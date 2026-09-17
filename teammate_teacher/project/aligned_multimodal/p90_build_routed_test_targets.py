"""Deploy the validated outer-router ensemble and build P87-S Test soft targets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import log_softmax, softmax

from audit_p87_sequence_decoder import align_metadata, build_sessions
from p89_deploy_supervised_router import load_decoder
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import softmax as imu_softmax
from p90_build_routed_p87s_targets import calibrated_probability, normalized_entropy
from p90_crossuser_visual_router import (
    CANDIDATE_NAME,
    SplitData,
    build_features,
    concatenate_splits,
    fit_ensemble,
    load_splits,
)
from p90_internvideo2_l_teacher import feature_sets as iv2_feature_sets
from p90_videomaev2_distilled_teacher import (
    class_sample_weights,
    feature_sets as base_feature_sets,
)
from train_p46_videomae_head import make_model


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TRAIN_BASE = REPO_ROOT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
TRAIN_IV2 = REPO_ROOT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
TEST_BASE = REPO_ROOT / "runs/p90_videomaev2_distilled_test_v1/complete_features.npz"
TEST_IV2 = REPO_ROOT / "runs/p90_internvideo2_l_k400_test_v1/complete_features.npz"
ROUTER_DIR = REPO_ROOT / "runs/p90_crossuser_visual_router_v1"
BASE_TARGETS = HERE / "runs/p87s_test_structured_targets_v1/structured_targets.npz"
TEST_METADATA = HERE / "data/p85_recording_metadata/test_recording_metadata.csv"
OUTPUT = HERE / "runs/p90_p87s_routed_test_targets_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def align(source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {sample_id: row for row, sample_id in enumerate(source_ids.astype(str))}
    missing = [sample_id for sample_id in target_ids.astype(str) if sample_id not in lookup]
    if missing:
        raise RuntimeError(f"Alignment source misses {len(missing)} rows")
    return np.asarray(values)[[lookup[sample_id] for sample_id in target_ids.astype(str)]]


def fit_full_ridge(
    train_values: np.ndarray,
    labels: np.ndarray,
    test_values: np.ndarray,
    alpha: float,
    power: float,
) -> np.ndarray:
    model = make_model(alpha)
    model.fit(
        train_values,
        labels,
        ridge__sample_weight=class_sample_weights(labels, power),
    )
    if not np.array_equal(model.named_steps["ridge"].classes_, np.arange(40)):
        raise RuntimeError("Full visual Ridge does not contain all 40 classes")
    return np.asarray(model.decision_function(test_values), dtype=np.float64)


def visual_test_probabilities() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_base = load_npz(TRAIN_BASE)
    train_iv2 = load_npz(TRAIN_IV2)
    test_base = load_npz(TEST_BASE)
    test_iv2 = load_npz(TEST_IV2)
    train_ids = train_base["sample_ids"].astype(str)
    if not np.array_equal(train_ids, train_iv2["sample_ids"].astype(str)):
        raise RuntimeError("Selected training visual teacher orders differ")
    test_ids = test_base["sample_ids"].astype(str)
    if not np.array_equal(test_ids, test_iv2["sample_ids"].astype(str)):
        raise RuntimeError("Selected Test visual teacher orders differ")
    labels = train_base["labels"].astype(np.int64)
    base_train = base_feature_sets(train_base)
    base_test = base_feature_sets(test_base)
    iv2_train = iv2_feature_sets(train_iv2)
    iv2_test = iv2_feature_sets(test_iv2)
    base_logits = fit_full_ridge(
        base_train["early_late"], labels, base_test["early_late"], 3000.0, 0.75
    )
    iv2_early_logits = fit_full_ridge(
        iv2_train["early_late"], labels, iv2_test["early_late"], 3000.0, 0.75
    )
    iv2_joint_logits = fit_full_ridge(
        iv2_train["early_late_plus_k400"],
        labels,
        iv2_test["early_late_plus_k400"],
        3000.0,
        0.75,
    )
    base_probability = softmax(base_logits, axis=1)
    iv2_early_probability = softmax(iv2_early_logits, axis=1)
    iv2_joint_probability = softmax(iv2_joint_logits, axis=1)
    equal_probability = softmax(
        0.5 * log_softmax(base_logits, axis=1)
        + 0.5 * log_softmax(iv2_joint_logits, axis=1),
        axis=1,
    )
    return test_ids, {
        "videomaev2_distilled_base": base_probability,
        "internvideo2_l_early_late": iv2_early_probability,
        "internvideo2_l_early_late_plus_k400": iv2_joint_probability,
        CANDIDATE_NAME: equal_probability,
    }


def read_predictions(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray([int(row["prediction"]) for row in csv.DictReader(handle)])


def test_safe_contract(target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = load_npz(HERE / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz")
    all_ids = source["sample_ids"].astype(str)
    base_probability = np.asarray(source["base_probability"], dtype=np.float64)
    imu = load_npz(HERE / "runs/p3_sd_imu_rf_full18/test_logits.npz")
    imu_logits = align(imu["sample_ids"], imu["imu_logits"], all_ids)
    imu_probability = imu_softmax(np.asarray(imu_logits, dtype=np.float64), 3.0)
    adjusted = 0.95 * base_probability + 0.05 * imu_probability
    adjusted /= adjusted.sum(axis=1, keepdims=True)

    transition, decoder = load_decoder()
    metadata = align_metadata(TEST_METADATA, all_ids)
    indices = np.arange(len(all_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    p87 = read_predictions(
        HERE / "runs/p87s_final_test_predictions_v1/submission_p87s_student_decoded.csv"
    )
    grouping_source = json.loads(
        (HERE / "runs/p89_global_joint_grouping_tuned_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    grouping = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
    protocol = (
        all_ids,
        None,
        adjusted,
        p87,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        None,
    )
    safe, _ = joint_decode(
        adjusted,
        p87,
        protocol,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )
    deployed = read_predictions(
        HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
    )
    if not np.array_equal(safe, deployed):
        raise RuntimeError("Failed to reproduce the P89 safe Test contract")
    return (
        align(all_ids, adjusted, target_ids),
        align(all_ids, safe, target_ids).astype(np.int64),
        align(all_ids, p87, target_ids).astype(np.int64),
    )


def test_quality(sample_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    with TEST_METADATA.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    values = []
    for sample_id in sample_ids.astype(str):
        row = rows[sample_id]
        trial = str(row.get("trial_id", ""))
        try:
            repeat_index = float(trial.rsplit("-", 1)[-1])
        except ValueError:
            repeat_index = 0.0
        values.append(
            [
                float(row.get("duration_seconds") or 0.0),
                float(row.get("timestamp_available") or 0.0),
                float(row.get("imu_csv_files") or 0.0),
                float(row.get("imu_parsed_rows") or 0.0),
                float(row.get("device_count") or 0.0),
                repeat_index,
            ]
        )
    return np.asarray(values, dtype=np.float32), [
        "duration_seconds",
        "timestamp_available",
        "imu_csv_files",
        "imu_parsed_rows",
        "device_count",
        "trial_repeat_index",
    ]


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(map(int, values)).items())}


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    test_ids, visual = visual_test_probabilities()
    safe_probability, safe_prediction, p87_prediction = test_safe_contract(test_ids)
    metadata = align_metadata(TEST_METADATA, test_ids)
    indices = np.arange(len(test_ids), dtype=np.int64)
    transition, decoder = load_decoder()
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    quality, quality_names = test_quality(test_ids)
    test_split = SplitData(
        name="official_test_401",
        sample_ids=test_ids,
        labels=np.full(len(test_ids), -1, dtype=np.int64),
        users=np.full(len(test_ids), "anonymous", dtype=str),
        safe_probability=safe_probability,
        safe_prediction=safe_prediction,
        p87_prediction=p87_prediction,
        sessions=sessions,
        visual_probability=visual,
        quality=quality,
        quality_names=quality_names,
    )
    test_x, test_feature_names = build_features(test_split)

    splits = load_splits()
    train_features: dict[str, np.ndarray] = {}
    for name, split in splits.items():
        train_features[name], names = build_features(split)
        if names != test_feature_names:
            raise RuntimeError("Train/Test router feature contracts differ")
    outer = {
        "H1_selection": ["H2_confirmation", "H3_independent_fold0"],
        "H2_confirmation": ["H1_selection", "H3_independent_fold0"],
        "H3_independent_fold0": ["H1_selection", "H2_confirmation"],
    }
    frozen = json.loads((ROUTER_DIR / "full_summary.json").read_text(encoding="utf-8"))
    candidate_prediction = visual[CANDIDATE_NAME].argmax(axis=1)
    disagreement = candidate_prediction != safe_prediction
    member_scores: dict[str, np.ndarray] = {}
    member_routes: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for held_name, train_names in outer.items():
        train_x, train_gain, _train_disagreement, _train_users = concatenate_splits(
            train_names, splits, train_features
        )
        score, _ = fit_ensemble(train_x, train_gain, test_x)
        threshold = float(
            frozen["cohorts"][held_name]["nested_threshold_selection"]["threshold"]
        )
        member_scores[held_name] = score
        member_routes[held_name] = disagreement & (score >= threshold)
        thresholds[held_name] = threshold
    votes = np.stack(list(member_routes.values()), axis=1).sum(axis=1)
    route = votes >= 2
    route_score = np.mean(
        np.stack(
            [member_scores["H1_selection"], member_scores["H3_independent_fold0"]],
            axis=1,
        ),
        axis=1,
    )

    train_candidate = np.concatenate(
        [splits[name].visual_probability[CANDIDATE_NAME] for name in splits]
    )
    train_labels = np.concatenate([splits[name].labels for name in splits])
    calibrated_visual, calibration = calibrated_probability(
        train_candidate, train_labels, visual[CANDIDATE_NAME]
    )

    with np.load(BASE_TARGETS, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    target_ids = arrays["sample_ids"].astype(str)
    target_mask = arrays["target_mask"].astype(bool)
    selected_ids = target_ids[target_mask]
    if not np.array_equal(selected_ids, test_ids):
        raise RuntimeError("P87-S Test target and P90 visual orders differ")
    rows = np.flatnonzero(target_mask)
    probability = arrays["structured_distillation_probability"].astype(np.float64)
    old = probability[rows].copy()
    new = old.copy()
    new[route] = (
        (1.0 - route_score[route, None]) * old[route]
        + route_score[route, None] * calibrated_visual[route]
    )
    new /= new.sum(axis=1, keepdims=True)
    probability[rows] = new
    arrays["structured_distillation_probability"] = probability.astype(np.float32)
    arrays["structured_distillation_prediction"] = probability.argmax(axis=1).astype(
        np.int64
    )
    confidence = arrays["structured_confidence"].astype(np.float64)
    confidence[rows] = 1.0 - normalized_entropy(new)
    arrays["structured_confidence"] = confidence.astype(np.float32)
    route_full = np.zeros(len(target_ids), dtype=bool)
    score_full = np.zeros(len(target_ids), dtype=np.float32)
    vote_full = np.zeros(len(target_ids), dtype=np.int8)
    route_full[rows] = route
    score_full[rows] = route_score.astype(np.float32)
    vote_full[rows] = votes.astype(np.int8)
    arrays["p90_router_route"] = route_full
    arrays["p90_router_score"] = score_full
    arrays["p90_router_votes"] = vote_full

    old_prediction = old.argmax(axis=1)
    new_prediction = new.argmax(axis=1)
    report: dict[str, Any] = {
        "stage": "P90 outer-router ensemble Test target deployment",
        "protocol": (
            "Each of the three frozen outer router fits predicts Test. A row routes only "
            "with at least two votes; because H2's frozen threshold is no-route, both "
            "independently active H1/H3 fits must agree. No Test label is available or used."
        ),
        "test_rows": int(len(test_ids)),
        "router_feature_count": int(test_x.shape[1]),
        "outer_thresholds": thresholds,
        "member_route_counts": {
            name: int(values.sum()) for name, values in member_routes.items()
        },
        "majority_routed_rows": int(route.sum()),
        "mean_route_score": float(route_score[route].mean()) if np.any(route) else 0.0,
        "visual_calibration": calibration,
        "changed_target_argmax": int(np.sum(new_prediction != old_prediction)),
        "target_histogram_before": histogram(old_prediction),
        "target_histogram_after": histogram(new_prediction),
        "safe_candidate_disagreements": int(disagreement.sum()),
        "routed_candidate_classes": histogram(candidate_prediction[route]),
        "large_teacher_required_at_final_inference": False,
        "base_targets": str(BASE_TARGETS.resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "structured_targets.npz", **arrays)
    np.savez_compressed(
        output / "router_test_predictions.npz",
        sample_ids=test_ids,
        safe_prediction=safe_prediction,
        candidate_prediction=candidate_prediction,
        route=route,
        route_votes=votes,
        route_score=route_score,
        calibrated_visual_probability=calibrated_visual.astype(np.float32),
        **{f"{name}_score": value for name, value in member_scores.items()},
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
