"""Deployable repeat-group teacher using only experts with matched Test outputs.

The validation path is strict outer cohort cross-fit.  Each held cohort is
predicted by a group classifier trained on the other two cohorts.  The Test
classifier is then refit on all scored OOF rows and uses the identical feature
construction, expert order, repeat geometry, and replacement rule.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p137_group_classifier_selector import (
    CONFIG,
    choose_threshold,
    fit_probability,
    group_features,
)
from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "runs/p165_deployable_group_teacher_v1"
A18_OOF = REPO / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
A18_TEST = HERE / "runs/a18_full_teacher_test_v1/test_predictions.npz"
TEST_BASE = HERE / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
TEST_IMU = HERE / "runs/p3_sd_imu_rf_full18/test_logits.npz"
TEST_SAFE = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
TRAIN_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
TEST_METADATA = HERE / "data/p85_recording_metadata/test_recording_metadata.csv"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def normalise(values: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    return probability / probability.sum(axis=1, keepdims=True)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def read_prediction(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray(
            [int(row["prediction"]) for row in csv.DictReader(handle)],
            dtype=np.int64,
        )


def aligned(values: np.ndarray, source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {value: row for row, value in enumerate(source_ids.astype(str))}
    missing = [value for value in target_ids.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"source misses {len(missing)} requested rows")
    rows = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(values)[rows]


def build_train_bank():
    splits = load_splits()
    teacher = np.load(HERE / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz")
    all_ids = teacher["oof_sample_ids"].astype(str)
    full_probability, full_names = full40.train_probabilities(all_ids)
    with np.load(A18_OOF, allow_pickle=False) as saved:
        a18_ids = saved["sample_ids"].astype(str)
        a18_probability = normalise(saved["best_session_probability"])

    result = {}
    for name in SPLITS:
        split = splits[name]
        ids = split.sample_ids.astype(str)
        bank = np.concatenate(
            (
                normalise(split.safe_probability)[:, None, :],
                aligned(full_probability, all_ids, ids),
                aligned(a18_probability, a18_ids, ids)[:, None, :],
            ),
            axis=1,
        )
        result[name] = {
            "ids": ids,
            "labels": split.labels.astype(np.int64),
            "users": split.users.astype(str),
            "base": split.safe_prediction.astype(np.int64),
            "bank": bank.astype(np.float32),
        }
    return result, ["p89_safe_probability", *full_names, "a18_best_session"]


def lookup_for(parts) -> dict[str, np.ndarray]:
    return {
        str(sample_id): probability
        for part in parts
        for sample_id, probability in zip(part["ids"], part["bank"])
    }


def concatenate(parts):
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in ("ids", "labels", "users", "base", "bank")
    }


def features(part, lookup, metadata_path: Path) -> np.ndarray:
    return group_features(
        part["ids"],
        part["base"],
        lookup,
        posterior_feature_mode="sqrt",
        group_config=CONFIG,
        group_feature_layout="full",
        teacher_subset="full",
        grouping_lookup=lookup,
        metadata_path=metadata_path,
    )


def metrics(labels, base, prediction) -> dict[str, object]:
    base_correct = base == labels
    selected_correct = prediction == labels
    return {
        **classification_metrics(labels, prediction),
        "base_correct": int(base_correct.sum()),
        "rescue": int(np.sum(~base_correct & selected_correct)),
        "harm": int(np.sum(base_correct & ~selected_correct)),
        "net": int(selected_correct.sum() - base_correct.sum()),
        "changed": int(np.sum(prediction != base)),
    }


def crossfit(train):
    reports = {}
    predictions = {}
    probabilities = {}
    thresholds = []
    for held_name in SPLITS:
        source_names = [name for name in SPLITS if name != held_name]
        source_parts = [train[name] for name in source_names]
        source = concatenate(source_parts)
        source_lookup = lookup_for(source_parts)
        source_x_parts = {
            name: features(train[name], source_lookup, TRAIN_METADATA)
            for name in source_names
        }
        source_probability = np.zeros((len(source["ids"]), 40), dtype=np.float64)
        offset = 0
        for target_name, calibration_name in (
            (source_names[0], source_names[1]),
            (source_names[1], source_names[0]),
        ):
            target = train[target_name]
            calibration = train[calibration_name]
            target_probability = fit_probability(
                source_x_parts[calibration_name],
                calibration["labels"],
                source_x_parts[target_name],
                0.03,
                2.0,
                False,
                0.0,
            )
            rows = slice(offset, offset + len(target["ids"]))
            source_probability[rows] = target_probability
            offset += len(target["ids"])
        source_proposal = source_probability.argmax(axis=1).astype(np.int64)
        threshold = choose_threshold(
            source["base"], source_proposal, source_probability, source["labels"]
        )
        thresholds.append(float(threshold["threshold"]))

        held = train[held_name]
        held_lookup = lookup_for([*source_parts, held])
        source_x = features(source, held_lookup, TRAIN_METADATA)
        held_x = features(held, held_lookup, TRAIN_METADATA)
        held_probability = fit_probability(
            source_x,
            source["labels"],
            held_x,
            0.03,
            2.0,
            False,
            0.0,
        )
        proposal = held_probability.argmax(axis=1).astype(np.int64)
        rows = np.arange(len(proposal))
        score = held_probability[rows, proposal] - held_probability[rows, held["base"]]
        route = (proposal != held["base"]) & (score >= float(threshold["threshold"]))
        prediction = held["base"].copy()
        prediction[route] = proposal[route]
        reports[held_name] = {
            "source_cohorts": source_names,
            "source_threshold": threshold,
            "held": metrics(held["labels"], held["base"], prediction),
        }
        predictions[held_name] = prediction
        probabilities[held_name] = held_probability.astype(np.float32)
    return reports, predictions, probabilities, thresholds


def build_test_bank(expert_names):
    with np.load(TEST_BASE, allow_pickle=False) as saved:
        ids = saved["sample_ids"].astype(str)
        base_probability = normalise(saved["base_probability"])
        detail_ids = saved["detail_sample_ids"].astype(str)
    with np.load(TEST_IMU, allow_pickle=False) as saved:
        imu_ids = saved["sample_ids"].astype(str)
        imu_probability = softmax(saved["imu_logits"], 3.0)
    adjusted = 0.95 * base_probability + 0.05 * aligned(imu_probability, imu_ids, ids)
    adjusted = normalise(adjusted)
    base = read_prediction(TEST_SAFE)
    full_probability, full_names = full40.test_probabilities(detail_ids)
    if full_names != expert_names[1:-1]:
        raise RuntimeError("Train/Test full40 expert order differs")
    full = np.repeat(adjusted[:, None, :], len(full_names), axis=1)
    positions = {value: row for row, value in enumerate(ids)}
    rows = np.asarray([positions[value] for value in detail_ids], dtype=np.int64)
    full[rows] = full_probability
    with np.load(A18_TEST, allow_pickle=False) as saved:
        a18 = aligned(saved["selected_probability"], saved["sample_ids"], ids)
    bank = np.concatenate((adjusted[:, None, :], full, normalise(a18)[:, None, :]), axis=1)
    return {
        "ids": ids,
        "base": base,
        "bank": bank.astype(np.float32),
        "visual_available": np.isin(ids, detail_ids),
    }


def main() -> None:
    train, expert_names = build_train_bank()
    reports, held_predictions, held_probabilities, thresholds = crossfit(train)
    all_train = concatenate([train[name] for name in SPLITS])
    all_lookup = lookup_for([train[name] for name in SPLITS])
    all_x = features(all_train, all_lookup, TRAIN_METADATA)
    test = build_test_bank(expert_names)
    test_lookup = {sample_id: value for sample_id, value in zip(test["ids"], test["bank"])}
    test_x = group_features(
        test["ids"],
        test["base"],
        test_lookup,
        posterior_feature_mode="sqrt",
        group_config=CONFIG,
        group_feature_layout="full",
        teacher_subset="full",
        grouping_lookup=test_lookup,
        metadata_path=TEST_METADATA,
    )
    test_probability = fit_probability(
        all_x, all_train["labels"], test_x, 0.03, 2.0, False, 0.0
    )
    proposal = test_probability.argmax(axis=1).astype(np.int64)
    rows = np.arange(len(proposal))
    score = test_probability[rows, proposal] - test_probability[rows, test["base"]]
    threshold = float(np.median(np.asarray(thresholds, dtype=np.float64)))
    route = (proposal != test["base"]) & (score >= threshold)
    route &= test["visual_available"]
    prediction = test["base"].copy()
    prediction[route] = proposal[route]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_rows = submission_io.read_rows(TEST_SAFE)
    submission = OUTPUT / "submission_p165_deployable_group_teacher.csv"
    submission_io.write_submission(submission, source_rows, prediction)
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        sample_ids=test["ids"],
        base_prediction=test["base"],
        proposal_prediction=proposal,
        probability=test_probability.astype(np.float32),
        route=route,
        prediction=prediction,
        **{
            f"{name}_held_prediction": held_predictions[name]
            for name in SPLITS
        },
        **{
            f"{name}_held_probability": held_probabilities[name]
            for name in SPLITS
        },
    )
    aggregate_rows = sum(len(train[name]["labels"]) for name in SPLITS)
    aggregate_correct = sum(
        int(np.sum(held_predictions[name] == train[name]["labels"])) for name in SPLITS
    )
    base_correct = sum(
        int(np.sum(train[name]["base"] == train[name]["labels"])) for name in SPLITS
    )
    report = {
        "stage": "P165_deployable_repeat_group_teacher",
        "status": "complete_strict_outer_crossfit_and_test_inference",
        "protocol": {
            "candidate_bank": "P89 safe probability + 19 matched Full40 experts + A18",
            "candidate_count": len(expert_names),
            "expert_names": expert_names,
            "group_head": "StandardScaler + LogisticRegression C=0.03",
            "peer_weight": 2.0,
            "held_inputs_or_labels_used_for_training": False,
            "user_id_used_as_feature": False,
            "test_labels_read": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": aggregate_rows,
            "base_correct": base_correct,
            "base_accuracy": base_correct / aggregate_rows,
            "correct": aggregate_correct,
            "accuracy": aggregate_correct / aggregate_rows,
            "net_vs_p89": aggregate_correct - base_correct,
        },
        "test": {
            "rows": len(test["ids"]),
            "matched_visual_expert_rows": int(test["visual_available"].sum()),
            "threshold_outer_median": threshold,
            "changes_vs_0.85572_p89_safe": int(route.sum()),
            "submission": str(submission.resolve()),
            "sha256": digest(submission),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
