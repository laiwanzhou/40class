"""Outer-cross-fit harm guard for A18-over-P89 selective replacement.

Only class-agnostic relative confidence, deployable P165 probability, and the
frozen micro-union agreement pattern are features.  User identity is used only
for source LOSO threshold validation and never enters a feature vector.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import p89_build_dual_consensus_submission as submission_io
from audit_a18_p89_selective_replacement import load_data
from deploy_p168_historical_micro_union import deploy_micro
from p117_transductive_multicandidate_router import select_threshold


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "runs/p169_relative_reliability_harm_guard_v1"
A18_OOF = REPO / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
P89_REFERENCE = REPO / "runs/p90_crossuser_visual_router_v1/full_predictions.npz"
P165 = HERE / "runs/p165_deployable_group_teacher_v1/predictions.npz"
MICRO = HERE / "runs/p89_verified_micro_union_audit_v1/validation_predictions.npz"
P89_TEST = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"
P89_TEST_PROB = HERE / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
IMU_TEST = HERE / "runs/p3_sd_imu_rf_full18/test_logits.npz"
A18_TEST = HERE / "runs/a18_full_teacher_test_v1/test_predictions.npz"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
EPSILON = 1e-12


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def normalise(values: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), EPSILON, None)
    return probability / probability.sum(axis=1, keepdims=True)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def margin(probability: np.ndarray) -> np.ndarray:
    top = np.partition(probability, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, EPSILON, 1.0)
    return -np.sum(values * np.log(values), axis=1) / np.log(values.shape[1])


def meta_features(base_features, base, a18, group_probability, micro):
    rows = np.arange(len(base), dtype=np.int64)
    group_probability = normalise(group_probability)
    group_prediction = group_probability.argmax(axis=1).astype(np.int64)
    matrices = [
        np.column_stack(
            [
                base_features[name]
                for name in (
                    "a18_confidence",
                    "p89_max_confidence",
                    "p89_safe_support",
                    "confidence_gap",
                    "safe_support_gap",
                    "margin_gap",
                )
            ]
        ),
        np.column_stack(
            (
                group_probability.max(axis=1),
                margin(group_probability),
                entropy(group_probability),
                group_probability[rows, a18],
                group_probability[rows, base],
                group_probability[rows, a18] - group_probability[rows, base],
                group_prediction == a18,
                group_prediction == base,
                micro == a18,
                micro == base,
                micro != base,
                a18 != base,
            )
        ),
    ]
    output = np.concatenate(matrices, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite P169 meta feature")
    return output


def fit_score(train_x, gain, predict_x):
    decisive = gain != 0
    target = (gain[decisive] > 0).astype(np.int64)
    if len(np.unique(target)) != 2:
        return np.zeros(len(predict_x), dtype=np.float64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.03,
            max_iter=1200,
            solver="liblinear",
            class_weight="balanced",
        ),
    )
    model.fit(train_x[decisive], target)
    return model.predict_proba(predict_x)[:, 1]


def loso_score(x, gain, users):
    score = np.zeros(len(x), dtype=np.float64)
    for user in sorted(set(users.astype(str).tolist())):
        held = users.astype(str) == user
        score[held] = fit_score(x[~held], gain[~held], x[held])
    return score


def read_prediction(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray([int(row["prediction"]) for row in csv.DictReader(handle)])


def align(values, source_ids, target_ids):
    lookup = {value: row for row, value in enumerate(source_ids.astype(str))}
    rows = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(values)[rows]


def load_oof():
    audit = load_data(A18_OOF, P89_REFERENCE)
    p165 = np.load(P165, allow_pickle=False)
    micro = np.load(MICRO, allow_pickle=False)
    output = {}
    prefixes = {
        "H1_selection": "h1",
        "H2_confirmation": "h2",
        "H3_independent_fold0": "h3",
    }
    for name in SPLITS:
        data = audit[name]
        prefix = prefixes[name]
        micro_ids = micro[f"{prefix}_sample_ids"].astype(str)
        if not np.array_equal(data.sample_ids, micro_ids):
            raise RuntimeError(f"{name}: micro sample order differs")
        group_probability = p165[f"{name}_held_probability"].astype(np.float64)
        group_prediction = p165[f"{name}_held_prediction"].astype(np.int64)
        micro_prediction = micro[f"{prefix}_union"].astype(np.int64)
        output[name] = {
            "ids": data.sample_ids,
            "labels": data.labels,
            "users": data.users,
            "base": data.p89_prediction,
            "a18": data.a18_prediction,
            "group": group_prediction,
            "micro": micro_prediction,
            "x": meta_features(
                data.features,
                data.p89_prediction,
                data.a18_prediction,
                group_probability,
                micro_prediction,
            ),
        }
    return output


def concatenate(parts):
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in ("ids", "labels", "users", "base", "a18", "group", "micro", "x")
    }


def apply_tail(base, guarded, group, micro):
    prediction = guarded.copy()
    available = prediction == base
    prediction[available & (group != base)] = group[available & (group != base)]
    available = prediction == base
    prediction[available & (micro != base)] = micro[available & (micro != base)]
    return prediction


def crossfit(data):
    reports = {}
    predictions = {}
    thresholds = []
    for held_name in SPLITS:
        source_names = [name for name in SPLITS if name != held_name]
        source = concatenate([data[name] for name in source_names])
        gain = (source["a18"] == source["labels"]).astype(np.int8) - (
            source["base"] == source["labels"]
        ).astype(np.int8)
        disagreement = source["a18"] != source["base"]
        nested = loso_score(source["x"], gain, source["users"])
        selected = select_threshold(
            nested, gain, disagreement, source["users"]
        )
        eligible = selected["net"] > 0 and selected["minimum_user_gain"] >= 0
        thresholds.append(float(selected["threshold"]))
        held = data[held_name]
        score = fit_score(source["x"], gain, held["x"])
        route = (
            eligible
            & (held["a18"] != held["base"])
            & (score >= float(selected["threshold"]))
        )
        guarded = held["base"].copy()
        guarded[route] = held["a18"][route]
        prediction = apply_tail(
            held["base"], guarded, held["group"], held["micro"]
        )
        base_correct = held["base"] == held["labels"]
        final_correct = prediction == held["labels"]
        reports[held_name] = {
            "source_cohorts": source_names,
            "source_threshold": selected,
            "eligible": bool(eligible),
            "held": {
                "rows": len(prediction),
                "base_correct": int(base_correct.sum()),
                "correct": int(final_correct.sum()),
                "net": int(final_correct.sum() - base_correct.sum()),
                "changed": int(np.sum(prediction != held["base"])),
                "a18_guard_routes": int(route.sum()),
                "rescue": int(np.sum(~base_correct & final_correct)),
                "harm": int(np.sum(base_correct & ~final_correct)),
            },
        }
        predictions[held_name] = prediction
    return reports, predictions, thresholds


def test_features():
    with np.load(A18_TEST, allow_pickle=False) as saved:
        ids = saved["sample_ids"].astype(str)
        a_probability = normalise(saved["selected_probability"])
    with np.load(P89_TEST_PROB, allow_pickle=False) as saved:
        base_ids = saved["sample_ids"].astype(str)
        base_probability = normalise(saved["base_probability"])
    if not np.array_equal(ids, base_ids):
        raise RuntimeError("P169 Test A18/P89 order differs")
    with np.load(IMU_TEST, allow_pickle=False) as saved:
        imu_probability = softmax(saved["imu_logits"], 3.0)
        imu_ids = saved["sample_ids"].astype(str)
    p_probability = normalise(
        0.95 * base_probability + 0.05 * align(imu_probability, imu_ids, ids)
    )
    base = read_prediction(P89_TEST)
    a18 = a_probability.argmax(axis=1).astype(np.int64)
    with np.load(P165, allow_pickle=False) as saved:
        if not np.array_equal(ids, saved["sample_ids"].astype(str)):
            raise RuntimeError("P169 Test P165 order differs")
        group_probability = saved["probability"].astype(np.float64)
        group = saved["prediction"].astype(np.int64)
    micro_ids, micro_base, _, _, micro, _ = deploy_micro()
    if not np.array_equal(ids, micro_ids) or not np.array_equal(base, micro_base):
        raise RuntimeError("P169 Test micro order/base differs")
    top_a = np.partition(a_probability, -2, axis=1)[:, -2:]
    top_p = np.partition(p_probability, -2, axis=1)[:, -2:]
    rows = np.arange(len(ids), dtype=np.int64)
    base_features = {
        "a18_confidence": a_probability.max(axis=1),
        "p89_max_confidence": p_probability.max(axis=1),
        "p89_safe_support": p_probability[rows, base],
        "confidence_gap": a_probability.max(axis=1) - p_probability.max(axis=1),
        "safe_support_gap": a_probability.max(axis=1) - p_probability[rows, base],
        "margin_gap": (top_a[:, 1] - top_a[:, 0]) - (top_p[:, 1] - top_p[:, 0]),
    }
    x = meta_features(base_features, base, a18, group_probability, micro)
    return ids, base, a18, group, micro, x


def main() -> None:
    data = load_oof()
    reports, held_predictions, thresholds = crossfit(data)
    all_data = concatenate([data[name] for name in SPLITS])
    prediction = np.concatenate([held_predictions[name] for name in SPLITS])
    correct = int(np.sum(prediction == all_data["labels"]))
    baseline_correct = int(np.sum(all_data["base"] == all_data["labels"]))
    fold_nets = [int(reports[name]["held"]["net"]) for name in SPLITS]
    passes = correct > 2146 and min(fold_nets) > 0

    report = {
        "stage": "P169_relative_reliability_harm_guard",
        "status": "passed_for_test" if passes else "rejected_keep_p168",
        "protocol": {
            "selector": "class-agnostic logistic C=0.03",
            "source_threshold": "LOSO by source user; user ID not a feature",
            "tail": "P165 group then verified micro-union",
            "test_labels_read": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": len(prediction),
            "base_correct": baseline_correct,
            "correct": correct,
            "accuracy": correct / len(prediction),
            "net_vs_p89": correct - baseline_correct,
            "fold_nets": fold_nets,
            "p168_correct_to_beat": 2146,
            "passes": passes,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        sample_ids=all_data["ids"],
        labels=all_data["labels"],
        base_prediction=all_data["base"],
        prediction=prediction,
    )
    if passes:
        gain = (all_data["a18"] == all_data["labels"]).astype(np.int8) - (
            all_data["base"] == all_data["labels"]
        ).astype(np.int8)
        ids, base, a18, group, micro, x = test_features()
        score = fit_score(all_data["x"], gain, x)
        threshold = float(np.median(np.asarray(thresholds)))
        route = (a18 != base) & (score >= threshold)
        guarded = base.copy()
        guarded[route] = a18[route]
        test_prediction = apply_tail(base, guarded, group, micro)
        submission = OUTPUT / "submission_p169_relative_reliability.csv"
        submission_io.write_submission(
            submission, submission_io.read_rows(P89_TEST), test_prediction
        )
        probability = np.full((len(ids), 40), 0.0005, dtype=np.float32)
        probability[np.arange(len(ids)), test_prediction] = 0.9805
        targets = OUTPUT / "student_test_targets.npz"
        np.savez_compressed(
            targets,
            sample_ids=ids,
            target_mask=np.ones(len(ids), dtype=bool),
            emission_probability=probability,
            structured_distillation_probability=probability,
            structured_confidence=np.full(len(ids), 0.9805, dtype=np.float32),
            emission_prediction=test_prediction,
            structured_distillation_prediction=test_prediction,
        )
        report["test"] = {
            "threshold_outer_median": threshold,
            "a18_guard_routes": int(route.sum()),
            "changes_vs_p89": int(np.sum(test_prediction != base)),
            "submission": str(submission.resolve()),
            "submission_sha256": digest(submission),
            "targets": str(targets.resolve()),
            "targets_sha256": digest(targets),
        }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
