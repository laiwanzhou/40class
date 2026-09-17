"""Add deployable P128 hierarchical probability to the P173 V-JEPA bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
from p137_group_classifier_selector import CONFIG, fit_probability, group_features
from p165_deployable_group_teacher import (
    SPLITS,
    TEST_METADATA,
    TRAIN_METADATA,
    concatenate,
    crossfit,
    features,
    lookup_for,
)
from p173_vjepa_augmented_group_teacher import (
    build_test_bank as build_p173_test_bank,
    build_train_bank as build_p173_train_bank,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p177_p128_vjepa_group_teacher_v1"
P128_OOF = HERE / "runs/p128_hierarchical_multimodal_full_oof_v1/oof_predictions.npz"
P128_TEST = HERE / "runs/p176_p128_hierarchical_test_v1/test_predictions.npz"
P89 = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def align(values, source_ids, target_ids):
    lookup = {value: row for row, value in enumerate(source_ids.astype(str))}
    rows = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(values)[rows]


def main() -> None:
    train, names = build_p173_train_bank()
    with np.load(P128_OOF, allow_pickle=False) as saved:
        p128_ids = saved["sample_ids"].astype(str)
        p128_probability = saved["probabilities"].astype(np.float32)
    for split_name in SPLITS:
        part = train[split_name]
        extra = align(p128_probability, p128_ids, part["ids"])[:, None, :]
        part["bank"] = np.concatenate((part["bank"], extra), axis=1)
    expert_names = [*names, "p128_hierarchical_multimodal"]
    reports, held_predictions, held_probabilities, thresholds = crossfit(train)
    all_train = concatenate([train[name] for name in SPLITS])
    all_lookup = lookup_for([train[name] for name in SPLITS])
    all_x = features(all_train, all_lookup, TRAIN_METADATA)
    test = build_p173_test_bank(names[:21], names)
    with np.load(P128_TEST, allow_pickle=False) as saved:
        p128_test = align(
            saved["probabilities"], saved["sample_ids"], test["ids"]
        ).astype(np.float32)
    test["bank"] = np.concatenate((test["bank"], p128_test[:, None, :]), axis=1)
    test_lookup = {
        sample_id: value for sample_id, value in zip(test["ids"], test["bank"])
    }
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
    rows = np.arange(len(proposal), dtype=np.int64)
    score = test_probability[rows, proposal] - test_probability[rows, test["base"]]
    threshold = float(np.median(np.asarray(thresholds, dtype=np.float64)))
    route = (proposal != test["base"]) & (score >= threshold) & test["visual_available"]
    prediction = test["base"].copy()
    prediction[route] = proposal[route]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    submission = OUTPUT / "submission_p177_p128_vjepa_group.csv"
    submission_io.write_submission(submission, submission_io.read_rows(P89), prediction)
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        sample_ids=test["ids"],
        base_prediction=test["base"],
        proposal_prediction=proposal,
        probability=test_probability.astype(np.float32),
        route=route,
        prediction=prediction,
        **{f"{name}_held_prediction": held_predictions[name] for name in SPLITS},
        **{f"{name}_held_probability": held_probabilities[name] for name in SPLITS},
    )
    total = sum(len(train[name]["labels"]) for name in SPLITS)
    correct = sum(
        int(np.sum(held_predictions[name] == train[name]["labels"])) for name in SPLITS
    )
    base_correct = sum(
        int(np.sum(train[name]["base"] == train[name]["labels"])) for name in SPLITS
    )
    report = {
        "stage": "P177_P128_VJEPA_deployable_group_teacher",
        "status": "complete",
        "protocol": {
            "expert_count": len(expert_names),
            "expert_names": expert_names,
            "new_expert": "p128_hierarchical_multimodal",
            "held_or_test_labels_used_for_training": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": total,
            "base_correct": base_correct,
            "correct": correct,
            "accuracy": correct / total,
            "net_vs_p89": correct - base_correct,
            "p173_reference_correct": 2151,
        },
        "test": {
            "threshold_outer_median": threshold,
            "changes_vs_p89": int(route.sum()),
            "submission": str(submission.resolve()),
            "submission_sha256": digest(submission),
            "test_labels_read": False,
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
