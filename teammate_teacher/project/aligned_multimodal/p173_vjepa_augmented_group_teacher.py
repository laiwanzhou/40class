"""Augment the deployable P165 group teacher with P142/P144/P149 Test heads."""

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
    build_test_bank as build_p165_test_bank,
    build_train_bank as build_p165_train_bank,
    concatenate,
    crossfit,
    features,
    lookup_for,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p173_vjepa_augmented_group_teacher_v1"
P142 = HERE / "runs/p142_vjepa_token_transformer_three_seed_v2/oof_predictions.npz"
P144 = HERE / "runs/p144_vjepa_hand_interaction_transformer_three_seed_v1/oof_predictions.npz"
P149 = HERE / "runs/p149_vjepa_repeat_consistency_three_seed_v2/oof_predictions.npz"
P172 = HERE / "runs/p172_vjepa_token_heads_test_v1/test_predictions.npz"
P89_SUBMISSION = HERE / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"


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


def build_train_bank():
    train, names = build_p165_train_bank()
    sources = []
    for name, path in (
        ("p142_all_token", P142),
        ("p144_hand_interaction", P144),
        ("p149_repeat_consistency", P149),
    ):
        with np.load(path, allow_pickle=False) as saved:
            sources.append(
                (name, saved["sample_ids"].astype(str), saved["probability"].astype(np.float32))
            )
    for split_name in SPLITS:
        part = train[split_name]
        extra = [
            align(probability, ids, part["ids"])[:, None, :]
            for _, ids, probability in sources
        ]
        part["bank"] = np.concatenate((part["bank"], *extra), axis=1)
    return train, [*names, *[name for name, _, _ in sources]]


def build_test_bank(base_names, expert_names):
    test = build_p165_test_bank(base_names)
    with np.load(P172, allow_pickle=False) as saved:
        ids = saved["sample_ids"].astype(str)
        sources = (
            saved["p142_all_probability"],
            saved["p144_hand_interaction_probability"],
            saved["p149_repeat_consistency_probability"],
        )
    positions = {value: row for row, value in enumerate(test["ids"])}
    rows = np.asarray([positions[value] for value in ids], dtype=np.int64)
    extras = []
    safe_probability = test["bank"][:, 0, :]
    for probability in sources:
        full = safe_probability.copy()
        full[rows] = probability
        extras.append(full[:, None, :])
    test["bank"] = np.concatenate((test["bank"], *extras), axis=1).astype(np.float32)
    if test["bank"].shape[1] != len(expert_names):
        raise RuntimeError("P173 Test expert count differs")
    return test


def main() -> None:
    train, expert_names = build_train_bank()
    reports, held_predictions, held_probabilities, thresholds = crossfit(train)
    all_train = concatenate([train[name] for name in SPLITS])
    all_lookup = lookup_for([train[name] for name in SPLITS])
    all_x = features(all_train, all_lookup, TRAIN_METADATA)
    # P165 base names are the first 21 entries; P172 adds the final three.
    test = build_test_bank(expert_names[:21], expert_names)
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
    submission = OUTPUT / "submission_p173_vjepa_augmented_group.csv"
    submission_io.write_submission(
        submission, submission_io.read_rows(P89_SUBMISSION), prediction
    )
    np.savez_compressed(
        OUTPUT / "predictions.npz",
        sample_ids=test["ids"],
        base_prediction=test["base"],
        proposal_prediction=proposal,
        probability=test_probability.astype(np.float32),
        route=route,
        prediction=prediction,
        **{
            f"{name}_held_prediction": held_predictions[name] for name in SPLITS
        },
        **{
            f"{name}_held_probability": held_probabilities[name] for name in SPLITS
        },
    )
    total = sum(len(train[name]["labels"]) for name in SPLITS)
    correct = sum(
        int(np.sum(held_predictions[name] == train[name]["labels"])) for name in SPLITS
    )
    base_correct = sum(
        int(np.sum(train[name]["base"] == train[name]["labels"])) for name in SPLITS
    )
    report = {
        "stage": "P173_VJEPA_augmented_deployable_group_teacher",
        "status": "complete",
        "protocol": {
            "expert_count": len(expert_names),
            "expert_names": expert_names,
            "new_experts": expert_names[-3:],
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
            "p165_reference_correct": 2123,
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
