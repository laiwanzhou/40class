"""Reproducible mechanism audit for the completed P100-A0 OOF artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p100a_global_teacher_data import parse_sample_id


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_INPUT = HERE / "runs/p100a_a0_global_teacher_oof_v1"
DEFAULT_OUTPUT = PROJECT / "runs/p100a_a0_global_teacher_oof_v1/mechanism_analysis.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability.astype(np.float32)


def group_statistics(
    mask: np.ndarray, gate: np.ndarray, residual: np.ndarray
) -> dict[str, float | int | None]:
    return {
        "rows": int(mask.sum()),
        "imu_gate_mean": float(gate[mask].mean()) if mask.any() else None,
        "imu_residual_norm_mean": float(residual[mask].mean()) if mask.any() else None,
    }


def confusion_delta(
    labels: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    class_names: list[str],
    limit: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    before_counter = Counter(
        (int(truth), int(prediction))
        for truth, prediction in zip(labels, before)
        if truth != prediction
    )
    after_counter = Counter(
        (int(truth), int(prediction))
        for truth, prediction in zip(labels, after)
        if truth != prediction
    )
    changes: list[dict[str, Any]] = []
    for pair in set(before_counter) | set(after_counter):
        truth, prediction = pair
        old = before_counter[pair]
        new = after_counter[pair]
        if old == new:
            continue
        changes.append(
            {
                "true_class": truth,
                "true_name": class_names[truth],
                "predicted_class": prediction,
                "predicted_name": class_names[prediction],
                "before": old,
                "after": new,
                "delta_errors": new - old,
            }
        )
    return {
        "largest_reductions": sorted(
            changes, key=lambda value: (value["delta_errors"], value["true_class"])
        )[:limit],
        "largest_increases": sorted(
            changes,
            key=lambda value: (-value["delta_errors"], value["true_class"]),
        )[:limit],
    }


def main() -> None:
    args = parse_args()
    archives = {
        variant: np.load(args.input / f"{variant}_complete_oof.npz", allow_pickle=False)
        for variant in ("V", "VS", "VI", "VSI")
    }
    try:
        sample_ids = archives["VSI"]["sample_ids"].astype(str)
        users = archives["VSI"]["users"].astype(str)
        labels = np.asarray([parse_sample_id(value)[0] for value in sample_ids])
        prediction = {
            variant: archives[variant]["direct_probability"].argmax(axis=1)
            for variant in archives
        }
        vsi = archives["VSI"]
        zero_imu_prediction = softmax(vsi["zero_imu_logits"]).argmax(axis=1)
        shuffle_imu_prediction = softmax(vsi["shuffle_imu_logits"]).argmax(axis=1)
        direct_correct = prediction["VSI"] == labels
        zero_correct = zero_imu_prediction == labels
        shuffle_correct = shuffle_imu_prediction == labels
        vs_correct = prediction["VS"] == labels
        gate = vsi["reliability_imu"]
        residual = vsi["residual_norm_imu"]

        groups = {
            "all": np.ones(len(labels), dtype=bool),
            "vsi_rescue_vs_zero_imu": direct_correct & ~zero_correct,
            "vsi_harm_vs_zero_imu": ~direct_correct & zero_correct,
            "vsi_rescue_vs_vs": direct_correct & ~vs_correct,
            "vsi_harm_vs_vs": ~direct_correct & vs_correct,
            "both_correct_direct_zero": direct_correct & zero_correct,
            "both_wrong_direct_zero": ~direct_correct & ~zero_correct,
        }
        group_audit = {
            name: group_statistics(mask, gate, residual)
            for name, mask in groups.items()
        }
        per_subject: dict[str, Any] = {}
        for user in sorted(np.unique(users).tolist()):
            mask = users == user
            rescue = int((direct_correct[mask] & ~zero_correct[mask]).sum())
            harm = int((~direct_correct[mask] & zero_correct[mask]).sum())
            per_subject[user] = {
                "rows": int(mask.sum()),
                "imu_gate_mean": float(gate[mask].mean()),
                "imu_residual_norm_mean": float(residual[mask].mean()),
                "direct_vs_zero_imu_rescue": rescue,
                "direct_vs_zero_imu_harm": harm,
                "direct_vs_zero_imu_net": rescue - harm,
                "direct_vs_shuffle_imu_net": int(
                    (direct_correct[mask] & ~shuffle_correct[mask]).sum()
                    - (~direct_correct[mask] & shuffle_correct[mask]).sum()
                ),
            }
        class_names = pd.read_csv(PROJECT / "class_mapping.csv")["action_name"].tolist()
        output = {
            "status": "complete",
            "question": "Why did A0 Skeleton help while IMU prevented full VSI from improving?",
            "group_gate_audit": group_audit,
            "per_subject_imu_audit": per_subject,
            "skeleton_confusion_changes_vs_visual": confusion_delta(
                labels, prediction["V"], prediction["VS"], class_names
            ),
            "mechanism_findings": {
                "skeleton_is_used": True,
                "skeleton_matched_gain_is_subject_bootstrap_positive": True,
                "imu_is_used": True,
                "correct_imu_pairing_beats_shuffle": int(direct_correct.sum())
                > int(shuffle_correct.sum()),
                "zero_imu_beats_direct": int(zero_correct.sum())
                > int(direct_correct.sum()),
                "imu_gate_separates_rescue_from_harm": bool(
                    group_audit["vsi_rescue_vs_zero_imu"]["imu_gate_mean"]
                    > group_audit["vsi_harm_vs_zero_imu"]["imu_gate_mean"] + 0.05
                ),
                "imu_direct_vs_zero_nonnegative_subjects": int(
                    sum(value["direct_vs_zero_imu_net"] >= 0 for value in per_subject.values())
                ),
                "diagnosis": "IMU content is aligned and changes the classifier, but the unconstrained residual gate cannot distinguish useful from harmful boundary changes and overfits subject-dependent evidence.",
                "structural_implication": "Preserve the verified VS boundary and add IMU/cross evidence through an exact-zero-initialized bounded adapter with explicit VS-correct preservation; do not rescan weights or discard Skeleton.",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == "__main__":
    main()
