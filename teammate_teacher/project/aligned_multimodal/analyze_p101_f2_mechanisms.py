"""Mechanism audit for P101-F2 source-safe shared-adapter OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_p101_f1_mechanisms import (
    describe,
    nested_vulnerability,
    paired,
    paired_binary,
    safe_correlation,
    topk_correct,
    true_rank,
)
from p101_finegrained_teacher_data import load_p101_data
from train_p100a_global_teacher_oof import softmax_numpy


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/p101_f2_source_safe_adapter_oof_v1"
F1_RUN = HERE / "runs/p101_f1_coarse_anchor_oof_v1"
P100_RUN = HERE / "runs/p100a_a0_global_teacher_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def class_changes(
    labels: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    before_top5: np.ndarray,
    after_top5: np.ndarray,
) -> list[dict[str, int]]:
    values: list[dict[str, int]] = []
    for label in range(40):
        mask = labels == label
        record = paired(before[mask], after[mask], labels[mask])
        top5 = paired_binary(before_top5[mask], after_top5[mask])
        if record["prediction_changes"] or top5["rescue"] or top5["harm"]:
            values.append(
                {
                    "class": label,
                    "rows": int(mask.sum()),
                    "top1_rescue": record["rescue"],
                    "top1_harm": record["harm"],
                    "top1_net": record["net"],
                    "top1_prediction_changes": record["prediction_changes"],
                    "top5_rescue": top5["rescue"],
                    "top5_harm": top5["harm"],
                    "top5_net": top5["net"],
                }
            )
    return values


def training_transfer(run: Path) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold in range(4):
        payload = torch.load(
            run / "F2_VSI" / f"fold{fold}_final.pt",
            map_location="cpu",
            weights_only=False,
        )
        history = payload["history"]
        source_error_rows = int(payload["source_safe_error_rows"])
        nested_rows = sum(
            value["prediction"]["rows"]
            for value in payload["nested_anchor_audits"].values()
        )
        folds.append(
            {
                "fold": fold,
                "source_safe_anchor_accuracy": 1.0 - source_error_rows / nested_rows,
                "epoch1_accuracy": float(history[0]["source_safe_accuracy"]),
                "final_accuracy": float(history[-1]["source_safe_accuracy"]),
                "final_residual_rms": float(history[-1]["residual"]),
                "final_pair_loss": float(history[-1]["pair"]),
                "final_per_anchor_accuracy": history[-1]["per_anchor_accuracy"],
            }
        )
    return {"folds": folds}


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    fine = load_p101_data()
    with np.load(run / "F2_VSI_complete_oof.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(P100_RUN / "VS_complete_oof.npz", allow_pickle=False) as archive:
        anchor_logits = np.asarray(archive["direct_logits"], dtype=np.float32)
        anchor_probability = np.asarray(archive["direct_probability"], dtype=np.float32)
    if not np.array_equal(arrays["sample_ids"].astype(str), fine.sample_ids):
        raise RuntimeError("P101-F2 mechanism row order changed")
    labels = fine.labels
    direct_probability = np.asarray(arrays["direct_probability"], dtype=np.float32)
    direct_prediction = direct_probability.argmax(axis=1)
    anchor_prediction = anchor_probability.argmax(axis=1)
    anchor_top5 = topk_correct(anchor_probability, labels, 5)
    direct_top5 = topk_correct(direct_probability, labels, 5)
    changed = direct_prediction != anchor_prediction
    rescue = (anchor_prediction != labels) & (direct_prediction == labels)
    harm = (anchor_prediction == labels) & (direct_prediction != labels)
    vulnerability, vulnerability_audit = nested_vulnerability(F1_RUN, len(labels))
    uncertainty = np.asarray(arrays["direct_uncertainty"], dtype=np.float32)
    residual = np.asarray(arrays["direct_fine_residual_rms"], dtype=np.float32)
    sorted_probability = np.sort(anchor_probability, axis=1)
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    direct_delta = np.asarray(arrays["direct_logits"], dtype=np.float32) - anchor_logits
    direct_delta_rms = np.sqrt(np.mean(direct_delta**2, axis=1))
    cf_audit: dict[str, Any] = {}
    imu_content: dict[str, Any] = {}
    for name in (
        "zero_imu",
        "reverse_imu",
        "shuffle_imu",
        "local_reverse_skeleton",
        "zero_skeleton",
        "shuffle_skeleton",
    ):
        probability = softmax_numpy(np.asarray(arrays[f"{name}_logits"], dtype=np.float32))
        prediction = probability.argmax(axis=1)
        cf_audit[name] = {
            "counterfactual_to_direct": paired(prediction, direct_prediction, labels),
            "top5_counterfactual_to_direct": paired_binary(
                topk_correct(probability, labels, 5), direct_top5
            ),
            "mean_probability_absolute_difference": float(
                np.abs(probability - direct_probability).mean()
            ),
            "max_probability_absolute_difference": float(
                np.abs(probability - direct_probability).max()
            ),
        }
        if name in {"reverse_imu", "shuffle_imu"}:
            delta = np.asarray(arrays[f"{name}_logits"], dtype=np.float32) - anchor_logits
            difference = np.sqrt(np.mean((direct_delta - delta) ** 2, axis=1))
            dot = (direct_delta * delta).sum(axis=1)
            norm = np.linalg.norm(direct_delta, axis=1) * np.linalg.norm(delta, axis=1)
            cosine = dot / np.maximum(norm, 1e-12)
            imu_content[name] = {
                "aligned_delta_rms": describe(direct_delta_rms),
                "aligned_minus_counterfactual_delta_rms": describe(difference),
                "difference_over_aligned_ratio": float(
                    difference.mean() / max(float(direct_delta_rms.mean()), 1e-12)
                ),
                "delta_cosine": describe(cosine),
            }
    rank_anchor = true_rank(anchor_probability, labels)
    rank_direct = true_rank(direct_probability, labels)
    true_delta = direct_delta[np.arange(len(labels)), labels]
    anchor_class_delta = direct_delta[np.arange(len(labels)), anchor_prediction]
    true_advantage = true_delta - anchor_class_delta
    mean_class_delta = direct_delta.mean(axis=0)
    class_bias_order = np.argsort(np.abs(mean_class_delta))[::-1][:12]
    row_details = [
        {
            "row": int(row),
            "sample_id": str(fine.sample_ids[row]),
            "user": str(fine.users[row]),
            "fold": int(fine.fold_ids[row]),
            "label": int(labels[row]),
            "anchor_prediction": int(anchor_prediction[row]),
            "candidate_prediction": int(direct_prediction[row]),
            "kind": "rescue" if rescue[row] else "harm" if harm[row] else "wrong_to_wrong",
            "nested_vulnerability": float(vulnerability[row]),
            "anchor_margin": float(margin[row]),
            "uncertainty": float(uncertainty[row]),
            "residual_rms": float(residual[row]),
            "true_rank_before": int(rank_anchor[row]),
            "true_rank_after": int(rank_direct[row]),
            "true_logit_advantage": float(true_advantage[row]),
        }
        for row in np.flatnonzero(changed)
    ]
    result = {
        "status": "complete",
        "matched": paired(anchor_prediction, direct_prediction, labels),
        "top5": paired_binary(anchor_top5, direct_top5),
        "routing": {
            "nested_vulnerability_audit": vulnerability_audit,
            "changed": {
                "nested_vulnerability": describe(vulnerability[changed]),
                "uncertainty": describe(uncertainty[changed]),
                "anchor_margin": describe(margin[changed]),
                "residual_rms": describe(residual[changed]),
            },
            "stable": {
                "nested_vulnerability": describe(vulnerability[~changed]),
                "uncertainty": describe(uncertainty[~changed]),
                "anchor_margin": describe(margin[~changed]),
                "residual_rms": describe(residual[~changed]),
            },
            "residual_uncertainty_correlation": safe_correlation(residual, uncertainty),
        },
        "source_safe_to_outer_transfer": training_transfer(run),
        "counterfactuals": cf_audit,
        "imu_class_content": imu_content,
        "label_direction": {
            "anchor_wrong_rows": int((anchor_prediction != labels).sum()),
            "true_rank_improved": int((rank_direct < rank_anchor).sum()),
            "true_rank_harmed": int((rank_direct > rank_anchor).sum()),
            "true_logit_advantage_anchor_wrong": describe(
                true_advantage[anchor_prediction != labels]
            ),
            "fraction_anchor_wrong_true_advantage_positive": float(
                (true_advantage[anchor_prediction != labels] > 0).mean()
            ),
            "largest_mean_class_logit_bias": [
                {"class": int(label), "mean_logit_delta": float(mean_class_delta[label])}
                for label in class_bias_order
            ],
        },
        "class_changes": class_changes(
            labels,
            anchor_prediction,
            direct_prediction,
            anchor_top5,
            direct_top5,
        ),
        "per_fold": {
            str(fold): paired(
                anchor_prediction[fine.fold_ids == fold],
                direct_prediction[fine.fold_ids == fold],
                labels[fine.fold_ids == fold],
            )
            for fold in range(4)
        },
        "architecture_audit": {
            "correspondence_logits_feed_residual": False,
            "residual_can_use_visual_or_skeleton_only_evidence_tokens": True,
            "imu_availability_multiplies_residual": True,
            "interpretation": "pair BCE trains a disconnected diagnostic head; classification can learn a V/S-driven residual that merely requires IMU availability",
        },
        "changed_rows": row_details,
        "h3_rows_loaded": 0,
        "student_started": False,
        "b_teacher_started": False,
    }
    path = run / "mechanism_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
