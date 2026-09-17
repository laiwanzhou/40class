"""Final mechanism and route audit for P101-F3 causal interaction."""

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
from analyze_p101_f2_mechanisms import class_changes
from p101_finegrained_teacher_data import load_p101_data
from train_p100a_global_teacher_oof import softmax_numpy


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/p101_f3_causal_interaction_oof_v1"
F1_RUN = HERE / "runs/p101_f1_coarse_anchor_oof_v1"
P100_RUN = HERE / "runs/p100a_a0_global_teacher_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def training_audit(run: Path) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold in range(4):
        payload = torch.load(
            run / "F3_VSI" / f"fold{fold}_final.pt",
            map_location="cpu",
            weights_only=False,
        )
        history = payload["history"]
        nested_rows = sum(
            value["prediction"]["rows"]
            for value in payload["nested_anchor_audits"].values()
        )
        source_accuracy = 1.0 - payload["source_safe_error_rows"] / nested_rows
        final = history[-1]
        folds.append(
            {
                "fold": fold,
                "source_safe_anchor_accuracy": source_accuracy,
                "final_accuracy": float(final["source_safe_accuracy"]),
                "source_safe_accuracy_gain_pp": float(
                    100.0 * (final["source_safe_accuracy"] - source_accuracy)
                ),
                "positive_pair_gate": float(final["pair_gate"]),
                "negative_pair_gate": float(final["negative_pair_gate"]),
                "aligned_residual_rms": float(final["residual"]),
                "negative_residual_rms": float(final["negative_residual"]),
                "negative_anchor_kl": float(final["negative_kl"]),
                "final_pair_loss": float(final["pair"]),
                "per_anchor_accuracy": final["per_anchor_accuracy"],
            }
        )
    return {"folds": folds}


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    fine = load_p101_data()
    with np.load(run / "F3_VSI_complete_oof.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(P100_RUN / "VS_complete_oof.npz", allow_pickle=False) as archive:
        anchor_logits = np.asarray(archive["direct_logits"], dtype=np.float32)
        anchor_probability = np.asarray(archive["direct_probability"], dtype=np.float32)
    if not np.array_equal(arrays["sample_ids"].astype(str), fine.sample_ids):
        raise RuntimeError("P101-F3 mechanism audit row order changed")
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
    counterfactuals: dict[str, Any] = {}
    causal_delta: dict[str, Any] = {}
    for name in (
        "zero_imu",
        "reverse_imu",
        "shuffle_imu",
        "local_reverse_skeleton",
        "zero_skeleton",
        "shuffle_skeleton",
    ):
        logits = np.asarray(arrays[f"{name}_logits"], dtype=np.float32)
        probability = softmax_numpy(logits)
        prediction = probability.argmax(axis=1)
        counterfactuals[name] = {
            "counterfactual_to_aligned": paired(prediction, direct_prediction, labels),
            "top5_counterfactual_to_aligned": paired_binary(
                topk_correct(probability, labels, 5), direct_top5
            ),
            "mean_probability_absolute_difference": float(
                np.abs(probability - direct_probability).mean()
            ),
            "max_probability_absolute_difference": float(
                np.abs(probability - direct_probability).max()
            ),
        }
        if name in {"zero_imu", "reverse_imu", "shuffle_imu"}:
            delta = logits - anchor_logits
            difference = np.sqrt(np.mean((direct_delta - delta) ** 2, axis=1))
            dot = (direct_delta * delta).sum(axis=1)
            norm = np.linalg.norm(direct_delta, axis=1) * np.linalg.norm(delta, axis=1)
            cosine = dot / np.maximum(norm, 1e-12)
            causal_delta[name] = {
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
    per_fold = {
        str(fold): paired(
            anchor_prediction[fine.fold_ids == fold],
            direct_prediction[fine.fold_ids == fold],
            labels[fine.fold_ids == fold],
        )
        for fold in range(4)
    }
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
        "training_causal_separation": training_audit(run),
        "counterfactuals": counterfactuals,
        "causal_delta": causal_delta,
        "label_direction": {
            "anchor_wrong_rows": int((anchor_prediction != labels).sum()),
            "true_rank_improved_all": int((rank_direct < rank_anchor).sum()),
            "true_rank_harmed_all": int((rank_direct > rank_anchor).sum()),
            "true_rank_improved_anchor_wrong": int(
                (rank_direct[anchor_prediction != labels] < rank_anchor[anchor_prediction != labels]).sum()
            ),
            "true_rank_harmed_anchor_wrong": int(
                (rank_direct[anchor_prediction != labels] > rank_anchor[anchor_prediction != labels]).sum()
            ),
            "true_logit_advantage_anchor_wrong": describe(
                true_advantage[anchor_prediction != labels]
            ),
            "fraction_anchor_wrong_true_advantage_positive": float(
                (true_advantage[anchor_prediction != labels] > 0).mean()
            ),
        },
        "class_changes": class_changes(
            labels,
            anchor_prediction,
            direct_prediction,
            anchor_top5,
            direct_top5,
        ),
        "per_fold": per_fold,
        "route_audit": {
            "causal_interaction_verified": True,
            "credible_effect_gate_passed": False,
            "student_raw_allowed": False,
            "h3_allowed": False,
            "b_teacher_allowed": False,
            "next_non_sweep_requirement": "new cross-subject information or representation, not residual scale/loss tuning",
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
