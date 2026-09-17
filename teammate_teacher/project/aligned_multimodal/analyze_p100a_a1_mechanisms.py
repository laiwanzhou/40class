"""Mechanism audit for the frozen P100-A1 protected IMU OOF result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p100a_global_teacher_data import load_p100a_data
from train_p100a_global_teacher_oof import paired_comparison, softmax_numpy


HERE = Path(__file__).resolve().parent
DEFAULT_A1 = HERE / "runs/p100a_a1_protected_imu_oof_v1"
DEFAULT_A0 = HERE / "runs/p100a_a0_global_teacher_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=Path, default=DEFAULT_A1)
    parser.add_argument("--a0", type=Path, default=DEFAULT_A0)
    return parser.parse_args()


def true_rank(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    truth = probability[np.arange(len(labels)), labels]
    return 1 + (probability > truth[:, None]).sum(axis=1)


def delta_summary(delta: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(len(delta)),
        "mean": float(delta.mean()),
        "mean_abs": float(np.abs(delta).mean()),
        "positive": int((delta > 0).sum()),
        "negative": int((delta < 0).sum()),
        "zero": int((delta == 0).sum()),
        "min": float(delta.min()),
        "max": float(delta.max()),
    }


def probability_audit(
    candidate: np.ndarray,
    anchor: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
) -> dict[str, Any]:
    rows = np.arange(len(labels))
    candidate_prediction = candidate.argmax(axis=1)
    anchor_prediction = anchor.argmax(axis=1)
    anchor_correct = anchor_prediction == labels
    truth_delta = candidate[rows, labels] - anchor[rows, labels]
    candidate_rank = true_rank(candidate, labels)
    anchor_rank = true_rank(anchor, labels)
    output = {
        "argmax_changed": int((candidate_prediction != anchor_prediction).sum()),
        "true_probability_delta_all": delta_summary(truth_delta),
        "true_probability_delta_anchor_correct": delta_summary(
            truth_delta[anchor_correct]
        ),
        "true_probability_delta_anchor_wrong": delta_summary(
            truth_delta[~anchor_correct]
        ),
        "true_rank_improved": int((candidate_rank < anchor_rank).sum()),
        "true_rank_worsened": int((candidate_rank > anchor_rank).sum()),
        "top3_rescue": int(((candidate_rank <= 3) & (anchor_rank > 3)).sum()),
        "top3_harm": int(((candidate_rank > 3) & (anchor_rank <= 3)).sum()),
        "top5_rescue": int(((candidate_rank <= 5) & (anchor_rank > 5)).sum()),
        "top5_harm": int(((candidate_rank > 5) & (anchor_rank <= 5)).sum()),
        "per_subject": {},
    }
    for user in sorted(np.unique(users).tolist()):
        mask = users == user
        output["per_subject"][user] = {
            "mean_true_probability_delta": float(truth_delta[mask].mean()),
            "mean_absolute_probability_change": float(
                np.abs(candidate[mask] - anchor[mask]).mean()
            ),
            "true_rank_improved": int((candidate_rank[mask] < anchor_rank[mask]).sum()),
            "true_rank_worsened": int((candidate_rank[mask] > anchor_rank[mask]).sum()),
        }
    return output


def changed_rows(
    candidate: np.ndarray,
    control: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    sample_ids: np.ndarray,
) -> list[dict[str, Any]]:
    candidate_prediction = candidate.argmax(axis=1)
    control_prediction = control.argmax(axis=1)
    indices = np.flatnonzero(candidate_prediction != control_prediction)
    return [
        {
            "row": int(row),
            "sample_id": str(sample_ids[row]),
            "user": str(users[row]),
            "label": int(labels[row]),
            "control_prediction": int(control_prediction[row]),
            "candidate_prediction": int(candidate_prediction[row]),
            "control_correct": bool(control_prediction[row] == labels[row]),
            "candidate_correct": bool(candidate_prediction[row] == labels[row]),
        }
        for row in indices
    ]


def adapter_scales(a1: Path) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    train_anchor_errors: list[int] = []
    all_scales: list[float] = []
    for fold in range(4):
        checkpoint = torch.load(
            a1 / f"fold{fold}_final.pt", map_location="cpu", weights_only=False
        )
        state = checkpoint["state_dict"]
        max_scale = float(checkpoint["model_config"]["protected_imu_max_scale"])
        values: dict[str, float] = {}
        for name, tensor in state.items():
            if ".protected_scale." not in name:
                continue
            scale = max_scale * float(torch.tanh(tensor).item())
            values[name] = scale
            all_scales.append(scale)
        final_history = checkpoint["history"][-1]
        anchor_wrong_estimate = int(
            round(
                (1.0 - float(checkpoint["history"][0]["train_accuracy"]))
                * ({0: 1481, 1: 1458, 2: 1428, 3: 1456}[fold])
            )
        )
        train_anchor_errors.append(anchor_wrong_estimate)
        folds[str(fold)] = {
            "held_users": list(checkpoint["held_users"]),
            "exact_anchor_max_abs_logit_error": float(
                checkpoint["exact_anchor_max_abs_logit_error"]
            ),
            "estimated_anchor_wrong_training_rows": anchor_wrong_estimate,
            "final_protected_kl": float(final_history["protected_kl"]),
            "scales": values,
            "mean_abs_scale": float(np.abs(list(values.values())).mean()),
            "max_abs_scale": float(np.abs(list(values.values())).max()),
        }
    return {
        "folds": folds,
        "estimated_anchor_wrong_training_rows_total": int(sum(train_anchor_errors)),
        "mean_signed_scale": float(np.mean(all_scales)),
        "mean_abs_scale": float(np.abs(all_scales).mean()),
        "max_abs_scale": float(np.abs(all_scales).max()),
        "configured_absolute_bound": 0.5,
    }


def main() -> None:
    args = parse_args()
    a1 = args.a1.resolve()
    a0 = args.a0.resolve()
    data = load_p100a_data()
    with np.load(a1 / "A1_complete_oof.npz", allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        direct_logits = np.asarray(archive["direct_logits"], dtype=np.float32)
        zero_imu_logits = np.asarray(archive["zero_imu_logits"], dtype=np.float32)
        shuffle_imu_logits = np.asarray(archive["shuffle_imu_logits"], dtype=np.float32)
    with np.load(a0 / "VS_complete_oof.npz", allow_pickle=False) as archive:
        source_ids = archive["sample_ids"].astype(str)
        vs_logits = np.asarray(archive["direct_logits"], dtype=np.float32)
        vs_probability = np.asarray(archive["direct_probability"], dtype=np.float32)
    if not np.array_equal(sample_ids, data.sample_ids.astype(str)):
        raise RuntimeError("A1 row contract changed")
    if not np.array_equal(source_ids, sample_ids):
        raise RuntimeError("VS/A1 row contract mismatch")

    direct_probability = softmax_numpy(direct_logits)
    zero_probability = softmax_numpy(zero_imu_logits)
    shuffle_probability = softmax_numpy(shuffle_imu_logits)
    output = {
        "rows": int(len(data.labels)),
        "h3_rows_loaded": 0,
        "zero_imu_reproduces_source_vs": {
            "max_abs_logit_error": float(np.abs(zero_imu_logits - vs_logits).max()),
            "max_abs_probability_error": float(
                np.abs(zero_probability - vs_probability).max()
            ),
            "argmax_disagreement": int(
                (zero_probability.argmax(axis=1) != vs_probability.argmax(axis=1)).sum()
            ),
        },
        "direct_vs_zero_imu": paired_comparison(
            direct_probability, zero_probability, data.labels, data.users
        ),
        "direct_vs_shuffle_imu": paired_comparison(
            direct_probability, shuffle_probability, data.labels, data.users
        ),
        "direct_vs_vs_changed_rows": changed_rows(
            direct_probability,
            vs_probability,
            data.labels,
            data.users,
            data.sample_ids,
        ),
        "direct_vs_shuffle_changed_rows": changed_rows(
            direct_probability,
            shuffle_probability,
            data.labels,
            data.users,
            data.sample_ids,
        ),
        "direct_vs_vs_probability_mechanism": probability_audit(
            direct_probability, vs_probability, data.labels, data.users
        ),
        "direct_vs_zero_logit_delta": delta_summary(
            (direct_logits - zero_imu_logits).reshape(-1)
        ),
        "adapter": adapter_scales(a1),
        "diagnosis": {
            "imu_adapter_changed_any_top1": bool(
                (direct_probability.argmax(axis=1) != vs_probability.argmax(axis=1)).any()
            ),
            "aligned_imu_outperformed_shuffle_top1": bool(
                (direct_probability.argmax(axis=1) == data.labels).sum()
                > (shuffle_probability.argmax(axis=1) == data.labels).sum()
            ),
            "mechanism": "PROTECTION_COLLAPSE_TO_VS_ANCHOR",
            "structural_cause": (
                "The fold-trained VS anchor is already nearly perfect on its own "
                "training subjects, so the correctness-conditioned protection loss "
                "exposes too few anchor-wrong rows to learn cross-subject IMU rescue."
            ),
            "post_hoc_hyperparameter_sweep_authorized": False,
        },
    }
    (a1 / "mechanism_analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output["diagnosis"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
