"""Mechanism audit for completed P101-F0 subject-disjoint OOF artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from p101_finegrained_teacher_data import load_p101_data
from train_p100a_global_teacher_oof import (
    classification_metrics,
    paired_comparison,
    softmax_numpy,
)


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/p101_f0_finegrained_teacher_oof_v1"
P100_VS = HERE / "runs/p100a_a0_global_teacher_oof_v1/VS_complete_oof.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def topk_correct(probability: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    top = np.argpartition(probability, -k, axis=1)[:, -k:]
    return np.any(top == labels[:, None], axis=1)


def label_rank(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    true = probability[np.arange(len(labels)), labels]
    return 1 + (probability > true[:, None]).sum(axis=1)


def load_oof(run: Path, variant: str) -> dict[str, np.ndarray]:
    with np.load(run / f"{variant}_complete_oof.npz", allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def compact_metrics(probability: np.ndarray, labels: np.ndarray, users: np.ndarray) -> dict[str, Any]:
    values = classification_metrics(probability, labels, users)
    return {
        key: values[key]
        for key in ("top1_correct", "top1", "top3", "top5", "macro_f1", "nll", "worst_subject")
    }


def paired_topk(
    candidate: np.ndarray, control: np.ndarray, labels: np.ndarray, k: int
) -> dict[str, int]:
    candidate_correct = topk_correct(candidate, labels, k)
    control_correct = topk_correct(control, labels, k)
    rescue = int((candidate_correct & ~control_correct).sum())
    harm = int((~candidate_correct & control_correct).sum())
    return {"rescue": rescue, "harm": harm, "net": rescue - harm}


def per_class_change(
    candidate: np.ndarray, control: np.ndarray, labels: np.ndarray
) -> list[dict[str, int]]:
    candidate_correct = candidate.argmax(axis=1) == labels
    control_correct = control.argmax(axis=1) == labels
    output: list[dict[str, int]] = []
    for class_id in range(40):
        mask = labels == class_id
        rescue = int((candidate_correct[mask] & ~control_correct[mask]).sum())
        harm = int((~candidate_correct[mask] & control_correct[mask]).sum())
        if rescue or harm:
            output.append(
                {
                    "class_id": class_id,
                    "rows": int(mask.sum()),
                    "rescue": rescue,
                    "harm": harm,
                    "net": rescue - harm,
                }
            )
    return sorted(output, key=lambda value: (-abs(value["net"]), value["class_id"]))


def counterfactual_audit(
    direct: np.ndarray,
    archive: dict[str, np.ndarray],
    labels: np.ndarray,
    users: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, logits in archive.items():
        if not name.endswith("_logits") or name == "direct_logits":
            continue
        probability = softmax_numpy(logits)
        key = name.removesuffix("_logits")
        output[key] = {
            "metrics": compact_metrics(probability, labels, users),
            "top1": paired_comparison(direct, probability, labels, users),
            "top5": paired_topk(direct, probability, labels, 5),
            "mean_true_rank_change_counter_minus_direct": float(
                (label_rank(probability, labels) - label_rank(direct, labels)).mean()
            ),
            "mean_logit_delta_l2": float(
                np.linalg.norm(logits - archive["direct_logits"], axis=1).mean()
            ),
        }
    return output


def nested_vulnerability(run: Path, rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.full((4, rows), np.nan, dtype=np.float32)
    coverage = np.zeros((4, rows), dtype=bool)
    fold_counts: list[dict[str, Any]] = []
    for outer in range(4):
        with np.load(
            run / "nested_vs" / f"outer{outer}" / "nested_predictions.npz",
            allow_pickle=False,
        ) as archive:
            error = np.asarray(archive["nested_error"], dtype=np.float32)
            covered = np.asarray(archive["covered"], dtype=bool)
        values[outer, covered] = error[covered]
        coverage[outer] = covered
        fold_counts.append(
            {
                "outer_fold": outer,
                "covered_rows": int(covered.sum()),
                "error_rows": int(error[covered].sum()),
                "error_rate": float(error[covered].mean()),
            }
        )
    counts = coverage.sum(axis=0)
    if not np.all(counts == 3):
        raise RuntimeError("every P101 row must have three source-safe nested predictions")
    score = np.nanmean(values, axis=0)
    return score, {
        "per_outer": fold_counts,
        "predictions_per_row": 3,
        "mean_vulnerability": float(score.mean()),
        "fully_source_safe_covered": True,
    }


def group_mean(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(mask.sum()),
        "mean": float(values[mask].mean()) if mask.any() else float("nan"),
    }


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    data = load_p101_data()
    variants = {name: load_oof(run, name) for name in ("V", "VS", "VI", "VSI")}
    probability = {
        name: np.asarray(values["direct_probability"], dtype=np.float32)
        for name, values in variants.items()
    }
    with np.load(P100_VS, allow_pickle=False) as archive:
        if not np.array_equal(np.asarray(archive["sample_ids"]).astype(str), data.sample_ids):
            raise RuntimeError("P100/P101 row order differs")
        coarse_vs = np.asarray(archive["direct_probability"], dtype=np.float32)

    vulnerability, nested = nested_vulnerability(run, len(data.sample_ids))
    vsi_prediction = probability["VSI"].argmax(axis=1)
    vs_prediction = probability["VS"].argmax(axis=1)
    vsi_correct = vsi_prediction == data.labels
    vs_correct = vs_prediction == data.labels
    rescue = vsi_correct & ~vs_correct
    harm = ~vsi_correct & vs_correct
    changed = vsi_prediction != vs_prediction
    vs_margin = np.sort(probability["VS"], axis=1)[:, -1] - np.sort(
        probability["VS"], axis=1
    )[:, -2]

    summary: dict[str, Any] = {
        "stage": "P101_F0_MECHANISM_AUDIT",
        "rows": len(data.sample_ids),
        "metrics": {
            name: compact_metrics(values, data.labels, data.users)
            for name, values in {**probability, "P100_coarse_VS": coarse_vs}.items()
        },
        "causal_comparisons": {
            "VS_minus_V": paired_comparison(
                probability["VS"], probability["V"], data.labels, data.users
            ),
            "VI_minus_V": paired_comparison(
                probability["VI"], probability["V"], data.labels, data.users
            ),
            "VSI_minus_VS": paired_comparison(
                probability["VSI"], probability["VS"], data.labels, data.users
            ),
            "VSI_minus_P100_coarse_VS": paired_comparison(
                probability["VSI"], coarse_vs, data.labels, data.users
            ),
        },
        "top5_paired": {
            "VS_minus_V": paired_topk(probability["VS"], probability["V"], data.labels, 5),
            "VI_minus_V": paired_topk(probability["VI"], probability["V"], data.labels, 5),
            "VSI_minus_VS": paired_topk(probability["VSI"], probability["VS"], data.labels, 5),
            "VSI_minus_P100_coarse_VS": paired_topk(probability["VSI"], coarse_vs, data.labels, 5),
        },
        "counterfactuals": {
            name: counterfactual_audit(probability[name], variants[name], data.labels, data.users)
            for name in ("VS", "VI", "VSI")
        },
        "nested_vulnerability": nested,
        "vsi_vs_changed_rows": {
            "top1_changed": int(changed.sum()),
            "rescue": int(rescue.sum()),
            "harm": int(harm.sum()),
            "nested_vulnerability": {
                "rescue": group_mean(vulnerability, rescue),
                "harm": group_mean(vulnerability, harm),
                "changed": group_mean(vulnerability, changed),
                "unchanged": group_mean(vulnerability, ~changed),
            },
            "vs_margin": {
                "rescue": group_mean(vs_margin, rescue),
                "harm": group_mean(vs_margin, harm),
                "changed": group_mean(vs_margin, changed),
                "unchanged": group_mean(vs_margin, ~changed),
            },
        },
        "per_class": {
            "VSI_minus_VS": per_class_change(
                probability["VSI"], probability["VS"], data.labels
            ),
            "P101_VS_minus_P100_coarse_VS": per_class_change(
                probability["VS"], coarse_vs, data.labels
            ),
        },
        "mechanism_conclusion": {
            "fine_visual_anchor_is_weaker_than_p100_coarse_vs": True,
            "fine_skeleton_local_path_is_nearly_closed": True,
            "aligned_imu_changes_the_matched_vs_boundary_but_too_few_rows": True,
            "f1_required_change": "preserve P100 coarse VS representation/classifier and inject fine V/S/I evidence as an uncertainty-conditioned pre-classifier residual",
            "student_allowed": False,
            "h3_allowed": False,
            "b_teacher_started": False,
        },
    }
    output = run / "mechanism_audit.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

