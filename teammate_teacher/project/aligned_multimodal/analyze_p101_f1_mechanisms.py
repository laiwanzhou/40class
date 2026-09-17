"""Mechanism audit for the completed P101-F1 subject-disjoint OOF run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p100a_global_teacher_data import (
    CANONICAL_VARIANTS as P100_VARIANTS,
    P100ADataset,
    load_p100a_data,
)
from p101_finegrained_teacher_data import load_p101_data
from train_p100a_global_teacher_oof import evaluate_model, softmax_numpy
from train_p101_f1_coarse_anchor_oof import load_outer_anchor, make_loader


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/p101_f1_coarse_anchor_oof_v1"
P100_RUN = HERE / "runs/p100a_a0_global_teacher_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def paired(before: np.ndarray, after: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    before_correct = before == labels
    after_correct = after == labels
    return {
        "rescue": int((~before_correct & after_correct).sum()),
        "harm": int((before_correct & ~after_correct).sum()),
        "net": int(after_correct.sum() - before_correct.sum()),
        "prediction_changes": int((before != after).sum()),
        "wrong_to_wrong": int((~before_correct & ~after_correct & (before != after)).sum()),
    }


def topk_correct(probability: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    top = np.argpartition(probability, -k, axis=1)[:, -k:]
    return np.any(top == labels[:, None], axis=1)


def paired_binary(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    return {
        "rescue": int((~before & after).sum()),
        "harm": int((before & ~after).sum()),
        "net": int(after.sum() - before.sum()),
    }


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "rows": int(values.size),
        "mean": float(values.mean()) if values.size else float("nan"),
        "std": float(values.std()) if values.size else float("nan"),
        "min": float(values.min()) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def true_rank(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    truth = probability[np.arange(len(labels)), labels]
    return 1 + (probability > truth[:, None]).sum(axis=1)


def nested_vulnerability(run: Path, rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    errors = np.zeros(rows, dtype=np.float32)
    counts = np.zeros(rows, dtype=np.int64)
    folds: list[dict[str, Any]] = []
    for fold in range(4):
        path = run / "nested_coarse_vs" / f"outer{fold}" / "nested_predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            covered = np.asarray(archive["covered"], dtype=bool)
            values = np.asarray(archive["nested_error"], dtype=np.float32)
        errors[covered] += values[covered]
        counts[covered] += 1
        folds.append(
            {
                "fold": fold,
                "covered_rows": int(covered.sum()),
                "error_rows": int(values[covered].sum()),
                "error_rate": float(values[covered].mean()),
            }
        )
    if not np.all(counts == 3):
        raise RuntimeError(
            f"P101-F1 nested vulnerability expected three source-safe values per row: "
            f"{np.unique(counts, return_counts=True)}"
        )
    return errors / counts, {"folds": folds, "values_per_row": 3}


def outer_in_domain_audit(
    run: Path, device: torch.device
) -> dict[str, Any]:
    coarse = load_p100a_data()
    folds: list[dict[str, Any]] = []
    totals = {
        "rows": 0,
        "outer_anchor_errors": 0,
        "nested_errors": 0,
        "both_wrong": 0,
        "nested_wrong_outer_correct": 0,
    }
    for fold in range(4):
        train, _ = coarse.indices_for_fold(fold)
        anchor, normalizer, _ = load_outer_anchor(fold, device)
        result = evaluate_model(
            anchor,
            make_loader(
                P100ADataset(coarse, train, normalizer, P100_VARIANTS["VS"]),
                64,
                False,
                9100 + fold,
            ),
            device,
        )
        rows = np.asarray(result["rows"], dtype=np.int64)
        outer_wrong = np.asarray(result["logits"]).argmax(axis=1) != coarse.labels[rows]
        with np.load(
            run / "nested_coarse_vs" / f"outer{fold}" / "nested_predictions.npz",
            allow_pickle=False,
        ) as archive:
            nested_wrong = np.asarray(archive["nested_error"], dtype=np.float32)[rows] > 0.5
        both = outer_wrong & nested_wrong
        record = {
            "fold": fold,
            "rows": int(len(rows)),
            "outer_anchor_error_rows": int(outer_wrong.sum()),
            "outer_anchor_error_rate": float(outer_wrong.mean()),
            "nested_error_rows": int(nested_wrong.sum()),
            "nested_error_rate": float(nested_wrong.mean()),
            "both_wrong_rows": int(both.sum()),
            "nested_wrong_outer_correct_rows": int((nested_wrong & ~outer_wrong).sum()),
            "fraction_nested_wrong_with_outer_label_gradient_already_correct": float(
                (nested_wrong & ~outer_wrong).sum() / max(int(nested_wrong.sum()), 1)
            ),
        }
        folds.append(record)
        totals["rows"] += record["rows"]
        totals["outer_anchor_errors"] += record["outer_anchor_error_rows"]
        totals["nested_errors"] += record["nested_error_rows"]
        totals["both_wrong"] += record["both_wrong_rows"]
        totals["nested_wrong_outer_correct"] += record[
            "nested_wrong_outer_correct_rows"
        ]
    totals["outer_anchor_error_rate"] = totals["outer_anchor_errors"] / totals["rows"]
    totals["nested_error_rate"] = totals["nested_errors"] / totals["rows"]
    totals["nested_wrong_outer_correct_fraction"] = totals[
        "nested_wrong_outer_correct"
    ] / max(totals["nested_errors"], 1)
    return {"folds": folds, "total": totals}


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    fine = load_p101_data()
    with np.load(run / "F1_VSI_complete_oof.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(P100_RUN / "VS_complete_oof.npz", allow_pickle=False) as archive:
        anchor_probability = np.asarray(archive["direct_probability"], dtype=np.float32)
        anchor_logits = np.asarray(archive["direct_logits"], dtype=np.float32)
    if not np.array_equal(arrays["sample_ids"].astype(str), fine.sample_ids):
        raise RuntimeError("P101-F1 mechanism audit row order changed")
    labels = fine.labels
    direct_probability = np.asarray(arrays["direct_probability"], dtype=np.float32)
    direct_prediction = direct_probability.argmax(axis=1)
    anchor_prediction = anchor_probability.argmax(axis=1)
    vulnerability, vulnerability_audit = nested_vulnerability(run, len(labels))
    changed = direct_prediction != anchor_prediction
    rescue = (anchor_prediction != labels) & (direct_prediction == labels)
    harm = (anchor_prediction == labels) & (direct_prediction != labels)
    stable = ~changed
    uncertainty = np.asarray(arrays["direct_uncertainty"], dtype=np.float32)
    residual = np.asarray(arrays["direct_fine_residual_rms"], dtype=np.float32)
    probability_sorted = np.sort(anchor_probability, axis=1)
    anchor_margin = probability_sorted[:, -1] - probability_sorted[:, -2]
    direct_delta = np.asarray(arrays["direct_logits"], dtype=np.float32) - anchor_logits
    true_delta = direct_delta[np.arange(len(labels)), labels]
    predicted_delta = direct_delta[np.arange(len(labels)), anchor_prediction]
    true_advantage = true_delta - predicted_delta
    anchor_wrong = anchor_prediction != labels
    cf_audit: dict[str, Any] = {}
    for name in (
        "local_reverse_skeleton",
        "zero_imu",
        "reverse_imu",
        "shuffle_imu",
        "zero_skeleton",
        "shuffle_skeleton",
        "zero_both",
    ):
        probability = softmax_numpy(np.asarray(arrays[f"{name}_logits"], dtype=np.float32))
        prediction = probability.argmax(axis=1)
        cf_audit[name] = {
            "direct_vs_counterfactual": paired(prediction, direct_prediction, labels),
            "prediction_disagreement_rows": int((prediction != direct_prediction).sum()),
            "top5_direct_vs_counterfactual": paired_binary(
                topk_correct(probability, labels, 5),
                topk_correct(direct_probability, labels, 5),
            ),
            "mean_probability_absolute_difference": float(
                np.abs(probability - direct_probability).mean()
            ),
            "max_probability_absolute_difference": float(
                np.abs(probability - direct_probability).max()
            ),
        }
    imu_delta_audit: dict[str, Any] = {}
    direct_delta_rms = np.sqrt(np.mean(direct_delta**2, axis=1))
    for name in ("reverse_imu", "shuffle_imu"):
        cf_delta = np.asarray(arrays[f"{name}_logits"], dtype=np.float32) - anchor_logits
        difference = np.sqrt(np.mean((direct_delta - cf_delta) ** 2, axis=1))
        dot = (direct_delta * cf_delta).sum(axis=1)
        norm = np.linalg.norm(direct_delta, axis=1) * np.linalg.norm(cf_delta, axis=1)
        cosine = dot / np.maximum(norm, 1e-12)
        imu_delta_audit[name] = {
            "aligned_delta_rms": describe(direct_delta_rms),
            "aligned_minus_counterfactual_delta_rms": describe(difference),
            "difference_over_aligned_ratio": float(
                difference.mean() / max(float(direct_delta_rms.mean()), 1e-12)
            ),
            "delta_cosine": describe(cosine),
        }
    rank_anchor = true_rank(anchor_probability, labels)
    rank_direct = true_rank(direct_probability, labels)
    changed_rows = np.flatnonzero(changed)
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
            "anchor_margin": float(anchor_margin[row]),
            "uncertainty": float(uncertainty[row]),
            "residual_rms": float(residual[row]),
            "nested_vulnerability": float(vulnerability[row]),
            "true_rank_before": int(rank_anchor[row]),
            "true_rank_after": int(rank_direct[row]),
            "true_logit_advantage": float(true_advantage[row]),
        }
        for row in changed_rows
    ]
    result = {
        "status": "complete",
        "run": str(run),
        "matched": paired(anchor_prediction, direct_prediction, labels),
        "top5": paired_binary(
            topk_correct(anchor_probability, labels, 5),
            topk_correct(direct_probability, labels, 5),
        ),
        "routing": {
            "nested_vulnerability_audit": vulnerability_audit,
            "nested_vulnerability_all": describe(vulnerability),
            "nested_vulnerability_changed": describe(vulnerability[changed]),
            "nested_vulnerability_stable": describe(vulnerability[stable]),
            "uncertainty_changed": describe(uncertainty[changed]),
            "uncertainty_stable": describe(uncertainty[stable]),
            "anchor_margin_changed": describe(anchor_margin[changed]),
            "anchor_margin_stable": describe(anchor_margin[stable]),
            "residual_changed": describe(residual[changed]),
            "residual_stable": describe(residual[stable]),
            "residual_uncertainty_correlation": safe_correlation(residual, uncertainty),
        },
        "label_direction": {
            "anchor_wrong_rows": int(anchor_wrong.sum()),
            "true_logit_advantage_on_anchor_wrong": describe(true_advantage[anchor_wrong]),
            "fraction_anchor_wrong_true_advantage_positive": float(
                (true_advantage[anchor_wrong] > 0).mean()
            ),
            "true_rank_improved_on_anchor_wrong_rows": int(
                (rank_direct[anchor_wrong] < rank_anchor[anchor_wrong]).sum()
            ),
            "true_rank_harmed_on_anchor_wrong_rows": int(
                (rank_direct[anchor_wrong] > rank_anchor[anchor_wrong]).sum()
            ),
            "true_logit_advantage_rescue": describe(true_advantage[rescue]),
            "true_logit_advantage_harm": describe(true_advantage[harm]),
        },
        "counterfactuals": cf_audit,
        "imu_class_content": imu_delta_audit,
        "outer_in_domain_supervision": outer_in_domain_audit(run, device),
        "per_fold": {
            str(fold): paired(
                anchor_prediction[fine.fold_ids == fold],
                direct_prediction[fine.fold_ids == fold],
                labels[fine.fold_ids == fold],
            )
            for fold in range(4)
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
