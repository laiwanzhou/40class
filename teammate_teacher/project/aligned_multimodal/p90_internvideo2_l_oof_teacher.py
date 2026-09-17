"""Three-fold OOF evaluation after the InternVideo2-L fold-0 gate cleared."""

from __future__ import annotations

import json

import numpy as np
from scipy.special import log_softmax

from p90_internvideo2_l_teacher import DEFAULT_OUTPUT, feature_sets
from p90_teacher_common import classification_metrics, load_protocol
from p90_videomaev2_distilled_teacher import (
    P85_OOF,
    P85_STRONG,
    accuracy,
    aligned,
    class_sample_weights,
)
from train_p46_videomae_head import make_model


P90_BASE_OOF = (
    DEFAULT_OUTPUT.parent / "p90_videomaev2_distilled_teacher_v1/oof_logits.npz"
)


def fold_accuracy(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> list[float]:
    return [accuracy(scores[folds == fold], labels[folds == fold]) for fold in range(3)]


def main() -> None:
    protocol = load_protocol()
    cache = DEFAULT_OUTPUT / "complete_features.npz"
    with np.load(cache, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    if not np.array_equal(data["sample_ids"].astype(str), protocol.sample_ids):
        raise ValueError("InternVideo2 cache and P90 protocol differ")

    recipes = {
        "early_late": (3000.0, 0.75),
        "window_mean": (1000.0, 0.75),
        "temporal_delta": (3000.0, 0.75),
        "k400_logits": (1000.0, 0.50),
        "early_late_plus_k400": (3000.0, 0.75),
    }
    matrices = feature_sets(data)
    labels = protocol.labels
    candidates = {
        name: np.zeros((len(labels), 40), dtype=np.float32) for name in matrices
    }
    per_fold: dict[str, dict[str, object]] = {}
    for fold in range(3):
        train_indices = protocol.train_indices(fold)
        val_indices = protocol.val_indices(fold)
        per_fold[str(fold)] = {}
        for name, values in matrices.items():
            alpha, power = recipes[name]
            model = make_model(alpha)
            model.fit(
                values[train_indices],
                labels[train_indices],
                ridge__sample_weight=class_sample_weights(
                    labels[train_indices], power
                ),
            )
            logits = np.asarray(
                model.decision_function(values[val_indices]), dtype=np.float32
            )
            candidates[name][val_indices] = logits
            metrics = classification_metrics(logits, labels[val_indices])
            per_fold[str(fold)][name] = metrics
            print(
                f"fold={fold} candidate={name} accuracy={metrics['accuracy']:.6f}",
                flush=True,
            )

    with np.load(P85_OOF, allow_pickle=False) as source:
        p85_visual = aligned(
            source["sample_ids"], source["early_late_logits"], protocol.sample_ids
        ).astype(np.float64)
    with np.load(P85_STRONG, allow_pickle=False) as source:
        p85_strong = np.log(
            np.clip(
                aligned(
                    source["oof_sample_ids"],
                    source["oof_teacher_probability"],
                    protocol.sample_ids,
                ),
                1e-8,
                1.0,
            )
        )
    with np.load(P90_BASE_OOF, allow_pickle=False) as source:
        p90_base = aligned(
            source["sample_ids"], source["early_late_logits"], protocol.sample_ids
        ).astype(np.float64)
    bases = {
        "p85_visual_ridge": log_softmax(p85_visual, axis=1),
        "p85_fullwindow_multimodal": p85_strong,
        "p90_videomaev2_distilled_base": log_softmax(p90_base, axis=1),
    }
    baseline_report = {
        name: {
            "metrics": classification_metrics(scores, labels),
            "fold_accuracy": fold_accuracy(scores, labels, protocol.fold_id),
        }
        for name, scores in bases.items()
    }

    candidate_report: dict[str, object] = {}
    for name, logits in candidates.items():
        candidate_logp = log_softmax(logits.astype(np.float64), axis=1)
        pair_blends: dict[str, object] = {}
        for base_name, base_scores in bases.items():
            fixed: dict[str, object] = {}
            for weight in (0.10, 0.25, 0.50):
                scores = (1.0 - weight) * base_scores + weight * candidate_logp
                fixed[str(weight)] = {
                    "metrics": classification_metrics(scores, labels),
                    "fold_accuracy": fold_accuracy(scores, labels, protocol.fold_id),
                }
            scan = [
                (
                    float(weight),
                    accuracy(
                        (1.0 - weight) * base_scores + weight * candidate_logp,
                        labels,
                    ),
                )
                for weight in np.arange(0.0, 1.0001, 0.025)
            ]
            best_weight, best_accuracy = max(scan, key=lambda row: row[1])
            pair_blends[base_name] = {
                "fixed_weights": fixed,
                "diagnostic_best_weight": best_weight,
                "diagnostic_best_accuracy": best_accuracy,
            }

        triple_fixed: dict[str, object] = {}
        for label, weights in {
            "equal_thirds": (1 / 3, 1 / 3, 1 / 3),
            "p85_half_base_quarter_iv2_quarter": (0.50, 0.25, 0.25),
            "p85_quarter_base_three_eighths_iv2_three_eighths": (
                0.25,
                0.375,
                0.375,
            ),
        }.items():
            p85_weight, base_weight, iv2_weight = weights
            scores = (
                p85_weight * p85_strong
                + base_weight * bases["p90_videomaev2_distilled_base"]
                + iv2_weight * candidate_logp
            )
            triple_fixed[label] = {
                "weights_p85_p90base_iv2": list(weights),
                "metrics": classification_metrics(scores, labels),
                "fold_accuracy": fold_accuracy(scores, labels, protocol.fold_id),
            }
        grid = []
        for p85_weight in np.arange(0.0, 1.0001, 0.05):
            for base_weight in np.arange(0.0, 1.0001 - p85_weight, 0.05):
                iv2_weight = 1.0 - p85_weight - base_weight
                scores = (
                    p85_weight * p85_strong
                    + base_weight * bases["p90_videomaev2_distilled_base"]
                    + iv2_weight * candidate_logp
                )
                grid.append(
                    (
                        accuracy(scores, labels),
                        float(p85_weight),
                        float(base_weight),
                        float(iv2_weight),
                    )
                )
        diagnostic = max(grid)
        candidate_report[name] = {
            "recipe": {
                "alpha": recipes[name][0],
                "class_weight_power": recipes[name][1],
                "dimensions": int(matrices[name].shape[1]),
            },
            "metrics": classification_metrics(logits, labels),
            "fold_accuracy": fold_accuracy(logits, labels, protocol.fold_id),
            "pair_blends": pair_blends,
            "triple_fixed": triple_fixed,
            "triple_diagnostic_best": {
                "accuracy": diagnostic[0],
                "weights_p85_p90base_iv2": list(diagnostic[1:]),
            },
        }

    report = {
        "protocol": (
            "P90 fixed subject-disjoint 3-fold OOF; executed after a clear fold-0 "
            "ensemble gate; Ridge recipes frozen before folds 1 and 2"
        ),
        "samples": int(len(labels)),
        "baselines": baseline_report,
        "candidates": candidate_report,
        "best_standalone": max(
            candidate_report,
            key=lambda name: candidate_report[name]["metrics"]["accuracy"],
        ),
    }
    np.savez_compressed(
        DEFAULT_OUTPUT / "oof_logits.npz",
        sample_ids=protocol.sample_ids,
        labels=labels,
        users=protocol.users,
        folds=protocol.fold_id,
        **{
            f"{name}_logits": values.astype(np.float32)
            for name, values in candidates.items()
        },
    )
    (DEFAULT_OUTPUT / "oof_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
