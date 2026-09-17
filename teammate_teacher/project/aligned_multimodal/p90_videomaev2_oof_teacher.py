"""Three-fold OOF evaluation for the VideoMAE V2 distilled visual teacher.

This stage is intentionally separate from the fold-0 gate.  It is run only
after ``p90_videomaev2_distilled_teacher.py`` clears that gate, and it reuses
the frozen feature cache so no video forward pass is repeated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import log_softmax

from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomaev2_distilled_teacher import (
    P85_OOF,
    P85_STRONG,
    accuracy,
    aligned,
    class_sample_weights,
    feature_sets,
)
from train_p46_videomae_head import make_model


OUTPUT = REPO_ROOT / "runs" / "p90_videomaev2_distilled_teacher_v1"
CACHE = OUTPUT / "complete_features.npz"


def main() -> None:
    protocol = load_protocol()
    with np.load(CACHE, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    if not np.array_equal(data["sample_ids"].astype(str), protocol.sample_ids):
        raise ValueError("VideoMAE V2 cache and P90 protocol differ")

    recipes = {
        "early_late": (3000.0, 0.75),
        "window_mean": (1000.0, 0.75),
        "temporal_delta": (3000.0, 0.75),
        "k710_logits": (1000.0, 0.50),
        "early_late_plus_k710": (3000.0, 0.75),
    }
    matrices = feature_sets(data)
    labels = protocol.labels
    candidate_logits = {
        name: np.zeros((len(labels), 40), dtype=np.float32) for name in matrices
    }
    fold_results: dict[str, dict[str, object]] = {}

    for fold in range(3):
        train_indices = protocol.train_indices(fold)
        val_indices = protocol.val_indices(fold)
        fold_results[str(fold)] = {}
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
            candidate_logits[name][val_indices] = logits
            fold_results[str(fold)][name] = classification_metrics(
                logits, labels[val_indices]
            )
            print(
                f"fold={fold} candidate={name} "
                f"accuracy={fold_results[str(fold)][name]['accuracy']:.6f}",
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

    baselines = {
        "p85_visual_ridge": classification_metrics(p85_visual, labels),
        "p85_fullwindow_multimodal": classification_metrics(p85_strong, labels),
    }
    aggregate: dict[str, object] = {}
    for name, logits in candidate_logits.items():
        candidate_logp = log_softmax(logits.astype(np.float64), axis=1)
        blends: dict[str, object] = {}
        for base_name, base_scores in (
            ("p85_visual_ridge", log_softmax(p85_visual, axis=1)),
            ("p85_fullwindow_multimodal", p85_strong),
        ):
            fixed: dict[str, object] = {}
            for weight in (0.10, 0.25, 0.50):
                scores = (1.0 - weight) * base_scores + weight * candidate_logp
                fixed[str(weight)] = {
                    "metrics": classification_metrics(scores, labels),
                    "fold_accuracy": [
                        accuracy(scores[protocol.val_indices(fold)], labels[protocol.val_indices(fold)])
                        for fold in range(3)
                    ],
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
            base_prediction = base_scores.argmax(axis=1)
            candidate_prediction = candidate_logp.argmax(axis=1)
            correct = labels
            blends[base_name] = {
                "fixed_weights": fixed,
                "diagnostic_best_weight": best_weight,
                "diagnostic_best_accuracy": best_accuracy,
                "pair_oracle_accuracy": float(
                    np.mean(
                        (base_prediction == correct)
                        | (candidate_prediction == correct)
                    )
                ),
                "candidate_rescues": int(
                    np.sum(
                        (base_prediction != correct)
                        & (candidate_prediction == correct)
                    )
                ),
                "candidate_harms": int(
                    np.sum(
                        (base_prediction == correct)
                        & (candidate_prediction != correct)
                    )
                ),
            }
        aggregate[name] = {
            "recipe": {
                "alpha": recipes[name][0],
                "class_weight_power": recipes[name][1],
                "dimensions": int(matrices[name].shape[1]),
            },
            "metrics": classification_metrics(logits, labels),
            "fold_accuracy": [
                float(fold_results[str(fold)][name]["accuracy"])
                for fold in range(3)
            ],
            "blends": blends,
        }

    best_name = max(
        aggregate,
        key=lambda name: aggregate[name]["metrics"]["accuracy"],
    )
    report = {
        "protocol": (
            "P90 fixed subject-disjoint 3-fold OOF; executed after the fold-0 "
            "gate cleared; feature recipes frozen before folds 1 and 2"
        ),
        "samples": int(len(labels)),
        "baselines": baselines,
        "candidates": aggregate,
        "best_oof_candidate": best_name,
    }
    np.savez_compressed(
        OUTPUT / "oof_logits.npz",
        sample_ids=protocol.sample_ids,
        labels=labels,
        users=protocol.users,
        folds=protocol.fold_id,
        **{
            f"{name}_logits": values.astype(np.float32)
            for name, values in candidate_logits.items()
        },
    )
    (OUTPUT / "oof_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
