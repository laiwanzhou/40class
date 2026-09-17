"""H1 selection and one frozen H2 confirmation for the P96 dense24 teacher.

Candidate representation and blend selection are entirely source-OOF on
H1+embargo users.  Only the single selected recipe is fitted on all source
users and evaluated once on H2.  This file intentionally has no H3 path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomaev2_distilled_teacher import class_sample_weights
from p91_hierarchical_multimodal_teacher import audit
from p91_unrestricted_fusion_teacher import build_cohorts
from p94_candidate_multimodal_reranker import (
    ROUTER,
    audit as user_audit,
    h2_p91_champion,
    load_npz,
    router_base,
)
from train_p46_videomae_head import l2_normalize, make_model, row_standardize


P96_RUN = REPO_ROOT / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
A1_RUN = REPO_ROOT / "runs/p92_vjepa2_vitl_ssv2_12view_fold0_v1"
VMAE = REPO_ROOT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IV2 = REPO_ROOT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
OUTPUT = REPO_ROOT / "runs/p96_vjepa2_dense24_teacher_h1h2_v1"


def align_indices(all_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {str(value): row for row, value in enumerate(all_ids)}
    return np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)


def feature_candidates() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    done = np.asarray(np.load(P96_RUN / "done.npy", mmap_mode="r"), dtype=bool)
    if not done.all():
        raise RuntimeError(f"P96 dense24 extraction incomplete: {int(done.sum())}/{len(done)}")
    dense = l2_normalize(
        np.asarray(np.load(P96_RUN / "features.npy", mmap_mode="r"), dtype=np.float32)
    )
    action = row_standardize(
        np.asarray(np.load(P96_RUN / "ssv2_logits.npy", mmap_mode="r"), dtype=np.float32)
    )
    if dense.shape[1:] != (24, 1024) or action.shape[1:] != (24, 174):
        raise ValueError(f"unexpected dense cache: {dense.shape}/{action.shape}")
    dense_group = l2_normalize(dense.reshape(len(dense), 8, 3, 1024).mean(axis=2))
    action_group = row_standardize(action.reshape(len(action), 8, 3, 174).mean(axis=2))

    a1 = l2_normalize(
        np.asarray(np.load(A1_RUN / "features.npy", mmap_mode="r"), dtype=np.float32)
    )
    vmae = load_npz(VMAE)
    iv2 = load_npz(IV2)
    protocol = load_protocol()
    for name, ids in (("vmae", vmae["sample_ids"]), ("iv2", iv2["sample_ids"])):
        if not np.array_equal(ids.astype(str), protocol.sample_ids.astype(str)):
            raise ValueError(f"{name} sample order mismatch")
    old_ir = np.concatenate(
        (
            l2_normalize(vmae["features"].astype(np.float32)).reshape(len(dense), -1),
            l2_normalize(iv2["features"].astype(np.float32)).reshape(len(dense), -1),
        ),
        axis=1,
    )
    dense_group_flat = dense_group.reshape(len(dense), -1)
    dense_group_action = np.concatenate(
        (dense_group_flat, action_group.reshape(len(dense), -1)), axis=1
    )
    candidates = {
        "old_ir_vmae_iv2": old_ir,
        "a1_vjepa_all12": a1.reshape(len(a1), -1),
        "dense24_group8": dense_group_flat,
        "dense24_all24": dense.reshape(len(dense), -1),
        "dense24_group8_ssv2": dense_group_action,
        "old_ir_plus_dense24_group8": np.concatenate(
            (old_ir, dense_group_flat), axis=1
        ),
        "old_ir_plus_dense24_group8_ssv2": np.concatenate(
            (old_ir, dense_group_action), axis=1
        ),
    }
    recipes = {
        "old_ir_vmae_iv2": 5000.0,
        "a1_vjepa_all12": 5000.0,
        "dense24_group8": 3000.0,
        "dense24_all24": 7000.0,
        "dense24_group8_ssv2": 4000.0,
        "old_ir_plus_dense24_group8": 8000.0,
        "old_ir_plus_dense24_group8_ssv2": 9000.0,
    }
    return candidates, recipes


def aligned_scores(model: Any, values: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(values), dtype=np.float64)
    classes = np.asarray(model.named_steps["ridge"].classes_, dtype=np.int64)
    output = np.full((len(values), 40), scores.min() - 1.0, dtype=np.float64)
    output[:, classes] = scores
    return output


def fit_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = make_model(alpha)
    model.fit(
        train_x,
        train_y,
        ridge__sample_weight=class_sample_weights(train_y, 0.75),
    )
    return aligned_scores(model, eval_x)


def conservative_blend(logits: np.ndarray, base: np.ndarray, weight: float) -> np.ndarray:
    probability = softmax(logits, axis=1)
    anchor = np.full((len(base), 40), 0.04 / 39.0, dtype=np.float64)
    anchor[np.arange(len(base)), base] = 0.96
    score = weight * np.log(np.clip(probability, 1e-9, 1.0))
    score += (1.0 - weight) * np.log(np.clip(anchor, 1e-9, 1.0))
    return score.argmax(axis=1)


def topk(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    values = np.argpartition(logits, -k, axis=1)[:, -k:]
    return float(np.mean((values == labels[:, None]).any(axis=1)))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    cohorts = build_cohorts()
    h1 = cohorts["H1_selection"]
    embargo = cohorts["E0_p87_sequence_source"]
    h2 = cohorts["H2_confirmation"]
    source_ids = np.concatenate((h1.sample_ids, embargo.sample_ids))
    source_users = np.concatenate((h1.users, embargo.users)).astype(str)
    source_labels = np.concatenate((h1.labels, embargo.labels)).astype(np.int64)
    source_index = align_indices(protocol.sample_ids, source_ids)
    h2_index = align_indices(protocol.sample_ids, h2.sample_ids)
    router = load_npz(ROUTER)
    source_base = np.concatenate(
        (
            router_base(router, "H1_selection", h1.sample_ids),
            embargo.safe_prediction.astype(np.int64),
        )
    )
    h2_base = h2_p91_champion(h2.sample_ids)
    candidates, recipes = feature_candidates()

    source_results: dict[str, Any] = {}
    selected_rows = []
    source_logits: dict[str, np.ndarray] = {}
    for candidate_name, all_values in candidates.items():
        values = all_values[source_index]
        logits = np.zeros((len(source_labels), 40), dtype=np.float64)
        for user in np.unique(source_users):
            held = source_users == user
            logits[held] = fit_scores(
                values[~held],
                source_labels[~held],
                values[held],
                recipes[candidate_name],
            )
        source_logits[candidate_name] = logits
        direct = logits.argmax(axis=1)
        blend_grid = []
        for weight in np.linspace(0.0, 1.0, 41):
            prediction = conservative_blend(logits, source_base, float(weight))
            blend_grid.append(
                {
                    "weight": float(weight),
                    **user_audit(
                        source_labels, source_base, prediction, source_users
                    ),
                }
            )
        selected_blend = max(
            blend_grid,
            key=lambda row: (
                row["candidate_correct"],
                -row["harm"],
                -row["negative_users"],
                row["worst_user_net"],
                -row["weight"],
            ),
        )
        source_results[candidate_name] = {
            "dimensions": int(values.shape[1]),
            "alpha": recipes[candidate_name],
            "direct_metrics": classification_metrics(logits, source_labels),
            "direct_vs_anchor": user_audit(
                source_labels, source_base, direct, source_users
            ),
            "top5_coverage": topk(logits, source_labels, 5),
            "selected_blend": selected_blend,
            "top_blends": sorted(
                blend_grid,
                key=lambda row: (row["candidate_correct"], -row["harm"]),
                reverse=True,
            )[:5],
        }
        selected_rows.append(
            {"candidate": candidate_name, **selected_blend}
        )

    champion = max(
        selected_rows,
        key=lambda row: (
            row["candidate_correct"],
            -row["harm"],
            -row["negative_users"],
            row["worst_user_net"],
            -row["weight"],
        ),
    )
    candidate_name = str(champion["candidate"])
    weight = float(champion["weight"])
    all_values = candidates[candidate_name]
    h2_logits = fit_scores(
        all_values[source_index],
        source_labels,
        all_values[h2_index],
        recipes[candidate_name],
    )
    h2_direct = h2_logits.argmax(axis=1)
    h2_prediction = conservative_blend(h2_logits, h2_base, weight)
    source_prediction = conservative_blend(
        source_logits[candidate_name], source_base, weight
    )
    source_audit = user_audit(
        source_labels, source_base, source_prediction, source_users
    )
    h2_audit = user_audit(h2.labels, h2_base, h2_prediction, h2.users)
    source_gate = source_audit["net"] >= 8 and source_audit["worst_user_net"] >= -2
    h2_gate = h2_audit["net"] >= 8 and h2_audit["worst_user_net"] >= -2
    report = {
        "status": "complete_h1_h2_only",
        "protocol": (
            "Candidate and blend selected by LOUO H1+embargo only; selected recipe "
            "refitted on source and confirmed once on H2; no H3 path."
        ),
        "source_candidates": source_results,
        "selected_recipe": {
            "candidate": candidate_name,
            "alpha": recipes[candidate_name],
            "blend_weight": weight,
        },
        "source_selected": source_audit,
        "h2": {
            "direct_metrics": classification_metrics(h2_logits, h2.labels),
            "direct_vs_anchor": user_audit(
                h2.labels, h2_base, h2_direct, h2.users
            ),
            "top5_coverage": topk(h2_logits, h2.labels, 5),
            "selected": h2_audit,
        },
        "gates": {
            "source_requires_net_8_and_worst_user_ge_minus_2": bool(source_gate),
            "h2_requires_net_8_and_worst_user_ge_minus_2": bool(h2_gate),
            "allow_h3": bool(source_gate and h2_gate),
        },
        "h3_evaluation_performed": False,
        "decision": (
            "eligible_for_separate_frozen_h3_script"
            if source_gate and h2_gate
            else "reject_without_h3"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT / "h2_predictions.npz",
        sample_ids=h2.sample_ids,
        labels=h2.labels,
        users=h2.users,
        base_prediction=h2_base,
        direct_logits=h2_logits.astype(np.float32),
        direct_prediction=h2_direct,
        selected_prediction=h2_prediction,
        selected_candidate=np.asarray(candidate_name),
        selected_alpha=np.asarray(recipes[candidate_name], dtype=np.float32),
        selected_blend_weight=np.asarray(weight, dtype=np.float32),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
