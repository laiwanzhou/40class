"""Audit P103 rich token assets and the frozen candidate-hit hard problem."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_p102_session_closure import load_npz, true_rank
from p100a_global_teacher_data import H3_USERS
from p101_finegrained_teacher_data import load_p101_data


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_B1 = HERE / "runs/p102_b1_visual_listwise_session_oof_v1/b1_oof_predictions.npz"
DEFAULT_OUTPUT = HERE / "runs/p103_rich_asset_audit_v1/summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-oof", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--hard-set", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--b1", type=Path, default=DEFAULT_B1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def array_record(values: np.ndarray, axes: list[str]) -> dict[str, Any]:
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "axes": axes,
        "finite_probe": bool(np.isfinite(np.asarray(values[:8], dtype=np.float32)).all()),
    }


def top_confusions(
    labels: np.ndarray, prediction: np.ndarray, selected: np.ndarray, limit: int = 30
) -> list[dict[str, int]]:
    counts = Counter(
        (int(labels[row]), int(prediction[row])) for row in np.flatnonzero(selected)
    )
    return [
        {"true": true, "a_prediction": predicted, "rows": count}
        for (true, predicted), count in counts.most_common(limit)
    ]


def main() -> None:
    args = parse_args()
    session = load_npz(args.session_oof.resolve())
    hard = load_npz(args.hard_set.resolve())
    b1 = load_npz(args.b1.resolve())
    data = load_p101_data()
    sample_ids = session["sample_ids"].astype(str)
    for values, name in ((hard["sample_ids"], "hard"), (b1["sample_ids"], "b1"), (data.sample_ids, "P101")):
        if not np.array_equal(np.asarray(values).astype(str), sample_ids):
            raise RuntimeError(f"P103/{name} row order differs")
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    folds = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    a_prediction = a_probability.argmax(axis=1)
    candidate_ids = np.asarray(hard["candidate_ids"], dtype=np.int64)
    candidate_hit = np.any(candidate_ids == labels[:, None], axis=1)
    a_error = a_prediction != labels
    attackable = a_error & candidate_hit
    if set(users.tolist()) & set(H3_USERS) or int(data.summary()["h3_rows"]) != 0:
        raise RuntimeError("H3 reached P103 asset audit")

    motion_rows = data.motion_rows
    assets = {
        "visual_videomaev2_temporal": array_record(
            data.visual_vmae_temporal, ["row", "window", "view", "time", "channel"]
        ),
        "visual_internvideo2_temporal": array_record(
            data.visual_iv2_temporal, ["row", "window", "view", "time", "channel"]
        ),
        "visual_videomaev2_pooled": array_record(
            data.visual_vmae_pooled, ["row", "window", "view", "channel"]
        ),
        "visual_internvideo2_pooled": array_record(
            data.visual_iv2_pooled, ["row", "window", "view", "channel"]
        ),
        "visual_videomaev2_action": array_record(
            data.visual_vmae_action, ["row", "window", "view", "action_channel"]
        ),
        "visual_internvideo2_action": array_record(
            data.visual_iv2_action, ["row", "window", "view", "action_channel"]
        ),
        "skeleton_joint_temporal": array_record(
            data.motion["skeleton_features"][motion_rows],
            ["row", "window", "time", "joint", "feature"],
        ),
        "skeleton_relations": array_record(
            data.motion["skeleton_relations"][motion_rows],
            ["row", "window", "time", "relation"],
        ),
        "imu_device_temporal": array_record(
            data.motion["imu_sequences"][motion_rows],
            ["row", "window", "time", "device", "channel", "substep"],
        ),
        "imu_device_statistics": array_record(
            data.motion["imu_bin_statistics"][motion_rows],
            ["row", "window", "time", "device", "statistic"],
        ),
    }
    sizes = np.sum(candidate_ids >= 0, axis=1)
    folds_report: list[dict[str, Any]] = []
    for fold in range(4):
        held = folds == fold
        fold_attackable = held & attackable
        folds_report.append(
            {
                "fold": fold,
                "held_users": sorted(set(users[held].tolist())),
                "rows": int(held.sum()),
                "a_errors": int(np.sum(held & a_error)),
                "candidate_hit_errors": int(fold_attackable.sum()),
                "candidate_hit_error_recall": float(np.mean(candidate_hit[held & a_error])),
                "top_confusions_candidate_hit": top_confusions(
                    labels, a_prediction, fold_attackable, 12
                ),
            }
        )
    category = hard["category"].astype(str)
    b1_prediction = np.asarray(b1["full_session_probability"]).argmax(axis=1)
    result = {
        "status": "complete",
        "conclusion_boundary": {
            "compressed_global_b_reranker_route_exhausted": True,
            "hard_case_b_teacher_route_exhausted": False,
            "b_teacher_hypothesis_closed": False,
            "p102_stop_scope": "B0/B1 compressed-global reranker family only",
        },
        "data": {
            "rows": 1941,
            "subjects": sorted(set(users.tolist())),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
            "p101_contract": data.summary(),
        },
        "rich_assets": assets,
        "depth_asset_status": {
            "historical_local_pixel_and_trial_feature_caches_exist": True,
            "p101_equivalent_row_complete_rich_token_contract": False,
            "initial_b2_authorized": False,
            "interpretation": "reserve Depth for a causally motivated later multimodal revision",
        },
        "hard_problem": {
            "a_correct": int(np.sum(~a_error)),
            "a_errors": int(a_error.sum()),
            "candidate_hit_errors": int(attackable.sum()),
            "candidate_hit_error_recall": float(np.mean(candidate_hit[a_error])),
            "candidate_oracle_correct": int(np.sum((~a_error) | candidate_hit)),
            "candidate_oracle_accuracy": float(np.mean((~a_error) | candidate_hit)),
            "candidate_size_mean": float(sizes.mean()),
            "candidate_size_distribution": {
                str(size): int(np.sum(sizes == size)) for size in sorted(set(sizes.tolist()))
            },
            "direct_true_rank_on_candidate_hit_errors": {
                "mean": float(true_rank(a_probability[attackable], labels[attackable]).mean()),
                "top5_rows": int(np.sum(category == "A")),
                "confusion_expanded_rows": int(np.sum(category == "B")),
            },
            "top_confusions_candidate_hit": top_confusions(labels, a_prediction, attackable),
            "folds": folds_report,
        },
        "b1_reference": {
            "conditional_post_session_correct": int(np.sum(attackable & (b1_prediction == labels))),
            "conditional_post_session_accuracy": float(np.mean(b1_prediction[attackable] == labels[attackable])),
        },
        "initial_architecture_decision": {
            "name": "P103-B2 rich visual candidate-query cross-attention",
            "why": [
                "two pretrained encoders retain 96 temporal view tokens per sample",
                "candidate query changes token attention before evidence scoring",
                "candidate-list and pairwise objectives directly compare hard negatives",
                "no mean/std summary, PCA, prototype, tree, class-linear direction or fixed A residual",
            ],
            "counterfactuals": ["aligned visual", "within-subject shuffle visual", "zero visual"],
            "next_mechanism_if_negative": "audit view/time attention and add explicit hand/object local representation",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
