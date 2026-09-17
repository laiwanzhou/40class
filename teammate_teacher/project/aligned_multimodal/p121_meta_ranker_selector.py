"""Outer-safe selector between the P120 meta champion and P121 LambdaRank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits, one_hot


HERE = Path(__file__).resolve().parent
META = HERE / "runs/p120_base_clip_router_meta_selector_v1"
RANKER = HERE / "runs/p121_row_level_candidate_ranker_v1"
OUTPUT = HERE / "runs/p121_meta_vs_ranker_selector_v1"


def features(
    safe: np.ndarray,
    meta: np.ndarray,
    ranker: np.ndarray,
    meta_score: np.ndarray,
    ranker_margin: np.ndarray,
) -> np.ndarray:
    scalar = np.column_stack(
        (
            meta_score,
            ranker_margin,
            meta != safe,
            ranker != safe,
            meta == ranker,
        )
    ).astype(np.float32)
    return np.concatenate(
        (scalar, one_hot(safe), one_hot(meta), one_hot(ranker)), axis=1
    ).astype(np.float32)


def fit_score(x: np.ndarray, target: np.ndarray, predict: np.ndarray) -> np.ndarray:
    outputs = []
    for model in (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, solver="liblinear", max_iter=1200),
        ),
        ExtraTreesClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=12131,
            n_jobs=-1,
        ),
    ):
        model.fit(x, target)
        outputs.append(model.predict_proba(predict)[:, 1])
    return np.mean(np.stack(outputs), axis=0)


def report(
    threshold: float,
    score: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    meta: np.ndarray,
    ranker: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    switch = (meta != ranker) & (score >= threshold)
    output = meta.copy()
    output[switch] = ranker[switch]
    per_user = {
        user: int(
            np.sum(output[users.astype(str) == user] == labels[users.astype(str) == user])
            - np.sum(meta[users.astype(str) == user] == labels[users.astype(str) == user])
        )
        for user in sorted(set(users.astype(str).tolist()))
    }
    return (
        {
            "threshold": float(threshold),
            "correct": int(np.sum(output == labels)),
            "meta_correct": int(np.sum(meta == labels)),
            "net_vs_meta": int(np.sum(output == labels) - np.sum(meta == labels)),
            "switches": int(switch.sum()),
            "minimum_user_gain": int(min(per_user.values())),
            "positive_users": int(sum(value > 0 for value in per_user.values())),
            "per_user_gain": per_user,
        },
        output,
    )


def main() -> None:
    meta_saved = np.load(META / "predictions.npz")
    ranker_saved = np.load(RANKER / "predictions.npz")
    ranker_summary = json.loads((RANKER / "summary.json").read_text(encoding="utf-8"))
    bank = load_candidate_splits(full_visual_bank=True, structured_bank=True)
    reports = {}
    payload = {}
    total_meta = total_output = 0
    for held_name in ("H1_selection", "H2_confirmation", "H3_independent_fold0"):
        ids = meta_saved[f"{held_name}_source_sample_ids"]
        if not np.array_equal(ids, ranker_saved[f"{held_name}_source_sample_ids"]):
            raise ValueError("source IDs differ")
        labels = meta_saved[f"{held_name}_source_labels"]
        users = meta_saved[f"{held_name}_source_users"]
        safe = meta_saved[f"{held_name}_source_safe_prediction"]
        meta = meta_saved[f"{held_name}_source_prediction"]
        ranker_candidate = ranker_saved[f"{held_name}_source_candidate_prediction"]
        ranker_margin = ranker_saved[f"{held_name}_source_margin"]
        ranker_threshold = float(
            ranker_summary["cohorts"][held_name]["selected_threshold"]["threshold"]
        )
        ranker = safe.copy()
        route = (ranker_candidate != safe) & (ranker_margin >= ranker_threshold)
        ranker[route] = ranker_candidate[route]
        x = features(
            safe,
            meta,
            ranker,
            meta_saved[f"{held_name}_source_meta_score"],
            ranker_margin,
        )
        disagreement = meta != ranker
        decisive = disagreement & ((meta == labels) != (ranker == labels))
        nested_score = np.zeros(len(labels), dtype=np.float64)
        for user in sorted(set(users.astype(str).tolist())):
            held_user = users.astype(str) == user
            train = decisive & ~held_user
            valid = disagreement & held_user
            target = (ranker[train] == labels[train]).astype(np.int64)
            if valid.any() and len(np.unique(target)) == 2:
                nested_score[valid] = fit_score(x[train], target, x[valid])
        grid = [
            report(threshold, nested_score, labels, users, meta, ranker)[0]
            for threshold in np.arange(0.20, 0.851, 0.025)
        ]
        grid.sort(
            key=lambda row: (
                row["net_vs_meta"],
                row["minimum_user_gain"],
                row["positive_users"],
                -row["switches"],
            ),
            reverse=True,
        )
        selected = grid[0]
        source_switch = (meta != ranker) & (nested_score >= selected["threshold"])
        source_output = meta.copy()
        source_output[source_switch] = ranker[source_switch]
        held_labels = meta_saved[f"{held_name}_labels"]
        held_meta = meta_saved[f"{held_name}_prediction"]
        held_ranker = ranker_saved[f"{held_name}_prediction"]
        held_safe = ranker_saved[f"{held_name}_safe_prediction"]
        held_x = features(
            held_safe,
            held_meta,
            held_ranker,
            meta_saved[f"{held_name}_meta_score"],
            ranker_saved[f"{held_name}_margin"],
        )
        target = (ranker[decisive] == labels[decisive]).astype(np.int64)
        held_score = fit_score(x[decisive], target, held_x)
        held_report, output = report(
            selected["threshold"],
            held_score,
            held_labels,
            bank[held_name].split.users,
            held_meta,
            held_ranker,
        )
        reports[held_name] = {
            "source_decisive_rows": int(decisive.sum()),
            "selected_threshold": selected,
            "held_result": held_report,
            "held_metrics": classification_metrics(held_labels, output),
            "top_source_thresholds": grid[:10],
        }
        total_meta += int(np.sum(held_meta == held_labels))
        total_output += int(np.sum(output == held_labels))
        payload[f"{held_name}_sample_ids"] = meta_saved[f"{held_name}_sample_ids"]
        payload[f"{held_name}_labels"] = held_labels
        payload[f"{held_name}_meta_prediction"] = held_meta
        payload[f"{held_name}_ranker_prediction"] = held_ranker
        payload[f"{held_name}_prediction"] = output
        payload[f"{held_name}_score"] = held_score
        payload[f"{held_name}_source_sample_ids"] = ids
        payload[f"{held_name}_source_labels"] = labels
        payload[f"{held_name}_source_users"] = users
        payload[f"{held_name}_source_safe_prediction"] = safe
        payload[f"{held_name}_source_prediction"] = source_output
        payload[f"{held_name}_source_score"] = nested_score
    summary = {
        "stage": "P121_meta_vs_LambdaRank_outer_selector_v1",
        "status": "complete",
        "protocol": {
            "outer_held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": 2470,
            "meta_correct": total_meta,
            "correct": total_output,
            "accuracy": total_output / 2470,
            "net_vs_meta": total_output - total_meta,
            "gap_to_0.91_correct": 2248 - total_output,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
