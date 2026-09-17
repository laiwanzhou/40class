"""Outer-safe selector between equal-score P121 and P123 route champions."""

from __future__ import annotations

import argparse
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
LEFT = HERE / "runs/p121_meta_vs_ranker_selector_v1"
RIGHT = HERE / "runs/p123_base_dense_router_meta_selector_v1"
OUTPUT = HERE / "runs/p124_equal_champion_selector_v1"


def make_features(
    safe: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_score: np.ndarray,
    right_score: np.ndarray,
) -> np.ndarray:
    scalar = np.column_stack(
        (
            left_score,
            right_score,
            right_score - left_score,
            left != safe,
            right != safe,
            left == right,
        )
    ).astype(np.float32)
    return np.concatenate(
        (scalar, one_hot(safe), one_hot(left), one_hot(right)), axis=1
    ).astype(np.float32)


def fit_score(x: np.ndarray, y: np.ndarray, predict: np.ndarray) -> np.ndarray:
    scores = []
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
            random_state=12401,
            n_jobs=-1,
        ),
    ):
        model.fit(x, y)
        scores.append(model.predict_proba(predict)[:, 1])
    return np.mean(np.stack(scores), axis=0)


def evaluate(
    threshold: float,
    score: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    switch = (left != right) & (score >= threshold)
    output = left.copy()
    output[switch] = right[switch]
    per_user = {
        user: int(
            np.sum(output[users.astype(str) == user] == labels[users.astype(str) == user])
            - np.sum(left[users.astype(str) == user] == labels[users.astype(str) == user])
        )
        for user in sorted(set(users.astype(str).tolist()))
    }
    return (
        {
            "threshold": float(threshold),
            "correct": int(np.sum(output == labels)),
            "left_correct": int(np.sum(left == labels)),
            "net_vs_left": int(np.sum(output == labels) - np.sum(left == labels)),
            "switches": int(switch.sum()),
            "minimum_user_gain": int(min(per_user.values())),
            "positive_users": int(sum(value > 0 for value in per_user.values())),
            "per_user_gain": per_user,
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--right", type=Path, default=RIGHT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    right_root = args.right.resolve()
    output_dir = args.output_dir.resolve()
    left_saved = np.load(LEFT / "predictions.npz")
    right_saved = np.load(right_root / "predictions.npz")
    bank = load_candidate_splits(full_visual_bank=True, structured_bank=True)
    reports = {}
    payload = {}
    total_left = total_output = 0
    for held_name, held in bank.items():
        source_ids = left_saved[f"{held_name}_source_sample_ids"]
        if not np.array_equal(source_ids, right_saved[f"{held_name}_source_sample_ids"]):
            raise ValueError("source IDs differ")
        labels = left_saved[f"{held_name}_source_labels"]
        users = left_saved[f"{held_name}_source_users"]
        safe = left_saved[f"{held_name}_source_safe_prediction"]
        left = left_saved[f"{held_name}_source_prediction"]
        right = right_saved[f"{held_name}_source_prediction"]
        x = make_features(
            safe,
            left,
            right,
            left_saved[f"{held_name}_source_score"],
            right_saved[f"{held_name}_source_meta_score"],
        )
        disagreement = left != right
        decisive = disagreement & ((left == labels) != (right == labels))
        nested_score = np.zeros(len(labels), dtype=np.float64)
        for user in sorted(set(users.astype(str).tolist())):
            held_user = users.astype(str) == user
            train = decisive & ~held_user
            valid = disagreement & held_user
            target = (right[train] == labels[train]).astype(np.int64)
            if valid.any() and len(np.unique(target)) == 2:
                nested_score[valid] = fit_score(x[train], target, x[valid])
        grid = [
            evaluate(threshold, nested_score, labels, users, left, right)[0]
            for threshold in np.arange(0.20, 0.851, 0.025)
        ]
        grid.sort(
            key=lambda row: (
                row["net_vs_left"],
                row["minimum_user_gain"],
                row["positive_users"],
                -row["switches"],
            ),
            reverse=True,
        )
        selected = grid[0]
        held_labels = left_saved[f"{held_name}_labels"]
        held_left = left_saved[f"{held_name}_prediction"]
        held_right = right_saved[f"{held_name}_prediction"]
        held_x = make_features(
            held.split.safe_prediction,
            held_left,
            held_right,
            left_saved[f"{held_name}_score"],
            right_saved[f"{held_name}_meta_score"],
        )
        target = (right[decisive] == labels[decisive]).astype(np.int64)
        held_score = fit_score(x[decisive], target, held_x)
        held_result, output = evaluate(
            selected["threshold"],
            held_score,
            held_labels,
            held.split.users,
            held_left,
            held_right,
        )
        reports[held_name] = {
            "source_decisive_rows": int(decisive.sum()),
            "selected_threshold": selected,
            "held_result": held_result,
            "held_metrics": classification_metrics(held_labels, output),
            "top_source_thresholds": grid[:10],
        }
        total_left += int(np.sum(held_left == held_labels))
        total_output += int(np.sum(output == held_labels))
        payload[f"{held_name}_sample_ids"] = held.split.sample_ids
        payload[f"{held_name}_labels"] = held_labels
        payload[f"{held_name}_left_prediction"] = held_left
        payload[f"{held_name}_right_prediction"] = held_right
        payload[f"{held_name}_prediction"] = output
        payload[f"{held_name}_score"] = held_score
    report = {
        "stage": "P124_equal_score_champion_outer_selector_v1",
        "status": "complete",
        "protocol": {
            "left": str(LEFT),
            "right": str(RIGHT),
            "outer_held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": 2470,
            "left_correct": total_left,
            "correct": total_output,
            "accuracy": total_output / 2470,
            "net_vs_left": total_output - total_left,
            "gap_to_0.91_correct": 2248 - total_output,
        },
    }
    report["protocol"]["right"] = str(right_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
