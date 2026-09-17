"""Outer-safe harm detector that may only roll P121 champion changes back to P89."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import (
    load_candidate_splits,
    one_hot,
    shared_features,
)


HERE = Path(__file__).resolve().parent
CHAMPION = HERE / "runs/p121_meta_vs_ranker_selector_v1"
OUTPUT = HERE / "runs/p126_current_champion_rollback_v1"


def full_feature_lookup():
    data = load_candidate_splits(full_visual_bank=True, structured_bank=True)
    lookup = {}
    users = {}
    for split_name, value in data.items():
        feature = shared_features(value)
        for row, sample_id in enumerate(value.split.sample_ids.astype(str)):
            lookup[sample_id] = feature[row]
            users[sample_id] = value.split.users[row]
    return data, lookup, users


def features(
    sample_ids: np.ndarray,
    safe: np.ndarray,
    champion: np.ndarray,
    score: np.ndarray,
    lookup,
) -> np.ndarray:
    shared = np.stack([lookup[value] for value in sample_ids.astype(str)]).astype(np.float32)
    scalar = np.column_stack((score, champion != safe)).astype(np.float32)
    return np.concatenate(
        (shared, scalar, one_hot(safe), one_hot(champion)), axis=1
    ).astype(np.float32)


def fit_score(x: np.ndarray, target: np.ndarray, predict: np.ndarray) -> np.ndarray:
    values = []
    for model in (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, solver="liblinear", max_iter=1200),
        ),
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=12601,
            n_jobs=-1,
        ),
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=5,
            min_samples_leaf=8,
            l2_regularization=15.0,
            random_state=12602,
        ),
    ):
        model.fit(x, target)
        values.append(model.predict_proba(predict)[:, 1])
    return np.mean(np.stack(values), axis=0)


def rollback_report(
    threshold: float,
    score: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    safe: np.ndarray,
    champion: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    rollback = (champion != safe) & (score >= threshold)
    output = champion.copy()
    output[rollback] = safe[rollback]
    per_user = {
        user: int(
            np.sum(output[users.astype(str) == user] == labels[users.astype(str) == user])
            - np.sum(
                champion[users.astype(str) == user] == labels[users.astype(str) == user]
            )
        )
        for user in sorted(set(users.astype(str).tolist()))
    }
    return (
        {
            "threshold": float(threshold),
            "correct": int(np.sum(output == labels)),
            "champion_correct": int(np.sum(champion == labels)),
            "net_vs_champion": int(
                np.sum(output == labels) - np.sum(champion == labels)
            ),
            "rollbacks": int(rollback.sum()),
            "recovered_harms": int(
                np.sum(rollback & (safe == labels) & (champion != labels))
            ),
            "lost_rescues": int(
                np.sum(rollback & (safe != labels) & (champion == labels))
            ),
            "minimum_user_gain": int(min(per_user.values())),
            "positive_users": int(sum(value > 0 for value in per_user.values())),
            "per_user_gain": per_user,
        },
        output,
    )


def main() -> None:
    champion = np.load(CHAMPION / "predictions.npz")
    data, lookup, user_lookup = full_feature_lookup()
    reports = {}
    payload = {}
    total_champion = total_output = 0
    for held_name, held in data.items():
        source_ids = champion[f"{held_name}_source_sample_ids"]
        labels = champion[f"{held_name}_source_labels"]
        users = champion[f"{held_name}_source_users"]
        safe = champion[f"{held_name}_source_safe_prediction"]
        prediction = champion[f"{held_name}_source_prediction"]
        source_score = champion[f"{held_name}_source_score"]
        x = features(source_ids, safe, prediction, source_score, lookup)
        changed = prediction != safe
        rescue = changed & (prediction == labels) & (safe != labels)
        harm = changed & (prediction != labels) & (safe == labels)
        decisive = rescue | harm
        nested_score = np.zeros(len(labels), dtype=np.float64)
        for user in sorted(set(users.astype(str).tolist())):
            held_user = users.astype(str) == user
            train = decisive & ~held_user
            valid = changed & held_user
            target = harm[train].astype(np.int64)
            if valid.any() and len(np.unique(target)) == 2:
                nested_score[valid] = fit_score(x[train], target, x[valid])
        thresholds = np.unique(
            np.concatenate(
                (
                    np.arange(0.10, 0.901, 0.025),
                    np.quantile(nested_score[changed], np.linspace(0.5, 0.98, 13)),
                )
            )
        )
        grid = [
            rollback_report(
                threshold, nested_score, labels, users, safe, prediction
            )[0]
            for threshold in thresholds
        ]
        grid.sort(
            key=lambda row: (
                row["net_vs_champion"],
                -row["lost_rescues"],
                row["positive_users"],
                row["minimum_user_gain"],
                -row["rollbacks"],
            ),
            reverse=True,
        )
        selected = grid[0]
        held_ids = champion[f"{held_name}_sample_ids"]
        held_labels = champion[f"{held_name}_labels"]
        held_safe = held.split.safe_prediction
        held_prediction = champion[f"{held_name}_prediction"]
        held_x = features(
            held_ids,
            held_safe,
            held_prediction,
            champion[f"{held_name}_score"],
            lookup,
        )
        target = harm[decisive].astype(np.int64)
        held_score = fit_score(x[decisive], target, held_x)
        held_result, output = rollback_report(
            selected["threshold"],
            held_score,
            held_labels,
            held.split.users,
            held_safe,
            held_prediction,
        )
        reports[held_name] = {
            "source_changes": int(changed.sum()),
            "source_rescues": int(rescue.sum()),
            "source_harms": int(harm.sum()),
            "selected_threshold": selected,
            "held_result": held_result,
            "held_metrics": classification_metrics(held_labels, output),
            "top_source_thresholds": grid[:10],
        }
        total_champion += int(np.sum(held_prediction == held_labels))
        total_output += int(np.sum(output == held_labels))
        payload[f"{held_name}_sample_ids"] = held_ids
        payload[f"{held_name}_labels"] = held_labels
        payload[f"{held_name}_safe_prediction"] = held_safe
        payload[f"{held_name}_champion_prediction"] = held_prediction
        payload[f"{held_name}_prediction"] = output
        payload[f"{held_name}_harm_score"] = held_score
    report = {
        "stage": "P126_current_champion_outer_safe_rollback_v1",
        "status": "complete",
        "protocol": {
            "allowed_action": "rollback current champion to P89 safe only",
            "source_score": "LOUO on source champion rescue-vs-harm rows",
            "outer_held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": 2470,
            "champion_correct": total_champion,
            "correct": total_output,
            "accuracy": total_output / 2470,
            "net_vs_champion": total_output - total_champion,
            "gap_to_0.91_correct": 2248 - total_output,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
