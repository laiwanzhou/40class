"""Row-level LambdaRank over P89 safe and eight frozen OOF candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBRanker

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import (
    CandidateSplit,
    load_candidate_splits,
    one_hot,
    probability_entropy,
    probability_margin,
    shared_features,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p121_row_level_candidate_ranker_v1"
EPSILON = 1e-8


def make_ranker(seed: int) -> XGBRanker:
    return XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg",
        n_estimators=140,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=5.0,
        subsample=0.85,
        colsample_bytree=0.35,
        reg_lambda=20.0,
        reg_alpha=0.1,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def branch_feature_bank(
    data: dict[str, CandidateSplit],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[str],
]:
    candidate_names = list(next(iter(data.values())).candidates)
    branch_names = ["safe", *candidate_names]
    feature_bank: dict[str, np.ndarray] = {}
    prediction_bank: dict[str, np.ndarray] = {}
    for split_name, value in data.items():
        split = value.split
        probabilities = [split.safe_probability, *value.candidates.values()]
        predictions = np.stack(
            [
                split.safe_prediction,
                *[probability.argmax(axis=1) for probability in value.candidates.values()],
            ],
            axis=1,
        ).astype(np.int64)
        shared = shared_features(value)
        all_vote = predictions
        rows = np.arange(len(split.labels))
        branch_matrices = []
        for branch_index, (probability, prediction) in enumerate(
            zip(probabilities, predictions.T)
        ):
            identity = np.zeros((len(rows), len(branch_names)), dtype=np.float32)
            identity[:, branch_index] = 1.0
            confidence = probability.max(axis=1)
            scalar = np.column_stack(
                (
                    confidence,
                    probability_margin(probability),
                    probability_entropy(probability),
                    probability[rows, prediction],
                    probability[rows, split.safe_prediction],
                    split.safe_probability[rows, prediction],
                    confidence - split.safe_probability[rows, split.safe_prediction],
                    probability[rows, prediction]
                    - probability[rows, split.safe_prediction],
                    split.safe_probability[rows, split.safe_prediction]
                    - split.safe_probability[rows, prediction],
                    np.mean(all_vote == prediction[:, None], axis=1),
                    np.mean(all_vote == split.safe_prediction[:, None], axis=1),
                    prediction != split.safe_prediction,
                )
            ).astype(np.float32)
            branch_matrices.append(
                np.concatenate(
                    (shared, scalar, one_hot(prediction), identity), axis=1
                ).astype(np.float32)
            )
        feature_bank[split_name] = np.stack(branch_matrices, axis=1)
        prediction_bank[split_name] = predictions
    return feature_bank, prediction_bank, branch_names


def concatenate_rows(
    names: list[str],
    data: dict[str, CandidateSplit],
    features: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([features[name] for name in names], axis=0)
    branch_prediction = np.concatenate([predictions[name] for name in names], axis=0)
    labels = np.concatenate([data[name].split.labels for name in names])
    users = np.concatenate([data[name].split.users for name in names])
    sample_ids = np.concatenate([data[name].split.sample_ids for name in names])
    relevance = (branch_prediction == labels[:, None]).astype(np.float32)
    return x, relevance, labels, users, sample_ids


def fit_predict_ranker(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    branches = train_x.shape[1]
    model = make_ranker(seed)
    model.fit(
        train_x.reshape(-1, train_x.shape[-1]),
        train_y.reshape(-1),
        group=np.full(len(train_x), branches, dtype=np.int32),
        verbose=False,
    )
    return model.predict(predict_x.reshape(-1, predict_x.shape[-1])).reshape(
        len(predict_x), branches
    )


def nested_scores(
    x: np.ndarray, relevance: np.ndarray, users: np.ndarray, seed: int
) -> np.ndarray:
    output = np.zeros(relevance.shape, dtype=np.float64)
    splitter = GroupKFold(n_splits=3)
    for inner, (train_rows, held_rows) in enumerate(
        splitter.split(np.arange(len(x)), groups=users.astype(str))
    ):
        output[held_rows] = fit_predict_ranker(
            x[train_rows], relevance[train_rows], x[held_rows], seed + inner
        )
    return output


def row_choice(
    scores: np.ndarray, branch_prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_candidate = scores[:, 1:].argmax(axis=1) + 1
    rows = np.arange(len(scores))
    margin = scores[rows, best_candidate] - scores[:, 0]
    candidate_label = branch_prediction[rows, best_candidate]
    return best_candidate, candidate_label, margin


def threshold_report(
    threshold: float,
    margin: np.ndarray,
    candidate_label: np.ndarray,
    safe: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    route = (candidate_label != safe) & (margin >= threshold)
    output = safe.copy()
    output[route] = candidate_label[route]
    rescue = int(np.sum((output == labels) & (safe != labels)))
    harm = int(np.sum((output != labels) & (safe == labels)))
    per_user = {
        user: int(
            np.sum(output[users.astype(str) == user] == labels[users.astype(str) == user])
            - np.sum(safe[users.astype(str) == user] == labels[users.astype(str) == user])
        )
        for user in sorted(set(users.astype(str).tolist()))
    }
    return (
        {
            "threshold": float(threshold),
            "correct": int(np.sum(output == labels)),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "changed": int(route.sum()),
            "minimum_user_gain": int(min(per_user.values())),
            "positive_users": int(sum(value > 0 for value in per_user.values())),
            "per_user_gain": per_user,
        },
        output,
    )


def select_threshold(
    margin: np.ndarray,
    candidate_label: np.ndarray,
    safe: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    disagreement = candidate_label != safe
    values = margin[disagreement]
    thresholds = np.unique(
        np.concatenate(
            (
                np.quantile(values, np.linspace(0.02, 0.98, 49)),
                [values.min() - 1e-6, values.max() + 1e-6],
            )
        )
    )
    reports = [
        threshold_report(
            threshold, margin, candidate_label, safe, labels, users
        )[0]
        for threshold in thresholds
    ]
    reports.sort(
        key=lambda row: (
            row["net"],
            -row["harm"],
            row["positive_users"],
            row["minimum_user_gain"],
            -row["changed"],
        ),
        reverse=True,
    )
    return reports[0], reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = load_candidate_splits(full_visual_bank=True, structured_bank=True)
    features, predictions, branch_names = branch_feature_bank(data)
    recipes = {
        "H1_selection": ["H2_confirmation", "H3_independent_fold0"],
        "H2_confirmation": ["H1_selection", "H3_independent_fold0"],
        "H3_independent_fold0": ["H1_selection", "H2_confirmation"],
    }
    reports = {}
    payload = {}
    total_safe = total_ranker = 0
    for outer_index, (held_name, source_names) in enumerate(recipes.items()):
        source_x, source_y, source_labels, source_users, source_ids = concatenate_rows(
            source_names, data, features, predictions
        )
        source_scores = nested_scores(
            source_x, source_y, source_users, 12100 + outer_index * 100
        )
        source_prediction = np.concatenate([predictions[name] for name in source_names])
        _, source_candidate, source_margin = row_choice(
            source_scores, source_prediction
        )
        source_safe = np.concatenate(
            [data[name].split.safe_prediction for name in source_names]
        )
        selected, grid = select_threshold(
            source_margin,
            source_candidate,
            source_safe,
            source_labels,
            source_users,
        )

        held = data[held_name]
        held_scores = fit_predict_ranker(
            source_x,
            source_y,
            features[held_name],
            12150 + outer_index * 100,
        )
        held_best, held_candidate, held_margin = row_choice(
            held_scores, predictions[held_name]
        )
        held_result, output = threshold_report(
            selected["threshold"],
            held_margin,
            held_candidate,
            held.split.safe_prediction,
            held.split.labels,
            held.split.users,
        )
        reports[held_name] = {
            "source_rows": int(len(source_labels)),
            "selected_threshold": selected,
            "held_result": held_result,
            "safe_metrics": classification_metrics(
                held.split.labels, held.split.safe_prediction
            ),
            "ranker_metrics": classification_metrics(held.split.labels, output),
            "top_source_thresholds": grid[:10],
        }
        total_safe += reports[held_name]["safe_metrics"]["correct"]
        total_ranker += reports[held_name]["ranker_metrics"]["correct"]
        payload[f"{held_name}_sample_ids"] = held.split.sample_ids
        payload[f"{held_name}_labels"] = held.split.labels
        payload[f"{held_name}_safe_prediction"] = held.split.safe_prediction
        payload[f"{held_name}_prediction"] = output
        payload[f"{held_name}_rank_scores"] = held_scores.astype(np.float32)
        payload[f"{held_name}_best_branch"] = held_best
        payload[f"{held_name}_margin"] = held_margin.astype(np.float32)
        payload[f"{held_name}_source_sample_ids"] = source_ids
        payload[f"{held_name}_source_labels"] = source_labels
        payload[f"{held_name}_source_users"] = source_users
        payload[f"{held_name}_source_safe_prediction"] = source_safe
        payload[f"{held_name}_source_candidate_prediction"] = source_candidate
        payload[f"{held_name}_source_margin"] = source_margin.astype(np.float32)
    report = {
        "stage": "P121_row_level_LambdaRank_candidate_selector_v1",
        "status": "complete",
        "protocol": {
            "branches": branch_names,
            "source_ranking_scores": "3-fold GroupKFold over source subjects",
            "outer_held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": reports,
        "aggregate": {
            "rows": 2470,
            "safe_correct": total_safe,
            "correct": total_ranker,
            "accuracy": total_ranker / 2470,
            "net": total_ranker - total_safe,
            "gap_to_0.91_correct": 2248 - total_ranker,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
