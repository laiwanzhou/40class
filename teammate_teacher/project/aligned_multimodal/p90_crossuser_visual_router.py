"""Nested cross-user router from the frozen P89 safe path to the P90 visual teacher.

The router is deliberately trained on *OOF teacher outputs*, never on in-fold
teacher predictions.  Its fixed model family predicts whether replacing the
safe decision by the visual decision is more likely to rescue than harm a row.
For every outer cohort, the route threshold is selected with leave-one-user-out
predictions from the other cohorts only.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p89_supported_template_gate import (
    h3_protocol,
    load_grouping,
    load_imu,
    safe_probability_and_prediction,
)
from p90_teacher_fusion_audit import align
from p90_visual_teacher_safe_fusion_audit import load_visual_candidates


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "runs/p90_crossuser_visual_router_v1"
METADATA_CSV = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
CANDIDATE_NAME = "videomaev2_base_plus_internvideo2_l_equal"
EPSILON = 1e-8


@dataclass(frozen=True)
class SplitData:
    name: str
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    safe_probability: np.ndarray
    safe_prediction: np.ndarray
    p87_prediction: np.ndarray
    sessions: list[np.ndarray]
    visual_probability: dict[str, np.ndarray]
    quality: np.ndarray
    quality_names: list[str]


def probability_margin(probability: np.ndarray) -> np.ndarray:
    top = np.partition(probability, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def probability_entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, EPSILON, 1.0)
    return -np.sum(values * np.log(values), axis=1) / np.log(values.shape[1])


def one_hot(values: np.ndarray, classes: int = 40) -> np.ndarray:
    output = np.zeros((len(values), classes), dtype=np.float32)
    output[np.arange(len(values)), np.asarray(values, dtype=np.int64)] = 1.0
    return output


def aligned_quality(sample_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    values: list[list[float]] = []
    for sample_id in sample_ids.astype(str):
        row = rows[sample_id]
        trial = str(row.get("trial_id", ""))
        try:
            repeat_index = float(trial.rsplit("-", 1)[-1])
        except ValueError:
            repeat_index = 0.0
        values.append(
            [
                float(row.get("duration_seconds") or 0.0),
                float(row.get("timestamp_available") or 0.0),
                float(row.get("imu_csv_files") or 0.0),
                float(row.get("imu_parsed_rows") or 0.0),
                float(row.get("device_count") or 0.0),
                repeat_index,
            ]
        )
    return np.asarray(values, dtype=np.float32), [
        "duration_seconds",
        "timestamp_available",
        "imu_csv_files",
        "imu_parsed_rows",
        "device_count",
        "trial_repeat_index",
    ]


def session_features(
    sessions: list[np.ndarray], safe: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    output = np.zeros((len(safe), 9), dtype=np.float32)
    for session in sessions:
        indices = np.asarray(session, dtype=np.int64)
        if not len(indices):
            continue
        safe_values = safe[indices]
        candidate_values = candidate[indices]
        for position, row in enumerate(indices):
            safe_class = safe[row]
            candidate_class = candidate[row]
            output[row] = (
                len(indices),
                position / max(len(indices) - 1, 1),
                np.mean(safe_values == safe_class),
                np.mean(candidate_values == candidate_class),
                np.mean(safe_values == candidate_class),
                float(position > 0 and safe_values[position - 1] == safe_class),
                float(position + 1 < len(indices) and safe_values[position + 1] == safe_class),
                float(position > 0 and candidate_values[position - 1] == candidate_class),
                float(
                    position + 1 < len(indices)
                    and candidate_values[position + 1] == candidate_class
                ),
            )
    return output, [
        "session_length",
        "session_position",
        "safe_class_session_fraction",
        "candidate_class_session_fraction",
        "candidate_class_in_safe_session_fraction",
        "previous_safe_agrees",
        "next_safe_agrees",
        "previous_candidate_agrees",
        "next_candidate_agrees",
    ]


def load_splits() -> dict[str, SplitData]:
    grouping = load_grouping()
    imu_ids, imu_logits = load_imu()
    raw = {
        "H1_selection": full40.protocol(full40.H1_RUN, full40.H1_USERS),
        "H2_confirmation": full40.protocol(full40.H2_RUN, full40.H2_USERS),
        "H3_independent_fold0": h3_protocol(imu_ids, imu_logits, grouping)[0],
    }
    reference_ids, visual = load_visual_candidates()
    splits: dict[str, SplitData] = {}
    for name, protocol in raw.items():
        safe_probability, safe_prediction = safe_probability_and_prediction(
            protocol, imu_ids, imu_logits, grouping
        )
        quality, quality_names = aligned_quality(protocol[0])
        splits[name] = SplitData(
            name=name,
            sample_ids=protocol[0].astype(str),
            labels=np.asarray(protocol[1], dtype=np.int64),
            users=protocol[4].users.astype(str),
            safe_probability=np.asarray(safe_probability, dtype=np.float64),
            safe_prediction=np.asarray(safe_prediction, dtype=np.int64),
            p87_prediction=np.asarray(protocol[3], dtype=np.int64),
            sessions=protocol[6],
            visual_probability={
                key: align(reference_ids, probability, protocol[0]).astype(np.float64)
                for key, probability in visual.items()
            },
            quality=quality,
            quality_names=quality_names,
        )
    return splits


def build_features(split: SplitData) -> tuple[np.ndarray, list[str]]:
    candidate = split.visual_probability[CANDIDATE_NAME]
    candidate_prediction = candidate.argmax(axis=1)
    safe_prediction = split.safe_prediction
    raw_safe_prediction = split.safe_probability.argmax(axis=1)
    experts = {
        "safe_probability": split.safe_probability,
        **split.visual_probability,
    }
    matrices: list[np.ndarray] = []
    names: list[str] = []

    # Full probability shape carries class-specific reliability.  Log-probability
    # is clipped so linear and tree members can share the exact same matrix.
    for expert_name, probability in experts.items():
        logp = np.log(np.clip(probability, EPSILON, 1.0)).astype(np.float32)
        matrices.append(logp)
        names.extend(f"{expert_name}_logp_{class_id}" for class_id in range(40))
        sorted_probability = np.sort(probability, axis=1)[:, ::-1]
        scalars = np.column_stack(
            (
                probability.max(axis=1),
                probability_margin(probability),
                probability_entropy(probability),
                sorted_probability[:, :5],
                probability[np.arange(len(probability)), safe_prediction],
                probability[np.arange(len(probability)), candidate_prediction],
            )
        ).astype(np.float32)
        matrices.append(scalars)
        names.extend(
            [
                f"{expert_name}_confidence",
                f"{expert_name}_margin",
                f"{expert_name}_entropy",
                *[f"{expert_name}_top_{rank}" for rank in range(1, 6)],
                f"{expert_name}_safe_class_probability",
                f"{expert_name}_candidate_class_probability",
            ]
        )

    predictions = {
        "safe_final": safe_prediction,
        "safe_raw": raw_safe_prediction,
        "p87_sequence": split.p87_prediction,
        **{
            f"{name}_prediction": probability.argmax(axis=1)
            for name, probability in split.visual_probability.items()
        },
    }
    for prediction_name, values in predictions.items():
        matrices.append(one_hot(values))
        names.extend(f"{prediction_name}_class_{class_id}" for class_id in range(40))

    visual_predictions = np.stack(
        [probability.argmax(axis=1) for probability in split.visual_probability.values()],
        axis=1,
    )
    pair = np.column_stack(
        (
            candidate_prediction != safe_prediction,
            raw_safe_prediction == safe_prediction,
            split.p87_prediction == safe_prediction,
            split.p87_prediction == candidate_prediction,
            np.mean(visual_predictions == candidate_prediction[:, None], axis=1),
            np.mean(visual_predictions == safe_prediction[:, None], axis=1),
            candidate[np.arange(len(candidate)), candidate_prediction]
            - split.safe_probability[np.arange(len(candidate)), safe_prediction],
            candidate[np.arange(len(candidate)), candidate_prediction]
            - candidate[np.arange(len(candidate)), safe_prediction],
            split.safe_probability[np.arange(len(candidate)), safe_prediction]
            - split.safe_probability[np.arange(len(candidate)), candidate_prediction],
        )
    ).astype(np.float32)
    matrices.append(pair)
    names.extend(
        [
            "candidate_disagrees_safe",
            "safe_final_equals_raw",
            "safe_final_equals_p87",
            "candidate_equals_p87",
            "visual_vote_fraction_candidate",
            "visual_vote_fraction_safe",
            "candidate_minus_safe_confidence",
            "candidate_internal_class_gap",
            "safe_internal_class_gap",
        ]
    )

    session, session_names = session_features(
        split.sessions, safe_prediction, candidate_prediction
    )
    matrices.extend((session, split.quality))
    names.extend(session_names)
    names.extend(split.quality_names)
    output = np.concatenate(matrices, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f"{split.name}: non-finite router features")
    return output, names


def model_factories() -> list[tuple[str, Callable[[], Any]]]:
    return [
        (
            "logistic_c003",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.03, max_iter=1200, solver="liblinear"),
            ),
        ),
        (
            "logistic_c030",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.30, max_iter=1200, solver="liblinear"),
            ),
        ),
        (
            "extra_depth5_leaf8",
            lambda: ExtraTreesClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=8,
                max_features=0.5,
                class_weight="balanced",
                random_state=9011,
                n_jobs=-1,
            ),
        ),
        (
            "extra_depth8_leaf5",
            lambda: ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=9021,
                n_jobs=-1,
            ),
        ),
        (
            "hist_leaf7",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=7,
                min_samples_leaf=15,
                l2_regularization=10.0,
                random_state=9031,
            ),
        ),
    ]


def gain_vector(split: SplitData) -> tuple[np.ndarray, np.ndarray]:
    candidate_prediction = split.visual_probability[CANDIDATE_NAME].argmax(axis=1)
    safe_correct = split.safe_prediction == split.labels
    candidate_correct = candidate_prediction == split.labels
    gain = candidate_correct.astype(np.int8) - safe_correct.astype(np.int8)
    disagreement = candidate_prediction != split.safe_prediction
    return gain, disagreement


def fit_ensemble(
    train_x: np.ndarray,
    train_gain: np.ndarray,
    predict_x: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    decisive = train_gain != 0
    target = (train_gain[decisive] > 0).astype(np.int64)
    if len(np.unique(target)) != 2:
        raise ValueError("router training rows do not contain both rescue and harm")
    members: dict[str, np.ndarray] = {}
    for name, factory in model_factories():
        model = factory()
        model.fit(train_x[decisive], target)
        members[name] = np.asarray(model.predict_proba(predict_x)[:, 1], dtype=np.float64)
    return np.mean(np.stack(list(members.values()), axis=1), axis=1), members


def cross_user_scores(
    features: np.ndarray, gain: np.ndarray, users: np.ndarray
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    scores = np.zeros(len(features), dtype=np.float64)
    audit: dict[str, dict[str, float]] = {}
    for user in sorted(set(users.astype(str).tolist())):
        held = users.astype(str) == user
        train = ~held
        scores[held], _ = fit_ensemble(features[train], gain[train], features[held])
        decisive = held & (gain != 0)
        audit[user] = {
            "rows": int(held.sum()),
            "decisive_rows": int(decisive.sum()),
            "rescue_rows": int(np.sum(gain[held] > 0)),
            "harm_rows": int(np.sum(gain[held] < 0)),
        }
    return scores, audit


def threshold_report(
    threshold: float,
    scores: np.ndarray,
    gain: np.ndarray,
    disagreement: np.ndarray,
    users: np.ndarray,
) -> dict[str, Any]:
    selected = disagreement & (scores >= threshold)
    rescue = int(np.sum(selected & (gain > 0)))
    harm = int(np.sum(selected & (gain < 0)))
    user_gain = {
        user: int(np.sum(gain[selected & (users.astype(str) == user)]))
        for user in sorted(set(users.astype(str).tolist()))
    }
    decisive = rescue + harm
    return {
        "threshold": float(threshold),
        "route_count": int(selected.sum()),
        "decisive_route_count": decisive,
        "rescue": rescue,
        "harm": harm,
        "net_gain": rescue - harm,
        "rescue_precision": float(rescue / decisive) if decisive else 0.0,
        "minimum_user_gain": min(user_gain.values()),
        "positive_users": int(sum(value > 0 for value in user_gain.values())),
        "user_gain": user_gain,
    }


def select_threshold(
    scores: np.ndarray,
    gain: np.ndarray,
    disagreement: np.ndarray,
    users: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixed = np.arange(0.20, 0.851, 0.025)
    quantiles = np.quantile(scores[disagreement], np.linspace(0.50, 0.95, 10))
    thresholds = np.unique(np.concatenate((fixed, quantiles)))
    reports = [
        threshold_report(value, scores, gain, disagreement, users)
        for value in thresholds
    ]
    valid = [
        row
        for row in reports
        if row["minimum_user_gain"] >= 0
        and row["decisive_route_count"] >= 10
        and row["net_gain"] > 0
    ]
    if not valid:
        empty = threshold_report(1.1, scores, gain, disagreement, users)
        empty["selection_note"] = "no positive threshold passed nested stability gate"
        return empty, reports
    valid.sort(
        key=lambda row: (
            row["net_gain"],
            row["positive_users"],
            row["rescue_precision"],
            -row["harm"],
            -row["route_count"],
        ),
        reverse=True,
    )
    selected = dict(valid[0])
    selected["selection_note"] = (
        "max nested leave-one-user-out net gain with no user regression"
    )
    return selected, reports


def concatenate_splits(
    names: list[str], splits: dict[str, SplitData], features: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrices = np.concatenate([features[name] for name in names], axis=0)
    gains = np.concatenate([gain_vector(splits[name])[0] for name in names])
    disagreements = np.concatenate(
        [gain_vector(splits[name])[1] for name in names]
    )
    users = np.concatenate([splits[name].users for name in names])
    return matrices, gains, disagreements, users


def evaluate_outer(
    held_name: str,
    train_names: list[str],
    splits: dict[str, SplitData],
    features: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_x, train_gain, train_disagreement, train_users = concatenate_splits(
        train_names, splits, features
    )
    nested_scores, nested_user_audit = cross_user_scores(
        train_x, train_gain, train_users
    )
    selected, grid = select_threshold(
        nested_scores, train_gain, train_disagreement, train_users
    )
    held = splits[held_name]
    held_gain, held_disagreement = gain_vector(held)
    held_scores, member_scores = fit_ensemble(
        train_x, train_gain, features[held_name]
    )
    route = held_disagreement & (held_scores >= float(selected["threshold"]))
    candidate_prediction = held.visual_probability[CANDIDATE_NAME].argmax(axis=1)
    prediction = held.safe_prediction.copy()
    prediction[route] = candidate_prediction[route]
    held_threshold = threshold_report(
        float(selected["threshold"]),
        held_scores,
        held_gain,
        held_disagreement,
        held.users,
    )
    report = {
        "held_split": held_name,
        "train_splits": train_names,
        "train_rows": int(len(train_x)),
        "train_users": sorted(set(train_users.astype(str).tolist())),
        "nested_threshold_selection": selected,
        "nested_user_audit": nested_user_audit,
        "nested_top_thresholds": sorted(
            grid,
            key=lambda row: (
                row["minimum_user_gain"] >= 0,
                row["net_gain"],
                row["rescue_precision"],
            ),
            reverse=True,
        )[:10],
        "safe_metrics": classification_metrics(held.labels, held.safe_prediction),
        "candidate_metrics": classification_metrics(held.labels, candidate_prediction),
        "router_metrics": classification_metrics(held.labels, prediction),
        "held_route_audit": held_threshold,
        "member_score_correlation": np.corrcoef(
            np.stack(list(member_scores.values()), axis=0)
        ).tolist(),
    }
    return report, prediction, held_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "full"), default="gate")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    splits = load_splits()
    feature_values: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    for name, split in splits.items():
        matrix, names = build_features(split)
        if feature_names is not None and names != feature_names:
            raise ValueError("router feature definitions differ by split")
        feature_names = names
        feature_values[name] = matrix

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "candidate": CANDIDATE_NAME,
        "feature_count": len(feature_names or []),
        "feature_names": feature_names,
        "model_members": [name for name, _ in model_factories()],
        "training_target": (
            "conditional pairwise correctness: rescue (+1) versus harm (-1); "
            "neutral disagreements do not affect accuracy"
        ),
        "threshold_rule": (
            "selected only on nested leave-one-user-out scores of outer-training "
            "cohorts; require >=10 decisive routes, positive net, and no train-user regression"
        ),
    }

    gate_report, gate_prediction, gate_scores = evaluate_outer(
        "H3_independent_fold0",
        ["H1_selection", "H2_confirmation"],
        splits,
        feature_values,
    )
    gate_passed = bool(gate_report["held_route_audit"]["net_gain"] >= 3)
    gate_payload = {
        "protocol": (
            "Frozen model family and feature definition; trained on H1+H2 OOF "
            "teachers with nested leave-one-user-out thresholding; H3 evaluated once."
        ),
        "frozen": frozen,
        "gate_passed": gate_passed,
        "minimum_gate_net_gain": 3,
        "result": gate_report,
    }
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "gate_predictions.npz",
        sample_ids=splits["H3_independent_fold0"].sample_ids,
        labels=splits["H3_independent_fold0"].labels,
        users=splits["H3_independent_fold0"].users,
        safe_prediction=splits["H3_independent_fold0"].safe_prediction,
        router_prediction=gate_prediction,
        route_score=gate_scores,
    )
    print(json.dumps(gate_payload, ensure_ascii=False, indent=2), flush=True)

    if args.stage != "full":
        return
    if not gate_passed:
        raise RuntimeError("H3 gate did not pass; refusing to run the other cohorts")

    outer_recipes = {
        "H1_selection": ["H2_confirmation", "H3_independent_fold0"],
        "H2_confirmation": ["H1_selection", "H3_independent_fold0"],
        "H3_independent_fold0": ["H1_selection", "H2_confirmation"],
    }
    reports: dict[str, Any] = {}
    saved: dict[str, np.ndarray] = {}
    total_safe = 0
    total_router = 0
    total_rows = 0
    for held_name, train_names in outer_recipes.items():
        report, prediction, scores = evaluate_outer(
            held_name, train_names, splits, feature_values
        )
        reports[held_name] = report
        split = splits[held_name]
        total_safe += int(np.sum(split.safe_prediction == split.labels))
        total_router += int(np.sum(prediction == split.labels))
        total_rows += len(split.labels)
        saved[f"{held_name}_sample_ids"] = split.sample_ids
        saved[f"{held_name}_labels"] = split.labels
        saved[f"{held_name}_users"] = split.users
        saved[f"{held_name}_safe_prediction"] = split.safe_prediction
        saved[f"{held_name}_router_prediction"] = prediction
        saved[f"{held_name}_route_score"] = scores
    full_payload = {
        "protocol": (
            "Three outer cohorts. For each held cohort, router models and route "
            "threshold use only the other two cohorts; threshold is nested LOSO."
        ),
        "frozen": frozen,
        "cohorts": reports,
        "aggregate": {
            "rows": total_rows,
            "safe_correct": total_safe,
            "router_correct": total_router,
            "safe_accuracy": total_safe / total_rows,
            "router_accuracy": total_router / total_rows,
            "net_gain": total_router - total_safe,
        },
    }
    (args.output_dir / "full_summary.json").write_text(
        json.dumps(full_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "full_predictions.npz", **saved)
    print(json.dumps(full_payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
