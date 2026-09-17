from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import classification_metrics
from p88_oof_candidate_ensemble import CANDIDATE_SOURCES, load_candidate, load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm
from p89_global_repeat_decoder import GlobalRepeatConfig, decode_global_repeat


PROJECT_DIR = Path(__file__).resolve().parent
CLASS_COUNT = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a high-precision alternative-class accept/reject model by user-LOO "
            "on H1, then transfer it once to untouched H2."
        )
    )
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--selection-users", nargs="+", required=True)
    parser.add_argument("--confirmation-run", type=Path, required=True)
    parser.add_argument("--confirmation-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--global-repeat-summary",
        type=Path,
        default=PROJECT_DIR / "runs/p89_global_repeat_h1_v1/summary.json",
    )
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        default=PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz",
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
    )
    parser.add_argument(
        "--repeat-config-summary",
        type=Path,
        default=PROJECT_DIR / "runs/p88_aligned_repeat_h1_v1/summary.json",
    )
    parser.add_argument("--class-bias", type=Path)
    return parser.parse_args()


def protocol(args: argparse.Namespace, run: Path, users: list[str]):
    return load_protocol(
        SimpleNamespace(
            base_run=run,
            holdout_users=users,
            teacher_targets=args.teacher_targets,
            train_metadata=args.train_metadata,
            repeat_config_summary=args.repeat_config_summary,
        )
    )


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(values))


def expert_probabilities(sample_ids: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = [np.asarray(base, dtype=np.float64)]
    names = ["p87"]
    for name in CANDIDATE_SOURCES:
        values.append(softmax(load_candidate(name, sample_ids)))
        names.append(name)
    return np.stack(values, axis=1), names


def entropy(probability: np.ndarray) -> np.ndarray:
    return -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=-1)


@dataclass
class ProposalTable:
    features: np.ndarray
    sample_index: np.ndarray
    proposed_class: np.ndarray
    target: np.ndarray | None


def build_proposals(
    probabilities: np.ndarray,
    base_prediction: np.ndarray,
    labels: np.ndarray | None,
) -> ProposalTable:
    """One row per distinct expert top-1 alternative, rather than per expert.

    Collapsing duplicate experts prevents the many P85 mechanism ablations from
    dominating training while retaining their agreement as an explicit feature.
    """

    n, expert_count, class_count = probabilities.shape
    if class_count != CLASS_COUNT:
        raise ValueError(f"expected {CLASS_COUNT} classes, got {class_count}")
    expert_top1 = probabilities.argmax(axis=2)
    expert_order = np.argsort(probabilities, axis=2)
    expert_top2 = expert_order[:, :, -2:]
    expert_entropy = entropy(probabilities)
    base_raw_class = probabilities[:, 0].argmax(axis=1)
    base_ordered = np.sort(probabilities[:, 0], axis=1)
    base_entropy = entropy(probabilities[:, 0])

    rows: list[np.ndarray] = []
    sample_indices: list[int] = []
    proposed_classes: list[int] = []
    targets: list[int] = []
    for i in range(n):
        alternatives = sorted(set(expert_top1[i].tolist()) - {int(base_prediction[i])})
        for candidate in alternatives:
            candidate_probability = probabilities[i, :, candidate]
            decoded_probability = probabilities[i, :, int(base_prediction[i])]
            difference = candidate_probability - decoded_probability
            top1_votes = (expert_top1[i] == candidate).astype(np.float64)
            top2_votes = np.any(expert_top2[i] == candidate, axis=1).astype(np.float64)
            aggregates = np.asarray(
                [
                    candidate_probability.mean(),
                    candidate_probability.max(),
                    candidate_probability.std(),
                    np.median(candidate_probability),
                    decoded_probability.mean(),
                    decoded_probability.max(),
                    decoded_probability.std(),
                    np.median(decoded_probability),
                    difference.mean(),
                    difference.max(),
                    difference.std(),
                    np.median(difference),
                    top1_votes.sum(),
                    top2_votes.sum(),
                    probabilities[i, 0, candidate],
                    probabilities[i, 0, int(base_prediction[i])],
                    probabilities[i, 0, candidate]
                    - probabilities[i, 0, int(base_prediction[i])],
                    base_ordered[i, -1],
                    base_ordered[i, -1] - base_ordered[i, -2],
                    base_entropy[i],
                    expert_entropy[i].mean(),
                    expert_entropy[i].std(),
                    float(base_raw_class[i] == base_prediction[i]),
                    float(candidate == base_raw_class[i]),
                ],
                dtype=np.float64,
            )
            base_one_hot = np.zeros(CLASS_COUNT, dtype=np.float64)
            candidate_one_hot = np.zeros(CLASS_COUNT, dtype=np.float64)
            base_one_hot[int(base_prediction[i])] = 1.0
            candidate_one_hot[candidate] = 1.0
            row = np.concatenate(
                (
                    candidate_probability,
                    decoded_probability,
                    difference,
                    top1_votes,
                    top2_votes,
                    aggregates,
                    base_one_hot,
                    candidate_one_hot,
                )
            )
            rows.append(row)
            sample_indices.append(i)
            proposed_classes.append(candidate)
            if labels is not None:
                targets.append(int(candidate == int(labels[i])))
    if not rows:
        raise RuntimeError("no alternative-class proposals")
    return ProposalTable(
        features=np.asarray(rows, dtype=np.float32),
        sample_index=np.asarray(sample_indices, dtype=np.int64),
        proposed_class=np.asarray(proposed_classes, dtype=np.int64),
        target=np.asarray(targets, dtype=np.int64) if labels is not None else None,
    )


def make_model(config: dict[str, Any]):
    if config["model"] == "hist":
        return HistGradientBoostingClassifier(
            learning_rate=float(config["learning_rate"]),
            max_iter=int(config["iterations"]),
            max_leaf_nodes=int(config["leaves"]),
            min_samples_leaf=int(config["min_leaf"]),
            l2_regularization=float(config["l2"]),
            random_state=20260816,
        )
    return LogisticRegression(
        C=float(config["regularization"]),
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=20260816,
    )


def fit_score(
    config: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    valid_scaled = scaler.transform(valid_x)
    model = make_model(config)
    if config["model"] == "hist":
        # A moderate positive weight asks the nonlinear model for recall; the
        # cross-fitted acceptance threshold below is responsible for precision.
        positive_weight = float(config["positive_weight"])
        sample_weight = np.where(train_y == 1, positive_weight, 1.0)
        model.fit(train_scaled, train_y, sample_weight=sample_weight)
    else:
        model.fit(train_scaled, train_y)
    return model.predict_proba(valid_scaled)[:, 1]


def accept_predictions(
    base_prediction: np.ndarray,
    proposals: ProposalTable,
    score: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = base_prediction.copy()
    accepted = np.zeros(len(base_prediction), dtype=bool)
    for sample_index in np.unique(proposals.sample_index):
        positions = np.flatnonzero(proposals.sample_index == sample_index)
        best_position = positions[int(np.argmax(score[positions]))]
        if float(score[best_position]) >= threshold:
            prediction[sample_index] = proposals.proposed_class[best_position]
            accepted[sample_index] = True
    return prediction, accepted


def decode(probability_value, protocol_value, config):
    _, labels, _, _, metadata, indices, _, transition, decoder, _ = protocol_value
    prediction, grouping = decode_global_repeat(
        np.log(np.maximum(probability_value, 1e-12)),
        indices,
        metadata,
        transition,
        decoder,
        config,
    )
    return prediction, grouping, classification_metrics(labels, prediction)


def main() -> None:
    args = parse_args()
    selection = protocol(args, args.selection_run.resolve(), list(args.selection_users))
    confirmation = protocol(args, args.confirmation_run.resolve(), list(args.confirmation_users))
    ids1, labels1, base1, _, metadata1, *_ = selection
    ids2, labels2, base2, _, metadata2, *_ = confirmation
    global_source = json.loads(args.global_repeat_summary.resolve().read_text(encoding="utf-8"))
    global_config = GlobalRepeatConfig(**global_source["selected_config"])
    base_prediction1, base_grouping1, base_metrics1 = decode(base1, selection, global_config)
    base_prediction2, base_grouping2, base_metrics2 = decode(base2, confirmation, global_config)
    experts1, expert_names = expert_probabilities(ids1, base1)
    experts2, expert_names2 = expert_probabilities(ids2, base2)
    if expert_names != expert_names2:
        raise RuntimeError("expert mismatch")
    proposals1 = build_proposals(experts1, base_prediction1, labels1)
    proposals2 = build_proposals(experts2, base_prediction2, labels2)
    users1 = metadata1.users.astype(str)

    configs: list[dict[str, Any]] = [
        {"model": "logistic", "regularization": c}
        for c in (0.0003, 0.001, 0.003, 0.01, 0.03)
    ]
    configs.extend(
        {
            "model": "hist",
            "learning_rate": 0.05,
            "iterations": 100,
            "leaves": leaves,
            "min_leaf": min_leaf,
            "l2": l2,
            "positive_weight": positive_weight,
        }
        for leaves in (3, 7)
        for min_leaf in (20, 40)
        for l2 in (3.0, 10.0)
        for positive_weight in (2.0, 4.0)
    )
    thresholds = np.round(np.arange(0.50, 0.981, 0.02), 3)
    all_candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key = None
    for config in configs:
        crossfit_score = np.zeros(len(proposals1.features), dtype=np.float64)
        for user in sorted(set(users1.tolist())):
            valid_samples = users1 == user
            valid_rows = valid_samples[proposals1.sample_index]
            train_rows = ~valid_rows
            crossfit_score[valid_rows] = fit_score(
                config,
                proposals1.features[train_rows],
                proposals1.target[train_rows],
                proposals1.features[valid_rows],
            )
        for threshold in thresholds:
            prediction, accepted = accept_predictions(
                base_prediction1, proposals1, crossfit_score, float(threshold)
            )
            metrics = classification_metrics(labels1, prediction)
            change = rescue_harm(labels1, base_prediction1, prediction)
            item = {
                "configuration": config,
                "threshold": float(threshold),
                "metrics": metrics,
                "rescue_harm": change,
                "accepted": int(accepted.sum()),
            }
            all_candidates.append(item)
            # Optimize net corrections, with explicit precision and sparsity
            # tie-breakers so a fragile high-change route is never preferred.
            precision = change["rescue"] / max(change["rescue"] + change["harm"], 1)
            key = (
                metrics["correct"],
                metrics["balanced_accuracy"],
                precision,
                -change["harm"],
                -int(accepted.sum()),
            )
            if best_key is None or key > best_key:
                best_key, best = key, item
        print(f"finished {config}", flush=True)
    assert best is not None

    final_score2 = fit_score(
        best["configuration"], proposals1.features, proposals1.target, proposals2.features
    )
    prediction2, accepted2 = accept_predictions(
        base_prediction2, proposals2, final_score2, float(best["threshold"])
    )
    confirmation_metrics = classification_metrics(labels2, prediction2)
    confirmation_change = rescue_harm(labels2, base_prediction2, prediction2)

    biased_result = None
    if args.class_bias:
        logits2 = np.asarray(
            np.load(args.confirmation_run.resolve() / "subject_holdout_logits.npy"),
            dtype=np.float64,
        )
        bias = np.asarray(np.load(args.class_bias.resolve()), dtype=np.float64)
        biased_base = softmax(logits2 + bias)
        biased_prediction, biased_grouping, biased_metrics = decode(
            biased_base, confirmation, global_config
        )
        # Keep only learned corrections that still disagree with the biased route.
        biased_corrected = biased_prediction.copy()
        biased_corrected[accepted2] = prediction2[accepted2]
        biased_result = {
            "base": biased_metrics,
            "metrics": classification_metrics(labels2, biased_corrected),
            "rescue_harm": rescue_harm(labels2, biased_prediction, biased_corrected),
            "grouping": biased_grouping,
        }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "P89_pairwise_high_precision_correction_H1_LOO_to_H2_v1",
        "status": "complete",
        "protocol": (
            "Alternative-class scorer and acceptance threshold selected from user-LOO "
            "predictions inside H1, refit on all H1, transferred once to untouched H2."
        ),
        "expert_names": expert_names,
        "feature_dim": int(proposals1.features.shape[1]),
        "selection_proposals": int(len(proposals1.features)),
        "selection_positive_proposals": int(proposals1.target.sum()),
        "selection_base": base_metrics1,
        "selection_base_grouping": base_grouping1,
        "selection_best_user_loo": best,
        "confirmation_base": base_metrics2,
        "confirmation_base_grouping": base_grouping2,
        "confirmation": {
            "metrics": confirmation_metrics,
            "rescue_harm": confirmation_change,
            "accepted": int(accepted2.sum()),
        },
        "confirmation_with_H1_class_bias": biased_result,
        "grid_size": len(all_candidates),
        "all_selection_candidates": all_candidates,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "confirmation_predictions.npz",
        sample_ids=ids2,
        labels=labels2,
        base_prediction=base_prediction2,
        corrected_prediction=prediction2,
        accepted=accepted2,
        proposal_sample_index=proposals2.sample_index,
        proposal_class=proposals2.proposed_class,
        proposal_score=final_score2,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "all_selection_candidates"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
