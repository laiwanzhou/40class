"""Source-OOF candidate-level multimodal reranker with a hard H3 firewall.

The model treats experts as evidence channels, not votes.  A LambdaRank model
scores every class from class-conditional visual/Skeleton/IMU probabilities,
then a source-OOF correction selector learns when the ranked candidate is safer
than the current anchor.  This file intentionally contains no H3 evaluation
path: only an H1+embargo OOF screen and one frozen H2 confirmation are allowed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

from p90_teacher_fusion_audit import align
from p91_hierarchical_multimodal_teacher import blend_prediction
from p91_unrestricted_fusion_teacher import Cohort, build_cohorts, concatenate


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTPUT = PROJECT / "runs/p94_candidate_multimodal_reranker_h1h2_v1"
ROUTER = PROJECT / "runs/p90_crossuser_visual_router_v1/full_predictions.npz"
P91 = PROJECT / "runs/p91_hierarchical_multimodal_h3_v3/inner_predictions.npz"
SELECTED_EXPERTS = (
    "p89_safe_raw",
    "internvideo2_l_early_late_plus_k400",
    "videomaev2_base_plus_internvideo2_l_equal",
    "videomaev2_distilled_base",
    "skeleton_invariant",
    "motionbert_front",
    "imu_sensorwise_deep",
)
VISUAL_SLICE = slice(1, 4)
SENSOR_SLICE = slice(4, 7)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def selected_probability(cohort: Cohort) -> np.ndarray:
    index = {name: offset for offset, name in enumerate(cohort.expert_names)}
    missing = set(SELECTED_EXPERTS) - set(index)
    if missing:
        raise ValueError(f"missing experts: {sorted(missing)}")
    return cohort.expert_probability[
        :, [index[name] for name in SELECTED_EXPERTS]
    ].astype(np.float32)


def evidence_features(probability: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Create one evidence token for every sample x class pair."""
    probability = np.clip(probability.astype(np.float32), 1e-7, 1.0)
    samples, experts, classes = probability.shape
    if classes != 40 or experts != len(SELECTED_EXPERTS):
        raise ValueError("unexpected expert evidence geometry")
    order = np.argsort(-probability, axis=2)
    rank = np.empty_like(order)
    rows = np.arange(samples)[:, None, None]
    expert_rows = np.arange(experts)[None, :, None]
    rank[rows, expert_rows, order] = np.arange(classes)[None, None, :]
    rank_score = 1.0 - rank.astype(np.float32) / float(classes - 1)
    top1 = (rank == 0).astype(np.float32)

    visual = probability[:, VISUAL_SLICE]
    sensor = probability[:, SENSOR_SLICE]
    class_values = [
        np.log(probability).transpose(0, 2, 1),
        probability.transpose(0, 2, 1),
        rank_score.transpose(0, 2, 1),
        top1.transpose(0, 2, 1),
        visual.mean(axis=1)[..., None],
        visual.max(axis=1)[..., None],
        sensor.mean(axis=1)[..., None],
        sensor.max(axis=1)[..., None],
        (sensor.mean(axis=1) - visual.mean(axis=1))[..., None],
        top1.sum(axis=1)[..., None],
    ]
    base_flag = np.zeros((samples, classes, 1), dtype=np.float32)
    base_flag[np.arange(samples), base.astype(np.int64), 0] = 1.0
    class_values.append(base_flag)
    class_identity = np.eye(classes, dtype=np.float32)[None].repeat(samples, axis=0)
    class_values.append(class_identity)

    sorted_probability = np.sort(probability, axis=2)
    maximum = sorted_probability[:, :, -1]
    margin = sorted_probability[:, :, -1] - sorted_probability[:, :, -2]
    entropy = -(probability * np.log(probability)).sum(axis=2)
    disagreement = (
        np.max(probability.argmax(axis=2), axis=1)
        != np.min(probability.argmax(axis=2), axis=1)
    ).astype(np.float32)[:, None]
    global_values = np.concatenate((maximum, margin, entropy, disagreement), axis=1)
    global_values = global_values[:, None, :].repeat(classes, axis=1)
    class_values.append(global_values)
    return np.concatenate(class_values, axis=2).reshape(samples * classes, -1)


def relevance(labels: np.ndarray) -> np.ndarray:
    output = np.zeros((len(labels), 40), dtype=np.int32)
    output[np.arange(len(labels)), labels.astype(np.int64)] = 1
    return output.reshape(-1)


def make_ranker(seed: int) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=320,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_ranker(
    probability: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> lgb.LGBMRanker:
    model = make_ranker(seed)
    model.fit(
        evidence_features(probability, base),
        relevance(labels),
        group=np.full(len(labels), 40, dtype=np.int32),
    )
    return model


def ranker_scores(
    model: lgb.LGBMRanker, probability: np.ndarray, base: np.ndarray
) -> np.ndarray:
    return np.asarray(
        model.predict(evidence_features(probability, base)), dtype=np.float64
    ).reshape(len(base), 40)


def candidate_prediction(scores: np.ndarray) -> np.ndarray:
    return scores.argmax(axis=1).astype(np.int64)


def selector_features(
    scores: np.ndarray,
    probability: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    samples = len(base)
    rows = np.arange(samples)
    sorted_scores = np.sort(scores, axis=1)
    base_score = scores[rows, base]
    candidate_score = scores[rows, candidate]
    base_rank = (scores > base_score[:, None]).sum(axis=1) / 39.0
    candidate_probability = probability[rows, :, candidate]
    base_probability = probability[rows, :, base]
    evidence_delta = candidate_probability - base_probability
    log_delta = np.log(np.clip(candidate_probability, 1e-7, 1.0)) - np.log(
        np.clip(base_probability, 1e-7, 1.0)
    )
    candidate_votes = (probability.argmax(axis=2) == candidate[:, None]).sum(axis=1)
    base_votes = (probability.argmax(axis=2) == base[:, None]).sum(axis=1)
    return np.concatenate(
        (
            (candidate_score - base_score)[:, None],
            (sorted_scores[:, -1] - sorted_scores[:, -2])[:, None],
            base_rank[:, None],
            candidate_votes[:, None],
            base_votes[:, None],
            evidence_delta,
            log_delta,
            np.eye(40, dtype=np.float32)[candidate],
            np.eye(40, dtype=np.float32)[base],
        ),
        axis=1,
    ).astype(np.float32)


def audit(
    labels: np.ndarray,
    base: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
) -> dict[str, Any]:
    base_correct = base == labels
    predicted_correct = prediction == labels
    per_user: dict[str, dict[str, int | float]] = {}
    nets = []
    for user in np.unique(users):
        selected = users == user
        old = int(base_correct[selected].sum())
        new = int(predicted_correct[selected].sum())
        nets.append(new - old)
        per_user[str(user)] = {
            "total": int(selected.sum()),
            "base_correct": old,
            "candidate_correct": new,
            "net": new - old,
            "candidate_accuracy": new / int(selected.sum()),
        }
    return {
        "total": int(len(labels)),
        "base_correct": int(base_correct.sum()),
        "candidate_correct": int(predicted_correct.sum()),
        "base_accuracy": float(base_correct.mean()),
        "candidate_accuracy": float(predicted_correct.mean()),
        "rescue": int((~base_correct & predicted_correct).sum()),
        "harm": int((base_correct & ~predicted_correct).sum()),
        "changed": int((prediction != base).sum()),
        "net": int(predicted_correct.sum() - base_correct.sum()),
        "worst_user_net": int(min(nets)),
        "negative_users": int(sum(value < 0 for value in nets)),
        "per_user": per_user,
    }


def topk_coverage(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    top = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean((top == labels[:, None]).any(axis=1)))


def router_base(router: dict[str, np.ndarray], name: str, ids: np.ndarray) -> np.ndarray:
    return align(
        router[f"{name}_sample_ids"].astype(str),
        router[f"{name}_router_prediction"],
        ids.astype(str),
    ).astype(np.int64)


def h2_p91_champion(ids: np.ndarray) -> np.ndarray:
    values = load_npz(P91)
    weight = float(np.asarray(values["selected_constant_weight"]).reshape(-1)[0])
    prediction = blend_prediction(
        values["direct_logits"], values["teacher_prediction"], weight
    )
    return align(values["sample_ids"].astype(str), prediction, ids.astype(str)).astype(
        np.int64
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohorts = build_cohorts()
    h1 = cohorts["H1_selection"]
    embargo = cohorts["E0_p87_sequence_source"]
    h2 = cohorts["H2_confirmation"]
    source = concatenate("P94_source", [h1, embargo])
    router = load_npz(ROUTER)
    source_base = np.concatenate(
        (
            router_base(router, "H1_selection", h1.sample_ids),
            embargo.safe_prediction.astype(np.int64),
        )
    )
    h2_base = h2_p91_champion(h2.sample_ids)
    source_probability = selected_probability(source)
    h2_probability = selected_probability(h2)

    oof_scores = np.zeros((len(source.labels), 40), dtype=np.float64)
    for fold, user in enumerate(np.unique(source.users)):
        held = source.users == user
        fit = ~held
        model = fit_ranker(
            source_probability[fit],
            source_base[fit],
            source.labels[fit],
            seed=20260820 + fold,
        )
        oof_scores[held] = ranker_scores(
            model, source_probability[held], source_base[held]
        )
    source_candidate = candidate_prediction(oof_scores)
    source_selector_x = selector_features(
        oof_scores, source_probability, source_base, source_candidate
    )
    affected = (source_candidate == source.labels) ^ (source_base == source.labels)
    selector_target = (source_candidate == source.labels).astype(np.int64)
    if (
        affected.sum() < 20
        or np.unique(selector_target[affected]).size < 2
    ):
        raise RuntimeError("source OOF does not contain enough rescue/harm selector rows")
    # The selector must be OOF too.  Fitting it once on all OOF ranker outcomes
    # and predicting the same samples would leak rescue/harm labels into the
    # threshold screen even though the ranker itself is cross-fitted.
    source_route_probability = np.zeros(len(source.labels), dtype=np.float64)
    for fold, user in enumerate(np.unique(source.users)):
        held = source.users == user
        fit = affected & ~held
        if fit.sum() < 20 or np.unique(selector_target[fit]).size < 2:
            raise RuntimeError(f"selector fold {user} lacks rescue/harm diversity")
        fold_selector = LogisticRegression(
            C=0.25,
            max_iter=3000,
            random_state=20260820 + fold,
        )
        fold_selector.fit(source_selector_x[fit], selector_target[fit])
        source_route_probability[held] = fold_selector.predict_proba(
            source_selector_x[held]
        )[:, 1]

    thresholds = np.linspace(0.50, 0.95, 19)
    source_grid = []
    for threshold in thresholds:
        route = (source_candidate != source_base) & (
            source_route_probability >= threshold
        )
        prediction = np.where(route, source_candidate, source_base)
        source_grid.append(
            {
                "threshold": float(threshold),
                **audit(
                    source.labels,
                    source_base,
                    prediction,
                    source.users,
                ),
            }
        )
    selected = max(
        source_grid,
        key=lambda row: (
            row["candidate_correct"],
            -row["harm"],
            -row["negative_users"],
            row["worst_user_net"],
            row["threshold"],
        ),
    )
    threshold = float(selected["threshold"])
    source_route = (source_candidate != source_base) & (
        source_route_probability >= threshold
    )
    source_prediction = np.where(source_route, source_candidate, source_base)

    final_ranker = fit_ranker(
        source_probability, source_base, source.labels, seed=20260820
    )
    selector = LogisticRegression(C=0.25, max_iter=3000, random_state=20260820)
    selector.fit(source_selector_x[affected], selector_target[affected])
    h2_scores = ranker_scores(final_ranker, h2_probability, h2_base)
    h2_candidate = candidate_prediction(h2_scores)
    h2_selector_x = selector_features(
        h2_scores, h2_probability, h2_base, h2_candidate
    )
    h2_route_probability = selector.predict_proba(h2_selector_x)[:, 1]
    h2_route = (h2_candidate != h2_base) & (h2_route_probability >= threshold)
    h2_prediction = np.where(h2_route, h2_candidate, h2_base)

    source_audit = audit(
        source.labels, source_base, source_prediction, source.users
    )
    h2_audit = audit(h2.labels, h2_base, h2_prediction, h2.users)
    source_gate = (
        source_audit["net"] >= 8
        and source_audit["worst_user_net"] >= -2
    )
    h2_gate = h2_audit["net"] >= 8 and h2_audit["worst_user_net"] >= -2
    report = {
        "status": "complete_h1_h2_only",
        "protocol": (
            "LOUO LambdaRank on H1+embargo selects one correction threshold; "
            "one frozen H2 confirmation against the P91 constant-blend champion. "
            "The script has no H3 evaluation path."
        ),
        "experts_as_channels_not_votes": list(SELECTED_EXPERTS),
        "source": {
            "samples": int(len(source.labels)),
            "users": sorted(np.unique(source.users).tolist()),
            "selected_threshold": threshold,
            "ranker_direct": audit(
                source.labels, source_base, source_candidate, source.users
            ),
            "selected": source_audit,
            "top5_coverage": topk_coverage(oof_scores, source.labels, 5),
            "top10_coverage": topk_coverage(oof_scores, source.labels, 10),
            "top_thresholds": sorted(
                source_grid,
                key=lambda row: (row["candidate_correct"], -row["harm"]),
                reverse=True,
            )[:5],
        },
        "h2": {
            "samples": int(len(h2.labels)),
            "ranker_direct": audit(h2.labels, h2_base, h2_candidate, h2.users),
            "selected": h2_audit,
            "top5_coverage": topk_coverage(h2_scores, h2.labels, 5),
            "top10_coverage": topk_coverage(h2_scores, h2.labels, 10),
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
        ranker_scores=h2_scores.astype(np.float32),
        ranker_prediction=h2_candidate,
        route_probability=h2_route_probability.astype(np.float32),
        selected_threshold=np.asarray(threshold, dtype=np.float32),
        selected_prediction=h2_prediction,
    )
    write_rows(
        OUTPUT / "h2_samples.csv",
        [
            {
                "sample_id": h2.sample_ids[row],
                "user_id": h2.users[row],
                "label": int(h2.labels[row]),
                "base_prediction": int(h2_base[row]),
                "ranker_prediction": int(h2_candidate[row]),
                "selected_prediction": int(h2_prediction[row]),
                "route_probability": float(h2_route_probability[row]),
                "routed": int(h2_route[row]),
            }
            for row in range(len(h2.labels))
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
