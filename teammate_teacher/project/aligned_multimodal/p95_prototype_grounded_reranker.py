"""Prototype-grounded candidate reranking with an H1/H2-only firewall.

P94 showed that the correct label is usually present in the candidate set but
that probability-only rescue reliability does not transfer across users.  P95
therefore adds class-conditional similarities from frozen visual, Skeleton,
MotionBERT, IMU, and cross-modal relation representations.  Every scaler, PCA,
prototype, ranker, and selector is fitted without the held user.
"""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

from p91_unrestricted_fusion_teacher import (
    Cohort,
    Prepared,
    Preprocessor,
    build_cohorts,
    concatenate,
)
from p94_candidate_multimodal_reranker import (
    P91,
    ROUTER,
    SELECTED_EXPERTS,
    audit,
    candidate_prediction,
    evidence_features,
    h2_p91_champion,
    load_npz,
    make_ranker,
    relevance,
    router_base,
    selected_probability,
    selector_features,
    topk_coverage,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTPUT = PROJECT / "runs/p95_prototype_grounded_reranker_h1h2_v1"
BLOCK_NAMES = (
    "vmae",
    "internvideo2",
    "motionbert",
    "skeleton",
    "imu",
    "relation",
)


def subset(source: Cohort, mask: np.ndarray, name: str) -> Cohort:
    values: dict[str, Any] = {"name": name}
    for field in fields(Cohort):
        if field.name == "name":
            continue
        value = getattr(source, field.name)
        values[field.name] = value if field.name == "expert_names" else value[mask]
    return Cohort(**values)


def representation_blocks(source: Prepared) -> list[np.ndarray]:
    return [
        source.vmae_tokens.mean(axis=1),
        source.iv2_tokens.mean(axis=1),
        source.motionbert_tokens.mean(axis=1),
        source.skeleton_statistics,
        source.imu_statistics,
        source.relation_statistics,
    ]


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def block_prototype_signals(
    train: np.ndarray,
    train_labels: np.ndarray,
    evaluation: np.ndarray,
    leave_one_out_train: bool,
) -> np.ndarray:
    train = np.asarray(train, dtype=np.float64)
    evaluation = np.asarray(evaluation, dtype=np.float64)
    if leave_one_out_train:
        if len(evaluation) != len(train):
            raise ValueError("leave-one-out prototype signals require train-sized input")
        evaluation = train
    classes = 40
    counts = np.bincount(train_labels, minlength=classes).astype(np.float64)
    sums = np.zeros((classes, train.shape[1]), dtype=np.float64)
    np.add.at(sums, train_labels, train)
    global_centroid = train.mean(axis=0)
    centroids = np.repeat(global_centroid[None], classes, axis=0)
    observed = counts > 0
    centroids[observed] = sums[observed] / counts[observed, None]

    evaluation_unit = unit(evaluation)
    centroid_unit = unit(centroids)
    cosine = evaluation_unit @ centroid_unit.T
    distance = -(
        (evaluation[:, None, :] - centroids[None, :, :]) ** 2
    ).mean(axis=2)

    if leave_one_out_train:
        rows = np.arange(len(train))
        own = train_labels.astype(np.int64)
        own_count = counts[own]
        loo_centroid = np.empty_like(train)
        repeated = own_count > 1
        loo_centroid[repeated] = (sums[own[repeated]] - train[repeated]) / (
            own_count[repeated, None] - 1.0
        )
        # A singleton class has no same-class evidence once its own row is
        # removed.  Falling back to the all-other-sample mean is deliberately
        # non-discriminative and does not encode the row's target label.
        singleton = ~repeated
        loo_centroid[singleton] = (
            train.sum(axis=0)[None] - train[singleton]
        ) / float(len(train) - 1)
        cosine[rows, own] = np.sum(unit(train) * unit(loo_centroid), axis=1)
        distance[rows, own] = -((train - loo_centroid) ** 2).mean(axis=1)
    return np.stack((cosine, distance), axis=2).astype(np.float32)


def prototype_signals(
    train: Prepared,
    evaluation: Prepared,
    leave_one_out_train: bool,
) -> np.ndarray:
    output = []
    for train_block, evaluation_block in zip(
        representation_blocks(train), representation_blocks(evaluation), strict=True
    ):
        # Passing the identical ndarray preserves shares_memory for the exact
        # leave-one-out training path.
        if leave_one_out_train:
            evaluation_block = train_block
        output.append(
            block_prototype_signals(
                train_block,
                train.labels,
                evaluation_block,
                leave_one_out_train,
            )
        )
    return np.concatenate(output, axis=2)


def augmented_evidence_features(
    probability: np.ndarray,
    base: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    samples = len(base)
    if prototypes.shape != (samples, 40, len(BLOCK_NAMES) * 2):
        raise ValueError(f"unexpected prototype geometry: {prototypes.shape}")
    return np.concatenate(
        (
            evidence_features(probability, base),
            prototypes.reshape(samples * 40, -1),
        ),
        axis=1,
    )


def fit_ranker(
    probability: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    seed: int,
) -> lgb.LGBMRanker:
    model = make_ranker(seed)
    model.fit(
        augmented_evidence_features(probability, base, prototypes),
        relevance(labels),
        group=np.full(len(labels), 40, dtype=np.int32),
    )
    return model


def ranker_scores(
    model: lgb.LGBMRanker,
    probability: np.ndarray,
    base: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    values = model.predict(augmented_evidence_features(probability, base, prototypes))
    return np.asarray(values, dtype=np.float64).reshape(len(base), 40)


def augmented_selector_features(
    scores: np.ndarray,
    probability: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    rows = np.arange(len(base))
    candidate_signal = prototypes[rows, candidate]
    base_signal = prototypes[rows, base]
    return np.concatenate(
        (
            selector_features(scores, probability, base, candidate),
            candidate_signal,
            base_signal,
            candidate_signal - base_signal,
        ),
        axis=1,
    ).astype(np.float32)


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
    source = concatenate("P95_source", [h1, embargo])
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
    oof_prototypes = np.zeros(
        (len(source.labels), 40, len(BLOCK_NAMES) * 2), dtype=np.float32
    )
    for fold, user in enumerate(np.unique(source.users)):
        held_mask = source.users == user
        fit_mask = ~held_mask
        fit_cohort = subset(source, fit_mask, f"fit_without_{user}")
        held_cohort = subset(source, held_mask, f"held_{user}")
        preprocessor = Preprocessor(statistics_dim=48, seed=20260820 + fold)
        preprocessor.fit(fit_cohort)
        fit_prepared = preprocessor.transform(fit_cohort)
        held_prepared = preprocessor.transform(held_cohort)
        fit_prototypes = prototype_signals(
            fit_prepared, fit_prepared, leave_one_out_train=True
        )
        held_prototypes = prototype_signals(
            fit_prepared, held_prepared, leave_one_out_train=False
        )
        model = fit_ranker(
            source_probability[fit_mask],
            source_base[fit_mask],
            source.labels[fit_mask],
            fit_prototypes,
            seed=20260820 + fold,
        )
        oof_scores[held_mask] = ranker_scores(
            model,
            source_probability[held_mask],
            source_base[held_mask],
            held_prototypes,
        )
        oof_prototypes[held_mask] = held_prototypes

    source_candidate = candidate_prediction(oof_scores)
    source_selector_x = augmented_selector_features(
        oof_scores,
        source_probability,
        source_base,
        source_candidate,
        oof_prototypes,
    )
    selector_target = (source_candidate == source.labels).astype(np.int64)
    affected = (source_candidate == source.labels) ^ (source_base == source.labels)
    if affected.sum() < 20 or np.unique(selector_target[affected]).size < 2:
        raise RuntimeError("source OOF lacks rescue/harm diversity")

    source_route_probability = np.zeros(len(source.labels), dtype=np.float64)
    for fold, user in enumerate(np.unique(source.users)):
        held = source.users == user
        fit = affected & ~held
        if fit.sum() < 20 or np.unique(selector_target[fit]).size < 2:
            raise RuntimeError(f"selector fold {user} lacks rescue/harm diversity")
        selector = LogisticRegression(
            C=0.20, max_iter=4000, random_state=20260820 + fold
        )
        selector.fit(source_selector_x[fit], selector_target[fit])
        source_route_probability[held] = selector.predict_proba(
            source_selector_x[held]
        )[:, 1]

    source_grid = []
    for threshold in np.linspace(0.50, 0.95, 19):
        route = (source_candidate != source_base) & (
            source_route_probability >= threshold
        )
        prediction = np.where(route, source_candidate, source_base)
        source_grid.append(
            {
                "threshold": float(threshold),
                **audit(source.labels, source_base, prediction, source.users),
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

    final_preprocessor = Preprocessor(statistics_dim=48, seed=20260820)
    final_preprocessor.fit(source)
    source_prepared = final_preprocessor.transform(source)
    h2_prepared = final_preprocessor.transform(h2)
    train_prototypes = prototype_signals(
        source_prepared, source_prepared, leave_one_out_train=True
    )
    h2_prototypes = prototype_signals(
        source_prepared, h2_prepared, leave_one_out_train=False
    )
    final_ranker = fit_ranker(
        source_probability,
        source_base,
        source.labels,
        train_prototypes,
        seed=20260820,
    )
    selector = LogisticRegression(C=0.20, max_iter=4000, random_state=20260820)
    selector.fit(source_selector_x[affected], selector_target[affected])
    h2_scores = ranker_scores(
        final_ranker, h2_probability, h2_base, h2_prototypes
    )
    h2_candidate = candidate_prediction(h2_scores)
    h2_selector_x = augmented_selector_features(
        h2_scores, h2_probability, h2_base, h2_candidate, h2_prototypes
    )
    h2_route_probability = selector.predict_proba(h2_selector_x)[:, 1]
    h2_route = (h2_candidate != h2_base) & (h2_route_probability >= threshold)
    h2_prediction = np.where(h2_route, h2_candidate, h2_base)

    source_audit = audit(source.labels, source_base, source_prediction, source.users)
    h2_audit = audit(h2.labels, h2_base, h2_prediction, h2.users)
    source_gate = source_audit["net"] >= 8 and source_audit["worst_user_net"] >= -2
    h2_gate = h2_audit["net"] >= 8 and h2_audit["worst_user_net"] >= -2
    report = {
        "status": "complete_h1_h2_only",
        "protocol": (
            "LOUO prototype/PCA/ranker/selector on H1+embargo; one frozen H2 "
            "confirmation against P91 constant-blend champion; no H3 evaluation."
        ),
        "prototype_blocks": list(BLOCK_NAMES),
        "prototype_signals_per_block": ["cosine", "negative_mean_squared_distance"],
        "statistics_pca_dim": 48,
        "experts": list(SELECTED_EXPERTS),
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
        prototype_signals=h2_prototypes,
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
