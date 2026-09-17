"""P91 sensor residual router on top of the strong P90 visual-first teacher.

Unlike P90's safe-to-visual router, this stage treats the P90 routed prediction
as the primary visual decision and asks a narrower question: can a skeleton or
IMU candidate correct that decision?  Scores are generated leave-one-user-out
on H1+H2; the route threshold is frozen before the independent H3 evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p90_crossuser_visual_router import build_features, load_splits
from p90_teacher_fusion_audit import align


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p91_sensor_residual_router_h3_v1"
EPSILON = 1e-8


@dataclass
class Split:
    name: str
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    base_prediction: np.ndarray
    candidate_names: list[str]
    candidate_prediction: np.ndarray
    features: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pca-dim", type=int, default=48)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def margin(probability: np.ndarray) -> np.ndarray:
    top = np.partition(probability, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, EPSILON, 1.0)
    return -np.sum(values * np.log(values), axis=1) / np.log(values.shape[1])


def one_hot(values: np.ndarray, classes: int = 40) -> np.ndarray:
    output = np.zeros((len(values), classes), dtype=np.float32)
    output[np.arange(len(values)), values.astype(np.int64)] = 1.0
    return output


def fit_reduce(
    source: np.ndarray, targets: list[np.ndarray], dimensions: int, seed: int
) -> list[np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(source.astype(np.float64))
    components = min(dimensions, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=seed,
    )
    pca.fit(scaled)
    return [
        pca.transform(scaler.transform(values.astype(np.float64))).astype(np.float32)
        for values in targets
    ]


def build_splits(args: argparse.Namespace) -> dict[str, Split]:
    raw = load_splits()
    names = ["H1_selection", "H2_confirmation", "H3_independent_fold0"]
    router = load_npz(PROJECT / "runs/p90_crossuser_visual_router_v1/full_predictions.npz")
    skeleton = load_npz(HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz")
    motionbert = load_npz(
        PROJECT / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz"
    )
    imu = load_npz(
        PROJECT / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz"
    )
    cross = load_npz(HERE / "runs/p89_crossmodal_statistics_v1/crossmodal_statistics.npz")
    vmae = load_npz(PROJECT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz")
    iv2 = load_npz(PROJECT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz")
    mb_features = load_npz(
        PROJECT / "runs/p90_motionbert_teacher_v1/features_pretrain_front_t81.npz"
    )

    candidate_names = ["skeleton_invariant", "motionbert_front", "imu_sensorwise_deep"]
    sensor_sources = [
        (skeleton, "skeleton_logits", True),
        (motionbert, "probabilities", False),
        (imu, "probabilities", False),
    ]
    base_features: dict[str, list[np.ndarray]] = {name: [] for name in names}
    candidates: dict[str, list[np.ndarray]] = {name: [] for name in names}
    raw_blocks: dict[str, dict[str, np.ndarray]] = {name: {} for name in names}
    for name in names:
        split = raw[name]
        ids = split.sample_ids.astype(str)
        visual_features, _ = build_features(split)
        base = router[f"{name}_router_prediction"].astype(np.int64)
        route_score = router[f"{name}_route_score"].astype(np.float32)
        sensor_probability: list[np.ndarray] = []
        for source, key, logits in sensor_sources:
            values = align(source["sample_ids"].astype(str), source[key], ids)
            probability = softmax(values, axis=1) if logits else values
            probability = np.clip(probability, EPSILON, None)
            probability /= probability.sum(axis=1, keepdims=True)
            sensor_probability.append(probability.astype(np.float32))
        sensor_prediction = np.stack([value.argmax(1) for value in sensor_probability], axis=1)
        candidates[name] = [value.argmax(1).astype(np.int64) for value in sensor_probability]
        matrices = [
            visual_features.astype(np.float32),
            one_hot(base),
            route_score[:, None],
        ]
        for sensor_name, probability, prediction in zip(
            candidate_names, sensor_probability, sensor_prediction.T
        ):
            matrices.extend(
                [
                    np.log(np.clip(probability, EPSILON, 1.0)).astype(np.float32),
                    one_hot(prediction),
                    np.column_stack(
                        (
                            probability.max(1),
                            margin(probability),
                            entropy(probability),
                            prediction != base,
                            probability[np.arange(len(base)), prediction],
                            probability[np.arange(len(base)), base],
                        )
                    ).astype(np.float32),
                ]
            )
        base_features[name] = matrices
        statistics = align(cross["sample_ids"].astype(str), cross["features"], ids).astype(
            np.float32
        )
        raw_blocks[name] = {
            "skeleton": np.concatenate((statistics[:, :2629], statistics[:, 5729:5740]), axis=1),
            "imu": np.concatenate((statistics[:, 2629:5729], statistics[:, 5740:5795]), axis=1),
            "relation": statistics[:, 5795:6195],
            "vmae": align(vmae["sample_ids"].astype(str), vmae["features"], ids).reshape(
                len(ids), -1
            ),
            "iv2": align(iv2["sample_ids"].astype(str), iv2["features"], ids).reshape(
                len(ids), -1
            ),
            "motionbert": align(
                mb_features["sample_ids"].astype(str), mb_features["features"], ids
            ),
        }

    for block_index, block in enumerate(
        ("skeleton", "imu", "relation", "vmae", "iv2", "motionbert")
    ):
        source_values = np.concatenate(
            [raw_blocks["H1_selection"][block], raw_blocks["H2_confirmation"][block]]
        )
        reduced = fit_reduce(
            source_values,
            [raw_blocks[name][block] for name in names],
            args.pca_dim,
            args.seed + block_index,
        )
        for name, values in zip(names, reduced):
            base_features[name].append(values)

    output: dict[str, Split] = {}
    for name in names:
        split = raw[name]
        output[name] = Split(
            name=name,
            sample_ids=split.sample_ids.astype(str),
            labels=split.labels.astype(np.int64),
            users=split.users.astype(str),
            base_prediction=router[f"{name}_router_prediction"].astype(np.int64),
            candidate_names=candidate_names,
            candidate_prediction=np.stack(candidates[name], axis=1),
            features=np.concatenate(base_features[name], axis=1).astype(np.float32),
        )
    return output


def concatenate(a: Split, b: Split) -> Split:
    return Split(
        name="H1_H2_source",
        sample_ids=np.concatenate((a.sample_ids, b.sample_ids)),
        labels=np.concatenate((a.labels, b.labels)),
        users=np.concatenate((a.users, b.users)),
        base_prediction=np.concatenate((a.base_prediction, b.base_prediction)),
        candidate_names=a.candidate_names,
        candidate_prediction=np.concatenate((a.candidate_prediction, b.candidate_prediction)),
        features=np.concatenate((a.features, b.features)),
    )


def models(seed: int, trees: int) -> list[Any]:
    return [
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.08, max_iter=1000, solver="liblinear", random_state=seed),
        ),
        ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=5,
            max_features=0.35,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=3.0,
            random_state=seed,
        ),
    ]


def fit_candidate(
    x: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    target_x: np.ndarray,
    seed: int,
    trees: int,
) -> np.ndarray:
    changed = candidate != base
    train = train & changed
    rescue = (base != labels) & (candidate == labels)
    harm = (base == labels) & (candidate != labels)
    y = rescue.astype(np.int64)
    weight = np.where(rescue, 9.0, np.where(harm, 2.0, 0.45)).astype(np.float64)
    if len(np.unique(y[train])) < 2:
        return np.zeros(len(target_x), dtype=np.float64)
    probabilities = []
    for model in models(seed, trees):
        model.fit(x[train], y[train], **({"sample_weight": weight[train]} if not hasattr(model, "steps") else {"logisticregression__sample_weight": weight[train]}))
        probabilities.append(model.predict_proba(target_x)[:, 1])
    return np.mean(probabilities, axis=0)


def oof_scores(source: Split, args: argparse.Namespace) -> np.ndarray:
    scores = np.zeros((len(source.labels), len(source.candidate_names)), dtype=np.float64)
    for user_index, user in enumerate(sorted(np.unique(source.users).tolist())):
        train = source.users != user
        target = source.users == user
        for candidate_index in range(len(source.candidate_names)):
            scores[target, candidate_index] = fit_candidate(
                source.features,
                source.base_prediction,
                source.candidate_prediction[:, candidate_index],
                source.labels,
                train,
                source.features[target],
                args.seed + 101 * user_index + candidate_index,
                args.trees,
            )
    return scores


def apply_route(split: Split, scores: np.ndarray, threshold: float) -> np.ndarray:
    best = scores.argmax(axis=1)
    selected = scores[np.arange(len(scores)), best] >= threshold
    candidate = split.candidate_prediction[np.arange(len(scores)), best]
    selected &= candidate != split.base_prediction
    output = split.base_prediction.copy()
    output[selected] = candidate[selected]
    return output


def route_audit(split: Split, prediction: np.ndarray) -> dict[str, Any]:
    base_correct = split.base_prediction == split.labels
    candidate_correct = prediction == split.labels
    per_user = {}
    for user in np.unique(split.users):
        selected = split.users == user
        per_user[user] = {
            "base_correct": int(base_correct[selected].sum()),
            "candidate_correct": int(candidate_correct[selected].sum()),
            "net": int(candidate_correct[selected].sum() - base_correct[selected].sum()),
            "changed": int(np.sum(prediction[selected] != split.base_prediction[selected])),
        }
    return {
        "accuracy": float(candidate_correct.mean()),
        "correct": int(candidate_correct.sum()),
        "base_correct": int(base_correct.sum()),
        "net": int(candidate_correct.sum() - base_correct.sum()),
        "rescue": int(np.sum(~base_correct & candidate_correct)),
        "harm": int(np.sum(base_correct & ~candidate_correct)),
        "changed": int(np.sum(prediction != split.base_prediction)),
        "per_user": per_user,
    }


def select_threshold(source: Split, scores: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    values = np.unique(np.concatenate((np.linspace(0.05, 0.95, 91), scores.ravel())))
    grid = []
    for threshold in values:
        prediction = apply_route(source, scores, float(threshold))
        audit = route_audit(source, prediction)
        negative_users = [row["net"] for row in audit["per_user"].values() if row["net"] < 0]
        audit["threshold"] = float(threshold)
        audit["negative_users"] = len(negative_users)
        audit["worst_user_net"] = min(
            [row["net"] for row in audit["per_user"].values()], default=0
        )
        grid.append(audit)
    eligible = [
        row
        for row in grid
        if row["net"] >= 3 and row["negative_users"] <= 2 and row["worst_user_net"] >= -1
    ]
    if not eligible:
        return 1.01, grid
    selected = max(
        eligible,
        key=lambda row: (row["net"], -row["negative_users"], row["worst_user_net"], -row["changed"]),
    )
    return float(selected["threshold"]), grid


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = build_splits(args)
    source = concatenate(splits["H1_selection"], splits["H2_confirmation"])
    target = splits["H3_independent_fold0"]
    scores = oof_scores(source, args)
    threshold, threshold_grid = select_threshold(source, scores)
    source_prediction = apply_route(source, scores, threshold)

    target_scores = np.zeros(
        (len(target.labels), len(source.candidate_names)), dtype=np.float64
    )
    all_source = np.ones(len(source.labels), dtype=bool)
    for candidate_index in range(len(source.candidate_names)):
        target_scores[:, candidate_index] = fit_candidate(
            source.features,
            source.base_prediction,
            source.candidate_prediction[:, candidate_index],
            source.labels,
            all_source,
            target.features,
            args.seed + 10000 + candidate_index,
            args.trees,
        )
    target_prediction = apply_route(target, target_scores, threshold)
    source_audit = route_audit(source, source_prediction)
    target_audit = route_audit(target, target_prediction)
    summary = {
        "stage": "P91_sensor_residual_router_H3_v1",
        "protocol": {
            "base": "P90 cross-user visual router",
            "source": "H1+H2 leave-one-user-out sensor rescue scores",
            "target": "H3 independent fold0, touched after threshold freeze",
            "features": "P90 visual-router features + sensor probabilities + source-only PCA teacher/statistics blocks",
        },
        "feature_dim": int(source.features.shape[1]),
        "candidate_names": source.candidate_names,
        "selected_threshold": threshold,
        "source": source_audit,
        "target": target_audit,
        "threshold_grid": threshold_grid,
    }
    np.savez_compressed(
        output / "predictions.npz",
        sample_ids=target.sample_ids,
        labels=target.labels,
        users=target.users,
        base_prediction=target.base_prediction,
        candidate_names=np.asarray(target.candidate_names),
        candidate_prediction=target.candidate_prediction,
        score=target_scores.astype(np.float32),
        prediction=target_prediction,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"threshold": threshold, "source": source_audit, "target": target_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
