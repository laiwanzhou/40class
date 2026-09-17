"""Source-selected subject-domain normalization probe for the P91 teacher.

The probe never uses H3 labels for model or hyper-parameter selection.  It
learns the feature basis on H1+embargo, selects the feature set, normalization,
ridge penalty and conservative blend on H2, then evaluates that single frozen
configuration on H3.  Per-subject statistics are label-free and can therefore
also be computed for an unlabeled deployment cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from p91_hierarchical_multimodal_teacher import (
    audit,
    blend_prediction,
    build_data,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p91_subject_domain_normalization_h3_v1"
CHAMPION = PROJECT / "runs/p91_hierarchical_multimodal_h3_v3"
STREAM_SETS = {
    "ir": ("vmae", "iv2"),
    "visual": ("vmae", "iv2", "depth", "thermal", "hand"),
    "all": ("vmae", "iv2", "depth", "thermal", "hand", "motionbert", "hdgcn"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def align(ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {str(sample_id): row for row, sample_id in enumerate(ids)}
    return np.asarray([values[lookup[str(sample_id)]] for sample_id in target_ids])


def champion_predictions(data: Any, h2: np.ndarray, h3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner = load_npz(CHAMPION / "inner_predictions.npz")
    target = load_npz(CHAMPION / "predictions.npz")
    h2_logits = align(inner["sample_ids"].astype(str), inner["direct_logits"], data.sample_ids[h2])
    h2_teacher = align(
        inner["sample_ids"].astype(str), inner["teacher_prediction"], data.sample_ids[h2]
    )
    weight = float(np.asarray(inner["selected_constant_weight"]).reshape(-1)[0])
    h2_prediction = blend_prediction(h2_logits, h2_teacher, weight)
    h3_prediction = align(
        target["sample_ids"].astype(str), target["blended_prediction"], data.sample_ids[h3]
    )
    return h2_prediction.astype(np.int64), h3_prediction.astype(np.int64)


def fit_basis(data: Any, train: np.ndarray, pca_dim: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset, (name, stream) in enumerate(data.streams.items()):
        pooled = stream.mean(axis=1).astype(np.float64)
        scaler = StandardScaler().fit(pooled[train])
        train_scaled = scaler.transform(pooled[train])
        dimensions = min(pca_dim, len(train) - 1, pooled.shape[1])
        pca = PCA(
            n_components=dimensions,
            svd_solver="randomized",
            random_state=seed + offset,
        ).fit(train_scaled)
        output[name] = {
            "values": pca.transform(scaler.transform(pooled)).astype(np.float32),
            "variance": float(pca.explained_variance_ratio_.sum()),
        }
    return output


def subject_adjust(
    values: np.ndarray,
    users: np.ndarray,
    reference_indices: np.ndarray,
    alpha_mean: float,
    alpha_scale: float,
) -> np.ndarray:
    """Shrink each subject's moments toward source-global moments without labels."""
    result = values.astype(np.float64, copy=True)
    source_mean = values[reference_indices].mean(axis=0, keepdims=True)
    source_std = np.maximum(values[reference_indices].std(axis=0, keepdims=True), 0.20)
    for user in np.unique(users):
        rows = np.flatnonzero(users == user)
        local = values[rows].astype(np.float64)
        local_mean = local.mean(axis=0, keepdims=True)
        local_std = np.maximum(local.std(axis=0, keepdims=True), 0.20)
        centered = local - alpha_mean * (local_mean - source_mean)
        scale = np.power(source_std / local_std, alpha_scale)
        result[rows] = source_mean + (centered - source_mean) * scale
    return result.astype(np.float32)


def feature_matrix(
    basis: dict[str, Any],
    names: tuple[str, ...],
    users: np.ndarray,
    reference_indices: np.ndarray,
    alpha_mean: float,
    alpha_scale: float,
) -> np.ndarray:
    blocks = []
    for name in names:
        values = basis[name]["values"]
        if alpha_mean or alpha_scale:
            values = subject_adjust(
                values, users, reference_indices, alpha_mean, alpha_scale
            )
        blocks.append(values)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def user_stability(
    labels: np.ndarray,
    base: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
) -> dict[str, int]:
    nets = []
    for user in np.unique(users):
        selected = users == user
        nets.append(
            int(np.sum(prediction[selected] == labels[selected]))
            - int(np.sum(base[selected] == labels[selected]))
        )
    return {
        "negative_users": int(np.sum(np.asarray(nets) < 0)),
        "worst_user_net": int(min(nets, default=0)),
    }


def conservative_blend(
    logits: np.ndarray,
    base: np.ndarray,
    weight: float,
) -> np.ndarray:
    neural = softmax(logits, axis=1)
    base_probability = np.full((len(base), 40), 0.04 / 39.0, dtype=np.float64)
    base_probability[np.arange(len(base)), base] = 0.96
    score = weight * np.log(np.clip(neural, 1e-9, 1.0))
    score += (1.0 - weight) * np.log(np.clip(base_probability, 1e-9, 1.0))
    return score.argmax(axis=1)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = build_data()
    h1 = data.boundaries["H1_selection"]
    h2 = data.boundaries["H2_confirmation"]
    embargo = data.boundaries["E0_p87_sequence_source"]
    h3 = data.boundaries["H3_independent_fold0"]
    inner_train = np.concatenate((h1, embargo))
    final_train = np.concatenate((h1, h2, embargo))
    h2_base, h3_base = champion_predictions(data, h2, h3)

    print("fitting source-only PCA bases", flush=True)
    inner_basis = fit_basis(data, inner_train, args.pca_dim, args.seed)
    normalization_grid = (
        ("global", 0.0, 0.0),
        ("center25", 0.25, 0.0),
        ("center50", 0.50, 0.0),
        ("center75", 0.75, 0.0),
        ("center100", 1.00, 0.0),
        ("center50_scale25", 0.50, 0.25),
        ("center75_scale25", 0.75, 0.25),
        ("center100_scale25", 1.00, 0.25),
        ("center75_scale50", 0.75, 0.50),
        ("center100_scale50", 1.00, 0.50),
    )
    grid: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str, float], tuple[StandardScaler, RidgeClassifier]] = {}
    for set_name, names in STREAM_SETS.items():
        for normalization, alpha_mean, alpha_scale in normalization_grid:
            features = feature_matrix(
                inner_basis,
                names,
                data.users,
                inner_train,
                alpha_mean,
                alpha_scale,
            )
            scaler = StandardScaler().fit(features[inner_train])
            train_x = scaler.transform(features[inner_train])
            valid_x = scaler.transform(features[h2])
            for ridge_alpha in (1.0, 10.0, 100.0, 1000.0):
                model = RidgeClassifier(alpha=ridge_alpha, solver="lsqr").fit(
                    train_x, data.labels[inner_train]
                )
                logits = model.decision_function(valid_x)
                for blend_weight in np.arange(0.10, 0.651, 0.025):
                    prediction = conservative_blend(logits, h2_base, float(blend_weight))
                    row = {
                        "stream_set": set_name,
                        "normalization": normalization,
                        "alpha_mean": alpha_mean,
                        "alpha_scale": alpha_scale,
                        "ridge_alpha": ridge_alpha,
                        "blend_weight": float(blend_weight),
                        **audit(data.labels[h2], h2_base, prediction),
                        **user_stability(data.labels[h2], h2_base, prediction, data.users[h2]),
                    }
                    grid.append(row)
                fitted[(set_name, normalization, ridge_alpha)] = (scaler, model)
            best_here = max(
                (row for row in grid if row["stream_set"] == set_name and row["normalization"] == normalization),
                key=lambda row: (row["correct"], -row["harm"], row["worst_user_net"]),
            )
            print(
                f"{set_name:6s} {normalization:20s} "
                f"acc={best_here['accuracy']:.6f} net={best_here['net']:+d} "
                f"rescue={best_here['rescue']} harm={best_here['harm']}",
                flush=True,
            )

    eligible = [
        row for row in grid if row["negative_users"] <= 1 and row["worst_user_net"] >= -1
    ]
    selected = max(
        eligible or grid,
        key=lambda row: (
            row["correct"],
            -row["harm"],
            -row["negative_users"],
            row["worst_user_net"],
            -row["blend_weight"],
        ),
    )
    print(f"selected on H2: {selected}", flush=True)

    # Refit the exact source-selected configuration on all source cohorts.
    final_basis = fit_basis(data, final_train, args.pca_dim, args.seed + 1000)
    names = STREAM_SETS[str(selected["stream_set"])]
    final_features = feature_matrix(
        final_basis,
        names,
        data.users,
        final_train,
        float(selected["alpha_mean"]),
        float(selected["alpha_scale"]),
    )
    final_scaler = StandardScaler().fit(final_features[final_train])
    final_model = RidgeClassifier(
        alpha=float(selected["ridge_alpha"]), solver="lsqr"
    ).fit(final_scaler.transform(final_features[final_train]), data.labels[final_train])
    h3_logits = final_model.decision_function(final_scaler.transform(final_features[h3]))
    h3_prediction = conservative_blend(
        h3_logits, h3_base, float(selected["blend_weight"])
    )
    h3_audit = audit(data.labels[h3], h3_base, h3_prediction)
    h3_audit.update(user_stability(data.labels[h3], h3_base, h3_prediction, data.users[h3]))
    report = {
        "protocol": (
            "PCA/classifier fit on H1+embargo and all choices selected on H2; exact choice "
            "refit on H1+H2+embargo; H3 labels used only for the final frozen audit."
        ),
        "h2_base": audit(data.labels[h2], data.teacher_prediction[h2], h2_base),
        "selected": selected,
        "h3": h3_audit,
        "pca_variance": {
            name: float(final_basis[name]["variance"]) for name in final_basis
        },
        "top_h2": sorted(
            grid,
            key=lambda row: (
                row["correct"], -row["harm"], -row["negative_users"], row["worst_user_net"]
            ),
            reverse=True,
        )[:30],
    }
    np.savez_compressed(
        output / "predictions.npz",
        sample_ids=data.sample_ids[h3],
        labels=data.labels[h3],
        base_prediction=h3_base,
        ridge_logits=h3_logits.astype(np.float32),
        prediction=h3_prediction,
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
