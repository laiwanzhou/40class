from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_CACHE = PROJECT_DIR / "runs/p86_motion_window_cache_t16_v1"
TEST_CACHE = PROJECT_DIR / "runs/p87s_test_motion_window_t16_v1"
OUTPUT = PROJECT_DIR / "runs/p89_skeleton_invariant_expert_v1"
SEED = 20260816


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def masked_fill(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, values, np.nan).astype(np.float64)


def robust_statistics(values: np.ndarray) -> np.ndarray:
    """Statistics along time for finite-filled [sample,time,signal] values."""

    valid = np.isfinite(values)
    count = np.maximum(valid.sum(axis=1), 1)
    safe = np.where(valid, values, 0.0)
    mean = safe.sum(axis=1) / count
    centered = np.where(valid, values - mean[:, None], 0.0)
    std = np.sqrt(np.sum(centered**2, axis=1) / count)
    minimum = np.min(np.where(valid, values, np.inf), axis=1)
    maximum = np.max(np.where(valid, values, -np.inf), axis=1)
    minimum[~np.isfinite(minimum)] = 0.0
    maximum[~np.isfinite(maximum)] = 0.0
    quantile_source = np.where(valid, values, np.nan)
    with np.errstate(all="ignore"):
        q10, q25, q50, q75, q90 = np.nanquantile(
            quantile_source, (0.10, 0.25, 0.50, 0.75, 0.90), axis=1
        )

    pair_valid = valid[:, 1:] & valid[:, :-1]
    difference = np.where(pair_valid, values[:, 1:] - values[:, :-1], 0.0)
    pair_count = np.maximum(pair_valid.sum(axis=1), 1)
    mean_abs_difference = np.sum(np.abs(difference), axis=1) / pair_count
    rms_difference = np.sqrt(np.sum(difference**2, axis=1) / pair_count)
    first = np.argmax(valid, axis=1)
    last = values.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    batch = np.arange(len(values))[:, None]
    signal = np.arange(values.shape[2])[None, :]
    endpoint_delta = safe[batch, last, signal] - safe[batch, first, signal]
    endpoint_delta[~valid.any(axis=1)] = 0.0

    blocks = (
        mean,
        std,
        np.sqrt(np.sum(safe**2, axis=1) / count),
        minimum,
        maximum,
        maximum - minimum,
        q10,
        q25,
        q50,
        q75,
        q90,
        q75 - q25,
        mean_abs_difference,
        rms_difference,
        endpoint_delta,
    )
    return np.nan_to_num(
        np.stack(blocks, axis=-1).reshape(len(values), -1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def phase_statistics(values: np.ndarray, bins: int = 8) -> np.ndarray:
    blocks = []
    for indices in np.array_split(np.arange(values.shape[1]), bins):
        part = values[:, indices]
        with np.errstate(all="ignore"):
            blocks.extend((np.nanmean(part, axis=1), np.nanstd(part, axis=1)))
    return np.nan_to_num(
        np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def bilateral_signals(norms: np.ndarray) -> np.ndarray:
    # H36M: legs (1,2,3)/(4,5,6), arms (14,15,16)/(11,12,13).
    pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
    output = []
    for right, left in pairs:
        output.extend(
            (
                0.5 * (norms[:, :, right] + norms[:, :, left]),
                np.abs(norms[:, :, right] - norms[:, :, left]),
            )
        )
    return np.stack(output, axis=2)


def extract(cache: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = read_rows(cache / "rows.csv")
    count = len(rows)
    skeleton = np.asarray(
        np.load(cache / "skeleton_features.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(count, 32, 17, 13)
    joint_mask = np.asarray(
        np.load(cache / "skeleton_joint_mask.npy", mmap_mode="r"), dtype=bool
    ).reshape(count, 32, 17)
    relations = np.asarray(
        np.load(cache / "skeleton_relations.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(count, 32, 18)
    relation_mask = np.asarray(
        np.load(cache / "skeleton_relation_mask.npy", mmap_mode="r"), dtype=bool
    ).reshape(count, 32, 18)
    quality = np.asarray(
        np.load(cache / "skeleton_frame_quality.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(count, 32, 1)

    vector = skeleton[..., :12].reshape(count, 32, -1)
    vector_mask = np.repeat(joint_mask[..., None], 12, axis=3).reshape(count, 32, -1)
    vector = masked_fill(vector, vector_mask)
    stream = skeleton[..., :12].reshape(count, 32, 17, 4, 3)
    norms = np.linalg.norm(stream, axis=-1)
    norm_mask = np.repeat(joint_mask[..., None], 4, axis=3)
    norms = masked_fill(norms.reshape(count, 32, -1), norm_mask.reshape(count, 32, -1))
    bone_norm = np.linalg.norm(skeleton[..., 3:6], axis=-1)
    bilateral = masked_fill(
        bilateral_signals(bone_norm),
        np.ones((count, 32, 12), dtype=bool),
    )
    invariant = np.concatenate(
        (
            norms,
            masked_fill(relations, relation_mask),
            bilateral,
            quality.astype(np.float64),
        ),
        axis=2,
    )

    features = [
        robust_statistics(vector),
        robust_statistics(invariant),
        phase_statistics(invariant, 8),
    ]
    # Preserve early/late asymmetry without doubling the large vector block.
    for window in (slice(0, 16), slice(16, 32)):
        features.append(robust_statistics(invariant[:, window]))
    coverage = np.concatenate(
        (
            joint_mask.mean(axis=(1, 2), keepdims=False)[:, None],
            relation_mask.mean(axis=(1, 2), keepdims=False)[:, None],
            quality.mean(axis=1),
            quality.std(axis=1),
        ),
        axis=1,
    )
    features.append(coverage.astype(np.float32))
    result = np.nan_to_num(
        np.concatenate(features, axis=1), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    return rows, result


def make_model(name: str):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_depth=28,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=40,
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=6,
            min_child_samples=18,
            subsample=0.85,
            colsample_bytree=0.30,
            max_bin=63,
            reg_alpha=1.0,
            reg_lambda=6.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(name)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_feature_path = OUTPUT / "train_features.npy"
    test_feature_path = OUTPUT / "test_features.npy"
    train_rows = read_rows(TRAIN_CACHE / "rows.csv")
    test_rows = read_rows(TEST_CACHE / "rows.csv")
    if train_feature_path.is_file() and test_feature_path.is_file():
        train_features = np.load(train_feature_path, mmap_mode="r")
        test_features = np.load(test_feature_path, mmap_mode="r")
    else:
        train_rows, train_features = extract(TRAIN_CACHE)
        test_rows, test_features = extract(TEST_CACHE)
        np.save(train_feature_path, train_features)
        np.save(test_feature_path, test_features)
    if len(train_rows) != len(train_features) or len(test_rows) != len(test_features):
        raise RuntimeError("skeleton feature row count changed")

    labels = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in train_rows]).astype(str)
    folds = json.loads(
        (PROJECT_DIR / "data/subject_folds/folds_summary.json").read_text(encoding="utf-8")
    )["folds"]
    results = {}
    oof_by_model = {}
    for name in ("extra_trees", "lightgbm"):
        probability = np.zeros((len(train_rows), 40), dtype=np.float64)
        per_fold = []
        for fold in folds:
            fit = np.isin(users, fold["train_users"])
            held = np.isin(users, fold["val_users"])
            estimator = make_model(name)
            estimator.fit(train_features[fit], labels[fit])
            held_probability = estimator.predict_proba(train_features[held])
            target = np.flatnonzero(held)
            probability[np.ix_(target, estimator.classes_.astype(np.int64))] = held_probability
            item = {
                "fold": int(fold["fold"]),
                **metrics(labels[held], probability[target].argmax(axis=1)),
            }
            per_fold.append(item)
            print(name, item, flush=True)
        results[name] = {
            "overall": metrics(labels, probability.argmax(axis=1)),
            "folds": per_fold,
        }
        oof_by_model[name] = probability
        print(name, results[name]["overall"], flush=True)

    selected = max(
        results,
        key=lambda name: (
            results[name]["overall"]["accuracy"],
            results[name]["overall"]["balanced_accuracy"],
        ),
    )
    final = make_model(selected)
    final.fit(train_features, labels)
    test_partial = final.predict_proba(test_features)
    test_probability = np.zeros((len(test_rows), 40), dtype=np.float64)
    test_probability[:, final.classes_.astype(np.int64)] = test_partial
    np.savez_compressed(
        OUTPUT / "oof_logits.npz",
        sample_ids=np.asarray([row["sample_id"] for row in train_rows]),
        labels=labels,
        skeleton_logits=np.log(np.maximum(oof_by_model[selected], 1e-12)),
    )
    np.savez_compressed(
        OUTPUT / "test_logits.npz",
        sample_ids=np.asarray([row["sample_id"] for row in test_rows]),
        skeleton_logits=np.log(np.maximum(test_probability, 1e-12)),
    )
    report = {
        "stage": "P89_body_invariant_multistream_skeleton_expert_v1",
        "protocol": (
            "Fixed subject-disjoint folds. Joint, bone, joint-motion and bone-motion "
            "vectors are combined with norm, bilateral, relation, phase and early/late "
            "statistics. Test labels and leaderboard feedback are not used."
        ),
        "feature_dimension": int(train_features.shape[1]),
        "selected_model": selected,
        "models": results,
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
