from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p27r2_event_data import (
    EVENT_NAMES,
    P27R2EventExtractor,
    load_event_cache,
    read_csv,
    save_event_cache,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "p27_r2_fold0.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_r2_event_audit"
DEFAULT_CACHE = DEFAULT_OUTPUT / "event_cache_v2.npz"
DEFAULT_ANCHORS = PROJECT_DIR / "data" / "p27_r2_outer_train_event_anchors.csv"

PAIR_IDS = (
    (19, 24, "Phone call / Use phone"),
    (19, 25, "Phone call / Watch TV"),
    (24, 26, "Use phone / Play games"),
    (25, 26, "Watch TV / Play games"),
    (37, 7, "Take medicine / Eat food"),
    (37, 6, "Take medicine / Drink water"),
    (9, 10, "Pour / Stir"),
    (10, 11, "Stir / Peel"),
    (21, 22, "Read / Turn pages"),
    (18, 22, "Write / Turn pages"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit P27-R2 event labels")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_eta_squared(values: np.ndarray, subjects: np.ndarray) -> float:
    if len(values) < 2:
        return 1.0
    mean = float(values.mean())
    total = float(np.sum((values - mean) ** 2))
    if total <= 1e-12:
        return 1.0
    between = 0.0
    for subject in np.unique(subjects):
        selected = subjects == subject
        between += int(selected.sum()) * float((values[selected].mean() - mean) ** 2)
    return float(np.clip(between / total, 0.0, 1.0))


def source_features(cache, source: str) -> np.ndarray:
    if source == "skeleton":
        token = cache.skeleton_tokens
    elif source == "imu":
        token = cache.imu_tokens[:, :, :25]
    elif source == "visual":
        token = cache.visual_tokens[:, :, -10:]
    elif source == "visual+imu":
        token = np.concatenate(
            [cache.visual_tokens[:, :, -10:], cache.imu_tokens[:, :, :25]], axis=2
        )
    else:
        raise ValueError(source)
    return np.concatenate(
        [
            token.reshape(len(token), -1),
            token.mean(axis=1),
            token.std(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def inner_prediction_audit(
    features: np.ndarray,
    target: np.ndarray,
    quality: np.ndarray,
    subjects: np.ndarray,
    inner_splits: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], float, int, float, int]:
    rows: list[dict[str, Any]] = []
    ridge_improvements: list[float] = []
    tree_improvements: list[float] = []
    for fold_name, held_subjects in inner_splits.items():
        train = (~np.isin(subjects, held_subjects)) & (quality > 0)
        held = np.isin(subjects, held_subjects) & (quality > 0)
        if train.sum() < 32 or held.sum() < 16:
            rows.append(
                {
                    "inner_fold": int(fold_name),
                    "train": int(train.sum()),
                    "held": int(held.sum()),
                    "baseline_mae": None,
                    "ridge_mae": None,
                    "ridge_improvement": None,
                    "tree_mae": None,
                    "tree_improvement": None,
                }
            )
            continue
        baseline = float(
            np.average(target[train], weights=np.maximum(quality[train], 1e-3))
        )
        baseline_mae = float(np.mean(np.abs(target[held] - baseline)))
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(features[train], target[train], ridge__sample_weight=quality[train])
        prediction = np.clip(model.predict(features[held]), 0.0, 1.0)
        ridge_mae = float(np.mean(np.abs(target[held] - prediction)))
        ridge_improvement = float(
            (baseline_mae - ridge_mae) / max(baseline_mae, 1e-8)
        )
        tree = ExtraTreesRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=27032 + int(fold_name),
        )
        tree.fit(features[train], target[train], sample_weight=quality[train])
        tree_prediction = np.clip(tree.predict(features[held]), 0.0, 1.0)
        tree_mae = float(np.mean(np.abs(target[held] - tree_prediction)))
        tree_improvement = float(
            (baseline_mae - tree_mae) / max(baseline_mae, 1e-8)
        )
        ridge_improvements.append(ridge_improvement)
        tree_improvements.append(tree_improvement)
        rows.append(
            {
                "inner_fold": int(fold_name),
                "train": int(train.sum()),
                "held": int(held.sum()),
                "baseline_mae": baseline_mae,
                "ridge_mae": ridge_mae,
                "ridge_improvement": ridge_improvement,
                "tree_mae": tree_mae,
                "tree_improvement": tree_improvement,
            }
        )
    return (
        rows,
        float(np.mean(ridge_improvements)) if ridge_improvements else -1.0,
        int(sum(value > 0 for value in ridge_improvements)),
        float(np.mean(tree_improvements)) if tree_improvements else -1.0,
        int(sum(value > 0 for value in tree_improvements)),
    )


def pair_audit(
    labels: np.ndarray,
    subjects: np.ndarray,
    target: np.ndarray,
    quality: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for left, right, name in PAIR_IDS:
        left_mask = (labels == left) & (quality > 0)
        right_mask = (labels == right) & (quality > 0)
        if left_mask.sum() < 3 or right_mask.sum() < 3:
            continue
        pooled = np.concatenate([target[left_mask], target[right_mask]])
        scale = float(pooled.std())
        effect = float(
            (target[left_mask].mean() - target[right_mask].mean()) / max(scale, 1e-6)
        )
        directions = []
        for subject in np.unique(subjects):
            left_subject = left_mask & (subjects == subject)
            right_subject = right_mask & (subjects == subject)
            if left_subject.sum() and right_subject.sum():
                direction = np.sign(
                    target[left_subject].mean() - target[right_subject].mean()
                )
                if direction:
                    directions.append(float(direction))
        agreement = (
            float(max(np.mean(np.asarray(directions) > 0), np.mean(np.asarray(directions) < 0)))
            if directions
            else 0.0
        )
        rows.append(
            {
                "pair": name,
                "left_class": left,
                "right_class": right,
                "left_mean": float(target[left_mask].mean()),
                "right_mean": float(target[right_mask].mean()),
                "standardized_effect": effect,
                "subjects_compared": len(directions),
                "direction_agreement": agreement,
            }
        )
    return rows


def anchor_audit(
    cache,
    outer_train: np.ndarray,
    target_index: int,
    anchors: list[dict[str, str]],
) -> tuple[int, int, float | None, list[dict[str, Any]]]:
    name = str(cache.event_names[target_index])
    lookup = {
        sample_id: index for index, sample_id in enumerate(cache.sample_ids.astype(str))
    }
    values = cache.event_targets[outer_train, target_index]
    quality = cache.event_quality[outer_train, target_index]
    valid = quality > 0
    if not valid.any():
        return 0, 0, None, []
    low, high = np.quantile(values[valid], [0.40, 0.60])
    rows = []
    correct = 0
    current = [row for row in anchors if row["event_name"] == name]
    anchor_values = {
        row["sample_id"]: float(
            cache.event_targets[lookup[row["sample_id"]], target_index]
        )
        for row in current
        if row["sample_id"] in lookup and outer_train[lookup[row["sample_id"]]]
    }
    high_values = [
        anchor_values[row["sample_id"]]
        for row in current
        if row["direction"] == "high" and row["sample_id"] in anchor_values
    ]
    low_values = [
        anchor_values[row["sample_id"]]
        for row in current
        if row["direction"] == "low" and row["sample_id"] in anchor_values
    ]
    has_contrast = bool(high_values and low_values)
    contrast_cut = (
        0.5 * (min(high_values) + max(low_values)) if has_contrast else None
    )
    for anchor in anchors:
        if anchor["event_name"] != name:
            continue
        index = lookup.get(anchor["sample_id"])
        if index is None or not outer_train[index]:
            raise ValueError(f"anchor is not an outer-train sample: {anchor}")
        value = float(cache.event_targets[index, target_index])
        anchor_quality = float(cache.event_quality[index, target_index])
        if has_contrast:
            passed = bool(
                anchor_quality > 0
                and (
                    (anchor["direction"] == "high" and value >= contrast_cut)
                    or (anchor["direction"] == "low" and value <= contrast_cut)
                )
            )
        else:
            passed = bool(
                anchor_quality > 0
                and (
                    (anchor["direction"] == "high" and value >= high)
                    or (anchor["direction"] == "low" and value <= low)
                )
            )
        correct += int(passed)
        rows.append(
            {
                "sample_id": anchor["sample_id"],
                "event_name": name,
                "direction": anchor["direction"],
                "value": value,
                "quality": anchor_quality,
                "low_cut": float(low),
                "high_cut": float(high),
                "contrast_cut": contrast_cut,
                "aligned": int(passed),
                "reason": anchor["reason"],
            }
        )
    return len(rows), correct, correct / len(rows) if rows else None, rows


def plot_anchor_traces(cache, outer_train: np.ndarray, output: Path) -> None:
    anchors = read_csv(DEFAULT_ANCHORS)
    sample_ids = []
    for row in anchors:
        if row["sample_id"] not in sample_ids:
            sample_ids.append(row["sample_id"])
    lookup = {
        sample_id: index for index, sample_id in enumerate(cache.sample_ids.astype(str))
    }
    selected = [lookup[value] for value in sample_ids if value in lookup and outer_train[lookup[value]]]
    figure, axes = plt.subplots(len(selected), 3, figsize=(13, max(3, 2.1 * len(selected))))
    if len(selected) == 1:
        axes = axes[None]
    for row_axis, index in zip(axes, selected, strict=True):
        token = cache.skeleton_tokens[index]
        proximity = np.exp(-((np.stack([token[:, 27], token[:, 28]], axis=1) / 0.65) ** 2))
        row_axis[0].plot(proximity[:, 0], label="left")
        row_axis[0].plot(proximity[:, 1], label="right")
        row_axis[0].set_ylim(0, 1.05)
        row_axis[0].legend(fontsize=6)
        invariant = cache.imu_tokens[index, :, :25].reshape(32, 5, 5)
        row_axis[1].plot(np.log1p(invariant[:, 1, 1]), label="left wrist")
        row_axis[1].plot(np.log1p(invariant[:, 2, 1]), label="right wrist")
        row_axis[1].legend(fontsize=6)
        row_axis[2].plot(cache.visual_tokens[index, :, -9], label="Depth local")
        row_axis[2].plot(cache.visual_tokens[index, :, -5], label="IR local")
        row_axis[2].legend(fontsize=6)
        row_axis[0].set_ylabel(
            f"c{int(cache.labels[index]):02d} {cache.subjects[index]}", fontsize=7
        )
    for axis, title in zip(axes[0], ["hand-head proximity", "wrist gyro", "local motion"]):
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_target_subjects(cache, outer_train: np.ndarray, output: Path) -> None:
    figure, axes = plt.subplots(6, 3, figsize=(15, 18))
    subjects = sorted(np.unique(cache.subjects[outer_train]).tolist())
    for index, axis in enumerate(axes.flat):
        values = []
        labels = []
        for subject in subjects:
            selected = (
                outer_train
                & (cache.subjects == subject)
                & (cache.event_quality[:, index] > 0)
            )
            if selected.any():
                values.append(cache.event_targets[selected, index])
                labels.append(subject.replace("user", "u"))
        if values:
            axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.set_title(str(cache.event_names[index]), fontsize=9)
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.set_ylim(-0.05, 1.05)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache.resolve()
    started = time.time()
    if args.reuse_cache and cache_path.is_file():
        cache = load_event_cache(cache_path)
    else:
        cache = P27R2EventExtractor().extract()
        save_event_cache(cache_path, cache)
    if tuple(cache.event_names.astype(str).tolist()) != EVENT_NAMES:
        raise ValueError("event cache names do not match code")

    outer_train = cache.outer_folds != int(config["outer_fold"])
    outer_held = ~outer_train
    anchors = read_csv(DEFAULT_ANCHORS)
    gates = config["event_audit"]
    audit_rows = []
    fold_rows = []
    anchor_rows = []
    pair_rows = []
    passed_names = []

    for event_index, name in enumerate(cache.event_names.astype(str)):
        source = str(cache.event_sources[event_index])
        quality = cache.event_quality[:, event_index]
        valid = outer_train & (quality > 0)
        coverage = float(valid.sum() / max(outer_train.sum(), 1))
        values = cache.event_targets[valid, event_index]
        iqr = (
            float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
            if len(values)
            else 0.0
        )
        eta = (
            subject_eta_squared(values, cache.subjects[valid]) if len(values) else 1.0
        )
        (
            prediction_rows,
            ridge_improvement,
            positive_ridge_folds,
            tree_improvement,
            positive_tree_folds,
        ) = inner_prediction_audit(
            source_features(cache, source)[outer_train],
            cache.event_targets[outer_train, event_index],
            quality[outer_train],
            cache.subjects[outer_train],
            config["inner_splits"],
        )
        for row in prediction_rows:
            fold_rows.append({"event_name": name, **row})
        event_pairs = pair_audit(
            cache.labels[outer_train],
            cache.subjects[outer_train],
            cache.event_targets[outer_train, event_index],
            quality[outer_train],
        )
        for row in event_pairs:
            pair_rows.append({"event_name": name, **row})
        best_pair = max(
            event_pairs,
            key=lambda row: abs(float(row["standardized_effect"])),
            default=None,
        )
        anchor_count, anchor_correct, anchor_agreement, current_anchor_rows = anchor_audit(
            cache, outer_train, event_index, anchors
        )
        anchor_rows.extend(current_anchor_rows)
        manual_support = (
            anchor_agreement is not None
            and anchor_agreement >= float(gates["minimum_anchor_agreement"])
        )
        if anchor_agreement is None and best_pair is not None:
            manual_support = bool(
                abs(float(best_pair["standardized_effect"])) >= 0.25
                and (
                    int(best_pair["subjects_compared"]) == 0
                    or float(best_pair["direction_agreement"]) >= 0.60
                )
            )
        passed = bool(
            coverage >= float(gates["minimum_outer_train_coverage"])
            and iqr >= float(gates["minimum_iqr"])
            and eta <= float(gates["maximum_subject_eta_squared"])
            and tree_improvement >= float(gates["minimum_nonlinear_mae_improvement"])
            and positive_tree_folds >= int(gates["minimum_positive_inner_folds"])
            and manual_support
        )
        if passed:
            passed_names.append(name)
        audit_rows.append(
            {
                "event_name": name,
                "source": source,
                "outer_train_valid": int(valid.sum()),
                "outer_train_coverage": coverage,
                "outer_held_values_not_read_for_selection": int(outer_held.sum()),
                "iqr": iqr,
                "subject_eta_squared": eta,
                "inner_mean_ridge_improvement": ridge_improvement,
                "positive_ridge_folds": positive_ridge_folds,
                "inner_mean_tree_improvement": tree_improvement,
                "positive_tree_folds": positive_tree_folds,
                "anchor_count": anchor_count,
                "anchor_correct": anchor_correct,
                "anchor_agreement": anchor_agreement,
                "best_pair": best_pair["pair"] if best_pair else None,
                "best_pair_effect": (
                    float(best_pair["standardized_effect"]) if best_pair else None
                ),
                "best_pair_subject_agreement": (
                    float(best_pair["direction_agreement"]) if best_pair else None
                ),
                "passed": int(passed),
            }
        )

    write_csv(output / "candidate_event_audit.csv", audit_rows)
    write_csv(output / "inner_event_prediction.csv", fold_rows)
    write_csv(output / "outer_train_pair_distributions.csv", pair_rows)
    write_csv(output / "outer_train_anchor_alignment.csv", anchor_rows)
    plot_anchor_traces(cache, outer_train, output / "outer_train_anchor_traces.png")
    plot_target_subjects(cache, outer_train, output / "outer_train_subject_distributions.png")
    summary = {
        "protocol": config["protocol"],
        "status": "complete",
        "outer_fold": int(config["outer_fold"]),
        "outer_train_samples": int(outer_train.sum()),
        "outer_held_samples": int(outer_held.sum()),
        "outer_held_policy": (
            "Raw cache arrays were generated by per-sample class-agnostic transforms, "
            "but no outer-held target distribution, metric, prediction, or label was "
            "read for R2 selection."
        ),
        "passed_events": passed_names,
        "passed_count": len(passed_names),
        "candidate_count": len(EVENT_NAMES),
        "event_cache": str(cache_path),
        "event_cache_sha256": sha256(cache_path),
        "elapsed_seconds": time.time() - started,
        "gates": gates,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
