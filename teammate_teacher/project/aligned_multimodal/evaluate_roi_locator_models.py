from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from skimage.feature import hog

from aligned_data import frame_map
from audit_motion_crop import analyse_trial
from local_roi_data import sample_positions


PROJECT_DIR = Path(__file__).resolve().parent
ROI_DIR = PROJECT_DIR / "data" / "local_roi_annotation_v2"
DEFAULT_ANNOTATIONS = ROI_DIR / "roi_annotations_final.csv"
DEFAULT_FIRST_PASS = ROI_DIR / "roi_annotations.csv"
DEFAULT_AUDIT = ROI_DIR / "roi_annotation_audit.csv"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p12_roi_locator_baselines"
RAW_WIDTH = 640
RAW_HEIGHT = 480
FEATURE_WIDTH = 160
FEATURE_HEIGHT = 120
SEED = 20260727


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed, motion, and small ExtraTrees ROI locators under the "
            "three subject-disjoint folds."
        )
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--first-pass", type=Path, default=DEFAULT_FIRST_PASS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def annotation_box(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for field in ("x0", "y0", "x1", "y1")],
        dtype=np.float32,
    )


def normalize_box(box: np.ndarray) -> np.ndarray:
    return box / np.asarray(
        [RAW_WIDTH - 1, RAW_HEIGHT - 1, RAW_WIDTH - 1, RAW_HEIGHT - 1],
        dtype=np.float32,
    )


def denormalize_box(box: np.ndarray) -> np.ndarray:
    result = box * np.asarray(
        [RAW_WIDTH - 1, RAW_HEIGHT - 1, RAW_WIDTH - 1, RAW_HEIGHT - 1],
        dtype=np.float32,
    )
    return result


def sanitize_box(box: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(box, dtype=np.float32), 0.0, 1.0)
    x0, y0, x1, y1 = clipped
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)
    minimum_width = 0.04
    minimum_height = 0.04
    if right - left < minimum_width:
        center = 0.5 * (left + right)
        left = max(0.0, center - minimum_width / 2)
        right = min(1.0, left + minimum_width)
        left = max(0.0, right - minimum_width)
    if bottom - top < minimum_height:
        center = 0.5 * (top + bottom)
        top = max(0.0, center - minimum_height / 2)
        bottom = min(1.0, top + minimum_height)
        top = max(0.0, bottom - minimum_height)
    return np.asarray([left, top, right, bottom], dtype=np.float32)


def map_box(box: list[float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.asarray(
        [
            x0 * RAW_WIDTH / 320,
            y0 * RAW_HEIGHT / 240,
            (x1 + 1) * RAW_WIDTH / 320 - 1,
            (y1 + 1) * RAW_HEIGHT / 240 - 1,
        ],
        dtype=np.float32,
    )


def feature_vector(row: dict[str, str]) -> np.ndarray:
    maps = {
        "depth": frame_map(Path(row["depth_dir"]), "depth"),
        "ir": frame_map(Path(row["ir_dir"]), "ir"),
        "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
    }
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    positions = sample_positions(len(common), 12, augment=False)
    frames: list[np.ndarray] = []
    for position in positions:
        with Image.open(maps["depth"][common[position]]) as image:
            rgb = np.asarray(
                image.convert("RGB").resize(
                    (FEATURE_WIDTH, FEATURE_HEIGHT),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
        frames.append(rgb)
    sequence = np.stack(frames).astype(np.float32) / 255.0
    gray = np.stack(
        [
            cv2.cvtColor(
                np.asarray(np.clip(frame * 255, 0, 255), dtype=np.uint8),
                cv2.COLOR_RGB2GRAY,
            )
            for frame in sequence
        ]
    ).astype(np.float32) / 255.0
    middle = gray[len(gray) // 2]
    temporal_mean = sequence.mean(axis=0)
    temporal_std = gray.std(axis=0)
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    first_last = np.abs(gray[-1] - gray[0])
    descriptors = [
        hog(
            middle,
            orientations=9,
            pixels_per_cell=(12, 12),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ),
        hog(
            motion,
            orientations=9,
            pixels_per_cell=(12, 12),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ),
    ]
    small_maps = [
        cv2.resize(temporal_mean, (20, 15), interpolation=cv2.INTER_AREA).reshape(-1),
        cv2.resize(temporal_std, (20, 15), interpolation=cv2.INTER_AREA).reshape(-1),
        cv2.resize(motion, (20, 15), interpolation=cv2.INTER_AREA).reshape(-1),
        cv2.resize(first_last, (20, 15), interpolation=cv2.INTER_AREA).reshape(-1),
    ]
    histograms: list[np.ndarray] = []
    for channel in range(3):
        histogram, _ = np.histogram(
            sequence[..., channel],
            bins=16,
            range=(0.0, 1.0),
            density=True,
        )
        histograms.append(histogram.astype(np.float32))
    return np.concatenate(
        [*descriptors, *small_maps, *histograms],
    ).astype(np.float32)


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]) + 1)
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]) + 1)
    intersection = width * height
    first_area = (first[2] - first[0] + 1) * (first[3] - first[1] + 1)
    second_area = (second[2] - second[0] + 1) * (second[3] - second[1] + 1)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def metric_summary(
    rows: list[dict[str, object]],
    prediction_field: str,
    target_field: str,
) -> dict[str, object]:
    ious = [
        box_iou(
            np.asarray(row[prediction_field], dtype=np.float32),
            np.asarray(row[target_field], dtype=np.float32),
        )
        for row in rows
    ]
    return {
        "count": len(ious),
        "mean_iou": mean(ious),
        "median_iou": median(ious),
        "iou_ge_0_3": sum(value >= 0.3 for value in ious) / len(ious),
        "iou_ge_0_5": sum(value >= 0.5 for value in ious) / len(ious),
        "iou_ge_0_7": sum(value >= 0.7 for value in ious) / len(ious),
    }


def training_bad_label(
    row: dict[str, str],
    audit_by_sample: dict[str, dict[str, str]],
) -> int:
    audited = audit_by_sample.get(row["sample_id"])
    if audited is not None:
        quality = audited["original_quality_interpretation"]
        return int(
            quality
            in {
                "serious_problem_by_region_choice",
                "serious_problem_four_edge_reframe",
                "substantial_adjustment_three_edges",
            }
        )
    if row["bbox_source"] == "pilot30_manual":
        return 1
    if row["bbox_source"] == "pilot30_accepted_auto":
        return 0
    raise ValueError(f"{row['sample_id']}: no machine-box quality label")


def select_gate_threshold(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    outer_train_mask: np.ndarray,
    trees: int,
    max_depth: int,
    held_fold: int,
) -> tuple[float, dict[str, float]]:
    inner_probabilities: list[float] = []
    inner_labels: list[int] = []
    inner_folds = sorted(set(int(value) for value in folds[outer_train_mask]))
    if len(inner_folds) != 2:
        raise AssertionError(f"Held fold {held_fold}: expected two inner folds")
    for inner_valid_fold in inner_folds:
        inner_train = outer_train_mask & (folds != inner_valid_fold)
        inner_valid = outer_train_mask & (folds == inner_valid_fold)
        classifier = ExtraTreesClassifier(
            n_estimators=trees,
            max_depth=max_depth,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight="balanced",
            random_state=SEED + 100 + held_fold * 10 + inner_valid_fold,
            n_jobs=-1,
        )
        classifier.fit(features[inner_train], labels[inner_train])
        inner_probabilities.extend(
            classifier.predict_proba(features[inner_valid])[:, 1].tolist()
        )
        inner_labels.extend(labels[inner_valid].tolist())
    probabilities = np.asarray(inner_probabilities, dtype=np.float64)
    truth = np.asarray(inner_labels, dtype=np.int64)
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.linspace(0.20, 0.80, 25):
        prediction = (probabilities >= threshold).astype(np.int64)
        balanced = balanced_accuracy_score(truth, prediction)
        _, recall, f1, _ = precision_recall_fscore_support(
            truth,
            prediction,
            average="binary",
            zero_division=0,
        )
        candidates.append((float(balanced), float(recall), float(threshold)))
    balanced, recall, threshold = max(
        candidates,
        key=lambda item: (item[0], item[1], -abs(item[2] - 0.5)),
    )
    return threshold, {
        "inner_balanced_accuracy": balanced,
        "inner_bad_recall": recall,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = read_csv(args.annotations.resolve())
    audit_by_sample = {
        row["sample_id"]: row
        for row in read_csv(args.audit.resolve())
        if row["annotation_mode"] == "correction"
    }
    first_pass = {
        row["sample_id"]: annotation_box(row)
        for row in read_csv(args.first_pass.resolve())
        if row["annotation_mode"] == "blind"
    }
    manifest = {row["sample_id"]: row for row in read_csv(args.manifest.resolve())}
    if len(annotations) != 216:
        raise ValueError("Locator evaluation requires 216 final ROI annotations")
    cache_path = output_dir / "roi_locator_features.npz"
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        sample_ids = [str(value) for value in cache["sample_ids"]]
        features = cache["features"]
        auto_boxes = cache["auto_boxes"]
        fallback = cache["fallback"].astype(bool)
    else:
        sample_ids = [row["sample_id"] for row in annotations]
        vectors: list[np.ndarray] = []
        auto: list[np.ndarray] = []
        fallback_values: list[bool] = []
        for index, sample_id in enumerate(sample_ids, start=1):
            row = manifest[sample_id]
            vectors.append(feature_vector(row))
            auto_record, _ = analyse_trial(row, 320, 240)
            auto.append(map_box([float(value) for value in auto_record["bbox"]]))
            fallback_values.append(bool(auto_record["fallback"]))
            if index % 10 == 0 or index == len(sample_ids):
                print(f"Locator features {index}/{len(sample_ids)}", flush=True)
        features = np.stack(vectors)
        auto_boxes = np.stack(auto)
        fallback = np.asarray(fallback_values, dtype=bool)
        np.savez_compressed(
            cache_path,
            sample_ids=np.asarray(sample_ids),
            features=features,
            auto_boxes=auto_boxes,
            fallback=fallback.astype(np.uint8),
        )
    if sample_ids != [row["sample_id"] for row in annotations]:
        raise ValueError("Feature cache sample order does not match ROI annotations")

    targets = np.stack([annotation_box(row) for row in annotations])
    normalized_targets = np.stack([normalize_box(box) for box in targets])
    normalized_auto = np.stack([normalize_box(box) for box in auto_boxes])
    folds = np.asarray([int(row["fold"]) for row in annotations], dtype=np.int64)
    modes = np.asarray([row["annotation_mode"] for row in annotations])
    bad_labels = np.asarray(
        [
            training_bad_label(row, audit_by_sample)
            if row["annotation_mode"] == "correction"
            else -1
            for row in annotations
        ],
        dtype=np.int64,
    )
    prediction_rows: list[dict[str, object]] = []
    model_sizes: dict[str, dict[str, int]] = {}
    gate_protocol: dict[str, object] = {}
    for held_fold in range(3):
        train_mask = (modes == "correction") & (folds != held_fold)
        eval_mask = (modes == "blind") & (folds == held_fold)
        if train_mask.sum() != 104 or eval_mask.sum() != 20:
            raise AssertionError(
                f"Fold {held_fold}: train={train_mask.sum()} eval={eval_mask.sum()}"
            )
        train_x = features[train_mask]
        train_y = normalized_targets[train_mask]
        train_residual = train_y - normalized_auto[train_mask]
        absolute = ExtraTreesRegressor(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=SEED + held_fold,
            n_jobs=-1,
        )
        residual = ExtraTreesRegressor(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=SEED + 10 + held_fold,
            n_jobs=-1,
        )
        absolute.fit(train_x, train_y)
        residual.fit(train_x, train_residual)
        gate_threshold, inner_gate = select_gate_threshold(
            features,
            bad_labels,
            folds,
            train_mask,
            args.trees,
            args.max_depth,
            held_fold,
        )
        gate = ExtraTreesClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight="balanced",
            random_state=SEED + 200 + held_fold,
            n_jobs=-1,
        )
        gate.fit(train_x, bad_labels[train_mask])
        absolute_path = output_dir / f"fold_{held_fold}_absolute_et.joblib"
        residual_path = output_dir / f"fold_{held_fold}_residual_et.joblib"
        gate_path = output_dir / f"fold_{held_fold}_gate_et.joblib"
        joblib.dump(absolute, absolute_path, compress=("gzip", 3))
        joblib.dump(residual, residual_path, compress=("gzip", 3))
        joblib.dump(gate, gate_path, compress=("gzip", 3))
        model_sizes[str(held_fold)] = {
            "absolute_bytes": absolute_path.stat().st_size,
            "residual_bytes": residual_path.stat().st_size,
            "gate_bytes": gate_path.stat().st_size,
        }
        gate_protocol[str(held_fold)] = {
            "threshold": gate_threshold,
            **inner_gate,
            "training_bad": int(bad_labels[train_mask].sum()),
            "training_good": int(train_mask.sum() - bad_labels[train_mask].sum()),
        }
        eval_indices = np.flatnonzero(eval_mask)
        absolute_prediction = absolute.predict(features[eval_mask])
        residual_prediction = (
            normalized_auto[eval_mask] + residual.predict(features[eval_mask])
        )
        gate_probability = gate.predict_proba(features[eval_mask])[:, 1]
        median_box = np.median(normalized_targets[train_mask], axis=0)
        for local_index, global_index in enumerate(eval_indices):
            absolute_box = denormalize_box(
                sanitize_box(absolute_prediction[local_index])
            )
            residual_box = denormalize_box(
                sanitize_box(residual_prediction[local_index])
            )
            fixed_box = denormalize_box(sanitize_box(median_box))
            auto_box = auto_boxes[global_index]
            is_fallback = bool(fallback[global_index])
            prediction_rows.append(
                {
                    "sample_id": sample_ids[global_index],
                    "held_fold": held_fold,
                    "fallback": int(is_fallback),
                    "machine_box_assessment": annotations[global_index][
                        "machine_box_assessment"
                    ],
                    "target_final": targets[global_index].tolist(),
                    "target_first_blind": first_pass[
                        sample_ids[global_index]
                    ].tolist(),
                    "original": auto_box.tolist(),
                    "fixed_median": fixed_box.tolist(),
                    "extra_trees_absolute": absolute_box.tolist(),
                    "extra_trees_residual": residual_box.tolist(),
                    "fallback_hybrid_absolute": (
                        absolute_box.tolist()
                        if is_fallback
                        else auto_box.tolist()
                    ),
                    "fallback_hybrid_residual": (
                        residual_box.tolist()
                        if is_fallback
                        else auto_box.tolist()
                    ),
                    "gate_probability": float(gate_probability[local_index]),
                    "gate_threshold": gate_threshold,
                    "gate_predicted_bad": int(
                        gate_probability[local_index] >= gate_threshold
                    ),
                    "gated_absolute": (
                        absolute_box.tolist()
                        if gate_probability[local_index] >= gate_threshold
                        else auto_box.tolist()
                    ),
                }
            )
        print(f"Locator models fold {held_fold} complete", flush=True)

    methods = [
        "original",
        "fixed_median",
        "extra_trees_absolute",
        "extra_trees_residual",
        "fallback_hybrid_absolute",
        "fallback_hybrid_residual",
        "gated_absolute",
    ]
    metrics: dict[str, object] = {}
    for method in methods:
        metrics[method] = {
            "final_reference": metric_summary(
                prediction_rows,
                method,
                "target_final",
            ),
            "first_blind_sensitivity": metric_summary(
                prediction_rows,
                method,
                "target_first_blind",
            ),
            "fallback_final": metric_summary(
                [row for row in prediction_rows if row["fallback"]],
                method,
                "target_final",
            ),
            "non_fallback_final": metric_summary(
                [row for row in prediction_rows if not row["fallback"]],
                method,
                "target_final",
            ),
            "machine_bad_final": metric_summary(
                [
                    row
                    for row in prediction_rows
                    if row["machine_box_assessment"] == "machine_bad"
                ],
                method,
                "target_final",
            ),
        }
    gate_truth = np.asarray(
        [
            int(row["machine_box_assessment"] == "machine_bad")
            for row in prediction_rows
        ],
        dtype=np.int64,
    )
    gate_prediction = np.asarray(
        [int(row["gate_predicted_bad"]) for row in prediction_rows],
        dtype=np.int64,
    )
    gate_precision, gate_recall, gate_f1, _ = precision_recall_fscore_support(
        gate_truth,
        gate_prediction,
        average="binary",
        zero_division=0,
    )
    gate_evaluation: dict[str, object] = {
        "pooled": {
            "balanced_accuracy": balanced_accuracy_score(
                gate_truth,
                gate_prediction,
            ),
            "bad_precision": gate_precision,
            "bad_recall": gate_recall,
            "bad_f1": gate_f1,
            "confusion_matrix_good_bad": confusion_matrix(
                gate_truth,
                gate_prediction,
                labels=[0, 1],
            ).tolist(),
        },
        "folds": {},
    }
    for held_fold in range(3):
        indices = np.asarray(
            [int(row["held_fold"]) == held_fold for row in prediction_rows]
        )
        truth = gate_truth[indices]
        prediction = gate_prediction[indices]
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            prediction,
            average="binary",
            zero_division=0,
        )
        gate_evaluation["folds"][str(held_fold)] = {
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "bad_precision": precision,
            "bad_recall": recall,
            "bad_f1": f1,
            "confusion_matrix_good_bad": confusion_matrix(
                truth,
                prediction,
                labels=[0, 1],
            ).tolist(),
        }
    csv_rows: list[dict[str, object]] = []
    for row in prediction_rows:
        flat: dict[str, object] = {
            key: value
            for key, value in row.items()
            if not isinstance(value, list)
        }
        for field in [
            "target_final",
            "target_first_blind",
            *methods,
        ]:
            values = row[field]
            for coordinate, value in zip(
                ("x0", "y0", "x1", "y1"),
                values,
                strict=True,
            ):
                flat[f"{field}_{coordinate}"] = value
        csv_rows.append(flat)
    with (output_dir / "reviewed60_predictions.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        "protocol": (
            "Each held fold uses only the 104 correction boxes from the other "
            "two folds; the 20 reviewed samples in the held fold are evaluation-only."
        ),
        "features": {
            "samples": len(sample_ids),
            "dimensions": int(features.shape[1]),
            "input": (
                "12 exact Depth frames; middle/motion HOG, temporal aggregates, "
                "low-resolution maps and color histograms"
            ),
        },
        "model_sizes": model_sizes,
        "gate_protocol": gate_protocol,
        "gate_evaluation": gate_evaluation,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
