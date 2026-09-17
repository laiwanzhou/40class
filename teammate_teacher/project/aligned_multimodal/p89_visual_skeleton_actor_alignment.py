from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from audit_yolo11_pose_skeleton import SEMANTIC_PAIRS, frame_map
from p31_skeleton_imu_preprocessing import build_skeleton_features, safe_trial_path
from p46_event_preprocessing import rotate_skeleton_cache
from p89_skeleton_identity_tracking_expert import (
    FOLDS,
    TEST_MANIFEST,
    TEST_PIXEL,
    TEST_ROWS,
    TEST_SOURCE,
    TRAIN_MANIFEST,
    TRAIN_PIXEL,
    TRAIN_ROWS,
    TRAIN_SOURCE,
    load_people,
    metrics,
    model,
    summarize_window,
)


PROJECT_DIR = Path(__file__).resolve().parent
BASELINE = PROJECT_DIR / "runs/p89_skeleton_invariant_expert_v1"
TRAIN_VISUAL = PROJECT_DIR / "runs/p28_adaptive_ir_pose_skeleton_full/trial_cache"
TEST_VISUAL = PROJECT_DIR / "runs/p28_adaptive_ir_pose_skeleton_test/trial_cache"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p89_visual_skeleton_actor_alignment_v1"
H1_USERS = ["user6", "user8", "user17", "user23"]
H2_USERS = ["user5", "user7", "user16", "user18", "user19"]

# Candidate Skeleton x/z has a strong, label-free calibration to the selected IR
# person's image x/y on single-person Train frames: corr(-x, image_x)=0.742 and
# corr(-z, image_y)=0.898.  Only large per-frame residual wins may replace slot 0.
STRATEGIES = {
    "visual_ratio_040": 0.40,
    "visual_ratio_050": 0.50,
    "visual_ratio_060": 0.60,
}
PAIRS = tuple((coco, h36m) for _, coco, h36m in SEMANTIC_PAIRS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the dataset Skeleton candidate aligned to the visually tracked IR actor."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def projection_residual(yolo: np.ndarray, skeleton: np.ndarray) -> float:
    image_points = []
    skeleton_points = []
    for coco, h36m in PAIRS:
        if (
            np.isfinite(yolo[coco, :2]).all()
            and float(yolo[coco, 2]) >= 0.25
            and np.isfinite(skeleton[h36m, :3]).all()
        ):
            image_points.append(yolo[coco, :2])
            # Coordinate signs are calibrated only from single-person Train frames.
            skeleton_points.append((-skeleton[h36m, 0], -skeleton[h36m, 2]))
    if len(image_points) < 6:
        return float("nan")
    image = np.asarray(image_points, dtype=np.float64)
    projected = np.asarray(skeleton_points, dtype=np.float64)
    image -= image.mean(axis=0, keepdims=True)
    projected -= projected.mean(axis=0, keepdims=True)
    # BBox-local image coordinates have a different horizontal/vertical scale.
    # Fit only the two positive axis scales; do not rotate or reflect candidates.
    for axis in range(2):
        denominator = float(np.sum(projected[:, axis] ** 2))
        scale = float(np.sum(image[:, axis] * projected[:, axis])) / max(denominator, 1e-8)
        if scale <= 0.0:
            return float("nan")
        projected[:, axis] *= scale
    numerator = float(np.sqrt(np.mean(np.sum((image - projected) ** 2, axis=1))))
    denominator = float(np.sqrt(np.mean(np.sum(image**2, axis=1))))
    return numerator / max(denominator, 1e-6)


def select_sequence(
    people_by_frame: list[list[np.ndarray]],
    visual_pose: np.ndarray,
    ratio_threshold: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    empty = np.full((17, 4), np.nan, dtype=np.float32)
    empty[:, 3] = 0.0
    selected = []
    changed = eligible = strong_people0 = 0
    ratios = []
    for people, yolo in zip(people_by_frame, visual_pose):
        if not people:
            selected.append(empty)
            continue
        if len(people) == 1:
            selected.append(people[0])
            continue
        residuals = np.asarray(
            [projection_residual(yolo, person) for person in people], dtype=np.float64
        )
        if not np.isfinite(residuals[0]) or not np.isfinite(residuals[1:]).any():
            selected.append(people[0])
            continue
        eligible += 1
        alternate_slot = 1 + int(np.nanargmin(residuals[1:]))
        ratio = float(residuals[alternate_slot] / max(residuals[0], 1e-8))
        ratios.append(ratio)
        if ratio < ratio_threshold:
            selected.append(people[alternate_slot])
            changed += 1
        else:
            selected.append(people[0])
            strong_people0 += int(residuals[0] < ratio_threshold * residuals[alternate_slot])
    return np.stack(selected).astype(np.float32), {
        "eligible_frames": eligible,
        "changed_frames": changed,
        "strong_people0_frames": strong_people0,
        "median_alternate_ratio": float(np.median(ratios)) if ratios else float("nan"),
    }


def make_feature(raw: np.ndarray, frame_times: np.ndarray, chosen: np.ndarray, present: np.ndarray) -> np.ndarray:
    built = build_skeleton_features(raw, frame_times)
    rotated = rotate_skeleton_cache(
        built["features"],
        built["feature_mask"],
        built["joint_mask"],
        built["relations"],
        built["relation_mask"],
        frame_times,
    )
    quality = np.asarray(built["frame_quality"], dtype=np.float32) * present.astype(np.float32)
    return summarize_window(
        rotated["features"][chosen].astype(np.float16).astype(np.float32),
        np.asarray(built["joint_mask"])[chosen],
        rotated["relations"][chosen].astype(np.float16).astype(np.float32),
        rotated["relation_mask"][chosen],
        quality[chosen].astype(np.float16).astype(np.float32),
    )


def build_split(
    *,
    rows_path: Path,
    pixel_cache: Path,
    source_cache: Path,
    manifest_path: Path,
    visual_cache: Path,
    baseline_path: Path,
    output: Path,
    split: str,
    rebuild: bool,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray], list[dict[str, str]]]:
    rows = read_csv(rows_path)
    baseline = np.load(baseline_path, mmap_mode="r")
    paths = {name: output / f"{split}_{name}_features.npy" for name in STRATEGIES}
    audit_path = output / f"{split}_visual_alignment_audit.csv"
    if not rebuild and all(path.is_file() for path in paths.values()) and audit_path.is_file():
        return rows, {name: np.load(path, mmap_mode="r") for name, path in paths.items()}, read_csv(audit_path)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=baseline.shape)
        for name, path in paths.items()
    }
    manifests = {row["sample_id"]: row for row in read_csv(manifest_path)}
    indices = np.asarray(np.load(pixel_cache / "source_frame_indices.npy", mmap_mode="r"))
    audits: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        for array in arrays.values():
            array[index] = baseline[index]
        source_id = row["source_id"]
        source_path = source_cache / "trial_motion_cache" / safe_trial_path(source_id).with_suffix(".npz")
        with np.load(source_path, allow_pickle=False) as source:
            frame_ids = np.asarray(source["frame_ids"]).astype(str)
            frame_times = np.asarray(source["frame_time_seconds"], dtype=np.float64)
        visual_path = visual_cache / safe_trial_path(source_id).with_suffix(".npz")
        if not visual_path.is_file():
            audits.append({
                "sample_id": row["sample_id"], "source_id": source_id, "class_id": row["class_id"],
                "user_id": row["user_id"], "frames": str(len(frame_ids)), "multi_frames": "0",
                "visual_available": "0", **{f"{name}_changed_frames": "0" for name in STRATEGIES},
            })
            continue
        with np.load(visual_path, allow_pickle=False) as visual:
            visual_ids = np.asarray(visual["frame_ids"]).astype(str)
            visual_pose = np.asarray(visual["ir_keypoints_bbox_local"], dtype=np.float32)
        if not np.array_equal(frame_ids, visual_ids):
            raise RuntimeError(f"visual/Skeleton frame mismatch: {source_id}")
        skeleton_paths = frame_map(Path(manifests[source_id]["skeleton_path"]), "skeleton")
        people = [load_people(skeleton_paths[frame_id]) for frame_id in frame_ids]
        audit = {
            "sample_id": row["sample_id"], "source_id": source_id, "class_id": row["class_id"],
            "user_id": row["user_id"], "frames": str(len(frame_ids)),
            "multi_frames": str(sum(len(item) > 1 for item in people)), "visual_available": "1",
        }
        chosen = np.asarray(indices[index], dtype=np.int64)
        for name, threshold in STRATEGIES.items():
            raw, diagnostics = select_sequence(people, visual_pose, threshold)
            audit[f"{name}_changed_frames"] = str(diagnostics["changed_frames"])
            audit[f"{name}_eligible_frames"] = str(diagnostics["eligible_frames"])
            audit[f"{name}_strong_people0_frames"] = str(diagnostics["strong_people0_frames"])
            audit[f"{name}_median_alternate_ratio"] = str(diagnostics["median_alternate_ratio"])
            if diagnostics["changed_frames"]:
                arrays[name][index] = make_feature(
                    raw, frame_times, chosen, np.asarray([bool(item) for item in people])
                )
        audits.append(audit)
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            for array in arrays.values():
                array.flush()
            print(json.dumps({"split": split, "processed": index + 1, "total": len(rows), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = sorted(set().union(*(row.keys() for row in audits)))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audits)
    return rows, arrays, audits


def evaluate_subset(labels: np.ndarray, prediction: np.ndarray, baseline: np.ndarray, users: np.ndarray, wanted: list[str], multi: np.ndarray) -> dict[str, object]:
    held = np.isin(users, wanted)
    result = {
        "correct": int(np.sum(labels[held] == prediction[held])),
        "baseline_correct": int(np.sum(labels[held] == baseline[held])),
        "net": int(np.sum(labels[held] == prediction[held]) - np.sum(labels[held] == baseline[held])),
        "ever_multi_net": int(np.sum(labels[held & multi] == prediction[held & multi]) - np.sum(labels[held & multi] == baseline[held & multi])),
        "per_user_net": {},
    }
    for user in wanted:
        selected = users == user
        result["per_user_net"][user] = int(np.sum(labels[selected] == prediction[selected]) - np.sum(labels[selected] == baseline[selected]))
    return result


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    train_rows, train_features, train_audit = build_split(
        rows_path=TRAIN_ROWS, pixel_cache=TRAIN_PIXEL, source_cache=TRAIN_SOURCE,
        manifest_path=TRAIN_MANIFEST, visual_cache=TRAIN_VISUAL,
        baseline_path=BASELINE / "train_features.npy", output=output, split="train",
        rebuild=args.rebuild_features,
    )
    test_rows, test_features, test_audit = build_split(
        rows_path=TEST_ROWS, pixel_cache=TEST_PIXEL, source_cache=TEST_SOURCE,
        manifest_path=TEST_MANIFEST, visual_cache=TEST_VISUAL,
        baseline_path=BASELINE / "test_features.npy", output=output, split="test",
        rebuild=args.rebuild_features,
    )
    labels = np.asarray([int(row["class_id"]) for row in train_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in train_rows]).astype(str)
    multi = np.asarray([int(row.get("multi_frames", "0")) > 0 for row in train_audit])
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    probabilities = {name: np.zeros((len(labels), 40), dtype=np.float64) for name in STRATEGIES}
    baseline_probability = np.zeros((len(labels), 40), dtype=np.float64)
    test_probabilities = {}
    results = {}
    # Freeze the model on the immutable people[0] baseline.  This isolates the
    # effect of actor replacement and prevents a handful of changed Train rows
    # from perturbing many unrelated ExtraTrees decisions.
    baseline_train = np.load(BASELINE / "train_features.npy", mmap_mode="r")
    baseline_test = np.load(BASELINE / "test_features.npy", mmap_mode="r")
    for fold in folds:
        fit = np.isin(users, fold["train_users"])
        held = np.isin(users, fold["val_users"])
        estimator = model()
        estimator.fit(baseline_train[fit], labels[fit])
        target = np.flatnonzero(held)
        for name, features in (("baseline", baseline_train), *train_features.items()):
            partial = estimator.predict_proba(features[held])
            destination = baseline_probability if name == "baseline" else probabilities[name]
            destination[np.ix_(target, estimator.classes_.astype(np.int64))] = partial

    final = model()
    final.fit(baseline_train, labels)
    for name in STRATEGIES:
        test_probability = np.zeros((len(test_rows), 40), dtype=np.float64)
        partial = final.predict_proba(test_features[name])
        test_probability[:, final.classes_.astype(np.int64)] = partial
        test_probabilities[name] = test_probability
    baseline_test_probability = np.zeros((len(test_rows), 40), dtype=np.float64)
    partial = final.predict_proba(baseline_test)
    baseline_test_probability[:, final.classes_.astype(np.int64)] = partial
    baseline_prediction = baseline_probability.argmax(1)
    for name in STRATEGIES:
        prediction = probabilities[name].argmax(1)
        results[name] = {
            "overall": metrics(labels, prediction),
            "overall_net": int(np.sum(labels == prediction) - np.sum(labels == baseline_prediction)),
            "ever_multi": metrics(labels[multi], prediction[multi]),
            "ever_multi_baseline_correct": int(np.sum(labels[multi] == baseline_prediction[multi])),
            "H1": evaluate_subset(labels, prediction, baseline_prediction, users, H1_USERS, multi),
            "H2": evaluate_subset(labels, prediction, baseline_prediction, users, H2_USERS, multi),
            "train_changed_trials": int(sum(int(row.get(f"{name}_changed_frames", "0")) > 0 for row in train_audit)),
            "train_changed_frames": int(sum(int(row.get(f"{name}_changed_frames", "0")) for row in train_audit)),
            "test_changed_trials": int(sum(int(row.get(f"{name}_changed_frames", "0")) > 0 for row in test_audit)),
            "test_changed_frames": int(sum(int(row.get(f"{name}_changed_frames", "0")) for row in test_audit)),
        }
    selected = max(
        STRATEGIES,
        key=lambda name: (
            results[name]["H1"]["net"],
            min(results[name]["H1"]["per_user_net"].values()),
            results[name]["H1"]["ever_multi_net"],
            -results[name]["train_changed_frames"],
        ),
    )
    np.savez_compressed(
        output / "oof_logits.npz", sample_ids=np.asarray([row["sample_id"] for row in train_rows]),
        labels=labels, skeleton_logits=np.log(np.maximum(probabilities[selected], 1e-12)),
    )
    np.savez_compressed(
        output / "test_logits.npz", sample_ids=np.asarray([row["sample_id"] for row in test_rows]),
        skeleton_logits=np.log(np.maximum(test_probabilities[selected], 1e-12)),
    )
    report = {
        "stage": "P89_visual_localization_then_Skeleton_candidate_alignment_v1",
        "protocol": "Use the already frozen IR YOLO target track, match each raw 3-D Skeleton candidate by calibrated 2-D projected pose residual, and keep the Skeleton classifier frozen on people[0] so only visually selected rows can change. Select the residual-ratio gate on H1 and transfer unchanged to H2/Test.",
        "calibration": {"image_x_vs_negative_skeleton_x": 0.7421, "image_y_vs_negative_skeleton_z": 0.8982},
        "baseline": metrics(labels, baseline_prediction),
        "strategies": results,
        "selected_on_H1": selected,
        "H2_confirmation": results[selected]["H2"],
        "test_prediction_changes_vs_baseline_skeleton": int(np.sum(test_probabilities[selected].argmax(1) != baseline_test_probability.argmax(1))),
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
