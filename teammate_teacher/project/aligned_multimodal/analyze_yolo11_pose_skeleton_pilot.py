from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from audit_yolo11_pose_skeleton import (
    COCO_EDGES,
    H36M_PARENTS,
    canonical_frame_id,
    draw_h36m,
    draw_yolo,
    frame_map,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "runs" / "p28_yolo11_pose_skeleton_40class_pilot"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"

UPPER_BODY = np.asarray((5, 6, 7, 8, 9, 10))
COCO_TO_H36M = {
    5: 11,
    7: 12,
    9: 13,
    6: 14,
    8: 15,
    10: 16,
    11: 4,
    13: 5,
    15: 6,
    12: 1,
    14: 2,
    16: 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse the 40-class YOLO D/IR + Skeleton pilot")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def box_iou_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, 0], b[:, 0])
    y0 = np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 2], b[:, 2])
    y1 = np.minimum(a[:, 3], b[:, 3])
    intersection = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-6)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) * 0.5
        start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    a = a[valid]
    b = b[valid]
    if len(a) < 5 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def motion_curve(points: np.ndarray, confidence: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    valid_point = np.isfinite(points).all(axis=2) & np.isfinite(confidence) & (confidence >= threshold)
    delta = np.linalg.norm(np.diff(points, axis=0), axis=2)
    valid_delta = valid_point[1:] & valid_point[:-1]
    delta[~valid_delta] = np.nan
    curve = np.nanmean(delta, axis=1)
    valid_curve = np.isfinite(curve)
    return curve, valid_curve


def peak_overlap(a: np.ndarray, b: np.ndarray, valid: np.ndarray, share: float = 0.20) -> float:
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 5:
        return float("nan")
    k = max(1, int(math.ceil(len(valid_indices) * share)))
    a_top = set(valid_indices[np.argsort(a[valid_indices])[-k:]].tolist())
    b_top = set(valid_indices[np.argsort(b[valid_indices])[-k:]].tolist())
    return len(a_top & b_top) / k


def finite_stat(values: list[float], fn: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    if fn == "mean":
        return float(np.mean(array))
    if fn == "median":
        return float(np.median(array))
    if fn == "p10":
        return float(np.percentile(array, 10))
    if fn == "p25":
        return float(np.percentile(array, 25))
    if fn == "p75":
        return float(np.percentile(array, 75))
    if fn == "p90":
        return float(np.percentile(array, 90))
    if fn == "p95":
        return float(np.percentile(array, 95))
    raise ValueError(fn)


def normalise_curve(curve: np.ndarray) -> np.ndarray:
    result = curve.copy().astype(np.float64)
    valid = np.isfinite(result)
    if not valid.any():
        return result
    low, high = np.percentile(result[valid], (5, 95))
    if high <= low + 1e-8:
        result[valid] = 0.0
    else:
        result[valid] = np.clip((result[valid] - low) / (high - low), 0.0, 1.0)
    return result


def load_manifest_lookup(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in read_csv(path)}


def analyse_trial(cache_path: Path, row: dict[str, str], threshold: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(cache_path) as cache:
        arrays = {key: cache[key] for key in cache.files}
    ir_boxes = arrays["ir_boxes_xyxy_conf"]
    depth_boxes = arrays["depth_boxes_xyxy_conf"]
    ir_keypoints = arrays["ir_keypoints_xy_conf"]
    depth_keypoints = arrays["depth_keypoints_xy_conf"]
    skeleton = arrays["skeleton_h36m_xyz_conf_normalised"]

    both_box = np.isfinite(ir_boxes[:, 4]) & np.isfinite(depth_boxes[:, 4])
    ious = box_iou_many(ir_boxes[both_box, :4], depth_boxes[both_box, :4]) if both_box.any() else np.asarray([])
    ir_centers = (ir_boxes[:, :2] + ir_boxes[:, 2:4]) * 0.5
    depth_centers = (depth_boxes[:, :2] + depth_boxes[:, 2:4]) * 0.5
    center_distance = np.linalg.norm(ir_centers - depth_centers, axis=1)
    center_distance[~both_box] = np.nan
    diagonal = math.hypot(640.0, 480.0)
    center_distance_image = center_distance / diagonal
    mean_box_height = ((ir_boxes[:, 3] - ir_boxes[:, 1]) + (depth_boxes[:, 3] - depth_boxes[:, 1])) * 0.5

    joint_valid = (
        np.isfinite(ir_keypoints[:, :, 2])
        & np.isfinite(depth_keypoints[:, :, 2])
        & (ir_keypoints[:, :, 2] >= threshold)
        & (depth_keypoints[:, :, 2] >= threshold)
    )
    joint_distance = np.linalg.norm(ir_keypoints[:, :, :2] - depth_keypoints[:, :, :2], axis=2)
    joint_distance[~joint_valid] = np.nan
    upper_distance = joint_distance[:, UPPER_BODY]
    upper_distance_height = upper_distance / mean_box_height[:, None]

    ir_local = arrays["ir_keypoints_bbox_local"]
    yolo_arm_points = ir_local[:, UPPER_BODY, :2]
    yolo_arm_conf = ir_local[:, UPPER_BODY, 2]
    h36m_upper = np.asarray((11, 14, 12, 15, 13, 16))
    skeleton_arm_points = skeleton[:, h36m_upper, :3]
    skeleton_arm_conf = skeleton[:, h36m_upper, 3]
    yolo_motion, yolo_motion_valid = motion_curve(yolo_arm_points, yolo_arm_conf, threshold)
    skeleton_motion, skeleton_motion_valid = motion_curve(skeleton_arm_points, skeleton_arm_conf, 0.25)
    motion_valid = yolo_motion_valid & skeleton_motion_valid
    motion_corr = spearman(yolo_motion, skeleton_motion, motion_valid)
    motion_peak_overlap = peak_overlap(yolo_motion, skeleton_motion, motion_valid)

    result: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "user_id": row["user_id"],
        "frames": len(ir_boxes),
        "both_box_frames": int(both_box.sum()),
        "both_box_rate": float(both_box.mean()),
        "box_iou_median": finite_stat(ious.tolist(), "median"),
        "box_iou_p10": finite_stat(ious.tolist(), "p10"),
        "center_distance_px_median": finite_stat(center_distance.tolist(), "median"),
        "center_distance_image_p90": finite_stat(center_distance_image.tolist(), "p90"),
        "upper_joint_pairs": int(np.isfinite(upper_distance).sum()),
        "upper_joint_distance_px_median": finite_stat(upper_distance.ravel().tolist(), "median"),
        "upper_joint_distance_box_height_median": finite_stat(
            upper_distance_height.ravel().tolist(), "median"
        ),
        "upper_joint_distance_box_height_p90": finite_stat(
            upper_distance_height.ravel().tolist(), "p90"
        ),
        "yolo_skeleton_motion_valid_transitions": int(motion_valid.sum()),
        "yolo_skeleton_motion_corr": motion_corr,
        "yolo_skeleton_top20_peak_overlap": motion_peak_overlap,
    }
    curves = {
        "yolo_motion": yolo_motion,
        "skeleton_motion": skeleton_motion,
        "motion_valid": motion_valid,
        "frame_ids": arrays["frame_ids"],
        "ir_boxes": ir_boxes,
        "depth_boxes": depth_boxes,
        "ir_keypoints": ir_keypoints,
        "depth_keypoints": depth_keypoints,
        "skeleton_raw": arrays["skeleton_h36m_xyz_conf_raw"],
    }
    return result, curves


def aggregate_trials(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "both_box_rate",
        "box_iou_median",
        "box_iou_p10",
        "center_distance_px_median",
        "center_distance_image_p90",
        "upper_joint_distance_px_median",
        "upper_joint_distance_box_height_median",
        "upper_joint_distance_box_height_p90",
        "yolo_skeleton_motion_corr",
        "yolo_skeleton_top20_peak_overlap",
    )
    output: dict[str, Any] = {"trials": len(rows), "frames": sum(int(row["frames"]) for row in rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        output[metric] = {
            "valid_trials": int(np.isfinite(np.asarray(values)).sum()),
            "mean": finite_stat(values, "mean"),
            "median": finite_stat(values, "median"),
            "p25": finite_stat(values, "p25"),
            "p75": finite_stat(values, "p75"),
            "p90": finite_stat(values, "p90"),
        }
    return output


def plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    items = (
        ("box_iou_median", "D/IR person box IoU per trial", (0.0, 1.0)),
        (
            "upper_joint_distance_box_height_median",
            "D/IR upper-joint distance / person height",
            (0.0, 0.5),
        ),
        ("yolo_skeleton_motion_corr", "IR-YOLO vs Skeleton arm-motion Spearman", (-1.0, 1.0)),
        ("yolo_skeleton_top20_peak_overlap", "Top-20% motion peak overlap", (0.0, 1.0)),
    )
    for axis, (metric, title, limits) in zip(axes.ravel(), items):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        axis.hist(values, bins=12, color="#4c78a8", alpha=0.85)
        if len(values):
            median = float(np.median(values))
            axis.axvline(median, color="#e45756", linewidth=2, label=f"median={median:.3f}")
            axis.legend()
        axis.set_title(title)
        axis.set_xlim(*limits)
        axis.set_ylabel("trials")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_motion_examples(
    trial_rows: list[dict[str, Any]], curves_by_sample: dict[str, dict[str, np.ndarray]], output: Path
) -> None:
    valid_rows = [row for row in trial_rows if np.isfinite(float(row["yolo_skeleton_motion_corr"]))]
    if not valid_rows:
        return
    valid_rows.sort(key=lambda row: float(row["yolo_skeleton_motion_corr"]))
    selected = [valid_rows[0], valid_rows[len(valid_rows) // 2], valid_rows[-1]]
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    for axis, row in zip(axes, selected):
        curves = curves_by_sample[row["sample_id"]]
        yolo = normalise_curve(curves["yolo_motion"])
        skeleton = normalise_curve(curves["skeleton_motion"])
        x = np.arange(1, len(yolo) + 1)
        axis.plot(x, yolo, label="IR-YOLO upper-limb motion", linewidth=1.8)
        axis.plot(x, skeleton, label="Dataset Skeleton upper-limb motion", linewidth=1.8)
        axis.set_ylim(-0.05, 1.05)
        axis.set_title(
            f"{row['class_name']} | {row['sample_id']} | Spearman={float(row['yolo_skeleton_motion_corr']):.3f}"
        )
        axis.set_ylabel("robust normalized motion")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("frame transition")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def make_transfer_sheet(
    sample_id: str,
    curves: dict[str, np.ndarray],
    manifest_row: dict[str, str],
    output: Path,
) -> None:
    maps = {
        "ir": frame_map(Path(manifest_row["ir_path"]), "ir"),
        "depth": frame_map(Path(manifest_row["depth_color_path"]), "depth"),
    }
    frame_ids = [str(value) for value in curves["frame_ids"]]
    wrist = curves["ir_keypoints"][:, (9, 10), :]
    valid = np.isfinite(wrist[:, :, 2]) & (wrist[:, :, 2] >= 0.25)
    motion = np.nanmean(np.where(valid[1:] & valid[:-1], np.linalg.norm(np.diff(wrist[:, :, :2], axis=0), axis=2), np.nan), axis=1)
    candidate = np.flatnonzero(np.isfinite(motion))
    if len(candidate):
        ranked = candidate[np.argsort(motion[candidate])]
        positions = np.unique(np.rint(np.linspace(0, len(ranked) - 1, min(6, len(ranked)))).astype(int))
        selected = (ranked[positions] + 1).tolist()
    else:
        selected = np.unique(np.rint(np.linspace(0, len(frame_ids) - 1, min(6, len(frame_ids)))).astype(int)).tolist()
    rows: list[np.ndarray] = []
    for index in selected:
        frame_id = frame_ids[index]
        ir = cv2.imread(str(maps["ir"][frame_id]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(maps["depth"][frame_id]), cv2.IMREAD_COLOR)
        if ir is None or depth is None:
            continue
        ir = cv2.resize(ir, (640, 480), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_AREA)
        ir_own = draw_yolo(ir, curves["ir_boxes"][index], curves["ir_keypoints"][index], "IR own pose")
        depth_transfer = draw_yolo(
            depth,
            curves["ir_boxes"][index],
            curves["ir_keypoints"][index],
            "IR pose copied exactly",
        )
        depth_own = draw_yolo(
            depth,
            curves["depth_boxes"][index],
            curves["depth_keypoints"][index],
            "Depth own pose",
        )
        cv2.putText(depth_own, f"frame={index}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.concatenate([ir_own, depth_transfer, depth_own], axis=1))
    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def make_all_class_transfer_contact_sheet(
    analysed: list[dict[str, Any]],
    curves_by_sample: dict[str, dict[str, np.ndarray]],
    manifest_lookup: dict[str, dict[str, str]],
    output: Path,
) -> None:
    tiles: list[np.ndarray] = []
    for row in sorted(analysed, key=lambda item: int(item["class_id"])):
        sample_id = row["sample_id"]
        curves = curves_by_sample[sample_id]
        confidence = curves["ir_keypoints"][:, (9, 10), 2]
        wrist_score = np.nanmean(confidence, axis=1)
        box_score = curves["ir_boxes"][:, 4]
        score = np.nan_to_num(wrist_score, nan=-1.0) + np.nan_to_num(box_score, nan=-1.0)
        index = int(np.argmax(score))
        frame_id = str(curves["frame_ids"][index])
        depth_map = frame_map(Path(manifest_lookup[sample_id]["depth_color_path"]), "depth")
        depth = cv2.imread(str(depth_map[frame_id]), cv2.IMREAD_COLOR)
        if depth is None:
            depth = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_AREA)
        tile = draw_yolo(
            depth,
            curves["ir_boxes"][index],
            curves["ir_keypoints"][index],
            "IR pose copied to Depth",
        )
        cv2.rectangle(tile, (0, 445), (640, 480), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"c{int(row['class_id']):02d} {row['class_name']}",
            (8, 469),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(tile, (320, 240), interpolation=cv2.INTER_AREA))
    columns = 5
    blank = np.zeros_like(tiles[0])
    rows: list[np.ndarray] = []
    for start in range(0, len(tiles), columns):
        chunk = tiles[start : start + columns]
        chunk.extend([blank] * (columns - len(chunk)))
        rows.append(np.concatenate(chunk, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    trial_rows = read_csv(run_dir / "trial_summary.csv")
    manifest_lookup = load_manifest_lookup(args.manifest.resolve())
    analysed: list[dict[str, Any]] = []
    curves_by_sample: dict[str, dict[str, np.ndarray]] = {}
    for row in trial_rows:
        relative = Path(*row["sample_id"].split("/"))
        cache_path = run_dir / "trial_cache" / relative.with_suffix(".npz")
        result, curves = analyse_trial(cache_path, row, args.keypoint_conf)
        analysed.append(result)
        curves_by_sample[row["sample_id"]] = curves
    write_csv(run_dir / "alignment_and_fusion_per_trial.csv", analysed)
    summary = aggregate_trials(analysed)
    (run_dir / "alignment_and_fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figures = run_dir / "analysis_figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_summary(analysed, figures / "pilot_metric_distributions.png")
    plot_motion_examples(analysed, curves_by_sample, figures / "yolo_skeleton_motion_examples.png")
    medicine = next((row for row in analysed if int(row["class_id"]) == 37), analysed[0])
    make_transfer_sheet(
        medicine["sample_id"],
        curves_by_sample[medicine["sample_id"]],
        manifest_lookup[medicine["sample_id"]],
        figures / "take_medicine_ir_pose_to_depth_audit.png",
    )
    make_all_class_transfer_contact_sheet(
        analysed,
        curves_by_sample,
        manifest_lookup,
        figures / "all_40_classes_ir_pose_copied_to_depth.png",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
