"""Build label-free P99-D1 hand/workspace Depth geometry descriptors.

P29 supplies synchronized ROI geometry derived from the frozen IR pose cache.
This builder reads the original colorized Depth frames, converts the monotonic
JET hue to a normalized relative-depth rank, and summarizes only explicit local
surface and geometry measurements.  It never writes a ground-truth label.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from audit_yolo11_pose_skeleton import frame_map, safe_name
from p99_depth_oof_expert import canonical_hash


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_depth_geometry_d1.json"
TEMPORAL_STATISTICS = (
    "mean",
    "std",
    "q10",
    "median",
    "q90",
    "mean_abs_delta",
    "max_abs_delta",
    "linear_slope",
    "early_mean",
    "middle_mean",
    "late_mean",
)
DEPTH_STATISTICS = ("valid_fraction", "mean", "std", "q10", "median", "q90")
JOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P99-D1 label-free descriptors")
    parser.add_argument("--stage", choices=("h1", "h2_confirmation"), default="h1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--h1-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def canonical_sample_id(row: dict[str, str]) -> str:
    return (
        f"train__c{int(row['class_id']):02d}__{row['user_id']}__{row['trial_id']}"
    )


def depth_color_to_rank(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert the fixed JET visualization to a monotonic relative-depth rank.

    OpenCV hue is 0 for red, 30 for yellow, 60 for green, 90 for cyan and
    120 for blue.  The source Depth_Color images use that interval.  Saturated
    nonblack pixels are valid; black holes remain explicitly missing.
    """

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Depth_Color image must be BGR with three channels")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    valid = (hsv[..., 1] >= 32) & (hsv[..., 2] >= 32)
    rank = np.clip(hsv[..., 0].astype(np.float32) / 120.0, 0.0, 1.0)
    rank[~valid] = 0.0
    return rank, valid


def roi_depth_statistics(
    rank: np.ndarray, valid_depth: np.ndarray, box: np.ndarray, roi_valid: bool
) -> np.ndarray:
    if not roi_valid or not np.isfinite(box).all():
        return np.zeros(len(DEPTH_STATISTICS), dtype=np.float32)
    height, width = rank.shape
    x1 = int(np.clip(np.floor(box[0]), 0, width - 1))
    y1 = int(np.clip(np.floor(box[1]), 0, height - 1))
    x2 = int(np.clip(np.ceil(box[2]) + 1, x1 + 1, width))
    y2 = int(np.clip(np.ceil(box[3]) + 1, y1 + 1, height))
    crop_valid = valid_depth[y1:y2, x1:x2]
    fraction = float(crop_valid.mean()) if crop_valid.size else 0.0
    if not crop_valid.any():
        return np.asarray((fraction, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    values = rank[y1:y2, x1:x2][crop_valid]
    q10, median, q90 = np.quantile(values, (0.10, 0.50, 0.90))
    return np.asarray(
        (fraction, values.mean(), values.std(), q10, median, q90),
        dtype=np.float32,
    )


def _safe_center(box: np.ndarray, valid: bool, width: float, height: float) -> np.ndarray:
    if not valid or not np.isfinite(box).all():
        return np.zeros(5, dtype=np.float32)
    box_width = max(float(box[2] - box[0]), 0.0) / width
    box_height = max(float(box[3] - box[1]), 0.0) / height
    return np.asarray(
        (
            0.5 * float(box[0] + box[2]) / width,
            0.5 * float(box[1] + box[3]) / height,
            box_width,
            box_height,
            box_width * box_height,
        ),
        dtype=np.float32,
    )


def frame_descriptors(
    cache: dict[str, np.ndarray],
    depth_ranks: list[tuple[np.ndarray, np.ndarray]],
    selected_regions: tuple[str, ...],
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    region_names = tuple(str(value) for value in cache["region_names"])
    region_lookup = {name: index for index, name in enumerate(region_names)}
    missing = [name for name in selected_regions if name not in region_lookup]
    if missing:
        raise KeyError(f"P29 cache misses D1 regions: {missing}")
    frame_count = len(cache["frame_ids"])
    if len(depth_ranks) != frame_count:
        raise ValueError("Depth frame count differs from P29 ROI frame count")
    width = float(np.asarray(cache["image_width"]).item())
    height = float(np.asarray(cache["image_height"]).item())
    boxes = cache["roi_boxes_xyxy"].astype(np.float32)
    valid = cache["roi_valid"].astype(bool)
    quality = cache["roi_quality"].astype(np.float32)
    clipped = cache["roi_clipped_ratio"].astype(np.float32)

    geometry_names: list[str] = []
    for region in selected_regions:
        geometry_names.extend(
            f"{region}_{suffix}"
            for suffix in ("cx", "cy", "width", "height", "area", "quality", "valid", "clipped")
        )
    for joint in JOINT_NAMES:
        geometry_names.extend((f"{joint}_x", f"{joint}_y", f"{joint}_quality"))
    geometry_names.extend(
        (
            "left_hand_body_dx", "left_hand_body_dy",
            "right_hand_body_dx", "right_hand_body_dy",
            "workspace_body_dx", "workspace_body_dy", "workspace_body_area_ratio",
            "left_right_dx", "left_right_dy", "left_right_distance",
            "left_hand_workspace_dx", "left_hand_workspace_dy",
            "right_hand_workspace_dx", "right_hand_workspace_dy",
            "pose_quality_factor", "left_right_ambiguous",
        )
    )

    depth_names: list[str] = []
    for region in selected_regions:
        depth_names.extend(f"{region}_depth_{suffix}" for suffix in DEPTH_STATISTICS)
    depth_names.extend(
        (
            "left_right_depth_mean_delta", "left_right_depth_median_delta",
            "left_body_depth_mean_delta", "left_body_depth_median_delta",
            "right_body_depth_mean_delta", "right_body_depth_median_delta",
            "workspace_body_depth_mean_delta", "workspace_body_depth_median_delta",
            "hands_workspace_depth_mean_delta", "hands_workspace_depth_median_delta",
        )
    )

    geometry_rows: list[np.ndarray] = []
    depth_rows: list[np.ndarray] = []
    for frame_index, (rank, valid_depth) in enumerate(depth_ranks):
        centers: dict[str, np.ndarray] = {}
        geometry: list[float] = []
        surface: dict[str, np.ndarray] = {}
        depth_values: list[float] = []
        for region in selected_regions:
            region_index = region_lookup[region]
            region_valid = bool(valid[frame_index, region_index])
            center = _safe_center(
                boxes[frame_index, region_index], region_valid, width, height
            )
            centers[region] = center
            geometry.extend(center.tolist())
            geometry.extend(
                (
                    float(quality[frame_index, region_index]),
                    float(region_valid),
                    float(clipped[frame_index, region_index]),
                )
            )
            stats = roi_depth_statistics(
                rank,
                valid_depth,
                boxes[frame_index, region_index],
                region_valid,
            )
            surface[region] = stats
            depth_values.extend(stats.tolist())

        joints = cache["arm_joint_xy_conf_for_roi"][frame_index].astype(np.float32)
        joint_quality = cache["arm_joint_quality_for_roi"][frame_index].astype(np.float32)
        for joint_index in range(len(JOINT_NAMES)):
            available = bool(
                np.isfinite(joints[joint_index, :2]).all() and joint_quality[joint_index] > 0
            )
            geometry.extend(
                (
                    float(joints[joint_index, 0] / width) if available else 0.0,
                    float(joints[joint_index, 1] / height) if available else 0.0,
                    float(joint_quality[joint_index]) if available else 0.0,
                )
            )

        body = centers["full_body"]
        left = centers["left_hand"]
        right = centers["right_hand"]
        workspace = centers["hand_workspace"]
        geometry.extend(
            (
                float(left[0] - body[0]), float(left[1] - body[1]),
                float(right[0] - body[0]), float(right[1] - body[1]),
                float(workspace[0] - body[0]), float(workspace[1] - body[1]),
                float(workspace[4] / max(body[4], 1e-6)),
                float(left[0] - right[0]), float(left[1] - right[1]),
                float(np.linalg.norm(left[:2] - right[:2])),
                float(left[0] - workspace[0]), float(left[1] - workspace[1]),
                float(right[0] - workspace[0]), float(right[1] - workspace[1]),
                float(cache["pose_quality_factor"][frame_index]),
                float(cache["left_right_ambiguous"][frame_index]),
            )
        )

        # Stats layout is valid_fraction, mean, std, q10, median, q90.
        body_depth = surface["full_body"]
        left_depth = surface["left_hand"]
        right_depth = surface["right_hand"]
        workspace_depth = surface["hand_workspace"]
        hand_mean = 0.5 * (left_depth + right_depth)
        depth_values.extend(
            (
                float(left_depth[1] - right_depth[1]),
                float(left_depth[4] - right_depth[4]),
                float(left_depth[1] - body_depth[1]),
                float(left_depth[4] - body_depth[4]),
                float(right_depth[1] - body_depth[1]),
                float(right_depth[4] - body_depth[4]),
                float(workspace_depth[1] - body_depth[1]),
                float(workspace_depth[4] - body_depth[4]),
                float(hand_mean[1] - workspace_depth[1]),
                float(hand_mean[4] - workspace_depth[4]),
            )
        )
        geometry_rows.append(np.asarray(geometry, dtype=np.float32))
        depth_rows.append(np.asarray(depth_values, dtype=np.float32))
    geometry_array = np.stack(geometry_rows)
    depth_array = np.stack(depth_rows)
    if geometry_array.shape[1] != len(geometry_names):
        raise RuntimeError("D1 geometry feature names do not match values")
    if depth_array.shape[1] != len(depth_names):
        raise RuntimeError("D1 depth feature names do not match values")
    return geometry_array, geometry_names, depth_array, depth_names


def temporal_summary(
    values: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] != len(feature_names):
        raise ValueError("temporal feature matrix is invalid")
    quantiles = np.quantile(values, (0.10, 0.50, 0.90), axis=0)
    if len(values) > 1:
        delta = np.abs(np.diff(values, axis=0))
        mean_delta = delta.mean(axis=0)
        max_delta = delta.max(axis=0)
        time = np.linspace(-1.0, 1.0, len(values), dtype=np.float32)
        slope = (time[:, None] * values).sum(axis=0) / max(float(np.square(time).sum()), 1e-6)
    else:
        mean_delta = max_delta = slope = np.zeros(values.shape[1], dtype=np.float32)
    phases = np.array_split(np.arange(len(values)), 3)
    global_mean = values.mean(axis=0)
    phase_means = [
        values[index].mean(axis=0) if len(index) else global_mean
        for index in phases
    ]
    statistics = (
        global_mean, values.std(axis=0), quantiles[0], quantiles[1], quantiles[2],
        mean_delta, max_delta, slope, phase_means[0], phase_means[1], phase_means[2],
    )
    output = np.concatenate(statistics).astype(np.float32)
    names = [
        f"{statistic}_{feature}"
        for statistic in TEMPORAL_STATISTICS
        for feature in feature_names
    ]
    return output, names


def load_depth_ranks(row: dict[str, str], frame_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    paths = frame_map(Path(row["depth_color_path"]), "depth")
    missing = [str(frame_id) for frame_id in frame_ids if str(frame_id) not in paths]
    if missing:
        raise FileNotFoundError(f"missing synchronized Depth frame {missing[0]}")
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_id in frame_ids:
        image = cv2.imread(str(paths[str(frame_id)]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read Depth frame {paths[str(frame_id)]}")
        result.append(depth_color_to_rank(image))
    return result


def extract_trial(row: dict[str, str], roi_path: Path, regions: tuple[str, ...]) -> dict[str, Any]:
    with np.load(roi_path, allow_pickle=False) as source:
        cache = {key: np.asarray(source[key]) for key in source.files}
    depth_ranks = load_depth_ranks(row, cache["frame_ids"])
    geometry, geometry_frame_names, depth, depth_frame_names = frame_descriptors(
        cache, depth_ranks, regions
    )
    geometry_summary, geometry_names = temporal_summary(geometry, geometry_frame_names)
    depth_summary, depth_names = temporal_summary(depth, depth_frame_names)
    return {
        "geometry": geometry_summary,
        "geometry_names": geometry_names,
        "depth_surface": depth_summary,
        "depth_surface_names": depth_names,
        "frames": len(cache["frame_ids"]),
        "both_hands_valid_rate": float(
            (
                cache["roi_valid"][:, list(cache["region_names"].astype(str)).index("left_hand")]
                & cache["roi_valid"][:, list(cache["region_names"].astype(str)).index("right_hand")]
            ).mean()
        ),
        "valid_depth_rate": float(np.mean([valid.mean() for _, valid in depth_ranks])),
    }


def save_descriptor_artifact(path: Path, arrays: dict[str, np.ndarray]) -> None:
    forbidden = {"label", "labels", "class_id", "class_ids"}
    if forbidden & set(arrays):
        raise ValueError("ground-truth fields cannot enter a D1 descriptor artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as written:
        if forbidden & set(written.files):
            raise RuntimeError("ground truth leaked into D1 descriptor artifact")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    config_sha256 = canonical_hash(config)
    if args.stage == "h2_confirmation":
        if args.h1_summary is None:
            raise ValueError("H2 descriptor extraction requires --h1-summary")
        frozen = json.loads(args.h1_summary.resolve().read_text(encoding="utf-8"))
        if frozen.get("stage") != "P99_D1_H1" or frozen.get("config_sha256") != config_sha256:
            raise ValueError("D1 H1 summary/config mismatch")
        selected_users = set(config["cohorts"]["source_only_users"])
        selected_users.update(config["cohorts"]["exploration_users"])
        selected_users.update(config["cohorts"]["confirmation_users"])
    else:
        if args.h1_summary is not None:
            raise ValueError("--h1-summary is only valid for H2 extraction")
        selected_users = set(config["cohorts"]["source_only_users"])
        selected_users.update(config["cohorts"]["exploration_users"])

    manifest_path = resolve(config["manifest"])
    roi_run = resolve(config["roi_run"])
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["user_id"] in selected_users and row["depth_color_usable"] == "1"
        ]
    rows.sort(key=lambda row: (int(row["class_id"]), row["user_id"], row["trial_id"]))
    rows = [
        row for row in rows
        if (roi_run / "trial_roi_cache" / safe_name(row["sample_id"]).with_suffix(".npz")).is_file()
    ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no D1 descriptor rows selected")

    sample_ids: list[str] = []
    users: list[str] = []
    geometry: list[np.ndarray] = []
    depth_surface: list[np.ndarray] = []
    trial_audit: list[dict[str, Any]] = []
    geometry_names: list[str] | None = None
    depth_names: list[str] | None = None
    regions = tuple(map(str, config["descriptor"]["regions"]))
    for index, row in enumerate(rows, start=1):
        roi_path = roi_run / "trial_roi_cache" / safe_name(row["sample_id"]).with_suffix(".npz")
        value = extract_trial(row, roi_path, regions)
        if geometry_names is None:
            geometry_names = value["geometry_names"]
            depth_names = value["depth_surface_names"]
        elif geometry_names != value["geometry_names"] or depth_names != value["depth_surface_names"]:
            raise RuntimeError("D1 descriptor schema changed between trials")
        sample_ids.append(canonical_sample_id(row))
        users.append(row["user_id"])
        geometry.append(value["geometry"])
        depth_surface.append(value["depth_surface"])
        trial_audit.append(
            {
                "frames": int(value["frames"]),
                "both_hands_valid_rate": float(value["both_hands_valid_rate"]),
                "valid_depth_rate": float(value["valid_depth_rate"]),
            }
        )
        if index == 1 or index % 50 == 0 or index == len(rows):
            print(f"D1 descriptor {index}/{len(rows)} {sample_ids[-1]}", flush=True)

    output = args.output_dir.resolve()
    artifact_path = output / "descriptors.npz"
    save_descriptor_artifact(
        artifact_path,
        {
            "sample_ids": np.asarray(sample_ids),
            "users": np.asarray(users),
            "geometry": np.stack(geometry).astype(np.float32),
            "geometry_feature_names": np.asarray(geometry_names),
            "depth_surface": np.stack(depth_surface).astype(np.float32),
            "depth_surface_feature_names": np.asarray(depth_names),
        },
    )
    summary = {
        "stage": "P99_D1_label_free_descriptor_H1" if args.stage == "h1" else "P99_D1_label_free_descriptor_H2",
        "status": "complete",
        "config_sha256": config_sha256,
        "config_path": str(args.config.resolve()),
        "artifact": str(artifact_path),
        "rows": len(sample_ids),
        "users": sorted(set(users)),
        "labels_written": False,
        "geometry_dim": int(geometry[0].shape[0]),
        "depth_surface_dim": int(depth_surface[0].shape[0]),
        "total_frames": int(sum(row["frames"] for row in trial_audit)),
        "mean_both_hands_valid_rate": float(np.mean([row["both_hands_valid_rate"] for row in trial_audit])),
        "mean_valid_depth_rate": float(np.mean([row["valid_depth_rate"] for row in trial_audit])),
        "h2_h3_labels_accessed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
