from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from audit_yolo11_pose_skeleton import frame_map
from build_p30_shared_dir_roi_feature_cache import read_depth, read_ir
from depth_encoding import decode_jet_rgb
from p30_shared_dir_roi_model import SharedResNet18Pyramid, imagenet_normalize
from p46_event_preprocessing import (
    P46_LOCAL_REGIONS,
    device_relative_imu,
    oriented_grid_crops,
    oriented_roi_geometry,
    rotate_skeleton_cache,
    upper_body_human_prior,
)
from p46_protocol import DEFAULT_MANIFEST


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_P28 = PROJECT_DIR / "runs" / "p28_adaptive_ir_pose_skeleton_full"
DEFAULT_P29 = PROJECT_DIR / "runs" / "p29_dir_multiscale_roi_full"
DEFAULT_P31 = PROJECT_DIR / "runs" / "p31_skeleton_imu_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_event_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build P46 [1]-[6] body-normalised, oriented all-frame event inputs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p28-run", type=Path, default=DEFAULT_P28)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--p31-run", type=Path, default=DEFAULT_P31)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--frame-batch", type=int, default=8)
    parser.add_argument("--backbone-batch", type=int, default=160)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use CPU-heavy np.savez_compressed. Default np.savez is lossless and faster.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cache_path(root: Path, source_id: str) -> Path:
    parts = source_id.split("/")
    if len(parts) != 3 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe source_id: {source_id}")
    return root / "trial_event_cache" / Path(*parts).with_suffix(".npz")


def source_cache(root: Path, folder: str, source_id: str) -> Path:
    parts = source_id.split("/")
    return root / folder / Path(*parts).with_suffix(".npz")


def atomic_npz(
    path: Path, arrays: dict[str, np.ndarray], compressed: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(args.manifest.resolve())
        if row["detail_selected"] == "1"
    ]
    if args.sample_id:
        requested = set(args.sample_id)
        rows = [
            row
            for row in rows
            if row["sample_id"] in requested or row["source_id"] in requested
        ]
        found = {row["sample_id"] for row in rows} | {row["source_id"] for row in rows}
        missing = sorted(requested - found)
        if missing:
            raise KeyError(f"P46 requested samples unavailable: {missing}")
    rows.sort(
        key=lambda row: (
            0 if row["p46_split"] == "train" else 1,
            int(row["class_id"]),
            row["user_id"],
            row["trial_id"],
        )
    )
    if args.max_trials > 0:
        rows = rows[: args.max_trials]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards")
    if args.num_shards > 1:
        rows = [
            row
            for position, row in enumerate(rows)
            if position % args.num_shards == args.shard_index
        ]
    return rows


def _feature_grids(
    crops: torch.Tensor,
    backbone: SharedResNet18Pyramid,
    backbone_batch: int,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    modalities, frames, regions = crops.shape[:3]
    flattened = crops.reshape(-1, *crops.shape[3:])
    maps: list[torch.Tensor] = []
    for start in range(0, len(flattened), backbone_batch):
        source = imagenet_normalize(flattened[start : start + backbone_batch])
        with torch.autocast(
            device_type=source.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            layer2, _, _ = backbone.forward_feature_maps(source)
        maps.append(layer2.float())
    feature_map = torch.cat(maps).reshape(
        modalities, frames, regions, 128, maps[0].shape[-2], maps[0].shape[-1]
    )
    arms = F.adaptive_avg_pool2d(
        feature_map[:, :, :2].reshape(-1, 128, feature_map.shape[-2], feature_map.shape[-1]),
        3,
    ).reshape(modalities, frames, 2, 128, 3, 3)
    detail = F.adaptive_avg_pool2d(
        feature_map[:, :, 2:].reshape(-1, 128, feature_map.shape[-2], feature_map.shape[-1]),
        5,
    ).reshape(modalities, frames, 3, 128, 5, 5)
    arms = arms.permute(1, 0, 2, 4, 5, 3).cpu().numpy().astype(np.float16)
    detail = detail.permute(1, 0, 2, 4, 5, 3).cpu().numpy().astype(np.float16)
    return arms, detail


def _geometry_grid(crops: torch.Tensor) -> np.ndarray:
    # channels: ordered depth, valid depth, IR, upper-body prior
    mean = F.adaptive_avg_pool2d(crops.reshape(-1, 4, *crops.shape[-2:]), 5)
    squared = F.adaptive_avg_pool2d(
        crops[:, :, :, (0, 2)].reshape(-1, 2, *crops.shape[-2:]).square(), 5
    )
    mean = mean.reshape(crops.shape[1], crops.shape[2], 4, 5, 5)
    squared = squared.reshape(crops.shape[1], crops.shape[2], 2, 5, 5)
    depth_std = torch.sqrt((squared[:, :, 0] - mean[:, :, 0].square()).clamp_min(0.0))
    ir_std = torch.sqrt((squared[:, :, 1] - mean[:, :, 2].square()).clamp_min(0.0))
    output = torch.stack(
        (
            mean[:, :, 0],
            depth_std,
            mean[:, :, 1],
            mean[:, :, 2],
            ir_std,
            mean[:, :, 3],
        ),
        dim=-1,
    )
    return output.cpu().numpy().astype(np.float16)


@torch.inference_mode()
def build_trial(
    row: dict[str, str],
    args: argparse.Namespace,
    backbone: SharedResNet18Pyramid,
    device: torch.device,
) -> dict[str, np.ndarray]:
    source_id = row["source_id"]
    p28_path = source_cache(args.p28_run.resolve(), "trial_cache", source_id)
    p29_path = source_cache(args.p29_run.resolve(), "trial_roi_cache", source_id)
    p31_path = source_cache(args.p31_run.resolve(), "trial_motion_cache", source_id)
    with np.load(p28_path, allow_pickle=False) as data:
        p28 = {key: np.asarray(data[key]) for key in data.files}
    with np.load(p29_path, allow_pickle=False) as data:
        p29 = {key: np.asarray(data[key]) for key in data.files}
    with np.load(p31_path, allow_pickle=False) as data:
        p31 = {key: np.asarray(data[key]) for key in data.files}
    frame_ids = [str(value) for value in p28["frame_ids"]]
    if frame_ids != [str(value) for value in p29["frame_ids"]]:
        raise RuntimeError(f"P28/P29 frame mismatch: {source_id}")
    if frame_ids != [str(value) for value in p31["frame_ids"]]:
        raise RuntimeError(f"P28/P31 frame mismatch: {source_id}")

    p29_regions = tuple(str(value) for value in p29["region_names"])
    local_indices = [p29_regions.index(name) for name in P46_LOCAL_REGIONS]
    local_boxes = p29["roi_boxes_xyxy"][:, local_indices]
    local_valid = p29["roi_valid"][:, local_indices]
    geometry, angle_valid = oriented_roi_geometry(
        local_boxes,
        local_valid,
        p29["arm_joint_xy_conf_for_roi"],
        p29["arm_joint_quality_for_roi"],
        int(p29["image_width"]),
        int(p29["image_height"]),
    )
    geometry_normalised = geometry.copy()
    geometry_normalised[..., 0] /= float(p29["image_width"])
    geometry_normalised[..., 1] /= float(p29["image_height"])
    geometry_normalised[..., 2] /= float(p29["image_width"])
    geometry_normalised[..., 3] /= float(p29["image_height"])

    depth_paths = frame_map(Path(row["depth_dir"]), "depth")
    ir_paths = frame_map(Path(row["ir_dir"]), "ir")
    missing = [fid for fid in frame_ids if fid not in depth_paths or fid not in ir_paths]
    if missing:
        raise RuntimeError(f"missing synchronized D/IR frame {missing[0]}: {source_id}")
    human_prior = upper_body_human_prior(
        p28["ir_keypoints_xy_conf"],
        p28["ir_boxes_xyxy_conf"][:, :4],
        int(p29["image_width"]),
        int(p29["image_height"]),
    )
    arm_output = np.zeros((len(frame_ids), 2, 2, 3, 3, 128), dtype=np.float16)
    detail_output = np.zeros((len(frame_ids), 2, 3, 5, 5, 128), dtype=np.float16)
    geometry_output = np.zeros((len(frame_ids), 5, 5, 5, 6), dtype=np.float16)
    use_amp = device.type == "cuda"
    for start in range(0, len(frame_ids), args.frame_batch):
        stop = min(start + args.frame_batch, len(frame_ids))
        chosen = frame_ids[start:stop]
        depth_rgb = np.stack([read_depth(depth_paths[fid]) for fid in chosen])
        ir_rgb = np.stack([read_ir(ir_paths[fid]) for fid in chosen])
        pixels = torch.from_numpy(np.stack((depth_rgb, ir_rgb))).to(
            device=device, dtype=torch.float32
        )
        pixels = pixels.permute(0, 1, 4, 2, 3).div_(255.0)
        batch_geometry = torch.from_numpy(geometry[start:stop]).to(
            device=device, dtype=torch.float32
        )
        crops = oriented_grid_crops(pixels, batch_geometry, args.crop_size)
        arms, detail = _feature_grids(
            crops, backbone, args.backbone_batch, use_amp
        )
        arm_output[start:stop] = arms
        detail_output[start:stop] = detail

        decoded: list[np.ndarray] = []
        valid_depth: list[np.ndarray] = []
        for image in depth_rgb:
            depth_index, depth_mask, _ = decode_jet_rgb(image, repair_unmatched=False)
            decoded.append(depth_index.astype(np.float32) / 255.0)
            valid_depth.append(depth_mask.astype(np.float32))
        scalar = np.stack(
            (
                np.stack(decoded),
                np.stack(valid_depth),
                ir_rgb[..., 0].astype(np.float32) / 255.0,
                human_prior[start:stop].astype(np.float32) / 255.0,
            ),
            axis=1,
        )
        scalar_tensor = torch.from_numpy(scalar).to(device=device)
        scalar_crops = oriented_grid_crops(
            scalar_tensor[None], batch_geometry, args.crop_size
        )
        geometry_output[start:stop] = _geometry_grid(scalar_crops)

    skeleton = rotate_skeleton_cache(
        p31["skeleton_features"],
        p31["skeleton_feature_mask"],
        p31["skeleton_joint_mask"],
        p31["skeleton_relations"],
        p31["skeleton_relation_mask"],
        p31["frame_time_seconds"],
    )
    imu = device_relative_imu(p31["imu_values"])
    return {
        "frame_ids": p31["frame_ids"],
        "frame_time_seconds": p31["frame_time_seconds"],
        "local_region_names": np.asarray(P46_LOCAL_REGIONS),
        "arm_spatial_features": arm_output,
        "detail_spatial_features": detail_output,
        "local_geometry_features": geometry_output,
        "oriented_roi_geometry": geometry_normalised.astype(np.float32),
        "oriented_angle_valid": angle_valid,
        "roi_valid": local_valid,
        "roi_quality": p29["roi_quality"][:, local_indices].astype(np.float32),
        "roi_source": p29["roi_source"][:, local_indices],
        "roi_clipped_ratio": p29["roi_clipped_ratio"][:, local_indices].astype(np.float32),
        "pose_quality_factor": p29["pose_quality_factor"].astype(np.float32),
        "skeleton_features": skeleton["features"],
        "skeleton_feature_mask": skeleton["feature_mask"],
        "skeleton_joint_mask": p31["skeleton_joint_mask"],
        "skeleton_relations": skeleton["relations"],
        "skeleton_relation_mask": skeleton["relation_mask"],
        "skeleton_frame_quality": p31["skeleton_frame_quality"],
        "body_axes_camera": skeleton["body_axes_camera"],
        "body_axes_raw_valid": skeleton["body_axes_raw_valid"],
        "body_axes_filled": skeleton["body_axes_filled"],
        "imu_values": imu["values"],
        "imu_raw_vectors": imu["raw_vectors"],
        "imu_time_seconds": p31["imu_time_seconds"],
        "imu_frame_index": p31["imu_frame_index"],
        "imu_device_offsets": p31["imu_device_offsets"],
        "imu_interval_counts": p31["imu_interval_counts"],
        "imu_device_mask": p31["imu_device_mask"],
    }


def validate_sources(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    missing: list[str] = []
    for row in rows:
        source_id = row["source_id"]
        for root, folder in (
            (args.p28_run.resolve(), "trial_cache"),
            (args.p29_run.resolve(), "trial_roi_cache"),
            (args.p31_run.resolve(), "trial_motion_cache"),
        ):
            if not source_cache(root, folder, source_id).is_file():
                missing.append(f"{folder}:{source_id}")
    if missing:
        raise RuntimeError(f"P46 source caches missing ({len(missing)}): {missing[:5]}")


def main() -> None:
    args = parse_args()
    rows = selected_rows(args)
    if not rows:
        raise RuntimeError("no P46 detail trials selected")
    validate_sources(rows, args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    backbone = SharedResNet18Pyramid(imagenet_pretrained=True).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    config = {
        "stage": "P46_steps_0_to_6_event_inputs",
        "version": 1,
        "manifest": str(args.manifest.resolve()),
        "selected_trials": len(rows),
        "selected_train": sum(row["p46_split"] == "train" for row in rows),
        "selected_val": sum(row["p46_split"] == "val" for row in rows),
        "all_frame_policy": "every synchronized frame; no uniform/motion-only sampling",
        "skeleton_coordinates": "pelvis-centred, clip-scaled, explicit per-frame body rotation",
        "imu_coordinates": (
            "device-relative trial orientation compensation; no unsupported claim of "
            "IMU-to-Skeleton body-frame extrinsic calibration"
        ),
        "local_regions": list(P46_LOCAL_REGIONS),
        "visual_grids": {"arms": "3x3", "hands_and_workspace": "5x5"},
        "geometry_channels": [
            "ordered_depth_mean",
            "ordered_depth_std",
            "depth_valid_fraction",
            "ir_mean",
            "ir_std",
            "upper_body_prior_fraction",
        ],
        "object_policy": (
            "no hard object labels; downstream object/surface queries attend over non-body "
            "workspace cells and ordered-depth/IR geometry"
        ),
        "depth_warning": "ordered JET index is not interpreted as metric metres",
        "backbone": "shared frozen ImageNet ResNet18 layer2",
        "crop_size": int(args.crop_size),
        "frame_batch": int(args.frame_batch),
        "backbone_batch": int(args.backbone_batch),
        "npz_compressed": bool(args.compressed),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "device": str(device),
    }
    artifact_suffix = (
        "" if args.num_shards == 1 else f".shard{args.shard_index}-of-{args.num_shards}"
    )
    atomic_json(output / f"config{artifact_suffix}.json", config)
    summaries: list[dict[str, Any]] = []
    begun = time.perf_counter()
    for index, row in enumerate(rows, 1):
        target = cache_path(output, row["source_id"])
        started = time.perf_counter()
        status = "cached"
        if args.overwrite or not target.is_file():
            arrays = build_trial(row, args, backbone, device)
            atomic_npz(target, arrays, compressed=args.compressed)
            status = "built"
        with np.load(target, allow_pickle=False) as data:
            frames = len(data["frame_ids"])
            body_raw_rate = float(data["body_axes_raw_valid"].mean())
            angle_rate = float(data["oriented_angle_valid"].mean())
            imu_points = len(data["imu_values"])
        elapsed = time.perf_counter() - started
        summaries.append(
            {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "p46_split": row["p46_split"],
                "class_id": int(row["class_id"]),
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "frames": frames,
                "imu_points": imu_points,
                "body_axes_raw_valid_rate": body_raw_rate,
                "oriented_angle_valid_rate": angle_rate,
                "status": status,
                "seconds": elapsed,
                "cache_bytes": target.stat().st_size,
            }
        )
        print(
            f"P46 cache [{index}/{len(rows)}] {row['source_id']} {status} "
            f"frames={frames} {elapsed:.2f}s {target.stat().st_size / 1024**2:.2f}MiB",
            flush=True,
        )
    write_csv(output / f"trial_summary{artifact_suffix}.csv", summaries)
    built = [row for row in summaries if row["status"] == "built"]
    summary = {
        **config,
        "completed_trials": len(summaries),
        "completed_frames": sum(int(row["frames"]) for row in summaries),
        "cache_bytes": sum(int(row["cache_bytes"]) for row in summaries),
        "mean_body_axes_raw_valid_rate": float(
            np.mean([float(row["body_axes_raw_valid_rate"]) for row in summaries])
        ),
        "mean_oriented_angle_valid_rate": float(
            np.mean([float(row["oriented_angle_valid_rate"]) for row in summaries])
        ),
        "built_trials": len(built),
        "elapsed_seconds": time.perf_counter() - begun,
    }
    atomic_json(output / f"summary{artifact_suffix}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
