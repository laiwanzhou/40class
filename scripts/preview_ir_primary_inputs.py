from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.data.pose_roi_dataset import PoseTrackCache, depth_frame_key, paired_frame_paths
from src.roi.ir_primary_input_builder import IRPrimaryInputROIBuilder, VIEW_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_SOURCE = DATA_ROOT.parent / "yulan" / "all_preview_samples.csv"
DEFAULT_OUTPUT = DATA_ROOT.parent / "yulan2"
POSE_CACHE_CANDIDATES = (
    PROJECT_ROOT / "outputs/depth_ir_person_crop_40class_fold0/person_crop_pose_tracks.npz",
    PROJECT_ROOT.parent / "40class/outputs/depth_ir_person_crop_40class_fold0/person_crop_pose_tracks.npz",
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pose-cache", type=Path)
    return parser.parse_args()


def resolve_pose_cache(value: Path | None) -> Path:
    if value is not None:
        if not value.exists():
            raise FileNotFoundError(value)
        return value
    for candidate in POSE_CACHE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Pose cache not found in: {POSE_CACHE_CANDIDATES}")


def contains_box(container: np.ndarray, item: np.ndarray) -> bool:
    return bool(
        np.isfinite(item).all()
        and item[0] >= container[0] - 1e-4
        and item[1] >= container[1] - 1e-4
        and item[2] <= container[2] + 1e-4
        and item[3] <= container[3] + 1e-4
    )


def crop_letterbox(path: Path, box: np.ndarray, mode: str, size: int = 256) -> Image.Image:
    with Image.open(path) as opened:
        crop = opened.convert(mode).crop(tuple(float(value) for value in box))
    fill = 0 if mode == "L" else (0, 0, 0)
    return ImageOps.pad(crop, (size, size), method=Image.Resampling.LANCZOS, color=fill)


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def save_input_grid(
    rows: list[dict[str, object]],
    view_indices: tuple[int, ...],
    path_key: str,
    mode: str,
    output: Path,
) -> None:
    figure, axes = plt.subplots(len(rows), len(view_indices), figsize=(3.0 * len(view_indices), 2.7 * len(rows)))
    axes = np.asarray(axes).reshape(len(rows), len(view_indices))
    for row_index, row in enumerate(rows):
        result = row["roi"]
        frame_index = int(row["frame_index"])
        for column, view_index in enumerate(view_indices):
            axis = axes[row_index, column]
            if bool(result.valid_mask[frame_index, view_index]):
                image = crop_letterbox(row[path_key], result.boxes[frame_index, view_index], mode)
                axis.imshow(image, cmap="gray" if mode == "L" else None)
            else:
                axis.imshow(np.zeros((256, 256)), cmap="gray", vmin=0, vmax=255)
            axis.set_title(
                f"{VIEW_NAMES[view_index]}\n{result.sources[frame_index, view_index]}", fontsize=8,
            )
            if column == 0:
                axis.set_ylabel(f"{row['user_id']} | p{row['progress_slot']}", fontsize=8)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(figure)


def save_overlay(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(12, 9))
    colors = ("lime", "cyan", "yellow", "magenta")
    for axis, row in zip(axes.flat, rows, strict=True):
        frame_index = int(row["frame_index"])
        with Image.open(row["ir_path"]) as opened:
            image = opened.convert("L")
        axis.imshow(image, cmap="gray")
        result = row["roi"]
        for view, color in enumerate(colors):
            if not result.valid_mask[frame_index, view]:
                continue
            x1, y1, x2, y2 = result.boxes[frame_index, view]
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1.5))
        axis.set_title(f"{row['user_id']} | progress {row['progress_slot']}", fontsize=9)
        axis.axis("off")
    figure.suptitle("green=context, cyan=left, yellow=right, magenta=adaptive", fontsize=11)
    figure.tight_layout()
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    config = args()
    source = pd.read_csv(config.source, encoding="utf-8-sig")
    manifest = pd.read_csv(PROJECT_ROOT / "metadata/manifest.csv", encoding="utf-8-sig")
    manifest = manifest.set_index("sample_id", drop=False)
    pose_cache = resolve_pose_cache(config.pose_cache)
    cache = PoseTrackCache(pose_cache)
    builder = IRPrimaryInputROIBuilder(
        interaction_config={
            "projection_alpha": 0.55,
            "long_axis_scale": 0.65,
            "short_axis_scale": 0.45,
            "two_hand_horizontal_padding": 0.30,
            "two_hand_bottom_padding": 0.45,
        },
    )
    config.output.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for (class_id, action_name), action_samples in source.groupby(["class_id", "action_name"], sort=True):
        action_dir = config.output / f"{int(class_id):02d}_{action_name}"
        action_dir.mkdir(parents=True, exist_ok=True)
        preview_rows: list[dict[str, object]] = []
        for sample_record in action_samples.to_dict(orient="records"):
            sample_id = str(sample_record["sample_id"])
            row = manifest.loc[sample_id]
            depth_dir = DATA_ROOT.joinpath(*Path(str(row["depth_color_path"]).replace("\\", "/")).parts)
            ir_dir = DATA_ROOT.joinpath(*Path(str(row["ir_path"]).replace("\\", "/")).parts)
            depth_paths, ir_paths = paired_frame_paths(depth_dir, ir_dir)
            with Image.open(depth_paths[0]) as opened:
                width, height = opened.size
            keys = [depth_frame_key(path) for path in depth_paths]
            person, keypoints, confidence = cache.trial_arrays(sample_id, keys)
            result = builder.build(person, keypoints, confidence, width, height)
            indices = [int(value) for value in str(sample_record["selected_depth_indices"]).split()]
            for slot, frame_index in enumerate(indices, start=1):
                preview_rows.append({
                    "sample_id": sample_id,
                    "user_id": str(sample_record["user_id"]),
                    "split": str(sample_record["split"]),
                    "progress_slot": slot,
                    "frame_index": frame_index,
                    "depth_path": depth_paths[frame_index],
                    "ir_path": ir_paths[frame_index],
                    "roi": result,
                })
            area = (result.boxes[..., 2] - result.boxes[..., 0]) * (result.boxes[..., 3] - result.boxes[..., 1])
            image_area = float(width * height)
            person_valid = np.isfinite(person).all(axis=1)
            person_contained = [
                contains_box(result.boxes[frame, 0], person[frame])
                for frame in range(len(person)) if person_valid[frame]
            ]
            keypoint_valid = (confidence >= 0.25) & np.isfinite(keypoints).all(axis=-1)
            keypoint_frames = []
            for frame in range(len(keypoints)):
                points = keypoints[frame, keypoint_valid[frame]]
                if not len(points):
                    continue
                context = result.boxes[frame, 0]
                keypoint_frames.append(bool(
                    (points[:, 0] >= context[0] - 1e-4).all()
                    and (points[:, 0] <= context[2] + 1e-4).all()
                    and (points[:, 1] >= context[1] - 1e-4).all()
                    and (points[:, 1] <= context[3] + 1e-4).all()
                ))
            audit_rows.append({
                "class_id": int(class_id),
                "action_name": action_name,
                "sample_id": sample_id,
                "user_id": sample_record["user_id"],
                "split": sample_record["split"],
                "frames": len(depth_paths),
                "context_area_ratio_mean": float(area[:, 0].mean() / image_area),
                "context_boundary_touch_rate": float(result.context_touches_boundary.mean()),
                "person_bbox_contained_rate": float(np.mean(person_contained)) if person_contained else np.nan,
                "valid_keypoints_contained_rate": float(np.mean(keypoint_frames)) if keypoint_frames else np.nan,
                "left_valid_rate": float(result.valid_mask[:, 1].mean()),
                "right_valid_rate": float(result.valid_mask[:, 2].mean()),
                "adaptive_valid_rate": float(result.valid_mask[:, 3].mean()),
                "left_right_duplicate_rate_before_suppression": float(np.nanmean(result.left_right_iou >= 0.70)),
                "hand_head_rate": float(np.mean(result.sources[:, 3] == "hand_head_union")),
                "two_hand_table_rate": float(np.mean(result.sources[:, 3] == "two_hand_table_context")),
            })
            sample_rows.append({
                "class_id": int(class_id), "action_name": action_name, "sample_id": sample_id,
                "user_id": sample_record["user_id"], "split": sample_record["split"],
                "selected_indices": " ".join(str(index) for index in indices),
            })
        if len(preview_rows) != 9:
            raise ValueError(f"Expected 9 preview rows for {action_name}, got {len(preview_rows)}")
        save_overlay(preview_rows, action_dir / "roi_overlay_ir.png")
        save_input_grid(preview_rows, (0, 1, 2, 3), "ir_path", "L", action_dir / "ir_primary_inputs.png")
        save_input_grid(preview_rows, (0, 3), "depth_path", "RGB", action_dir / "depth_geometry_inputs.png")
        pd.DataFrame([row for row in sample_rows if row["action_name"] == action_name]).to_csv(
            action_dir / "samples.csv", index=False, encoding="utf-8-sig",
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(config.output / "roi_audit_samples.csv", index=False, encoding="utf-8-sig")
    summary = audit.groupby(["class_id", "action_name"], as_index=False).agg({
        "context_area_ratio_mean": "mean",
        "context_boundary_touch_rate": "mean",
        "person_bbox_contained_rate": "mean",
        "valid_keypoints_contained_rate": "mean",
        "left_valid_rate": "mean",
        "right_valid_rate": "mean",
        "adaptive_valid_rate": "mean",
        "left_right_duplicate_rate_before_suppression": "mean",
        "hand_head_rate": "mean",
        "two_hand_table_rate": "mean",
    })
    summary.to_csv(config.output / "roi_audit_by_action.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sample_rows).to_csv(config.output / "preview_samples.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "actions": int(summary.shape[0]),
        "samples": int(audit.shape[0]),
        "preview_frames": int(audit.shape[0] * 3),
        "ir_views": list(VIEW_NAMES),
        "depth_views": [VIEW_NAMES[0], VIEW_NAMES[3]],
        "pose_cache": str(pose_cache),
        "test_read": False,
    }
    (config.output / "preview_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
