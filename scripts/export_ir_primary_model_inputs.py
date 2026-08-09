from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_ir_primary_depth_residual_fullseq import build_datasets, resolve_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_primary_depth_residual_fullseq.yaml"
DEFAULT_OUTPUT = Path(
    r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_model_inputs_256"
)
IR_VIEWS = (
    (0, "ir_context"),
    (1, "ir_left"),
    (2, "ir_right"),
    (3, "ir_relation"),
)
DEPTH_VIEWS = ((0, "depth_context"), (3, "depth_relation"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the exact 256x256 IR/Depth crops used by the full-sequence model.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--png-compress-level", type=int, default=3, choices=range(0, 10))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def crop_input(
    source: Path,
    box: np.ndarray,
    valid: bool,
    mode: str,
    image_size: int,
) -> Image.Image:
    if not valid:
        return Image.new(mode, (image_size, image_size), color=0)
    with Image.open(source) as opened:
        image = opened.convert(mode)
        crop = image.crop(tuple(float(value) for value in box))
    fill: int | tuple[int, int, int] = 0 if mode == "L" else (0, 0, 0)
    return ImageOps.pad(
        crop,
        (image_size, image_size),
        method=Image.Resampling.LANCZOS,
        color=fill,
    )


def save_png(image: Image.Image, path: Path, compress_level: int) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    image.save(temporary, format="PNG", compress_level=compress_level)
    os.replace(temporary, path)


def read_recovery_metadata(pose_cache: Path) -> dict[str, Any]:
    summary_path = pose_cache.parent / "person_crop_pose_cache_summary.json"
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    image_size = int(config["image_size"])
    pose_cache = resolve_path(config["pose_cache"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "_SUCCESS").exists():
        raise FileExistsError(f"Completed export already exists: {output}")

    train_dataset, val_dataset = build_datasets(config)
    datasets = (("train", train_dataset), ("val", val_dataset))
    input_partial = output / "all_frame_inputs.csv.partial"
    roi_partial = output / "roi_frame_audit.csv.partial"
    input_header = [
        "split", "class_id", "action_name", "sample_id", "user_id", "source_frame_index",
        *(f"{name}_path" for _, name in (*IR_VIEWS, *DEPTH_VIEWS)),
        *(f"{name}_valid" for _, name in (*IR_VIEWS, *DEPTH_VIEWS)),
        *(f"{name}_source" for _, name in (*IR_VIEWS, *DEPTH_VIEWS)),
        *(f"{name}_confidence" for _, name in IR_VIEWS),
    ]
    roi_header = [
        "split", "class_id", "action_name", "sample_id", "user_id", "source_frame_index",
        "view_name", "modality", "valid", "source", "confidence", "x1", "y1", "x2", "y2",
        "output_path",
    ]
    action_counts: dict[tuple[str, int, str], dict[str, int]] = {}
    total_samples = 0
    unique_frames = 0
    output_images = 0
    started = time.perf_counter()

    with input_partial.open("w", newline="", encoding="utf-8-sig") as input_file, roi_partial.open(
        "w", newline="", encoding="utf-8-sig",
    ) as roi_file:
        input_writer = csv.DictWriter(input_file, fieldnames=input_header)
        roi_writer = csv.DictWriter(roi_file, fieldnames=roi_header)
        input_writer.writeheader()
        roi_writer.writeheader()
        for split, dataset in datasets:
            for sample in dataset.samples:
                if args.limit_samples is not None and total_samples >= args.limit_samples:
                    break
                paths = sample["paths"]
                ir_paths = sample["ir_paths"]
                roi = sample["input_roi"]
                confidence = sample["roi_confidence"]
                if not isinstance(paths, tuple) or not isinstance(ir_paths, tuple):
                    raise TypeError("Invalid paired paths")
                class_id = int(sample["original_class_id"])
                action_name = str(dataset.class_names[int(sample["label"])])
                sample_id = str(sample["sample_id"])
                user_id = str(sample["user_id"])
                selected_frames = np.arange(len(paths), dtype=np.int64)
                sample_root = output / split / f"c{class_id:02d}" / sample_id
                path_lookup: dict[tuple[int, str], str] = {}
                for frame in selected_frames:
                    frame_index = int(frame)
                    for view, name in IR_VIEWS:
                        destination = sample_root / name / f"f{frame_index:04d}.png"
                        valid = bool(roi.valid_mask[frame_index, view])
                        save_png(
                            crop_input(
                                ir_paths[frame_index], roi.boxes[frame_index, view], valid, "L", image_size,
                            ),
                            destination,
                            args.png_compress_level,
                        )
                        path_lookup[(frame_index, name)] = relative(destination, output)
                        box = roi.boxes[frame_index, view]
                        roi_writer.writerow({
                            "split": split, "class_id": class_id, "action_name": action_name,
                            "sample_id": sample_id, "user_id": user_id,
                            "source_frame_index": frame_index, "view_name": name, "modality": "IR",
                            "valid": int(valid), "source": str(roi.sources[frame_index, view]),
                            "confidence": float(confidence[frame_index, view]),
                            "x1": float(box[0]), "y1": float(box[1]),
                            "x2": float(box[2]), "y2": float(box[3]),
                            "output_path": relative(destination, output),
                        })
                        output_images += 1
                    for view, name in DEPTH_VIEWS:
                        destination = sample_root / name / f"f{frame_index:04d}.png"
                        valid = bool(roi.valid_mask[frame_index, view])
                        save_png(
                            crop_input(
                                paths[frame_index], roi.boxes[frame_index, view], valid, "RGB", image_size,
                            ),
                            destination,
                            args.png_compress_level,
                        )
                        path_lookup[(frame_index, name)] = relative(destination, output)
                        box = roi.boxes[frame_index, view]
                        roi_writer.writerow({
                            "split": split, "class_id": class_id, "action_name": action_name,
                            "sample_id": sample_id, "user_id": user_id,
                            "source_frame_index": frame_index, "view_name": name, "modality": "Depth_Color",
                            "valid": int(valid), "source": str(roi.sources[frame_index, view]),
                            "confidence": float(confidence[frame_index, view]),
                            "x1": float(box[0]), "y1": float(box[1]),
                            "x2": float(box[2]), "y2": float(box[3]),
                            "output_path": relative(destination, output),
                        })
                        output_images += 1
                    input_writer.writerow({
                        "split": split, "class_id": class_id, "action_name": action_name,
                        "sample_id": sample_id, "user_id": user_id,
                        "source_frame_index": frame_index,
                        **{
                            f"{name}_path": path_lookup[(frame_index, name)]
                            for _, name in (*IR_VIEWS, *DEPTH_VIEWS)
                        },
                        **{
                            f"{name}_valid": int(bool(roi.valid_mask[frame_index, view]))
                            for view, name in (*IR_VIEWS, *DEPTH_VIEWS)
                        },
                        **{
                            f"{name}_source": str(roi.sources[frame_index, view])
                            for view, name in (*IR_VIEWS, *DEPTH_VIEWS)
                        },
                        **{
                            f"{name}_confidence": float(confidence[frame_index, view])
                            for view, name in IR_VIEWS
                        },
                    })
                key = split, class_id, action_name
                counts = action_counts.setdefault(key, {"samples": 0, "unique_frames": 0, "images": 0})
                counts["samples"] += 1
                counts["unique_frames"] += len(selected_frames)
                counts["images"] += len(selected_frames) * 6
                total_samples += 1
                unique_frames += len(selected_frames)
                if total_samples % 25 == 0:
                    elapsed = time.perf_counter() - started
                    print(json.dumps({
                        "samples": total_samples, "unique_frames": unique_frames,
                        "images": output_images, "seconds": round(elapsed, 1),
                    }), flush=True)
            if args.limit_samples is not None and total_samples >= args.limit_samples:
                break

    os.replace(input_partial, output / "all_frame_inputs.csv")
    os.replace(roi_partial, output / "roi_frame_audit.csv")
    summary_rows = [
        {
            "split": split, "class_id": class_id, "action_name": action,
            **counts,
        }
        for (split, class_id, action), counts in sorted(action_counts.items())
    ]
    with (output / "export_summary_by_action.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(
            target, fieldnames=["split", "class_id", "action_name", "samples", "unique_frames", "images"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    recovery = read_recovery_metadata(pose_cache)
    metadata = {
        "status": "complete",
        "config": str(args.config.resolve()),
        "pose_cache": str(pose_cache),
        "pose_cache_sha256": sha256(pose_cache),
        "pose_detector_image_size": 640,
        "base_pose_detection_confidence": 0.25,
        "recovery_detection_confidence": recovery.get("recovery_detection_confidence"),
        "recovery_trial_count": recovery.get("recovery_trial_count", 0),
        "recovered_frames": recovery.get("recovered_frames", 0),
        "keypoint_confidence_threshold": float(config["roi"]["keypoint_threshold"]),
        "image_size": image_size,
        "samples": total_samples,
        "unique_source_frames": unique_frames,
        "output_images": output_images,
        "ir_views": [name for _, name in IR_VIEWS],
        "depth_views": [name for _, name in DEPTH_VIEWS],
        "all_original_paired_frames_exported": True,
        "temporal_sampling_applied": False,
        "invalid_views_are_black_preview_images": True,
        "loader_requirement": "After image normalization, set invalid views to an all-zero tensor using *_valid.",
        "test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output / "_SUCCESS").write_text("complete\n", encoding="ascii")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
