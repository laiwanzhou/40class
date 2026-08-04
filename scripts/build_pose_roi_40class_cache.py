from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pose_roi_dataset import depth_frame_key, paired_frame_key, paired_frame_paths
from src.roi.pose_locator import UltralyticsPoseLocator


DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/depth_ir_pose_roi_40class_probe/pose_tracks.npz"
OLD_CACHE = PROJECT_ROOT / "outputs/depth_pose_roi_probe/pose_tracks.npz"
QUALITY_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_pose_quality.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolo11n-pose.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=8192)
    return parser.parse_args()


def read_ir_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode audited IR frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def load_detection_lookup(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]]:
    if not path.is_file():
        return {}
    with np.load(path) as data:
        sample_ids = data["sample_ids"].astype(str)
        frame_keys = data["frame_keys"].astype(str)
        processed = data["processed"].astype(bool) if "processed" in data else np.ones(len(sample_ids), dtype=bool)
        bbox_confidence = data["bbox_confidence"] if "bbox_confidence" in data else np.zeros(len(sample_ids), dtype=np.float32)
        return {
            (sample_id, frame_key): (
                data["bbox_xyxy"][index].astype(np.float32),
                float(bbox_confidence[index]),
                data["keypoints_xy"][index].astype(np.float32),
                data["keypoints_confidence"][index].astype(np.float32),
                bool(processed[index]),
            )
            for index, (sample_id, frame_key) in enumerate(zip(sample_ids, frame_keys, strict=True))
        }


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(PROJECT_ROOT / "metadata/manifest.csv", encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    audit = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_pairing.csv", encoding="utf-8-sig")
    valid_ids = set(audit.loc[audit["complete_pairing"], "sample_id"].astype(str))
    split_by_sample = dict(zip(audit["sample_id"].astype(str), audit["split"].astype(str), strict=True))
    selected = manifest[manifest["sample_id"].isin(valid_ids)].sort_values(["class_id", "sample_id"])
    records: list[dict[str, object]] = []
    for row in selected.itertuples():
        depth_paths, ir_paths = paired_frame_paths(DATA_ROOT / row.depth_color_path, DATA_ROOT / row.ir_path)
        with Image.open(depth_paths[0]) as image:
            width, height = image.size
        for depth_path, ir_path in zip(depth_paths, ir_paths, strict=True):
            timestamp, frame_id = paired_frame_key(depth_path, "Depth")
            records.append(
                {
                    "sample_id": str(row.sample_id),
                    "frame_key": depth_frame_key(depth_path),
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "width": width,
                    "height": height,
                    "class_id": int(row.class_id),
                    "action_name": str(row.action_name),
                    "split": split_by_sample[str(row.sample_id)],
                    "ir_path": ir_path,
                }
            )
    count = len(records)
    if count != int(audit.loc[audit["complete_pairing"], "paired_frames"].sum()):
        raise ValueError("Cache record count differs from the strict pairing audit.")
    bbox = np.full((count, 4), np.nan, dtype=np.float32)
    bbox_confidence = np.zeros(count, dtype=np.float32)
    keypoints = np.full((count, 17, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((count, 17), dtype=np.float32)
    processed = np.zeros(count, dtype=bool)
    existing = load_detection_lookup(args.output)
    legacy = load_detection_lookup(OLD_CACHE)
    reused_current = 0
    reused_legacy = 0
    for index, record in enumerate(records):
        key = str(record["sample_id"]), str(record["frame_key"])
        value = existing.get(key)
        source = "current"
        if value is None:
            value = legacy.get(key)
            source = "legacy"
        if value is None or not value[4]:
            continue
        bbox[index], bbox_confidence[index], keypoints[index], confidence[index], processed[index] = value
        reused_current += source == "current"
        reused_legacy += source == "legacy"
    print(json.dumps({"records": count, "resumed": reused_current, "reused_11class": reused_legacy, "missing": int((~processed).sum())}), flush=True)

    def save_cache() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            sample_ids=np.asarray([record["sample_id"] for record in records]),
            frame_keys=np.asarray([record["frame_key"] for record in records]),
            timestamps=np.asarray([record["timestamp"] for record in records]),
            frame_ids=np.asarray([record["frame_id"] for record in records], dtype=np.int64),
            original_width=np.asarray([record["width"] for record in records], dtype=np.int32),
            original_height=np.asarray([record["height"] for record in records], dtype=np.int32),
            class_ids=np.asarray([record["class_id"] for record in records], dtype=np.int64),
            action_names=np.asarray([record["action_name"] for record in records]),
            splits=np.asarray([record["split"] for record in records]),
            bbox_xyxy=bbox,
            bbox_confidence=bbox_confidence,
            keypoints_xy=keypoints,
            keypoints_confidence=confidence,
            processed=processed,
        )

    missing_indices = np.flatnonzero(~processed)
    if len(missing_indices):
        locator = UltralyticsPoseLocator(args.weights)
        since_checkpoint = 0
        for start in range(0, len(missing_indices), args.batch_size):
            indices = missing_indices[start : start + args.batch_size]
            detections = locator.predict([read_ir_rgb(Path(records[int(index)]["ir_path"])) for index in indices])
            for index, detection in zip(indices, detections, strict=True):
                processed[index] = True
                if detection is not None:
                    bbox[index] = detection.bbox_xyxy
                    bbox_confidence[index] = detection.bbox_confidence
                    keypoints[index] = detection.keypoints_xy
                    confidence[index] = detection.keypoints_confidence
            since_checkpoint += len(indices)
            completed = start + len(indices)
            if since_checkpoint >= args.checkpoint_interval or completed == len(missing_indices):
                save_cache()
                since_checkpoint = 0
                print(f"pose cache new frames {completed}/{len(missing_indices)}; total processed {int(processed.sum())}/{count}", flush=True)
    else:
        save_cache()
    if not processed.all():
        raise RuntimeError("Pose cache contains unprocessed records.")
    class_ids = np.asarray([record["class_id"] for record in records])
    action_names = np.asarray([record["action_name"] for record in records])
    rows = []
    for class_id in range(40):
        selected_frames = class_ids == class_id
        rows.append(
            {
                "class_id": class_id,
                "action_name": str(action_names[selected_frames][0]),
                "frames": int(selected_frames.sum()),
                "person_success": float(np.isfinite(bbox[selected_frames]).all(axis=1).mean()),
                "left_wrist_success": float((confidence[selected_frames, 9] >= 0.25).mean()),
                "right_wrist_success": float((confidence[selected_frames, 10] >= 0.25).mean()),
                "at_least_one_wrist_success": float(((confidence[selected_frames, 9] >= 0.25) | (confidence[selected_frames, 10] >= 0.25)).mean()),
            }
        )
    pd.DataFrame(rows).to_csv(QUALITY_PATH, index=False, encoding="utf-8-sig")
    summary = {
        "frames": count,
        "samples": len(selected),
        "person_success": float(np.isfinite(bbox).all(axis=1).mean()),
        "left_wrist_success": float((confidence[:, 9] >= 0.25).mean()),
        "right_wrist_success": float((confidence[:, 10] >= 0.25).mean()),
        "at_least_one_wrist_success": float(((confidence[:, 9] >= 0.25) | (confidence[:, 10] >= 0.25)).mean()),
        "reused_11class_frames": reused_legacy,
        "newly_inferred_frames": len(missing_indices),
    }
    (args.output.parent / "pose_cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
