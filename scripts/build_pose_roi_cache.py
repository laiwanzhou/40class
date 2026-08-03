from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.roi.pose_locator import UltralyticsPoseLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/depth_pose_roi_expert.yaml")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolo11n-pose.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/depth_pose_roi_probe/pose_tracks.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def frame_key(path: Path, prefix: str) -> str:
    if not path.stem.startswith(prefix):
        raise ValueError(f"Unexpected frame name: {path.name}")
    return path.stem[len(prefix) :].removesuffix("_Color")


def read_ir_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read IR frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = pd.read_csv(resolve(config["manifest"]), encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    fold = json.loads(resolve(config["fold"]).read_text(encoding="utf-8"))
    users = set(fold["train_users"]) | set(fold["val_users"])
    hard_actions = list(config["hard_actions"])
    selected = manifest[
        manifest["action_name"].isin(hard_actions)
        & manifest["user_id"].isin(users)
        & manifest["depth_color_path"].fillna("").ne("")
        & manifest["ir_path"].fillna("").ne("")
    ].copy()
    if set(selected["action_name"]) != set(hard_actions):
        raise ValueError("Hard action selection is incomplete.")
    data_root = Path(config["data_root"])
    records: list[tuple[str, str, Path]] = []
    for row in selected.itertuples():
        depth_dir = data_root / row.depth_color_path
        ir_dir = data_root / row.ir_path
        depth_keys = {frame_key(path, "Depth_") for path in depth_dir.glob("*.png")}
        ir = {frame_key(path, "IR_"): path for path in ir_dir.glob("*.png")}
        keys = sorted(depth_keys & ir.keys())
        if not keys:
            raise ValueError(f"No Depth/IR pairing for {row.sample_id}")
        records.extend((row.sample_id, key, ir[key]) for key in keys)
    locator = UltralyticsPoseLocator(args.weights)
    bbox = np.full((len(records), 4), np.nan, dtype=np.float32)
    keypoints = np.full((len(records), 17, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(records), 17), dtype=np.float32)
    bbox_confidence = np.zeros(len(records), dtype=np.float32)
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        detections = locator.predict([read_ir_rgb(record[2]) for record in batch])
        for offset, detection in enumerate(detections):
            if detection is None:
                continue
            index = start + offset
            bbox[index] = detection.bbox_xyxy
            keypoints[index] = detection.keypoints_xy
            confidence[index] = detection.keypoints_confidence
            bbox_confidence[index] = detection.bbox_confidence
        if start == 0 or (start + len(batch)) % 1024 == 0 or start + len(batch) == len(records):
            print(f"pose cache {start + len(batch)}/{len(records)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray([record[0] for record in records]),
        frame_keys=np.asarray([record[1] for record in records]),
        bbox_xyxy=bbox,
        bbox_confidence=bbox_confidence,
        keypoints_xy=keypoints,
        keypoints_confidence=confidence,
    )
    split_map = {user: "train" for user in fold["train_users"]} | {user: "validation" for user in fold["val_users"]}
    class_rows = []
    for (class_id, action), group in selected.groupby(["class_id", "action_name"]):
        class_rows.append({
            "class_id": int(class_id),
            "action_name": action,
            "expert_label": sorted(selected["class_id"].unique()).index(class_id),
            "train_samples": int(group["user_id"].map(split_map).eq("train").sum()),
            "validation_samples": int(group["user_id"].map(split_map).eq("validation").sum()),
        })
    pd.DataFrame(class_rows).sort_values("expert_label").to_csv(
        PROJECT_ROOT / "reports/depth_pose_roi_hard_classes.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "frames": len(records),
        "trials": len(selected),
        "person_success": float(np.isfinite(bbox).all(axis=1).mean()),
        "left_wrist_success": float((confidence[:, 9] >= 0.25).mean()),
        "right_wrist_success": float((confidence[:, 10] >= 0.25).mean()),
        "at_least_one_wrist_success": float(((confidence[:, 9] >= 0.25) | (confidence[:, 10] >= 0.25)).mean()),
        "weight_bytes": args.weights.stat().st_size,
    }
    (args.output.parent / "pose_cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    print(args.output)


if __name__ == "__main__":
    main()
