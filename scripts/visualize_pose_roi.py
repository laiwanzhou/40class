from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import PoseROIDataset, load_modality_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/depth_pose_roi_expert.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/depth_pose_roi_probe/roi_checks")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _, val_frame = load_modality_frames(resolve(config["manifest"]), resolve(config["fold"]), Path(config["data_root"]), config["path_column"])
    dataset = PoseROIDataset(val_frame, list(config["hard_actions"]), int(config["num_frames"]), int(config["image_size"]), False, True, resolve(config["pose_cache"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("lime", "cyan", "magenta")
    quality_rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for sample in dataset.samples:
        paths = sample["paths"]
        boxes = sample["roi_boxes"]
        sources = sample["roi_sources"]
        assert isinstance(paths, tuple) and isinstance(boxes, np.ndarray) and isinstance(sources, np.ndarray)
        for frame_index, frame_boxes in enumerate(boxes):
            with Image.open(paths[frame_index]) as opened:
                width, height = opened.size
            for view_index, (box, source) in enumerate(zip(frame_boxes, sources[frame_index], strict=True)):
                quality_rows.append({
                    "sample_id": sample["sample_id"], "view": ("upper_body", "left_hand", "right_hand")[view_index],
                    "source": source, "area_ratio": float((box[2] - box[0]) * (box[3] - box[1]) / width / height),
                })
        label = int(sample["label"])
        if label in seen:
            continue
        seen.add(label)
        middle = len(paths) // 2
        with Image.open(paths[middle]) as opened:
            image = opened.convert("RGB")
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for view_index, (box, color) in enumerate(zip(boxes[middle], colors, strict=True), start=1):
            draw.rectangle(tuple(box), outline=color, width=4)
            draw.text((box[0] + 4, box[1] + 4), f"ROI {view_index}: {sources[middle, view_index - 1]}", fill=color, stroke_width=2, stroke_fill="black")
        crops = [image.crop(tuple(box)) for box in boxes[middle]]
        panels = [image, overlay, *crops]
        thumb_height = 320
        resized = []
        for panel in panels:
            copy = panel.copy()
            copy.thumbnail((640, thumb_height))
            resized.append(copy)
        canvas = Image.new("RGB", (sum(panel.width for panel in resized), thumb_height), "white")
        x = 0
        for panel in resized:
            canvas.paste(panel, (x, 0)); x += panel.width
        canvas.save(args.output_dir / f"{sample['sample_id']}.jpg", quality=90)
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(args.output_dir.parent / "roi_quality.csv", index=False, encoding="utf-8-sig")
    summary = quality.groupby(["view", "source"]).agg(frames=("sample_id", "size"), mean_area_ratio=("area_ratio", "mean")).reset_index()
    summary["source_ratio"] = summary["frames"] / summary.groupby("view")["frames"].transform("sum")
    summary.to_csv(args.output_dir.parent / "roi_fallback_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(args.output_dir)


if __name__ == "__main__":
    main()
