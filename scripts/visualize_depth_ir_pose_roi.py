from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pose_roi_dataset import PoseTrackCache, depth_frame_key
from src.train_unimodal import build_datasets, load_config
COLORS = ("white", "lime", "cyan", "magenta")
VIEW_NAMES = ("global", "upper", "left_hand", "right_hand")


def config_args() -> argparse.Namespace:
    return argparse.Namespace(
        data_root=None, manifest=None, fold=None, output_root=None, device=None, seed=None,
        smoke_test=False, max_epochs=None, num_workers=0, max_train_batches=None,
        max_val_batches=None, run_id=None,
    )


def overlay(axis: plt.Axes, image: Image.Image, boxes: np.ndarray, keypoints: np.ndarray) -> None:
    axis.imshow(image, cmap="gray" if image.mode == "L" else None)
    width, height = image.size
    all_boxes = np.vstack(([0, 0, width, height], boxes))
    for name, color, box in zip(VIEW_NAMES, COLORS, all_boxes, strict=True):
        x1, y1, x2, y2 = box
        axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1.4))
        axis.text(x1 + 2, y1 + 10, name, color=color, fontsize=6, backgroundcolor="black")
    valid = np.isfinite(keypoints).all(axis=1)
    axis.scatter(keypoints[valid, 0], keypoints[valid, 1], s=5, c="yellow")
    axis.axis("off")


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs/experiments/depth_ir_pose_roi_expert.yaml", config_args())
    _, dataset = build_datasets(config)
    cache = PoseTrackCache(Path(config["pose_cache"]))
    chosen = []
    for label in range(len(dataset.class_names)):
        chosen.append(next(index for index, sample in enumerate(dataset.samples) if sample["label"] == label))
    chosen.append(max(range(len(dataset.samples)), key=lambda index: int(dataset.samples[index]["length"])))
    output_dir = PROJECT_ROOT / "outputs/depth_ir_pose_roi_probe/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    for order, index in enumerate(chosen):
        sample = dataset.samples[index]
        frame_index = int(sample["length"]) // 2
        depth_path = sample["paths"][frame_index]
        ir_path = sample["ir_paths"][frame_index]
        depth = Image.open(depth_path).convert("RGB")
        ir = Image.open(ir_path).convert("L")
        if depth.size != ir.size:
            raise ValueError(f"Native image size mismatch: {depth_path} / {ir_path}")
        boxes = sample["roi_boxes"][frame_index]
        cached = cache.lookup.get((str(sample["sample_id"]), depth_frame_key(depth_path)))
        if cached is None:
            raise KeyError(f"Pose cache key absent: {sample['sample_id']} / {depth_frame_key(depth_path)}")
        _, keypoints, _ = cached
        figure, axes = plt.subplots(3, 4, figsize=(12, 8))
        axes[0, 0].imshow(depth); axes[0, 0].set_title("Depth original"); axes[0, 0].axis("off")
        overlay(axes[0, 1], depth, boxes, keypoints); axes[0, 1].set_title("Depth shared ROI")
        axes[0, 2].imshow(ir, cmap="gray"); axes[0, 2].set_title("IR original"); axes[0, 2].axis("off")
        overlay(axes[0, 3], ir, boxes, keypoints); axes[0, 3].set_title("IR same ROI")
        for modality_row, image in ((1, depth), (2, ir)):
            width, height = image.size
            all_boxes = np.vstack(([0, 0, width, height], boxes))
            for view_index, (name, box) in enumerate(zip(VIEW_NAMES, all_boxes, strict=True)):
                axes[modality_row, view_index].imshow(image.crop(tuple(float(value) for value in box)), cmap="gray" if image.mode == "L" else None)
                axes[modality_row, view_index].set_title(f"{'Depth' if modality_row == 1 else 'IR'} {name}")
                axes[modality_row, view_index].axis("off")
        action = dataset.class_names[int(sample["label"])]
        figure.suptitle(f"{sample['sample_id']} | {action} | {depth_frame_key(depth_path)}", fontsize=10)
        figure.tight_layout()
        figure.savefig(output_dir / f"{order:02d}_{action}_{sample['sample_id']}.png", dpi=130)
        plt.close(figure)
    print(f"Saved {len(chosen)} paired visualizations to {output_dir}")


if __name__ == "__main__":
    main()
