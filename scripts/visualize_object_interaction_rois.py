from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.data.object_interaction_roi_dataset import ObjectInteractionROIDataset
from src.data.pose_roi_dataset import PoseTrackCache, depth_frame_key
from src.roi.object_interaction_builder import VIEW_NAMES
from src.train_object_interaction_tcn_expert import PROJECT_ROOT, filtered_frames, load_base_logits, resolve_config, roi_config


ACTIONS = (
    "Take_and_use_tableware", "Write", "Make_a_phone_call", "Watch_TV", "Take_medicine",
    "Use_a_mobile_phone", "Wipe_bowls", "Take_body_temperature", "Peel_fruits", "Stir_drinks",
    "Play_games", "Take_a_selfie", "Turn_pages", "Read_documents", "Drink_water", "Eat_food",
    "Wash_face", "Wipe_windows_and_tables", "Put_on_clothes", "Take_off_clothes", "Lie_down",
)
COLORS = ("white", "lime", "cyan", "magenta", "yellow")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/depth_ir_object_interaction_tcn_expert.yaml")
    parser.add_argument(
        "--run-dir", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_object_interaction_tcn_expert_fold0/depth_ir_object_interaction_tcn_expert_14train_4val",
    )
    args = parser.parse_args()
    generic = argparse.Namespace(
        config=args.config, smoke_test=False, probe=False, num_workers=None,
        max_train_batches=None, max_val_batches=None, run_id=None,
    )
    config = resolve_config(generic)
    _, val = filtered_frames(config)
    dataset = ObjectInteractionROIDataset(
        val, Path(config["data_root"]), Path(config["pose_cache"]), int(config["expert_num_frames"]),
        int(config["image_size"]), False, roi_config(config), load_base_logits(config),
    )
    archive = np.load(args.run_dir / "val_predictions_best_target16.npz", allow_pickle=False)
    result_index = {str(value): index for index, value in enumerate(archive["sample_ids"])}
    class_names = dataset.class_names
    pose_cache = PoseTrackCache(Path(config["pose_cache"]))
    output_dir = args.run_dir / "roi_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    for action in ACTIONS:
        candidates = [sample for sample in dataset.samples if sample["action_name"] == action][:2]
        for sample_number, sample in enumerate(candidates):
            sample_id = str(sample["sample_id"])
            result = result_index[sample_id]
            sampled = archive["sampled_indices"][result]
            positions = (0, len(sampled) // 2, int(archive["temporal_mask"][result].sum()) - 1)
            figure, axes = plt.subplots(3, 2, figsize=(13, 13))
            roi = sample["roi"]
            for row, position in enumerate(positions):
                frame = int(sampled[position])
                for column, (paths_key, mode, title) in enumerate(
                    (("depth_paths", "RGB", "Depth"), ("ir_paths", "L", "IR"))
                ):
                    path = sample[paths_key][frame]
                    with Image.open(path) as opened:
                        image = np.asarray(opened.convert(mode))
                    axis = axes[row, column]
                    axis.imshow(image, cmap="gray" if mode == "L" else None)
                    _, keypoints, confidence = pose_cache.lookup[(sample_id, depth_frame_key(sample["depth_paths"][frame]))]
                    valid = (confidence >= float(config["pose_keypoint_confidence_threshold"])) & np.isfinite(keypoints).all(axis=1)
                    axis.scatter(keypoints[valid, 0], keypoints[valid, 1], s=12, c="red")
                    for view, color in enumerate(COLORS):
                        if not roi.valid_mask[frame, view] or view == 0:
                            continue
                        box = roi.boxes[frame, view]
                        axis.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color=color, linewidth=1.5))
                    for projected in roi.projected_points[frame]:
                        if np.isfinite(projected).all():
                            axis.scatter(projected[0], projected[1], marker="x", s=35, c="blue")
                    weights = archive["view_weights"][result, position]
                    masks = archive["view_valid_mask"][result, position].astype(int)
                    axis.set_title(f"{title} {('start', 'middle', 'end')[row]} | mask={masks.tolist()}\nw={np.round(weights, 3).tolist()}")
                    axis.axis("off")
            true_id = int(archive["labels"][result])
            base_id = int(archive["base_logits"][result].argmax())
            final_id = int(archive["logits"][result].argmax())
            figure.suptitle(
                f"{sample_id}\ntrue={class_names[true_id]} base={class_names[base_id]} final={class_names[final_id]} | "
                f"two_hand={bool(archive['two_hand_roi_valid'][result])} hand_head={bool(archive['hand_head_roi_valid'][result])}"
            )
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            figure.savefig(output_dir / f"{int(sample['label']):02d}_{action}_{sample_number}.png", dpi=130)
            plt.close(figure)
    print(f"Saved ROI visualizations to {output_dir}")


if __name__ == "__main__":
    main()
