from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.object_interaction_roi_dataset import ObjectInteractionROIDataset
from src.train_object_interaction_tcn_expert import PROJECT_ROOT, filtered_frames, resolve_config, roi_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/depth_ir_object_interaction_tcn_expert.yaml")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_object_interaction_tcn_expert_fold0/object_interaction_rois.npz",
    )
    args = parser.parse_args()
    generic = argparse.Namespace(
        config=args.config, smoke_test=False, probe=False, num_workers=None,
        max_train_batches=None, max_val_batches=None, run_id=None,
    )
    config = resolve_config(generic)
    train, val = filtered_frames(config)
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "sample_ids", "split", "frame_index", "boxes", "raw_boxes", "valid_mask", "sources",
            "keypoint_valid", "projected_points", "two_hand_merge", "hand_head_trigger",
        )
    }
    for split, frame in (("train", train), ("validation", val)):
        dataset = ObjectInteractionROIDataset(
            frame, Path(config["data_root"]), Path(config["pose_cache"]), int(config["expert_num_frames"]),
            int(config["image_size"]), False, roi_config(config), None,
        )
        for sample in dataset.samples:
            roi = sample["roi"]
            count = int(sample["length"])
            arrays["sample_ids"].append(np.repeat(str(sample["sample_id"]), count))
            arrays["split"].append(np.repeat(split, count))
            arrays["frame_index"].append(np.arange(count, dtype=np.int32))
            arrays["boxes"].append(roi.boxes)
            arrays["raw_boxes"].append(roi.raw_boxes)
            arrays["valid_mask"].append(roi.valid_mask)
            arrays["sources"].append(roi.sources)
            arrays["keypoint_valid"].append(roi.keypoint_valid)
            arrays["projected_points"].append(roi.projected_points)
            arrays["two_hand_merge"].append(roi.two_hand_merge)
            arrays["hand_head_trigger"].append(roi.hand_head_trigger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{name: np.concatenate(values) for name, values in arrays.items()})
    print(f"Saved {sum(len(value) for value in arrays['frame_index'])} frame ROIs to {args.output}")


if __name__ == "__main__":
    main()
