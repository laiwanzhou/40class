from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.object_interaction_roi_dataset import ObjectInteractionROIDataset
from src.train_object_interaction_tcn_expert import PROJECT_ROOT, filtered_frames, resolve_config, roi_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/depth_ir_object_interaction_tcn_expert.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/depth_ir_object_interaction_tcn_expert_fold0/roi_audit")
    args = parser.parse_args()
    generic = argparse.Namespace(
        config=args.config, smoke_test=False, probe=False, num_workers=None,
        max_train_batches=None, max_val_batches=None, run_id=None,
    )
    config = resolve_config(generic)
    train, val = filtered_frames(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in (("train", train), ("validation", val)):
        dataset = ObjectInteractionROIDataset(
            frame, Path(config["data_root"]), Path(config["pose_cache"]), int(config["expert_num_frames"]),
            int(config["image_size"]), False, roi_config(config), None,
        )
        dataset.roi_audit_rows().to_csv(args.output_dir / f"roi_audit_{split}.csv", index=False, encoding="utf-8-sig")
        if split == "validation":
            dataset.temporal_diagnostic_rows().to_csv(
                args.output_dir / "temporal_sampling_validation.csv", index=False, encoding="utf-8-sig"
            )
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
