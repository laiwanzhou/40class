from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.object_interaction_roi_dataset import ObjectInteractionROIDataset
from src.models.object_interaction_tcn_expert import ObjectInteractionTCNExpert
from src.train_object_interaction_tcn_expert import (
    PROJECT_ROOT, evaluate, filtered_frames, load_base_logits, make_loader, resolve_config, roi_config, target_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mobile-run-dir", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_object_interaction_tcn_expert_fold0/depth_ir_object_interaction_tcn_expert_14train_4val",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_object_interaction_resnet18_expert_fold0/depth_ir_object_interaction_resnet18_expert_14train_4val/mobilenet_validation_benchmark.json",
    )
    args = parser.parse_args()
    namespace = argparse.Namespace(
        config=args.mobile_run_dir / "config.yaml", smoke_test=False, probe=False,
        num_workers=None, max_train_batches=None, max_val_batches=None, run_id=None,
    )
    config = resolve_config(namespace)
    _, val_frame = filtered_frames(config)
    class_map = pd.read_csv(config["class_map"], encoding="utf-8-sig").sort_values("class_id")
    targets = target_ids(config, class_map)
    dataset = ObjectInteractionROIDataset(
        val_frame, Path(config["data_root"]), Path(config["pose_cache"]), int(config["expert_num_frames"]),
        int(config["image_size"]), False, roi_config(config), load_base_logits(config),
    )
    loader = make_loader(dataset, config, False)
    model = ObjectInteractionTCNExpert(
        targets, frame_feature_dim=int(config["frame_feature_dim"]), tcn_channels=int(config["tcn_channels"]),
        embedding_dim=int(config["expert_embedding_dim"]), kernel_size=int(config["tcn_kernel_size"]),
        dilations=tuple(config["tcn_dilations"]), dropout=float(config["dropout"]),
    )
    checkpoint = torch.load(args.mobile_run_dir / "best_target16_macro_f1.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(config["device"])
    model.to(device)
    result = evaluate(model, loader, device, config, np.asarray(targets))
    record = {
        "checkpoint_epoch": int(checkpoint["epoch"]), "validation_samples": len(result["labels"]),
        "validation_seconds": result["validation_seconds"], "accuracy": result["metrics"]["accuracy"],
        "macro_f1": result["metrics"]["macro_f1"], "target16_macro_f1": result["metrics"]["target16_macro_f1"],
        "sample_ids_match_saved": bool(np.array_equal(
            result["sample_ids"].astype(str),
            np.load(args.mobile_run_dir / "val_predictions_best_target16.npz", allow_pickle=False)["sample_ids"].astype(str),
        )),
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
