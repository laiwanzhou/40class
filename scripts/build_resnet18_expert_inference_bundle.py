from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_object_interaction_resnet18_expert_fold0/depth_ir_object_interaction_resnet18_expert_14train_4val",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "outputs/depth_ir_pose_roi_40class_fold0/depth_ir_pose_roi_40class_fold0_14train_4val/best_macro_f1.pt")
    parser.add_argument("--pose-weights", type=Path, default=PROJECT_ROOT / "yolo11n-pose.pt")
    args = parser.parse_args()
    expert = torch.load(args.run_dir / "best_target16_macro_f1.pt", map_location="cpu", weights_only=True)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    output = args.run_dir / "resnet18_expert_inference_bundle.pt"
    torch.save(
        {
            "format": "depth_ir_resnet18_object_interaction_bundle_v1",
            "base_model_state_dict": base["model_state_dict"],
            "expert_model_state_dict": expert["model_state_dict"],
            "pose_weights_bytes": args.pose_weights.read_bytes(),
            "base_epoch": int(base["epoch"]),
            "expert_epoch": int(expert["epoch"]),
            "target_class_ids": expert["target_class_ids"],
        },
        output,
    )
    components = {
        "frozen_base_fp32_bytes": sum(v.numel() * v.element_size() for v in base["model_state_dict"].values()),
        "resnet18_expert_fp32_bytes": sum(v.numel() * v.element_size() for v in expert["model_state_dict"].values()),
        "yolo11n_pose_file_bytes": args.pose_weights.stat().st_size,
        "actual_bundle_bytes": output.stat().st_size,
        "actual_bundle_mib": output.stat().st_size / 1048576,
        "under_100_mib": output.stat().st_size < 100 * 1048576,
    }
    (args.run_dir / "inference_weight_budget.json").write_text(json.dumps(components, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(components, indent=2))


if __name__ == "__main__":
    main()
