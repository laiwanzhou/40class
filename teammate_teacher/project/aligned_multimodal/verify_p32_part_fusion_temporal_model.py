from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from p32_fused_data import P32FusedTrialDataset, collate_p32_trials
from p32_part_fusion_temporal_model import (
    P32PartFusionTemporalModel,
    model_size_mib,
    parameter_count,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VISUAL = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_MOTION = PROJECT_DIR / "runs" / "p31_skeleton_imu_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p32_steps13_14_verification"
MISSING_IMU_SAMPLE = "8_Take_and_use_tableware/user1/2-1-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Steps 13/14 on real sequence extremes")
    parser.add_argument("--visual-run", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--motion-run", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_verification = json.loads(
        (args.motion_run.resolve() / "verification.json").read_text(encoding="utf-8")
    )
    sample_ids = (
        motion_verification["maximum_frame_trial"],
        motion_verification["maximum_point_trial"],
        MISSING_IMU_SAMPLE,
    )
    dataset = P32FusedTrialDataset(
        args.visual_run.resolve(), args.motion_run.resolve()
    )
    lookup = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    batch = collate_p32_trials([dataset[lookup[sample_id]] for sample_id in sample_ids])
    device = torch.device(args.device)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model = P32PartFusionTemporalModel().to(device).train()
    disposable_head = nn.Linear(384, 40).to(device).train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=device.type in {"cuda", "cpu"},
    ):
        output = model(batch)
        logits = disposable_head(output["trial_embedding"])
        loss = nn.functional.cross_entropy(logits, batch["label"])
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gate_error = (
        output["modality_gate"].sum(dim=3)
        - output["fused_part_mask"].to(output["modality_gate"].dtype)
    ).abs().max()
    result = {
        "verified": True,
        "device": str(device),
        "samples": list(sample_ids),
        "batch_T": int(batch["frame_mask"].shape[1]),
        "batch_N_per_device": int(batch["imu_point_mask"].shape[2]),
        "trial_embedding_shape": list(output["trial_embedding"].shape),
        "fused_part_shape": list(output["fused_part_tokens"].shape),
        "temporal_sequence_shape": list(output["temporal_sequence"].shape),
        "temporal_tcn_dilations": [1, 2, 4, 8, 16, 32, 64],
        "temporal_receptive_field_frames": 255,
        "cached_imu_points": int(batch["imu_point_mask"].sum().item()),
        "encoder_counted_imu_points": int(output["imu_interval_count"].sum().item()),
        "maximum_modality_gate_sum_error": float(gate_error.detach()),
        "finite_outputs": bool(
            all(
                torch.isfinite(output[key]).all()
                for key in (
                    "trial_embedding",
                    "fused_part_tokens",
                    "temporal_sequence",
                    "modality_gate",
                )
            )
        ),
        "finite_gradients": bool(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        ),
        "parameters": parameter_count(model),
        "fp32_mib": model_size_mib(model, 4),
        "fp16_mib": model_size_mib(model, 2),
        "forward_backward_seconds_including_disposable_ce_head": time.perf_counter()
        - started,
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "no_checkpoint_saved": True,
    }
    if result["cached_imu_points"] != result["encoder_counted_imu_points"]:
        raise AssertionError("P32 lost IMU points")
    if result["maximum_modality_gate_sum_error"] > 1e-5:
        raise AssertionError("P32 modality gates do not sum to one on valid parts")
    if not result["finite_outputs"] or not result["finite_gradients"]:
        raise FloatingPointError("P32 output/gradient is not finite")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
