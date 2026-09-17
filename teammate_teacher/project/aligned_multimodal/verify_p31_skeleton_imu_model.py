from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from p31_skeleton_imu_data import P31SkeletonIMUDataset, collate_p31_trials
from p31_skeleton_imu_model import (
    P31SkeletonIMUPartEncoders,
    model_size_mib,
    parameter_count,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "runs" / "p31_skeleton_imu_full"
MISSING_IMU_AUDIT_SAMPLE = "8_Take_and_use_tableware/user1/2-1-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise P31 encoders on real maximum-size trials")
    parser.add_argument("--cache-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.cache_run.resolve()
    cache_verification = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    sample_ids = (
        cache_verification["maximum_frame_trial"],
        cache_verification["maximum_point_trial"],
        MISSING_IMU_AUDIT_SAMPLE,
    )
    dataset = P31SkeletonIMUDataset(run)
    lookup = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    batch = collate_p31_trials([dataset[lookup[sample_id]] for sample_id in sample_ids])
    device = torch.device(args.device)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model = P31SkeletonIMUPartEncoders().to(device).train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = model(batch)
    loss = output["skeleton_part_tokens"].square().mean()
    loss = loss + output["imu_part_tokens"].square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "verified": True,
        "device": str(device),
        "batch_samples": list(sample_ids),
        "batch_shape_T": int(batch["frame_mask"].shape[1]),
        "batch_shape_N_per_device": int(batch["imu_point_mask"].shape[2]),
        "skeleton_tokens": list(output["skeleton_part_tokens"].shape),
        "imu_tokens": list(output["imu_part_tokens"].shape),
        "cached_imu_points": int(batch["imu_point_mask"].sum().item()),
        "encoder_counted_imu_points": int(output["imu_interval_count"].sum().item()),
        "finite_outputs": bool(
            torch.isfinite(output["skeleton_part_tokens"]).all()
            and torch.isfinite(output["imu_part_tokens"]).all()
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
        "forward_backward_seconds": time.perf_counter() - started,
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    if result["cached_imu_points"] != result["encoder_counted_imu_points"]:
        raise AssertionError("P31 encoder interval pooling lost IMU points")
    if not result["finite_outputs"] or not result["finite_gradients"]:
        raise FloatingPointError("P31 encoder produced non-finite output/gradient")
    (run / "model_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
