from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from benchmark_p32_short_train import (
    make_loader,
    run_eval_epoch,
    stratified_length_sample,
)
from p32_fused_data import P32FusedTrialDataset
from p32_part_fusion_temporal_model import P32PartFusionTemporalModel


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure cold and persistent-worker P32 eval passes")
    parser.add_argument(
        "--visual-run",
        type=Path,
        default=PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full",
    )
    parser.add_argument(
        "--motion-run",
        type=Path,
        default=PROJECT_DIR / "runs" / "p31_skeleton_imu_full",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p32_steps13_14_short_benchmark_80_tcn_w2"
        / "eval_repeat.json",
    )
    parser.add_argument("--trials-per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = P32FusedTrialDataset(args.visual_run.resolve(), args.motion_run.resolve())
    selected = stratified_length_sample(full.rows, args.trials_per_class)
    dataset = P32FusedTrialDataset(
        args.visual_run.resolve(), args.motion_run.resolve(), selected
    )
    loader, sampler = make_loader(
        dataset,
        args.batch_size,
        args.workers,
        shuffle=False,
        seed=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P32PartFusionTemporalModel().to(device)
    head = nn.Linear(384, 40).to(device)
    passes = []
    for index in range(args.repeats):
        metrics = run_eval_epoch(model, head, loader, sampler, device)
        passes.append(metrics)
        print(json.dumps({"pass": index + 1, **metrics}, ensure_ascii=False), flush=True)
    result = {
        "device": str(device),
        "workers": args.workers,
        "trials": len(dataset),
        "real_frames": sum(dataset.frame_lengths),
        "passes": passes,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
