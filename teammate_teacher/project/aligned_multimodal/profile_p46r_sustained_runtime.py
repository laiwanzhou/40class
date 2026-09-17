from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from p46_event_data import P46EventDataset, collate_p46_events
from p46_step10_model import cross_subject_supervised_contrastive
from p46r_event_bottleneck_model import (
    P46R_OFFSET_FRACTIONS,
    P46REventModel,
    circular_shift_visual_batch,
    event_localization_loss,
)
from train_p46_step10 import move_batch, seed_everything


PROJECT_DIR = Path(__file__).resolve().parent


def gpu_state() -> str:
    command = (
        "nvidia-smi --query-gpu=pstate,temperature.gpu,utilization.gpu,"
        "clocks.current.graphics,power.draw,"
        "clocks_event_reasons.sw_power_cap,"
        "clocks_event_reasons.sw_thermal_slowdown "
        "--format=csv,noheader"
    )
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.stdout.strip() or result.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    seed_everything(20260807)
    device = torch.device("cuda")
    dataset = P46EventDataset(
        PROJECT_DIR / "runs" / "p46_event_inputs_full",
        PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full",
        split="train",
        load_context=False,
    )
    # This benchmark deliberately holds one real batch in memory and repeats it.
    # It therefore contains no repeated NPZ reads or epoch-dependent sampling.
    raw = collate_p46_events(
        [dataset[index] for index in range(args.batch_size)], include_context=False
    )
    batch = move_batch(raw, device)
    model = P46REventModel(width=128, dropout=0.12).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.03)
    subjects = torch.tensor(
        [hash(value) for value in batch["user_id"]], dtype=torch.long, device=device
    )
    labels = batch["detail_index"]
    offsets = torch.arange(len(labels), device=device).remainder(
        len(P46R_OFFSET_FRACTIONS)
    )
    shifted, _ = circular_shift_visual_batch(batch, offsets)

    rows: list[dict[str, object]] = []
    chunk_started = time.perf_counter()
    total_started = chunk_started
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
            shifted_output = model(shifted)
            classification = F.cross_entropy(output["detail_logits"], labels)
            offset = F.cross_entropy(shifted_output["offset_logits"], offsets)
            localization = event_localization_loss(output)
            contrast = cross_subject_supervised_contrastive(
                output["contrast_embedding"], labels, subjects
            )
            loss = classification + 0.5 * offset + 0.15 * localization + 0.05 * contrast
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if step % args.log_every == 0:
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            row = {
                "step": step,
                "chunk_seconds": now - chunk_started,
                "seconds_per_step": (now - chunk_started) / args.log_every,
                "elapsed_seconds": now - total_started,
                "loss": float(loss.detach()),
                "gpu": gpu_state(),
            }
            rows.append(row)
            args.output.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            chunk_started = time.perf_counter()


if __name__ == "__main__":
    main()
