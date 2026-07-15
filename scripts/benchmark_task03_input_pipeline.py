from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = Path(r"D:\Anaconda\envs\pyTorch2.7\python.exe")
REPORT_CSV = PROJECT_ROOT / "reports" / "task03_input_pipeline_benchmark.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "task03_input_pipeline_benchmark.md"
MODALITY_ORDER = ("depth_color", "imu", "skeleton", "radar", "ir", "thermal")
WARMUP_BATCHES = 10
MEASURED_BATCHES = 100
FIELDS = (
    "modality",
    "num_workers",
    "batch_size",
    "measured_batches",
    "measured_samples",
    "elapsed_seconds",
    "samples_per_second",
    "batches_per_second",
    "mean_batch_seconds",
    "torch_peak_allocated_mb",
    "torch_peak_reserved_mb",
    "status",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Task 03 input pipeline configurations.")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_benchmark_config(path: Path, workers: int, batch_size: int) -> dict[str, Any]:
    config_path = path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config.update(
        {
            "config_path": str(config_path),
            "data_root": str(Path(config["data_root"]).resolve()),
            "manifest": str((PROJECT_ROOT / config.get("manifest", "metadata/manifest.csv")).resolve()),
            "fold": str((PROJECT_ROOT / config.get("fold", "metadata/splits/fold_0.json")).resolve()),
            "num_workers": workers,
            "batch_size": batch_size,
            "smoke_test": False,
        }
    )
    return config


def benchmark_single(config_path: Path, workers: int, batch_size: int) -> dict[str, Any]:
    from src.data.common import compute_sequence_normalization
    from src.train_unimodal import (
        VISUAL_MODALITIES,
        build_datasets,
        build_model,
        loader_for,
        set_seed,
    )

    if workers < 0:
        raise ValueError("num_workers must be non-negative.")
    config = load_benchmark_config(config_path, workers, batch_size)
    set_seed(int(config["seed"]))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    train_dataset, val_dataset = build_datasets(config)
    del val_dataset
    if str(config["modality"]) not in VISUAL_MODALITIES:
        mean, std = compute_sequence_normalization(train_dataset)  # type: ignore[arg-type]
        train_dataset.set_normalization(mean, std)  # type: ignore[attr-defined]
    sample = train_dataset[0]
    model = build_model(config, sample).to(device)
    loader = loader_for(train_dataset, config, training=True)
    if len(loader) <= WARMUP_BATCHES:
        raise ValueError(f"Only {len(loader)} batches are available; warmup requires {WARMUP_BATCHES}.")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    amp_enabled = bool(config.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    iterator = iter(loader)

    def train_batch(batch: dict[str, object]) -> int:
        inputs = batch["input"].to(device, non_blocking=True)  # type: ignore[union-attr]
        labels = batch["label"].to(device, non_blocking=True)  # type: ignore[union-attr]
        mask = batch["temporal_mask"].to(device, non_blocking=True)  # type: ignore[union-attr]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            output = model(inputs, temporal_mask=mask)
            loss = criterion(output["logits"], labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        return int(labels.shape[0])

    model.train()
    for _ in range(WARMUP_BATCHES):
        train_batch(next(iterator))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured_limit = min(MEASURED_BATCHES, len(loader) - WARMUP_BATCHES)
    measured_samples = 0
    started = time.perf_counter()
    for _ in range(measured_limit):
        measured_samples += train_batch(next(iterator))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "modality": str(config["modality"]),
        "num_workers": workers,
        "batch_size": batch_size,
        "measured_batches": measured_limit,
        "measured_samples": measured_samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": measured_samples / elapsed,
        "batches_per_second": measured_limit / elapsed,
        "mean_batch_seconds": elapsed / measured_limit,
        "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "torch_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "status": "passed",
        "error": "",
    }
    print(f"BENCHMARK_JSON={json.dumps(result, ensure_ascii=False)}")
    return result


def run_subprocess(modality: str, workers: int, batch_size: int) -> dict[str, Any]:
    command = [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "--single",
        "--config",
        str(PROJECT_ROOT / "configs" / "task03" / f"{modality}.yaml"),
        "--num-workers",
        str(workers),
        "--batch-size",
        str(batch_size),
    ]
    print(f"Running benchmark: modality={modality}, workers={workers}, batch={batch_size}")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=1800,
    )
    print(completed.stdout)
    lines = [line for line in completed.stdout.splitlines() if line.startswith("BENCHMARK_JSON=")]
    if completed.returncode == 0 and lines:
        return json.loads(lines[-1].split("=", 1)[1])
    error = (completed.stderr or completed.stdout or "Benchmark process failed without output.")[-4000:]
    status = "cuda_oom" if "out of memory" in error.casefold() else "failed"
    return {
        "modality": modality,
        "num_workers": workers,
        "batch_size": batch_size,
        "measured_batches": 0,
        "measured_samples": 0,
        "elapsed_seconds": 0.0,
        "samples_per_second": 0.0,
        "batches_per_second": 0.0,
        "mean_batch_seconds": 0.0,
        "torch_peak_allocated_mb": 0.0,
        "torch_peak_reserved_mb": 0.0,
        "status": status,
        "error": error.replace("\n", " "),
    }


def choose_smallest_near_best(rows: list[dict[str, Any]], field: str) -> int:
    passed = [row for row in rows if row["status"] == "passed"]
    if not passed:
        raise RuntimeError("No stable benchmark configuration passed.")
    best_rate = max(float(row["samples_per_second"]) for row in passed)
    near_best = [row for row in passed if float(row["samples_per_second"]) >= best_rate * 0.95]
    return min(int(row[field]) for row in near_best)


def write_reports(rows: list[dict[str, Any]], selections: dict[str, dict[str, int]]) -> None:
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIELDS).to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    lines = [
        "# Task 03 input pipeline benchmark",
        "",
        f"Each passed combination used {WARMUP_BATCHES} warmup batches and up to {MEASURED_BATCHES} measured training batches in an isolated Python process.",
        "Measurements include real data loading and decoding, H2D transfer, CUDA AMP forward/backward, gradient clipping, and AdamW step.",
        "",
        "| Modality | Workers | Batch | Measured batches | Samples/s | Batch s | Peak allocated MB | Peak reserved MB | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['modality']} | {row['num_workers']} | {row['batch_size']} | {row['measured_batches']} | "
            f"{float(row['samples_per_second']):.3f} | {float(row['mean_batch_seconds']):.4f} | "
            f"{float(row['torch_peak_allocated_mb']):.2f} | {float(row['torch_peak_reserved_mb']):.2f} | {row['status']} |"
        )
    lines.extend(["", "## Selected configurations", ""])
    for modality in MODALITY_ORDER:
        selected = selections[modality]
        lines.append(f"- `{modality}`: `num_workers={selected['num_workers']}`, `batch_size={selected['batch_size']}`")
    lines.extend(
        [
            "",
            "Selection requires a passed run with no CUDA OOM or worker failure. Throughput is primary; when configurations are within 5% of the best rate, the smaller worker or batch value is selected.",
            "CPU utilization may remain high; success is judged by stable completion and measured throughput rather than a single utilization snapshot.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_matrix() -> None:
    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, int]] = {}
    depth_worker_rows = [run_subprocess("depth_color", workers, 4) for workers in (0, 2, 4)]
    rows.extend(depth_worker_rows)
    depth_workers = choose_smallest_near_best(depth_worker_rows, "num_workers")
    depth_batch_rows = [row for row in depth_worker_rows if int(row["num_workers"]) == depth_workers]
    for batch_size in (8, 12, 16):
        depth_batch_rows.append(run_subprocess("depth_color", depth_workers, batch_size))
    rows.extend(row for row in depth_batch_rows if row not in rows)
    depth_batch = choose_smallest_near_best(depth_batch_rows, "batch_size")
    selections["depth_color"] = {"num_workers": depth_workers, "batch_size": depth_batch}

    for modality in ("imu", "skeleton", "radar"):
        modality_rows = [run_subprocess(modality, workers, 16) for workers in (0, 2, 4)]
        rows.extend(modality_rows)
        selections[modality] = {
            "num_workers": choose_smallest_near_best(modality_rows, "num_workers"),
            "batch_size": 16,
        }

    for modality in ("ir", "thermal"):
        confirmation = run_subprocess(modality, depth_workers, depth_batch)
        rows.append(confirmation)
        if confirmation["status"] != "passed":
            raise RuntimeError(f"Selected visual configuration failed for {modality}: {confirmation['error']}")
        selections[modality] = {"num_workers": depth_workers, "batch_size": depth_batch}
    write_reports(rows, selections)
    print(f"Benchmark reports written to {REPORT_CSV} and {REPORT_MD}")


def main() -> None:
    args = parse_args()
    if args.single:
        if args.config is None or args.num_workers is None or args.batch_size is None:
            raise SystemExit("--single requires --config, --num-workers, and --batch-size")
        benchmark_single(args.config, args.num_workers, args.batch_size)
    else:
        run_matrix()


if __name__ == "__main__":
    main()
