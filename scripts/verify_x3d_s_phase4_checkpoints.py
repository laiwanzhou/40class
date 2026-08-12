from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.x3d_clip_dataset import X3DClipDataset, collate_x3d_clips
import src.train_x3d_s_visual_expert as trainer


TOLERANCE = 1e-6


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-evaluate Phase-4 formal X3D checkpoints")
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--entry", action="append", required=True, help="seed:fold:fold_directory")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_entry(value: str) -> tuple[int, int, Path]:
    seed, fold, path = value.split(":", maxsplit=2)
    return int(seed), int(fold), Path(path).resolve()


def aligned_max_delta(saved: np.ndarray, fresh: np.ndarray) -> float:
    if saved.shape != fresh.shape:
        raise ValueError(f"Array shape mismatch: {saved.shape} != {fresh.shape}")
    if saved.dtype.kind in "OUSb" or fresh.dtype.kind in "OUSb":
        if not np.array_equal(saved, fresh):
            raise ValueError("Non-floating archive fields differ")
        return 0.0
    return float(np.max(np.abs(saved.astype(np.float64) - fresh.astype(np.float64)), initial=0.0))


def verify_entry(
    *, seed: int, fold_index: int, fold_directory: Path, assignment: dict[str, Any], assignment_sha: str
) -> dict[str, Any]:
    checkpoint_path = fold_directory / "formal_outer_refit.pt"
    archive_path = fold_directory / "formal_outer_predictions.npz"
    summary_path = fold_directory / "run_summary.json"
    for path in (checkpoint_path, archive_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config_path = fold_directory.parent / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = dict(config)
    config["seed"] = seed
    trainer.validate_config(config)
    split = json.loads(Path(str(config["split_path"])).read_text(encoding="utf-8"))
    folds = trainer.validate_oof_assignment(
        assignment, allowed_users=set(str(user) for user in split["train_users"])
    )
    fold = folds[fold_index]
    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    frame = trainer.prepare_partition_manifest(
        manifest,
        train_user_ids=fold.train_user_ids,
        validation_user_ids=fold.validation_user_ids,
    )
    dataset = X3DClipDataset(frame, split="val", training=False, seed=seed)
    loader_config = config["loader"]
    sampler = trainer.ClipBudgetBatchSampler(
        dataset.num_clips,
        max_trials_per_batch=int(loader_config["max_trials_per_batch"]),
        max_valid_clips_per_batch=int(loader_config["max_valid_clips_per_batch"]),
        shuffle=False,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_x3d_clips,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("strict_oof_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Missing strict_oof_provenance: {checkpoint_path}")
    expected = {
        "actual_seed": seed,
        "outer_fold": fold_index,
        "outer_train_user_ids": list(fold.train_user_ids),
        "outer_validation_user_ids": list(fold.validation_user_ids),
        "assignment_sha256": assignment_sha,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Checkpoint provenance mismatch for {key}: {provenance.get(key)!r} != {value!r}")

    model = trainer._build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(str(config["device"]))
    model.to(device)
    outcome = trainer.run_model_epoch(
        model,
        loader,
        device=device,
        optimizer=None,
        gradient_accumulation=1,
        gradient_clip=float(config["optimizer"]["gradient_clip"]),
        amp_enabled=bool(config["amp"]["enabled"]),
        max_batches=None,
    )
    fresh = outcome.predictions
    fresh_arrays = {
        "sample_ids": np.asarray(fresh.sample_ids),
        "user_ids": np.asarray(fresh.user_ids),
        "labels": fresh.labels.numpy(),
        "logits": fresh.output.main_logits.numpy(),
        "embeddings": fresh.output.embedding.numpy(),
        "quality": fresh.output.quality.numpy(),
        "quality_mask": fresh.output.quality_mask.numpy(),
        "availability": fresh.output.availability.numpy(),
        "num_frames": fresh.num_frames.numpy(),
        "num_clips": fresh.num_clips.numpy(),
    }
    with np.load(archive_path, allow_pickle=False) as saved:
        order = {str(sample): index for index, sample in enumerate(fresh_arrays["sample_ids"])}
        try:
            indices = np.asarray([order[str(sample)] for sample in saved["sample_ids"]])
        except KeyError as exc:
            raise ValueError(f"Fresh prediction is missing sample {exc.args[0]}") from exc
        deltas = {
            key: aligned_max_delta(saved[key], value[indices])
            for key, value in fresh_arrays.items()
        }
    maximum_delta = max(deltas.values())
    if maximum_delta > TOLERANCE:
        raise ValueError(f"Prediction regeneration exceeded tolerance: {maximum_delta}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metric_deltas = {
        "accuracy": abs(float(outcome.metrics["accuracy"]) - float(summary["formal_outer_accuracy"])),
        "macro_f1": abs(float(outcome.metrics["macro_f1"]) - float(summary["formal_outer_macro_f1"])),
        "worst_user_accuracy": abs(
            float(outcome.metrics["worst_user_accuracy"])
            - float(summary["formal_outer_worst_user_accuracy"])
        ),
    }
    if max(metric_deltas.values()) > TOLERANCE:
        raise ValueError(f"Metric regeneration exceeded tolerance: {metric_deltas}")
    return {
        "seed": seed,
        "fold": fold_index,
        "fold_directory": str(fold_directory),
        "selected_epoch": int(provenance["selected_epoch"]),
        "checkpoint_sha256": trainer._sha256_file(checkpoint_path),
        "archive_sha256": trainer._sha256_file(archive_path),
        "sample_count": len(fresh.sample_ids),
        "maximum_array_delta": maximum_delta,
        "array_deltas": deltas,
        "metric_deltas": metric_deltas,
        "status": "passed",
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
    assignment_sha = trainer._sha256_file(args.assignment)
    entries = [parse_entry(value) for value in args.entry]
    if len({(seed, fold) for seed, fold, _ in entries}) != len(entries):
        raise ValueError("Duplicate seed/fold entries")
    results = [
        verify_entry(
            seed=seed,
            fold_index=fold,
            fold_directory=path,
            assignment=assignment,
            assignment_sha=assignment_sha,
        )
        for seed, fold, path in entries
    ]
    report = {
        "schema_version": 1,
        "assignment_sha256": assignment_sha,
        "tolerance": TOLERANCE,
        "entry_count": len(results),
        "all_passed": all(result["status"] == "passed" for result in results),
        "entries": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
