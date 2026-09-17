from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from p46_event_data import P46EventDataset, collate_p46_events
from p46_protocol import HARD_CLASS_IDS
from p46_unified_repair_model import P46UnifiedRepairV3Model
from train_p46_step10 import FrameBudgetBatchSampler
from train_p46_unified_repair import evaluate


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the frozen four-user P46-v3 checkpoint's Detail21 logits."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "runs/p46_unified_repair_v3_clean/best_accuracy.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "runs/p46_v3_train_oof_v1/target_validation_logits.npz",
    )
    parser.add_argument("--event-run", type=Path, default=PROJECT_DIR / "runs/p46_event_inputs_full")
    parser.add_argument("--context-run", type=Path, default=PROJECT_DIR / "runs/p30_shared_dir_roi_features_full")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--eval-frame-budget", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("checkpoint_stage") != "P46_unified_repair_v3_stageB":
        raise RuntimeError("Checkpoint is not a P46-v3 Stage-B model")
    expected_val_users = ["user1", "user2", "user8", "user9"]
    if sorted(config.get("val_subjects", [])) != expected_val_users:
        raise RuntimeError("Checkpoint does not use the frozen four-user P46 validation split")
    model = P46UnifiedRepairV3Model(
        width=int(config["model_width"]),
        dropout=float(config["model_dropout"]),
        subjects=len(config["train_subjects"]),
        raw_axis_rotation_degrees=float(config["raw_axis_rotation_degrees"]),
        raw_coordinate_dropout=float(config["raw_coordinate_dropout"]),
        relationship_maximum_scale=float(config["relationship_maximum_scale"]),
        relationship_initial_scale=float(config["relationship_initial_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = P46EventDataset(
        args.event_run.resolve(), args.context_run.resolve(), split="val", load_context=True
    )
    sampler = FrameBudgetBatchSampler(
        dataset.frame_lengths,
        maximum_batch_size=args.eval_batch_size,
        frame_budget=args.eval_frame_budget,
    )
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": False,
        "persistent_workers": False,
        "collate_fn": collate_p46_events,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
    evaluation = evaluate(model, loader, device, str(config["amp_dtype"]))
    row_by_source = {str(row["source_id"]): row for row in dataset.rows}
    source_ids = np.asarray(evaluation["source_ids"])
    sample_ids = np.asarray([row_by_source[value]["sample_id"] for value in source_ids])
    labels = np.asarray(
        [int(row_by_source[value]["class_id"]) for value in source_ids], dtype=np.int64
    )
    detail_labels = np.asarray(evaluation["labels"], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(HARD_CLASS_IDS, dtype=np.int64)[detail_labels]):
        raise RuntimeError("Exported evaluation labels disagree with frozen manifest")
    logits = np.asarray(evaluation["logits"], dtype=np.float32)
    prediction = logits.argmax(axis=1)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=sample_ids,
            source_ids=source_ids,
            labels=labels,
            users=np.asarray(evaluation["users"]),
            logits=logits,
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[prediction],
            checkpoint_epoch=np.asarray(int(checkpoint["epoch"]), dtype=np.int64),
        )
    temporary.replace(output)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "samples": int(len(labels)),
        "correct": int((prediction == detail_labels).sum()),
        "accuracy": float((prediction == detail_labels).mean()),
        "users": sorted(set(evaluation["users"])),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
