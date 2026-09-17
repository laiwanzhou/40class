"""Fold-local diagnostic that separates LoRA backbone changes from head drift.

The P90 LoRA screen initialized its six position-specific heads from the frozen
P85 early/late Ridge teacher, but then optimized one clip at a time.  Final
evaluation averages all six clips, so that training objective can destroy a
good joint head even when the adapted backbone remains useful.  This script
loads only the trained LoRA/LayerNorm tensors, restores the original fold-local
Ridge head, and evaluates the held subject fold once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomae_lora_teacher import (
    MODEL_NAME,
    EvalTrialDataset,
    VideoCollator,
    build_model_and_processor,
    evaluate,
    initialize_ir_head_from_fold_ridge,
    read_aligned_rows,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "p90_videomae_lora_teacher_v1"
    / "checkpoints"
    / "videomae_large_ir_lora_r8_fold0.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "p90_videomae_lora_head_rescue_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--modality", choices=("ir",), default="ir")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.10)
    parser.add_argument(
        "--train-layernorm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--eval-trial-batch", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    fold = int(checkpoint["fold"])
    if checkpoint["modality"] != args.modality:
        raise ValueError("checkpoint modality does not match")

    protocol = load_protocol()
    rows = read_aligned_rows()
    train_indices = protocol.train_indices(fold)
    val_indices = protocol.val_indices(fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, snapshot, bias_report, lora_report = build_model_and_processor(
        args, device
    )
    ridge_report = initialize_ir_head_from_fold_ridge(
        model.classifier, protocol.sample_ids, protocol.labels, train_indices
    )

    restored = []
    skipped = []
    current_state = model.state_dict()
    for name, value in checkpoint["trainable_state"].items():
        if name.startswith("classifier."):
            skipped.append(name)
            continue
        if name not in current_state or current_state[name].shape != value.shape:
            raise ValueError(f"checkpoint tensor mismatch: {name}")
        current_state[name].copy_(value)
        restored.append(name)

    dataset = EvalTrialDataset(
        [rows[index] for index in val_indices],
        protocol.labels[val_indices],
        args.modality,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_trial_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VideoCollator(processor),
        pin_memory=True,
    )
    logits, labels, sample_ids = evaluate(model, loader, device)
    if sample_ids != protocol.sample_ids[val_indices].tolist():
        raise ValueError("validation sample order changed")
    if not np.array_equal(labels, protocol.labels[val_indices]):
        raise ValueError("validation labels changed")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(logits, labels)
    artifact = output / f"videomae_large_ir_lora_backbone_ridge_fold{fold}.npz"
    np.savez_compressed(
        artifact,
        sample_ids=np.asarray(sample_ids),
        labels=labels,
        fold_id=np.full(len(labels), fold, dtype=np.int8),
        logits=logits.astype(np.float32),
    )
    report = {
        "protocol": "fold-local LoRA/LayerNorm backbone with the untouched P85 six-clip Ridge head",
        "fold": fold,
        "metrics": metrics,
        "checkpoint": str(args.checkpoint.resolve()),
        "artifact": str(artifact),
        "snapshot": str(snapshot),
        "restored_backbone_tensors": len(restored),
        "skipped_trained_head_tensors": skipped,
        "ridge_initialization": ridge_report,
        "attention_bias": bias_report,
        "lora": lora_report,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main(parse_args())
