"""Refit one deployable P87-S Student on all available P150 OOF targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from adapt_p87s_structured_student import (
    LabelFreePseudoDataset,
    make_loader,
    metric_delta,
    model_build_args,
    resolve_config_path,
    train_label_free,
)
from p86_cached_motion_data import P86CachedSequenceMotionDataset
from train_p86_mobind_fusion_proxy import build_model, evaluate


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "runs/p87s_fusion_all2914_v1/unified_student.pt"
DEFAULT_TARGETS = HERE / "runs/p162_p150_student_targets_v1/all_structured_targets.npz"
DEFAULT_OUTPUT = HERE / "runs/p162_p150_student_final_refit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--structured-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", choices=("structured",), default="structured")
    parser.add_argument("--adaptation-scope", default="heads_motion_encoder")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-4)
    parser.add_argument("--visual-head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--emission-warmup-epochs", type=int, default=0)
    parser.add_argument("--confidence-power", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    base_path = args.base_checkpoint.resolve()
    base_summary = json.loads((base_path.parent / "summary.json").read_text(encoding="utf-8"))
    if base_summary.get("stage") != "P87S_mobind_fusion_all2914_refit":
        raise ValueError("final P162 refit requires the frozen all-2914 P87-S base")
    build_args = model_build_args(base_path, base_summary)
    model, visual_config, pretrain_config = build_model(build_args)
    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    model.load_state_dict(base_checkpoint["model_state"], strict=True)

    config = base_summary["config"]
    repository_root = HERE.parent
    full = P86CachedSequenceMotionDataset(
        resolve_config_path(config["sequence_cache"], repository_root),
        resolve_config_path(config["motion_cache"], repository_root),
        resolve_config_path(config["pixel_cache"], repository_root),
        resolve_config_path(config["teacher_features"], repository_root),
        resolve_config_path(config["teacher_logits"], repository_root),
    )
    targets = np.load(args.structured_targets.resolve(), allow_pickle=False)
    selected_ids = targets["sample_ids"].astype(str)[targets["target_mask"].astype(bool)]
    if len(selected_ids) != 2470 or len(np.unique(selected_ids)) != 2470:
        raise RuntimeError("combined P150 target universe changed")
    missing = sorted(set(selected_ids.tolist()) - set(full.index_lookup))
    if missing:
        raise RuntimeError(f"all-2914 Student cache misses targets: {missing[:3]}")
    indices = np.asarray([full.index_lookup[value] for value in selected_ids], dtype=np.int64)
    probability = targets["structured_distillation_probability"].astype(np.float32)
    confidence = targets["structured_confidence"].astype(np.float32)
    probability_by_id = dict(zip(selected_ids, probability))
    confidence_by_id = {key: float(value) for key, value in zip(selected_ids, confidence)}

    def dataset(augment: bool) -> P86CachedSequenceMotionDataset:
        return P86CachedSequenceMotionDataset(
            full.sequence_cache,
            full.motion_cache,
            resolve_config_path(config["pixel_cache"], repository_root),
            resolve_config_path(config["teacher_features"], repository_root),
            resolve_config_path(config["teacher_logits"], repository_root),
            indices=indices,
            temporal_augment=augment,
        )

    evaluation = dataset(False)
    pseudo = LabelFreePseudoDataset(dataset(True), probability_by_id, confidence_by_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    base_metrics, base_rows, base_logits = evaluate(
        model,
        make_loader(evaluation, args.batch_size, args.workers, False),
        device,
        args.max_eval_batches,
    )
    history = train_label_free(
        model, make_loader(pseudo, args.batch_size, args.workers, True), args, device
    )
    adapted_metrics, rows, logits = evaluate(
        model,
        make_loader(evaluation, args.batch_size, args.workers, False),
        device,
        args.max_eval_batches,
    )
    target_prediction = probability.argmax(axis=1)
    prediction = logits.argmax(axis=1)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": "P162_P150_final_student_refit",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "visual_config": visual_config,
        "pretrain_config": pretrain_config,
        "modality": base_summary["modality"],
        "base_checkpoint": str(base_path),
        "target_source": str(args.structured_targets.resolve()),
    }
    checkpoint_path = output / "model.pth"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    if checkpoint_bytes >= 100_000_000:
        raise RuntimeError(f"final checkpoint exceeds 100 MB: {checkpoint_bytes}")
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(reloaded["model_state"]) != set(checkpoint["model_state"]):
        raise RuntimeError("final checkpoint reload changed state keys")
    write_rows(output / "target_fit_predictions.csv", rows)
    np.save(output / "target_fit_logits.npy", logits)
    report = {
        "stage": "P162_P150_final_student_refit",
        "status": "complete_single_checkpoint_under_100MB",
        "protocol": (
            "Initialize one Student from the all-2914 true-label refit, then adapt on "
            "2470 strict P150 OOF targets with no ground-truth field in adaptation batches."
        ),
        "validation_note": (
            "The target-fit metric below is in-sample diagnostic only. The honest "
            "subject-disjoint score is the separate three-cohort P162 OOF audit."
        ),
        "target_rows": len(selected_ids),
        "ground_truth_fields_seen_during_adaptation": 0,
        "base_target_rows_metrics_diagnostic": base_metrics,
        "adapted_target_rows_metrics_diagnostic": adapted_metrics,
        "delta_diagnostic": metric_delta(base_metrics, adapted_metrics),
        "teacher_agreement_diagnostic": float(np.mean(prediction == target_prediction)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mb_decimal": checkpoint_bytes / 1_000_000,
        "checkpoint_sha256": sha256(checkpoint_path),
        "single_checkpoint_under_100000000_bytes": checkpoint_bytes < 100_000_000,
        "large_teacher_required_at_inference": False,
        "config": vars(args),
    }
    report["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["config"].items()
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
