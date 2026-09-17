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
    model_build_args,
    seed_all,
    train_label_free,
)
from p87s_test_data import P87STestCachedSequenceMotionDataset
from p87s_deploy_model import deployment_model_config
from train_p86_mobind_fusion_proxy import build_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGETS = PROJECT_DIR / "runs/p87s_test_structured_targets_v1/structured_targets.npz"
DEFAULT_SEQUENCE = PROJECT_DIR / "runs/p87s_test_mc3_sequence_v1"
DEFAULT_MOTION = PROJECT_DIR / "runs/p87s_test_motion_window_t16_v1"
DEFAULT_PIXELS = PROJECT_DIR / "runs/p87s_test_pixel_cache_t16_r160_v1"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p87s_test_adapt_structured12_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-recipe label-free adaptation of the all-2914 P87-S Student on "
            "the 401 Test rows with precomputed structured posterior targets."
        )
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--structured-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--sequence-cache", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fusion-learning-rate", type=float, default=5e-5)
    parser.add_argument("--visual-head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence-power", type=float, default=0.0)
    parser.add_argument(
        "--adaptation-scope",
        choices=("heads", "heads_motion_encoder"),
        default="heads",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.temperature <= 0:
        raise ValueError("epochs and temperature must be positive")
    if args.confidence_power < 0:
        raise ValueError("confidence-power must be nonnegative")
    if args.smoke:
        args.epochs = 1
        args.max_train_batches = args.max_train_batches or 2
    # Reuse the already-audited label-free trainer's structured branch.
    args.target = "structured"
    args.emission_warmup_epochs = 0
    seed_all(args.seed)

    base_checkpoint_path = args.base_checkpoint.resolve()
    base_dir = base_checkpoint_path.parent
    base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    if base_summary.get("stage") != "P87S_mobind_fusion_all2914_refit":
        raise ValueError("base checkpoint is not the terminal all-2914 P87-S fusion refit")
    build_args = model_build_args(base_checkpoint_path, base_summary)
    model, visual_config, pretrain_config = build_model(build_args)
    base_checkpoint = torch.load(
        base_checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(base_checkpoint["model_state"], strict=True)

    full = P87STestCachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        temporal_augment=False,
    )
    with np.load(args.structured_targets.resolve(), allow_pickle=False) as targets:
        target_ids = targets["sample_ids"].astype(str)
        target_mask = targets["target_mask"].astype(bool)
        selected_ids = target_ids[target_mask]
        probability = targets["structured_distillation_probability"].astype(np.float32)[
            target_mask
        ]
        confidence = targets["structured_confidence"].astype(np.float32)[target_mask]
    if len(selected_ids) not in (401, 405):
        raise RuntimeError("final adaptation requires 401 or 405 structured targets")
    missing = sorted(set(selected_ids.tolist()) - set(full.index_lookup))
    if missing:
        raise RuntimeError(f"final Student cache is missing targets: {missing[:3]}")
    indices = np.asarray([full.index_lookup[value] for value in selected_ids], dtype=np.int64)
    pseudo_base = P87STestCachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        indices=indices,
        temporal_augment=True,
    )
    probability_by_id = dict(zip(selected_ids, probability, strict=True))
    confidence_by_id = {
        sample_id: float(value)
        for sample_id, value in zip(selected_ids, confidence, strict=True)
    }
    pseudo_dataset = LabelFreePseudoDataset(
        pseudo_base,
        probability_by_id,
        confidence_by_id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = train_label_free(
        model,
        make_loader(pseudo_dataset, args.batch_size, args.workers, shuffle=True),
        args,
        device,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = {
        "stage": "P87S_label_free_test_adaptation",
        "target": "structured",
        "adaptation_scope": args.adaptation_scope,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "visual_config": visual_config,
        "pretrain_config": pretrain_config,
        "modality": base_summary["modality"],
        "base_checkpoint": str(base_checkpoint_path),
        "deployment_model_config": deployment_model_config(
            visual_config,
            pretrain_config,
            base_summary["modality"],
            base_summary["config"],
        ),
    }
    checkpoint_path = output / "unified_student.pt"
    torch.save(checkpoint, checkpoint_path)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "stage": "P87S_label_free_test_adaptation",
        "status": "smoke" if args.smoke else "formal_frozen_recipe",
        "protocol": (
            f"Start from the all-2914 true-label Student refit, then adapt only on {len(selected_ids)} "
            "unlabeled Test inputs with structured targets whose decoder recipe was "
            "frozen by nested Train OOF. No Test labels, Large model, old submission "
            "prediction or Test ground truth enters adaptation."
        ),
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_sha256": sha256(base_checkpoint_path),
        "structured_targets": str(args.structured_targets.resolve()),
        "structured_targets_sha256": sha256(args.structured_targets.resolve()),
        "pseudo_rows": len(selected_ids),
        "nonadapted_test_rows": len(full) - len(selected_ids),
        "ground_truth_fields_seen_during_training": 0,
        "teacher_model_executed_during_training": False,
        "parameters": parameters,
        "fp32_mib": parameters * 4 / 1024**2,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
