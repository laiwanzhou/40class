from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import yaml

from scripts.run_x3d_s_fold0_dev import (
    CANONICAL_ARTIFACTS,
    CANONICAL_SEED,
    collect_canonical_artifact_hashes,
)
from src.data.x3d_clip_dataset import X3DClipDataset
import src.train_x3d_s_visual_expert as trainer


DEV_OUTPUT_ROOT = Path("outputs/x3d_s_ir_context_train12_val2_dev")
EXPECTED_TRAIN_USERS = (
    "user1", "user2", "user3", "user5", "user6", "user7",
    "user8", "user9", "user16", "user18", "user19", "user20",
)
EXPECTED_VALIDATION_USERS = ("user21", "user22")


def validate_split_contract(
    split: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train = tuple(str(user) for user in split.get("train_user_ids", ()))
    validation = tuple(str(user) for user in split.get("validation_user_ids", ()))
    if train != EXPECTED_TRAIN_USERS or validation != EXPECTED_VALIDATION_USERS:
        raise ValueError("Development split must match the exact frozen train12/val2 users")
    if not bool(split.get("development_only", False)):
        raise ValueError("Development split must be explicitly development_only")
    metric_policy = split.get("metric_policy", {})
    if not isinstance(metric_policy, Mapping) or int(metric_policy.get("num_classes", -1)) != 40:
        raise ValueError("Development split must freeze the 40-class metric policy")
    if set(train) & set(validation):
        raise ValueError("Development split users must be disjoint")
    return train, validation


def resolve_effective_config(
    config: Mapping[str, Any],
    *,
    split: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    if Path(str(config.get("output_root", ""))) != DEV_OUTPUT_ROOT:
        raise ValueError(f"Train12/val2 development must use protected output root {DEV_OUTPUT_ROOT}")
    if int(seed) != CANONICAL_SEED:
        raise ValueError(f"Train12/val2 development is frozen to seed {CANONICAL_SEED}")
    train, validation = validate_split_contract(split)
    resolved = dict(config)
    resolved["seed"] = int(seed)
    resolved["development_partition"] = {
        "train_user_ids": list(train),
        "validation_user_ids": list(validation),
    }
    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run protected X3D-S development on the shared train12/val2 split"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-split", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    split = json.loads(args.development_split.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping) or not isinstance(split, Mapping):
        raise ValueError("Config and development split roots must be mappings")
    config = resolve_effective_config(raw_config, split=split, seed=int(args.seed))
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a non-empty path-safe name")

    if args.smoke_test:
        config["training"] = {
            **dict(config["training"]),
            "epochs": 2,
            "scheduler_horizon_epochs": 2,
            "warmup_epochs": 1,
            "early_stopping_enabled": False,
        }
    trainer.validate_config(config)

    canonical_before = collect_canonical_artifact_hashes(CANONICAL_ARTIFACTS)
    partition = config["development_partition"]
    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    partition_manifest = trainer.prepare_partition_manifest(
        manifest,
        train_user_ids=partition["train_user_ids"],
        validation_user_ids=partition["validation_user_ids"],
    )
    augmentation = config.get("augmentation")
    train_dataset = X3DClipDataset(
        partition_manifest,
        split="train",
        training=True,
        augmentation_config=augmentation,
        train_clip_keep_fraction=1.0,
        seed=CANONICAL_SEED,
    )
    validation_dataset = X3DClipDataset(
        partition_manifest,
        split="val",
        training=False,
        augmentation_config=augmentation,
        train_clip_keep_fraction=1.0,
        seed=CANONICAL_SEED,
    )
    if set(train_dataset.sample_ids) & set(validation_dataset.sample_ids):
        raise RuntimeError("Train12/val2 trial ownership overlaps")

    run_directory = trainer.prepare_run_directory(DEV_OUTPUT_ROOT, str(args.run_id))
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "role": "train12_val2_development_tuning",
        "unbiased_oof": False,
        "canonical_evidence_mutation_permitted": False,
        "seed": CANONICAL_SEED,
        "train_user_ids": partition["train_user_ids"],
        "validation_user_ids": partition["validation_user_ids"],
        "train_trial_count": len(train_dataset),
        "validation_trial_count": len(validation_dataset),
        "train_class_count": 40,
        "validation_class_count": 36,
        "validation_missing_class_ids": [25, 26, 33, 35],
        "development_split_sha256": trainer._sha256_file(args.development_split),
        "resolved_config_sha256": trainer.resolved_config_sha256(config),
        "temporal_training_policy": {
            "train_clip_keep_fraction": 1.0,
            "validation_clip_keep_fraction": 1.0,
            "aggregation": "mean_probability",
        },
        "canonical_artifact_sha256_before": canonical_before,
        "smoke_test": bool(args.smoke_test),
    }
    provenance_path = run_directory / "development_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    trainer._set_seed(CANONICAL_SEED)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summary = trainer.train_partition(
        model=trainer._build_model(config),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=config,
        run_directory=run_directory,
        device=device,
        max_train_batches=1 if args.smoke_test else None,
        max_val_batches=1 if args.smoke_test else None,
    )

    canonical_after = collect_canonical_artifact_hashes(CANONICAL_ARTIFACTS)
    if canonical_after != canonical_before:
        raise RuntimeError("Canonical Phase 4/5 artifact hash changed during development")
    provenance["canonical_artifact_sha256_after"] = canonical_after
    provenance["canonical_artifacts_unchanged"] = True
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
