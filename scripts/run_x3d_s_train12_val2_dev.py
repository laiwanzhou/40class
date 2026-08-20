from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MappingProxyType
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
SPLIT_PROFILES = MappingProxyType({
    "train12_val2_user21_user22": {
        "train_user_ids": (
            "user1", "user2", "user3", "user5", "user6", "user7",
            "user8", "user9", "user16", "user18", "user19", "user20",
        ),
        "validation_user_ids": ("user21", "user22"),
        "heldout_user_ids": None,
        "ir_audit": {
            "train_usable_trials": 1996,
            "validation_usable_trials": 324,
            "train_class_count": 40,
            "validation_class_count": 36,
            "validation_missing_class_ids": (25, 26, 33, 35),
        },
    },
    "train12_val2_user6_user7": {
        "train_user_ids": (
            "user1", "user2", "user3", "user5", "user8", "user9",
            "user16", "user18", "user19", "user20", "user21", "user22",
        ),
        "validation_user_ids": ("user6", "user7"),
        "heldout_user_ids": ("user4", "user17", "user23", "user24"),
        "ir_audit": {
            "train_usable_trials": 1935,
            "validation_usable_trials": 385,
            "train_class_count": 40,
            "validation_class_count": 40,
            "validation_missing_class_ids": (),
            "validation_minimum_class_support": 2,
        },
    },
})


def validate_split_contract(
    split: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    name = str(split.get("name", ""))
    if name not in SPLIT_PROFILES:
        raise ValueError("Development split must use a registered named profile")
    profile = SPLIT_PROFILES[name]
    train = tuple(str(user) for user in split.get("train_user_ids", ()))
    validation = tuple(str(user) for user in split.get("validation_user_ids", ()))
    if (
        train != profile["train_user_ids"]
        or validation != profile["validation_user_ids"]
    ):
        raise ValueError("Development split must match the exact frozen train12/val2 users")
    expected_heldout = profile["heldout_user_ids"]
    if expected_heldout is not None:
        heldout = tuple(str(user) for user in split.get("heldout_user_ids", ()))
        if heldout != expected_heldout:
            raise ValueError("Development split heldout users changed")
    if not bool(split.get("development_only", False)):
        raise ValueError("Development split must be explicitly development_only")
    metric_policy = split.get("metric_policy", {})
    if not isinstance(metric_policy, Mapping) or int(metric_policy.get("num_classes", -1)) != 40:
        raise ValueError("Development split must freeze the 40-class metric policy")
    if expected_heldout is not None and tuple(
        str(user) for user in metric_policy.get("worst_user_population", ())
    ) != validation:
        raise ValueError("Development split worst-user population changed")
    audit = split.get("ir_audit", {})
    if not isinstance(audit, Mapping):
        raise ValueError("Development split IR audit must be a mapping")
    normalized_audit = {
        key: tuple(value) if key == "validation_missing_class_ids" else value
        for key, value in audit.items()
    }
    if normalized_audit != profile["ir_audit"]:
        raise ValueError("Development split IR audit changed")
    heldout_set = set(expected_heldout or ())
    if set(train) & set(validation) or set(train) & heldout_set or set(validation) & heldout_set:
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
    resolved["development_split_name"] = str(split["name"])
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
        temporal_sampling_mode=trainer._resolved_temporal_sampling_mode(config),
        **trainer._dataset_spatial_input_kwargs(config),
        seed=CANONICAL_SEED,
    )
    validation_dataset = X3DClipDataset(
        partition_manifest,
        split="val",
        training=False,
        augmentation_config=augmentation,
        train_clip_keep_fraction=1.0,
        temporal_sampling_mode=trainer._resolved_temporal_sampling_mode(config),
        **trainer._dataset_spatial_input_kwargs(config),
        seed=CANONICAL_SEED,
    )
    if set(train_dataset.sample_ids) & set(validation_dataset.sample_ids):
        raise RuntimeError("Train12/val2 trial ownership overlaps")
    audit = split["ir_audit"]
    if len(train_dataset) != int(audit["train_usable_trials"]):
        raise RuntimeError("Train trial count differs from frozen split IR audit")
    if len(validation_dataset) != int(audit["validation_usable_trials"]):
        raise RuntimeError("Validation trial count differs from frozen split IR audit")

    run_directory = trainer.prepare_run_directory(DEV_OUTPUT_ROOT, str(args.run_id))
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    spatial_input_policy = dict(config.get("spatial_input", {}))
    pose_cache_path = spatial_input_policy.get("pose_cache")
    if pose_cache_path is not None:
        spatial_input_policy["pose_cache_sha256"] = trainer._sha256_file(
            Path(str(pose_cache_path))
        )
    provenance = {
        "schema_version": 1,
        "role": "train12_val2_development_tuning",
        "unbiased_oof": False,
        "canonical_evidence_mutation_permitted": False,
        "seed": CANONICAL_SEED,
        "development_split_name": str(split["name"]),
        "development_split_path": str(args.development_split.resolve()),
        "train_user_ids": partition["train_user_ids"],
        "validation_user_ids": partition["validation_user_ids"],
        "heldout_user_ids": list(split.get("heldout_user_ids", ())),
        "train_trial_count": len(train_dataset),
        "validation_trial_count": len(validation_dataset),
        "train_class_count": int(audit["train_class_count"]),
        "validation_class_count": int(audit["validation_class_count"]),
        "validation_missing_class_ids": list(audit["validation_missing_class_ids"]),
        "development_split_sha256": trainer._sha256_file(args.development_split),
        "resolved_config_sha256": trainer.resolved_config_sha256(config),
        "temporal_training_policy": {
            "sampling_mode": trainer._resolved_temporal_sampling_mode(config),
            "train_clip_keep_fraction": 1.0,
            "validation_clip_keep_fraction": 1.0,
            "aggregation": "mean_probability",
        },
        "spatial_input_policy": spatial_input_policy,
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
