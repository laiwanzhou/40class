from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import yaml

from src.data.x3d_clip_dataset import X3DClipDataset
import src.train_x3d_s_visual_expert as trainer


FOLD_INDEX = 2
CANONICAL_SEED = 20260715


def select_fold2(assignment: Mapping[str, Any], *, allowed_users: set[str]) -> trainer.UserFold:
    folds = trainer.validate_oof_assignment(assignment, allowed_users=allowed_users)
    selected = [fold for fold in folds if fold.fold == FOLD_INDEX]
    if len(selected) != 1:
        raise ValueError("Frozen OOF assignment must contain exactly one fold 2")
    return selected[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only frozen strict-v3 X3D-S OOF fold 2 without rerunning folds 0/1"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oof-fold-assignment", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    return parser


def run(args: argparse.Namespace) -> None:
    if int(args.seed) != CANONICAL_SEED:
        raise ValueError(f"fold-2-only continuation is frozen to seed {CANONICAL_SEED}")

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")
    config = dict(config)
    config["seed"] = CANONICAL_SEED
    trainer.validate_config(config)

    trainer._set_seed(CANONICAL_SEED)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    split_path = Path(str(config["split_path"]))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    official_train_users = set(str(user) for user in split["train_users"])

    assignment = json.loads(args.oof_fold_assignment.read_text(encoding="utf-8"))
    partition = select_fold2(assignment, allowed_users=official_train_users)
    assignment_sha256 = trainer._sha256_file(args.oof_fold_assignment)

    output_root = Path(str(config["output_root"]))
    parent_run_directory = trainer.prepare_run_directory(output_root, args.run_id)
    partition_directory = parent_run_directory / "fold_2"
    partition_directory.mkdir(parents=False, exist_ok=False)

    (parent_run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    continuation_provenance = {
        "schema_version": 1,
        "role": "phase4_fold2_only_continuation",
        "source_run_id": str(args.source_run_id),
        "continuation_run_id": str(args.run_id),
        "selected_fold": FOLD_INDEX,
        "seed": CANONICAL_SEED,
        "assignment_sha256": assignment_sha256,
        "protocol_change": False,
        "purpose": "Continue untouched fold 2 without rerunning or overwriting completed folds 0/1",
        "fold0_or_fold1_training_permitted": False,
    }
    (parent_run_directory / "continuation_provenance.json").write_text(
        json.dumps(continuation_provenance, indent=2) + "\n", encoding="utf-8"
    )

    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    outer_manifest = trainer.prepare_partition_manifest(
        manifest,
        train_user_ids=partition.train_user_ids,
        validation_user_ids=partition.validation_user_ids,
    )
    outer_train_dataset = X3DClipDataset(
        outer_manifest, split="train", training=True, seed=CANONICAL_SEED
    )
    outer_validation_dataset = X3DClipDataset(
        outer_manifest, split="val", training=False, seed=CANONICAL_SEED
    )

    raw_fold = assignment["folds"][FOLD_INDEX]
    epoch_selection = raw_fold.get("epoch_selection")
    if not isinstance(epoch_selection, Mapping):
        raise ValueError("Frozen fold 2 has no epoch_selection")
    inner_fit_users = tuple(str(user) for user in epoch_selection["fit_user_ids"])
    inner_validation_users = tuple(
        str(user) for user in epoch_selection["validation_user_ids"]
    )
    if (
        set(inner_fit_users) & set(inner_validation_users)
        or set(inner_fit_users) | set(inner_validation_users) != set(partition.train_user_ids)
    ):
        raise ValueError("Frozen fold-2 epoch-selection users must partition outer-train")

    inner_manifest = trainer.prepare_partition_manifest(
        manifest,
        train_user_ids=inner_fit_users,
        validation_user_ids=inner_validation_users,
    )
    inner_fit_dataset = X3DClipDataset(
        inner_manifest, split="train", training=True, seed=CANONICAL_SEED
    )
    inner_validation_dataset = X3DClipDataset(
        inner_manifest, split="val", training=False, seed=CANONICAL_SEED
    )

    summary = trainer.train_strict_oof_partition(
        model_factory=lambda: trainer._build_model(config),
        inner_fit_dataset=inner_fit_dataset,
        inner_validation_dataset=inner_validation_dataset,
        outer_train_dataset=outer_train_dataset,
        outer_validation_dataset=outer_validation_dataset,
        config=config,
        run_directory=partition_directory,
        device=device,
        fold_provenance={
            "outer_fold": FOLD_INDEX,
            "inner_fit_user_ids": list(inner_fit_users),
            "inner_validation_user_ids": list(inner_validation_users),
            "outer_train_user_ids": list(partition.train_user_ids),
            "outer_validation_user_ids": list(partition.validation_user_ids),
            "assignment_sha256": assignment_sha256,
            "continuation_source_run_id": str(args.source_run_id),
            "continuation_run_id": str(args.run_id),
        },
        max_train_batches=None,
        max_val_batches=None,
    )
    summary["fold"] = FOLD_INDEX
    summary["train_user_ids"] = list(partition.train_user_ids)
    summary["validation_user_ids"] = list(partition.validation_user_ids)
    (parent_run_directory / "partition_summaries.json").write_text(
        json.dumps([summary], indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
