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


SEED = 20260717
FOLD1_SELECTED_EPOCH = 10


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refit seed17 fold 1 at epoch 10, then run fold 2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oof-fold-assignment", type=Path, required=True)
    parser.add_argument("--source-run-directory", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _datasets(
    manifest: pd.DataFrame,
    fold: trainer.UserFold,
    *,
    seed: int,
) -> tuple[X3DClipDataset, X3DClipDataset]:
    frame = trainer.prepare_partition_manifest(
        manifest,
        train_user_ids=fold.train_user_ids,
        validation_user_ids=fold.validation_user_ids,
    )
    return (
        X3DClipDataset(frame, split="train", training=True, seed=seed),
        X3DClipDataset(frame, split="val", training=False, seed=seed),
    )


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")
    config = dict(config)
    config["seed"] = SEED
    trainer.validate_config(config)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    split = json.loads(Path(str(config["split_path"])).read_text(encoding="utf-8"))
    allowed_users = set(str(user) for user in split["train_users"])
    assignment = json.loads(args.oof_fold_assignment.read_text(encoding="utf-8"))
    folds = trainer.validate_oof_assignment(assignment, allowed_users=allowed_users)
    fold1, fold2 = folds[1], folds[2]
    assignment_sha = trainer._sha256_file(args.oof_fold_assignment)

    source = args.source_run_directory.resolve()
    source_history = source / "fold_1" / "epoch_selection" / "history.csv"
    if not source_history.is_file():
        raise FileNotFoundError(source_history)
    rows = pd.read_csv(source_history)
    if rows["epoch"].tolist() != list(range(1, 29)):
        raise ValueError("Source fold-1 history must contain exactly epochs 1..28")
    selected = rows.loc[rows["epoch"] == FOLD1_SELECTED_EPOCH].iloc[0]
    if float(selected["val_accuracy"]) != float(rows["val_accuracy"].max()):
        raise ValueError("Frozen epoch 10 is not the best observed fold-1 Accuracy")

    output_root = Path(str(config["output_root"]))
    run_directory = trainer.prepare_run_directory(output_root, args.run_id)
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "role": "seed17_manual_fold1_refit_then_strict_fold2",
        "seed": SEED,
        "source_run_directory": str(source),
        "source_history_sha256": trainer._sha256_file(source_history),
        "assignment_sha256": assignment_sha,
        "fold1_selected_epoch": FOLD1_SELECTED_EPOCH,
        "fold1_selection_policy": "human_approved_best_observed_accuracy_after_28_of_30_epochs",
        "protocol_deviation": True,
        "canonical_phase5_evidence": False,
    }
    (run_directory / "continuation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    fold1_directory = run_directory / "fold_1"
    fold1_directory.mkdir()
    fold1_train, fold1_val = _datasets(manifest, fold1, seed=SEED)
    fold1_summary = trainer.refit_strict_oof_partition(
        model_factory=lambda: trainer._build_model(config),
        outer_train_dataset=fold1_train,
        outer_validation_dataset=fold1_val,
        config=config,
        run_directory=fold1_directory,
        device=device,
        selected_epoch=FOLD1_SELECTED_EPOCH,
        fold_provenance={
            "outer_fold": 1,
            "outer_train_user_ids": list(fold1.train_user_ids),
            "outer_validation_user_ids": list(fold1.validation_user_ids),
            "assignment_sha256": assignment_sha,
            "selection_policy": provenance["fold1_selection_policy"],
            "source_history_sha256": provenance["source_history_sha256"],
        },
        selection_summary={
            "status": "incomplete_human_frozen",
            "epochs_completed": 28,
            "selected_epoch": FOLD1_SELECTED_EPOCH,
            "selected_accuracy": float(selected["val_accuracy"]),
            "selected_macro_f1": float(selected["val_macro_f1"]),
        },
        max_train_batches=None,
        max_val_batches=None,
    )
    fold1_summary["fold"] = 1

    raw_fold2 = assignment["folds"][2]
    epoch_selection = raw_fold2.get("epoch_selection")
    if not isinstance(epoch_selection, Mapping):
        raise ValueError("Frozen fold 2 has no epoch_selection")
    inner_fit = tuple(str(user) for user in epoch_selection["fit_user_ids"])
    inner_val = tuple(str(user) for user in epoch_selection["validation_user_ids"])
    inner_frame = trainer.prepare_partition_manifest(
        manifest, train_user_ids=inner_fit, validation_user_ids=inner_val
    )
    fold2_train, fold2_val = _datasets(manifest, fold2, seed=SEED)
    fold2_directory = run_directory / "fold_2"
    fold2_directory.mkdir()
    fold2_summary = trainer.train_strict_oof_partition(
        model_factory=lambda: trainer._build_model(config),
        inner_fit_dataset=X3DClipDataset(inner_frame, split="train", training=True, seed=SEED),
        inner_validation_dataset=X3DClipDataset(inner_frame, split="val", training=False, seed=SEED),
        outer_train_dataset=fold2_train,
        outer_validation_dataset=fold2_val,
        config=config,
        run_directory=fold2_directory,
        device=device,
        fold_provenance={
            "outer_fold": 2,
            "inner_fit_user_ids": list(inner_fit),
            "inner_validation_user_ids": list(inner_val),
            "outer_train_user_ids": list(fold2.train_user_ids),
            "outer_validation_user_ids": list(fold2.validation_user_ids),
            "assignment_sha256": assignment_sha,
        },
        max_train_batches=None,
        max_val_batches=None,
    )
    fold2_summary["fold"] = 2
    (run_directory / "partition_summaries.json").write_text(
        json.dumps([fold1_summary, fold2_summary], indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
