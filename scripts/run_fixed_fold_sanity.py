from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.x3d_clip_dataset import X3DClipDataset
import src.train_x3d_s_visual_expert as trainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one fixed-epoch outer-fold sanity experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> None:
    if args.fold != 0 or args.epochs != 10 or args.seed != 20260715:
        raise ValueError("Phase-4 sanity experiment is frozen to fold 0, 10 epochs, seed 20260715")
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    config = dict(config)
    config["seed"] = args.seed
    trainer.validate_config(config)
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if amendment.get("replacement_protocol", {}).get("epochs") != args.epochs:
        raise ValueError("Amendment does not authorize the requested epoch budget")

    split = json.loads(Path(str(config["split_path"])).read_text(encoding="utf-8"))
    assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
    folds = trainer.validate_oof_assignment(
        assignment, allowed_users=set(str(user) for user in split["train_users"])
    )
    fold = folds[args.fold]
    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    frame = trainer.prepare_partition_manifest(
        manifest,
        train_user_ids=fold.train_user_ids,
        validation_user_ids=fold.validation_user_ids,
    )
    train_dataset = X3DClipDataset(frame, split="train", training=True, seed=args.seed)
    validation_dataset = X3DClipDataset(frame, split="val", training=False, seed=args.seed)
    run_directory = trainer.prepare_run_directory(Path(str(config["output_root"])), args.run_id)
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "role": "phase4_fixed_budget_matched_baseline_sanity",
        "amendment_sha256": trainer._sha256_file(args.amendment),
        "assignment_sha256": trainer._sha256_file(args.assignment),
        "fold": args.fold,
        "epochs": args.epochs,
        "seed": args.seed,
        "outer_validation_evaluations": 1,
        "threshold_accuracy": 0.53,
    }
    (run_directory / "sanity_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    summary = trainer.refit_strict_oof_partition(
        model_factory=lambda: trainer._build_model(config),
        outer_train_dataset=train_dataset,
        outer_validation_dataset=validation_dataset,
        config=config,
        run_directory=run_directory,
        device=torch.device(str(config["device"])),
        selected_epoch=args.epochs,
        fold_provenance={
            "outer_fold": args.fold,
            "outer_train_user_ids": list(fold.train_user_ids),
            "outer_validation_user_ids": list(fold.validation_user_ids),
            "assignment_sha256": provenance["assignment_sha256"],
            "selection_policy": "fixed_compute_budget_no_validation_selection",
            "amendment_sha256": provenance["amendment_sha256"],
        },
        selection_summary={
            "status": "not_applicable_fixed_budget",
            "selected_epoch": args.epochs,
            "outer_validation_used_for_selection": False,
        },
        max_train_batches=None,
        max_val_batches=None,
    )
    summary["sanity_threshold_accuracy"] = 0.53
    summary["sanity_decision"] = (
        "stop_baseline_compute_keep_x3d"
        if float(summary["formal_outer_accuracy"]) < 0.53
        else "unexpectedly_close_reconsider_full_matched_experiment"
    )
    (run_directory / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
