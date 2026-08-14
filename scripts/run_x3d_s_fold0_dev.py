from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import yaml

from src.data.x3d_clip_dataset import X3DClipDataset
import src.train_x3d_s_visual_expert as trainer


FOLD_INDEX = 0
CANONICAL_SEED = 20260715
DEV_OUTPUT_ROOT = Path("outputs/x3d_s_ir_context_fold0_dev")
FORBIDDEN_RUN_ID_PARTS = ("strict_v3", "phase4", "phase5")
CANONICAL_ARTIFACTS = (
    Path("metadata/splits/train14_oof_3fold.json"),
    Path(
        "outputs/x3d_s_ir_context_oof/"
        "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715/"
        "fold_0/formal_outer_refit.pt"
    ),
    Path(
        "outputs/x3d_s_ir_context_oof/"
        "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715/"
        "fold_0/formal_outer_predictions.npz"
    ),
    Path("outputs/x3d_s_ir_evidence/ir_x3d_s_k400_pure/oof_evidence.npz"),
    Path("reports/x3d_s_phase5_evidence_summary.json"),
)


def collect_canonical_artifact_hashes(paths: Sequence[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def select_fold0(
    assignment: Mapping[str, Any], *, allowed_users: set[str]
) -> trainer.UserFold:
    folds = trainer.validate_oof_assignment(assignment, allowed_users=allowed_users)
    selected = [fold for fold in folds if fold.fold == FOLD_INDEX]
    if len(selected) != 1:
        raise ValueError("Frozen OOF assignment must contain exactly one fold 0")
    return selected[0]


def validate_dev_contract(*, output_root: Path, run_id: str, seed: int) -> None:
    if Path(output_root) != DEV_OUTPUT_ROOT:
        raise ValueError(f"Fold0 tuning must use development output root {DEV_OUTPUT_ROOT}")
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id must be a non-empty path-safe name")
    if any(part in run_id.lower() for part in FORBIDDEN_RUN_ID_PARTS):
        raise ValueError("Development run-id must not resemble canonical Phase 4/5 artifacts")
    if int(seed) != CANONICAL_SEED:
        raise ValueError(f"Fold0 development is frozen to seed {CANONICAL_SEED}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run protected development-only X3D-S tuning on frozen fold 0"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oof-fold-assignment", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping):
        raise ValueError("Config root must be a mapping")
    config = dict(raw_config)
    config["seed"] = int(args.seed)
    output_root = Path(str(config["output_root"]))
    validate_dev_contract(output_root=output_root, run_id=str(args.run_id), seed=args.seed)

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
    split = json.loads(Path(str(config["split_path"])).read_text(encoding="utf-8"))
    allowed_users = set(str(user) for user in split["train_users"])
    assignment = json.loads(args.oof_fold_assignment.read_text(encoding="utf-8"))
    fold = select_fold0(assignment, allowed_users=allowed_users)

    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    partition_manifest = trainer.prepare_partition_manifest(
        manifest,
        train_user_ids=fold.train_user_ids,
        validation_user_ids=fold.validation_user_ids,
    )
    augmentation_config = config.get("augmentation")
    train_dataset = X3DClipDataset(
        partition_manifest,
        split="train",
        training=True,
        augmentation_config=augmentation_config,
        seed=CANONICAL_SEED,
    )
    validation_dataset = X3DClipDataset(
        partition_manifest,
        split="val",
        training=False,
        augmentation_config=augmentation_config,
        seed=CANONICAL_SEED,
    )

    run_directory = trainer.prepare_run_directory(output_root, str(args.run_id))
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "role": "fold0_development_tuning",
        "unbiased_oof": False,
        "canonical_evidence_mutation_permitted": False,
        "fold": FOLD_INDEX,
        "seed": CANONICAL_SEED,
        "train_user_ids": list(fold.train_user_ids),
        "validation_user_ids": list(fold.validation_user_ids),
        "assignment_sha256": trainer._sha256_file(args.oof_fold_assignment),
        "canonical_artifact_sha256_before": canonical_before,
        "target": {
            "accuracy_minimum": 0.63,
            "macro_f1_minimum": 0.52,
            "worst_user_accuracy_minimum": 0.5338,
        },
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
        raise RuntimeError("Canonical Phase 4/5 artifact hash changed during fold0 development")
    provenance["canonical_artifact_sha256_after"] = canonical_after
    provenance["canonical_artifacts_unchanged"] = True
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
