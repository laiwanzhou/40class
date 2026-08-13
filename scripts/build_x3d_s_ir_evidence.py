from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.x3d_clip_dataset import X3DClipDataset, collate_x3d_clips
from src.fusion.expert_evidence import ExpertEvidence
import src.train_x3d_s_visual_expert as trainer


ROOT = Path(__file__).resolve().parents[1]
EXPERT_ID = "ir_x3d_s_k400_pure"
CANONICAL_SEED = 20260715
ASSIGNMENT_SHA256 = "2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76"
QUALITY_MAPPING = "constant_1_for_first_generation_fusion"
QUALITY_MAPPING_SHA256 = hashlib.sha256(QUALITY_MAPPING.encode("utf-8")).hexdigest()
FINALIZATION_REPORT = ROOT / "reports/x3d_s_phase5_finalization_policy.json"
CANONICAL_FOLDS = (
    ROOT / "outputs/x3d_s_ir_context_oof/x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715/fold_0",
    ROOT / "outputs/x3d_s_ir_context_oof/x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715/fold_1",
    ROOT / "outputs/x3d_s_ir_context_oof/x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715_fold2_only/fold_2",
)
SOURCE_EPOCHS = (10, 29, 11, 16, 20, 12, 8, 10, 7)
FINALIZE_EPOCHS = 11


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or audit sparse X3D-S IR ExpertEvidence")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/x3d_s_ir_evidence/ir_x3d_s_k400_pure"))
    parser.add_argument("--assignment", type=Path, default=Path("metadata/splits/train14_oof_3fold.json"))
    parser.add_argument("--build-oof", action="store_true")
    parser.add_argument("--prepare-finalization", action="store_true")
    parser.add_argument("--build-heldout", action="store_true")
    parser.add_argument("--final-run-directory", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_sha256(path: Path) -> str:
    return trainer._sha256_file(path.resolve())


def validate_assignment(path: Path, config: Mapping[str, Any]) -> tuple[trainer.UserFold, ...]:
    if trainer._sha256_file(path) != ASSIGNMENT_SHA256:
        raise ValueError("Phase 5 assignment SHA differs from frozen Phase 4 assignment")
    assignment = json.loads(path.read_text(encoding="utf-8"))
    split = json.loads(Path(str(config["split_path"])).read_text(encoding="utf-8"))
    folds = trainer.validate_oof_assignment(
        assignment, allowed_users=set(str(user) for user in split["train_users"])
    )
    for index, fold in enumerate(folds):
        epoch_selection = assignment["folds"][index].get("epoch_selection")
        if not isinstance(epoch_selection, Mapping):
            raise ValueError(f"Fold {index} lacks frozen epoch_selection")
        fit = set(str(user) for user in epoch_selection["fit_user_ids"])
        validation = set(str(user) for user in epoch_selection["validation_user_ids"])
        if fit & validation or fit | validation != set(fold.train_user_ids):
            raise ValueError(f"Fold {index} epoch-selection users do not partition outer-train")
    return folds


def load_canonical_archives(
    *, config: Mapping[str, Any], assignment: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    folds = validate_assignment(assignment, config)
    archives: list[dict[str, np.ndarray]] = []
    provenance: list[dict[str, object]] = []
    for fold_index, (fold, directory) in enumerate(zip(folds, CANONICAL_FOLDS, strict=True)):
        checkpoint_path = directory / "formal_outer_refit.pt"
        archive_path = directory / "formal_outer_predictions.npz"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        strict = checkpoint.get("strict_oof_provenance")
        if not isinstance(strict, Mapping):
            raise ValueError(f"Fold {fold_index} lacks strict OOF provenance")
        expected = {
            "actual_seed": CANONICAL_SEED,
            "outer_fold": fold_index,
            "outer_train_user_ids": list(fold.train_user_ids),
            "outer_validation_user_ids": list(fold.validation_user_ids),
            "assignment_sha256": ASSIGNMENT_SHA256,
        }
        for key, value in expected.items():
            if strict.get(key) != value:
                raise ValueError(f"Fold {fold_index} provenance mismatch for {key}")
        if strict.get("outer_validation_labels_used_for_selection") is not False:
            raise ValueError(f"Fold {fold_index} does not prove strict checkpoint selection")
        if int(strict.get("scheduler_horizon_epochs", -1)) != 30:
            raise ValueError(f"Fold {fold_index} scheduler horizon differs from Phase 4")
        run_summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
        if run_summary.get("resolved_config_sha256") != strict.get("resolved_config_sha256"):
            raise ValueError(f"Fold {fold_index} resolved config provenance mismatch")
        with np.load(archive_path, allow_pickle=False) as source:
            archive = {key: source[key] for key in source.files}
        if set(archive["user_ids"].astype(str)) != set(fold.validation_user_ids):
            raise ValueError(f"Fold {fold_index} archive users differ from assignment")
        archives.append(archive)
        provenance.append(
            {
                "fold": fold_index,
                "selected_epoch": int(strict["selected_epoch"]),
                "checkpoint_sha256": trainer._sha256_file(checkpoint_path),
                "prediction_sha256": trainer._sha256_file(archive_path),
                "sample_count": len(archive["sample_ids"]),
                "outer_train_user_ids": list(fold.train_user_ids),
                "outer_validation_user_ids": list(fold.validation_user_ids),
            }
        )
    row_keys = {
        key
        for key, value in archives[0].items()
        if value.ndim > 0 and value.shape[0] == len(archives[0]["sample_ids"])
    }
    combined = {
        key: np.concatenate([archive[key] for archive in archives]) for key in row_keys
    }
    hashes = {str(archive["class_map_hash"].item()) for archive in archives}
    if len(hashes) != 1:
        raise ValueError("Canonical fold class-map hashes differ")
    combined["class_map_hash"] = np.asarray(next(iter(hashes)))
    sample_ids = combined["sample_ids"].astype(str)
    if len(sample_ids) != 2320 or len(np.unique(sample_ids)) != 2320:
        raise ValueError("Canonical OOF must contain exactly 2,320 unique samples")
    if sorted(np.unique(combined["labels"]).astype(int).tolist()) != list(range(40)):
        raise ValueError("Canonical OOF does not cover all 40 classes")
    return combined, provenance


def build_oof(config_path: Path, assignment: Path, output_root: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    combined, folds = load_canonical_archives(config=config, assignment=assignment)
    model_lineage_sha = sha256_json([fold["checkpoint_sha256"] for fold in folds])
    evidence = ExpertEvidence(
        role="oof_train14",
        expert_id=EXPERT_ID,
        sample_ids=combined["sample_ids"],
        user_ids=combined["user_ids"],
        labels=combined["labels"],
        logits=combined["logits"],
        embeddings=combined["embeddings"],
        availability=combined["availability"],
        quality=combined["quality"],
        quality_mask=combined["quality_mask"],
        fusion_quality_score=np.ones((len(combined["sample_ids"]), 1), dtype=np.float32),
        class_map_hash=str(combined["class_map_hash"].item()),
        model_sha256=model_lineage_sha,
        config_sha256=config_sha256(config_path),
        deployed_weight_bytes=20_644_200,
        preprocessing_dependencies=(
            "yolo11n-pose:869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
            "pose-guided-ir-context",
            "adaptive-multiclip-13x182",
        ),
        quality_mapping=QUALITY_MAPPING,
        quality_mapping_sha256=QUALITY_MAPPING_SHA256,
        diagnostics={
            "num_frames": combined["num_frames"],
            "num_clips": combined["num_clips"],
        },
    )
    output_root.mkdir(parents=True, exist_ok=False)
    evidence_path = output_root / "oof_evidence.npz"
    evidence.save(evidence_path)
    provenance = {
        "schema_version": 1,
        "role": "oof_train14",
        "expert_id": EXPERT_ID,
        "canonical_seed": CANONICAL_SEED,
        "assignment_sha256": ASSIGNMENT_SHA256,
        "config_sha256": config_sha256(config_path),
        "model_lineage_sha256": model_lineage_sha,
        "quality_mapping": QUALITY_MAPPING,
        "quality_mapping_sha256": QUALITY_MAPPING_SHA256,
        "sample_count": 2320,
        "class_count": 40,
        "folds": folds,
        "evidence_sha256": trainer._sha256_file(evidence_path),
        "deployed_weight_bytes": 20_644_200,
        "heldout_access": False,
    }
    (output_root / "oof_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def finalization_policy() -> dict[str, object]:
    if tuple(sorted(SOURCE_EPOCHS))[4] != FINALIZE_EPOCHS:
        raise AssertionError("Frozen finalization median is inconsistent")
    return {
        "schema_version": 1,
        "status": "frozen_before_train14_finalization",
        "expert_id": EXPERT_ID,
        "source_epochs": list(SOURCE_EPOCHS),
        "median_finalize_epochs": FINALIZE_EPOCHS,
        "final_seed": CANONICAL_SEED,
        "scheduler_horizon_epochs": 30,
        "selection_uses_heldout": False,
        "heldout_role": "evaluation_only",
        "heldout_labels_serialized": False,
    }


def prepare_finalization(output_root: Path) -> None:
    policy = finalization_policy()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "finalization_policy.json"
    if path.exists():
        raise FileExistsError(path)
    serialized = json.dumps(policy, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")
    if FINALIZATION_REPORT.exists():
        if FINALIZATION_REPORT.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(FINALIZATION_REPORT)
    else:
        FINALIZATION_REPORT.write_text(serialized, encoding="utf-8")


@torch.no_grad()
def predict_label_free(
    *, model: torch.nn.Module, dataset: X3DClipDataset, config: Mapping[str, Any], device: torch.device
) -> dict[str, np.ndarray]:
    loader_config = config["loader"]
    sampler = trainer.ClipBudgetBatchSampler(
        dataset.num_clips,
        max_trials_per_batch=int(loader_config["max_trials_per_batch"]),
        max_valid_clips_per_batch=int(loader_config["max_valid_clips_per_batch"]),
        shuffle=False,
        seed=CANONICAL_SEED,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_x3d_clips,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model.eval().to(device)
    rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("logits", "embeddings", "quality", "quality_mask", "availability", "num_frames", "num_clips")
    }
    sample_ids: list[str] = []
    user_ids: list[str] = []
    for batch in loader:
        output = model(
            batch["clips"].to(device, non_blocking=True),
            clip_mask=batch["clip_mask"].to(device, non_blocking=True),
            quality=batch["quality"].to(device, non_blocking=True),
            quality_mask=batch["quality_mask"].to(device, non_blocking=True),
            availability=batch["availability"].to(device, non_blocking=True),
        )
        sample_ids.extend(str(value) for value in batch["sample_ids"])
        user_ids.extend(str(value) for value in batch["user_ids"])
        for key, value in (
            ("logits", output.main_logits),
            ("embeddings", output.embedding),
            ("quality", output.quality),
            ("quality_mask", output.quality_mask),
            ("availability", output.availability),
            ("num_frames", batch["num_frames"]),
            ("num_clips", batch["num_clips"]),
        ):
            rows[key].append(value.detach().cpu().numpy())
    return {
        "sample_ids": np.asarray(sample_ids),
        "user_ids": np.asarray(user_ids),
        **{key: np.concatenate(value) for key, value in rows.items()},
    }


def build_heldout(
    config_path: Path, assignment: Path, output_root: Path, final_run_directory: Path
) -> None:
    policy_path = output_root / "finalization_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("median_finalize_epochs") != FINALIZE_EPOCHS:
        raise ValueError("Finalization policy epoch mismatch")
    config = yaml.safe_load((final_run_directory / "resolved_config.yaml").read_text(encoding="utf-8"))
    if int(config["seed"]) != CANONICAL_SEED or int(config["training"]["epochs"]) != FINALIZE_EPOCHS:
        raise ValueError("Final run does not match frozen seed/epoch policy")
    validate_assignment(assignment, config)
    checkpoint_path = final_run_directory / "final_train14.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != FINALIZE_EPOCHS:
        raise ValueError("Final checkpoint epoch mismatch")
    model = trainer._build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    heldout_manifest = manifest.loc[manifest["split"].astype(str) == "val"].copy()
    if set(heldout_manifest["user_id"].astype(str)) != {"user4", "user17", "user23", "user24"}:
        raise ValueError("Held-out manifest users differ from frozen outer split")
    dataset = X3DClipDataset(
        heldout_manifest, split="val", training=False, seed=CANONICAL_SEED
    )
    predictions = predict_label_free(
        model=model,
        dataset=dataset,
        config=config,
        device=torch.device(str(config["device"])),
    )
    if len(predictions["sample_ids"]) != 590 or len(np.unique(predictions["sample_ids"])) != 590:
        raise ValueError("Held-out IR evidence must contain 590 unique samples")
    class_hash = dataset.class_map_hash
    checkpoint_sha = trainer._sha256_file(checkpoint_path)
    evidence = ExpertEvidence(
        role="heldout",
        expert_id=EXPERT_ID,
        sample_ids=predictions["sample_ids"],
        user_ids=predictions["user_ids"],
        labels=None,
        logits=predictions["logits"],
        embeddings=predictions["embeddings"],
        availability=predictions["availability"],
        quality=predictions["quality"],
        quality_mask=predictions["quality_mask"],
        fusion_quality_score=np.ones((590, 1), dtype=np.float32),
        class_map_hash=class_hash,
        model_sha256=checkpoint_sha,
        config_sha256=config_sha256(config_path),
        deployed_weight_bytes=int(checkpoint_path.stat().st_size + 6_255_593),
        preprocessing_dependencies=(
            "yolo11n-pose:869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
            "pose-guided-ir-context",
            "adaptive-multiclip-13x182",
        ),
        quality_mapping=QUALITY_MAPPING,
        quality_mapping_sha256=QUALITY_MAPPING_SHA256,
        diagnostics={
            "num_frames": predictions["num_frames"],
            "num_clips": predictions["num_clips"],
        },
    )
    evidence_path = output_root / "heldout_evidence.npz"
    if evidence_path.exists():
        raise FileExistsError(evidence_path)
    evidence.save(evidence_path)
    provenance = {
        "schema_version": 1,
        "role": "heldout",
        "expert_id": EXPERT_ID,
        "evaluation_only": True,
        "labels_present": False,
        "labels_inspected": False,
        "sample_count": 590,
        "user_ids": sorted(set(predictions["user_ids"].astype(str))),
        "final_seed": CANONICAL_SEED,
        "finalize_epochs": FINALIZE_EPOCHS,
        "checkpoint_sha256": checkpoint_sha,
        "evidence_sha256": trainer._sha256_file(evidence_path),
        "config_sha256": config_sha256(config_path),
        "assignment_sha256": ASSIGNMENT_SHA256,
        "deployed_weight_bytes": evidence.deployed_weight_bytes,
        "quarantine_until_phase": 10,
    }
    (output_root / "heldout_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def audit(config_path: Path, assignment: Path, output_root: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_assignment(assignment, config)
    report: dict[str, object] = {
        "assignment_sha256": trainer._sha256_file(assignment),
        "assignment_valid": True,
        "expert_id": EXPERT_ID,
    }
    for role, filename in (("oof_train14", "oof_evidence.npz"), ("heldout", "heldout_evidence.npz")):
        path = output_root / filename
        if path.exists():
            evidence = ExpertEvidence.load(path)
            if evidence.role != role:
                raise ValueError(f"Evidence role mismatch for {filename}")
            report[role] = {
                "exists": True,
                "sample_count": len(evidence.sample_ids),
                "sha256": trainer._sha256_file(path),
                "labels_present": evidence.labels is not None,
            }
        else:
            report[role] = {"exists": False}
    print(json.dumps(report, indent=2))


def main() -> None:
    args = build_arg_parser().parse_args()
    actions = sum(
        bool(value)
        for value in (args.build_oof, args.prepare_finalization, args.build_heldout, args.audit_only)
    )
    if actions != 1:
        raise ValueError("Select exactly one evidence action")
    if args.build_oof:
        build_oof(args.config, args.assignment, args.output_root)
    elif args.prepare_finalization:
        prepare_finalization(args.output_root)
    elif args.build_heldout:
        if args.final_run_directory is None:
            raise ValueError("--build-heldout requires --final-run-directory")
        build_heldout(
            args.config, args.assignment, args.output_root, args.final_run_directory.resolve()
        )
    else:
        audit(args.config, args.assignment, args.output_root)


if __name__ == "__main__":
    main()
