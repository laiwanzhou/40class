from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import yaml

from src.data.ir_primary_full_sequence_dataset import class_map_hash
from src.train_x3d_s_visual_expert import resolved_config_sha256, validate_oof_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/x3d_s_ir_context_oof.yaml"
DEFAULT_ASSIGNMENT = PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json"
DEFAULT_EXPERIMENT_MANIFEST = (
    PROJECT_ROOT / "outputs/x3d_s_ir_context_oof/phase4_experiment_manifest.json"
)
PHASE4_SEEDS = (20260715, 20260716, 20260717)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trial_rows(frame: pd.DataFrame, *, allowed_users: set[str]) -> pd.DataFrame:
    required = {"sample_id", "user_id", "class_id", "action_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing Phase 4 columns: {sorted(missing)}")
    selected = frame[frame["user_id"].astype(str).isin(allowed_users)].copy()
    if set(selected["user_id"].astype(str).unique()) != allowed_users:
        raise ValueError("Manifest does not contain every official train-14 user")
    consistency = selected.groupby("sample_id", sort=False).agg(
        user_count=("user_id", "nunique"),
        class_count=("class_id", "nunique"),
        action_count=("action_name", "nunique"),
    )
    if (consistency != 1).any().any():
        raise ValueError("A Phase 4 sample has inconsistent user, class, or action metadata")
    return (
        selected[["sample_id", "user_id", "class_id", "action_name"]]
        .drop_duplicates("sample_id")
        .reset_index(drop=True)
    )


def _epoch_selection_split(
    trials: pd.DataFrame,
    *,
    outer_train_users: set[str],
    outer_fold: int,
) -> dict[str, Any]:
    outer_train = trials[trials["user_id"].astype(str).isin(outer_train_users)].reset_index(
        drop=True
    )
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=PHASE4_SEEDS[0] + outer_fold,
    )
    candidates = []
    for candidate_index, (fit_indices, validation_indices) in enumerate(
        splitter.split(
            outer_train,
            outer_train["class_id"],
            groups=outer_train["user_id"],
        )
    ):
        fit = outer_train.iloc[fit_indices]
        validation = outer_train.iloc[validation_indices]
        fit_class_count = int(fit["class_id"].nunique())
        if fit_class_count != 40:
            continue
        validation_classes = sorted(validation["class_id"].astype(int).unique().tolist())
        candidates.append(
            {
                "candidate_index": candidate_index,
                "fit_user_ids": sorted(fit["user_id"].astype(str).unique().tolist()),
                "validation_user_ids": sorted(
                    validation["user_id"].astype(str).unique().tolist()
                ),
                "fit_trial_count": int(len(fit)),
                "validation_trial_count": int(len(validation)),
                "fit_class_count": fit_class_count,
                "validation_class_count": len(validation_classes),
                "validation_missing_class_ids": sorted(set(range(40)) - set(validation_classes)),
            }
        )
    if not candidates:
        raise ValueError(f"Outer fold {outer_fold} has no inner split with 40-class fit coverage")
    selected = max(
        candidates,
        key=lambda candidate: (
            int(candidate["validation_class_count"]),
            -int(candidate["candidate_index"]),
        ),
    )
    return {
        "method": "StratifiedGroupKFold_candidate_selection",
        "n_splits": 3,
        "shuffle": True,
        "random_state": PHASE4_SEEDS[0] + outer_fold,
        "selection_rule": "fit_class_count_40_then_max_validation_coverage_then_lower_index",
        **selected,
    }


def generate_train14_oof_assignment(
    frame: pd.DataFrame,
    *,
    allowed_users: set[str],
    random_state: int,
) -> dict[str, Any]:
    trials = _trial_rows(frame, allowed_users=allowed_users)
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state,
    )
    folds = []
    validation_occurrences: list[str] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(trials, trials["class_id"], groups=trials["user_id"])
    ):
        train = trials.iloc[train_indices]
        validation = trials.iloc[validation_indices]
        train_users = sorted(train["user_id"].astype(str).unique().tolist())
        validation_users = sorted(validation["user_id"].astype(str).unique().tolist())
        train_class_count = int(train["class_id"].nunique())
        validation_class_count = int(validation["class_id"].nunique())
        if train_class_count != 40:
            raise ValueError(
                f"Fold {fold_index} outer-train lacks complete 40-class coverage"
            )
        validation_classes = set(validation["class_id"].astype(int).unique().tolist())
        validation_occurrences.extend(validation_users)
        folds.append(
            {
                "fold": fold_index,
                "train_user_ids": train_users,
                "validation_user_ids": validation_users,
                "train_trial_count": int(len(train)),
                "validation_trial_count": int(len(validation)),
                "train_class_count": train_class_count,
                "validation_class_count": validation_class_count,
                "validation_missing_class_ids": sorted(set(range(40)) - validation_classes),
                "epoch_selection": _epoch_selection_split(
                    trials,
                    outer_train_users=set(train_users),
                    outer_fold=fold_index,
                ),
            }
        )
    if sorted(validation_occurrences) != sorted(allowed_users):
        raise ValueError("Each train-14 user must be validation exactly once")
    combined_validation_class_count = int(trials["class_id"].nunique())
    if combined_validation_class_count != 40:
        raise ValueError("Combined outer-validation population lacks 40-class coverage")
    assignment = {
        "schema_version": 1,
        "method": "StratifiedGroupKFold",
        "n_splits": 3,
        "shuffle": True,
        "random_state": int(random_state),
        "group_field": "user_id",
        "label_field": "class_id",
        "population": "official_train14_canonical_union",
        "trial_count": int(len(trials)),
        "user_count": len(allowed_users),
        "class_count": int(trials["class_id"].nunique()),
        "combined_validation_class_count": combined_validation_class_count,
        "folds": folds,
        "immutable_policy": "write_once_and_verify_sha256",
    }
    validate_oof_assignment(assignment, allowed_users=allowed_users)
    return assignment


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"Frozen artifact already exists: {path}") from error
    return sha256_file(path)


def create_frozen_oof_assignment(path: Path, assignment: Mapping[str, Any]) -> str:
    return _write_json_once(path, assignment)


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    ).strip()


def _load_train14_manifest(path: Path, allowed_users: set[str]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=("sample_id", "user_id", "class_id", "action_name"),
        chunksize=100_000,
    ):
        selected = chunk[chunk["user_id"].astype(str).isin(allowed_users)]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError("No train-14 rows were found in the input manifest")
    return pd.concat(chunks, ignore_index=True)


def freeze_phase4_inputs(
    *,
    config_path: Path,
    assignment_path: Path,
    experiment_manifest_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Config root must be a mapping")
    split_path = (PROJECT_ROOT / Path(str(config["split_path"]))).resolve()
    outer_split = json.loads(split_path.read_text(encoding="utf-8"))
    train_users = set(str(user) for user in outer_split["train_users"])
    heldout_users = set(str(user) for user in outer_split["val_users"])
    if train_users & heldout_users or len(train_users) != 14 or len(heldout_users) != 4:
        raise ValueError("Outer split must contain disjoint 14/4 user sets")

    input_manifest_path = Path(str(config["input_manifest"])).resolve()
    train14_frame = _load_train14_manifest(input_manifest_path, train_users)
    ir_trials = _trial_rows(train14_frame, allowed_users=train_users)
    canonical_manifest_path = (PROJECT_ROOT / "metadata/manifest.csv").resolve()
    canonical_frame = pd.read_csv(canonical_manifest_path, encoding="utf-8-sig")
    canonical_trials = _trial_rows(canonical_frame, allowed_users=train_users)
    assignment = generate_train14_oof_assignment(
        canonical_trials,
        allowed_users=train_users,
        random_state=PHASE4_SEEDS[0],
    )
    for fold in assignment["folds"]:
        validation_users = set(fold["validation_user_ids"])
        train_fold_users = set(fold["train_user_ids"])
        ir_validation = ir_trials[ir_trials["user_id"].astype(str).isin(validation_users)]
        ir_train = ir_trials[ir_trials["user_id"].astype(str).isin(train_fold_users)]
        ir_validation_classes = set(ir_validation["class_id"].astype(int).unique().tolist())
        fold["ir_train_trial_count"] = int(len(ir_train))
        fold["ir_validation_trial_count"] = int(len(ir_validation))
        fold["ir_train_class_count"] = int(ir_train["class_id"].nunique())
        fold["ir_validation_class_count"] = len(ir_validation_classes)
        fold["ir_validation_missing_class_ids"] = sorted(
            set(range(40)) - ir_validation_classes
        )
    assignment_sha256 = create_frozen_oof_assignment(assignment_path, assignment)

    environment_probe_path = (
        PROJECT_ROOT / Path(str(config["deployment_artifacts"]["phase0_probe"]))
    ).resolve()
    environment_probe = json.loads(environment_probe_path.read_text(encoding="utf-8"))
    x3d_path = Path(str(environment_probe["components"]["x3d_s"]["path"])).resolve()
    yolo_path = Path(str(config["deployment_artifacts"]["yolo_checkpoint"])).resolve()
    smoke_checkpoint_path = (
        PROJECT_ROOT
        / "outputs/x3d_s_ir_context_fold0/x3d_s_ir_context_adaptive_smoke/best_accuracy.pt"
    ).resolve()
    for required_path in (x3d_path, yolo_path, smoke_checkpoint_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required frozen input is missing: {required_path}")

    compliance_path = (PROJECT_ROOT / "docs/x3d_s_rule_compliance.md").resolve()
    compliance_text = compliance_path.read_text(encoding="utf-8")
    access_match = re.search(r"Access date:\s*([^\n]+)", compliance_text)
    rule_urls = re.findall(r"https://[^\s)]+", compliance_text.split("## Working Interpretation")[0])
    route_bytes = smoke_checkpoint_path.stat().st_size + yolo_path.stat().st_size
    size_limit = int(config["size_gate"]["internal_limit_bytes"])
    manifest = {
        "schema_version": 1,
        "role": "phase4_train14_oof_preregistration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_inspected_before_freeze": False,
        "heldout_prediction_archive_loaded": False,
        "heldout_labels_used_for_fitting_or_selection": False,
        "git": {
            "sha": _git_output("rev-parse", "HEAD"),
            "tracked_worktree_clean": subprocess.call(
                ("git", "diff", "--quiet"), cwd=PROJECT_ROOT
            )
            == 0,
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": sha256_file(input_manifest_path),
            "train14_frame_rows": int(len(train14_frame)),
            "train14_trials": int(len(ir_trials)),
        },
        "canonical_union_manifest": {
            "path": str(canonical_manifest_path),
            "sha256": sha256_file(canonical_manifest_path),
            "train14_trials": int(len(canonical_trials)),
        },
        "class_map_hash": class_map_hash(
            canonical_trials[["class_id", "action_name"]].drop_duplicates()
        ),
        "outer_split": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "train_user_ids": sorted(train_users),
            "heldout_user_ids": sorted(heldout_users),
            "heldout_user_count": len(heldout_users),
        },
        "inner_oof_assignment": {
            "path": str(assignment_path.resolve()),
            "sha256": assignment_sha256,
            "folds": assignment["folds"],
        },
        "seeds": [
            {
                "seed": seed,
                "role": "canonical_phase5_evidence" if seed == PHASE4_SEEDS[0] else "stability_only",
                "resolved_config_sha256": resolved_config_sha256(
                    {**copy.deepcopy(dict(config)), "seed": seed}
                ),
            }
            for seed in PHASE4_SEEDS
        ],
        "canonical_oof_evidence_seed": PHASE4_SEEDS[0],
        "pretrained_weights": {
            "x3d_s": {"path": str(x3d_path), "sha256": sha256_file(x3d_path)},
            "yolo11n_pose": {"path": str(yolo_path), "sha256": sha256_file(yolo_path)},
        },
        "compliance": {
            "path": str(compliance_path),
            "sha256": sha256_file(compliance_path),
            "rule_source_urls": rule_urls,
            "rule_access_date": access_match.group(1).strip() if access_match else None,
        },
        "environment_probe": {
            "path": str(environment_probe_path),
            "sha256": sha256_file(environment_probe_path),
            "runtime": environment_probe["runtime"],
        },
        "provisional_ir_route": {
            "x3d_checkpoint_path": str(smoke_checkpoint_path),
            "x3d_checkpoint_sha256": sha256_file(smoke_checkpoint_path),
            "x3d_checkpoint_bytes": smoke_checkpoint_path.stat().st_size,
            "yolo_checkpoint_bytes": yolo_path.stat().st_size,
            "serialized_weight_subtotal": route_bytes,
            "internal_size_limit_bytes": size_limit,
            "ir_route_provisional_size_gate_passed": route_bytes < size_limit,
            "complete_submission_package_claimed": False,
        },
        "decision_contract": {
            "primary_checkpoint": "best_accuracy.pt",
            "formal_seeds": list(PHASE4_SEEDS),
            "complementary_worst_user_accuracy_delta_minimum": -0.02,
            "below_threshold_action": "interrupt_for_human_review_and_preserve_all_artifacts",
        },
    }
    if not manifest["provisional_ir_route"]["ir_route_provisional_size_gate_passed"]:
        raise RuntimeError("Provisional IR-route size gate failed")
    _write_json_once(experiment_manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze or summarize Phase 4 X3D-S evidence")
    parser.add_argument("--freeze-inputs", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assignment", type=Path, default=DEFAULT_ASSIGNMENT)
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        default=DEFAULT_EXPERIMENT_MANIFEST,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.freeze_inputs:
        raise ValueError("Select --freeze-inputs")
    manifest = freeze_phase4_inputs(
        config_path=args.config,
        assignment_path=args.assignment,
        experiment_manifest_path=args.experiment_manifest,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
