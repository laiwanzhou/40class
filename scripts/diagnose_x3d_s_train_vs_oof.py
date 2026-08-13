from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.x3d_clip_dataset import X3DClipDataset, collate_x3d_clips
import src.train_x3d_s_visual_expert as trainer


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SEED = 20260715
ASSIGNMENT_SHA256 = "2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76"
CANONICAL_FOLDS = (
    ROOT / "outputs/x3d_s_ir_context_oof/x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715/fold_0",
    ROOT / "outputs/x3d_s_ir_context_oof/x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715/fold_1",
    ROOT / "outputs/x3d_s_ir_context_oof/x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715_fold2_only/fold_2",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose canonical X3D train-to-OOF gap")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def deterministic_eval_manifest(
    frame: pd.DataFrame, user_ids: Sequence[str]
) -> pd.DataFrame:
    requested = set(str(user) for user in user_ids)
    selected = frame.loc[frame["user_id"].astype(str).isin(requested)].copy()
    observed = set(selected["user_id"].astype(str).unique())
    if observed != requested:
        raise ValueError(f"Manifest is missing requested users: {sorted(requested - observed)}")
    selected["split"] = "val"
    return selected


def summarize_fold_metrics(
    *,
    fold: int,
    training_log: Mapping[str, float],
    train_eval: Mapping[str, float],
    outer_val: Mapping[str, float],
    train_count: int,
    val_count: int,
) -> dict[str, float | int]:
    return {
        "fold": fold,
        "outer_train_sample_count": train_count,
        "outer_val_sample_count": val_count,
        "training_log_train_accuracy": float(training_log["train_accuracy"]),
        "training_log_train_macro_f1": float(training_log["train_macro_f1"]),
        "deterministic_train_eval_accuracy": float(train_eval["accuracy"]),
        "deterministic_train_eval_macro_f1": float(train_eval["macro_f1"]),
        "deterministic_train_eval_worst_user_accuracy": float(
            train_eval["worst_user_accuracy"]
        ),
        "formal_outer_val_accuracy": float(outer_val["accuracy"]),
        "formal_outer_val_macro_f1": float(outer_val["macro_f1"]),
        "formal_outer_val_worst_user_accuracy": float(outer_val["worst_user_accuracy"]),
        "train_to_val_accuracy_gap": float(train_eval["accuracy"] - outer_val["accuracy"]),
        "train_to_val_macro_f1_gap": float(
            train_eval["macro_f1"] - outer_val["macro_f1"]
        ),
        "train_to_val_worst_user_accuracy_gap": float(
            train_eval["worst_user_accuracy"] - outer_val["worst_user_accuracy"]
        ),
    }


def make_loader(
    dataset: X3DClipDataset, config: Mapping[str, Any]
) -> DataLoader[Mapping[str, object]]:
    loader = config["loader"]
    sampler = trainer.ClipBudgetBatchSampler(
        dataset.num_clips,
        max_trials_per_batch=int(loader["max_trials_per_batch"]),
        max_valid_clips_per_batch=int(loader["max_valid_clips_per_batch"]),
        shuffle=False,
        seed=CANONICAL_SEED,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_x3d_clips,
        num_workers=int(loader["num_workers"]),
        pin_memory=True,
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def run_fold(
    *,
    fold_index: int,
    fold: trainer.UserFold,
    fold_directory: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    assignment_sha: str,
    output_directory: Path,
) -> tuple[dict[str, float | int | str | list[str]], dict[str, np.ndarray]]:
    checkpoint_path = fold_directory / "formal_outer_refit.pt"
    outer_archive_path = fold_directory / "formal_outer_predictions.npz"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("strict_oof_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Fold {fold_index} lacks strict OOF provenance")
    expected = {
        "actual_seed": CANONICAL_SEED,
        "outer_fold": fold_index,
        "outer_train_user_ids": list(fold.train_user_ids),
        "outer_validation_user_ids": list(fold.validation_user_ids),
        "assignment_sha256": assignment_sha,
        "outer_validation_labels_used_for_selection": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Fold {fold_index} checkpoint provenance mismatch for {key}")

    train_frame = deterministic_eval_manifest(manifest, fold.train_user_ids)
    if set(train_frame["user_id"].astype(str)) & set(fold.validation_user_ids):
        raise ValueError("Outer-train diagnostic contains outer-validation users")
    train_dataset = X3DClipDataset(
        train_frame, split="val", training=False, seed=CANONICAL_SEED
    )
    model = trainer._build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(str(config["device"]))
    model.to(device)
    outcome = trainer.run_model_epoch(
        model,
        make_loader(train_dataset, config),
        device=device,
        optimizer=None,
        gradient_accumulation=1,
        gradient_clip=float(config["optimizer"]["gradient_clip"]),
        amp_enabled=bool(config["amp"]["enabled"]),
        max_batches=None,
    )
    predictions = outcome.predictions
    train_arrays = {
        "sample_ids": np.asarray(predictions.sample_ids),
        "user_ids": np.asarray(predictions.user_ids),
        "labels": predictions.labels.numpy(),
        "logits": predictions.output.main_logits.numpy(),
        "num_frames": predictions.num_frames.numpy(),
        "num_clips": predictions.num_clips.numpy(),
    }
    train_archive_path = output_directory / f"fold_{fold_index}_outer_train_predictions.npz"
    np.savez_compressed(train_archive_path, **train_arrays)

    outer_archive = load_npz(outer_archive_path)
    outer_labels = outer_archive["labels"].astype(np.int64)
    outer_predictions = outer_archive["logits"].argmax(axis=1)
    outer_result = trainer.classification_metrics(
        outer_labels, outer_predictions, num_classes=40
    )
    outer_users = outer_archive["user_ids"].astype(str)
    outer_result["worst_user_accuracy"] = min(
        float((outer_predictions[outer_users == user] == outer_labels[outer_users == user]).mean())
        for user in np.unique(outer_users)
    )
    history = pd.read_csv(fold_directory / "finalize_history.csv")
    training_log = history.loc[history["epoch"] == int(provenance["selected_epoch"])].iloc[-1]
    summary = summarize_fold_metrics(
        fold=fold_index,
        training_log=training_log,
        train_eval=outcome.metrics,
        outer_val=outer_result,
        train_count=len(train_arrays["sample_ids"]),
        val_count=len(outer_labels),
    )
    summary.update(
        {
            "selected_epoch": int(provenance["selected_epoch"]),
            "checkpoint_sha256": trainer._sha256_file(checkpoint_path),
            "saved_outer_val_archive_sha256": trainer._sha256_file(outer_archive_path),
            "outer_train_prediction_sha256": trainer._sha256_file(train_archive_path),
            "outer_train_user_ids": list(fold.train_user_ids),
            "outer_val_user_ids": list(fold.validation_user_ids),
        }
    )
    return summary, train_arrays


def population_metrics(archives: Sequence[Mapping[str, np.ndarray]]) -> dict[str, float | int]:
    labels = np.concatenate([archive["labels"].astype(np.int64) for archive in archives])
    logits = np.concatenate([archive["logits"] for archive in archives])
    users = np.concatenate([archive["user_ids"].astype(str) for archive in archives])
    predictions = logits.argmax(axis=1)
    result = trainer.classification_metrics(labels, predictions, num_classes=40)
    return {
        "sample_count": len(labels),
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "worst_user_accuracy": min(
            float((predictions[users == user] == labels[users == user]).mean())
            for user in np.unique(users)
        ),
    }


def aggregate(
    folds: Sequence[Mapping[str, Any]],
    train_archives: Sequence[Mapping[str, np.ndarray]],
    outer_val_archives: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    metric_names = (
        "training_log_train_accuracy",
        "training_log_train_macro_f1",
        "deterministic_train_eval_accuracy",
        "deterministic_train_eval_macro_f1",
        "deterministic_train_eval_worst_user_accuracy",
        "formal_outer_val_accuracy",
        "formal_outer_val_macro_f1",
        "formal_outer_val_worst_user_accuracy",
        "train_to_val_accuracy_gap",
        "train_to_val_macro_f1_gap",
        "train_to_val_worst_user_accuracy_gap",
    )
    fold_mean = {
        name: float(np.mean([float(fold[name]) for fold in folds])) for name in metric_names
    }
    train_total = sum(int(fold["outer_train_sample_count"]) for fold in folds)
    val_total = sum(int(fold["outer_val_sample_count"]) for fold in folds)
    weighted: dict[str, float | int | str] = {
        "outer_train_evaluation_rows": train_total,
        "outer_val_unique_rows": val_total,
        "outer_train_population_note": (
            "Each canonical train-14 trial is evaluated by the two fold models that trained on it."
        ),
    }
    for prefix, denominator, weight_key in (("training_log_train", train_total, "outer_train_sample_count"),):
        for metric in ("accuracy", "macro_f1"):
            name = f"{prefix}_{metric}"
            weighted[name] = sum(
                float(fold[name]) * int(fold[weight_key]) for fold in folds
            ) / denominator
    train_population = population_metrics(train_archives)
    val_population = population_metrics(outer_val_archives)
    combined = {
        "outer_train_evaluation": train_population,
        "formal_outer_val_oof": val_population,
        "train_to_val_accuracy_gap": float(
            train_population["accuracy"] - val_population["accuracy"]
        ),
        "train_to_val_macro_f1_gap": float(
            train_population["macro_f1"] - val_population["macro_f1"]
        ),
        "train_to_val_worst_user_accuracy_gap": float(
            train_population["worst_user_accuracy"] - val_population["worst_user_accuracy"]
        ),
    }
    return {
        "fold_mean": fold_mean,
        "training_log_trial_weighted": weighted,
        "concatenated_population": combined,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# X3D-S IR Train-vs-OOF Generalization Diagnostic",
        "",
        "This is a pure-inference diagnostic using only canonical seed `20260715`. It does not change training, checkpoint selection, canonical evidence, retention, or finalization.",
        "",
        "| Fold | Epoch | training-log train Acc/F1 | deterministic train_eval Acc/F1 | formal OOF val Acc/F1 | Acc gap | F1 gap | train/val worst-user Acc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in report["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['selected_epoch']} | "
            f"{fold['training_log_train_accuracy']:.6f} / {fold['training_log_train_macro_f1']:.6f} | "
            f"{fold['deterministic_train_eval_accuracy']:.6f} / {fold['deterministic_train_eval_macro_f1']:.6f} | "
            f"{fold['formal_outer_val_accuracy']:.6f} / {fold['formal_outer_val_macro_f1']:.6f} | "
            f"{fold['train_to_val_accuracy_gap']:.6f} | {fold['train_to_val_macro_f1_gap']:.6f} | "
            f"{fold['deterministic_train_eval_worst_user_accuracy']:.6f} / {fold['formal_outer_val_worst_user_accuracy']:.6f} |"
        )
    combined = report["aggregate"]["concatenated_population"]
    train = combined["outer_train_evaluation"]
    val = combined["formal_outer_val_oof"]
    lines.extend(
        [
            "",
            "## Combined view",
            "",
            f"Across `{train['sample_count']}` outer-train evaluation rows and `{val['sample_count']}` unique formal OOF rows, concatenated deterministic train_eval Accuracy/Macro-F1 are `{train['accuracy']:.6f}` / `{train['macro_f1']:.6f}`; formal OOF Accuracy/Macro-F1 are `{val['accuracy']:.6f}` / `{val['macro_f1']:.6f}`. The concatenated-population gaps are `{combined['train_to_val_accuracy_gap']:.6f}` Accuracy and `{combined['train_to_val_macro_f1_gap']:.6f}` Macro-F1.",
            "",
            "The 4,640 outer-train rows are not 4,640 unique trials: each of the 2,320 canonical train-14 trials is evaluated by the two fold models whose training population included that trial. Formal OOF remains the 2,320-row exactly-once unseen-user population.",
            "",
            "`training-log train_acc` was measured during the final training epoch with training-mode stochastic augmentation/dropout. `deterministic train_eval_acc` is a new eval-mode deterministic pass over outer-train. `formal OOF val_acc` is the saved untouched-user prediction. Only the latter two form the reported generalization gap.",
            "",
            "## Freeze decision",
            "",
            "Diagnostic only. The frozen combined frame manifest was opened as the path-index source, then immediately projected to the 14 train users. No heldout4 row entered a dataset, model forward, metric, or output; no heldout evidence/prediction or competition-test path was accessed. `ir_x3d_s_k400_pure`, the Phase 4 competition-retention decision, Phase 5 registration, and the final checkpoint remain unchanged. The IR/X3D route is frozen; no new IR single-modality training, tuning, matched baseline, or ablation may start without explicit approval.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = build_arg_parser().parse_args()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    if trainer._sha256_file(args.assignment) != ASSIGNMENT_SHA256:
        raise ValueError("Assignment differs from the frozen canonical assignment")
    assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = dict(config)
    config["seed"] = CANONICAL_SEED
    trainer.validate_config(config)
    split = json.loads(Path(str(config["split_path"])).read_text(encoding="utf-8"))
    folds = trainer.validate_oof_assignment(
        assignment, allowed_users=set(str(user) for user in split["train_users"])
    )
    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    heldout_users = set(str(user) for user in split["val_users"])
    if heldout_users & set(manifest.loc[manifest["split"] == "train", "user_id"].astype(str)):
        raise ValueError("Held-out users appear on the canonical train side")
    train14_manifest = manifest.loc[
        manifest["user_id"].astype(str).isin(set(str(user) for user in split["train_users"]))
    ].copy()
    if set(train14_manifest["user_id"].astype(str)) & heldout_users:
        raise ValueError("Train-14 diagnostic manifest contains held-out users")
    fold_reports = []
    train_archives = []
    outer_val_archives = []
    for fold_index, (fold, directory) in enumerate(zip(folds, CANONICAL_FOLDS, strict=True)):
        report, train_archive = run_fold(
            fold_index=fold_index,
            fold=fold,
            fold_directory=directory,
            config=config,
            manifest=train14_manifest,
            assignment_sha=ASSIGNMENT_SHA256,
            output_directory=output_directory,
        )
        fold_reports.append(report)
        train_archives.append(train_archive)
        outer_val_archives.append(load_npz(directory / "formal_outer_predictions.npz"))
    report = {
        "schema_version": 1,
        "status": "passed_diagnostic_only",
        "canonical_seed": CANONICAL_SEED,
        "assignment_sha256": ASSIGNMENT_SHA256,
        "combined_manifest_opened_as_path_index_source": True,
        "heldout4_rows_selected_for_dataset_or_metrics": False,
        "heldout4_evidence_or_predictions_accessed": False,
        "competition_test_accessed": False,
        "weights_updated": False,
        "checkpoint_or_epoch_reselected": False,
        "canonical_evidence_changed": False,
        "folds": fold_reports,
        "aggregate": aggregate(fold_reports, train_archives, outer_val_archives),
        "route_status_after_diagnostic": "frozen",
    }
    json_path = output_directory / "x3d_s_train_vs_oof_generalization.json"
    markdown_path = output_directory / "x3d_s_train_vs_oof_generalization.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
