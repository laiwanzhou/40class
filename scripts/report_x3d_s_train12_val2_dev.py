from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.report_x3d_s_fold0_dev import recompute_metrics
from scripts.run_x3d_s_fold0_dev import (
    CANONICAL_ARTIFACTS,
    collect_canonical_artifact_hashes,
)
import src.train_x3d_s_visual_expert as trainer


EXPECTED_VALIDATION_USERS = ("user21", "user22")
EXPECTED_VALIDATION_TRIALS = 324
EXPECTED_MISSING_CLASS_IDS = (25, 26, 33, 35)


def validate_prediction_population(
    user_ids: np.ndarray, sample_ids: np.ndarray
) -> None:
    if len(sample_ids) != EXPECTED_VALIDATION_TRIALS:
        raise ValueError(
            f"Prediction archive must contain exactly {EXPECTED_VALIDATION_TRIALS} trials"
        )
    if len(set(sample_ids.astype(str).tolist())) != len(sample_ids):
        raise ValueError("Prediction archive contains duplicate sample IDs")
    actual_users = tuple(sorted(set(user_ids.astype(str).tolist())))
    if actual_users != EXPECTED_VALIDATION_USERS:
        raise ValueError("Prediction archive must contain the exact frozen validation users")


def _load_prediction_contract(archive_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"labels", "user_ids", "sample_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Prediction archive is missing fields: {sorted(missing)}")
        labels = archive["labels"].astype(np.int64)
        user_ids = archive["user_ids"].astype(str)
        sample_ids = archive["sample_ids"].astype(str)
    validate_prediction_population(user_ids, sample_ids)
    observed = set(labels.tolist())
    missing_classes = tuple(class_id for class_id in range(40) if class_id not in observed)
    if missing_classes != EXPECTED_MISSING_CLASS_IDS:
        raise ValueError(
            f"Validation missing classes changed: expected {EXPECTED_MISSING_CLASS_IDS}, "
            f"got {missing_classes}"
        )
    return user_ids, sample_ids


def build_report(run_directory: Path) -> dict[str, Any]:
    provenance_path = run_directory / "development_provenance.json"
    summary_path = run_directory / "run_summary.json"
    history_path = run_directory / "history.csv"
    archive_path = run_directory / "val_predictions_best_accuracy.npz"
    checkpoint_path = run_directory / "best_accuracy.pt"
    for path in (
        provenance_path,
        summary_path,
        history_path,
        archive_path,
        checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if provenance.get("role") != "train12_val2_development_tuning":
        raise ValueError("Run is not registered as train12/val2 development tuning")
    if provenance.get("unbiased_oof") is not False:
        raise ValueError("Development evidence must never claim unbiased OOF status")
    if tuple(provenance.get("validation_user_ids", ())) != EXPECTED_VALIDATION_USERS:
        raise ValueError("Provenance validation users do not match the frozen split")

    _load_prediction_contract(archive_path)
    metrics = recompute_metrics(archive_path)
    history = pd.read_csv(history_path)
    selected_epoch = int(summary["best_accuracy"]["epoch"])
    selected_rows = history.loc[history["epoch"].astype(int) == selected_epoch]
    if len(selected_rows) != 1:
        raise ValueError("Selected checkpoint epoch is absent or duplicated in history")
    selected = selected_rows.iloc[0]

    canonical_after = collect_canonical_artifact_hashes(CANONICAL_ARTIFACTS)
    canonical_unchanged = (
        canonical_after == provenance.get("canonical_artifact_sha256_before")
        and provenance.get("canonical_artifacts_unchanged") is True
    )
    if not canonical_unchanged:
        raise RuntimeError("Canonical Phase 4/5 artifacts changed during development")

    train_accuracy = float(selected["train_accuracy"])
    return {
        "schema_version": 1,
        "role": "train12_val2_development_tuning_report",
        "unbiased_oof": False,
        "run_id": run_directory.name,
        "decision": "freeze_result_no_matched_baseline_claim",
        "interpretation": (
            "Standalone partial-backbone result on the frozen shared development split. "
            "No matched full-backbone run exists on this split, so the effect of partial "
            "unfreezing is not causally identified."
        ),
        "split": {
            "train_user_ids": provenance["train_user_ids"],
            "validation_user_ids": provenance["validation_user_ids"],
            "train_trial_count": int(provenance["train_trial_count"]),
            "validation_trial_count": int(provenance["validation_trial_count"]),
            "validation_observed_class_count": int(provenance["validation_class_count"]),
            "validation_missing_class_ids": list(EXPECTED_MISSING_CLASS_IDS),
            "macro_f1_class_labels": list(range(40)),
            "development_split_sha256": provenance["development_split_sha256"],
        },
        "metrics": metrics,
        "selected_epoch": selected_epoch,
        "epochs_completed": int(summary["epochs_completed"]),
        "training_log_at_selected_epoch": {
            "train_accuracy": train_accuracy,
            "train_macro_f1": float(selected["train_macro_f1"]),
            "val_accuracy": float(selected["val_accuracy"]),
            "val_macro_f1": float(selected["val_macro_f1"]),
            "train_minus_val_accuracy": train_accuracy - float(metrics["accuracy"]),
        },
        "best_macro_f1_checkpoint": summary["best_macro_f1"],
        "unfrozen_backbone_blocks": int(summary["unfrozen_backbone_blocks"]),
        "trainable_backbone_parameters": int(
            summary["trainable_backbone_parameters_last_epoch"]
        ),
        "full_temporal_coverage": provenance["temporal_training_policy"],
        "artifacts": {
            "prediction_archive": str(archive_path.resolve()),
            "prediction_archive_sha256": trainer._sha256_file(archive_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": trainer._sha256_file(checkpoint_path),
            "resolved_config_sha256": summary["resolved_config_sha256"],
        },
        "canonical_artifact_sha256_after": canonical_after,
        "canonical_artifacts_unchanged": True,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    training = report["training_log_at_selected_epoch"]
    split = report["split"]
    lines = [
        "# X3D-S Train12/Val2 Partial-Backbone Result",
        "",
        "> Development-only evidence on user21/user22. This is not unbiased OOF, does not replace Phase 4/5 evidence, and is not matched to the former fold0 experiments.",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.6f} |",
        f"| Macro-F1 (fixed 40 classes) | {metrics['macro_f1']:.6f} |",
        f"| Worst-user Accuracy | {metrics['worst_user_accuracy']:.6f} |",
        f"| Selected epoch | {report['selected_epoch']} |",
        f"| Train Accuracy at selected epoch | {training['train_accuracy']:.6f} |",
        f"| Train minus validation Accuracy | {training['train_minus_val_accuracy']:.6f} |",
        "",
        "Validation contains 324 usable-IR trials across 36 observed classes. "
        f"Missing class IDs are `{split['validation_missing_class_ids']}` and contribute zero to fixed-40 Macro-F1.",
        "",
        "## Per User",
        "",
        "| User | Accuracy |",
        "|---|---:|",
    ]
    for user, accuracy in metrics["per_user_accuracy"].items():
        lines.append(f"| {user} | {accuracy:.6f} |")
    lines.extend(
        ["", "## Duration", "", "| Frames | N | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"]
    )
    for bucket, values in metrics["duration_buckets"].items():
        lines.append(
            f"| {bucket} | {values['sample_count']} | {values['accuracy']:.6f} | "
            f"{values['macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "The selected checkpoint still has a substantial train-to-validation gap, so this intervention does not by itself resolve cross-user overfitting.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report train12/val2 X3D evidence")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/x3d_s_train12_val2_partial2_report.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/x3d_s_train12_val2_partial2_report.md"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(args.run_directory)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
