from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.engine.metrics import classification_metrics
import src.train_x3d_s_visual_expert as trainer
from scripts.run_x3d_s_fold0_dev import (
    CANONICAL_ARTIFACTS,
    collect_canonical_artifact_hashes,
)


TARGET = {
    "accuracy": 0.63,
    "macro_f1": 0.52,
    "worst_user_accuracy": 0.5338,
}
CANONICAL_BASELINE = {
    "accuracy": 0.571250,
    "macro_f1": 0.486918,
    "worst_user_accuracy": 0.533835,
}


def evaluate_candidate(
    metrics: Mapping[str, float], parent_metrics: Mapping[str, float]
) -> str:
    if float(metrics["accuracy"]) < float(parent_metrics["accuracy"]) - 0.02 - 1e-12:
        return "human_review_regression"
    if all(float(metrics[name]) >= threshold for name, threshold in TARGET.items()):
        return "target_met"
    return "continue_stage_a"


def _duration_bucket(num_frames: int) -> str:
    if num_frames <= 13:
        return "<=13"
    if num_frames <= 32:
        return "14-32"
    if num_frames <= 64:
        return "33-64"
    return ">64"


def recompute_metrics(archive_path: Path) -> dict[str, Any]:
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"labels", "logits", "user_ids", "num_frames", "sample_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Prediction archive is missing fields: {sorted(missing)}")
        labels = archive["labels"].astype(np.int64)
        logits = archive["logits"].astype(np.float64)
        users = archive["user_ids"].astype(str)
        num_frames = archive["num_frames"].astype(np.int64)
        sample_ids = archive["sample_ids"].astype(str)
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("Prediction archive contains duplicate sample IDs")
    predictions = logits.argmax(axis=1)
    overall = classification_metrics(labels, predictions, num_classes=40)
    per_user = {
        user: float((predictions[users == user] == labels[users == user]).mean())
        for user in sorted(np.unique(users).tolist())
    }
    duration: dict[str, dict[str, float | int]] = {}
    buckets = np.asarray([_duration_bucket(int(value)) for value in num_frames])
    for bucket in ("<=13", "14-32", "33-64", ">64"):
        mask = buckets == bucket
        bucket_metrics = classification_metrics(
            labels[mask], predictions[mask], num_classes=40
        )
        duration[bucket] = {
            "sample_count": int(mask.sum()),
            "accuracy": float(bucket_metrics["accuracy"]),
            "macro_f1": float(bucket_metrics["macro_f1"]),
        }
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(overall["accuracy"]),
        "macro_f1": float(overall["macro_f1"]),
        "worst_user_accuracy": min(per_user.values()),
        "worst_user_id": min(per_user, key=per_user.get),
        "per_user_accuracy": per_user,
        "duration_buckets": duration,
    }


def build_report(run_directory: Path) -> dict[str, Any]:
    provenance_path = run_directory / "development_provenance.json"
    summary_path = run_directory / "run_summary.json"
    archive_path = run_directory / "val_predictions_best_accuracy.npz"
    history_path = run_directory / "history.csv"
    for path in (provenance_path, archive_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("role") != "fold0_development_tuning":
        raise ValueError("Run is not registered as fold0 development tuning")
    if provenance.get("unbiased_oof") is not False:
        raise ValueError("Development report must never claim unbiased OOF status")
    metrics = recompute_metrics(archive_path)
    history = pd.read_csv(history_path)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected_epoch = int(summary["best_accuracy"]["epoch"])
        run_status = "completed"
    else:
        ranked = history.sort_values(
            ["val_accuracy", "val_macro_f1", "epoch"],
            ascending=[False, False, True],
        )
        selected_epoch = int(ranked.iloc[0]["epoch"])
        config_path = run_directory / "resolved_config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        summary = {
            "resolved_config_sha256": trainer.resolved_config_sha256(config),
            "unfrozen_backbone_blocks": config["training"].get(
                "unfrozen_backbone_blocks"
            ),
            "trainable_backbone_parameters_last_epoch": int(
                history.iloc[-1]["trainable_backbone_parameters"]
            ),
        }
        run_status = "interrupted_by_regression_guard"
    selected = history.loc[history["epoch"].astype(int) == selected_epoch]
    if len(selected) != 1:
        raise ValueError("Selected checkpoint epoch is absent or duplicated in history")
    selected_row = selected.iloc[0]
    decision = evaluate_candidate(metrics, CANONICAL_BASELINE)
    canonical_after = collect_canonical_artifact_hashes(CANONICAL_ARTIFACTS)
    canonical_unchanged = (
        canonical_after == provenance.get("canonical_artifact_sha256_before")
    )
    if not canonical_unchanged:
        raise RuntimeError("Canonical Phase 4/5 artifacts changed during development run")
    return {
        "schema_version": 1,
        "role": "fold0_development_tuning_report",
        "unbiased_oof": False,
        "run_id": run_directory.name,
        "decision": decision,
        "run_status": run_status,
        "metrics": metrics,
        "target": TARGET,
        "target_gap": {
            name: float(metrics[name] - threshold) for name, threshold in TARGET.items()
        },
        "canonical_baseline": CANONICAL_BASELINE,
        "canonical_delta": {
            name: float(metrics[name] - baseline)
            for name, baseline in CANONICAL_BASELINE.items()
        },
        "selected_epoch": selected_epoch,
        "training_log_at_selected_epoch": {
            "train_accuracy": float(selected_row["train_accuracy"]),
            "train_macro_f1": float(selected_row["train_macro_f1"]),
            "val_accuracy": float(selected_row["val_accuracy"]),
            "val_macro_f1": float(selected_row["val_macro_f1"]),
        },
        "unfrozen_backbone_blocks": summary.get("unfrozen_backbone_blocks"),
        "trainable_backbone_parameters": summary.get(
            "trainable_backbone_parameters_last_epoch"
        ),
        "artifacts": {
            "prediction_archive": str(archive_path.resolve()),
            "prediction_archive_sha256": trainer._sha256_file(archive_path),
            "checkpoint": str((run_directory / "best_accuracy.pt").resolve()),
            "checkpoint_sha256": trainer._sha256_file(run_directory / "best_accuracy.pt"),
            "resolved_config_sha256": summary["resolved_config_sha256"],
        },
        "canonical_artifact_sha256_after": canonical_after,
        "canonical_artifacts_unchanged": canonical_unchanged,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# X3D-S Fold0 Generalization Tuning",
        "",
        "> Development-only fold0 evidence. This is not unbiased OOF and does not replace Phase 4/5 evidence.",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "| Metric | Candidate | Target | Canonical fold0 | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("accuracy", "macro_f1", "worst_user_accuracy"):
        lines.append(
            f"| {name} | {metrics[name]:.6f} | {report['target'][name]:.6f} | "
            f"{report['canonical_baseline'][name]:.6f} | {report['canonical_delta'][name]:+.6f} |"
        )
    lines.extend(["", "## Per User", "", "| User | Accuracy |", "|---|---:|"])
    for user, accuracy in metrics["per_user_accuracy"].items():
        lines.append(f"| {user} | {accuracy:.6f} |")
    lines.extend(["", "## Duration", "", "| Frames | N | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"])
    for bucket, values in metrics["duration_buckets"].items():
        lines.append(
            f"| {bucket} | {values['sample_count']} | {values['accuracy']:.6f} | {values['macro_f1']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report protected X3D fold0 tuning evidence")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/x3d_s_fold0_generalization_tuning.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/x3d_s_fold0_generalization_tuning.md"),
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
