from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/experiments/ir_primary_depth_residual_fullseq.yaml"
REPRESENTATIONS = ("raw", "relative", "raw+relative")
SLUGS = {"raw": "raw", "relative": "relative", "raw+relative": "raw_relative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--skip-formal", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_training(
    representation: str,
    run_id: str,
    config: Path,
    *,
    epochs: int,
    patience: int,
) -> None:
    command = [
        sys.executable,
        "-m",
        "src.train_ir_primary_depth_residual_fullseq",
        "--config",
        str(config),
        "--depth-representation",
        representation,
        "--run-id",
        run_id,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def user_rows(prediction_path: Path, representation: str) -> list[dict[str, Any]]:
    data = np.load(prediction_path, allow_pickle=False)
    labels = data["labels"].astype(np.int64)
    logits = data["logits"].astype(np.float64)
    users = data["user_ids"].astype(str)
    predictions = logits.argmax(axis=1)
    rows = []
    for user in sorted(set(users)):
        selected = users == user
        rows.append({
            "depth_representation": representation,
            "user_id": user,
            "samples": int(selected.sum()),
            "accuracy": float(accuracy_score(labels[selected], predictions[selected])),
            "macro_f1": float(f1_score(labels[selected], predictions[selected], labels=range(40), average="macro", zero_division=0)),
        })
    return rows


def summarize_pilot(output_root: Path, run_id: str, representation: str, small_ids: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = output_root / run_id
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    checkpoint = summary["checkpoints"]["best_macro_f1"]
    epoch = int(checkpoint["epoch"])
    history = pd.read_csv(run_dir / "history.csv", encoding="utf-8-sig")
    epoch_row = history.loc[history.epoch == epoch].iloc[0]
    per_class = pd.read_csv(run_dir / "per_class_best_macro_f1.csv", encoding="utf-8-sig")
    user_metrics = user_rows(run_dir / "val_predictions_best_macro_f1.npz", representation)
    prediction = np.load(run_dir / "val_predictions_best_macro_f1.npz", allow_pickle=False)
    logits = prediction["logits"].astype(np.float64)
    labels = prediction["labels"].astype(np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1)
    high_confidence_errors = int(((predictions != labels) & (probabilities.max(axis=1) >= 0.8)).sum())
    f1_column = "f1" if "f1" in per_class.columns else "f1_score"
    small_macro = float(per_class.loc[per_class.class_id.isin(small_ids), f1_column].mean())
    row = {
        "depth_representation": representation,
        "run_id": run_id,
        "best_macro_epoch": epoch,
        "accuracy": float(checkpoint["accuracy"]),
        "macro_f1": float(checkpoint["macro_f1"]),
        "small_action_macro_f1": small_macro,
        "val_loss": float(checkpoint["val_loss"]),
        "train_accuracy": float(epoch_row.train_accuracy),
        "generalization_gap": float(epoch_row.generalization_gap),
        "weighted_f1": float(epoch_row.val_weighted_f1),
        "top3_accuracy": float(epoch_row.val_top3_accuracy),
        "top5_accuracy": float(epoch_row.val_top5_accuracy),
        "zero_f1_class_count": int(epoch_row.zero_f1_class_count),
        "predicted_class_count": int(epoch_row.number_of_predicted_classes),
        "worst_user_accuracy": min(item["accuracy"] for item in user_metrics),
        "worst_user_macro_f1": min(item["macro_f1"] for item in user_metrics),
        "high_confidence_errors_ge_0_8": high_confidence_errors,
    }
    return row, user_metrics


def select_representation(rows: list[dict[str, Any]], accuracy_tolerance: float) -> tuple[str, list[dict[str, Any]]]:
    best_accuracy = max(float(row["accuracy"]) for row in rows)
    for row in rows:
        row["accuracy_eligible"] = float(row["accuracy"]) >= best_accuracy - accuracy_tolerance
    eligible = [row for row in rows if row["accuracy_eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["small_action_macro_f1"]),
            -abs(float(row["generalization_gap"])),
            float(row["worst_user_macro_f1"]),
            float(row["accuracy"]),
        ),
    )
    return str(selected["depth_representation"]), rows


def write_report(rows: list[dict[str, Any]], user_metrics: list[dict[str, Any]], selected: str, tolerance: float) -> None:
    report_csv = PROJECT_ROOT / "reports/depth_ordinal_stage8_pilot_comparison.csv"
    users_csv = PROJECT_ROOT / "reports/depth_ordinal_stage8_pilot_per_user.csv"
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(user_metrics).to_csv(users_csv, index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows)[[
        "depth_representation", "best_macro_epoch", "accuracy", "macro_f1",
        "small_action_macro_f1", "generalization_gap", "worst_user_macro_f1",
        "zero_f1_class_count", "predicted_class_count", "accuracy_eligible",
    ]].to_markdown(index=False)
    markdown = f"""# Stage 8 Depth representation pilot comparison

All three pilots used the same split, seed, model, optimizer, frame budget, and fixed epoch budget.

Selection rule fixed before pilot results: retain representations within `{tolerance:.3f}` absolute Accuracy of the best pilot, then maximize validation Macro-F1, small-action Macro-F1, smaller absolute train/validation gap, worst-user Macro-F1, and Accuracy in that order.

{table}

Selected representation: **`{selected}`**.

Competition test was not read. Skeleton/IMU/Radar were not read or connected.
"""
    (PROJECT_ROOT / "reports/depth_ordinal_stage8_pilot_comparison.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    pilot_epochs = int(config["pilot_epochs"])
    pilot_patience = int(config["pilot_patience"])
    tolerance = float(config["pilot_accuracy_tolerance"])
    state_path = output_root / "stage8_stage9_pipeline_status.json"
    state: dict[str, Any] = {
        "status": "running_pilots",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pilot_epochs": pilot_epochs,
        "representations": list(REPRESENTATIONS),
        "completed_pilots": [],
        "competition_test_read": False,
        "sensor_modalities_read": False,
    }
    write_json(state_path, state)
    for representation in REPRESENTATIONS:
        run_id = f"stage8_pilot_{SLUGS[representation]}_e{pilot_epochs}"
        summary_path = output_root / run_id / "run_summary.json"
        if not summary_path.exists():
            state["active_representation"] = representation
            write_json(state_path, state)
            run_training(
                representation, run_id, config_path,
                epochs=pilot_epochs, patience=pilot_patience,
            )
        state["completed_pilots"].append(representation)
        write_json(state_path, state)

    rows: list[dict[str, Any]] = []
    per_user: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        run_id = f"stage8_pilot_{SLUGS[representation]}_e{pilot_epochs}"
        row, users = summarize_pilot(
            output_root, run_id, representation, list(config["small_object_class_ids"]),
        )
        rows.append(row)
        per_user.extend(users)
    selected, rows = select_representation(rows, tolerance)
    write_report(rows, per_user, selected, tolerance)
    state.update({"status": "pilot_selection_complete", "selected_representation": selected})
    write_json(state_path, state)
    if args.skip_formal:
        return

    formal_run_id = f"stage9_selected_{SLUGS[selected]}_formal"
    state.update({"status": "running_selected_formal", "formal_run_id": formal_run_id})
    write_json(state_path, state)
    formal_summary = output_root / formal_run_id / "run_summary.json"
    if not formal_summary.exists():
        run_training(
            selected, formal_run_id, config_path,
            epochs=int(config["epochs"]), patience=int(config["patience"]),
        )
    state.update({
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    write_json(state_path, state)


if __name__ == "__main__":
    main()
