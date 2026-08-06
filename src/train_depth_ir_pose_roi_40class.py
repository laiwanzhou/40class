from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.engine import classification_metrics, collect_predictions, run_epoch
from src.train_unimodal import (
    PROJECT_ROOT,
    build_datasets,
    build_model,
    load_config,
    loader_for,
    save_confusion_matrix,
    set_seed,
)


RUN_ID = "depth_ir_pose_roi_40class_fold0_14train_4val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/depth_ir_pose_roi_40class.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args()


def generic_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=None,
        manifest=None,
        fold=None,
        output_root=None,
        device=None,
        seed=None,
        smoke_test=args.smoke_test,
        max_epochs=1 if args.smoke_test else None,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        run_id=args.run_id,
    )


def detailed_metrics(labels: np.ndarray, logits: np.ndarray, loss: float) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    base = classification_metrics(labels, predictions, logits.shape[1])
    probabilities = torch.from_numpy(logits).softmax(dim=1).numpy()
    ranks = 1 + (logits > logits[np.arange(len(labels)), labels, None]).sum(axis=1)
    predicted_count = np.bincount(predictions, minlength=logits.shape[1])
    top3 = np.argpartition(logits, -3, axis=1)[:, -3:]
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    return {
        **base,
        "loss": float(loss),
        "macro_precision": float(np.mean(base["per_class_precision"])),
        "macro_recall": float(np.mean(base["per_class_recall"])),
        "top3_accuracy": float(np.mean(np.any(top3 == labels[:, None], axis=1))),
        "top5_accuracy": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
        "zero_f1_class_count": int(np.sum(np.asarray(base["per_class_f1"]) == 0)),
        "zero_recall_class_count": int(np.sum(np.asarray(base["per_class_recall"]) == 0)),
        "never_predicted_class_count": int(np.sum(predicted_count == 0)),
        "number_of_predicted_classes": int(np.sum(predicted_count > 0)),
        "predicted_count": predicted_count.tolist(),
        "true_class_rank_mean": float(ranks.mean()),
        "probabilities": probabilities,
        "true_class_rank": ranks.astype(np.int64),
    }


def per_class_rows(
    epoch: int,
    labels: np.ndarray,
    logits: np.ndarray,
    metrics: dict[str, Any],
    class_names: list[str],
    train_support: np.ndarray,
    val_support: np.ndarray,
) -> list[dict[str, Any]]:
    predictions = logits.argmax(axis=1)
    probabilities = np.asarray(metrics["probabilities"])
    ranks = np.asarray(metrics["true_class_rank"])
    matrix = np.asarray(metrics["confusion_matrix"])
    rows = []
    for class_id, action_name in enumerate(class_names):
        selected = labels == class_id
        confused = matrix[class_id].copy()
        confused[class_id] = 0
        top_confused = int(confused.argmax()) if confused.sum() else -1
        rows.append(
            {
                "epoch": epoch,
                "class_id": class_id,
                "action_name": action_name,
                "train_support": int(train_support[class_id]),
                "val_support": int(val_support[class_id]),
                "precision": metrics["per_class_precision"][class_id],
                "recall": metrics["per_class_recall"][class_id],
                "f1": metrics["per_class_f1"][class_id],
                "predicted_count": int(np.sum(predictions == class_id)),
                "correct_count": int(matrix[class_id, class_id]),
                "true_class_probability_mean": float(probabilities[selected, class_id].mean()) if selected.any() else np.nan,
                "true_class_rank_mean": float(ranks[selected].mean()) if selected.any() else np.nan,
                "true_class_rank_median": float(np.median(ranks[selected])) if selected.any() else np.nan,
                "top_confused_class_id": top_confused,
                "top_confused_action": class_names[top_confused] if top_confused >= 0 else "",
                "top_confusion_count": int(confused[top_confused]) if top_confused >= 0 else 0,
            }
        )
    return rows


def save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict[str, Any]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def plot_history(history: pd.DataFrame, run_dir: Path) -> None:
    plots = {
        "accuracy_macro_f1.png": (("accuracy", "macro_f1"), "Validation score"),
        "val_loss.png": (("val_loss",), "Validation loss"),
        "zero_f1_classes.png": (("zero_f1_class_count",), "Zero-F1 classes"),
        "never_predicted_classes.png": (("never_predicted_class_count",), "Never-predicted classes"),
        "predicted_class_count.png": (("number_of_predicted_classes",), "Predicted class count"),
    }
    for filename, (columns, title) in plots.items():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for column in columns:
            axis.plot(history["epoch"], history[column], marker="o", markersize=2, label=column)
        axis.set_xlabel("Epoch")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if len(columns) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(run_dir / filename, dpi=140)
        plt.close(figure)


def evaluate_checkpoint(
    suffix: str,
    checkpoint_path: Path,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    run_dir: Path,
    class_names: list[str],
    original_class_ids: np.ndarray,
    train_support: np.ndarray,
    val_support: np.ndarray,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    output = collect_predictions(model, loader, criterion, device, amp_enabled)
    labels = np.asarray(output["labels"])
    logits = np.asarray(output["logits"])
    metrics = detailed_metrics(labels, logits, float(output["metrics"]["loss"]))
    probabilities = np.asarray(metrics.pop("probabilities"))
    true_ranks = np.asarray(metrics.pop("true_class_rank"))
    predictions = logits.argmax(axis=1)
    matrix = np.asarray(metrics["confusion_matrix"])
    rows = per_class_rows(
        int(checkpoint["epoch"]), labels, logits, {**metrics, "probabilities": probabilities, "true_class_rank": true_ranks},
        class_names, train_support, val_support,
    )
    max_confidence = probabilities.max(axis=1)
    for row in rows:
        class_id = int(row["class_id"])
        selected = labels == class_id
        correct = selected & (predictions == labels)
        wrong = selected & (predictions != labels)
        row["correct_confidence_mean"] = float(max_confidence[correct].mean()) if correct.any() else np.nan
        row["wrong_confidence_mean"] = float(max_confidence[wrong].mean()) if wrong.any() else np.nan
    pd.DataFrame(rows).to_csv(run_dir / f"per_class_{suffix}.csv", index=False, encoding="utf-8-sig")
    save_confusion_matrix(matrix.tolist(), run_dir / f"confusion_matrix_{suffix}.png")
    np.savez_compressed(
        run_dir / f"val_predictions_{suffix}.npz",
        sample_ids=output["sample_ids"],
        labels=labels,
        predicted=predictions,
        logits=logits,
        probabilities=probabilities,
        embeddings=output["embeddings"],
        roi_attention=output["roi_attention"],
        modality_gate=output["modality_gate"],
        temporal_mask=output["temporal_mask"],
        true_class_rank=true_ranks,
        original_class_ids=original_class_ids,
        action_names=np.asarray(class_names),
    )
    serializable = {
        key: value for key, value in metrics.items()
        if key not in {"confusion_matrix", "per_class_precision", "per_class_recall", "per_class_f1", "per_class_support", "predicted_count"}
    }
    serializable.update(
        {
            "checkpoint": checkpoint_path.name,
            "epoch": int(checkpoint["epoch"]),
            "confusion_matrix": metrics["confusion_matrix"],
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"],
            "per_class_support": metrics["per_class_support"],
            "predicted_count": metrics["predicted_count"],
        }
    )
    (run_dir / f"metrics_{suffix}.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    return serializable


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config, generic_args(args))
    set_seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    run_id = f"{args.run_id}_smoke" if args.smoke_test and not str(args.run_id).endswith("_smoke") else args.run_id
    run_dir = Path(config["output_root"]) / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config["run_dir"] = str(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    train_dataset, val_dataset = build_datasets(config)
    if len(getattr(train_dataset, "class_names")) != 40 or getattr(train_dataset, "original_class_ids") != list(range(40)):
        raise ValueError("Dataset does not preserve original class IDs 0-39.")
    class_names = list(getattr(train_dataset, "class_names"))
    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv")
    train_support = class_map.sort_values("class_id")["train_support"].to_numpy(dtype=np.int64)
    val_support = class_map.sort_values("class_id")["val_support"].to_numpy(dtype=np.int64)
    model = build_model(config, train_dataset[0]).to(device)
    train_loader = loader_for(train_dataset, config, training=True)
    val_loader = loader_for(val_dataset, config, training=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_accuracy = (-1.0, -1.0)
    best_macro = (-1.0, -1.0)
    history: list[dict[str, Any]] = []
    class_history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(
            model, train_loader, criterion, device, amp_enabled, optimizer=optimizer, scaler=scaler,
            max_batches=config.get("max_train_batches"), gradient_clip=float(config["gradient_clip"]),
        )
        validation = collect_predictions(model, val_loader, criterion, device, amp_enabled)
        labels = np.asarray(validation["labels"])
        logits = np.asarray(validation["logits"])
        metrics = detailed_metrics(labels, logits, float(validation["metrics"]["loss"]))
        elapsed = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "top3_accuracy": metrics["top3_accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "zero_f1_class_count": metrics["zero_f1_class_count"],
            "zero_recall_class_count": metrics["zero_recall_class_count"],
            "never_predicted_class_count": metrics["never_predicted_class_count"],
            "number_of_predicted_classes": metrics["number_of_predicted_classes"],
            "epoch_time_seconds": elapsed,
            "gpu_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
            "gpu_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        class_history.extend(
            per_class_rows(epoch, labels, logits, metrics, class_names, train_support, val_support)
        )
        accuracy_key = (float(metrics["accuracy"]), float(metrics["macro_f1"]))
        macro_key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
        if accuracy_key > best_accuracy:
            best_accuracy = accuracy_key
            save_checkpoint(run_dir / "best_accuracy.pt", model, epoch, metrics)
        if macro_key > best_macro:
            best_macro = macro_key
            save_checkpoint(run_dir / "best_macro_f1.pt", model, epoch, metrics)
        scheduler.step()
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(class_history).to_csv(run_dir / "epoch_per_class_diagnostics.csv", index=False, encoding="utf-8-sig")
        save_checkpoint(run_dir / "last_complete.pt", model, epoch, metrics)
        print(json.dumps(row), flush=True)
    save_checkpoint(run_dir / "last_model.pt", model, epochs, metrics)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(run_dir / "epoch_summary.csv", index=False, encoding="utf-8-sig")
    plot_history(history_frame, run_dir)
    evaluations = {}
    for suffix, filename in (
        ("best_accuracy", "best_accuracy.pt"),
        ("best_macro_f1", "best_macro_f1.pt"),
        ("last", "last_model.pt"),
    ):
        evaluations[suffix] = evaluate_checkpoint(
            suffix, run_dir / filename, model, val_loader, criterion, device, amp_enabled, run_dir,
            class_names, np.arange(40, dtype=np.int64), train_support, val_support,
        )
    result = {
        "status": "passed",
        "smoke_test": bool(args.smoke_test),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_completed": epochs,
        "batch_size": int(config["batch_size"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "total_training_time_seconds": time.perf_counter() - started,
        "mean_epoch_time_seconds": float(history_frame["epoch_time_seconds"].mean()),
        "peak_allocated_mb": float(history_frame["gpu_peak_allocated_mb"].max()),
        "peak_reserved_mb": float(history_frame["gpu_peak_reserved_mb"].max()),
        "evaluations": evaluations,
        "output_dir": str(run_dir),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT_JSON={json.dumps(result)}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
