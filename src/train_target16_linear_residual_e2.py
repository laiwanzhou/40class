from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.target16_linear_residual import Target16LinearResidual
from src.train_unimodal import PROJECT_ROOT


RUN_ID = "target16_linear_residual_e2_14train_4val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_target16_linear_residual_e2.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args()


def metrics(labels: np.ndarray, logits: np.ndarray, loss: float) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(16), zero_division=0,
    )
    return {
        "loss": float(loss),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "predicted_class_count": int(np.sum(np.bincount(predictions, minlength=16) > 0)),
        "zero_f1_class_count": int(np.sum(f1 == 0)),
        "per_class_f1": f1.tolist(),
    }


def target_arrays(archive: np.lib.npyio.NpzFile, target_ids: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(archive["labels"], dtype=np.int64)
    selected = np.isin(labels, target_ids)
    local_map = np.full(40, -1, dtype=np.int64)
    local_map[target_ids] = np.arange(16)
    return {
        "sample_ids": np.asarray(archive["sample_ids"], dtype=str)[selected],
        "user_ids": np.asarray(archive["user_ids"], dtype=str)[selected],
        "labels": local_map[labels[selected]],
        "original_labels": labels[selected],
        "embeddings": np.asarray(archive["embeddings"], dtype=np.float32)[selected],
        "base_logits": np.asarray(archive["logits"], dtype=np.float32)[selected],
    }


def make_loader(arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays["embeddings"]),
        torch.from_numpy(arrays["base_logits"]),
        torch.from_numpy(arrays["labels"]),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def run_epoch(
    model: Target16LinearResidual,
    loader: DataLoader,
    device: torch.device,
    residual_l2: float,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[dict[str, Any], np.ndarray]:
    training = optimizer is not None
    model.train(training)
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    total_loss = 0.0
    samples = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for embeddings, base_logits, labels in loader:
            embeddings = embeddings.to(device)
            base_logits = base_logits.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(embeddings, base_logits)
            ce = nn.functional.cross_entropy(output["logits"], labels)
            penalty = output["delta_logits"].square().mean()
            loss = ce + residual_l2 * penalty
            if training:
                loss.backward()
                optimizer.step()
            count = len(labels)
            total_loss += float(loss.detach()) * count
            samples += count
            labels_all.append(labels.detach().cpu())
            logits_all.append(output["logits"].detach().cpu())
    labels_np = torch.cat(labels_all).numpy()
    logits_np = torch.cat(logits_all).numpy()
    return metrics(labels_np, logits_np, total_loss / samples), logits_np


def save_checkpoint(
    path: Path,
    model: Target16LinearResidual,
    epoch: int,
    result: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "val_accuracy": result["accuracy"],
            "val_macro_f1": result["macro_f1"],
            "target_class_ids": list(model.target_class_ids),
            "b2_epoch": 25,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "config": config,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def infer(
    model: Target16LinearResidual,
    embeddings: np.ndarray,
    base_logits: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            output = model(
                torch.from_numpy(embeddings[start : start + batch_size]).to(device),
                torch.from_numpy(base_logits[start : start + batch_size]).to(device),
            )
            logits.append(output["logits"].cpu().numpy())
            deltas.append(output["delta_logits"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(deltas)


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    output_root = (PROJECT_ROOT / str(config["output_root"])).resolve()
    run_id = f"{args.run_id}_smoke" if args.smoke_test and not args.run_id.endswith("_smoke") else args.run_id
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config["run_dir"] = str(run_dir)
    config["smoke_test"] = bool(args.smoke_test)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv").sort_values("class_id")
    lookup = dict(zip(class_map["action_name"], class_map["class_id"], strict=True))
    target_ids = np.asarray(sorted(lookup[name] for name in config["target_actions"]), dtype=np.int64)
    base_dir = (PROJECT_ROOT / str(config["b2_archive_dir"])).resolve()
    train_raw = np.load(base_dir / "train_predictions_best_epoch25.npz", allow_pickle=False)
    val_raw = np.load(base_dir / "val_predictions_best_epoch25_with_users.npz", allow_pickle=False)
    train = target_arrays(train_raw, target_ids)
    val = target_arrays(val_raw, target_ids)
    if len(train["labels"]) != 880 or len(val["labels"]) != 222 or len(val_raw["labels"]) != 590:
        raise ValueError("B2 archive split integrity failed.")
    if set(train["user_ids"]) & set(val["user_ids"]):
        raise ValueError("Train and validation users overlap.")

    model = Target16LinearResidual(192, target_ids).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    if trainable != 3088:
        raise ValueError(f"Expected 3,088 trainable parameters, got {trainable}.")
    batch_size = int(config["batch_size"])
    train_loader = make_loader(train, batch_size, True, seed)
    val_loader = make_loader(val, batch_size, False, seed + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    epochs = 2 if args.smoke_test else int(config["epochs"])
    patience = 2 if args.smoke_test else int(config["early_stopping_patience"])
    residual_l2 = float(config["residual_output_l2"])
    initial_train, _ = run_epoch(model, train_loader, device, residual_l2, None)
    initial_val, _ = run_epoch(model, val_loader, device, residual_l2, None)
    initial_row = {
        "epoch": 0,
        "train_loss": initial_train["loss"],
        "train_accuracy": initial_train["accuracy"],
        "train_macro_f1": initial_train["macro_f1"],
        "val_loss": initial_val["loss"],
        "val_accuracy": initial_val["accuracy"],
        "val_macro_f1": initial_val["macro_f1"],
        "weighted_f1": initial_val["weighted_f1"],
        "predicted_class_count": initial_val["predicted_class_count"],
        "zero_f1_class_count": initial_val["zero_f1_class_count"],
        "generalization_gap": initial_train["accuracy"] - initial_val["accuracy"],
    }
    best_key = (float(initial_val["macro_f1"]), float(initial_val["accuracy"]))
    no_improvement = 0
    history: list[dict[str, Any]] = [initial_row]
    save_checkpoint(run_dir / "best_macro_f1.pt", model, 0, initial_val, config)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(initial_row), flush=True)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, device, residual_l2, optimizer)
        val_metrics, _ = run_epoch(model, val_loader, device, residual_l2, None)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "predicted_class_count": val_metrics["predicted_class_count"],
            "zero_f1_class_count": val_metrics["zero_f1_class_count"],
            "generalization_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
        }
        history.append(row)
        key = (float(val_metrics["macro_f1"]), float(val_metrics["accuracy"]))
        if key > best_key:
            best_key = key
            no_improvement = 0
            save_checkpoint(run_dir / "best_macro_f1.pt", model, epoch, val_metrics, config)
        else:
            no_improvement += 1
        save_checkpoint(run_dir / "last_complete.pt", model, epoch, val_metrics, config)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(row), flush=True)
        if no_improvement >= patience:
            print(json.dumps({"early_stop_epoch": epoch, "patience": patience}), flush=True)
            break

    checkpoint = torch.load(run_dir / "best_macro_f1.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    target_logits, target_delta = infer(model, val["embeddings"], val["base_logits"], device, batch_size)
    all_logits, all_delta = infer(
        model,
        np.asarray(val_raw["embeddings"], dtype=np.float32),
        np.asarray(val_raw["logits"], dtype=np.float32),
        device,
        batch_size,
    )
    np.savez_compressed(
        run_dir / "val_predictions_target16_best_macro_f1.npz",
        sample_ids=val["sample_ids"], user_ids=val["user_ids"], labels=val["labels"],
        original_labels=val["original_labels"], logits=target_logits,
        probabilities=torch.from_numpy(target_logits).softmax(1).numpy(),
        delta_logits=target_delta, embeddings=val["embeddings"], original_class_ids=target_ids,
    )
    np.savez_compressed(
        run_dir / "val_predictions_all40_best_macro_f1.npz",
        sample_ids=np.asarray(val_raw["sample_ids"], dtype=str),
        user_ids=np.asarray(val_raw["user_ids"], dtype=str),
        labels=np.asarray(val_raw["labels"], dtype=np.int64),
        logits=all_logits, probabilities=torch.from_numpy(all_logits).softmax(1).numpy(),
        delta_logits=all_delta, embeddings=np.asarray(val_raw["embeddings"], dtype=np.float32),
        original_class_ids=target_ids,
    )
    summary = {
        "status": "passed",
        "test_read": False,
        "train_samples": len(train["labels"]),
        "target_val_samples": len(val["labels"]),
        "full_val_samples": len(val_raw["labels"]),
        "epochs_completed": int(history[-1]["epoch"]),
        "best_macro_epoch": int(checkpoint["epoch"]),
        "best_macro_f1": float(checkpoint["val_macro_f1"]),
        "best_macro_accuracy": float(checkpoint["val_accuracy"]),
        "trainable_parameters": trainable,
        "runtime_seconds": time.perf_counter() - started,
        "target_class_ids": target_ids.tolist(),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run(parse_args())
