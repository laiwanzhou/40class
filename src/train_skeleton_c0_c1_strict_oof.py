from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from src.data.clean_skeleton_dataset import CleanSkeletonDataset
from src.data.common import compute_sequence_normalization
from src.engine import collect_predictions, run_epoch
from src.models import TemporalClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict three-fold Skeleton C0/C1 OOF experiment.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/skeleton_c0_c1_strict_oof.yaml")
    parser.add_argument("--representations", nargs="*", choices=("C0", "C1"))
    parser.add_argument("--folds", nargs="*", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(dataset: CleanSkeletonDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def model_for(config: dict[str, Any]) -> TemporalClassifier:
    return TemporalClassifier(
        input_features=102, embedding_dim=int(config["embedding_dim"]),
        num_classes=int(config["num_classes"]),
        channels=tuple(int(value) for value in config["tcn_channels"]),
        dropout=float(config["dropout"]),
    )


def set_normalization(
    train: CleanSkeletonDataset, validation: CleanSkeletonDataset, output_path: Path, source_users: list[str],
) -> None:
    mean, std = compute_sequence_normalization(train)
    train.set_normalization(mean, std)
    validation.set_normalization(mean, std)
    output_path.write_text(json.dumps({
        "source_user_ids": sorted(source_users), "samples_used": len(train),
        "mean": mean.tolist(), "std": std.tolist(),
    }, indent=2) + "\n", encoding="utf-8")


def optimizer_for(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )


def train_selection(
    config: dict[str, Any], train: CleanSkeletonDataset, validation: CleanSkeletonDataset,
    device: torch.device, seed: int, output_dir: Path,
    model_builder: Callable[[dict[str, Any]], nn.Module] | None = None,
) -> tuple[int, pd.DataFrame]:
    set_seed(seed)
    model = (model_builder or model_for)(config).to(device)
    optimizer = optimizer_for(model, config)
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()
    train_loader = loader(train, int(config["batch_size"]), True, seed)
    validation_loader = loader(validation, int(config["batch_size"]), False, seed + 1)
    best_accuracy = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, amp, optimizer=optimizer, scaler=scaler,
            gradient_clip=float(config["gradient_clip"]),
        )
        validation_metrics = run_epoch(model, validation_loader, criterion, device, amp)
        row = {
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"], "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1_40class": validation_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row))
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= int(config["early_stopping_patience"]):
            break
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "inner_selection_history.csv", index=False, encoding="utf-8-sig")
    return best_epoch, frame


def formal_refit_and_predict(
    config: dict[str, Any], train: CleanSkeletonDataset, validation: CleanSkeletonDataset,
    selected_epoch: int, device: torch.device, seed: int, output_dir: Path,
    model_builder: Callable[[dict[str, Any]], nn.Module] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    set_seed(seed)
    model = (model_builder or model_for)(config).to(device)
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=selected_epoch)
    amp = bool(config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()
    train_loader = loader(train, int(config["batch_size"]), True, seed)
    validation_loader = loader(validation, int(config["batch_size"]), False, seed + 1)
    history = []
    for epoch in range(1, selected_epoch + 1):
        started = time.perf_counter()
        metrics = run_epoch(
            model, train_loader, criterion, device, amp, optimizer=optimizer, scaler=scaler,
            gradient_clip=float(config["gradient_clip"]),
        )
        history.append({
            "epoch": epoch, "train_loss": metrics["loss"], "train_accuracy": metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started,
        })
        scheduler.step()
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "formal_refit_history.csv", index=False, encoding="utf-8-sig")
    outcome = collect_predictions(model, validation_loader, criterion, device, amp)
    logits = np.asarray(outcome["logits"])
    labels = np.asarray(outcome["labels"])
    predictions = logits.argmax(axis=1)
    sample_to_user = {str(group.iloc[0]["sample_id"]): str(group.iloc[0]["user_id"]) for group in validation.groups}
    prediction_frame = pd.DataFrame({
        "sample_id": outcome["sample_ids"],
        "user_id": [sample_to_user[str(sample_id)] for sample_id in outcome["sample_ids"]],
        "label": labels, "prediction": predictions,
    })
    for class_id in range(int(config["num_classes"])):
        prediction_frame[f"logit_{class_id:02d}"] = logits[:, class_id]
    metrics = dict(outcome["metrics"])
    metrics["supported_class_macro_f1"] = float(f1_score(
        labels, predictions, labels=np.unique(labels), average="macro", zero_division=0
    ))
    by_user = prediction_frame.groupby("user_id").apply(
        lambda group: pd.Series({
            "samples": len(group), "accuracy": accuracy_score(group["label"], group["prediction"]),
            "macro_f1_supported": f1_score(
                group["label"], group["prediction"], labels=np.unique(group["label"]),
                average="macro", zero_division=0,
            ),
        }), include_groups=False,
    ).reset_index()
    by_user.to_csv(output_dir / "formal_outer_per_user.csv", index=False, encoding="utf-8-sig")
    return prediction_frame, metrics, by_user


def dataset_for(view: Path, data_root: Path, users: list[str], policy: str, sequence_length: int) -> CleanSkeletonDataset:
    return CleanSkeletonDataset(view, data_root, set(users), policy, sequence_length)


def run_arm_fold(
    config: dict[str, Any], assignment: dict, arm: str, policy: str, fold: dict,
    device: torch.device,
) -> dict[str, Any]:
    fold_index = int(fold["fold"])
    seed = int(config["seed"]) + fold_index
    run_dir = resolve(config["output_root"]) / arm / f"fold_{fold_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    inner_view = resolve(config["strict_views_root"]) / f"fold_{fold_index}/inner_selection/clean_view.csv"
    formal_view = resolve(config["strict_views_root"]) / f"fold_{fold_index}/formal_outer/clean_view.csv"
    inner_fit_users = list(fold["epoch_selection"]["fit_user_ids"])
    inner_validation_users = list(fold["epoch_selection"]["validation_user_ids"])
    outer_train_users = list(fold["train_user_ids"])
    outer_validation_users = list(fold["validation_user_ids"])
    inner_fit = dataset_for(inner_view, resolve(config["data_root"]), inner_fit_users, policy, int(config["sequence_length"]))
    inner_validation = dataset_for(inner_view, resolve(config["data_root"]), inner_validation_users, policy, int(config["sequence_length"]))
    inner_fit_trials = len(inner_fit)
    inner_validation_trials = len(inner_validation)
    set_normalization(inner_fit, inner_validation, run_dir / "inner_normalization.json", inner_fit_users)
    selected_epoch, selection_history = train_selection(
        config, inner_fit, inner_validation, device, seed, run_dir
    )
    del inner_fit, inner_validation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    outer_train = dataset_for(formal_view, resolve(config["data_root"]), outer_train_users, policy, int(config["sequence_length"]))
    outer_validation = dataset_for(formal_view, resolve(config["data_root"]), outer_validation_users, policy, int(config["sequence_length"]))
    set_normalization(outer_train, outer_validation, run_dir / "outer_normalization.json", outer_train_users)
    predictions, metrics, by_user = formal_refit_and_predict(
        config, outer_train, outer_validation, selected_epoch, device, seed, run_dir
    )
    predictions.insert(0, "fold", fold_index)
    predictions.insert(0, "representation", arm)
    predictions.to_csv(run_dir / "formal_outer_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "representation": arm, "scale_policy": policy, "fold": fold_index,
        "selected_epoch": selected_epoch,
        "inner_fit_users": sorted(inner_fit_users), "inner_validation_users": sorted(inner_validation_users),
        "outer_train_users": sorted(outer_train_users), "outer_validation_users": sorted(outer_validation_users),
        "inner_fit_trials": inner_fit_trials, "inner_validation_trials": inner_validation_trials,
        "outer_train_trials": len(outer_train), "outer_validation_trials": len(outer_validation),
        "outer_accuracy": float(metrics["accuracy"]), "outer_macro_f1_40class": float(metrics["macro_f1"]),
        "outer_macro_f1_supported": float(metrics["supported_class_macro_f1"]),
        "outer_weighted_f1": float(metrics["weighted_f1"]),
        "outer_worst_user_accuracy": float(by_user["accuracy"].min()),
        "oof_assignment_sha256": sha256(resolve(config["oof_folds"])),
        "inner_clean_view_sha256": sha256(inner_view), "formal_clean_view_sha256": sha256(formal_view),
        "outer_validation_labels_used_for_selection": False,
        "preprocessing_fit_excludes_scope_validation_users": True,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"FOLD_RESULT={json.dumps(summary)}")
    return summary


def summarize(config: dict[str, Any], arms: list[str], folds: list[int]) -> None:
    report_dir = resolve(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for arm in arms:
        for fold in folds:
            path = resolve(config["output_root"]) / arm / f"fold_{fold}/summary.json"
            if path.is_file():
                fold_rows.append(json.loads(path.read_text(encoding="utf-8")))
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(report_dir / "fold_results.csv", index=False, encoding="utf-8-sig")
    combined_rows = []
    all_predictions = []
    per_class_rows = []
    for arm in arms:
        paths = [resolve(config["output_root"]) / arm / f"fold_{fold}/formal_outer_predictions.csv" for fold in folds]
        if not all(path.is_file() for path in paths):
            continue
        predictions = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in paths], ignore_index=True)
        if predictions["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate OOF sample IDs for {arm}")
        labels = predictions["label"].to_numpy()
        predicted = predictions["prediction"].to_numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predicted, labels=np.arange(40), zero_division=0
        )
        combined_rows.append({
            "representation": arm, "samples": len(predictions),
            "accuracy": accuracy_score(labels, predicted),
            "macro_f1_40class": f1_score(labels, predicted, labels=np.arange(40), average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, predicted, average="weighted", zero_division=0),
        })
        for class_id in range(40):
            per_class_rows.append({
                "representation": arm, "class_id": class_id, "support": int(support[class_id]),
                "precision": precision[class_id], "recall": recall[class_id], "f1": f1[class_id],
            })
        all_predictions.append(predictions)
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(report_dir / "combined_oof_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class_rows).to_csv(report_dir / "combined_oof_per_class.csv", index=False, encoding="utf-8-sig")
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            report_dir / "combined_oof_predictions.csv", index=False, encoding="utf-8-sig"
        )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if args.max_epochs:
        config["epochs"] = args.max_epochs
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    assignment = json.loads(resolve(config["oof_folds"]).read_text(encoding="utf-8"))
    arms = args.representations or list(config["representations"])
    folds = args.folds or [0, 1, 2]
    for arm in arms:
        policy = str(config["representations"][arm])
        for fold_index in folds:
            fold = next(item for item in assignment["folds"] if int(item["fold"]) == fold_index)
            run_arm_fold(config, assignment, arm, policy, fold, device)
    summarize(config, arms, folds)


if __name__ == "__main__":
    main()
