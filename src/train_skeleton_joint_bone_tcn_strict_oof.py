from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.clean_skeleton_joint_bone_dataset import CleanSkeletonJointBoneDataset
from src.engine import run_epoch
from src.models import TemporalClassifier
from src.train_skeleton_c0_c1_strict_oof import (
    formal_refit_and_predict,
    resolve,
    set_normalization,
    set_seed,
    sha256,
    summarize,
    train_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "D2_joint_bone_tcn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen D2-v1 Joint/Bone TCN strict OOF experiment.")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/skeleton_joint_bone_tcn_strict_oof.yaml",
    )
    parser.add_argument("--folds", nargs="*", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def model_for(config: dict[str, Any]) -> TemporalClassifier:
    return TemporalClassifier(
        input_features=int(config["input_features"]),
        embedding_dim=int(config["embedding_dim"]),
        num_classes=int(config["num_classes"]),
        channels=tuple(int(value) for value in config["tcn_channels"]),
        dropout=float(config["dropout"]),
    )


def dataset_for(
    view: Path, data_root: Path, users: list[str], config: dict[str, Any],
) -> CleanSkeletonJointBoneDataset:
    if str(config["scale_policy"]) != "per_frame" or int(config["input_features"]) != 198:
        raise ValueError("D2-v1 is frozen to C1 per-frame scale and 198 Joint/Bone features")
    return CleanSkeletonJointBoneDataset(
        view, data_root, set(users), sequence_length=int(config["sequence_length"])
    )


def run_smoke_test(config: dict[str, Any], assignment: dict[str, Any], device: torch.device) -> None:
    fold = next(item for item in assignment["folds"] if int(item["fold"]) == 0)
    view = resolve(config["strict_views_root"]) / "fold_0/inner_selection/clean_view.csv"
    dataset = dataset_for(
        view, resolve(config["data_root"]), list(fold["epoch_selection"]["fit_user_ids"]), config
    )
    subset = Subset(dataset, range(min(2, len(dataset))))
    batch_loader = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0)
    set_seed(int(config["seed"]))
    model = model_for(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]) and device.type == "cuda")
    metrics = run_epoch(
        model, batch_loader, nn.CrossEntropyLoss(), device,
        bool(config["amp"]) and device.type == "cuda", optimizer=optimizer, scaler=scaler, max_batches=1,
        gradient_clip=float(config["gradient_clip"]),
    )
    print(json.dumps({
        "status": "passed", "mode": "real_data_single_batch_train", "fold": 0,
        "scope": "inner_selection_fit_users_only", "samples": len(subset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loss": metrics["loss"], "device": str(device),
    }, indent=2))


def run_fold(config: dict[str, Any], fold: dict[str, Any], device: torch.device) -> dict[str, Any]:
    fold_index = int(fold["fold"])
    seed = int(config["seed"]) + fold_index
    run_dir = resolve(config["output_root"]) / MODEL_NAME / f"fold_{fold_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    inner_view = resolve(config["strict_views_root"]) / f"fold_{fold_index}/inner_selection/clean_view.csv"
    formal_view = resolve(config["strict_views_root"]) / f"fold_{fold_index}/formal_outer/clean_view.csv"
    inner_fit_users = list(fold["epoch_selection"]["fit_user_ids"])
    inner_validation_users = list(fold["epoch_selection"]["validation_user_ids"])
    outer_train_users = list(fold["train_user_ids"])
    outer_validation_users = list(fold["validation_user_ids"])

    inner_fit = dataset_for(inner_view, resolve(config["data_root"]), inner_fit_users, config)
    inner_validation = dataset_for(inner_view, resolve(config["data_root"]), inner_validation_users, config)
    inner_counts = (len(inner_fit), len(inner_validation))
    set_normalization(inner_fit, inner_validation, run_dir / "inner_normalization.json", inner_fit_users)
    selected_epoch, _ = train_selection(
        config, inner_fit, inner_validation, device, seed, run_dir, model_builder=model_for
    )
    del inner_fit, inner_validation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    outer_train = dataset_for(formal_view, resolve(config["data_root"]), outer_train_users, config)
    outer_validation = dataset_for(formal_view, resolve(config["data_root"]), outer_validation_users, config)
    set_normalization(outer_train, outer_validation, run_dir / "outer_normalization.json", outer_train_users)
    predictions, metrics, by_user = formal_refit_and_predict(
        config, outer_train, outer_validation, selected_epoch, device, seed, run_dir,
        model_builder=model_for,
    )
    predictions.insert(0, "fold", fold_index)
    predictions.insert(0, "representation", MODEL_NAME)
    predictions.to_csv(run_dir / "formal_outer_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "representation": MODEL_NAME, "scale_policy": "per_frame", "fold": fold_index,
        "selected_epoch": selected_epoch,
        "inner_fit_users": sorted(inner_fit_users), "inner_validation_users": sorted(inner_validation_users),
        "outer_train_users": sorted(outer_train_users), "outer_validation_users": sorted(outer_validation_users),
        "inner_fit_trials": inner_counts[0], "inner_validation_trials": inner_counts[1],
        "outer_train_trials": len(outer_train), "outer_validation_trials": len(outer_validation),
        "outer_accuracy": float(metrics["accuracy"]), "outer_macro_f1_40class": float(metrics["macro_f1"]),
        "outer_macro_f1_supported": float(metrics["supported_class_macro_f1"]),
        "outer_weighted_f1": float(metrics["weighted_f1"]),
        "outer_worst_user_accuracy": float(by_user["accuracy"].min()),
        "parameter_count": sum(parameter.numel() for parameter in model_for(config).parameters()),
        "feature_order": "joint_17x6_then_h36m_edge_order_bone_16x6",
        "oof_assignment_sha256": sha256(resolve(config["oof_folds"])),
        "inner_clean_view_sha256": sha256(inner_view), "formal_clean_view_sha256": sha256(formal_view),
        "outer_validation_labels_used_for_selection": False,
        "preprocessing_fit_excludes_scope_validation_users": True,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"FOLD_RESULT={json.dumps(summary)}")
    return summary


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if args.max_epochs:
        config["epochs"] = args.max_epochs
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    assignment = json.loads(resolve(config["oof_folds"]).read_text(encoding="utf-8"))
    if args.smoke_test:
        run_smoke_test(config, assignment, device)
        return
    folds = args.folds or [0, 1, 2]
    for fold_index in folds:
        fold = next(item for item in assignment["folds"] if int(item["fold"]) == fold_index)
        run_fold(config, fold, device)
    summarize(config, [MODEL_NAME], folds)


if __name__ == "__main__":
    main()
