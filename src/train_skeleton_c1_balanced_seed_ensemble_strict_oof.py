from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.clean_skeleton_dataset import CleanSkeletonDataset
from src.train_skeleton_c0_c1_strict_oof import (
    formal_refit_and_predict,
    resolve,
    set_normalization,
    set_seed,
    sha256,
    train_selection,
)
from src.train_skeleton_c1_seed_ensemble_strict_oof import (
    dataset_for,
    ensemble_predictions,
    model_for,
    prediction_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATION = "E2_c1_tcn_inverse_sqrt_balanced_3seed_probability_ensemble"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen E2-v1 balanced E1 strict OOF experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/skeleton_c1_balanced_seed_ensemble_strict_oof.yaml",
    )
    parser.add_argument("--folds", nargs="*", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def balanced_sample_weights(labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Training labels must be a non-empty one-dimensional array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Training labels must be integer class IDs")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("Training labels are outside the configured class range")
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / np.sqrt(counts[labels].astype(np.float64))
    return weights, counts


def make_balanced_sampler(weights: np.ndarray, seed: int) -> WeightedRandomSampler:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0 or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Sampler weights must be a non-empty vector of finite positive values")
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def dataset_labels(dataset: CleanSkeletonDataset) -> np.ndarray:
    return np.asarray([int(group.iloc[0]["class_id"]) for group in dataset.groups], dtype=np.int64)


def balanced_loader(dataset: CleanSkeletonDataset, batch_size: int, seed: int) -> DataLoader:
    weights, _ = balanced_sample_weights(dataset_labels(dataset), num_classes=40)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=make_balanced_sampler(weights, seed),
        num_workers=0,
        pin_memory=True,
    )


def write_sampling_provenance(
    dataset: CleanSkeletonDataset,
    output_path: Path,
    source_users: list[str],
    seed: int,
    num_classes: int,
) -> dict[str, Any]:
    labels = dataset_labels(dataset)
    weights, counts = balanced_sample_weights(labels, num_classes)
    per_class_weight = {
        str(class_id): (None if count == 0 else float(1.0 / np.sqrt(count)))
        for class_id, count in enumerate(counts)
    }
    payload = {
        "strategy": "inverse_sqrt_class_frequency",
        "replacement": True,
        "num_samples_per_epoch": len(dataset),
        "source_user_ids": sorted(source_users),
        "source_trial_count": len(dataset),
        "seed": seed,
        "class_counts": {str(class_id): int(count) for class_id, count in enumerate(counts)},
        "per_sample_class_weight": per_class_weight,
        "sample_weight_min": float(weights.min()),
        "sample_weight_max": float(weights.max()),
        "validation_labels_used": False,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_member_fold(
    config: dict[str, Any], fold: dict[str, Any], member_seed: int, device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_index = int(fold["fold"])
    training_seed = member_seed + fold_index
    run_dir = resolve(config["output_root"]) / f"member_seed_{member_seed}" / f"fold_{fold_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    inner_view = resolve(config["strict_views_root"]) / f"fold_{fold_index}/inner_selection/clean_view.csv"
    formal_view = resolve(config["strict_views_root"]) / f"fold_{fold_index}/formal_outer/clean_view.csv"
    inner_fit_users = list(fold["epoch_selection"]["fit_user_ids"])
    inner_validation_users = list(fold["epoch_selection"]["validation_user_ids"])
    outer_train_users = list(fold["train_user_ids"])
    outer_validation_users = list(fold["validation_user_ids"])
    data_root = resolve(config["data_root"])

    inner_fit = dataset_for(inner_view, data_root, inner_fit_users, config)
    inner_validation = dataset_for(inner_view, data_root, inner_validation_users, config)
    inner_fit_trials, inner_validation_trials = len(inner_fit), len(inner_validation)
    set_normalization(inner_fit, inner_validation, run_dir / "inner_normalization.json", inner_fit_users)
    inner_sampling = write_sampling_provenance(
        inner_fit, run_dir / "inner_sampling.json", inner_fit_users, training_seed, int(config["num_classes"])
    )
    selected_epoch, _ = train_selection(
        config,
        inner_fit,
        inner_validation,
        device,
        training_seed,
        run_dir,
        model_builder=model_for,
        train_loader_builder=balanced_loader,
    )
    del inner_fit, inner_validation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    outer_train = dataset_for(formal_view, data_root, outer_train_users, config)
    outer_validation = dataset_for(formal_view, data_root, outer_validation_users, config)
    set_normalization(outer_train, outer_validation, run_dir / "outer_normalization.json", outer_train_users)
    outer_sampling = write_sampling_provenance(
        outer_train, run_dir / "outer_sampling.json", outer_train_users, training_seed, int(config["num_classes"])
    )
    predictions, metrics, by_user = formal_refit_and_predict(
        config,
        outer_train,
        outer_validation,
        selected_epoch,
        device,
        training_seed,
        run_dir,
        model_builder=model_for,
        train_loader_builder=balanced_loader,
    )
    predictions.insert(0, "seed", member_seed)
    predictions.insert(0, "fold", fold_index)
    predictions.insert(0, "representation", "E2_member")
    predictions.to_csv(run_dir / "formal_outer_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "representation": "E2_member",
        "member_seed": member_seed,
        "training_seed": training_seed,
        "scale_policy": str(config["scale_policy"]),
        "sequence_length": int(config["sequence_length"]),
        "model_class": "TemporalClassifier",
        "sampling_strategy": inner_sampling["strategy"],
        "sampling_replacement": True,
        "samples_per_epoch_policy": "train_scope_size",
        "fold": fold_index,
        "selected_epoch": selected_epoch,
        "inner_fit_users": sorted(inner_fit_users),
        "inner_validation_users": sorted(inner_validation_users),
        "outer_train_users": sorted(outer_train_users),
        "outer_validation_users": sorted(outer_validation_users),
        "inner_fit_trials": inner_fit_trials,
        "inner_validation_trials": inner_validation_trials,
        "outer_train_trials": len(outer_train),
        "outer_validation_trials": len(outer_validation),
        "inner_sampling_source_trials": inner_sampling["source_trial_count"],
        "outer_sampling_source_trials": outer_sampling["source_trial_count"],
        "outer_accuracy": float(metrics["accuracy"]),
        "outer_macro_f1_40class": float(metrics["macro_f1"]),
        "outer_weighted_f1": float(metrics["weighted_f1"]),
        "outer_worst_user_accuracy": float(by_user["accuracy"].min()),
        "parameter_count_per_member": sum(parameter.numel() for parameter in model_for(config).parameters()),
        "oof_assignment_sha256": sha256(resolve(config["oof_folds"])),
        "inner_clean_view_sha256": sha256(inner_view),
        "formal_clean_view_sha256": sha256(formal_view),
        "outer_validation_labels_used_for_selection": False,
        "sampling_fit_excludes_scope_validation_users": True,
        "preprocessing_fit_excludes_scope_validation_users": True,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"MEMBER_RESULT={json.dumps(summary)}")
    return predictions, summary


def summarize(config: dict[str, Any], folds: list[int], seeds: list[int]) -> None:
    report_dir = resolve(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    ensemble_frames: list[pd.DataFrame] = []
    ensemble_summaries: list[dict[str, Any]] = []
    member_summaries: list[dict[str, Any]] = []
    for fold in folds:
        ensemble_dir = resolve(config["output_root"]) / "ensemble" / f"fold_{fold}"
        ensemble_path = ensemble_dir / "formal_outer_predictions.csv"
        summary_path = ensemble_dir / "summary.json"
        if ensemble_path.is_file() and summary_path.is_file():
            ensemble_frames.append(pd.read_csv(ensemble_path, encoding="utf-8-sig"))
            ensemble_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        for seed in seeds:
            path = resolve(config["output_root"]) / f"member_seed_{seed}" / f"fold_{fold}/summary.json"
            if path.is_file():
                member_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    pd.DataFrame(member_summaries).to_csv(
        report_dir / "member_fold_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(ensemble_summaries).to_csv(
        report_dir / "ensemble_fold_results.csv", index=False, encoding="utf-8-sig"
    )
    if not ensemble_frames:
        return
    combined = pd.concat(ensemble_frames, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError("Duplicate combined E2 OOF sample IDs")
    combined.to_csv(report_dir / "combined_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = prediction_metrics(combined, int(config["num_classes"]))
    pd.DataFrame([{"representation": REPRESENTATION, "samples": len(combined), **metrics}]).to_csv(
        report_dir / "combined_oof_summary.csv", index=False, encoding="utf-8-sig"
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        combined["label"],
        combined["prediction"],
        labels=np.arange(int(config["num_classes"])),
        zero_division=0,
    )
    pd.DataFrame([{
        "representation": REPRESENTATION,
        "class_id": class_id,
        "support": int(support[class_id]),
        "precision": precision[class_id],
        "recall": recall[class_id],
        "f1": f1[class_id],
    } for class_id in range(int(config["num_classes"]))]).to_csv(
        report_dir / "combined_oof_per_class.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if args.max_epochs:
        config["epochs"] = args.max_epochs
    if args.smoke_test:
        config["epochs"] = 1
        config["output_root"] = "outputs/skeleton_c1_balanced_seed_ensemble_strict_oof_smoke"
        config["report_dir"] = "reports/skeleton_c1_balanced_seed_ensemble_strict_oof_smoke"
    expected_sampling = {
        "strategy": "inverse_sqrt_class_frequency",
        "replacement": True,
        "num_samples": "train_scope_size",
    }
    if config.get("sampling") != expected_sampling:
        raise ValueError("E2-v1 sampling contract differs from the frozen inverse-sqrt policy")
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    assignment = json.loads(resolve(config["oof_folds"]).read_text(encoding="utf-8"))
    folds = args.folds or ([0] if args.smoke_test else [0, 1, 2])
    seeds = [int(seed) for seed in config["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("E2-v1 requires exactly the three frozen E1 seeds")
    set_seed(seeds[0])
    for fold_index in folds:
        fold = next(item for item in assignment["folds"] if int(item["fold"]) == fold_index)
        members: list[pd.DataFrame] = []
        member_summaries: list[dict[str, Any]] = []
        for seed in seeds:
            predictions, summary = run_member_fold(config, fold, seed, device)
            members.append(predictions)
            member_summaries.append(summary)
        ensemble = ensemble_predictions(members, int(config["num_classes"]))
        ensemble["representation"] = REPRESENTATION
        ensemble_dir = resolve(config["output_root"]) / "ensemble" / f"fold_{fold_index}"
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        ensemble.to_csv(ensemble_dir / "formal_outer_predictions.csv", index=False, encoding="utf-8-sig")
        metrics = prediction_metrics(ensemble, int(config["num_classes"]))
        fold_summary = {
            "representation": REPRESENTATION,
            "fold": fold_index,
            "member_seeds": seeds,
            "selected_epochs_by_seed": {
                str(item["member_seed"]): int(item["selected_epoch"]) for item in member_summaries
            },
            "outer_validation_users": sorted(fold["validation_user_ids"]),
            "outer_validation_trials": len(ensemble),
            "outer_accuracy": metrics["accuracy"],
            "outer_macro_f1_40class": metrics["macro_f1_40class"],
            "outer_weighted_f1": metrics["weighted_f1"],
            "parameter_count_per_member": int(member_summaries[0]["parameter_count_per_member"]),
            "parameter_count_total": int(sum(item["parameter_count_per_member"] for item in member_summaries)),
            "sampling_strategy": "inverse_sqrt_class_frequency",
            "oof_assignment_sha256": member_summaries[0]["oof_assignment_sha256"],
            "outer_validation_labels_used_for_selection": False,
            "sampling_fit_excludes_scope_validation_users": True,
            "preprocessing_fit_excludes_scope_validation_users": True,
            "ensemble_weights_fitted_on_outer_validation": False,
            "ensemble_rule": "equal_mean_softmax_probability",
        }
        (ensemble_dir / "summary.json").write_text(json.dumps(fold_summary, indent=2) + "\n", encoding="utf-8")
        print(f"ENSEMBLE_FOLD_RESULT={json.dumps(fold_summary)}")
    summarize(config, folds, seeds)


if __name__ == "__main__":
    main()
