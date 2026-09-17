from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from p32_fused_data import (
    BalancedCostBucketBatchSampler,
    LengthBucketBatchSampler,
)
from p44c_model import GROUP_CLASS_IDS, P44CPretrainModel, P44CStructuredExpert
from p44c_spatial_fused_data import P44CSpatialFusedDataset, collate_p44c
from train_p44c_pretrain_fold0 import (
    CALIBRATION_SUBJECTS,
    atomic_checkpoint,
    atomic_json,
    make_optimizer,
    move_batch,
    seed_everything,
    supervised_contrastive,
    write_history,
)


PROJECT_DIR = Path(__file__).resolve().parent
INTAKE_CLASSES = {6, 7, 37}
GROUP_LOOKUP = {class_id: index for index, class_id in enumerate(GROUP_CLASS_IDS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P44-C structured Group-A experts and final fold0 audit.")
    parser.add_argument(
        "--fold-csv", type=Path, default=PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv"
    )
    parser.add_argument(
        "--visual-run", type=Path, default=PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
    )
    parser.add_argument(
        "--motion-run", type=Path, default=PROJECT_DIR / "runs" / "p31_skeleton_imu_full"
    )
    parser.add_argument(
        "--spatial-run", type=Path, default=PROJECT_DIR / "runs" / "p44c_spatial_roi_fold0"
    )
    parser.add_argument(
        "--pretrain-run", type=Path, default=PROJECT_DIR / "runs" / "p44c_pretrain_fold0"
    )
    parser.add_argument(
        "--base-logits",
        type=Path,
        default=PROJECT_DIR / "runs" / "p44_p12_base_inner_oof" / "fold0_pilot" / "fold0_base_logits.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "runs" / "p44c_group_experts_fold0"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=45)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=44031)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def selected_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {"fit": [], "calibration": [], "train_all": [], "final_val": []}
    for row in rows:
        if int(row["class_id"]) not in GROUP_CLASS_IDS:
            continue
        row["source_id"] = f"{row['class_name']}/{row['user_id']}/{row['trial_id']}"
        if row["split"] == "train":
            output["train_all"].append(row)
            key = "calibration" if row["user_id"] in CALIBRATION_SUBJECTS else "fit"
            output[key].append(row)
        elif row["split"] == "val":
            output["final_val"].append(row)
    return output


def dataset(args: argparse.Namespace, rows: list[dict[str, str]]) -> P44CSpatialFusedDataset:
    return P44CSpatialFusedDataset(
        args.visual_run.resolve(),
        args.motion_run.resolve(),
        args.spatial_run.resolve(),
        {row["source_id"] for row in rows},
    )


def loader(
    source: P44CSpatialFusedDataset,
    batch_size: int,
    workers: int,
    seed: int,
    train: bool,
) -> tuple[DataLoader, Any]:
    if train:
        sampler = BalancedCostBucketBatchSampler(
            source.frame_lengths,
            source.imu_point_lengths,
            [GROUP_LOOKUP[int(row["class_id"])] for row in source.rows],
            maximum_batch_size=batch_size,
            seed=seed,
            bucket_multiplier=10,
        )
    else:
        sampler = LengthBucketBatchSampler(
            source.frame_lengths, batch_size, shuffle=False, seed=seed, bucket_multiplier=10
        )
    return (
        DataLoader(
            source,
            batch_sampler=sampler,
            collate_fn=collate_p44c,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        ),
        sampler,
    )


def load_pretrained_encoder(path: Path, device: torch.device) -> tuple[P44CStructuredExpert, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    pretrained = P44CPretrainModel()
    pretrained.load_state_dict(checkpoint["model_state_dict"])
    model = P44CStructuredExpert(pretrained.encoder)
    del pretrained.classifier, pretrained.multimodal_aux, pretrained.spatial_aux
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return model.to(device), checkpoint


def mapped_labels(labels: torch.Tensor) -> torch.Tensor:
    lookup = torch.full((40,), -1, dtype=torch.long, device=labels.device)
    for index, class_id in enumerate(GROUP_CLASS_IDS):
        lookup[class_id] = index
    return lookup[labels]


def expert_losses(
    output: dict[str, torch.Tensor], labels: torch.Tensor, users: list[str]
) -> dict[str, torch.Tensor]:
    targets = mapped_labels(labels)
    logits = output["group_logits"]
    ce = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=0.03)
    subgroup_target = torch.tensor(
        [0 if int(value) in INTAKE_CLASSES else 1 for value in labels.tolist()],
        dtype=torch.long,
        device=labels.device,
    )
    subgroup = torch.nn.functional.cross_entropy(output["subgroup_logits"], subgroup_target)
    true_score = logits.gather(1, targets[:, None]).squeeze(1)
    wrong = logits.masked_fill(
        torch.nn.functional.one_hot(targets, len(GROUP_CLASS_IDS)).bool(), -1e4
    ).amax(dim=1)
    ranking = torch.nn.functional.softplus((wrong - true_score) / 0.35).mean()
    contrast = supervised_contrastive(output["expert_embedding"], targets, users, 0.12)
    return {
        "total": ce + 0.20 * subgroup + 0.35 * ranking + 0.08 * contrast,
        "ce": ce,
        "subgroup": subgroup,
        "ranking": ranking,
        "contrast": contrast,
    }


def metric_dict(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(GROUP_CLASS_IDS)[logits.argmax(1)]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=GROUP_CLASS_IDS).tolist(),
    }


def perturb_spatial(batch: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "none":
        return batch
    output = dict(batch)
    for key in ("spatial_features", "spatial_valid", "spatial_quality", "spatial_clipped_ratio"):
        output[key] = batch[key].clone()
    if mode == "zero":
        output["spatial_features"].zero_()
        output["spatial_valid"].zero_()
        output["spatial_quality"].zero_()
    elif mode == "shuffle":
        permutation = torch.arange(len(batch["label"]) - 1, -1, -1, device=batch["label"].device)
        for key in ("spatial_features", "spatial_valid", "spatial_quality", "spatial_clipped_ratio"):
            output[key] = output[key][permutation]
    else:
        raise ValueError(mode)
    return output


def run_epoch(
    model: P44CStructuredExpert,
    source_loader: DataLoader,
    sampler: Any,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    perturbation: str = "none",
    maximum_batches: int = 0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    model.encoder.eval()  # frozen pretrained features must remain deterministic
    if training:
        sampler.set_epoch(epoch)
    totals = {key: 0.0 for key in ("total", "ce", "subgroup", "ranking", "contrast")}
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    users: list[str] = []
    qualities: list[np.ndarray] = []
    begun = time.perf_counter()
    for batch_index, batch in enumerate(source_loader):
        if maximum_batches and batch_index >= maximum_batches:
            break
        batch = perturb_spatial(move_batch(batch, device), perturbation)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(batch)
            batch_losses = expert_losses(output, batch["label"], batch["user_id"])
        if training:
            assert scaler is not None
            scaler.scale(batch_losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 2.0
            )
            scaler.step(optimizer)
            scaler.update()
        size = len(batch["label"])
        for key in totals:
            totals[key] += float(batch_losses[key].detach()) * size
        labels_all.append(batch["label"].detach().cpu().numpy())
        logits_all.append(output["group_logits"].detach().float().cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        users.extend(batch["user_id"])
        qualities.append(output["mean_roi_quality"].detach().float().cpu().numpy())
    labels = np.concatenate(labels_all)
    logits = np.concatenate(logits_all)
    return {
        "losses": {key: value / len(labels) for key, value in totals.items()},
        "metrics": metric_dict(labels, logits),
        "labels": labels,
        "logits": logits,
        "sample_ids": np.asarray(sample_ids),
        "users": np.asarray(users),
        "qualities": np.concatenate(qualities),
        "seconds": time.perf_counter() - begun,
        "perturbation": perturbation,
    }


def lr_for_epoch(epoch: int, epochs: int, maximum: float, minimum: float) -> float:
    if epoch <= 2:
        return maximum * epoch / 2
    progress = (epoch - 2) / max(epochs - 2, 1)
    return minimum + 0.5 * (maximum - minimum) * (1 + math.cos(math.pi * progress))


def train_select(
    args: argparse.Namespace,
    fit: P44CSpatialFusedDataset,
    calibration: P44CSpatialFusedDataset,
    output: Path,
    device: torch.device,
) -> int:
    model, pretrain = load_pretrained_encoder(
        args.pretrain_run.resolve() / "best_subject_calibration.pt", device
    )
    fit_loader, fit_sampler = loader(fit, args.batch_size, args.workers, args.seed, True)
    cal_loader, cal_sampler = loader(
        calibration, args.eval_batch_size, args.workers, args.seed, False
    )
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    maximum = 2 if args.smoke else args.max_epochs
    for epoch in range(1, maximum + 1):
        lr = lr_for_epoch(epoch, maximum, args.learning_rate, args.minimum_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train = run_epoch(
            model, fit_loader, fit_sampler, device, epoch, optimizer, scaler,
            maximum_batches=2 if args.smoke else 0,
        )
        calibration_result = run_epoch(
            model, cal_loader, cal_sampler, device, epoch, None, None,
            maximum_batches=2 if args.smoke else 0,
        )
        score = calibration_result["metrics"]["macro_f1"]
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            stale = 0
            atomic_checkpoint(
                output / "best_subject_calibration.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "calibration_metrics": {
                        key: value
                        for key, value in calibration_result["metrics"].items()
                        if key != "predictions"
                    },
                    "pretrain_epoch": int(pretrain["epoch"]),
                },
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train["losses"]["total"],
            "train_accuracy": train["metrics"]["accuracy"],
            "calibration_accuracy": calibration_result["metrics"]["accuracy"],
            "calibration_macro_f1": score,
            "best_epoch": best_epoch,
            "stale": stale,
            "train_seconds": train["seconds"],
            "calibration_seconds": calibration_result["seconds"],
        }
        history.append(row)
        write_history(output / "selection_history.csv", history)
        print(json.dumps({"stage": "expert_select", **row}, ensure_ascii=False), flush=True)
        if not args.smoke and epoch >= args.min_epochs and stale >= args.patience:
            break
    return best_epoch


def refit(
    args: argparse.Namespace,
    all_train: P44CSpatialFusedDataset,
    epochs: int,
    output: Path,
    device: torch.device,
) -> P44CStructuredExpert:
    model, pretrain = load_pretrained_encoder(
        args.pretrain_run.resolve() / "refit_all_fold0_train.pt", device
    )
    source_loader, sampler = loader(
        all_train, args.batch_size, args.workers, args.seed + 101, True
    )
    seed_everything(args.seed + 101)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    epochs = 1 if args.smoke else max(1, epochs)
    for epoch in range(1, epochs + 1):
        lr = lr_for_epoch(epoch, epochs, args.learning_rate, args.minimum_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        result = run_epoch(
            model, source_loader, sampler, device, epoch, optimizer, scaler,
            maximum_batches=2 if args.smoke else 0,
        )
        row = {
            "epoch": epoch,
            "lr": lr,
            "loss": result["losses"]["total"],
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "seconds": result["seconds"],
        }
        history.append(row)
        write_history(output / "refit_history.csv", history)
        print(json.dumps({"stage": "expert_refit", **row}, ensure_ascii=False), flush=True)
    atomic_checkpoint(
        output / "refit_all_group_train.pt",
        {
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "pretrain_epochs": int(pretrain["epochs"]),
            "outer_held_predictions_generated": False,
        },
    )
    return model


def restricted_base(
    path: Path, final_rows: list[dict[str, str]], sample_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if bool(data["outer_held_predictions_generated"].item()):
            raise RuntimeError("Base artifact contains outer-held predictions")
        base_by_id = {
            str(sample_id): logits.astype(np.float32)
            for sample_id, logits in zip(data["sample_ids"], data["base_logits"])
        }
    manifest_by_source = {row["source_id"]: row["sample_id"] for row in final_rows}
    logits = np.stack([base_by_id[manifest_by_source[str(source_id)]] for source_id in sample_ids])
    return logits[:, list(GROUP_CLASS_IDS)], logits


def fixed_safe_residual(
    base_group: np.ndarray,
    detail: np.ndarray,
    roi_quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    def log_softmax(value: np.ndarray) -> np.ndarray:
        shifted = value - value.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    base_logp = log_softmax(base_group)
    detail_logp = log_softmax(detail)
    probability = np.exp(base_logp)
    ordered = np.sort(probability, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    uncertainty = 1.0 / (1.0 + np.exp((margin - 0.22) / 0.08))
    quality = np.clip((roi_quality - 0.45) / 0.35, 0.0, 1.0)
    gate = quality * uncertainty
    centered = detail_logp - detail_logp.mean(axis=1, keepdims=True)
    combined = base_logp + 0.60 * gate[:, None] * np.clip(centered, -2.0, 2.0)
    return combined, gate


def save_evaluation(path: Path, result: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        sample_ids=result["sample_ids"],
        users=result["users"],
        labels=result["labels"],
        logits=result["logits"].astype(np.float16),
        predictions=result["metrics"]["predictions"],
        roi_quality=result["qualities"].astype(np.float16),
        perturbation=np.asarray(result["perturbation"]),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(args.fold_csv.resolve())
    expected = {"fit": 287, "calibration": 71, "train_all": 358, "final_val": 201}
    actual = {key: len(value) for key, value in rows.items()}
    if actual != expected:
        raise RuntimeError(f"Group-A partition mismatch: {actual}")
    fit = dataset(args, rows["fit"])
    calibration = dataset(args, rows["calibration"])
    all_train = dataset(args, rows["train_all"])
    final_val = dataset(args, rows["final_val"])
    device = torch.device(args.device)
    config = {
        "protocol": "p44c-group-a-structured-experts-fold0-v1",
        "status": "exploratory because Group-A was identified from this fold0 validation",
        "class_ids": list(GROUP_CLASS_IDS),
        "intake_expert": [6, 7, 37],
        "operation_expert": [8, 9, 10, 11, 14],
        "selection_subjects": sorted(CALIBRATION_SUBJECTS),
        "counts": actual,
        "encoder_policy": "40-class pretrained; frozen for specialist to control overfit",
        "loss": "8-way CE + 0.20 subgroup CE + 0.35 hardest-rival ranking + 0.08 cross-subject SupCon",
        "primary_safe_residual": "Base logp + 0.60 * ROI-quality * Base-uncertainty * clipped centered Detail logp",
        "final_val_used_for_training_or_epoch_selection": False,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    atomic_json(output / "config.json", config)
    print(json.dumps({"stage": "start", **config}, ensure_ascii=False), flush=True)
    best_epoch = train_select(args, fit, calibration, output, device)
    model = refit(args, all_train, best_epoch, output, device)
    final_loader, final_sampler = loader(
        final_val, args.eval_batch_size, args.workers, args.seed, False
    )
    normal = run_epoch(model, final_loader, final_sampler, device, 0, None, None)
    shuffled = run_epoch(
        model, final_loader, final_sampler, device, 0, None, None, perturbation="shuffle"
    )
    zeroed = run_epoch(
        model, final_loader, final_sampler, device, 0, None, None, perturbation="zero"
    )
    save_evaluation(output / "final_detail.npz", normal)
    save_evaluation(output / "final_spatial_shuffle.npz", shuffled)
    save_evaluation(output / "final_spatial_zero.npz", zeroed)
    base_group, _ = restricted_base(args.base_logits.resolve(), rows["final_val"], normal["sample_ids"])
    base_metrics = metric_dict(normal["labels"], base_group)
    combined_logits, gates = fixed_safe_residual(
        base_group, normal["logits"], normal["qualities"]
    )
    combined_metrics = metric_dict(normal["labels"], combined_logits)
    np.savez_compressed(
        output / "final_base_detail_residual.npz",
        sample_ids=normal["sample_ids"],
        labels=normal["labels"],
        base_group_logits=base_group.astype(np.float16),
        detail_logits=normal["logits"].astype(np.float16),
        gates=gates.astype(np.float16),
        combined_logits=combined_logits.astype(np.float16),
        predictions=combined_metrics["predictions"],
    )
    both_correct = int(
        ((base_metrics["predictions"] == normal["labels"]) & (normal["metrics"]["predictions"] == normal["labels"])).sum()
    )
    rescues = int(
        ((base_metrics["predictions"] != normal["labels"]) & (normal["metrics"]["predictions"] == normal["labels"])).sum()
    )
    new_errors = int(
        ((base_metrics["predictions"] == normal["labels"]) & (normal["metrics"]["predictions"] != normal["labels"])).sum()
    )
    summary = {
        **config,
        "selected_specialist_epoch": best_epoch,
        "final_metrics": {
            "base_restricted": {key: value for key, value in base_metrics.items() if key != "predictions"},
            "detail_standalone": {key: value for key, value in normal["metrics"].items() if key != "predictions"},
            "base_plus_fixed_safe_residual": {key: value for key, value in combined_metrics.items() if key != "predictions"},
            "spatial_shuffle": {key: value for key, value in shuffled["metrics"].items() if key != "predictions"},
            "spatial_zero": {key: value for key, value in zeroed["metrics"].items() if key != "predictions"},
        },
        "detail_vs_base_counts": {
            "both_correct": both_correct,
            "rescues": rescues,
            "new_errors": new_errors,
        },
        "gate": {
            "mean": float(gates.mean()),
            "median": float(np.median(gates)),
            "maximum": float(gates.max()),
        },
        "outer_held_predictions_generated": False,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
