from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from p46_complete_repair_model import P46CompleteRepairModel, parameter_count
from p46_event_data import P46EventDataset
from p46_protocol import EXPECTED, HARD_CLASS_TO_INDEX
from p46r_event_bottleneck_model import P46R_OFFSET_FRACTIONS, circular_shift_visual_batch
from train_p46_step10 import (
    atomic_checkpoint,
    atomic_json,
    batch_coverage_report,
    cosine_lr,
    detail_metrics,
    make_class_weights,
    make_loaders,
    move_batch,
    save_evaluation,
    seed_everything,
    source_coverage_report,
    write_csv,
)
from train_p46r_mechanism import select_smoke_ids


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_RUN = PROJECT_DIR / "runs" / "p46_event_inputs_full"
DEFAULT_CONTEXT_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_BASE_CHECKPOINT = (
    PROJECT_DIR / "runs" / "p46_step10_detail21_fullcoverage_v2" / "best_macro_f1.pt"
)
DEFAULT_RELATION_CHECKPOINT = PROJECT_DIR / "runs" / "p46r_full_30_v1" / "best.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_complete_repair_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature-level P46 repair using frozen validated P46-R relationship evidence."
    )
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--context-run", type=Path, default=DEFAULT_CONTEXT_RUN)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--relation-checkpoint", type=Path, default=DEFAULT_RELATION_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--minimum-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--frame-budget", type=int, default=1024)
    parser.add_argument("--eval-frame-budget", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--maximum-delta-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--base-anchor-weight", type=float, default=0.05)
    parser.add_argument("--delta-penalty-weight", type=float, default=1e-4)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def amp_context(device: torch.device, name: str) -> torch.autocast:
    dtype = torch.bfloat16 if name == "bfloat16" else torch.float16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and name != "float32",
    )


def make_datasets(args: argparse.Namespace) -> tuple[P46EventDataset, P46EventDataset]:
    train_full = P46EventDataset(
        args.event_run.resolve(), args.context_run.resolve(), split="train", load_context=True
    )
    val_full = P46EventDataset(
        args.event_run.resolve(), args.context_run.resolve(), split="val", load_context=True
    )
    if not args.smoke:
        return train_full, val_full
    train_ids = select_smoke_ids(train_full.rows, per_class=3)
    val_ids = select_smoke_ids(val_full.rows, per_class=2)
    return (
        P46EventDataset(
            args.event_run,
            args.context_run,
            split="train",
            sample_ids=train_ids,
            load_context=True,
        ),
        P46EventDataset(
            args.event_run,
            args.context_run,
            split="val",
            sample_ids=val_ids,
            load_context=True,
        ),
    )


def make_model(args: argparse.Namespace, subjects: int) -> tuple[P46CompleteRepairModel, dict[str, Any]]:
    model = P46CompleteRepairModel(
        subjects=subjects,
        dropout=0.12,
        adapter_rank=args.adapter_rank,
        maximum_delta_norm=args.maximum_delta_norm,
    )
    metadata = model.load_pretrained(args.base_checkpoint, args.relation_checkpoint)
    model.freeze_pretrained()
    return model, metadata


def make_optimizer(model: P46CompleteRepairModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay = [
        parameter
        for parameter in model.relationship_parameters()
        if parameter.ndim >= 2
    ]
    no_decay = [
        parameter
        for parameter in model.relationship_parameters()
        if parameter.ndim < 2
    ]
    return torch.optim.AdamW(
        (
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=args.learning_rate,
    )


@torch.inference_mode()
def evaluate(
    model: P46CompleteRepairModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    *,
    relationship_scale: float = 1.0,
    relationship_offset_index: int | None = None,
) -> dict[str, Any]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    base_logits: list[torch.Tensor] = []
    source_ids: list[str] = []
    users: list[str] = []
    delta_norm = 0.0
    samples = 0
    started = time.perf_counter()
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        relationship_batch = None
        if relationship_offset_index is not None:
            offset = torch.full(
                (len(batch["detail_index"]),),
                relationship_offset_index,
                device=device,
                dtype=torch.long,
            )
            relationship_batch, _ = circular_shift_visual_batch(batch, offset)
        with amp_context(device, amp_dtype):
            output = model(
                batch,
                relationship_scale=relationship_scale,
                relationship_batch=relationship_batch,
            )
        batch_size = len(batch["detail_index"])
        labels.append(batch["detail_index"].cpu())
        logits.append(output["detail_logits"].float().cpu())
        base_logits.append(output["base_detail_logits"].float().cpu())
        delta_norm += float(output["relationship_delta"].float().norm(dim=1).sum())
        samples += batch_size
        source_ids.extend(batch["source_id"])
        users.extend(batch["user_id"])
    label_tensor = torch.cat(labels)
    logit_tensor = torch.cat(logits)
    base_logit_tensor = torch.cat(base_logits)
    label_array = label_tensor.numpy()
    logit_array = logit_tensor.numpy()
    base_logit_array = base_logit_tensor.numpy()
    return {
        "loss": float(F.cross_entropy(logit_tensor, label_tensor)),
        "metrics": detail_metrics(label_array, logit_array),
        "base_metrics": detail_metrics(label_array, base_logit_array),
        "labels": label_array,
        "logits": logit_array,
        "base_logits": base_logit_array,
        "source_ids": source_ids,
        "users": users,
        "samples": samples,
        "mean_relationship_delta_norm": delta_norm / max(samples, 1),
        "seconds": time.perf_counter() - started,
    }


def train_epoch(
    model: P46CompleteRepairModel,
    loader: DataLoader,
    sampler: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    class_weights: torch.Tensor,
) -> dict[str, Any]:
    model.train()
    sampler.set_epoch(3000 + epoch)
    totals: Counter[str] = Counter()
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    observed: list[str] = []
    samples = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, raw_batch in enumerate(loader):
        observed.extend(raw_batch["source_id"])
        batch = move_batch(raw_batch, device)
        labels = batch["detail_index"]
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.amp_dtype):
            output = model(batch)
            classification = F.cross_entropy(
                output["detail_logits"],
                labels,
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
            anchor = F.kl_div(
                F.log_softmax(output["detail_logits"].float(), dim=1),
                F.softmax(output["base_detail_logits"].detach().float(), dim=1),
                reduction="batchmean",
            )
            delta = output["relationship_delta"].float().square().mean()
            total = (
                classification
                + args.base_anchor_weight * anchor
                + args.delta_penalty_weight * delta
            )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite complete-repair loss at epoch {epoch}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.relationship_parameters(), 2.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite complete-repair gradient at epoch {epoch}")
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(labels)
        for key, value in {
            "total": total,
            "classification": classification,
            "base_anchor": anchor,
            "delta_penalty": delta,
            "gradient_norm": gradient,
        }.items():
            totals[key] += float(value.detach()) * batch_size
        labels_all.append(labels.detach().cpu())
        logits_all.append(output["detail_logits"].detach().float().cpu())
        samples += batch_size
        if args.log_every and (batch_index + 1) % args.log_every == 0:
            print(
                f"repair epoch={epoch} batch={batch_index+1}/{len(loader)} "
                f"samples={samples} loss={totals['total']/samples:.4f}",
                flush=True,
            )
    coverage = source_coverage_report(observed, sampler.source_ids)
    if not coverage["exact_once"]:
        raise RuntimeError(f"complete-repair epoch coverage failed: {coverage}")
    labels_array = torch.cat(labels_all).numpy()
    logits_array = torch.cat(logits_all).numpy()
    return {
        "losses": {key: value / samples for key, value in totals.items()},
        "metrics": detail_metrics(labels_array, logits_array),
        "coverage": coverage,
        "seconds": time.perf_counter() - started,
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }


def preflight(
    model: P46CompleteRepairModel,
    train: P46EventDataset,
    val: P46EventDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sampler: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    epoch_reports = []
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(3000 + epoch)
        report = sampler.coverage_report()
        if not report["exact_once"]:
            raise RuntimeError(f"complete-repair preflight coverage failed: {report}")
        epoch_reports.append(report)
    raw_batch = next(iter(train_loader))
    batch = move_batch(raw_batch, device)
    model.eval()
    with amp_context(device, args.amp_dtype):
        output = model(batch)
    maximum_logit_difference = float(
        (output["detail_logits"] - output["base_detail_logits"]).detach().abs().max()
    )
    maximum_delta = float(output["relationship_delta"].detach().abs().max())
    if maximum_logit_difference != 0.0 or maximum_delta != 0.0:
        raise RuntimeError(
            "zero-initialised repair does not preserve original P46: "
            f"logit={maximum_logit_difference} delta={maximum_delta}"
        )
    model.train()
    for parameter in model.relationship_parameters():
        parameter.grad = None
    with amp_context(device, args.amp_dtype):
        trained = model(batch)
        loss = F.cross_entropy(trained["detail_logits"], batch["detail_index"])
    loss.backward()
    adapter_gradients = [
        parameter.grad for parameter in model.relationship_parameters() if parameter.grad is not None
    ]
    adapter_gradient_finite = bool(adapter_gradients) and all(
        torch.isfinite(value).all() for value in adapter_gradients
    )
    adapter_gradient_nonzero = any(float(value.abs().sum()) > 0.0 for value in adapter_gradients)
    frozen_gradients_absent = all(
        parameter.grad is None
        for module in (model.base, model.relation)
        for parameter in module.parameters()
    )
    model.zero_grad(set_to_none=True)
    val_batches = list(val_loader.batch_sampler)
    validation_coverage = batch_coverage_report(val_batches, len(val))
    return {
        "stage": "P46_complete_repair_preflight",
        "train_trials": len(train),
        "val_trials": len(val),
        "subject_overlap": sorted(
            {row["user_id"] for row in train.rows}
            & {row["user_id"] for row in val.rows}
        ),
        "sampler_epochs": epoch_reports,
        "validation_coverage": validation_coverage,
        "zero_initialisation": {
            "maximum_logit_difference_vs_p46": maximum_logit_difference,
            "maximum_relationship_delta": maximum_delta,
        },
        "adapter_gradient_finite": adapter_gradient_finite,
        "adapter_gradient_nonzero": adapter_gradient_nonzero,
        "frozen_pretrained_gradients_absent": frozen_gradients_absent,
        "forward_finite": bool(torch.isfinite(output["detail_logits"]).all()),
        "relationship_input_finite": bool(torch.isfinite(output["relationship_input"]).all()),
    }


def checkpoint_payload(
    model: P46CompleteRepairModel,
    epoch: int,
    evaluation: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "P46_complete_repair_feature_input",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": evaluation["metrics"],
        "base_metrics": evaluation["base_metrics"],
        "config": config,
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.minimum_epochs < 1 or args.patience < 1:
        raise ValueError("epoch and patience settings must be positive")
    if args.minimum_epochs > args.epochs and not args.smoke:
        raise ValueError("minimum-epochs exceeds epochs")
    if os.name == "nt" and args.workers > 0:
        print(f"forcing --workers {args.workers} to 0 on Windows", flush=True)
        args.workers = 0
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.minimum_epochs = 1
        args.patience = 2
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train, val = make_datasets(args)
    if not args.smoke and (
        len(train) != EXPECTED["train_detail_trials"]
        or len(val) != EXPECTED["val_detail_trials"]
    ):
        raise RuntimeError(f"P46 complete-repair split changed: {len(train)}/{len(val)}")
    train_users = sorted({row["user_id"] for row in train.rows})
    val_users = sorted({row["user_id"] for row in val.rows})
    if set(train_users) & set(val_users):
        raise RuntimeError("P46 complete-repair subject leakage")
    train_loader, train_sampler, val_loader, _ = make_loaders(train, val, args)
    # The frozen P46 checkpoint contains a 14-subject adversarial head even though
    # that head is not trained in the repair stage. Preserve its exact shape.
    model, pretrained = make_model(args, 14)
    model.to(device)
    preflight_result = preflight(
        model, train, val, train_loader, val_loader, train_sampler, device, args
    )
    atomic_json(output / "preflight.json", preflight_result)
    if not all(
        (
            not preflight_result["subject_overlap"],
            preflight_result["validation_coverage"]["exact_once"],
            preflight_result["adapter_gradient_finite"],
            preflight_result["adapter_gradient_nonzero"],
            preflight_result["frozen_pretrained_gradients_absent"],
            preflight_result["forward_finite"],
            preflight_result["relationship_input_finite"],
        )
    ):
        raise RuntimeError(f"P46 complete-repair preflight failed: {preflight_result}")
    if args.preflight_only:
        print(json.dumps(preflight_result, ensure_ascii=False, indent=2), flush=True)
        return

    initial = evaluate(model, val_loader, device, args.amp_dtype)
    initial_scale_zero = evaluate(
        model, val_loader, device, args.amp_dtype, relationship_scale=0.0
    )
    initial_difference = float(np.max(np.abs(initial["logits"] - initial["base_logits"])))
    if initial_difference != 0.0:
        raise RuntimeError(f"full-validation initial logits differ from P46 by {initial_difference}")
    if not args.smoke:
        expected_accuracy = float(pretrained["base_metrics"]["accuracy"])
        if abs(initial["metrics"]["accuracy"] - expected_accuracy) > 1.0 / len(val) + 1e-9:
            raise RuntimeError(
                "runtime precision changed the P46 baseline beyond one sample: "
                f"stored={expected_accuracy} runtime={initial['metrics']['accuracy']}"
            )
    save_evaluation(output, "initial_p46_equivalent", initial, epoch=0)

    train_labels = [HARD_CLASS_TO_INDEX[int(row["class_id"])] for row in train.rows]
    class_weights, class_counts = make_class_weights(
        train_labels, len(HARD_CLASS_TO_INDEX), args.class_weight_power
    )
    class_weights = class_weights.to(device)
    config = {
        "stage": "P46_complete_repair_feature_input",
        "architecture": (
            "P46 trial embedding + zero-initialised adapter(P46-R ordered relation embedding, "
            "five-part lag correlation, event centres) before original P46 Detail21 head"
        ),
        "output_gating_used": False,
        "pretrained": pretrained,
        "train_trials": len(train),
        "val_trials": len(val),
        "train_subjects": train_users,
        "val_subjects": val_users,
        "epochs": args.epochs,
        "minimum_epochs": args.minimum_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "weight_decay": args.weight_decay,
        "adapter_rank": args.adapter_rank,
        "maximum_delta_norm": args.maximum_delta_norm,
        "label_smoothing": args.label_smoothing,
        "base_anchor_weight": args.base_anchor_weight,
        "delta_penalty_weight": args.delta_penalty_weight,
        "amp_dtype": args.amp_dtype,
        "class_counts": class_counts,
        "model_parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "pretrained_branches_frozen": True,
        "seed": args.seed,
        "smoke": args.smoke,
        "device": str(device),
        "initial_metrics": initial["metrics"],
        "initial_scale_zero_metrics": initial_scale_zero["metrics"],
    }
    atomic_json(output / "frozen_config.json", config)

    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.amp_dtype == "float16",
    )
    history: list[dict[str, Any]] = []
    best_accuracy = initial["metrics"]["accuracy"]
    best_macro_f1 = initial["metrics"]["macro_f1"]
    best_score = best_macro_f1 + 0.25 * best_accuracy
    best_epoch = 0
    best_accuracy_epoch = 0
    best_macro_epoch = 0
    initial_payload = checkpoint_payload(model, 0, initial, config)
    atomic_checkpoint(output / "best.pt", initial_payload)
    atomic_checkpoint(output / "best_accuracy.pt", initial_payload)
    atomic_checkpoint(output / "best_macro_f1.pt", initial_payload)
    stale = 0
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        rate = cosine_lr(
            epoch,
            args.epochs,
            args.learning_rate,
            args.minimum_learning_rate,
            warmup=1,
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        training = train_epoch(
            model,
            train_loader,
            train_sampler,
            optimizer,
            scaler,
            device,
            args,
            epoch,
            class_weights,
        )
        validation = evaluate(model, val_loader, device, args.amp_dtype)
        accuracy = validation["metrics"]["accuracy"]
        macro_f1 = validation["metrics"]["macro_f1"]
        score = macro_f1 + 0.25 * accuracy
        accuracy_improved = accuracy > best_accuracy + args.min_delta
        macro_improved = macro_f1 > best_macro_f1 + args.min_delta
        score_improved = score > best_score + args.min_delta
        payload = checkpoint_payload(model, epoch, validation, config)
        if score_improved:
            best_score = score
            best_epoch = epoch
            atomic_checkpoint(output / "best.pt", payload)
            save_evaluation(output, "best", validation, epoch)
        if accuracy_improved:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch
            atomic_checkpoint(output / "best_accuracy.pt", payload)
            save_evaluation(output, "best_accuracy", validation, epoch)
        if macro_improved:
            best_macro_f1 = macro_f1
            best_macro_epoch = epoch
            atomic_checkpoint(output / "best_macro_f1.pt", payload)
            save_evaluation(output, "best_macro_f1", validation, epoch)
        stale = 0 if (accuracy_improved or macro_improved) else stale + 1
        row = {
            "epoch": epoch,
            "learning_rate": rate,
            **{f"train_{key}": value for key, value in training["losses"].items()},
            **{f"train_{key}": value for key, value in training["metrics"].items()},
            **{f"val_{key}": value for key, value in validation["metrics"].items()},
            "val_base_accuracy": validation["base_metrics"]["accuracy"],
            "val_mean_relationship_delta_norm": validation["mean_relationship_delta_norm"],
            "train_seconds": training["seconds"],
            "val_seconds": validation["seconds"],
            "peak_cuda_mib": training["peak_cuda_mib"],
            "coverage_exact_once": training["coverage"]["exact_once"],
            "stale_epochs": stale,
        }
        history.append(row)
        write_csv(output / "history.csv", history)
        atomic_checkpoint(
            output / "last.pt",
            {
                **payload,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "best_accuracy_epoch": best_accuracy_epoch,
                "best_macro_f1": best_macro_f1,
                "best_macro_epoch": best_macro_epoch,
                "stale_epochs": stale,
            },
        )
        atomic_json(
            output / "progress.json",
            {
                "stage": "epoch_completed",
                "epoch": epoch,
                "epochs": args.epochs,
                "val_accuracy": accuracy,
                "val_macro_f1": macro_f1,
                "initial_p46_accuracy": initial["metrics"]["accuracy"],
                "stale_epochs": stale,
            },
        )
        print(
            f"repair epoch={epoch}/{args.epochs} train_acc={training['metrics']['accuracy']:.4f} "
            f"val_acc={accuracy:.4f} val_macro={macro_f1:.4f} "
            f"delta_norm={validation['mean_relationship_delta_norm']:.3f} "
            f"stale={stale}/{args.patience}",
            flush=True,
        )
        if epoch >= args.minimum_epochs and stale >= args.patience:
            stopped_early = True
            break

    best_checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model.to(device)
    final = evaluate(model, val_loader, device, args.amp_dtype)
    without_relationship = evaluate(
        model, val_loader, device, args.amp_dtype, relationship_scale=0.0
    )
    shift_results = {}
    for offset_index, fraction in enumerate(P46R_OFFSET_FRACTIONS):
        if fraction == 0.0:
            continue
        shifted = evaluate(
            model,
            val_loader,
            device,
            args.amp_dtype,
            relationship_offset_index=offset_index,
        )
        shift_results[str(fraction)] = {
            "metrics": shifted["metrics"],
            "accuracy_delta_pp": 100.0
            * (shifted["metrics"]["accuracy"] - final["metrics"]["accuracy"]),
            "prediction_changed": int(
                (shifted["logits"].argmax(1) != final["logits"].argmax(1)).sum()
            ),
        }
    audit = {
        "baseline": final["metrics"],
        "without_relationship": without_relationship["metrics"],
        "accuracy_delta_vs_without_relationship_pp": 100.0
        * (final["metrics"]["accuracy"] - without_relationship["metrics"]["accuracy"]),
        "macro_f1_delta_vs_without_relationship_pp": 100.0
        * (final["metrics"]["macro_f1"] - without_relationship["metrics"]["macro_f1"]),
        "prediction_changed_vs_p46": int(
            (final["logits"].argmax(1) != without_relationship["logits"].argmax(1)).sum()
        ),
        "relationship_only_visual_shifts": shift_results,
        "mean_relationship_delta_norm": final["mean_relationship_delta_norm"],
    }
    atomic_json(output / "relationship_audit.json", audit)
    save_evaluation(output, "final_best", final, best_checkpoint["epoch"])
    summary = {
        "stage": "P46_complete_repair_feature_input",
        "best_epoch": int(best_checkpoint["epoch"]),
        "initial_p46_metrics": initial["metrics"],
        "best_metrics": final["metrics"],
        "accuracy_delta_vs_p46_pp": 100.0
        * (final["metrics"]["accuracy"] - initial["metrics"]["accuracy"]),
        "macro_f1_delta_vs_p46_pp": 100.0
        * (final["metrics"]["macro_f1"] - initial["metrics"]["macro_f1"]),
        "above_p46": final["metrics"]["accuracy"] > initial["metrics"]["accuracy"],
        "above_p12": final["metrics"]["accuracy"] > 0.45517241379310347,
        "relationship_audit": audit,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "all_epochs_exact_once": all(bool(row["coverage_exact_once"]) for row in history),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "best_checkpoint": str((output / "best.pt").resolve()),
            "last_checkpoint": str((output / "last.pt").resolve()),
            "history": str((output / "history.csv").resolve()),
            "relationship_audit": str((output / "relationship_audit.json").resolve()),
        },
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
