"""Train P101-F0 with fixed outer OOF and nested subject-OOF supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from p100a_global_teacher_data import FOLD_USERS
from p101_finegrained_teacher_data import (
    CANONICAL_VARIANTS,
    DEV_USER_SET,
    P101Data,
    P101Dataset,
    class_user_sample_weights,
    load_p101_data,
    within_subject_permutation,
    within_subject_wrong_label_source,
)
from p101_finegrained_teacher_model import (
    P101FineGrainedTeacher,
    P101ModelConfig,
    trainable_imu_adapter_parameters,
)
from train_p100a_global_teacher_oof import (
    classification_metrics,
    confusion_change_groups,
    paired_comparison,
    softmax_numpy,
    topk_accuracy,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p101_f0_finegrained_teacher.json"
DEFAULT_OUTPUT = HERE / "runs/p101_f0_finegrained_teacher_oof_v1"
COARSE_VS = HERE / "runs/p100a_a0_global_teacher_oof_v1/VS_complete_oof.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stages", default="base,nested,vsi")
    parser.add_argument("--variants", default="V,VS,VI")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_loader(
    dataset: P101Dataset, batch_size: int, shuffle: bool, seed: int
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return max((step + 1) / max(warmup, 1), 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def model_config(config: dict[str, Any], variant: str) -> P101ModelConfig:
    return P101ModelConfig(
        modalities=CANONICAL_VARIANTS[variant], **config["model"]
    )


def build_causal_base_model(
    config: dict[str, Any], variant: str, seed: int, device: torch.device
) -> P101FineGrainedTeacher:
    """Give V/VS/VI the same visual/global/classifier initialization per fold."""
    set_seed(seed)
    visual = P101FineGrainedTeacher(model_config(config, "V"))
    if variant == "V":
        model = visual
    else:
        model = P101FineGrainedTeacher(model_config(config, variant))
        incompatible = model.load_state_dict(visual.state_dict(), strict=False)
        if incompatible.unexpected_keys or not incompatible.missing_keys:
            raise RuntimeError(
                f"P101 causal Visual initialization failed for {variant}: {incompatible}"
            )
    # Re-align loader/dropout randomness after modality-specific construction.
    set_seed(seed)
    return model.to(device)


def train_base(
    model: P101FineGrainedTeacher,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    config: dict[str, Any],
    epochs: int,
) -> list[dict[str, float]]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.98),
    )
    total = max(epochs * len(loader), 1)
    warmup = int(total * float(training["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(step, total, warmup)
    )
    use_amp = device.type == "cuda" and bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        correct = rows = 0
        started = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch)["logits"]
                losses = F.cross_entropy(
                    logits,
                    batch["label"],
                    reduction="none",
                    label_smoothing=float(training["label_smoothing"]),
                )
                loss = (losses * batch["weight"]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(batch["label"])
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            rows += len(batch["label"])
        record = {
            "epoch": float(epoch + 1),
            "loss": loss_sum / max(rows, 1),
            "accuracy": correct / max(rows, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    return history


def set_adapter_mode(model: P101FineGrainedTeacher) -> None:
    model.eval()
    for module in (
        model.imu_encoder,
        model.imu_projection,
        model.imu_attention,
        model.imu_delta,
        model.imu_reliability,
        model.correspondence,
    ):
        if module is not None:
            module.train()


def train_vsi_adapter(
    candidate: P101FineGrainedTeacher,
    anchor: P101FineGrainedTeacher,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    config: dict[str, Any],
    epochs: int,
) -> tuple[list[dict[str, float]], list[str], list[str]]:
    training = config["training"]
    parameters, trainable, frozen = trainable_imu_adapter_parameters(candidate)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["adapter_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.98),
    )
    total = max(epochs * len(loader), 1)
    warmup = int(total * float(training["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(step, total, warmup)
    )
    use_amp = device.type == "cuda" and bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    temperature = float(training["anchor_kl_temperature"])
    anchor.eval()
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        set_adapter_mode(candidate)
        sums = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "pair": 0.0}
        correct = rows = 0
        started = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                anchor_logits = anchor(batch)["logits"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = candidate(batch)
                logits = output["logits"]
                error = batch["nested_vs_error"]
                ce_weight = batch["weight"] * (
                    1.0 + float(training["nested_error_ce_multiplier"]) * error
                )
                ce = (
                    F.cross_entropy(
                        logits,
                        batch["label"],
                        reduction="none",
                        label_smoothing=float(training["label_smoothing"]),
                    )
                    * ce_weight
                ).mean()
                anchor_probability = F.softmax(anchor_logits / temperature, dim=-1)
                kl_rows = F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    anchor_probability,
                    reduction="none",
                ).sum(dim=-1) * temperature**2
                kl_weight = torch.where(
                    error > 0.5,
                    torch.full_like(error, float(training["anchor_kl_wrong_weight"])),
                    torch.full_like(error, float(training["anchor_kl_correct_weight"])),
                )
                kl = (kl_rows * kl_weight).mean()
                positive = output["positive_correspondence_logits"]
                negative = output["negative_correspondence_logits"]
                pair = 0.5 * (
                    F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
                    + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
                )
                loss = ce + kl + float(training["correspondence_weight"]) * pair
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            size = len(batch["label"])
            for key, value in (("loss", loss), ("ce", ce), ("kl", kl), ("pair", pair)):
                sums[key] += float(value.detach()) * size
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            rows += size
        record = {
            "epoch": float(epoch + 1),
            **{key: value / max(rows, 1) for key, value in sums.items()},
            "accuracy": correct / max(rows, 1),
            "imu_scale": float(
                torch.sigmoid(candidate.imu_scale_logit).detach()
                * candidate.config.maximum_imu_scale
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    return history, trainable, frozen


def attention_audit(
    attention: torch.Tensor,
    visual_time: torch.Tensor,
    motion_time: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # attention [B,W,V,T,S,P]
    distribution = attention.sum(dim=-1)
    delta = motion_time[:, :, None, None, :] - visual_time[:, :, None, :, None]
    signed = (distribution * delta).sum(dim=-1).mean(dim=(1, 2, 3))
    absolute = (distribution * delta.abs()).sum(dim=-1).mean(dim=(1, 2, 3))
    entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=(-2, -1))
    entropy = entropy.mean(dim=(1, 2, 3))
    return (
        signed.float().cpu().numpy(),
        absolute.float().cpu().numpy(),
        entropy.float().cpu().numpy(),
    )


@torch.no_grad()
def evaluate(
    model: P101FineGrainedTeacher,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {"rows": [], "logits": []}
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        collected["rows"].append(batch["row"].cpu().numpy())
        collected["logits"].append(output["logits"].float().cpu().numpy())
        for modality in ("skeleton", "imu"):
            reliability = output.get(f"{modality}_reliability")
            attention = output.get(f"{modality}_attention")
            if reliability is not None:
                collected.setdefault(f"{modality}_reliability", []).append(
                    reliability.mean(dim=(1, 2, 3, 4)).float().cpu().numpy()
                )
            if attention is not None:
                signed, absolute, entropy = attention_audit(
                    attention, batch["visual_time"], batch["motion_time"]
                )
                collected.setdefault(f"{modality}_signed_offset", []).append(signed)
                collected.setdefault(f"{modality}_absolute_offset", []).append(absolute)
                collected.setdefault(f"{modality}_attention_entropy", []).append(entropy)
        if "positive_correspondence_logits" in output:
            collected.setdefault("positive_correspondence", []).append(
                output["positive_correspondence_logits"]
                .mean(dim=(1, 2, 3, 4))
                .float()
                .cpu()
                .numpy()
            )
        if "imu_residual_rms" in output:
            size = len(batch["row"])
            collected.setdefault("imu_residual_rms", []).append(
                np.full(size, float(output["imu_residual_rms"].cpu()), dtype=np.float32)
            )
            collected.setdefault("imu_scale", []).append(
                np.full(size, float(output["imu_scale"].cpu()), dtype=np.float32)
            )
    return {key: np.concatenate(values) for key, values in collected.items()}


def counterfactual_dataset(
    data: P101Data,
    indices: np.ndarray,
    modalities: tuple[str, ...],
    kind: str,
    seed: int,
) -> P101Dataset:
    zero: tuple[str, ...] = ()
    reverse: tuple[str, ...] = ()
    skeleton_source = imu_source = None
    if kind == "zero_skeleton":
        zero = ("skeleton",)
    elif kind == "reverse_skeleton":
        reverse = ("skeleton",)
    elif kind == "shuffle_skeleton":
        skeleton_source = within_subject_permutation(data, indices, "skeleton", seed)
    elif kind == "zero_imu":
        zero = ("imu",)
    elif kind == "reverse_imu":
        reverse = ("imu",)
    elif kind == "shuffle_imu":
        imu_source = within_subject_permutation(data, indices, "imu", seed)
    elif kind == "zero_both":
        zero = ("skeleton", "imu")
    else:
        raise ValueError(kind)
    return P101Dataset(
        data,
        indices,
        modalities,
        skeleton_source=skeleton_source,
        imu_source=imu_source,
        zero_modalities=zero,
        reverse_modalities=reverse,
    )


def evaluate_fold_with_counterfactuals(
    model: P101FineGrainedTeacher,
    data: P101Data,
    held: np.ndarray,
    variant: str,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    batch_size = int(config["training"]["eval_batch_size"])
    modalities = CANONICAL_VARIANTS[variant]
    direct = evaluate(
        model,
        make_loader(P101Dataset(data, held, modalities), batch_size, False, seed),
        device,
    )
    arrays = {"rows": direct.pop("rows"), "direct_logits": direct.pop("logits")}
    arrays.update({f"direct_{key}": value for key, value in direct.items()})
    kinds: list[str] = []
    if "skeleton" in modalities:
        kinds += ["zero_skeleton", "reverse_skeleton", "shuffle_skeleton"]
    if "imu" in modalities:
        kinds += ["zero_imu", "reverse_imu", "shuffle_imu"]
    if {"skeleton", "imu"}.issubset(modalities):
        kinds.append("zero_both")
    for number, kind in enumerate(kinds):
        result = evaluate(
            model,
            make_loader(
                counterfactual_dataset(data, held, modalities, kind, seed + 1000 + number),
                batch_size,
                False,
                seed,
            ),
            device,
        )
        if not np.array_equal(result["rows"], arrays["rows"]):
            raise RuntimeError("P101 counterfactual row order changed")
        arrays[f"{kind}_logits"] = result["logits"]
    return arrays


def save_checkpoint(
    path: Path,
    model: P101FineGrainedTeacher,
    history: list[dict[str, float]],
    held_users: tuple[str, ...],
    config: P101ModelConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_config": config.__dict__,
        "history": history,
        "held_users": held_users,
        "outer_label_used_for_selection": False,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def run_base_fold(
    data: P101Data,
    variant: str,
    fold: int,
    config: dict[str, Any],
    output: Path,
    epochs: int,
    device: torch.device,
    resume: bool,
) -> dict[str, np.ndarray]:
    fold_output = output / variant
    fold_output.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_output / f"fold{fold}_final.pt"
    predictions = fold_output / f"fold{fold}_predictions.npz"
    if resume and checkpoint.exists() and predictions.exists():
        with np.load(predictions, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    train, held = data.indices_for_fold(fold)
    seed = int(config["training"]["seed"]) + fold * 101
    weights = class_user_sample_weights(data, train)
    dataset = P101Dataset(data, train, CANONICAL_VARIANTS[variant], sample_weights=weights)
    model_cfg = model_config(config, variant)
    model = build_causal_base_model(config, variant, seed, device)
    print(
        json.dumps(
            {
                "stage": "base",
                "variant": variant,
                "fold": fold,
                "held_users": FOLD_USERS[fold],
                "train_rows": len(train),
                "held_rows": len(held),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "epochs": epochs,
            }
        ),
        flush=True,
    )
    history = train_base(
        model,
        make_loader(dataset, int(config["training"]["batch_size"]), True, seed),
        device,
        config,
        epochs,
    )
    save_checkpoint(checkpoint, model, history, FOLD_USERS[fold], model_cfg)
    arrays = evaluate_fold_with_counterfactuals(
        model, data, held, variant, config, seed, device
    )
    np.savez_compressed(predictions, **arrays)
    return arrays


def nested_vs_errors(
    data: P101Data,
    outer_fold: int,
    config: dict[str, Any],
    output: Path,
    epochs: int,
    device: torch.device,
    resume: bool,
) -> np.ndarray:
    nested_output = output / "nested_vs" / f"outer{outer_fold}"
    nested_output.mkdir(parents=True, exist_ok=True)
    complete_path = nested_output / "nested_predictions.npz"
    outer_train, _ = data.indices_for_fold(outer_fold)
    if resume and complete_path.exists():
        with np.load(complete_path, allow_pickle=False) as archive:
            errors = np.asarray(archive["nested_error"], dtype=np.float32)
            covered = np.asarray(archive["covered"], dtype=bool)
        if covered[outer_train].all():
            return errors
    logits = np.full((len(data.sample_ids), 40), np.nan, dtype=np.float32)
    covered = np.zeros(len(data.sample_ids), dtype=bool)
    for inner_fold in range(4):
        if inner_fold == outer_fold:
            continue
        checkpoint = nested_output / f"inner{inner_fold}_final.pt"
        prediction = nested_output / f"inner{inner_fold}_predictions.npz"
        inner_held = np.flatnonzero(data.fold_ids == inner_fold).astype(np.int64)
        inner_train = np.flatnonzero(
            (data.fold_ids != outer_fold) & (data.fold_ids != inner_fold)
        ).astype(np.int64)
        if set(data.users[inner_train]) & set(data.users[inner_held]):
            raise RuntimeError("P101 nested subject leakage")
        if resume and checkpoint.exists() and prediction.exists():
            with np.load(prediction, allow_pickle=False) as archive:
                rows = np.asarray(archive["rows"], dtype=np.int64)
                values = np.asarray(archive["logits"], dtype=np.float32)
        else:
            seed = int(config["training"]["seed"]) + 1000 + outer_fold * 10 + inner_fold
            set_seed(seed)
            weights = class_user_sample_weights(data, inner_train)
            cfg = model_config(config, "VS")
            model = build_causal_base_model(config, "VS", seed, device)
            print(
                json.dumps(
                    {
                        "stage": "nested_vs",
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "train_users": sorted(set(data.users[inner_train].tolist())),
                        "held_users": sorted(set(data.users[inner_held].tolist())),
                        "epochs": epochs,
                    }
                ),
                flush=True,
            )
            history = train_base(
                model,
                make_loader(
                    P101Dataset(
                        data,
                        inner_train,
                        CANONICAL_VARIANTS["VS"],
                        sample_weights=weights,
                    ),
                    int(config["training"]["batch_size"]),
                    True,
                    seed,
                ),
                device,
                config,
                epochs,
            )
            save_checkpoint(
                checkpoint,
                model,
                history,
                tuple(sorted(set(data.users[inner_held].tolist()))),
                cfg,
                extra={
                    "outer_excluded_users": FOLD_USERS[outer_fold],
                    "nested_source_safe": True,
                },
            )
            result = evaluate(
                model,
                make_loader(
                    P101Dataset(data, inner_held, CANONICAL_VARIANTS["VS"]),
                    int(config["training"]["eval_batch_size"]),
                    False,
                    seed,
                ),
                device,
            )
            rows, values = result["rows"], result["logits"]
            np.savez_compressed(prediction, rows=rows, logits=values)
        logits[rows] = values
        covered[rows] = True
    if not covered[outer_train].all() or not np.isfinite(logits[outer_train]).all():
        raise RuntimeError("P101 nested VS does not cover the outer training rows")
    error = np.zeros(len(data.sample_ids), dtype=np.float32)
    error[outer_train] = (
        logits[outer_train].argmax(axis=1) != data.labels[outer_train]
    ).astype(np.float32)
    np.savez_compressed(
        complete_path,
        sample_ids=data.sample_ids,
        logits=logits,
        nested_error=error,
        covered=covered,
        outer_fold=outer_fold,
        outer_train_users=np.asarray(sorted(set(data.users[outer_train].tolist()))),
    )
    return error


def load_model_from_checkpoint(
    checkpoint: Path, config: P101ModelConfig, device: torch.device
) -> P101FineGrainedTeacher:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = P101FineGrainedTeacher(config)
    model.load_state_dict(payload["state_dict"])
    return model.to(device)


@torch.no_grad()
def exact_anchor_error(
    anchor: P101FineGrainedTeacher,
    candidate: P101FineGrainedTeacher,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    anchor.eval()
    candidate.eval()
    batch = move_batch(batch, device)
    return float((anchor(batch)["logits"] - candidate(batch)["logits"]).abs().max())


def run_vsi_fold(
    data: P101Data,
    fold: int,
    nested_error: np.ndarray,
    config: dict[str, Any],
    output: Path,
    epochs: int,
    device: torch.device,
    resume: bool,
) -> dict[str, np.ndarray]:
    fold_output = output / "VSI"
    fold_output.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_output / f"fold{fold}_final.pt"
    predictions = fold_output / f"fold{fold}_predictions.npz"
    if resume and checkpoint.exists() and predictions.exists():
        with np.load(predictions, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    vs_checkpoint = output / "VS" / f"fold{fold}_final.pt"
    if not vs_checkpoint.exists():
        raise FileNotFoundError(f"P101 VSI requires matched VS checkpoint: {vs_checkpoint}")
    train, held = data.indices_for_fold(fold)
    vs_cfg = model_config(config, "VS")
    vsi_cfg = model_config(config, "VSI")
    anchor = load_model_from_checkpoint(vs_checkpoint, vs_cfg, device)
    candidate = P101FineGrainedTeacher(vsi_cfg)
    incompatible = candidate.load_state_dict(anchor.state_dict(), strict=False)
    if incompatible.unexpected_keys or not incompatible.missing_keys:
        raise RuntimeError(f"P101 VS->VSI initialization changed: {incompatible}")
    candidate = candidate.to(device)
    seed = int(config["training"]["seed"]) + 2000 + fold
    weights = class_user_sample_weights(data, train)
    negatives = within_subject_wrong_label_source(data, train, seed)
    dataset = P101Dataset(
        data,
        train,
        CANONICAL_VARIANTS["VSI"],
        sample_weights=weights,
        nested_vs_error=nested_error,
        negative_imu_source=negatives,
    )
    initial_batch = next(
        iter(make_loader(P101Dataset(data, train[:2], CANONICAL_VARIANTS["VSI"]), 2, False, seed))
    )
    initial_error = exact_anchor_error(anchor, candidate, initial_batch, device)
    if initial_error > 1e-7:
        raise RuntimeError(f"P101 VSI did not exactly initialize from VS: {initial_error}")
    print(
        json.dumps(
            {
                "stage": "vsi_adapter",
                "fold": fold,
                "nested_vs_errors": int(nested_error[train].sum()),
                "nested_vs_error_rate": float(nested_error[train].mean()),
                "exact_anchor_max_abs_error": initial_error,
                "epochs": epochs,
            }
        ),
        flush=True,
    )
    history, trainable, frozen = train_vsi_adapter(
        candidate,
        anchor,
        make_loader(dataset, int(config["training"]["batch_size"]), True, seed),
        device,
        config,
        epochs,
    )
    save_checkpoint(
        checkpoint,
        candidate,
        history,
        FOLD_USERS[fold],
        vsi_cfg,
        extra={
            "matched_vs_checkpoint": str(vs_checkpoint),
            "exact_anchor_max_abs_error": initial_error,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "nested_vs_error_rows": int(nested_error[train].sum()),
        },
    )
    arrays = evaluate_fold_with_counterfactuals(
        candidate, data, held, "VSI", config, seed, device
    )
    np.savez_compressed(predictions, **arrays)
    return arrays


def aggregate_variant(
    data: P101Data,
    variant: str,
    fold_outputs: list[dict[str, np.ndarray]],
    output: Path,
) -> dict[str, Any]:
    full: dict[str, np.ndarray] = {}
    for fold_output in fold_outputs:
        rows = np.asarray(fold_output["rows"], dtype=np.int64)
        for name, values in fold_output.items():
            if name == "rows":
                continue
            shape = (len(data.sample_ids),) + values.shape[1:]
            target = full.setdefault(name, np.full(shape, np.nan, dtype=np.float32))
            target[rows] = values
    if not np.isfinite(full["direct_logits"]).all():
        raise RuntimeError(f"P101 {variant} OOF is incomplete")
    probability = softmax_numpy(full["direct_logits"])
    np.savez_compressed(
        output / f"{variant}_complete_oof.npz",
        sample_ids=data.sample_ids,
        users=data.users,
        fold_ids=data.fold_ids,
        direct_probability=probability,
        **full,
    )
    counterfactuals = {
        name.removesuffix("_logits"): classification_metrics(
            softmax_numpy(values), data.labels, data.users
        )
        for name, values in full.items()
        if name.endswith("_logits") and name != "direct_logits"
    }
    audits = {
        name.removeprefix("direct_"): {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for name, values in full.items()
        if name.startswith("direct_") and name != "direct_logits"
    }
    return {
        "variant": variant,
        "modalities": list(CANONICAL_VARIANTS[variant]),
        "direct": classification_metrics(probability, data.labels, data.users),
        "counterfactuals": counterfactuals,
        "mechanism": audits,
        "_probability": probability,
    }


def top5_correct(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    top = np.argpartition(probability, -5, axis=1)[:, -5:]
    return np.any(top == labels[:, None], axis=1)


def development_gate(
    vsi: dict[str, Any],
    vs: dict[str, Any],
    coarse_probability: np.ndarray,
    comparison: dict[str, Any],
    data: P101Data,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    subject = list(comparison["per_subject"].values())
    vsi_probability = vsi["_probability"]
    vs_probability = vs["_probability"]
    top5_delta_rows = int(
        top5_correct(vsi_probability, data.labels).sum()
        - top5_correct(vs_probability, data.labels).sum()
    )
    counter = vsi["counterfactuals"]
    aligned_imu = all(
        vsi["direct"]["top1"] > counter[name]["top1"]
        or vsi["direct"]["top5"] > counter[name]["top5"]
        for name in ("zero_imu", "shuffle_imu", "reverse_imu")
    )
    stability = {
        "subjects_nonnegative_at_least_8": sum(value["net"] >= 0 for value in subject)
        >= int(gate["min_nonnegative_subjects"]),
        "subjects_positive_at_least_4": sum(value["net"] > 0 for value in subject)
        >= int(gate["min_positive_subjects"]),
        "worst_subject_drop_within_2pp": min(value["accuracy_pp"] for value in subject)
        >= -float(gate["max_subject_drop_pp"]),
    }
    top1_checks = {
        "matched_net_at_least_10": comparison["net"]
        >= int(gate["matched_vsi_minus_vs_min_net"]),
        "mcnemar_p_at_most_0_10": comparison["mcnemar_exact_p"]
        <= float(gate["mcnemar_max_p"]),
        "top5_not_lower": vsi["direct"]["top5"] >= vs["direct"]["top5"],
        "above_p100_coarse_vs": vsi["direct"]["top1"]
        > topk_accuracy(coarse_probability, data.labels, 1),
        "aligned_imu_over_all_counterfactuals": aligned_imu,
        **stability,
    }
    top5_checks = {
        "top5_delta_rows_at_least_10": top5_delta_rows
        >= int(gate["top5_only_min_rows"]),
        "top1_drop_at_most_2_rows": comparison["net"]
        >= -int(gate["top5_only_max_top1_drop_rows"]),
        "macro_f1_drop_within_0_2pp": vsi["direct"]["macro_f1"]
        >= vs["direct"]["macro_f1"] - float(gate["top5_only_max_macro_f1_drop"]),
        "aligned_imu_over_all_counterfactuals": aligned_imu,
        **stability,
    }
    return {
        "top1_path": {"checks": top1_checks, "passed": bool(all(top1_checks.values()))},
        "top5_only_path": {"checks": top5_checks, "passed": bool(all(top5_checks.values()))},
        "top5_delta_rows": top5_delta_rows,
        "passed": bool(all(top1_checks.values()) or all(top5_checks.values())),
    }


def load_fold_predictions(output: Path, variant: str) -> list[dict[str, np.ndarray]]:
    values: list[dict[str, np.ndarray]] = []
    for fold in range(4):
        path = output / variant / f"fold{fold}_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            values.append({key: np.asarray(archive[key]) for key in archive.files})
    return values


def finalize(
    data: P101Data, config: dict[str, Any], config_hash: str, output: Path
) -> dict[str, Any]:
    summaries = {
        variant: aggregate_variant(
            data, variant, load_fold_predictions(output, variant), output
        )
        for variant in CANONICAL_VARIANTS
    }
    comparisons = {
        "VS_minus_V_skeleton": paired_comparison(
            summaries["VS"]["_probability"], summaries["V"]["_probability"], data.labels, data.users
        ),
        "VI_minus_V_imu": paired_comparison(
            summaries["VI"]["_probability"], summaries["V"]["_probability"], data.labels, data.users
        ),
        "VSI_minus_VS_imu": paired_comparison(
            summaries["VSI"]["_probability"], summaries["VS"]["_probability"], data.labels, data.users
        ),
        "VSI_minus_VI_skeleton": paired_comparison(
            summaries["VSI"]["_probability"], summaries["VI"]["_probability"], data.labels, data.users
        ),
        "VSI_minus_V_joint": paired_comparison(
            summaries["VSI"]["_probability"], summaries["V"]["_probability"], data.labels, data.users
        ),
    }
    with np.load(COARSE_VS, allow_pickle=False) as archive:
        coarse_ids = np.asarray(archive["sample_ids"]).astype(str)
        if not np.array_equal(coarse_ids, data.sample_ids):
            raise RuntimeError("P100 coarse VS row order differs from P101")
        coarse_probability = np.asarray(archive["direct_probability"], dtype=np.float32)
    comparisons["P101_VS_minus_P100_coarse_VS"] = paired_comparison(
        summaries["VS"]["_probability"], coarse_probability, data.labels, data.users
    )
    comparisons["P101_VSI_minus_P100_coarse_VS"] = paired_comparison(
        summaries["VSI"]["_probability"], coarse_probability, data.labels, data.users
    )
    comparisons["VSI_minus_VS_imu"]["confusion_changes"] = confusion_change_groups(
        summaries["VSI"]["_probability"], summaries["VS"]["_probability"], data.labels
    )
    gate = development_gate(
        summaries["VSI"],
        summaries["VS"],
        coarse_probability,
        comparisons["VSI_minus_VS_imu"],
        data,
        config,
    )
    clean = {
        variant: {key: value for key, value in summary.items() if key != "_probability"}
        for variant, summary in summaries.items()
    }
    result = {
        "status": "complete",
        "protocol": "P101 fixed four-fold 12-subject OOF; nested subject-OOF VS errors; fixed final epochs",
        "git_commit": git_commit(),
        "config_sha256": config_hash,
        "data": data.summary(),
        "variants": clean,
        "comparisons": comparisons,
        "a_teacher_positive_gate": gate,
        "next_stage": "STUDENT_RAW_ALLOWED" if gate["passed"] else "MECHANISM_AUDIT_REQUIRED",
        "leakage_audit": {
            "development_allow_list_exact": set(data.users.tolist()) == DEV_USER_SET,
            "h3_rows_loaded": 0,
            "historical_40class_predictions_loaded": False,
            "outer_subject_disjoint": True,
            "nested_subject_disjoint": True,
            "outer_label_used_for_checkpoint_selection": False,
            "b_teacher_started": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = [value.strip() for value in args.stages.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    folds = list(range(4)) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    if not set(stages) <= {"base", "nested", "vsi", "finalize"}:
        raise ValueError(f"unknown P101 stages: {stages}")
    if not set(variants) <= {"V", "VS", "VI"}:
        raise ValueError(f"base variants must be V/VS/VI: {variants}")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    data = load_p101_data()
    base_epochs = 1 if args.smoke else int(config["training"]["base_epochs"])
    nested_epochs = 1 if args.smoke else int(config["training"]["nested_epochs"])
    vsi_epochs = 1 if args.smoke else int(config["training"]["vsi_epochs"])
    manifest = {
        "status": "smoke" if args.smoke else "formal_started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": str(config_path),
        "config_sha256": config_hash,
        "stages": stages,
        "variants": variants,
        "folds": folds,
        "epochs": {"base": base_epochs, "nested": nested_epochs, "vsi": vsi_epochs},
        "device": str(device),
        "data": data.summary(),
        "outer_label_used_for_checkpoint_selection": False,
        "h3_code_path_present": False,
        "b_teacher_code_path_present": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "base" in stages:
        for variant in variants:
            for fold in folds:
                run_base_fold(
                    data, variant, fold, config, output, base_epochs, device, args.resume
                )
    nested_by_fold: dict[int, np.ndarray] = {}
    if "nested" in stages or "vsi" in stages:
        for fold in folds:
            nested_by_fold[fold] = nested_vs_errors(
                data, fold, config, output, nested_epochs, device, args.resume
            )
    if "vsi" in stages:
        for fold in folds:
            run_vsi_fold(
                data,
                fold,
                nested_by_fold[fold],
                config,
                output,
                vsi_epochs,
                device,
                args.resume,
            )
    if "finalize" in stages or (
        not args.smoke and folds == list(range(4)) and set(variants) == {"V", "VS", "VI"}
    ):
        result = finalize(data, config, config_hash, output)
        print(json.dumps(result["a_teacher_positive_gate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
