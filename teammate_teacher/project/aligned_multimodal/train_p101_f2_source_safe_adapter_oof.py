"""Train P101-F2 shared local VSI adapters on source-safe nested anchors."""

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
from torch.utils.data import DataLoader

from p100a_global_teacher_data import (
    CANONICAL_VARIANTS as P100_VARIANTS,
    FOLD_USERS,
    FoldNormalizer,
    P100ADataset,
    load_p100a_data,
)
from p100a_global_teacher_model import P100AGlobalTeacher, P100AModelConfig
from p101_f1_coarse_anchor_data import P101F1Dataset
from p101_f1_coarse_anchor_model import (
    P101F1CoarseAnchoredTeacher,
    P101F1Config,
    select_f1_trainable_parameters,
)
from p101_finegrained_teacher_data import (
    class_user_sample_weights,
    load_p101_data,
    within_subject_wrong_label_source,
)
from train_p100a_global_teacher_oof import (
    classification_metrics,
    confusion_change_groups,
    evaluate_model as evaluate_p100,
    paired_comparison,
    softmax_numpy,
)
from train_p101_f1_coarse_anchor_oof import (
    counterfactual_dataset,
    evaluate,
    exact_initial_error,
    gate_result,
    load_outer_anchor,
    make_loader,
    move_batch,
    rebase_to_canonical_anchor,
    set_adapter_mode,
    summarize_candidate,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p101_f2_source_safe_adapter.json"
DEFAULT_OUTPUT = HERE / "runs/p101_f2_source_safe_adapter_oof_v1"
F1_RUN = HERE / "runs/p101_f1_coarse_anchor_oof_v1"
F0_RUN = HERE / "runs/p101_f0_finegrained_teacher_oof_v1"
P100_RUN = HERE / "runs/p100a_a0_global_teacher_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stages", default="adapter,finalize")
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


def f2_config(config: dict[str, Any]) -> P101F1Config:
    return P101F1Config(**config["model"])


def source_safe_inner_folds(outer_fold: int) -> tuple[int, ...]:
    if outer_fold not in range(4):
        raise ValueError(outer_fold)
    return tuple(fold for fold in range(4) if fold != outer_fold)


def load_nested_anchor(
    outer_fold: int, inner_fold: int, device: torch.device
) -> tuple[P100AGlobalTeacher, FoldNormalizer, dict[str, Any]]:
    if inner_fold not in source_safe_inner_folds(outer_fold):
        raise RuntimeError("P101-F2 nested anchor cannot use the outer-held fold")
    path = (
        F1_RUN
        / "nested_coarse_vs"
        / f"outer{outer_fold}"
        / f"inner{inner_fold}_final.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_held = tuple(FOLD_USERS[inner_fold])
    expected_excluded = tuple(FOLD_USERS[outer_fold])
    if tuple(payload["held_users"]) != expected_held:
        raise RuntimeError(f"P101-F2 nested held-subject contract changed: {path}")
    if tuple(payload["outer_excluded_users"]) != expected_excluded:
        raise RuntimeError(f"P101-F2 outer exclusion contract changed: {path}")
    if not bool(payload["nested_source_safe"]):
        raise RuntimeError(f"P101-F2 nested checkpoint is not source-safe: {path}")
    values = dict(payload["model_config"])
    values["modalities"] = tuple(values["modalities"])
    model = P100AGlobalTeacher(P100AModelConfig(**values))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalizer = FoldNormalizer(
        means={
            key: np.asarray(value) for key, value in payload["normalizer_means"].items()
        },
        stds={
            key: np.asarray(value) for key, value in payload["normalizer_stds"].items()
        },
    )
    return model.to(device), normalizer, {
        "path": str(path),
        "held_users": list(expected_held),
        "outer_excluded_users": list(expected_excluded),
        "nested_source_safe": True,
    }


def allocate_group_batch_counts(sizes: list[int], batch_size: int) -> list[int]:
    if not sizes or any(size <= 0 for size in sizes) or batch_size <= 0:
        raise ValueError("P101-F2 batch allocation requires positive group sizes")
    total_batches = math.ceil(sum(sizes) / batch_size)
    raw = np.asarray(sizes, dtype=np.float64) / sum(sizes) * total_batches
    counts = np.maximum(np.floor(raw).astype(np.int64), 1)
    while int(counts.sum()) < total_batches:
        index = int(np.argmax(raw - counts))
        counts[index] += 1
    while int(counts.sum()) > total_batches:
        candidates = np.flatnonzero(counts > 1)
        if not len(candidates):
            raise RuntimeError("P101-F2 could not match the fixed optimizer-step budget")
        index = int(candidates[np.argmin(raw[candidates] - counts[candidates])])
        counts[index] -= 1
    return counts.astype(int).tolist()


def make_epoch_loader(
    dataset: P101F1Dataset, batch_count: int, seed: int
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = np.random.default_rng(seed)
    positions = generator.permutation(len(dataset))
    batches = [values.tolist() for values in np.array_split(positions, batch_count)]
    if any(not values for values in batches):
        raise RuntimeError("P101-F2 produced an empty source-safe batch")
    return DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def non_anchor_state(model: P101F1CoarseAnchoredTeacher) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("anchor.")
    }
    if not state or any(name.startswith("anchor.") for name in state):
        raise RuntimeError("P101-F2 adapter state isolation failed")
    return state


def load_non_anchor_state(
    model: P101F1CoarseAnchoredTeacher, state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [name for name in missing if not name.startswith("anchor.")]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"P101-F2 adapter state contract failed: missing={invalid_missing[:3]} "
            f"unexpected={unexpected[:3]}"
        )
    return {
        "adapter_tensors": len(state),
        "outer_anchor_tensors_preserved": len(missing),
    }


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return max((step + 1) / max(warmup, 1), 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def matched_nested_arrays(
    outer_fold: int, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        F1_RUN
        / "nested_coarse_vs"
        / f"outer{outer_fold}"
        / "nested_predictions.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        covered = np.asarray(archive["covered"], dtype=bool)
        error = np.asarray(archive["nested_error"], dtype=np.float32)
        uncertainty = np.asarray(archive["nested_uncertainty"], dtype=np.float32)
    if not np.all(covered[rows]):
        raise RuntimeError("P101-F2 outer-train rows lack nested source-safe predictions")
    return error, uncertainty


@torch.no_grad()
def verify_nested_prediction(
    anchor: P100AGlobalTeacher,
    normalizer: FoldNormalizer,
    coarse: Any,
    rows: np.ndarray,
    outer_fold: int,
    inner_fold: int,
    device: torch.device,
) -> dict[str, Any]:
    result = evaluate_p100(
        anchor,
        make_loader(
            P100ADataset(coarse, rows, normalizer, P100_VARIANTS["VS"]),
            64,
            False,
            12000 + outer_fold * 10 + inner_fold,
        ),
        device,
    )
    path = (
        F1_RUN
        / "nested_coarse_vs"
        / f"outer{outer_fold}"
        / f"inner{inner_fold}_predictions.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        saved_rows = np.asarray(archive["rows"], dtype=np.int64)
        saved_logits = np.asarray(archive["logits"], dtype=np.float32)
    evaluated_rows = np.asarray(result["rows"], dtype=np.int64)
    if not np.array_equal(evaluated_rows, saved_rows) or not np.array_equal(rows, saved_rows):
        raise RuntimeError("P101-F2 nested prediction row order changed")
    current_probability = softmax_numpy(np.asarray(result["logits"], dtype=np.float32))
    saved_probability = softmax_numpy(saved_logits)
    drift = np.abs(current_probability - saved_probability)
    if float(drift.max()) > 1e-3:
        raise RuntimeError("P101-F2 nested checkpoint no longer reproduces saved predictions")
    return {
        "rows": int(len(rows)),
        "max_probability_recompute_drift": float(drift.max()),
        "argmax_rows_different": int(
            (current_probability.argmax(axis=1) != saved_probability.argmax(axis=1)).sum()
        ),
    }


def train_source_safe_adapter(
    model: P101F1CoarseAnchoredTeacher,
    anchors: dict[int, P100AGlobalTeacher],
    datasets: dict[int, P101F1Dataset],
    batch_counts: dict[int, int],
    device: torch.device,
    config: dict[str, Any],
    epochs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    training = config["training"]
    parameters, trainable, frozen = select_f1_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["adapter_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.98),
    )
    steps_per_epoch = sum(batch_counts.values())
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup = int(total_steps * float(training["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(step, total_steps, warmup)
    )
    use_amp = device.type == "cuda" and bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    temperature = float(training["anchor_kl_temperature"])
    inner_folds = sorted(anchors)
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        loaders = {
            fold: make_epoch_loader(
                datasets[fold], batch_counts[fold], seed + epoch * 100 + fold
            )
            for fold in inner_folds
        }
        iterators = {fold: iter(loader) for fold, loader in loaders.items()}
        order = inner_folds[epoch % len(inner_folds) :] + inner_folds[: epoch % len(inner_folds)]
        sums = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "pair": 0.0, "residual": 0.0}
        correct = rows_seen = steps = 0
        per_anchor: dict[int, dict[str, int]] = {
            fold: {"rows": 0, "correct": 0} for fold in inner_folds
        }
        started = time.perf_counter()
        active = True
        while active:
            active = False
            for fold in order:
                try:
                    batch = next(iterators[fold])
                except StopIteration:
                    continue
                active = True
                model.anchor = anchors[fold]
                set_adapter_mode(model)
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    output = model(batch)
                    error = batch["nested_anchor_error"]
                    ce_weight = batch["weight"] * (
                        1.0 + float(training["nested_error_ce_multiplier"]) * error
                    )
                    ce = (
                        F.cross_entropy(
                            output["logits"],
                            batch["label"],
                            reduction="none",
                            label_smoothing=float(training["label_smoothing"]),
                        )
                        * ce_weight
                    ).mean()
                    anchor_probability = F.softmax(
                        output["anchor_logits"].detach() / temperature, dim=-1
                    )
                    kl_rows = (
                        F.kl_div(
                            F.log_softmax(output["logits"] / temperature, dim=-1),
                            anchor_probability,
                            reduction="none",
                        ).sum(dim=-1)
                        * temperature**2
                    )
                    kl_weight = torch.where(
                        error > 0.5,
                        torch.full_like(
                            error, float(training["anchor_kl_wrong_weight"])
                        ),
                        torch.full_like(
                            error, float(training["anchor_kl_correct_weight"])
                        ),
                    )
                    kl = (kl_rows * kl_weight).mean()
                    positive = output["positive_correspondence_logits"]
                    negative = output["negative_correspondence_logits"]
                    pair = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            positive, torch.ones_like(positive)
                        )
                        + F.binary_cross_entropy_with_logits(
                            negative, torch.zeros_like(negative)
                        )
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
                sums["residual"] += float(output["fine_residual_rms"].mean().detach()) * size
                batch_correct = int((output["logits"].argmax(dim=1) == batch["label"]).sum())
                correct += batch_correct
                rows_seen += size
                steps += 1
                per_anchor[fold]["rows"] += size
                per_anchor[fold]["correct"] += batch_correct
        if steps != steps_per_epoch or rows_seen != sum(len(dataset) for dataset in datasets.values()):
            raise RuntimeError("P101-F2 source-safe epoch coverage changed")
        record = {
            "epoch": float(epoch + 1),
            **{key: value / max(rows_seen, 1) for key, value in sums.items()},
            "source_safe_accuracy": correct / max(rows_seen, 1),
            "per_anchor_accuracy": {
                str(fold): value["correct"] / value["rows"]
                for fold, value in per_anchor.items()
            },
            "optimizer_steps": steps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    return history, trainable, frozen


def evaluate_with_counterfactuals(
    model: P101F1CoarseAnchoredTeacher,
    coarse: Any,
    fine: Any,
    held: np.ndarray,
    normalizer: FoldNormalizer,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    batch_size = int(config["training"]["eval_batch_size"])
    direct = evaluate(
        model,
        make_loader(P101F1Dataset(coarse, fine, held, normalizer), batch_size, False, seed),
        device,
    )
    arrays = {"rows": direct.pop("rows"), "direct_logits": direct.pop("logits")}
    arrays.update({f"direct_{key}": value for key, value in direct.items()})
    kinds = (
        "zero_skeleton",
        "local_reverse_skeleton",
        "shuffle_skeleton",
        "zero_imu",
        "reverse_imu",
        "shuffle_imu",
        "zero_both",
    )
    for number, kind in enumerate(kinds):
        result = evaluate(
            model,
            make_loader(
                counterfactual_dataset(
                    coarse, fine, held, normalizer, kind, seed + 1000 + number
                ),
                batch_size,
                False,
                seed,
            ),
            device,
        )
        if not np.array_equal(result["rows"].astype(np.int64), arrays["rows"].astype(np.int64)):
            raise RuntimeError("P101-F2 counterfactual row order changed")
        arrays[f"{kind}_logits"] = result["logits"]
    if not np.array_equal(arrays["zero_imu_logits"], arrays["direct_anchor_logits"]):
        raise RuntimeError("P101-F2 zero IMU did not exactly return the outer anchor")
    return arrays


def run_fold(
    coarse: Any,
    fine: Any,
    fold: int,
    config: dict[str, Any],
    output: Path,
    epochs: int,
    device: torch.device,
    resume: bool,
) -> dict[str, np.ndarray]:
    fold_output = output / "F2_VSI"
    fold_output.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_output / f"fold{fold}_final.pt"
    predictions = fold_output / f"fold{fold}_predictions.npz"
    if resume and checkpoint.exists() and predictions.exists():
        with np.load(predictions, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    train, held = coarse.indices_for_fold(fold)
    nested_error, nested_uncertainty = matched_nested_arrays(fold, train)
    seed = int(config["training"]["seed"]) + 5000 + fold
    set_seed(seed)
    anchors: dict[int, P100AGlobalTeacher] = {}
    normalizers: dict[int, FoldNormalizer] = {}
    source_audits: dict[int, dict[str, Any]] = {}
    group_rows: dict[int, np.ndarray] = {}
    for inner_fold in source_safe_inner_folds(fold):
        anchor, normalizer, contract = load_nested_anchor(fold, inner_fold, device)
        anchor.requires_grad_(False).eval()
        rows = train[np.isin(coarse.users[train], np.asarray(FOLD_USERS[inner_fold]))]
        if set(coarse.users[rows].tolist()) != set(FOLD_USERS[inner_fold]):
            raise RuntimeError("P101-F2 source-safe subject group coverage changed")
        anchors[inner_fold] = anchor
        normalizers[inner_fold] = normalizer
        group_rows[inner_fold] = rows
        source_audits[inner_fold] = {
            **contract,
            "prediction": verify_nested_prediction(
                anchor, normalizer, coarse, rows, fold, inner_fold, device
            ),
        }
    first_fold = source_safe_inner_folds(fold)[0]
    model = P101F1CoarseAnchoredTeacher(anchors[first_fold], f2_config(config))
    f0_path = F0_RUN / "VSI" / f"fold{fold}_final.pt"
    f0_payload = torch.load(f0_path, map_location="cpu", weights_only=False)
    local_audit = model.local.load_f0_state(f0_payload["state_dict"])
    model = model.to(device)
    exact: dict[int, float] = {}
    for inner_fold in source_safe_inner_folds(fold):
        model.anchor = anchors[inner_fold]
        dataset = P101F1Dataset(
            coarse, fine, group_rows[inner_fold][:2], normalizers[inner_fold]
        )
        batch = next(iter(make_loader(dataset, 2, False, seed + inner_fold)))
        exact[inner_fold] = exact_initial_error(model, batch, device)
    if any(value > 1e-7 for value in exact.values()):
        raise RuntimeError(f"P101-F2 nested exact-zero initialization failed: {exact}")
    weights = class_user_sample_weights(coarse, train)
    negatives = within_subject_wrong_label_source(fine, train, seed)
    datasets = {
        inner_fold: P101F1Dataset(
            coarse,
            fine,
            group_rows[inner_fold],
            normalizers[inner_fold],
            sample_weights=weights,
            nested_anchor_error=nested_error,
            nested_anchor_uncertainty=nested_uncertainty,
            negative_imu_source=negatives,
        )
        for inner_fold in source_safe_inner_folds(fold)
    }
    sizes = [len(datasets[inner_fold]) for inner_fold in source_safe_inner_folds(fold)]
    counts = allocate_group_batch_counts(sizes, int(config["training"]["batch_size"]))
    batch_counts = dict(zip(source_safe_inner_folds(fold), counts))
    print(
        json.dumps(
            {
                "stage": "f2_source_safe_adapter",
                "fold": fold,
                "source_safe_error_rows": int(nested_error[train].sum()),
                "source_safe_error_rate": float(nested_error[train].mean()),
                "exact_nested_anchor_errors": exact,
                "group_rows": {str(key): len(value) for key, value in group_rows.items()},
                "optimizer_steps_per_epoch": sum(counts),
                "matched_f1_steps_per_epoch": math.ceil(len(train) / int(config["training"]["batch_size"])),
                "epochs": epochs,
            }
        ),
        flush=True,
    )
    history, trainable, frozen = train_source_safe_adapter(
        model,
        anchors,
        datasets,
        batch_counts,
        device,
        config,
        epochs,
        seed,
    )
    adapter_state = non_anchor_state(model)
    outer_anchor, outer_normalizer, outer_audit = load_outer_anchor(fold, device)
    inference_model = P101F1CoarseAnchoredTeacher(outer_anchor, f2_config(config)).to(device)
    state_audit = load_non_anchor_state(inference_model, adapter_state)
    zero_dataset = P101F1Dataset(
        coarse, fine, held[:2], outer_normalizer, zero_modalities=("imu",)
    )
    zero_batch = next(iter(make_loader(zero_dataset, 2, False, seed)))
    with torch.no_grad():
        zero_output = inference_model(move_batch(zero_batch, device))
        zero_error = float(
            (zero_output["logits"] - zero_output["anchor_logits"]).abs().max()
        )
    if zero_error != 0.0:
        raise RuntimeError("P101-F2 deployed zero-IMU path is not exact outer anchor")
    torch.save(
        {
            "adapter_state": adapter_state,
            "f2_config": inference_model.config.__dict__,
            "history": history,
            "held_users": FOLD_USERS[fold],
            "outer_anchor": outer_audit,
            "nested_anchor_audits": source_audits,
            "f0_local_source": str(f0_path),
            "local_initialization": local_audit,
            "exact_nested_anchor_errors": exact,
            "deployed_zero_imu_exact_anchor_error": zero_error,
            "source_safe_error_rows": int(nested_error[train].sum()),
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "adapter_state_audit": state_audit,
            "outer_label_used_for_selection": False,
        },
        checkpoint,
    )
    arrays = evaluate_with_counterfactuals(
        inference_model,
        coarse,
        fine,
        held,
        outer_normalizer,
        config,
        seed,
        device,
    )
    np.savez_compressed(predictions, **arrays)
    return arrays


def load_fold_predictions(output: Path) -> list[dict[str, np.ndarray]]:
    values: list[dict[str, np.ndarray]] = []
    for fold in range(4):
        path = output / "F2_VSI" / f"fold{fold}_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            values.append({name: np.asarray(archive[name]) for name in archive.files})
    return values


def aggregate_arrays(fine: Any, fold_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    full: dict[str, np.ndarray] = {}
    for fold_output in fold_outputs:
        rows = fold_output["rows"].astype(np.int64)
        for name, values in fold_output.items():
            if name == "rows":
                continue
            shape = (len(fine.sample_ids),) + values.shape[1:]
            full.setdefault(name, np.full(shape, np.nan, dtype=np.float32))[rows] = values
    if not np.isfinite(full["direct_logits"]).all():
        raise RuntimeError("P101-F2 OOF is incomplete")
    full["direct_probability"] = softmax_numpy(full["direct_logits"])
    return full


def save_complete_oof(fine: Any, arrays: dict[str, np.ndarray], output: Path) -> None:
    np.savez_compressed(
        output / "F2_VSI_complete_oof.npz",
        sample_ids=fine.sample_ids,
        users=fine.users,
        fold_ids=fine.fold_ids,
        **arrays,
    )


def finalize(fine: Any, config: dict[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    arrays = aggregate_arrays(fine, load_fold_predictions(output))
    with np.load(P100_RUN / "VS_complete_oof.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["sample_ids"].astype(str), fine.sample_ids):
            raise RuntimeError("P101-F2 canonical anchor row order changed")
        anchor_logits = np.asarray(archive["direct_logits"], dtype=np.float32)
        anchor_probability = np.asarray(archive["direct_probability"], dtype=np.float32)
    arrays, rebase_audit = rebase_to_canonical_anchor(arrays, anchor_logits)
    save_complete_oof(fine, arrays, output)
    candidate = summarize_candidate(fine, arrays)
    comparison = paired_comparison(
        arrays["direct_probability"], anchor_probability, fine.labels, fine.users
    )
    comparison["confusion_changes"] = confusion_change_groups(
        arrays["direct_probability"], anchor_probability, fine.labels
    )
    gate = gate_result(
        candidate,
        arrays["direct_probability"],
        anchor_probability,
        comparison,
        fine,
        config,
    )
    result = {
        "status": "complete",
        "protocol": "P101-F2 shared local VSI pre-classifier adapter trained on source-safe nested coarse VS boundaries; outer P100 VS inference anchor",
        "git_commit": git_commit(),
        "config_sha256": config_hash,
        "candidate": candidate,
        "anchor": classification_metrics(anchor_probability, fine.labels, fine.users),
        "comparison": comparison,
        "a_teacher_positive_gate": gate,
        "next_stage": "STUDENT_RAW_ALLOWED" if gate["passed"] else "MECHANISM_AUDIT_REQUIRED",
        "leakage_audit": {
            "rows": len(fine.sample_ids),
            "h3_rows_loaded": 0,
            "outer_subject_disjoint": True,
            "adapter_training_anchor_subject_disjoint_for_every_row": True,
            "nested_anchor_count": 12,
            "nested_checkpoints_selected_by_held_metric": False,
            "outer_label_used_for_checkpoint_selection": False,
            "nested_heads_present_in_inference": False,
            "historical_40class_logits_used_as_model_input": False,
            "canonical_anchor_rebase": rebase_audit,
            "student_started": False,
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
    if not set(stages) <= {"adapter", "finalize"}:
        raise ValueError(f"unknown P101-F2 stages: {stages}")
    folds = list(range(4)) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    coarse = load_p100a_data()
    fine = load_p101_data()
    if not np.array_equal(coarse.sample_ids, fine.sample_ids):
        raise RuntimeError("P101-F2 coarse/fine row contract differs")
    epochs = 1 if args.smoke else int(config["training"]["adapter_epochs"])
    manifest = {
        "status": "smoke" if args.smoke else "formal_started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": str(config_path),
        "config_sha256": config_hash,
        "stages": stages,
        "folds": folds,
        "adapter_epochs": epochs,
        "device": str(device),
        "rows": len(fine.sample_ids),
        "subjects": sorted(set(fine.users.tolist())),
        "nested_source_run": str(F1_RUN),
        "h3_rows": 0,
        "student_started": False,
        "b_teacher_started": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs: list[dict[str, np.ndarray]] = []
    if "adapter" in stages:
        for fold in folds:
            outputs.append(
                run_fold(
                    coarse,
                    fine,
                    fold,
                    config,
                    output,
                    epochs,
                    device,
                    args.resume,
                )
            )
    if "finalize" in stages and not args.smoke and folds == list(range(4)):
        result = finalize(fine, config, config_hash, output)
        print(json.dumps(result["a_teacher_positive_gate"], ensure_ascii=False))
    elif outputs:
        print(json.dumps({"status": "partial", "folds": folds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
