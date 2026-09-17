"""Run P101-F1 coarse-anchored fine local VSI subject-disjoint OOF."""

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
    class_user_sample_weights as p100_weights,
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
    within_subject_permutation,
    within_subject_wrong_label_source,
)
from train_p100a_global_teacher_oof import (
    classification_metrics,
    confusion_change_groups,
    evaluate_model as evaluate_p100,
    paired_comparison,
    softmax_numpy,
    topk_accuracy,
    train_model as train_p100,
)
from train_p101_finegrained_teacher_oof import attention_audit


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p101_f1_coarse_anchor.json"
DEFAULT_OUTPUT = HERE / "runs/p101_f1_coarse_anchor_oof_v1"
P100_RUN = HERE / "runs/p100a_a0_global_teacher_oof_v1"
F0_RUN = HERE / "runs/p101_f0_finegrained_teacher_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stages", default="nested,adapter,finalize")
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
    dataset: Any, batch_size: int, shuffle: bool, seed: int
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


def load_outer_anchor(
    fold: int, device: torch.device
) -> tuple[P100AGlobalTeacher, FoldNormalizer, dict[str, Any]]:
    path = P100_RUN / "VS" / f"fold{fold}_final.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_values = dict(payload["model_config"])
    model_values["modalities"] = tuple(model_values["modalities"])
    model = P100AGlobalTeacher(P100AModelConfig(**model_values))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalizer = FoldNormalizer(
        means={key: np.asarray(value) for key, value in payload["normalizer_means"].items()},
        stds={key: np.asarray(value) for key, value in payload["normalizer_stds"].items()},
    )
    return model.to(device), normalizer, {"path": str(path), "held_users": payload["held_users"]}


def f1_config(config: dict[str, Any]) -> P101F1Config:
    return P101F1Config(**config["model"])


def nested_p100_training(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    return {
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "warmup_fraction": training["warmup_fraction"],
        "label_smoothing": training["label_smoothing"],
        "gradient_clip": training["gradient_clip"],
        "amp": training["amp"],
    }


def nested_coarse_errors(
    coarse: Any,
    outer_fold: int,
    config: dict[str, Any],
    output: Path,
    epochs: int,
    device: torch.device,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nested_output = output / "nested_coarse_vs" / f"outer{outer_fold}"
    nested_output.mkdir(parents=True, exist_ok=True)
    complete_path = nested_output / "nested_predictions.npz"
    outer_train, _ = coarse.indices_for_fold(outer_fold)
    if resume and complete_path.exists():
        with np.load(complete_path, allow_pickle=False) as archive:
            error = np.asarray(archive["nested_error"], dtype=np.float32)
            uncertainty = np.asarray(
                archive["nested_uncertainty"], dtype=np.float32
            )
            covered = np.asarray(archive["covered"], dtype=bool)
        if covered[outer_train].all():
            return error, uncertainty
    logits = np.full((len(coarse.sample_ids), 40), np.nan, dtype=np.float32)
    covered = np.zeros(len(coarse.sample_ids), dtype=bool)
    for inner_fold in range(4):
        if inner_fold == outer_fold:
            continue
        checkpoint = nested_output / f"inner{inner_fold}_final.pt"
        prediction = nested_output / f"inner{inner_fold}_predictions.npz"
        held = np.flatnonzero(coarse.fold_ids == inner_fold).astype(np.int64)
        train = np.flatnonzero(
            (coarse.fold_ids != outer_fold) & (coarse.fold_ids != inner_fold)
        ).astype(np.int64)
        if set(coarse.users[train]) & set(coarse.users[held]):
            raise RuntimeError("P101-F1 nested coarse VS subject leakage")
        if resume and checkpoint.exists() and prediction.exists():
            with np.load(prediction, allow_pickle=False) as archive:
                rows = np.asarray(archive["rows"], dtype=np.int64)
                values = np.asarray(archive["logits"], dtype=np.float32)
        else:
            seed = int(config["training"]["seed"]) + 3000 + outer_fold * 10 + inner_fold
            set_seed(seed)
            normalizer = FoldNormalizer.fit(coarse, train)
            weights = p100_weights(coarse, train)
            model_cfg = P100AModelConfig(
                modalities=P100_VARIANTS["VS"], **config["nested_model"]
            )
            model = P100AGlobalTeacher(model_cfg).to(device)
            print(
                json.dumps(
                    {
                        "stage": "nested_coarse_vs",
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "train_users": sorted(set(coarse.users[train].tolist())),
                        "held_users": sorted(set(coarse.users[held].tolist())),
                        "epochs": epochs,
                    }
                ),
                flush=True,
            )
            history = train_p100(
                model,
                make_loader(
                    P100ADataset(
                        coarse,
                        train,
                        normalizer,
                        P100_VARIANTS["VS"],
                        sample_weights=weights,
                    ),
                    int(config["training"]["batch_size"]),
                    True,
                    seed,
                ),
                device,
                nested_p100_training(config),
                epochs,
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_cfg.__dict__,
                    "normalizer_means": normalizer.means,
                    "normalizer_stds": normalizer.stds,
                    "history": history,
                    "held_users": tuple(sorted(set(coarse.users[held].tolist()))),
                    "outer_excluded_users": FOLD_USERS[outer_fold],
                    "nested_source_safe": True,
                    "outer_label_used_for_selection": False,
                },
                checkpoint,
            )
            result = evaluate_p100(
                model,
                make_loader(
                    P100ADataset(
                        coarse, held, normalizer, P100_VARIANTS["VS"]
                    ),
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
        raise RuntimeError("P101-F1 nested coarse VS coverage failed")
    error = np.zeros(len(coarse.sample_ids), dtype=np.float32)
    error[outer_train] = (
        logits[outer_train].argmax(axis=1) != coarse.labels[outer_train]
    ).astype(np.float32)
    uncertainty = np.zeros(len(coarse.sample_ids), dtype=np.float32)
    probability = softmax_numpy(logits[outer_train])
    top = np.sort(probability, axis=1)[:, -2:]
    # Source-safe wrong rows must remain open even if the nested model is
    # confidently wrong; correct rows use its continuous label-free margin.
    uncertainty[outer_train] = np.maximum(
        1.0 - (top[:, 1] - top[:, 0]), error[outer_train]
    )
    np.savez_compressed(
        complete_path,
        sample_ids=coarse.sample_ids,
        logits=logits,
        nested_error=error,
        nested_uncertainty=uncertainty,
        covered=covered,
        outer_fold=outer_fold,
    )
    return error, uncertainty


def set_adapter_mode(model: P101F1CoarseAnchoredTeacher) -> None:
    model.eval()
    for module in (
        model.local.imu_encoder,
        model.local.imu_projection,
        model.local.imu_attention,
        model.local.imu_reliability,
        model.local.correspondence,
        model.evidence_encoder,
        model.evidence_norm,
        model.residual_projection,
    ):
        module.train()


def train_adapter(
    model: P101F1CoarseAnchoredTeacher,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    config: dict[str, Any],
    epochs: int,
) -> tuple[list[dict[str, float]], list[str], list[str]]:
    training = config["training"]
    parameters, trainable, frozen = select_f1_trainable_parameters(model)
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
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        set_adapter_mode(model)
        sums = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "pair": 0.0, "residual": 0.0}
        correct = rows = 0
        started = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
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
                anchor_probability = F.softmax(output["anchor_logits"].detach() / temperature, dim=-1)
                kl_rows = F.kl_div(
                    F.log_softmax(output["logits"] / temperature, dim=-1),
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
            sums["residual"] += float(output["fine_residual_rms"].mean().detach()) * size
            correct += int((output["logits"].argmax(dim=1) == batch["label"]).sum())
            rows += size
        record = {
            "epoch": float(epoch + 1),
            **{key: value / max(rows, 1) for key, value in sums.items()},
            "accuracy": correct / max(rows, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    return history, trainable, frozen


@torch.no_grad()
def evaluate(
    model: P101F1CoarseAnchoredTeacher,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        for key, value in (
            ("rows", batch["row"]),
            ("logits", output["logits"]),
            ("anchor_logits", output["anchor_logits"]),
            ("uncertainty", output["uncertainty"]),
            ("fine_residual_rms", output["fine_residual_rms"]),
            ("raw_residual_rms", output["raw_residual_rms"]),
        ):
            collected.setdefault(key, []).append(value.float().cpu().numpy())
        for modality in ("skeleton", "imu"):
            attention = output[f"{modality}_attention"]
            signed, absolute, entropy = attention_audit(
                attention, batch["visual_time"], batch["motion_time"]
            )
            collected.setdefault(f"{modality}_signed_offset", []).append(signed)
            collected.setdefault(f"{modality}_absolute_offset", []).append(absolute)
            collected.setdefault(f"{modality}_attention_entropy", []).append(entropy)
        collected.setdefault("positive_correspondence", []).append(
            output["positive_correspondence_logits"]
            .mean(dim=(1, 2, 3, 4))
            .float()
            .cpu()
            .numpy()
        )
    return {key: np.concatenate(values) for key, values in collected.items()}


def counterfactual_dataset(
    coarse: Any,
    fine: Any,
    indices: np.ndarray,
    normalizer: FoldNormalizer,
    kind: str,
    seed: int,
) -> P101F1Dataset:
    zero: tuple[str, ...] = ()
    reverse: tuple[str, ...] = ()
    skeleton_source = imu_source = None
    if kind == "zero_skeleton":
        zero = ("skeleton",)
    elif kind == "local_reverse_skeleton":
        reverse = ("skeleton",)
    elif kind == "shuffle_skeleton":
        skeleton_source = within_subject_permutation(fine, indices, "skeleton", seed)
    elif kind == "zero_imu":
        zero = ("imu",)
    elif kind == "reverse_imu":
        reverse = ("imu",)
    elif kind == "shuffle_imu":
        imu_source = within_subject_permutation(fine, indices, "imu", seed)
    elif kind == "zero_both":
        zero = ("skeleton", "imu")
    else:
        raise ValueError(kind)
    return P101F1Dataset(
        coarse,
        fine,
        indices,
        normalizer,
        skeleton_source=skeleton_source,
        imu_source=imu_source,
        zero_modalities=zero,
        reverse_modalities=reverse,
    )


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
            raise RuntimeError("P101-F1 counterfactual row order changed")
        arrays[f"{kind}_logits"] = result["logits"]
    if not np.array_equal(arrays["zero_imu_logits"], arrays["direct_anchor_logits"]):
        raise RuntimeError("P101-F1 zero IMU did not exactly return the coarse anchor")
    return arrays


@torch.no_grad()
def exact_initial_error(
    model: P101F1CoarseAnchoredTeacher,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    output = model(move_batch(batch, device))
    return float((output["logits"] - output["anchor_logits"]).abs().max())


def run_adapter_fold(
    coarse: Any,
    fine: Any,
    fold: int,
    nested_error: np.ndarray,
    nested_uncertainty: np.ndarray,
    config: dict[str, Any],
    output: Path,
    epochs: int,
    device: torch.device,
    resume: bool,
) -> dict[str, np.ndarray]:
    fold_output = output / "F1_VSI"
    fold_output.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_output / f"fold{fold}_final.pt"
    predictions = fold_output / f"fold{fold}_predictions.npz"
    if resume and checkpoint.exists() and predictions.exists():
        with np.load(predictions, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    train, held = coarse.indices_for_fold(fold)
    anchor, normalizer, anchor_audit = load_outer_anchor(fold, device)
    model = P101F1CoarseAnchoredTeacher(anchor, f1_config(config))
    f0_path = F0_RUN / "VSI" / f"fold{fold}_final.pt"
    f0_payload = torch.load(f0_path, map_location="cpu", weights_only=False)
    local_audit = model.local.load_f0_state(f0_payload["state_dict"])
    model = model.to(device)
    seed = int(config["training"]["seed"]) + 4000 + fold
    initial_dataset = P101F1Dataset(coarse, fine, train[:2], normalizer)
    initial_batch = next(iter(make_loader(initial_dataset, 2, False, seed)))
    initial_error = exact_initial_error(model, initial_batch, device)
    if initial_error > 1e-7:
        raise RuntimeError(f"P101-F1 did not exactly initialize from P100 VS: {initial_error}")
    weights = class_user_sample_weights(fine, train)
    negatives = within_subject_wrong_label_source(fine, train, seed)
    dataset = P101F1Dataset(
        coarse,
        fine,
        train,
        normalizer,
        sample_weights=weights,
        nested_anchor_error=nested_error,
        nested_anchor_uncertainty=nested_uncertainty,
        negative_imu_source=negatives,
    )
    print(
        json.dumps(
            {
                "stage": "f1_adapter",
                "fold": fold,
                "nested_coarse_errors": int(nested_error[train].sum()),
                "nested_coarse_error_rate": float(nested_error[train].mean()),
                "exact_anchor_max_abs_error": initial_error,
                "epochs": epochs,
            }
        ),
        flush=True,
    )
    history, trainable, frozen = train_adapter(
        model,
        make_loader(dataset, int(config["training"]["batch_size"]), True, seed),
        device,
        config,
        epochs,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "f1_config": model.config.__dict__,
            "history": history,
            "held_users": FOLD_USERS[fold],
            "anchor": anchor_audit,
            "f0_local_source": str(f0_path),
            "local_initialization": local_audit,
            "exact_anchor_max_abs_error": initial_error,
            "nested_coarse_error_rows": int(nested_error[train].sum()),
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "outer_label_used_for_selection": False,
        },
        checkpoint,
    )
    arrays = evaluate_with_counterfactuals(
        model, coarse, fine, held, normalizer, config, seed, device
    )
    np.savez_compressed(predictions, **arrays)
    return arrays


def load_fold_predictions(output: Path) -> list[dict[str, np.ndarray]]:
    values: list[dict[str, np.ndarray]] = []
    for fold in range(4):
        path = output / "F1_VSI" / f"fold{fold}_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            values.append({key: np.asarray(archive[key]) for key in archive.files})
    return values


def summarize_candidate(fine: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    probability = arrays["direct_probability"]
    counters = {
        name.removesuffix("_logits"): classification_metrics(
            softmax_numpy(values), fine.labels, fine.users
        )
        for name, values in arrays.items()
        if name.endswith("_logits") and name not in {"direct_logits", "direct_anchor_logits"}
    }
    mechanism = {
        name.removeprefix("direct_"): {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for name, values in arrays.items()
        if name.startswith("direct_") and not name.endswith("logits")
    }
    return {
        "direct": classification_metrics(probability, fine.labels, fine.users),
        "counterfactuals": counters,
        "mechanism": mechanism,
    }


def save_complete_oof(
    fine: Any, arrays: dict[str, np.ndarray], output: Path
) -> None:
    np.savez_compressed(
        output / "F1_VSI_complete_oof.npz",
        sample_ids=fine.sample_ids,
        users=fine.users,
        fold_ids=fine.fold_ids,
        **arrays,
    )


def aggregate(
    fine: Any, fold_outputs: list[dict[str, np.ndarray]], output: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    full: dict[str, np.ndarray] = {}
    for fold_output in fold_outputs:
        rows = fold_output["rows"].astype(np.int64)
        for name, values in fold_output.items():
            if name == "rows":
                continue
            shape = (len(fine.sample_ids),) + values.shape[1:]
            full.setdefault(name, np.full(shape, np.nan, dtype=np.float32))[rows] = values
    if not np.isfinite(full["direct_logits"]).all():
        raise RuntimeError("P101-F1 OOF is incomplete")
    probability = softmax_numpy(full["direct_logits"])
    arrays = {**full, "direct_probability": probability}
    save_complete_oof(fine, arrays, output)
    return summarize_candidate(fine, arrays), arrays


def verify_frozen_anchor_states(output: Path) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold in range(4):
        canonical = torch.load(
            P100_RUN / "VS" / f"fold{fold}_final.pt",
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        candidate = torch.load(
            output / "F1_VSI" / f"fold{fold}_final.pt",
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        missing = [name for name in canonical if f"anchor.{name}" not in candidate]
        changed = [
            name
            for name, tensor in canonical.items()
            if f"anchor.{name}" in candidate
            and not torch.equal(candidate[f"anchor.{name}"], tensor)
        ]
        if missing or changed:
            raise RuntimeError(
                f"P101-F1 frozen anchor state differs in fold {fold}: "
                f"missing={missing[:3]} changed={changed[:3]}"
            )
        folds.append(
            {
                "fold": fold,
                "tensor_count": len(canonical),
                "all_tensors_bitwise_equal": True,
            }
        )
    return {"all_folds_bitwise_equal": True, "folds": folds}


def rebase_to_canonical_anchor(
    arrays: dict[str, np.ndarray], canonical_logits: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Remove harmless GPU re-evaluation drift from matched-anchor comparisons.

    The frozen anchor is evaluated inside the larger F1 graph.  CUDA GEMM/attention
    kernels can differ by roughly 1e-3 logits across processes even when every
    checkpoint tensor is bitwise identical.  Preserve the learned pre-classifier
    logit delta while placing it on the already-published P100 OOF anchor.
    """

    values = {name: np.asarray(value).copy() for name, value in arrays.items()}
    reconstructed = values["direct_anchor_logits"]
    reconstructed_probability = softmax_numpy(reconstructed)
    canonical_probability = softmax_numpy(canonical_logits)
    drift = np.abs(reconstructed_probability - canonical_probability)
    if float(drift.max()) > 1e-3:
        raise RuntimeError(
            "P101-F1 reconstructed anchor exceeds numerical-drift contract: "
            f"max_probability_difference={float(drift.max())}"
        )
    anchor_preserving = (
        "direct_logits",
        "local_reverse_skeleton_logits",
        "zero_imu_logits",
        "reverse_imu_logits",
        "shuffle_imu_logits",
    )
    for name in anchor_preserving:
        values[name] = canonical_logits + (values[name] - reconstructed)
    values["direct_anchor_logits"] = canonical_logits.copy()
    values["direct_probability"] = softmax_numpy(values["direct_logits"])
    return values, {
        "method": "canonical P100 OOF logits plus learned F1 logit delta",
        "max_probability_drift_before_rebase": float(drift.max()),
        "mean_probability_drift_before_rebase": float(drift.mean()),
        "argmax_rows_different_before_rebase": int(
            (reconstructed.argmax(axis=1) != canonical_logits.argmax(axis=1)).sum()
        ),
        "anchor_preserving_outputs_rebased": list(anchor_preserving),
    }


def top5_correct(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    top = np.argpartition(probability, -5, axis=1)[:, -5:]
    return np.any(top == labels[:, None], axis=1)


def gate_result(
    candidate: dict[str, Any],
    candidate_probability: np.ndarray,
    anchor_probability: np.ndarray,
    comparison: dict[str, Any],
    fine: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    subject = list(comparison["per_subject"].values())
    top5_delta = int(
        top5_correct(candidate_probability, fine.labels).sum()
        - top5_correct(anchor_probability, fine.labels).sum()
    )
    counters = candidate["counterfactuals"]
    direct = candidate["direct"]
    aligned = (
        direct["top1"] > counters["zero_imu"]["top1"]
        and direct["top1"] > counters["shuffle_imu"]["top1"]
        and direct["top1"] >= counters["reverse_imu"]["top1"]
        and direct["top5"] >= counters["reverse_imu"]["top5"]
    )
    stability = {
        "subjects_nonnegative_at_least_8": sum(value["net"] >= 0 for value in subject)
        >= int(gate["min_nonnegative_subjects"]),
        "subjects_positive_at_least_4": sum(value["net"] > 0 for value in subject)
        >= int(gate["min_positive_subjects"]),
        "worst_subject_drop_within_2pp": min(value["accuracy_pp"] for value in subject)
        >= -float(gate["max_subject_drop_pp"]),
    }
    top1 = {
        "matched_net_at_least_10": comparison["net"] >= int(gate["candidate_minus_anchor_min_net"]),
        "mcnemar_p_at_most_0_10": comparison["mcnemar_exact_p"] <= float(gate["mcnemar_max_p"]),
        "top5_not_lower": direct["top5"] >= topk_accuracy(anchor_probability, fine.labels, 5),
        "above_coarse_anchor": direct["top1"] > topk_accuracy(anchor_probability, fine.labels, 1),
        "aligned_imu_over_counterfactuals": aligned,
        **stability,
    }
    anchor_metrics = classification_metrics(anchor_probability, fine.labels, fine.users)
    top5 = {
        "top5_delta_rows_at_least_10": top5_delta >= int(gate["top5_only_min_rows"]),
        "top1_drop_at_most_2_rows": comparison["net"] >= -int(gate["top5_only_max_top1_drop_rows"]),
        "macro_f1_drop_within_0_2pp": direct["macro_f1"]
        >= anchor_metrics["macro_f1"] - float(gate["top5_only_max_macro_f1_drop"]),
        "aligned_imu_over_counterfactuals": aligned,
        **stability,
    }
    return {
        "top1_path": {"checks": top1, "passed": bool(all(top1.values()))},
        "top5_only_path": {"checks": top5, "passed": bool(all(top5.values()))},
        "top5_delta_rows": top5_delta,
        "passed": bool(all(top1.values()) or all(top5.values())),
    }


def finalize(fine: Any, config: dict[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    candidate, arrays = aggregate(fine, load_fold_predictions(output), output)
    anchor_state_audit = verify_frozen_anchor_states(output)
    with np.load(P100_RUN / "VS_complete_oof.npz", allow_pickle=False) as archive:
        if not np.array_equal(np.asarray(archive["sample_ids"]).astype(str), fine.sample_ids):
            raise RuntimeError("P101-F1 anchor OOF row order changed")
        anchor_logits = np.asarray(archive["direct_logits"], dtype=np.float32)
        anchor_probability = np.asarray(archive["direct_probability"], dtype=np.float32)
    arrays, anchor_rebase_audit = rebase_to_canonical_anchor(arrays, anchor_logits)
    candidate = summarize_candidate(fine, arrays)
    save_complete_oof(fine, arrays, output)
    if not np.array_equal(arrays["direct_anchor_logits"], anchor_logits):
        raise RuntimeError("P101-F1 canonical anchor rebase failed")
    if not np.allclose(
        softmax_numpy(arrays["direct_anchor_logits"]),
        anchor_probability,
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError("P101-F1 canonical anchor probability changed")
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
        "protocol": "P101-F1 frozen P100 coarse VS plus fine local VSI pre-classifier residual; fixed four-fold OOF",
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
            "nested_coarse_vs_subject_disjoint": True,
            "outer_label_used_for_checkpoint_selection": False,
            "zero_imu_exact_anchor": True,
            "historical_40class_logits_used_as_model_input": False,
            "b_teacher_started": False,
            "frozen_anchor_state": anchor_state_audit,
            "canonical_anchor_rebase": anchor_rebase_audit,
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
    folds = list(range(4)) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    if not set(stages) <= {"nested", "adapter", "finalize"}:
        raise ValueError(f"unknown P101-F1 stages: {stages}")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    coarse = load_p100a_data()
    fine = load_p101_data()
    if not np.array_equal(coarse.sample_ids, fine.sample_ids):
        raise RuntimeError("P101-F1 coarse/fine contract differs")
    nested_epochs = 1 if args.smoke else int(config["training"]["nested_epochs"])
    adapter_epochs = 1 if args.smoke else int(config["training"]["adapter_epochs"])
    manifest = {
        "status": "smoke" if args.smoke else "formal_started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": str(config_path),
        "config_sha256": config_hash,
        "stages": stages,
        "folds": folds,
        "epochs": {"nested": nested_epochs, "adapter": adapter_epochs},
        "device": str(device),
        "rows": len(fine.sample_ids),
        "subjects": sorted(set(fine.users.tolist())),
        "h3_rows": 0,
        "b_teacher_started": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    nested: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if "nested" in stages or "adapter" in stages:
        for fold in folds:
            nested[fold] = nested_coarse_errors(
                coarse, fold, config, output, nested_epochs, device, args.resume
            )
    if "adapter" in stages:
        for fold in folds:
            run_adapter_fold(
                coarse,
                fine,
                fold,
                nested[fold][0],
                nested[fold][1],
                config,
                output,
                adapter_epochs,
                device,
                args.resume,
            )
    if "finalize" in stages or (not args.smoke and folds == list(range(4))):
        result = finalize(fine, config, config_hash, output)
        print(json.dumps(result["a_teacher_positive_gate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
