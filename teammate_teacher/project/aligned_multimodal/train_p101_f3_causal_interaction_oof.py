"""Train P101-F3 causal-centered interaction on source-safe nested anchors."""

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

from p100a_global_teacher_data import FOLD_USERS, load_p100a_data
from p101_f1_coarse_anchor_data import P101F1Dataset
from p101_f1_coarse_anchor_model import P101F1Config, select_f1_trainable_parameters
from p101_f3_causal_interaction_model import P101F3CausalInteractionTeacher
from p101_finegrained_teacher_data import (
    class_user_sample_weights,
    load_p101_data,
    within_subject_wrong_label_source,
)
from train_p100a_global_teacher_oof import (
    classification_metrics,
    confusion_change_groups,
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
from train_p101_f2_source_safe_adapter_oof import (
    allocate_group_batch_counts,
    load_nested_anchor,
    load_non_anchor_state,
    make_epoch_loader,
    matched_nested_arrays,
    non_anchor_state,
    source_safe_inner_folds,
    verify_nested_prediction,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p101_f3_causal_interaction.json"
DEFAULT_OUTPUT = HERE / "runs/p101_f3_causal_interaction_oof_v1"
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


def f3_config(config: dict[str, Any]) -> P101F1Config:
    return P101F1Config(**config["model"])


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return max((step + 1) / max(warmup, 1), 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_source_safe_adapter(
    model: P101F3CausalInteractionTeacher,
    anchors: dict[int, Any],
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
        sums = {
            "loss": 0.0,
            "ce": 0.0,
            "kl": 0.0,
            "pair": 0.0,
            "negative_kl": 0.0,
            "residual": 0.0,
            "negative_residual": 0.0,
            "pair_gate": 0.0,
            "negative_pair_gate": 0.0,
        }
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
                    negative_kl = (
                        F.kl_div(
                            F.log_softmax(
                                output["negative_logits"] / temperature, dim=-1
                            ),
                            anchor_probability,
                            reduction="batchmean",
                        )
                        * temperature**2
                    )
                    loss = (
                        ce
                        + kl
                        + float(training["correspondence_weight"]) * pair
                        + float(training["negative_imu_anchor_kl_weight"])
                        * negative_kl
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                size = len(batch["label"])
                for key, value in (
                    ("loss", loss),
                    ("ce", ce),
                    ("kl", kl),
                    ("pair", pair),
                    ("negative_kl", negative_kl),
                ):
                    sums[key] += float(value.detach()) * size
                for key, value in (
                    ("residual", output["fine_residual_rms"]),
                    ("negative_residual", output["negative_fine_residual_rms"]),
                    ("pair_gate", output["pair_gate"]),
                    ("negative_pair_gate", output["negative_pair_gate"]),
                ):
                    sums[key] += float(value.mean().detach()) * size
                batch_correct = int((output["logits"].argmax(dim=1) == batch["label"]).sum())
                correct += batch_correct
                rows_seen += size
                steps += 1
                per_anchor[fold]["rows"] += size
                per_anchor[fold]["correct"] += batch_correct
        if steps != steps_per_epoch or rows_seen != sum(len(value) for value in datasets.values()):
            raise RuntimeError("P101-F3 source-safe epoch coverage changed")
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
    fold_output = output / "F3_VSI"
    fold_output.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_output / f"fold{fold}_final.pt"
    predictions = fold_output / f"fold{fold}_predictions.npz"
    if resume and checkpoint.exists() and predictions.exists():
        with np.load(predictions, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    train, held = coarse.indices_for_fold(fold)
    nested_error, nested_uncertainty = matched_nested_arrays(fold, train)
    seed = int(config["training"]["seed"]) + 6000 + fold
    set_seed(seed)
    anchors: dict[int, Any] = {}
    normalizers: dict[int, Any] = {}
    source_audits: dict[int, dict[str, Any]] = {}
    group_rows: dict[int, np.ndarray] = {}
    for inner_fold in source_safe_inner_folds(fold):
        anchor, normalizer, contract = load_nested_anchor(fold, inner_fold, device)
        anchor.requires_grad_(False).eval()
        rows = train[np.isin(coarse.users[train], np.asarray(FOLD_USERS[inner_fold]))]
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
    model = P101F3CausalInteractionTeacher(anchors[first_fold], f3_config(config))
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
        raise RuntimeError(f"P101-F3 exact-zero nested initialization failed: {exact}")
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
    sizes = [len(datasets[value]) for value in source_safe_inner_folds(fold)]
    counts = allocate_group_batch_counts(sizes, int(config["training"]["batch_size"]))
    batch_counts = dict(zip(source_safe_inner_folds(fold), counts))
    print(
        json.dumps(
            {
                "stage": "f3_causal_interaction",
                "fold": fold,
                "source_safe_error_rows": int(nested_error[train].sum()),
                "source_safe_error_rate": float(nested_error[train].mean()),
                "exact_nested_anchor_errors": exact,
                "optimizer_steps_per_epoch": sum(counts),
                "matched_f2_steps_per_epoch": math.ceil(
                    len(train) / int(config["training"]["batch_size"])
                ),
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
    inference_model = P101F3CausalInteractionTeacher(
        outer_anchor, f3_config(config)
    ).to(device)
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
        raise RuntimeError("P101-F3 deployed zero-IMU path is not exact outer anchor")
    torch.save(
        {
            "adapter_state": adapter_state,
            "f3_config": inference_model.config.__dict__,
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
    from train_p101_f2_source_safe_adapter_oof import evaluate_with_counterfactuals

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
    outputs: list[dict[str, np.ndarray]] = []
    for fold in range(4):
        path = output / "F3_VSI" / f"fold{fold}_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            outputs.append({name: np.asarray(archive[name]) for name in archive.files})
    return outputs


def aggregate_arrays(fine: Any, outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    full: dict[str, np.ndarray] = {}
    for output in outputs:
        rows = output["rows"].astype(np.int64)
        for name, values in output.items():
            if name == "rows":
                continue
            shape = (len(fine.sample_ids),) + values.shape[1:]
            full.setdefault(name, np.full(shape, np.nan, dtype=np.float32))[rows] = values
    if not np.isfinite(full["direct_logits"]).all():
        raise RuntimeError("P101-F3 OOF is incomplete")
    full["direct_probability"] = softmax_numpy(full["direct_logits"])
    return full


def save_complete_oof(fine: Any, arrays: dict[str, np.ndarray], output: Path) -> None:
    np.savez_compressed(
        output / "F3_VSI_complete_oof.npz",
        sample_ids=fine.sample_ids,
        users=fine.users,
        fold_ids=fine.fold_ids,
        **arrays,
    )


def finalize(fine: Any, config: dict[str, Any], config_hash: str, output: Path) -> dict[str, Any]:
    arrays = aggregate_arrays(fine, load_fold_predictions(output))
    with np.load(P100_RUN / "VS_complete_oof.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["sample_ids"].astype(str), fine.sample_ids):
            raise RuntimeError("P101-F3 canonical anchor row order changed")
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
        "protocol": "P101-F3 source-safe causal-centered correspondence-gated fine VSI pre-classifier interaction",
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
            "nested_checkpoints_selected_by_held_metric": False,
            "outer_label_used_for_checkpoint_selection": False,
            "causal_vsi_minus_vs_centering": True,
            "correspondence_gate_in_classification_path": True,
            "negative_imu_logits_present_in_inference": False,
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
        raise ValueError(f"unknown P101-F3 stages: {stages}")
    folds = list(range(4)) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    coarse = load_p100a_data()
    fine = load_p101_data()
    if not np.array_equal(coarse.sample_ids, fine.sample_ids):
        raise RuntimeError("P101-F3 coarse/fine row contract differs")
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
