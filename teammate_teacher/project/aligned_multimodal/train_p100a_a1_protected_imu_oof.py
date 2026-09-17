"""Train the single frozen P100-A1 protected IMU structural revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from p100a_global_teacher_data import (
    CANONICAL_VARIANTS,
    DEV_USER_SET,
    FOLD_USERS,
    FoldNormalizer,
    P100AData,
    P100ADataset,
    class_user_sample_weights,
    load_p100a_data,
    write_contract,
)
from p100a_global_teacher_model import P100AGlobalTeacher, P100AModelConfig
from train_p100a_global_teacher_oof import (
    classification_metrics,
    confusion_change_groups,
    cosine_with_warmup,
    counterfactual_dataset,
    evaluate_model,
    git_commit,
    make_loader,
    move_batch,
    paired_comparison,
    set_seed,
    softmax_numpy,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p100a_a1_protected_imu.json"
DEFAULT_OUTPUT = HERE / "runs/p100a_a1_protected_imu_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def source_run_path(config: dict[str, Any]) -> Path:
    value = Path(config["source_run"])
    return value if value.is_absolute() else PROJECT / value


def load_source_checkpoint(
    source_run: Path, fold: int, device: torch.device
) -> tuple[dict[str, Any], FoldNormalizer]:
    path = source_run / "VS" / f"fold{fold}_final.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing source-safe VS checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if tuple(checkpoint["held_users"]) != tuple(FOLD_USERS[fold]):
        raise RuntimeError("VS source checkpoint held-user contract changed")
    normalizer = FoldNormalizer(
        means=checkpoint["normalizer_means"],
        stds=checkpoint["normalizer_stds"],
    )
    return checkpoint, normalizer


def build_anchor_and_candidate(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[P100AGlobalTeacher, P100AGlobalTeacher, list[str], list[str]]:
    anchor_config = P100AModelConfig(**checkpoint["model_config"])
    if anchor_config.modalities != CANONICAL_VARIANTS["VS"]:
        raise RuntimeError("A1 must start from a VS checkpoint")
    anchor = P100AGlobalTeacher(anchor_config).to(device)
    anchor.load_state_dict(checkpoint["state_dict"], strict=True)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)

    candidate_config = P100AModelConfig(
        modalities=CANONICAL_VARIANTS["VSI"], **config["model"]
    )
    candidate = P100AGlobalTeacher(candidate_config).to(device)
    incompatible = candidate.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected source keys: {sorted(incompatible.unexpected_keys)}"
        )
    allowed_missing_fragments = (
        "imu_encoder.",
        "cross_statistics.",
        ".evidence.imu.",
        ".evidence.cross.",
        ".reliability.imu.",
        ".reliability.cross.",
        ".protected_scale.imu",
        ".protected_scale.cross",
    )
    illegal_missing = [
        name
        for name in incompatible.missing_keys
        if not any(fragment in name for fragment in allowed_missing_fragments)
    ]
    if illegal_missing:
        raise RuntimeError(f"non-adapter state missing from VS anchor: {illegal_missing}")

    trainable_fragments = allowed_missing_fragments
    trainable_names: list[str] = []
    for name, parameter in candidate.named_parameters():
        trainable = any(fragment in name for fragment in trainable_fragments)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("protected adapter has no trainable parameters")
    frozen_names = [
        name for name, parameter in candidate.named_parameters() if not parameter.requires_grad
    ]
    return anchor, candidate, trainable_names, frozen_names


def set_adapter_train_mode(model: P100AGlobalTeacher) -> None:
    # Preserve the frozen VS computation exactly in inference mode. Only the
    # newly introduced IMU/cross modules use training-time dropout.
    model.eval()
    assert model.imu_encoder is not None and model.cross_statistics is not None
    model.imu_encoder.train()
    model.cross_statistics.train()
    for block in model.blocks:
        for name in ("imu", "cross"):
            block.evidence[name].train()
            block.reliability[name].train()


@torch.no_grad()
def exact_anchor_error(
    anchor: P100AGlobalTeacher,
    candidate: P100AGlobalTeacher,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    anchor.eval()
    candidate.eval()
    moved = move_batch(batch, device)
    anchor_logits = anchor(moved)["logits"]
    candidate_logits = candidate(moved)["logits"]
    return float((anchor_logits - candidate_logits).abs().max().cpu())


def train_adapter(
    anchor: P100AGlobalTeacher,
    candidate: P100AGlobalTeacher,
    loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    training: dict[str, Any],
    epochs: int,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.98),
    )
    total_steps = epochs * len(loader)
    warmup_steps = int(total_steps * float(training["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_with_warmup(step, total_steps, warmup_steps),
    )
    use_amp = device.type == "cuda" and bool(training.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    temperature = float(training["protection_temperature"])
    protection_weight = float(training["protection_weight"])
    history: list[dict[str, float]] = []
    anchor.eval()
    for epoch in range(epochs):
        set_adapter_train_mode(candidate)
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        correct = 0
        rows = 0
        started = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                anchor_logits = anchor(batch)["logits"]
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = candidate(batch)["logits"]
                ce_rows = F.cross_entropy(
                    logits,
                    batch["label"],
                    reduction="none",
                    label_smoothing=float(training["label_smoothing"]),
                )
                kl_rows = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(anchor_logits / temperature, dim=1),
                    reduction="none",
                ).sum(dim=1) * temperature**2
                anchor_correct = anchor_logits.argmax(dim=1) == batch["label"]
                boundary_weight = torch.where(
                    anchor_correct,
                    torch.full_like(kl_rows, float(training["anchor_correct_weight"])),
                    torch.full_like(kl_rows, float(training["anchor_wrong_weight"])),
                )
                ce = (ce_rows * batch["weight"]).mean()
                protected_kl = (kl_rows * boundary_weight * batch["weight"]).mean()
                loss = ce + protection_weight * protected_kl
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_rows = len(batch["label"])
            total_loss += float(loss.detach()) * batch_rows
            total_ce += float(ce.detach()) * batch_rows
            total_kl += float(protected_kl.detach()) * batch_rows
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            rows += batch_rows
        scale_values = [
            float(
                (
                    block.protected_imu_max_scale
                    * torch.tanh(block.protected_scale[name].detach())
                ).cpu()
            )
            for block in candidate.blocks
            for name in ("imu", "cross")
        ]
        epoch_log = {
            "epoch": float(epoch + 1),
            "train_loss": total_loss / rows,
            "train_ce": total_ce / rows,
            "protected_kl": total_kl / rows,
            "train_accuracy": correct / rows,
            "mean_signed_adapter_scale": float(np.mean(scale_values)),
            "max_abs_adapter_scale": float(np.max(np.abs(scale_values))),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_log)
        print(json.dumps(epoch_log), flush=True)
    candidate.eval()
    return history


def run_fold(
    data: P100AData,
    fold: int,
    config: dict[str, Any],
    output: Path,
    epochs: int,
    device: torch.device,
    resume: bool,
) -> dict[str, np.ndarray]:
    prediction_path = output / f"fold{fold}_predictions.npz"
    checkpoint_path = output / f"fold{fold}_final.pt"
    if resume and prediction_path.exists():
        print(f"resume A1 fold {fold}", flush=True)
        with np.load(prediction_path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}

    source_run = source_run_path(config)
    source, normalizer = load_source_checkpoint(source_run, fold, device)
    seed = int(config["training"]["seed"]) + fold
    set_seed(seed)
    anchor, candidate, trainable_names, frozen_names = build_anchor_and_candidate(
        source, config, device
    )
    train_indices, held_indices = data.indices_for_fold(fold)
    weights = class_user_sample_weights(data, train_indices)
    train_loader = make_loader(
        P100ADataset(
            data,
            train_indices,
            normalizer,
            CANONICAL_VARIANTS["VSI"],
            sample_weights=weights,
        ),
        int(config["training"]["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    verification_loader = make_loader(
        P100ADataset(
            data, train_indices[:8], normalizer, CANONICAL_VARIANTS["VSI"]
        ),
        8,
        shuffle=False,
        seed=seed,
    )
    anchor_error = exact_anchor_error(
        anchor, candidate, next(iter(verification_loader)), device
    )
    if anchor_error > 1e-5:
        raise RuntimeError(f"protected zero point is not the VS anchor: {anchor_error}")
    print(
        json.dumps(
            {
                "fold": fold,
                "held_users": FOLD_USERS[fold],
                "train_rows": len(train_indices),
                "held_rows": len(held_indices),
                "parameters": candidate.parameter_count,
                "trainable_parameters": int(
                    sum(p.numel() for p in candidate.parameters() if p.requires_grad)
                ),
                "frozen_parameters": int(
                    sum(p.numel() for p in candidate.parameters() if not p.requires_grad)
                ),
                "exact_anchor_max_abs_logit_error": anchor_error,
                "epochs": epochs,
            }
        ),
        flush=True,
    )
    history = train_adapter(
        anchor, candidate, train_loader, device, config["training"], epochs
    )
    torch.save(
        {
            "state_dict": candidate.state_dict(),
            "model_config": candidate.config.__dict__,
            "normalizer_means": normalizer.means,
            "normalizer_stds": normalizer.stds,
            "epochs": epochs,
            "history": history,
            "held_users": FOLD_USERS[fold],
            "source_vs_checkpoint": str(source_run / "VS" / f"fold{fold}_final.pt"),
            "exact_anchor_max_abs_logit_error": anchor_error,
            "trainable_names": trainable_names,
            "frozen_names": frozen_names,
        },
        checkpoint_path,
    )

    eval_batch_size = int(config["training"]["eval_batch_size"])
    direct = evaluate_model(
        candidate,
        make_loader(
            P100ADataset(
                data, held_indices, normalizer, CANONICAL_VARIANTS["VSI"]
            ),
            eval_batch_size,
            shuffle=False,
            seed=seed,
        ),
        device,
    )
    arrays: dict[str, np.ndarray] = {
        "rows": direct["rows"],
        "direct_logits": direct["logits"],
    }
    for group, prefix in (
        (direct["reliability"], "direct_reliability_"),
        (direct["residual_norm"], "direct_residual_norm_"),
        (direct["evidence_scale"], "direct_evidence_scale_"),
    ):
        for name, values in group.items():
            arrays[f"{prefix}{name}"] = values
    for index, kind in enumerate(
        (
            "zero_skeleton",
            "shuffle_skeleton",
            "zero_imu",
            "shuffle_imu",
            "zero_both",
        )
    ):
        counter = evaluate_model(
            candidate,
            make_loader(
                counterfactual_dataset(
                    data,
                    held_indices,
                    normalizer,
                    CANONICAL_VARIANTS["VSI"],
                    kind,
                    seed + 1000 + index,
                ),
                eval_batch_size,
                shuffle=False,
                seed=seed,
            ),
            device,
        )
        if not np.array_equal(counter["rows"], direct["rows"]):
            raise RuntimeError("counterfactual row order changed")
        arrays[f"{kind}_logits"] = counter["logits"]
    np.savez_compressed(prediction_path, **arrays)
    return arrays


def load_source_oof(source_run: Path, name: str, data: P100AData) -> np.ndarray:
    path = source_run / f"{name}_complete_oof.npz"
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["sample_ids"].astype(str), data.sample_ids.astype(str)):
            raise RuntimeError(f"source OOF row order changed: {name}")
        return np.asarray(archive["direct_probability"], dtype=np.float32)


def aggregate(
    data: P100AData,
    fold_outputs: list[dict[str, np.ndarray]],
    output: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    direct_logits = np.full((len(data.sample_ids), 40), np.nan, dtype=np.float32)
    counter_logits: dict[str, np.ndarray] = {}
    diagnostics: dict[str, np.ndarray] = {}
    for fold_output in fold_outputs:
        rows = fold_output["rows"].astype(np.int64)
        direct_logits[rows] = fold_output["direct_logits"]
        for name, values in fold_output.items():
            if name.endswith("_logits") and name != "direct_logits":
                counter_logits.setdefault(
                    name, np.full_like(direct_logits, np.nan)
                )[rows] = values
            elif name.startswith("direct_") and name != "direct_logits":
                diagnostics.setdefault(
                    name, np.full(len(data.sample_ids), np.nan, dtype=np.float32)
                )[rows] = values
    if not np.isfinite(direct_logits).all():
        raise RuntimeError("A1 does not cover every OOF row")
    direct_probability = softmax_numpy(direct_logits)
    np.savez_compressed(
        output / "A1_complete_oof.npz",
        sample_ids=data.sample_ids,
        users=data.users,
        fold_ids=data.fold_ids,
        direct_logits=direct_logits,
        direct_probability=direct_probability,
        **counter_logits,
        **diagnostics,
    )
    summary = {
        "direct": classification_metrics(direct_probability, data.labels, data.users),
        "counterfactuals": {
            name.removesuffix("_logits"): classification_metrics(
                softmax_numpy(values), data.labels, data.users
            )
            for name, values in counter_logits.items()
        },
        "diagnostics": {
            name.removeprefix("direct_"): {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in diagnostics.items()
        },
    }
    return summary, direct_probability


def positive_gate(
    a1: dict[str, Any],
    versus_v: dict[str, Any],
    versus_vs: dict[str, Any],
) -> dict[str, Any]:
    subject_values = list(versus_v["per_subject"].values())
    direct = a1["direct"]
    checks = {
        "top1_above_visual": versus_v["net"] > 0,
        "top5_above_visual": versus_v["top5_difference_pp"] > 0,
        "top1_above_vs": versus_vs["net"] > 0,
        "top5_above_vs": versus_vs["top5_difference_pp"] > 0,
        "a1_vs_vs_net_at_least_10": versus_vs["net"] >= 10,
        "a1_vs_vs_mcnemar_p_at_most_0_10": versus_vs["mcnemar_exact_p"] <= 0.10,
        "net_vs_visual_at_least_10": versus_v["net"] >= 10,
        "subjects_vs_visual_nonnegative_at_least_8": sum(
            value["net"] >= 0 for value in subject_values
        )
        >= 8,
        "subjects_vs_visual_positive_at_least_4": sum(
            value["net"] > 0 for value in subject_values
        )
        >= 4,
        "worst_subject_vs_visual_drop_within_2pp": min(
            value["accuracy_pp"] for value in subject_values
        )
        >= -2.0,
        "mcnemar_vs_visual_p_at_most_0_10": versus_v["mcnemar_exact_p"] <= 0.10,
        "aligned_imu_beats_shuffle": direct["top1"]
        > a1["counterfactuals"]["shuffle_imu"]["top1"],
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "strong_confirmation": bool(
            all(checks.values())
            and (
                versus_vs["mcnemar_exact_p"] <= 0.05
                or versus_vs["subject_bootstrap"]["lower_95"] > 0
            )
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    epochs = int(args.epochs or config["training"]["epochs"])
    if args.smoke:
        epochs = 1
    folds = (
        list(range(4))
        if args.folds == "all"
        else [int(value) for value in args.folds.split(",")]
    )
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    data = load_p100a_data()
    write_contract(output / "data_contract.json", data)
    manifest = {
        "status": "smoke" if args.smoke else "formal_started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "folds": folds,
        "epochs": epochs,
        "device": str(device),
        "data": data.summary(),
        "source_run": str(source_run_path(config)),
        "source_variant": "VS",
        "outer_label_used_for_checkpoint_selection": False,
        "h3_code_path_present": False,
        "b_teacher_code_path_present": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fold_outputs = [
        run_fold(data, fold, config, output, epochs, device, args.resume)
        for fold in folds
    ]
    if folds != list(range(4)):
        return

    a1_summary, a1_probability = aggregate(data, fold_outputs, output)
    source_run = source_run_path(config)
    visual_probability = load_source_oof(source_run, "V", data)
    vs_probability = load_source_oof(source_run, "VS", data)
    a0_vsi_probability = load_source_oof(source_run, "VSI", data)
    comparisons = {
        "A1_minus_V": paired_comparison(
            a1_probability, visual_probability, data.labels, data.users
        ),
        "A1_minus_VS_imu": paired_comparison(
            a1_probability, vs_probability, data.labels, data.users
        ),
        "A1_minus_A0_VSI": paired_comparison(
            a1_probability, a0_vsi_probability, data.labels, data.users
        ),
    }
    comparisons["A1_minus_V"]["confusion_changes"] = confusion_change_groups(
        a1_probability, visual_probability, data.labels
    )
    comparisons["A1_minus_VS_imu"]["confusion_changes"] = confusion_change_groups(
        a1_probability, vs_probability, data.labels
    )
    gate = positive_gate(
        a1_summary,
        comparisons["A1_minus_V"],
        comparisons["A1_minus_VS_imu"],
    )
    final_summary = {
        "status": "complete",
        "protocol": "P100-A1 fixed 4-fold / 12-subject OOF; protected source-safe VS anchor; fixed epoch",
        "git_commit": git_commit(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "epochs": epochs,
        "data": data.summary(),
        "A1": a1_summary,
        "comparisons": comparisons,
        "a_teacher_positive_gate": gate,
        "leakage_audit": {
            "development_allow_list_exact": set(data.users) == DEV_USER_SET,
            "h3_rows_loaded": 0,
            "historical_40class_expert_probability_loaded": False,
            "outer_subject_disjoint": True,
            "source_vs_checkpoint_is_fold_matched": True,
            "fixed_epoch_without_outer_label_selection": True,
            "student_started": False,
            "b_teacher_started": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
