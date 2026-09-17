"""Train and audit the four frozen P100-A subject-disjoint OOF variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from torch.utils.data import DataLoader

from p100a_global_teacher_data import (
    CANONICAL_VARIANTS,
    DEV_USER_SET,
    FOLD_USERS,
    FoldNormalizer,
    P100AData,
    P100ADataset,
    class_user_sample_weights,
    load_p100a_data,
    within_subject_permutation,
    write_contract,
)
from p100a_global_teacher_model import P100AGlobalTeacher, P100AModelConfig


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p100a_a0.json"
DEFAULT_OUTPUT = HERE / "runs/p100a_a0_global_teacher_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--epochs", type=int)
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


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    probability = np.exp(values)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability.astype(np.float32)


def topk_accuracy(probability: np.ndarray, labels: np.ndarray, k: int) -> float:
    top = np.argpartition(probability, -k, axis=1)[:, -k:]
    return float(np.any(top == labels[:, None], axis=1).mean())


def classification_metrics(
    probability: np.ndarray, labels: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    prediction = probability.argmax(axis=1)
    output: dict[str, Any] = {
        "rows": int(len(labels)),
        "top1_correct": int((prediction == labels).sum()),
        "top1": float((prediction == labels).mean()),
        "top3": topk_accuracy(probability, labels, 3),
        "top5": topk_accuracy(probability, labels, 5),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "nll": float(log_loss(labels, probability, labels=np.arange(40))),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=np.arange(40)).tolist(),
        "per_subject": {},
    }
    for user in sorted(np.unique(users).tolist()):
        mask = users == user
        subject_probability = probability[mask]
        subject_labels = labels[mask]
        subject_prediction = prediction[mask]
        output["per_subject"][user] = {
            "rows": int(mask.sum()),
            "correct": int((subject_prediction == subject_labels).sum()),
            "top1": float((subject_prediction == subject_labels).mean()),
            "top5": topk_accuracy(subject_probability, subject_labels, 5),
            "macro_f1": float(
                f1_score(
                    subject_labels,
                    subject_prediction,
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    worst_user = min(output["per_subject"], key=lambda name: output["per_subject"][name]["top1"])
    output["worst_subject"] = {
        "user": worst_user,
        "top1": output["per_subject"][worst_user]["top1"],
    }
    return output


def subject_bootstrap_difference(
    candidate_correct: np.ndarray,
    control_correct: np.ndarray,
    users: np.ndarray,
    seed: int = 20260823,
    draws: int = 10000,
) -> dict[str, float]:
    unique_users = np.unique(users)
    per_user = np.asarray(
        [
            candidate_correct[users == user].mean()
            - control_correct[users == user].mean()
            for user in unique_users
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique_users), size=(draws, len(unique_users)))
    distribution = per_user[sampled].mean(axis=1)
    lower, median, upper = np.quantile(distribution, (0.025, 0.5, 0.975))
    return {
        "subject_mean_difference": float(per_user.mean()),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


def paired_comparison(
    candidate_probability: np.ndarray,
    control_probability: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
) -> dict[str, Any]:
    candidate = candidate_probability.argmax(axis=1)
    control = control_probability.argmax(axis=1)
    candidate_correct = candidate == labels
    control_correct = control == labels
    rescue = int((candidate_correct & ~control_correct).sum())
    harm = int((~candidate_correct & control_correct).sum())
    discordant = rescue + harm
    p_value = (
        float(binomtest(min(rescue, harm), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    per_subject: dict[str, Any] = {}
    for user in sorted(np.unique(users).tolist()):
        mask = users == user
        user_rescue = int((candidate_correct[mask] & ~control_correct[mask]).sum())
        user_harm = int((~candidate_correct[mask] & control_correct[mask]).sum())
        per_subject[user] = {
            "rows": int(mask.sum()),
            "rescue": user_rescue,
            "harm": user_harm,
            "net": user_rescue - user_harm,
            "accuracy_pp": float(
                100.0 * (candidate_correct[mask].mean() - control_correct[mask].mean())
            ),
        }
    return {
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "changed": int((candidate != control).sum()),
        "wrong_to_wrong": int((~candidate_correct & ~control_correct & (candidate != control)).sum()),
        "mcnemar_exact_p": p_value,
        "top5_difference_pp": float(
            100.0
            * (
                topk_accuracy(candidate_probability, labels, 5)
                - topk_accuracy(control_probability, labels, 5)
            )
        ),
        "per_subject": per_subject,
        "subject_bootstrap": subject_bootstrap_difference(
            candidate_correct, control_correct, users
        ),
    }


def confusion_change_groups(
    candidate_probability: np.ndarray,
    control_probability: np.ndarray,
    labels: np.ndarray,
    limit: int = 20,
) -> list[dict[str, int]]:
    candidate = candidate_probability.argmax(axis=1)
    control = control_probability.argmax(axis=1)
    counter: Counter[tuple[int, int, int]] = Counter()
    for truth, before, after in zip(labels, control, candidate):
        if before != after:
            counter[(int(truth), int(before), int(after))] += 1
    return [
        {"true": truth, "control_prediction": before, "candidate_prediction": after, "rows": rows}
        for (truth, before, after), rows in counter.most_common(limit)
    ]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def make_loader(
    dataset: P100ADataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
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


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-3)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_model(
    model: P100AGlobalTeacher,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    training: dict[str, Any],
    epochs: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        correct = 0
        rows = 0
        started = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(batch)
                losses = F.cross_entropy(
                    output["logits"],
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
            correct += int((output["logits"].argmax(dim=1) == batch["label"]).sum())
            rows += len(batch["label"])
        epoch_log = {
            "epoch": float(epoch + 1),
            "train_loss": loss_sum / max(rows, 1),
            "train_accuracy": correct / max(rows, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_log)
        print(json.dumps(epoch_log), flush=True)
    return history


@torch.no_grad()
def evaluate_model(
    model: P100AGlobalTeacher,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    rows: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    reliability: dict[str, list[np.ndarray]] = {}
    residual_norm: dict[str, list[np.ndarray]] = {}
    evidence_scale: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        rows.append(batch["row"].cpu().numpy())
        logits.append(output["logits"].float().cpu().numpy())
        for name, values in output["reliability"].items():
            reliability.setdefault(name, []).append(values.float().cpu().numpy())
        for name, values in output["evidence_residual_norm"].items():
            residual_norm.setdefault(name, []).append(values.float().cpu().numpy())
        for name, values in output.get("evidence_scale", {}).items():
            evidence_scale.setdefault(name, []).append(values.float().cpu().numpy())
    return {
        "rows": np.concatenate(rows),
        "logits": np.concatenate(logits),
        "reliability": {
            name: np.concatenate(values) for name, values in reliability.items()
        },
        "residual_norm": {
            name: np.concatenate(values) for name, values in residual_norm.items()
        },
        "evidence_scale": {
            name: np.concatenate(values) for name, values in evidence_scale.items()
        },
    }


def counterfactual_dataset(
    data: P100AData,
    held_indices: np.ndarray,
    normalizer: FoldNormalizer,
    modalities: tuple[str, ...],
    kind: str,
    seed: int,
) -> P100ADataset:
    zero: tuple[str, ...] = ()
    skeleton_source = None
    imu_source = None
    cross_available = True
    if kind == "zero_skeleton":
        zero = ("skeleton",)
    elif kind == "zero_imu":
        zero = ("imu",)
    elif kind == "zero_both":
        zero = ("skeleton", "imu")
    elif kind == "shuffle_skeleton":
        skeleton_source = within_subject_permutation(
            data, held_indices, "skeleton", seed
        )
        cross_available = False
    elif kind == "shuffle_imu":
        imu_source = within_subject_permutation(data, held_indices, "imu", seed)
        cross_available = False
    else:
        raise ValueError(kind)
    return P100ADataset(
        data,
        held_indices,
        normalizer,
        modalities,
        skeleton_source=skeleton_source,
        imu_source=imu_source,
        zero_modalities=zero,
        cross_available=cross_available,
    )


def run_fold(
    data: P100AData,
    variant: str,
    fold: int,
    config: dict[str, Any],
    output: Path,
    epochs: int,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    variant_output = output / variant
    variant_output.mkdir(parents=True, exist_ok=True)
    prediction_path = variant_output / f"fold{fold}_predictions.npz"
    checkpoint_path = variant_output / f"fold{fold}_final.pt"
    if resume and prediction_path.exists():
        print(f"resume {variant} fold {fold}", flush=True)
        with np.load(prediction_path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}

    modalities = CANONICAL_VARIANTS[variant]
    train_indices, held_indices = data.indices_for_fold(fold)
    normalizer = FoldNormalizer.fit(data, train_indices)
    weights = class_user_sample_weights(data, train_indices)
    seed = int(config["training"]["seed"]) + fold
    set_seed(seed)
    dataset = P100ADataset(
        data, train_indices, normalizer, modalities, sample_weights=weights
    )
    loader = make_loader(
        dataset,
        int(config["training"]["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    model_config = P100AModelConfig(
        modalities=modalities,
        **config["model"],
    )
    model = P100AGlobalTeacher(model_config).to(device)
    print(
        json.dumps(
            {
                "variant": variant,
                "fold": fold,
                "held_users": FOLD_USERS[fold],
                "train_rows": len(train_indices),
                "held_rows": len(held_indices),
                "parameters": model.parameter_count,
                "epochs": epochs,
            }
        ),
        flush=True,
    )
    history = train_model(model, loader, device, config["training"], epochs)
    # Save weights before any held-label metric is computed.  The OOF label can
    # therefore never affect checkpoint/epoch selection.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config.__dict__,
            "normalizer_means": normalizer.means,
            "normalizer_stds": normalizer.stds,
            "epochs": epochs,
            "history": history,
            "held_users": FOLD_USERS[fold],
        },
        checkpoint_path,
    )

    eval_batch_size = int(config["training"]["eval_batch_size"])
    direct_loader = make_loader(
        P100ADataset(data, held_indices, normalizer, modalities),
        eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    direct = evaluate_model(model, direct_loader, device)
    arrays: dict[str, np.ndarray] = {
        "rows": direct["rows"],
        "direct_logits": direct["logits"],
    }
    for name, values in direct["reliability"].items():
        arrays[f"direct_reliability_{name}"] = values
    for name, values in direct["residual_norm"].items():
        arrays[f"direct_residual_norm_{name}"] = values

    counterfactuals: list[str] = []
    if "skeleton" in modalities:
        counterfactuals.extend(("zero_skeleton", "shuffle_skeleton"))
    if "imu" in modalities:
        counterfactuals.extend(("zero_imu", "shuffle_imu"))
    if {"skeleton", "imu"}.issubset(modalities):
        counterfactuals.append("zero_both")
    for index, kind in enumerate(counterfactuals):
        counter_loader = make_loader(
            counterfactual_dataset(
                data,
                held_indices,
                normalizer,
                modalities,
                kind,
                seed + 1000 + index,
            ),
            eval_batch_size,
            shuffle=False,
            seed=seed,
        )
        counter = evaluate_model(model, counter_loader, device)
        if not np.array_equal(counter["rows"], direct["rows"]):
            raise RuntimeError("counterfactual row order changed")
        arrays[f"{kind}_logits"] = counter["logits"]
    np.savez_compressed(prediction_path, **arrays)
    return arrays


def aggregate_variant(
    data: P100AData,
    variant: str,
    fold_outputs: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    direct_logits = np.full((len(data.sample_ids), 40), np.nan, dtype=np.float32)
    counter_logits: dict[str, np.ndarray] = {}
    reliability: dict[str, np.ndarray] = {}
    residual_norm: dict[str, np.ndarray] = {}
    for fold_output in fold_outputs:
        rows = np.asarray(fold_output["rows"], dtype=np.int64)
        direct_logits[rows] = fold_output["direct_logits"]
        for name, values in fold_output.items():
            if name.endswith("_logits") and name != "direct_logits":
                counter_logits.setdefault(
                    name,
                    np.full((len(data.sample_ids), 40), np.nan, dtype=np.float32),
                )[rows] = values
            elif name.startswith("direct_reliability_"):
                reliability.setdefault(
                    name.removeprefix("direct_reliability_"),
                    np.full(len(data.sample_ids), np.nan, dtype=np.float32),
                )[rows] = values
            elif name.startswith("direct_residual_norm_"):
                residual_norm.setdefault(
                    name.removeprefix("direct_residual_norm_"),
                    np.full(len(data.sample_ids), np.nan, dtype=np.float32),
                )[rows] = values
    if not np.isfinite(direct_logits).all():
        raise RuntimeError(f"{variant} does not cover every OOF row")
    direct_probability = softmax_numpy(direct_logits)
    np.savez_compressed(
        output / f"{variant}_complete_oof.npz",
        sample_ids=data.sample_ids,
        users=data.users,
        fold_ids=data.fold_ids,
        direct_logits=direct_logits,
        direct_probability=direct_probability,
        **counter_logits,
        **{f"reliability_{name}": values for name, values in reliability.items()},
        **{f"residual_norm_{name}": values for name, values in residual_norm.items()},
    )
    counterfactual_metrics = {
        name.removesuffix("_logits"): classification_metrics(
            softmax_numpy(values), data.labels, data.users
        )
        for name, values in counter_logits.items()
    }
    return {
        "variant": variant,
        "modalities": list(CANONICAL_VARIANTS[variant]),
        "direct": classification_metrics(direct_probability, data.labels, data.users),
        "counterfactuals": counterfactual_metrics,
        "reliability": {
            name: {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in reliability.items()
        },
        "evidence_residual_norm": {
            name: {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in residual_norm.items()
        },
        "_probability": direct_probability,
    }


def positive_gate(
    summaries: dict[str, dict[str, Any]], comparison: dict[str, Any]
) -> dict[str, Any]:
    vsi = summaries["VSI"]["direct"]
    visual = summaries["V"]["direct"]
    subject_values = list(comparison["per_subject"].values())
    checks = {
        "top1_positive": vsi["top1"] > visual["top1"],
        "top5_positive": vsi["top5"] > visual["top5"],
        "net_at_least_10": comparison["net"] >= 10,
        "subjects_nonnegative_at_least_8": sum(
            value["net"] >= 0 for value in subject_values
        )
        >= 8,
        "subjects_positive_at_least_4": sum(value["net"] > 0 for value in subject_values)
        >= 4,
        "worst_subject_drop_within_2pp": min(
            value["accuracy_pp"] for value in subject_values
        )
        >= -2.0,
        "mcnemar_p_at_most_0_10": comparison["mcnemar_exact_p"] <= 0.10,
    }
    checks["aligned_modality_evidence"] = any(
        summaries["VSI"]["direct"]["top1"]
        > summaries["VSI"]["counterfactuals"][kind]["top1"]
        for kind in ("shuffle_skeleton", "shuffle_imu")
    )
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "strong_confirmation": bool(
            all(checks.values())
            and (
                comparison["mcnemar_exact_p"] <= 0.05
                or comparison["subject_bootstrap"]["lower_95"] > 0
            )
        ),
    }


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()


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
    variants = (
        list(CANONICAL_VARIANTS)
        if args.variants == "all"
        else [value.strip() for value in args.variants.split(",")]
    )
    folds = (
        list(range(4))
        if args.folds == "all"
        else [int(value) for value in args.folds.split(",")]
    )
    unknown = set(variants) - set(CANONICAL_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    data = load_p100a_data()
    write_contract(output / "data_contract.json", data)
    run_manifest = {
        "status": "smoke" if args.smoke else "formal_started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "variants": variants,
        "folds": folds,
        "epochs": epochs,
        "device": str(device),
        "data": data.summary(),
        "outer_label_used_for_checkpoint_selection": False,
        "h3_code_path_present": False,
        "b_teacher_code_path_present": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        fold_outputs = [
            run_fold(
                data,
                variant,
                fold,
                config,
                output,
                epochs,
                device,
                args.resume,
            )
            for fold in folds
        ]
        if folds == list(range(4)):
            summaries[variant] = aggregate_variant(
                data, variant, fold_outputs, output
            )

    if folds == list(range(4)) and set(variants) == set(CANONICAL_VARIANTS):
        comparisons = {
            "VS_minus_V": paired_comparison(
                summaries["VS"]["_probability"],
                summaries["V"]["_probability"],
                data.labels,
                data.users,
            ),
            "VI_minus_V": paired_comparison(
                summaries["VI"]["_probability"],
                summaries["V"]["_probability"],
                data.labels,
                data.users,
            ),
            "VSI_minus_V": paired_comparison(
                summaries["VSI"]["_probability"],
                summaries["V"]["_probability"],
                data.labels,
                data.users,
            ),
            "VSI_minus_VI_skeleton": paired_comparison(
                summaries["VSI"]["_probability"],
                summaries["VI"]["_probability"],
                data.labels,
                data.users,
            ),
            "VSI_minus_VS_imu": paired_comparison(
                summaries["VSI"]["_probability"],
                summaries["VS"]["_probability"],
                data.labels,
                data.users,
            ),
        }
        comparisons["VSI_minus_V"]["confusion_changes"] = confusion_change_groups(
            summaries["VSI"]["_probability"],
            summaries["V"]["_probability"],
            data.labels,
        )
        gate = positive_gate(summaries, comparisons["VSI_minus_V"])
        clean_summaries: dict[str, dict[str, Any]] = {}
        for variant, summary in summaries.items():
            clean_summaries[variant] = {
                name: value for name, value in summary.items() if name != "_probability"
            }
        final_summary = {
            "status": "complete",
            "protocol": "P100-A fixed 4-fold / 12-subject OOF; fixed epoch; no outer-label checkpoint selection",
            "git_commit": git_commit(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "epochs": epochs,
            "data": data.summary(),
            "variants": clean_summaries,
            "comparisons": comparisons,
            "a_teacher_positive_gate": gate,
            "leakage_audit": {
                "development_allow_list_exact": set(data.users) == DEV_USER_SET,
                "h3_rows_loaded": 0,
                "historical_40class_expert_probability_loaded": False,
                "outer_subject_disjoint": True,
                "fixed_epoch_without_outer_label_selection": True,
                "b_teacher_started": False,
            },
        }
        (output / "summary.json").write_text(
            json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
