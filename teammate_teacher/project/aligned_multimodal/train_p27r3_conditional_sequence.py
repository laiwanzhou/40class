from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from p27r2_event_data import load_event_cache
from p27r3_conditional_model import ConditionalEventSequenceModel, parameter_count
from probe_p27r3_incremental_information import (
    FOCUS_IDS,
    HARD_IDS,
    SMALL_IDS,
    explicit_event_sequence,
    load_fold_core,
    metric_bundle,
    transformed_sequence,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "p27_r3_conditional_sequence.json"
DEFAULT_CACHE = (
    PROJECT_DIR / "runs" / "p27_r2_event_audit" / "event_cache_v2.npz"
)
DEFAULT_CORE_DIR = PROJECT_DIR / "runs" / "p27_r2_fold0"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_r3_conditional_sequence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P27-R3 conditional phase-preserving event sequence probe"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--core-dir", type=Path, default=DEFAULT_CORE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_normalizer(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = sequence.reshape(-1, sequence.shape[-1])
    median = np.median(flattened, axis=0)
    q25, q75 = np.quantile(flattened, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-3)
    return median.astype(np.float32), scale.astype(np.float32)


def normalize(
    sequence: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.clip((sequence - median) / scale, -8.0, 8.0).astype(np.float32)


def make_tensors(
    cache,
    indices: np.ndarray,
    base_logits: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
) -> tuple[torch.Tensor, ...]:
    sequence = normalize(explicit_event_sequence(cache, indices), median, scale)
    return (
        torch.from_numpy(base_logits.astype(np.float32)),
        torch.from_numpy(sequence),
        torch.from_numpy(cache.modality_mask[indices].astype(np.float32)),
        torch.from_numpy(cache.event_quality[indices].astype(np.float32)),
        torch.from_numpy(cache.labels[indices].astype(np.int64)),
    )


def class_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    weights = np.zeros(40, dtype=np.float32)
    present = counts > 0
    weights[present] = np.power(counts[present], -float(power)).astype(np.float32)
    weights[present] /= weights[present].mean()
    return weights


def transform_batch(
    sequence: torch.Tensor,
    mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if mode != "event_sequence_orderless":
        return sequence
    output = torch.empty_like(sequence)
    for index in range(len(sequence)):
        order = torch.randperm(sequence.shape[1], generator=generator).to(
            sequence.device
        )
        output[index] = sequence[index, order]
    return output


def train_model(
    train_tensors: tuple[torch.Tensor, ...],
    held_tensors: tuple[torch.Tensor, ...],
    config: dict[str, Any],
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[ConditionalEventSequenceModel, dict[str, Any], np.ndarray]:
    seed_everything(seed)
    architecture = config["architecture"]
    training = config["training"]
    model = ConditionalEventSequenceModel(
        modality_hidden=int(architecture["modality_hidden"]),
        gru_hidden=int(architecture["gru_hidden"]),
        gru_layers=int(architecture["gru_layers"]),
        segment_count=int(architecture["segment_count"]),
        base_hidden=int(architecture["base_hidden"]),
        joint_hidden=int(architecture["joint_hidden"]),
        dropout=float(architecture["dropout"]),
        delta_limit=float(architecture["delta_limit"]),
        gate_initial_bias=float(architecture["gate_initial_bias"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"])
    )
    weights = torch.from_numpy(
        class_weights(
            train_tensors[-1].numpy(), float(training["class_weight_power"])
        )
    ).to(device)
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    order_generator = torch.Generator().manual_seed(seed + 77)
    history: list[dict[str, float]] = []
    initial_error = 0.0
    with torch.no_grad():
        batch = tuple(value[: min(64, len(value))].to(device) for value in train_tensors)
        output = model(
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            disable_event=mode == "base_only",
        )
        initial_error = float((output["logits"] - batch[0]).abs().max().cpu())
    if initial_error > 1e-7:
        raise RuntimeError(f"Initial logits do not reproduce base: {initial_error}")

    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "gate": 0.0}
        seen = 0
        for batch in loader:
            base, sequence, presence, quality, labels = (
                value.to(device, non_blocking=True) for value in batch
            )
            sequence = transform_batch(sequence, mode, order_generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(
                    base,
                    sequence,
                    presence,
                    quality,
                    disable_event=mode == "base_only",
                )
                ce = F.cross_entropy(
                    output["logits"],
                    labels,
                    weight=weights,
                    label_smoothing=float(training["label_smoothing"]),
                )
                base_probability = torch.softmax(base.detach(), dim=1)
                kl = F.kl_div(
                    torch.log_softmax(output["logits"], dim=1),
                    base_probability,
                    reduction="batchmean",
                )
                gate_penalty = output["gate"].mean()
                loss = (
                    ce
                    + float(training["base_kl_anchor_weight"]) * kl
                    + float(training["gate_l1_weight"]) * gate_penalty
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(labels)
            seen += count
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["ce"] += float(ce.detach().cpu()) * count
            totals["kl"] += float(kl.detach().cpu()) * count
            totals["gate"] += float(gate_penalty.detach().cpu()) * count
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "loss": totals["loss"] / seen,
                "ce": totals["ce"] / seen,
                "kl": totals["kl"] / seen,
                "gate": totals["gate"] / seen,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    held = tuple(value.to(device) for value in held_tensors)
    if mode == "event_sequence_orderless":
        deterministic = torch.Generator().manual_seed(seed + 7007)
        held_sequence = transform_batch(held[1].cpu(), mode, deterministic).to(device)
    else:
        held_sequence = held[1]
    model.eval()
    with torch.no_grad():
        output = model(
            held[0],
            held_sequence,
            held[2],
            held[3],
            disable_event=mode == "base_only",
        )
    predictions = output["logits"].argmax(1).cpu().numpy()
    info = {
        "mode": mode,
        "parameters": parameter_count(model),
        "fp16_parameter_mib": parameter_count(model) * 2 / (1024**2),
        "initial_max_abs_logit_error": initial_error,
        "train_seconds": time.perf_counter() - started,
        "final_train": history[-1],
        "held_gate_mean": float(output["gate"].mean().cpu()),
        "held_gate_std": float(output["gate"].std().cpu()),
        "metrics": metric_bundle(held_tensors[-1].numpy(), predictions),
    }
    return model, info, predictions


def predict_with_ablation(
    model: ConditionalEventSequenceModel,
    held_tensors: tuple[torch.Tensor, ...],
    ablation: str,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    base, sequence, presence, quality, _ = held_tensors
    array = sequence.numpy()
    mapping = {
        "event_zero": "zero",
        "event_cross_sample_shuffle": "cross_sample_shuffle",
        "time_reverse": "time_reverse",
        "within_sample_time_permutation": "within_sample_time_permutation",
    }
    transformed = transformed_sequence(array, mapping[ablation], seed)
    model.eval()
    with torch.no_grad():
        output = model(
            base.to(device),
            torch.from_numpy(transformed).to(device),
            presence.to(device),
            quality.to(device),
            disable_event=False,
        )
    return output["logits"].argmax(1).cpu().numpy()


def rows_by_subject_and_class(
    cache,
    indices: np.ndarray,
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    subjects = cache.subjects[indices].astype(str)
    for method, values in predictions.items():
        for subject in sorted(np.unique(subjects).tolist()):
            mask = subjects == subject
            metrics = {
                "accuracy": float(accuracy_score(labels[mask], values[mask])),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[mask], values[mask])
                ),
                "macro_f1": float(
                    f1_score(
                        labels[mask], values[mask], average="macro", zero_division=0
                    )
                ),
            }
            subject_rows.append(
                {
                    "method": method,
                    "subject": subject,
                    "samples": int(mask.sum()),
                    **metrics,
                }
            )
        for class_id in range(40):
            mask = labels == class_id
            if mask.any():
                class_rows.append(
                    {
                        "method": method,
                        "class_id": class_id,
                        "samples": int(mask.sum()),
                        "recall": float(np.mean(values[mask] == class_id)),
                    }
                )
    return subject_rows, class_rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    cache = load_event_cache(args.cache.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_train = cache.outer_folds != int(config["outer_fold"])
    fold_summaries: dict[str, Any] = {}
    fold_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    all_subject_rows: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(3):
        core = load_fold_core(cache, args.core_dir.resolve(), fold)
        train_indices = core["train_indices"]
        held_indices = core["held_indices"]
        if not np.all(outer_train[train_indices]) or not np.all(outer_train[held_indices]):
            raise RuntimeError("Conditional probe attempted to include outer-held")
        train_raw = explicit_event_sequence(cache, train_indices)
        median, scale = fit_normalizer(train_raw)
        train_tensors = make_tensors(
            cache, train_indices, core["train_logits"], median, scale
        )
        held_tensors = make_tensors(
            cache, held_indices, core["held_logits"], median, scale
        )
        labels = held_tensors[-1].numpy()
        predictions: dict[str, np.ndarray] = {
            "base_argmax": core["held_logits"].argmax(axis=1)
        }
        models: dict[str, ConditionalEventSequenceModel] = {}
        model_info: dict[str, Any] = {}
        for model_index, mode in enumerate(config["models"]):
            model, info, values = train_model(
                train_tensors,
                held_tensors,
                config,
                str(mode),
                int(config["seed"]) + fold * 100 + model_index,
                device,
            )
            models[str(mode)] = model
            model_info[str(mode)] = info
            predictions[str(mode)] = values
            print(
                f"fold={fold} mode={mode} "
                f"overall={info['metrics']['overall']['accuracy']:.4f} "
                f"hard={info['metrics']['hard']['accuracy']:.4f} "
                f"seconds={info['train_seconds']:.1f}",
                flush=True,
            )
        for method, values in predictions.items():
            metrics = metric_bundle(labels, values)
            fold_rows.append(
                {
                    "inner_fold": fold,
                    "method": method,
                    **{
                        f"{subset}_{metric}": value
                        for subset, subset_values in metrics.items()
                        for metric, value in subset_values.items()
                    },
                }
            )
        event_reference = metric_bundle(labels, predictions["event_sequence"])
        for ablation_index, ablation in enumerate(config["ablations"]):
            values = predict_with_ablation(
                models["event_sequence"],
                held_tensors,
                str(ablation),
                int(config["seed"]) + fold * 1000 + ablation_index,
                device,
            )
            metrics = metric_bundle(labels, values)
            ablation_rows.append(
                {
                    "inner_fold": fold,
                    "ablation": ablation,
                    **{
                        f"{subset}_{metric}": value
                        for subset, subset_values in metrics.items()
                        for metric, value in subset_values.items()
                    },
                    "overall_delta_pp": 100
                    * (
                        metrics["overall"]["accuracy"]
                        - event_reference["overall"]["accuracy"]
                    ),
                    "hard_delta_pp": 100
                    * (
                        metrics["hard"]["accuracy"]
                        - event_reference["hard"]["accuracy"]
                    ),
                    "focus_delta_pp": 100
                    * (
                        metrics["focus"]["accuracy"]
                        - event_reference["focus"]["accuracy"]
                    ),
                }
            )
        subject_rows, class_rows = rows_by_subject_and_class(
            cache, held_indices, labels, predictions
        )
        for row in subject_rows:
            row["inner_fold"] = fold
        for row in class_rows:
            row["inner_fold"] = fold
        all_subject_rows.extend(subject_rows)
        all_class_rows.extend(class_rows)
        rescues = int(
            np.sum(
                (predictions["base_only"] != labels)
                & (predictions["event_sequence"] == labels)
            )
        )
        new_errors = int(
            np.sum(
                (predictions["base_only"] == labels)
                & (predictions["event_sequence"] != labels)
            )
        )
        ledger_rows.append(
            {
                "inner_fold": fold,
                "hypothesis": config["development_hypothesis"],
                "change": "GRU ordered segment states + sample-adaptive unbounded-enough gated delta; no scalar event loss",
                "rescues_vs_base_only": rescues,
                "new_errors_vs_base_only": new_errors,
                "net": rescues - new_errors,
                "event_minus_base_only_overall_pp": 100
                * (
                    event_reference["overall"]["accuracy"]
                    - metric_bundle(labels, predictions["base_only"])["overall"][
                        "accuracy"
                    ]
                ),
                "event_minus_base_only_hard_pp": 100
                * (
                    event_reference["hard"]["accuracy"]
                    - metric_bundle(labels, predictions["base_only"])["hard"][
                        "accuracy"
                    ]
                ),
            }
        )
        fold_summaries[str(fold)] = {
            "train_samples": int(len(train_indices)),
            "held_samples": int(len(held_indices)),
            "train_subjects": sorted(np.unique(cache.subjects[train_indices]).tolist()),
            "held_subjects": sorted(np.unique(cache.subjects[held_indices]).tolist()),
            "models": model_info,
            "normalizer": {
                "median": median.tolist(),
                "scale": scale.tolist(),
                "fit_scope": "inner-train only",
            },
            "outer_held_predictions_generated": False,
        }
        (output / f"fold_{fold}_progress.json").write_text(
            json.dumps(
                {
                    "protocol": config["protocol"],
                    "inner_fold": fold,
                    "models": model_info,
                    "ledger": ledger_rows[-1],
                    "outer_held_predictions_generated": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "ablations.csv", ablation_rows)
    write_csv(output / "per_subject.csv", all_subject_rows)
    write_csv(output / "per_class.csv", all_class_rows)
    write_csv(output / "development_ledger.csv", ledger_rows)
    means: dict[str, Any] = {}
    for method in ["base_argmax", *config["models"]]:
        rows = [row for row in fold_rows if row["method"] == method]
        means[str(method)] = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in rows[0]
            if key not in {"inner_fold", "method"}
        }
    summary = {
        "protocol": config["protocol"],
        "status": "complete",
        "device": str(device),
        "outer_train_samples": int(outer_train.sum()),
        "outer_held_samples_excluded": int((~outer_train).sum()),
        "outer_held_predictions_generated": False,
        "mean_metrics": means,
        "folds": fold_summaries,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
