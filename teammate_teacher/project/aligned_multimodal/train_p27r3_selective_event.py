from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from p27r2_event_data import EVENT_NAMES, load_event_cache
from p27r3_selective_model import SelectiveEventCorrector, parameter_count
from probe_p27r3_incremental_information import (
    explicit_event_sequence,
    load_fold_core,
    metric_bundle,
    transformed_sequence,
    write_csv,
)
from train_p27r3_conditional_sequence import (
    class_weights,
    fit_normalizer,
    normalize,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "p27_r3_selective_event.json"
DEFAULT_CACHE = (
    PROJECT_DIR / "runs" / "p27_r2_event_audit" / "event_cache_v2.npz"
)
DEFAULT_CORE_DIR = PROJECT_DIR / "runs" / "p27_r2_fold0"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_r3_selective_event"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P27-R3 selective event correction inner development"
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


def make_tensors(
    cache,
    indices: np.ndarray,
    base_logits: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
    subject_lookup: dict[str, int],
) -> tuple[torch.Tensor, ...]:
    sequence = normalize(
        explicit_event_sequence(cache, indices), median, scale
    )
    subject_ids = np.asarray(
        [subject_lookup[str(value)] for value in cache.subjects[indices]],
        dtype=np.int64,
    )
    return (
        torch.from_numpy(base_logits.astype(np.float32)),
        torch.from_numpy(sequence),
        torch.from_numpy(cache.modality_mask[indices].astype(np.float32)),
        torch.from_numpy(cache.event_quality[indices].astype(np.float32)),
        torch.from_numpy(cache.event_targets[indices].astype(np.float32)),
        torch.from_numpy(subject_ids),
        torch.from_numpy(cache.labels[indices].astype(np.int64)),
    )


def permute_each_sequence(sequence: torch.Tensor) -> torch.Tensor:
    order = torch.rand(
        sequence.shape[:2], device=sequence.device
    ).argsort(dim=1)
    return torch.gather(
        sequence, 1, order[:, :, None].expand_as(sequence)
    )


def supervised_cross_subject_contrastive(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    embedding = F.normalize(embedding.float(), dim=1)
    similarity = embedding @ embedding.T / float(temperature)
    self_mask = torch.eye(
        len(embedding), dtype=torch.bool, device=embedding.device
    )
    similarity = similarity.masked_fill(self_mask, -1e4)
    positive = (
        (labels[:, None] == labels[None, :])
        & (subjects[:, None] != subjects[None, :])
        & ~self_mask
    )
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embedding.sum() * 0.0
    log_probability = similarity - torch.logsumexp(
        similarity, dim=1, keepdim=True
    )
    per_anchor = -(
        (log_probability * positive.float()).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return per_anchor[valid].mean()


def automatic_confusion_rank_loss(
    event_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    order = torch.argsort(base_logits.detach(), dim=1, descending=True)
    top1 = order[:, 0]
    top2 = order[:, 1]
    competitor = torch.where(top1 == labels, top2, top1)
    row = torch.arange(len(labels), device=labels.device)
    gap = event_logits[row, labels] - event_logits[row, competitor]
    base_wrong = (top1 != labels).float()
    weights = 1.0 + base_wrong
    return (
        weights
        * 0.5
        * F.softplus((float(margin) - gap) / 0.5)
    ).mean()


def masked_event_probe_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    quality: torch.Tensor,
    indices: list[int],
) -> torch.Tensor:
    selected_target = targets[:, indices]
    selected_quality = quality[:, indices].clamp(0.0, 1.0)
    error = F.smooth_l1_loss(
        prediction, selected_target, reduction="none", beta=0.1
    )
    return (error * selected_quality).sum() / selected_quality.sum().clamp_min(
        1.0
    )


def instantiate_model(
    config: dict[str, Any],
    event_target_count: int,
) -> SelectiveEventCorrector:
    architecture = config["architecture"]
    return SelectiveEventCorrector(
        modality_hidden=int(architecture["modality_hidden"]),
        gru_hidden=int(architecture["gru_hidden"]),
        gru_layers=int(architecture["gru_layers"]),
        segment_count=int(architecture["segment_count"]),
        base_hidden=int(architecture["base_hidden"]),
        representation_dim=int(architecture["representation_dim"]),
        projection_dim=int(architecture["projection_dim"]),
        dropout=float(architecture["dropout"]),
        delta_limit=float(architecture["delta_limit"]),
        scale_limit=float(architecture["scale_limit"]),
        event_target_count=event_target_count,
    )


def train_model(
    train_tensors: tuple[torch.Tensor, ...],
    held_tensors: tuple[torch.Tensor, ...],
    config: dict[str, Any],
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[
    SelectiveEventCorrector,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    seed_everything(seed)
    training = config["training"]
    event_indices = [int(value) for value in config["passed_event_indices"]]
    use_rank = mode in {"selective_fine", "selective_fine_orderless", "selective_rank"}
    use_supcon = mode in {
        "selective_fine",
        "selective_fine_orderless",
        "selective_supcon",
    }
    use_event_probe = mode in {
        "selective_fine",
        "selective_fine_orderless",
        "selective_probe",
    }
    use_order = mode in {
        "selective_fine",
        "selective_fine_orderless",
        "selective_order",
    }
    fine = use_rank or use_supcon or use_event_probe or use_order
    orderless = mode.endswith("orderless")
    model = instantiate_model(config, len(event_indices)).to(device)
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
            train_tensors[-1].numpy(),
            float(training["class_weight_power"]),
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
    with torch.no_grad():
        batch = tuple(
            value[: min(64, len(value))].to(device)
            for value in train_tensors
        )
        initial = model(
            batch[0], batch[1], batch[2], batch[3]
        )["logits"]
        initial_error = float((initial - batch[0]).abs().max().cpu())
    if initial_error > 1e-7:
        raise RuntimeError(
            f"Initial logits do not reproduce base: {initial_error}"
        )

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "final_ce": 0.0,
            "event_ce": 0.0,
            "error_bce": 0.0,
            "kl": 0.0,
            "rank": 0.0,
            "supcon": 0.0,
            "event_probe": 0.0,
            "order": 0.0,
            "mix_scale": 0.0,
        }
        seen = 0
        for batch in loader:
            (
                base,
                sequence,
                presence,
                quality,
                event_targets,
                subjects,
                labels,
            ) = (value.to(device, non_blocking=True) for value in batch)
            if orderless:
                sequence = permute_each_sequence(sequence)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(base, sequence, presence, quality)
                final_ce = F.cross_entropy(
                    output["logits"],
                    labels,
                    weight=weights,
                    label_smoothing=float(training["label_smoothing"]),
                )
                event_ce = F.cross_entropy(
                    output["event_logits"],
                    labels,
                    weight=weights,
                    label_smoothing=float(training["label_smoothing"]),
                )
                base_wrong = (
                    base.detach().argmax(dim=1) != labels
                ).float()
                error_bce = F.binary_cross_entropy_with_logits(
                    output["base_error_logit"],
                    base_wrong,
                )
                kl = F.kl_div(
                    torch.log_softmax(output["logits"], dim=1),
                    torch.softmax(base.detach(), dim=1),
                    reduction="batchmean",
                )
                loss = (
                    float(training["final_ce_weight"]) * final_ce
                    + float(training["event_ce_weight"]) * event_ce
                    + float(training["base_error_bce_weight"]) * error_bce
                    + float(training["base_kl_anchor_weight"]) * kl
                    + float(training["mix_scale_l1_weight"])
                    * output["mix_scale"].abs()
                )
                rank = output["logits"].sum() * 0.0
                supcon = rank
                event_probe = rank
                order_loss = rank
                if fine:
                    if use_rank:
                        rank = automatic_confusion_rank_loss(
                            output["event_logits"],
                            base,
                            labels,
                            float(training["rank_margin"]),
                        )
                        loss = (
                            loss
                            + float(training["fine_rank_weight"]) * rank
                        )
                    if use_supcon:
                        supcon = supervised_cross_subject_contrastive(
                            output["metric_embedding"],
                            labels,
                            subjects,
                            float(training["supcon_temperature"]),
                        )
                        loss = (
                            loss
                            + float(training["fine_supcon_weight"]) * supcon
                        )
                    if use_event_probe:
                        event_probe = masked_event_probe_loss(
                            output["event_targets"],
                            event_targets,
                            quality,
                            event_indices,
                        )
                        loss = (
                            loss
                            + float(training["fine_event_probe_weight"])
                            * event_probe
                        )
                    if use_order:
                        reversed_representation = model.encode_event(
                            torch.flip(sequence, dims=[1]), presence
                        )
                        reversed_order = model.order_head(
                            reversed_representation
                        ).squeeze(1)
                        order_loss = 0.5 * (
                            F.binary_cross_entropy_with_logits(
                                output["order_logit"],
                                torch.ones_like(output["order_logit"]),
                            )
                            + F.binary_cross_entropy_with_logits(
                                reversed_order,
                                torch.zeros_like(reversed_order),
                            )
                        )
                        loss = (
                            loss
                            + float(training["fine_order_weight"])
                            * order_loss
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
            values = {
                "loss": loss,
                "final_ce": final_ce,
                "event_ce": event_ce,
                "error_bce": error_bce,
                "kl": kl,
                "rank": rank,
                "supcon": supcon,
                "event_probe": event_probe,
                "order": order_loss,
                "mix_scale": output["mix_scale"].abs(),
            }
            for key, value in values.items():
                totals[key] += float(value.detach().cpu()) * count
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                **{key: value / seen for key, value in totals.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    held = tuple(value.to(device) for value in held_tensors)
    held_sequence = held[1]
    if orderless:
        transformed = transformed_sequence(
            held[1].cpu().numpy(),
            "within_sample_time_permutation",
            seed + 9001,
        )
        held_sequence = torch.from_numpy(transformed).to(device)
    model.eval()
    with torch.no_grad():
        output = model(
            held[0], held_sequence, held[2], held[3]
        )
        reverse_representation = model.encode_event(
            torch.flip(held_sequence, dims=[1]), held[2]
        )
        reverse_order = model.order_head(
            reverse_representation
        ).squeeze(1)
    arrays = {
        "predictions": output["logits"].argmax(1).cpu().numpy(),
        "logits": output["logits"].float().cpu().numpy(),
        "event_predictions": output["event_targets"].float().cpu().numpy(),
        "event_logits": output["event_logits"].float().cpu().numpy(),
        "base_error_probability": output[
            "base_error_probability"
        ].float().cpu().numpy()[:, 0],
    }
    order_accuracy = 0.5 * (
        float((output["order_logit"] > 0).float().mean().cpu())
        + float((reverse_order < 0).float().mean().cpu())
    )
    info = {
        "mode": mode,
        "parameters": parameter_count(model),
        "fp16_parameter_mib": parameter_count(model) * 2 / (1024**2),
        "initial_max_abs_logit_error": initial_error,
        "train_seconds": time.perf_counter() - started,
        "final_train": history[-1],
        "held_mix_scale": float(output["mix_scale"].cpu()),
        "held_base_error_probability_mean": float(
            output["base_error_probability"].mean().cpu()
        ),
        "held_base_error_prediction_accuracy": float(
            np.mean(
                (arrays["base_error_probability"] >= 0.5)
                == (
                    held_tensors[0].numpy().argmax(1)
                    != held_tensors[-1].numpy()
                )
            )
        ),
        "held_order_direction_accuracy": order_accuracy,
        "metrics": metric_bundle(
            held_tensors[-1].numpy(), arrays["predictions"]
        ),
        "event_branch_metrics": metric_bundle(
            held_tensors[-1].numpy(),
            arrays["event_logits"].argmax(1),
        ),
    }
    return model, info, arrays


def predict_ablation(
    model: SelectiveEventCorrector,
    held_tensors: tuple[torch.Tensor, ...],
    ablation: str,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    base, sequence, presence, quality = held_tensors[:4]
    sequence_array = sequence.numpy().copy()
    presence_array = presence.numpy().copy()
    mapping = {
        "event_zero": "zero",
        "event_cross_sample_shuffle": "cross_sample_shuffle",
        "time_reverse": "time_reverse",
        "within_sample_time_permutation": "within_sample_time_permutation",
    }
    if ablation in mapping:
        sequence_array = transformed_sequence(
            sequence_array, mapping[ablation], seed
        )
    elif ablation == "skeleton_zero":
        sequence_array[:, :, :16] = 0
        presence_array[:, 0] = 0
    elif ablation == "imu_zero":
        sequence_array[:, :, 16:41] = 0
        presence_array[:, 1] = 0
    elif ablation == "visual_zero":
        sequence_array[:, :, 41:51] = 0
        presence_array[:, 2:4] = 0
    else:
        raise ValueError(ablation)
    model.eval()
    with torch.no_grad():
        output = model(
            base.to(device),
            torch.from_numpy(sequence_array).to(device),
            torch.from_numpy(presence_array).to(device),
            quality.to(device),
        )
    return output["logits"].argmax(1).cpu().numpy()


def event_probe_rows(
    cache,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
    event_predictions: np.ndarray,
    event_indices: list[int],
    fold: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_index, event_index in enumerate(event_indices):
        train_quality = cache.event_quality[train_indices, event_index]
        held_quality = cache.event_quality[held_indices, event_index]
        train_target = cache.event_targets[train_indices, event_index]
        held_target = cache.event_targets[held_indices, event_index]
        valid_train = train_quality > 0
        valid_held = held_quality > 0
        if not valid_train.any() or not valid_held.any():
            continue
        mean = float(
            np.average(
                train_target[valid_train],
                weights=train_quality[valid_train],
            )
        )
        baseline_mae = float(
            np.average(
                np.abs(held_target[valid_held] - mean),
                weights=held_quality[valid_held],
            )
        )
        model_mae = float(
            np.average(
                np.abs(
                    held_target[valid_held]
                    - event_predictions[valid_held, output_index]
                ),
                weights=held_quality[valid_held],
            )
        )
        rows.append(
            {
                "inner_fold": fold,
                "event_index": event_index,
                "event_name": EVENT_NAMES[event_index],
                "held_valid": int(valid_held.sum()),
                "constant_mean_mae": baseline_mae,
                "model_mae": model_mae,
                "relative_improvement": (
                    (baseline_mae - model_mae) / baseline_mae
                    if baseline_mae > 1e-8
                    else math.nan
                ),
            }
        )
    return rows


def subject_and_class_rows(
    cache,
    indices: np.ndarray,
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subjects = cache.subjects[indices].astype(str)
    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for method, prediction in predictions.items():
        for subject in sorted(np.unique(subjects).tolist()):
            selected = subjects == subject
            subject_rows.append(
                {
                    "inner_fold": fold,
                    "method": method,
                    "subject": subject,
                    "samples": int(selected.sum()),
                    "accuracy": float(
                        accuracy_score(
                            labels[selected], prediction[selected]
                        )
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            labels[selected], prediction[selected]
                        )
                    ),
                    "macro_f1": float(
                        f1_score(
                            labels[selected],
                            prediction[selected],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
        for class_id in range(40):
            selected = labels == class_id
            if selected.any():
                class_rows.append(
                    {
                        "inner_fold": fold,
                        "method": method,
                        "class_id": class_id,
                        "samples": int(selected.sum()),
                        "recall": float(
                            np.mean(prediction[selected] == class_id)
                        ),
                    }
                )
    return subject_rows, class_rows


def flatten_metrics(
    metrics: dict[str, dict[str, float | int]]
) -> dict[str, float | int]:
    return {
        f"{subset}_{name}": value
        for subset, values in metrics.items()
        for name, value in values.items()
    }


def main() -> None:
    args = parse_args()
    config = json.loads(
        args.config.resolve().read_text(encoding="utf-8")
    )
    cache = load_event_cache(args.cache.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_train_mask = cache.outer_folds != int(config["outer_fold"])
    subject_lookup = {
        value: index
        for index, value in enumerate(
            sorted(np.unique(cache.subjects[outer_train_mask]).astype(str))
        )
    }
    event_indices = [
        int(value) for value in config["passed_event_indices"]
    ]
    primary_method = str(
        config.get("primary_method", "selective_fine")
    )
    ce_method = str(config.get("ce_method", "selective_ce"))
    fold_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    fold_summaries: dict[str, Any] = {}
    started = time.perf_counter()
    for fold in range(3):
        core = load_fold_core(cache, args.core_dir.resolve(), fold)
        train_indices = core["train_indices"]
        held_indices = core["held_indices"]
        if not np.all(outer_train_mask[train_indices]) or not np.all(
            outer_train_mask[held_indices]
        ):
            raise RuntimeError("outer-held entered selective development")
        median, scale = fit_normalizer(
            explicit_event_sequence(cache, train_indices)
        )
        train_tensors = make_tensors(
            cache,
            train_indices,
            core["train_logits"],
            median,
            scale,
            subject_lookup,
        )
        held_tensors = make_tensors(
            cache,
            held_indices,
            core["held_logits"],
            median,
            scale,
            subject_lookup,
        )
        labels = held_tensors[-1].numpy()
        predictions: dict[str, np.ndarray] = {
            "p12_core": core["held_logits"].argmax(1)
        }
        model_info: dict[str, Any] = {}
        fine_model: SelectiveEventCorrector | None = None
        fine_arrays: dict[str, np.ndarray] | None = None
        for model_index, mode in enumerate(config["models"]):
            model, info, arrays = train_model(
                train_tensors,
                held_tensors,
                config,
                str(mode),
                int(config["seed"]) + fold * 100,
                device,
            )
            predictions[str(mode)] = arrays["predictions"]
            model_info[str(mode)] = info
            if mode == primary_method:
                fine_model = model
                fine_arrays = arrays
                probe_rows.extend(
                    event_probe_rows(
                        cache,
                        train_indices,
                        held_indices,
                        arrays["event_predictions"],
                        event_indices,
                        fold,
                    )
                )
            print(
                f"fold={fold} mode={mode} "
                f"overall={info['metrics']['overall']['accuracy']:.4f} "
                f"hard={info['metrics']['hard']['accuracy']:.4f} "
                f"scale={info['held_mix_scale']:.3f} "
                f"seconds={info['train_seconds']:.1f}",
                flush=True,
            )
        if fine_model is None or fine_arrays is None:
            raise RuntimeError(f"{primary_method} was not trained")
        for method, prediction in predictions.items():
            fold_rows.append(
                {
                    "inner_fold": fold,
                    "method": method,
                    **flatten_metrics(metric_bundle(labels, prediction)),
                }
            )
        fine_metrics = metric_bundle(labels, predictions[primary_method])
        for ablation_index, ablation in enumerate(config["ablations"]):
            prediction = predict_ablation(
                fine_model,
                held_tensors,
                str(ablation),
                int(config["seed"]) + fold * 1000 + ablation_index,
                device,
            )
            metrics = metric_bundle(labels, prediction)
            ablation_rows.append(
                {
                    "inner_fold": fold,
                    "ablation": ablation,
                    **flatten_metrics(metrics),
                    "overall_delta_pp": 100
                    * (
                        metrics["overall"]["accuracy"]
                        - fine_metrics["overall"]["accuracy"]
                    ),
                    "hard_delta_pp": 100
                    * (
                        metrics["hard"]["accuracy"]
                        - fine_metrics["hard"]["accuracy"]
                    ),
                    "focus_delta_pp": 100
                    * (
                        metrics["focus"]["accuracy"]
                        - fine_metrics["focus"]["accuracy"]
                    ),
                }
            )
        new_subject, new_class = subject_and_class_rows(
            cache, held_indices, labels, predictions, fold
        )
        subject_rows.extend(new_subject)
        class_rows.extend(new_class)
        for reference in ("p12_core", ce_method):
            rescue = int(
                np.sum(
                    (predictions[reference] != labels)
                    & (predictions[primary_method] == labels)
                )
            )
            new_error = int(
                np.sum(
                    (predictions[reference] == labels)
                    & (predictions[primary_method] != labels)
                )
            )
            rescue_rows.append(
                {
                    "inner_fold": fold,
                    "reference": reference,
                    "rescues": rescue,
                    "new_errors": new_error,
                    "net": rescue - new_error,
                }
            )
        fold_summaries[str(fold)] = {
            "train_samples": int(len(train_indices)),
            "held_samples": int(len(held_indices)),
            "train_subjects": sorted(
                np.unique(cache.subjects[train_indices]).astype(str).tolist()
            ),
            "held_subjects": sorted(
                np.unique(cache.subjects[held_indices]).astype(str).tolist()
            ),
            "models": model_info,
            "normalizer_fit_scope": "inner-train only",
            "outer_held_predictions_generated": False,
        }
        (output / f"fold_{fold}_progress.json").write_text(
            json.dumps(
                {
                    "protocol": config["protocol"],
                    "fold": fold,
                    "models": model_info,
                    "outer_held_predictions_generated": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "ablations.csv", ablation_rows)
    write_csv(output / "event_probe.csv", probe_rows)
    write_csv(output / "per_subject.csv", subject_rows)
    write_csv(output / "per_class.csv", class_rows)
    write_csv(output / "rescue_new_error.csv", rescue_rows)
    mean_metrics: dict[str, dict[str, float]] = {}
    for method in sorted({str(row["method"]) for row in fold_rows}):
        selected = [row for row in fold_rows if row["method"] == method]
        mean_metrics[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in selected[0]
            if key not in {"inner_fold", "method"}
        }
    gate = config["freeze_gate"]
    fine = mean_metrics[primary_method]
    ce = mean_metrics[ce_method]
    hard_deltas = [
        100
        * (
            float(row["hard_accuracy"])
            - next(
                float(other["hard_accuracy"])
                for other in fold_rows
                if other["inner_fold"] == row["inner_fold"]
                and other["method"] == ce_method
            )
        )
        for row in fold_rows
        if row["method"] == primary_method
    ]
    rescue_vs_ce = [
        row
        for row in rescue_rows
        if row["reference"] == ce_method
    ]
    freeze_checks = {
        "mean_hard_delta_pp": 100
        * (fine["hard_accuracy"] - ce["hard_accuracy"]),
        "positive_hard_folds": int(
            sum(value > 0 for value in hard_deltas)
        ),
        "mean_overall_delta_pp": 100
        * (fine["overall_accuracy"] - ce["overall_accuracy"]),
        "small_non_regression": bool(
            fine["small_accuracy"] >= ce["small_accuracy"]
        ),
        "balanced_non_regression": bool(
            fine["overall_balanced_accuracy"]
            >= ce["overall_balanced_accuracy"]
        ),
        "macro_f1_non_regression": bool(
            fine["overall_macro_f1"] >= ce["overall_macro_f1"]
        ),
        "rescues": int(
            sum(int(row["rescues"]) for row in rescue_vs_ce)
        ),
        "new_errors": int(
            sum(int(row["new_errors"]) for row in rescue_vs_ce)
        ),
    }
    freeze_passed = bool(
        freeze_checks["mean_hard_delta_pp"]
        >= float(gate["fine_minus_ce_mean_hard_pp"])
        and freeze_checks["positive_hard_folds"]
        >= int(gate["minimum_positive_hard_folds"])
        and freeze_checks["mean_overall_delta_pp"]
        >= float(gate["fine_minus_ce_mean_overall_pp"])
        and freeze_checks["small_non_regression"]
        and freeze_checks["balanced_non_regression"]
        and freeze_checks["macro_f1_non_regression"]
        and freeze_checks["rescues"] > freeze_checks["new_errors"]
    )
    summary = {
        "protocol": config["protocol"],
        "status": "complete",
        "device": str(device),
        "outer_train_samples": int(outer_train_mask.sum()),
        "outer_held_samples_excluded": int((~outer_train_mask).sum()),
        "outer_held_predictions_generated": False,
        "hypothesis": config["development_hypothesis"],
        "primary_method": primary_method,
        "ce_method": ce_method,
        "shared_seed_across_variants": True,
        "mean_metrics": mean_metrics,
        "folds": fold_summaries,
        "freeze_checks": freeze_checks,
        "freeze_gate_passed": freeze_passed,
        "next_action": (
            "eligible_to_freeze_for_one_outer_held_evaluation"
            if freeze_passed
            else "do_not_evaluate_outer_held; diagnose and iterate"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
