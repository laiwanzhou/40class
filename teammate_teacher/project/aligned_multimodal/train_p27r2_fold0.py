from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from p27r2_event_data import load_event_cache, write_csv
from p27r2_model import P27R2ResidualModel, parameter_count


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "p27_r2_fold0.json"
DEFAULT_CACHE = PROJECT_DIR / "runs" / "p27_r2_event_audit" / "event_cache_v2.npz"
DEFAULT_AUDIT = PROJECT_DIR / "runs" / "p27_r2_event_audit" / "summary.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_r2_fold0"
DEFAULT_P12 = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"

SMALL_IDS = np.asarray(
    [1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39]
)
HARD_IDS = np.asarray(
    [7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 35, 37, 38, 39]
)
FOCUS_IDS = np.asarray([19, 24, 25, 26, 37, 9, 10, 21, 22])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P27-R2 legal fold-0 pilot")
    parser.add_argument("--stage", choices=("dev", "final"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=DEFAULT_OUTPUT / "frozen_final_config.json",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dense_log_probabilities(
    model: ExtraTreesClassifier, features: np.ndarray
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    dense = np.full((len(features), 40), 1e-6, dtype=np.float32)
    dense[:, model.classes_.astype(np.int64)] = probabilities.astype(np.float32)
    dense /= dense.sum(axis=1, keepdims=True)
    return np.log(np.clip(dense, 1e-6, 1.0))


def fit_core(
    cache,
    train: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
    seed: int,
    save_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    core = config["p12_core"]
    sources = {
        "skeleton": (cache.core_skeleton_features, cache.modality_mask[:, 0] > 0),
        "depth": (cache.core_depth_features, cache.modality_mask[:, 2] > 0),
        "imu": (cache.core_imu_features, cache.modality_mask[:, 1] > 0),
    }
    logits = {}
    target_presence = {}
    sizes = {}
    timings = {}
    for source_index, (name, (features, presence)) in enumerate(sources.items()):
        fit_mask = train & presence
        model = ExtraTreesClassifier(
            n_estimators=int(core["n_estimators"]),
            max_depth=int(core["max_depth"]),
            min_samples_leaf=int(core["min_samples_leaf"]),
            max_features=str(core["max_features"]),
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + source_index,
        )
        started = time.perf_counter()
        model.fit(features[fit_mask], cache.labels[fit_mask])
        target_presence[name] = presence[target]
        source_logits = np.zeros((int(target.sum()), 40), dtype=np.float32)
        if target_presence[name].any():
            source_logits[target_presence[name]] = dense_log_probabilities(
                model, features[target][target_presence[name]]
            )
        logits[name] = source_logits
        timings[name] = time.perf_counter() - started
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"{name}.joblib"
            joblib.dump(model, path, compress=3)
            sizes[name] = path.stat().st_size
        del model
    numerator = np.zeros((int(target.sum()), 40), dtype=np.float32)
    denominator = np.zeros((int(target.sum()), 1), dtype=np.float32)
    for name, weight in core["weights"].items():
        present = target_presence[name].astype(np.float32)[:, None]
        numerator += float(weight) * present * logits[name]
        denominator += float(weight) * present
    no_core_modality = denominator[:, 0] <= 0
    if no_core_modality.any():
        counts = np.bincount(cache.labels[train], minlength=40).astype(np.float64) + 1.0
        prior = counts / counts.sum()
        numerator[no_core_modality] = np.log(prior.astype(np.float32))
        denominator[no_core_modality] = 1.0
    fused = numerator / denominator
    return fused, {
        "fit_samples": int(train.sum()),
        "target_samples": int(target.sum()),
        "prior_fallback_no_core_modality": int(no_core_modality.sum()),
        "timings_seconds": timings,
        "saved_model_bytes": sizes,
    }


def nested_core_predictions(
    cache,
    train: np.ndarray,
    held: np.ndarray,
    config: dict[str, Any],
    seed: int,
    save_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    oof = np.zeros((int(train.sum()), 40), dtype=np.float32)
    train_positions = np.flatnonzero(train)
    lookup = {value: index for index, value in enumerate(train_positions.tolist())}
    summaries = []
    train_subjects = sorted(np.unique(cache.subjects[train]).tolist())
    for subject_index, subject in enumerate(train_subjects):
        target = train & (cache.subjects == subject)
        source = train & (cache.subjects != subject)
        prediction, summary = fit_core(
            cache, source, target, config, seed + subject_index * 10
        )
        positions = [lookup[value] for value in np.flatnonzero(target).tolist()]
        oof[np.asarray(positions)] = prediction
        summaries.append({"held_subject": subject, **summary})
    held_logits, held_summary = fit_core(
        cache, train, held, config, seed + 100, save_dir=save_dir
    )
    return oof, held_logits, {
        "prediction_protocol": "leave-one-subject-out within residual-train",
        "oof_core": summaries,
        "held_core": held_summary,
    }


def outer_train_core_predictions(
    cache,
    outer_train: np.ndarray,
    outer_held: np.ndarray,
    config: dict[str, Any],
    seed: int,
    save_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    outer_indices = np.flatnonzero(outer_train)
    lookup = {value: index for index, value in enumerate(outer_indices.tolist())}
    oof = np.zeros((len(outer_indices), 40), dtype=np.float32)
    summaries = []
    train_subjects = sorted(np.unique(cache.subjects[outer_train]).tolist())
    for subject_index, subject in enumerate(train_subjects):
        target = outer_train & (cache.subjects == subject)
        train = outer_train & (cache.subjects != subject)
        prediction, summary = fit_core(
            cache, train, target, config, seed + subject_index * 10
        )
        positions = [lookup[value] for value in np.flatnonzero(target).tolist()]
        oof[np.asarray(positions)] = prediction
        summaries.append(
            {
                "held_subject": subject,
                "train_subjects": sorted(np.unique(cache.subjects[train]).tolist()),
                **summary,
            }
        )
    held_logits, held_summary = fit_core(
        cache,
        outer_train,
        outer_held,
        config,
        seed + 100,
        save_dir=save_dir,
    )
    return oof, held_logits, {
        "inner_oof": summaries,
        "outer_held": held_summary,
    }


def fit_normalizers(cache, train: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for name, values, present in (
        ("skeleton", cache.skeleton_tokens, cache.modality_mask[:, 0] > 0),
        ("imu", cache.imu_tokens, cache.modality_mask[:, 1] > 0),
        (
            "visual",
            cache.visual_tokens,
            np.maximum(cache.modality_mask[:, 2], cache.modality_mask[:, 3]) > 0,
        ),
    ):
        selected = values[train & present].reshape(-1, values.shape[2])
        mean = selected.mean(axis=0).astype(np.float32)
        std = np.maximum(selected.std(axis=0), 1e-4).astype(np.float32)
        output[name] = (mean, std)
    return output


def normalize_tokens(values: np.ndarray, normalizer: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = normalizer
    return ((values - mean[None, None]) / std[None, None]).astype(np.float32)


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def metric_bundle(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "overall": metric_dict(labels, predictions),
        "small": metric_dict(labels[np.isin(labels, SMALL_IDS)], predictions[np.isin(labels, SMALL_IDS)]),
        "hard": metric_dict(labels[np.isin(labels, HARD_IDS)], predictions[np.isin(labels, HARD_IDS)]),
        "focus": metric_dict(labels[np.isin(labels, FOCUS_IDS)], predictions[np.isin(labels, FOCUS_IDS)]),
    }


def prepare_tensors(
    cache,
    selected: np.ndarray,
    base_logits: np.ndarray,
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]],
    event_indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "skeleton": torch.from_numpy(
            normalize_tokens(cache.skeleton_tokens[selected], normalizers["skeleton"])
        ).to(device),
        "imu": torch.from_numpy(
            normalize_tokens(cache.imu_tokens[selected], normalizers["imu"])
        ).to(device),
        "visual": torch.from_numpy(
            normalize_tokens(cache.visual_tokens[selected], normalizers["visual"])
        ).to(device),
        "modality_mask": torch.from_numpy(cache.modality_mask[selected].astype(np.float32)).to(device),
        "base_logits": torch.from_numpy(base_logits.astype(np.float32)).to(device),
        "labels": torch.from_numpy(cache.labels[selected].astype(np.int64)).to(device),
        "event_target": torch.from_numpy(
            cache.event_targets[selected][:, event_indices].astype(np.float32)
        ).to(device),
        "event_quality": torch.from_numpy(
            cache.event_quality[selected][:, event_indices].astype(np.float32)
        ).to(device),
    }


def forward_model(model, tensors, indices: torch.Tensor | None = None, **kwargs):
    if indices is None:
        indices = torch.arange(len(tensors["labels"]), device=tensors["labels"].device)
    return model(
        tensors["skeleton"][indices],
        tensors["imu"][indices],
        tensors["visual"][indices],
        tensors["modality_mask"][indices],
        tensors["base_logits"][indices],
        **kwargs,
    )


@torch.inference_mode()
def evaluate_model(
    model: P27R2ResidualModel,
    tensors: dict[str, torch.Tensor],
    event_names: list[str],
) -> dict[str, Any]:
    model.eval()
    output = forward_model(model, tensors)
    logits = output["logits"].float().cpu().numpy()
    predictions = logits.argmax(axis=1)
    labels = tensors["labels"].cpu().numpy()
    event_prediction = output["event_prediction"].float().cpu().numpy()
    event_target = tensors["event_target"].cpu().numpy()
    event_quality = tensors["event_quality"].cpu().numpy()
    event_metrics = {}
    for index, name in enumerate(event_names):
        valid = event_quality[:, index] > 0
        if not valid.any():
            continue
        event_metrics[name] = {
            "samples": int(valid.sum()),
            "mae": float(np.mean(np.abs(event_prediction[valid, index] - event_target[valid, index]))),
        }
    return {
        "logits": logits,
        "predictions": predictions,
        "metrics": metric_bundle(labels, predictions),
        "event_metrics": event_metrics,
        "alpha": float(output["alpha"].cpu()),
        "event_prediction": event_prediction,
    }


def train_residual(
    cache,
    train: np.ndarray,
    held: np.ndarray,
    train_base: np.ndarray,
    held_base: np.ndarray,
    event_names: list[str],
    config: dict[str, Any],
    event_supervision: bool,
    seed: int,
    save_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], P27R2ResidualModel, dict[str, Any]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizers = fit_normalizers(cache, train)
    name_lookup = {name: index for index, name in enumerate(cache.event_names.astype(str))}
    event_indices = np.asarray([name_lookup[name] for name in event_names], dtype=np.int64)
    residual = config["residual"]
    target_weights = torch.tensor(
        [
            float(residual.get("event_target_weights", {}).get(name, 1.0))
            for name in event_names
        ],
        dtype=torch.float32,
        device=device,
    )
    train_tensor = prepare_tensors(
        cache, train, train_base, normalizers, event_indices, device
    )
    held_tensor = prepare_tensors(
        cache, held, held_base, normalizers, event_indices, device
    )
    model = P27R2ResidualModel(
        skeleton_dim=cache.skeleton_tokens.shape[2],
        imu_dim=cache.imu_tokens.shape[2],
        visual_dim=cache.visual_tokens.shape[2],
        event_count=len(event_names),
        hidden=int(residual["hidden_channels"]),
        event_dim=int(residual["event_dim"]),
        dropout=float(residual["dropout"]),
        alpha_limit=float(residual["alpha_limit"]),
        delta_limit=float(residual["delta_logit_limit"]),
    ).to(device)
    model.eval()
    with torch.inference_mode():
        initial = forward_model(model, held_tensor)["logits"]
        initial_error = float(
            torch.max(torch.abs(initial - held_tensor["base_logits"])).cpu()
        )
    if initial_error > 1e-7:
        raise RuntimeError(f"zero residual check failed: {initial_error}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(residual["learning_rate"]),
        weight_decay=float(residual["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(residual["epochs"])
    )
    labels = train_tensor["labels"]
    counts = torch.bincount(labels, minlength=40).float().clamp_min(1.0)
    weights = 1.0 / counts[labels]
    generator = torch.Generator(device=device).manual_seed(seed)
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(residual["epochs"]) + 1):
        model.train()
        sampled = torch.multinomial(
            weights, len(weights), replacement=True, generator=generator
        )
        loss_sum = ce_sum = event_sum = 0.0
        count = 0
        for start in range(0, len(sampled), int(residual["batch_size"])):
            indices = sampled[start : start + int(residual["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            output = forward_model(model, train_tensor, indices)
            ce = F.cross_entropy(
                output["logits"],
                labels[indices],
                label_smoothing=float(residual["label_smoothing"]),
            )
            if event_supervision:
                error = F.smooth_l1_loss(
                    output["event_prediction"],
                    train_tensor["event_target"][indices],
                    reduction="none",
                )
                quality = (
                    train_tensor["event_quality"][indices]
                    * target_weights.unsqueeze(0)
                )
                event_loss = (error * quality).sum() / quality.sum().clamp_min(1.0)
            else:
                event_loss = ce.new_zeros(())
            base_soft = torch.softmax(train_tensor["base_logits"][indices], dim=1)
            base_anchor = F.kl_div(
                F.log_softmax(output["logits"], dim=1),
                base_soft,
                reduction="batchmean",
            )
            loss = (
                ce
                + float(residual["event_loss_weight"]) * event_loss
                + float(residual["base_kl_anchor_weight"]) * base_anchor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch = len(indices)
            loss_sum += float(loss.detach()) * batch
            ce_sum += float(ce.detach()) * batch
            event_sum += float(event_loss.detach()) * batch
            count += batch
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(count, 1),
                "ce_loss": ce_sum / max(count, 1),
                "event_loss": event_sum / max(count, 1),
                "alpha": float(
                    float(residual["alpha_limit"])
                    * torch.tanh(model.alpha_raw.detach()).cpu()
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    train_seconds = time.perf_counter() - started
    train_result = evaluate_model(model, train_tensor, event_names)
    held_result = evaluate_model(model, held_tensor, event_names)
    for event_index, event_name in enumerate(event_names):
        train_quality = train_tensor["event_quality"][:, event_index].cpu().numpy()
        held_quality = held_tensor["event_quality"][:, event_index].cpu().numpy()
        train_valid = train_quality > 0
        held_valid = held_quality > 0
        if not train_valid.any() or not held_valid.any():
            continue
        train_targets = train_tensor["event_target"][:, event_index].cpu().numpy()
        held_targets = held_tensor["event_target"][:, event_index].cpu().numpy()
        train_mean = float(
            np.average(train_targets[train_valid], weights=train_quality[train_valid])
        )
        baseline_mae = float(np.mean(np.abs(held_targets[held_valid] - train_mean)))
        model_mae = float(held_result["event_metrics"][event_name]["mae"])
        held_result["event_metrics"][event_name].update(
            {
                "train_mean_baseline_mae": baseline_mae,
                "relative_improvement": float(
                    (baseline_mae - model_mae) / max(baseline_mae, 1e-8)
                ),
            }
        )
    model_info = {
        "parameters": parameter_count(model),
        "fp16_parameter_mib": parameter_count(model) * 2 / 1024**2,
        "initial_max_abs_logit_error": initial_error,
        "train_seconds": train_seconds,
        "history": history,
        "normalizers": {
            name: {"mean": mean.tolist(), "std": std.tolist()}
            for name, (mean, std) in normalizers.items()
        },
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "event_names": event_names,
                "normalizers": normalizers,
                "config": config,
                "event_supervision": event_supervision,
                "fixed_final_epoch": int(residual["epochs"]),
            },
            save_path,
        )
        model_info["checkpoint"] = str(save_path)
        model_info["checkpoint_bytes"] = save_path.stat().st_size
    return train_result, held_result, model, model_info


def class_rows(cache, selected: np.ndarray, methods: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    labels = cache.labels[selected]
    for class_id in range(40):
        class_mask = labels == class_id
        row: dict[str, Any] = {"class_id": class_id, "samples": int(class_mask.sum())}
        for name, prediction in methods.items():
            row[f"{name}_recall"] = (
                float(np.mean(prediction[class_mask] == class_id)) if class_mask.any() else None
            )
        rows.append(row)
    return rows


def subject_rows(cache, selected: np.ndarray, methods: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    labels = cache.labels[selected]
    subjects = cache.subjects[selected]
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        row: dict[str, Any] = {"subject": subject, "samples": int(mask.sum())}
        for name, prediction in methods.items():
            row[f"{name}_accuracy"] = float(np.mean(prediction[mask] == labels[mask]))
        rows.append(row)
    return rows


def development_stage(cache, config: dict[str, Any], event_names: list[str], output: Path) -> dict[str, Any]:
    outer_train = cache.outer_folds != int(config["outer_fold"])
    folds = {}
    ledger = []
    for fold_name, held_subjects in config["inner_splits"].items():
        held = outer_train & np.isin(cache.subjects, held_subjects)
        train = outer_train & ~np.isin(cache.subjects, held_subjects)
        core_cache_path = output / f"development_core_loso_fold_{fold_name}.npz"
        if core_cache_path.is_file():
            with np.load(core_cache_path, allow_pickle=False) as source:
                expected_train = cache.sample_ids[train].astype(str)
                expected_held = cache.sample_ids[held].astype(str)
                if (
                    source["protocol"].item()
                    != "p12-core-et-subject-loso-v1"
                    or not np.array_equal(source["train_sample_ids"].astype(str), expected_train)
                    or not np.array_equal(source["held_sample_ids"].astype(str), expected_held)
                ):
                    raise ValueError(f"stale development core cache: {core_cache_path}")
                base_train = source["train_logits"].astype(np.float32)
                base_held = source["held_logits"].astype(np.float32)
                base_summary = json.loads(source["summary_json"].item())
        else:
            base_train, base_held, base_summary = nested_core_predictions(
                cache, train, held, config, int(config["seed"]) + 1000 * int(fold_name)
            )
            np.savez_compressed(
                core_cache_path,
                protocol=np.asarray("p12-core-et-subject-loso-v1"),
                train_sample_ids=cache.sample_ids[train],
                held_sample_ids=cache.sample_ids[held],
                train_logits=base_train,
                held_logits=base_held,
                summary_json=np.asarray(json.dumps(base_summary)),
            )
        base_prediction = base_held.argmax(axis=1)
        ce_train, ce_held, _, ce_info = train_residual(
            cache,
            train,
            held,
            base_train,
            base_held,
            event_names,
            config,
            False,
            int(config["seed"]) + 100 * int(fold_name),
        )
        event_train, event_held, _, event_info = train_residual(
            cache,
            train,
            held,
            base_train,
            base_held,
            event_names,
            config,
            True,
            int(config["seed"]) + 100 * int(fold_name),
        )
        base_metrics = metric_bundle(cache.labels[held], base_prediction)
        folds[fold_name] = {
            "held_subjects": held_subjects,
            "train_samples": int(train.sum()),
            "held_samples": int(held.sum()),
            "base": base_metrics,
            "ce_only": ce_held["metrics"],
            "event": event_held["metrics"],
            "event_prediction": event_held["event_metrics"],
            "base_protocol": base_summary,
            "ce_model": ce_info,
            "event_model": event_info,
        }
        ledger.append(
            {
                "iteration": f"inner-{fold_name}",
                "hypothesis": "full-sequence invariant event supervision improves the same residual architecture",
                "base_overall": base_metrics["overall"]["accuracy"],
                "ce_overall": ce_held["metrics"]["overall"]["accuracy"],
                "event_overall": event_held["metrics"]["overall"]["accuracy"],
                "base_hard": base_metrics["hard"]["accuracy"],
                "ce_hard": ce_held["metrics"]["hard"]["accuracy"],
                "event_hard": event_held["metrics"]["hard"]["accuracy"],
                "event_minus_ce_overall_pp": 100
                * (
                    event_held["metrics"]["overall"]["accuracy"]
                    - ce_held["metrics"]["overall"]["accuracy"]
                ),
                "event_minus_ce_hard_pp": 100
                * (
                    event_held["metrics"]["hard"]["accuracy"]
                    - ce_held["metrics"]["hard"]["accuracy"]
                ),
            }
        )
    write_csv(output / "development_ledger.csv", ledger)
    event_overall_gain = [
        float(row["event_minus_ce_overall_pp"]) for row in ledger
    ]
    event_hard_gain = [float(row["event_minus_ce_hard_pp"]) for row in ledger]
    strongly_weighted = [
        name
        for name in event_names
        if float(config["residual"]["event_target_weights"].get(name, 1.0)) >= 0.99
    ]
    strong_prediction_pass = all(
        all(
            float(folds[fold_name]["event_prediction"][name]["relative_improvement"])
            > 0.0
            for fold_name in folds
        )
        for name in strongly_weighted
    )
    gate = bool(
        all(value >= -1.0 for value in event_overall_gain)
        and all(value >= -1.0 for value in event_hard_gain)
        and np.mean(event_hard_gain) > 0.0
        and any(value > 0.0 for value in event_hard_gain)
        and strong_prediction_pass
    )
    return {
        "protocol": config["protocol"],
        "status": "complete",
        "outer_held_predictions_generated": False,
        "passed_events": event_names,
        "folds": folds,
        "gate_to_freeze_final": gate,
        "mean_event_minus_ce_overall_pp": float(np.mean(event_overall_gain)),
        "mean_event_minus_ce_hard_pp": float(np.mean(event_hard_gain)),
        "strong_event_prediction_pass": strong_prediction_pass,
        "strongly_weighted_events": strongly_weighted,
    }


@torch.inference_mode()
def ablations(
    model: P27R2ResidualModel,
    tensors: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    model.eval()
    labels = tensors["labels"].cpu().numpy()
    generator = torch.Generator(device=tensors["labels"].device).manual_seed(seed)
    permutation = torch.randperm(len(labels), generator=generator, device=tensors["labels"].device)
    standard = forward_model(model, tensors)
    shuffled = forward_model(model, tensors, event_shuffle=permutation)

    imu_tensors = {key: value.clone() if torch.is_tensor(value) else value for key, value in tensors.items()}
    imu_tensors["imu"].zero_()
    imu_tensors["modality_mask"][:, 1] = 0
    imu_zero = forward_model(model, imu_tensors)

    ir_tensors = {key: value.clone() if torch.is_tensor(value) else value for key, value in tensors.items()}
    ir_tensors["visual"][:, :, 48:96] = 0
    ir_tensors["visual"][:, :, 99:102] = 0
    ir_tensors["modality_mask"][:, 3] = 0
    ir_zero = forward_model(model, ir_tensors)

    base_predictions = tensors["base_logits"].argmax(1).cpu().numpy()
    return {
        "standard": metric_bundle(labels, standard["logits"].argmax(1).cpu().numpy()),
        "event_representation_shuffle": metric_bundle(
            labels, shuffled["logits"].argmax(1).cpu().numpy()
        ),
        "event_representation_zero_delta": metric_bundle(labels, base_predictions),
        "imu_zero": metric_bundle(labels, imu_zero["logits"].argmax(1).cpu().numpy()),
        "ir_zero": metric_bundle(labels, ir_zero["logits"].argmax(1).cpu().numpy()),
    }


def final_stage(
    cache,
    config: dict[str, Any],
    frozen: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if not frozen.get("configuration_frozen", False):
        raise RuntimeError("Final held evaluation requires configuration_frozen=true")
    event_names = list(frozen["passed_events"])
    outer_train = cache.outer_folds != int(config["outer_fold"])
    outer_held = ~outer_train
    model_dir = output / "models"
    base_train, base_held, base_summary = outer_train_core_predictions(
        cache,
        outer_train,
        outer_held,
        config,
        int(config["seed"]) + 9000,
        model_dir / "p12_core",
    )
    base_prediction = base_held.argmax(axis=1)
    ce_train, ce_held, ce_model, ce_info = train_residual(
        cache,
        outer_train,
        outer_held,
        base_train,
        base_held,
        event_names,
        config,
        False,
        int(config["seed"]) + 9100,
        model_dir / "ce_only_residual.pt",
    )
    event_train, event_held, event_model, event_info = train_residual(
        cache,
        outer_train,
        outer_held,
        base_train,
        base_held,
        event_names,
        config,
        True,
        int(config["seed"]) + 9100,
        model_dir / "event_residual.pt",
    )
    normalizers = fit_normalizers(cache, outer_train)
    event_lookup = {name: index for index, name in enumerate(cache.event_names.astype(str))}
    event_indices = np.asarray([event_lookup[name] for name in event_names], dtype=np.int64)
    device = next(event_model.parameters()).device
    held_tensors = prepare_tensors(
        cache, outer_held, base_held, normalizers, event_indices, device
    )
    ablation = ablations(event_model, held_tensors, int(config["seed"]) + 9200)
    labels = cache.labels[outer_held]
    methods = {
        "p12_core": base_prediction,
        "ce_only": ce_held["predictions"],
        "event": event_held["predictions"],
    }
    write_csv(output / "per_class.csv", class_rows(cache, outer_held, methods))
    write_csv(output / "per_subject.csv", subject_rows(cache, outer_held, methods))
    prediction_rows = []
    for sample_id, subject, label, base, ce, event in zip(
        cache.sample_ids[outer_held],
        cache.subjects[outer_held],
        labels,
        methods["p12_core"],
        methods["ce_only"],
        methods["event"],
        strict=True,
    ):
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "label": int(label),
                "p12_core_prediction": int(base),
                "ce_only_prediction": int(ce),
                "event_prediction": int(event),
            }
        )
    write_csv(output / "fold0_predictions.csv", prediction_rows)
    np.savez_compressed(
        output / "fold0_logits.npz",
        sample_ids=cache.sample_ids[outer_held],
        labels=labels,
        p12_core_logits=base_held,
        ce_only_logits=ce_held["logits"],
        event_logits=event_held["logits"],
    )
    rescues = int(
        np.sum((methods["ce_only"] != labels) & (methods["event"] == labels))
    )
    new_errors = int(
        np.sum((methods["ce_only"] == labels) & (methods["event"] != labels))
    )

    p12_reference = {}
    with np.load(DEFAULT_P12, allow_pickle=False) as source:
        p12_ids = source["sample_ids"].astype(str)
        p12_labels = source["labels"].astype(np.int64)
        logits_key = (
            "final_logits"
            if "final_logits" in source.files
            else "thermal_routed_final_logits"
        )
        p12_logits = source[logits_key]
        lookup = {sample_id: index for index, sample_id in enumerate(p12_ids.tolist())}
        selected_indices = [
            lookup[sample_id]
            for sample_id in cache.sample_ids[outer_held].astype(str)
            if sample_id in lookup
        ]
        p12_prediction = p12_logits[selected_indices].argmax(axis=1)
        p12_reference = {
            "scope": "read-only full P12 common sample reference",
            "samples": len(selected_indices),
            "metrics": metric_bundle(p12_labels[selected_indices], p12_prediction),
        }

    return {
        "protocol": config["protocol"],
        "status": "complete",
        "held_evaluation_count": 1,
        "passed_events": event_names,
        "metrics": {
            "p12_core": metric_bundle(labels, methods["p12_core"]),
            "ce_only": ce_held["metrics"],
            "event": event_held["metrics"],
            "full_p12_read_only": p12_reference,
        },
        "event_prediction": event_held["event_metrics"],
        "rescue_new_error_vs_ce": {
            "rescues": rescues,
            "new_errors": new_errors,
            "net": rescues - new_errors,
        },
        "base_protocol": base_summary,
        "ce_model": ce_info,
        "event_model": event_info,
        "ablations": ablation,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    audit = json.loads(args.audit.resolve().read_text(encoding="utf-8"))
    cache = load_event_cache(args.cache.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    event_names = list(audit["passed_events"])
    if not event_names:
        raise RuntimeError("No R2 events passed the outer-train-only audit")
    started = time.time()
    if args.stage == "dev":
        summary = development_stage(cache, config, event_names, output)
        summary["elapsed_seconds"] = time.time() - started
        summary["event_cache_sha256"] = sha256(args.cache.resolve())
        (output / "development_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        frozen = json.loads(args.frozen_config.resolve().read_text(encoding="utf-8"))
        summary = final_stage(cache, config, frozen, output)
        summary["elapsed_seconds"] = time.time() - started
        summary["event_cache_sha256"] = sha256(args.cache.resolve())
        (output / "final_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
