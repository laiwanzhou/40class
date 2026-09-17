from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from p22_feature_fusion_model import (
    MODALITY_ORDER,
    build_p22_model,
    parameter_count,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
MODEL_NAMES = ("P22-L", "P22-F", "P22-U")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single preregistered three-fold P22-A head experiment. "
            "Encoders remain frozen because this script reads cached features only."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "p22_joint_pooled_fusion.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "experiment",
    )
    return parser.parse_args()


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def tensor_batch(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = {
        modality: torch.from_numpy(
            cache[f"{modality}_embedding"][indices].astype(np.float32)
        ).to(device)
        for modality in MODALITY_ORDER
    }
    modality_logits = torch.from_numpy(
        cache["per_modality_logits"][indices].astype(np.float32)
    ).to(device)
    presence = torch.from_numpy(
        cache["presence"][indices].astype(np.float32)
    ).to(device)
    labels = torch.from_numpy(cache["labels"][indices].astype(np.int64)).to(device)
    return embeddings, modality_logits, presence, labels


def apply_modality_dropout(
    embeddings: dict[str, torch.Tensor],
    modality_logits: torch.Tensor,
    presence: torch.Tensor,
    imu_probability: float,
    thermal_probability: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    augmented_presence = presence.clone()
    for modality, probability in (
        ("thermal", thermal_probability),
        ("imu", imu_probability),
    ):
        index = MODALITY_ORDER.index(modality)
        drop = (
            torch.rand(
                len(augmented_presence),
                device=augmented_presence.device,
            )
            < probability
        ) & (augmented_presence[:, index] > 0)
        augmented_presence[drop, index] = 0.0
    augmented_embeddings = {
        modality: value * augmented_presence[:, index : index + 1]
        for index, (modality, value) in enumerate(embeddings.items())
    }
    augmented_logits = modality_logits * augmented_presence.unsqueeze(-1)
    return augmented_embeddings, augmented_logits, augmented_presence


def infer(
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            embeddings, logits, presence, _ = tensor_batch(
                cache, batch_indices, device
            )
            outputs.append(
                model(embeddings, logits, presence).detach().cpu().numpy()
            )
    return np.concatenate(outputs).astype(np.float32), time.perf_counter() - started


def train_one(
    model_name: str,
    fold: int,
    cache: dict[str, np.ndarray],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_config = config["training"]
    seed = int(train_config["seed"]) + fold
    seed_everything(seed)
    model = build_p22_model(
        model_name,
        projection_dim=int(config["models"]["P22-F"]["projection_dim"]),
        hidden_dim=int(config["models"]["P22-F"]["hidden_dim"]),
        dropout=float(config["models"]["P22-F"]["dropout"]),
    ).to(device)
    parameters = parameter_count(model)
    if model_name == "P22-L":
        feature_parameters = parameter_count(
            build_p22_model(
                "P22-F",
                projection_dim=int(config["models"]["P22-F"]["projection_dim"]),
                hidden_dim=int(config["models"]["P22-F"]["hidden_dim"]),
                dropout=float(config["models"]["P22-F"]["dropout"]),
            )
        )
        if parameters > feature_parameters:
            raise ValueError("P22-L has more parameters than P22-F")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    epochs = int(train_config["epochs"])
    batch_size = int(train_config["batch_size"])
    train_indices = np.flatnonzero(cache["is_outer_train"] > 0)
    val_indices = np.flatnonzero(cache["folds"] == fold)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, Any]] = []
    started = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        for start in range(0, len(order), batch_size):
            indices = train_indices[order[start : start + batch_size]]
            embeddings, logits, presence, labels = tensor_batch(
                cache, indices, device
            )
            embeddings, logits, presence = apply_modality_dropout(
                embeddings,
                logits,
                presence,
                imu_probability=float(train_config["imu_modality_dropout"]),
                thermal_probability=float(
                    train_config["thermal_modality_dropout"]
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(embeddings, logits, presence)
            loss = F.cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
            epoch_correct += int((predictions.argmax(1) == labels).sum().item())
            epoch_samples += len(indices)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            record = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(epoch_samples, 1),
                "train_accuracy": epoch_correct / max(epoch_samples, 1),
            }
            history.append(record)
            print(
                f"{model_name} fold={fold} epoch={epoch:03d}/{epochs} "
                f"loss={record['train_loss']:.5f} "
                f"acc={record['train_accuracy']:.4f}",
                flush=True,
            )

    val_logits, inference_seconds = infer(
        model, cache, val_indices, batch_size, device
    )
    val_labels = cache["labels"][val_indices].astype(np.int64)
    val_predictions = val_logits.argmax(1)
    peak_vram_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )

    fold_dir = output_dir / model_name / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = fold_dir / "final.pt"
    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_name": model_name,
        "outer_fold": fold,
        "epoch": epochs,
        "fixed_final_epoch": True,
        "config": config,
        "cache_code_commit": str(cache["code_commit"].item()),
    }
    torch.save(checkpoint, checkpoint_path)
    np.savez_compressed(
        fold_dir / "val_logits.npz",
        sample_ids=cache["sample_ids"][val_indices],
        labels=val_labels,
        subjects=cache["subjects"][val_indices],
        folds=cache["folds"][val_indices],
        presence=cache["presence"][val_indices],
        logits=val_logits.astype(np.float16),
        predictions=val_predictions.astype(np.int64),
    )
    summary = {
        "model": model_name,
        "fold": fold,
        "fixed_final_epoch": epochs,
        "held_fold_used_for_selection": False,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "parameters": parameters,
        "fp32_parameter_mib": parameters * 4 / 1024**2,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics": metrics(val_labels, val_predictions),
        "inference_seconds": inference_seconds,
        "peak_vram_mib": peak_vram_mib,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (fold_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_logits, summary


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing P22 experiment: {output_dir}"
        )
    preflight_path = cache_dir / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(
            "Stage-one preflight is missing; training is forbidden"
        )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("training_started"):
        raise RuntimeError("Stage-one preflight did not pass")
    if sorted(preflight.get("folds_built", [])) != [0, 1, 2]:
        raise RuntimeError("Stage-one caches are incomplete")
    commit = current_commit()
    if preflight.get("code_commit") != commit:
        raise RuntimeError(
            "Current code commit differs from the preflight cache commit; "
            "regenerate stage-one caches before training"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P22-A training requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    canonical_ids: np.ndarray | None = None
    canonical_labels: np.ndarray | None = None
    canonical_subjects: np.ndarray | None = None
    canonical_folds: np.ndarray | None = None
    canonical_presence: np.ndarray | None = None
    oof_logits = {
        model_name: np.zeros((2914, 40), dtype=np.float32)
        for model_name in MODEL_NAMES
    }
    fold_summaries: list[dict[str, Any]] = []
    total_started = time.time()

    for fold in range(3):
        cache_path = cache_dir / f"fold_{fold}_cache.npz"
        cache = load_cache(cache_path)
        if str(cache["code_commit"].item()) != commit:
            raise RuntimeError(f"Cache commit mismatch: {cache_path}")
        if canonical_ids is None:
            canonical_ids = cache["sample_ids"].copy()
            canonical_labels = cache["labels"].copy()
            canonical_subjects = cache["subjects"].copy()
            canonical_folds = cache["folds"].copy()
            canonical_presence = cache["presence"].copy()
        else:
            for key, expected in (
                ("sample_ids", canonical_ids),
                ("labels", canonical_labels),
                ("subjects", canonical_subjects),
                ("folds", canonical_folds),
                ("presence", canonical_presence),
            ):
                if not np.array_equal(cache[key], expected):
                    raise ValueError(f"Fold-conditioned cache mismatch: {key}")
        held_indices = np.flatnonzero(cache["folds"] == fold)
        for model_name in MODEL_NAMES:
            logits, summary = train_one(
                model_name,
                fold,
                cache,
                config,
                output_dir,
                device,
            )
            oof_logits[model_name][held_indices] = logits
            fold_summaries.append(summary)

    assert canonical_ids is not None
    assert canonical_labels is not None
    assert canonical_subjects is not None
    assert canonical_folds is not None
    assert canonical_presence is not None
    for model_name in MODEL_NAMES:
        predictions = oof_logits[model_name].argmax(1)
        np.savez_compressed(
            output_dir / f"{model_name}_oof_logits.npz",
            sample_ids=canonical_ids,
            labels=canonical_labels,
            subjects=canonical_subjects,
            folds=canonical_folds,
            presence=canonical_presence,
            logits=oof_logits[model_name].astype(np.float16),
            predictions=predictions.astype(np.int64),
            fixed_config=np.asarray(json.dumps(config, sort_keys=True)),
            code_commit=np.asarray(commit),
        )
    summary = {
        "status": "fixed_three_fold_training_complete",
        "protocol": config["protocol_version"],
        "code_commit": commit,
        "device": str(device),
        "held_fold_used_for_selection": False,
        "models": {
            model_name: metrics(
                canonical_labels, oof_logits[model_name].argmax(1)
            )
            for model_name in MODEL_NAMES
        },
        "folds": fold_summaries,
        "elapsed_seconds": time.time() - total_started,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
