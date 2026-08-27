from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import yaml

from scripts.cache_ir_depth_videomaev2_p2a import _metrics
from src.data.motionbert_skeleton_dataset import MotionBERTSkeletonDataset
from src.experiments.motionbert_p6b_config import (
    load_motionbert_p6b_config,
    project_path,
)
from src.models.motionbert_lite_skeleton import (
    MotionBERTLiteSkeletonExpert,
    build_motionbert_lite_expert,
    set_motionbert_train_stage,
)
from third_party.motionbert import DSTformer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _supported_indices(dataset: MotionBERTSkeletonDataset, count: int) -> list[int]:
    result = []
    for index, trial in enumerate(dataset.trials):
        if trial.sample_id in dataset.lookup:
            result.append(index)
            if len(result) == count:
                break
    if len(result) != count:
        raise ValueError("MotionBERT smoke lacks supported rows")
    return result


def _batch(dataset: MotionBERTSkeletonDataset, indices: list[int]) -> dict[str, Any]:
    return next(
        iter(
            DataLoader(
                Subset(dataset, indices),
                batch_size=len(indices),
                shuffle=False,
                num_workers=0,
            )
        )
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _random_expert(config: dict[str, Any]) -> MotionBERTLiteSkeletonExpert:
    model = config["model"]
    backbone = DSTformer(
        dim_in=int(model["dim_in"]),
        dim_out=3,
        dim_feat=int(model["dim_feat"]),
        dim_rep=int(model["dim_rep"]),
        depth=int(model["depth"]),
        num_heads=int(model["num_heads"]),
        mlp_ratio=int(model["mlp_ratio"]),
        num_joints=int(model["num_joints"]),
        maxlen=int(model["maxlen"]),
        att_fuse=bool(model["att_fuse"]),
    )
    return MotionBERTLiteSkeletonExpert(
        backbone=backbone,
        dim_rep=int(model["dim_rep"]),
        classes=40,
        dropout=float(model["dropout"]),
    )


def run_motionbert_smoke(
    config_path: Path, *, output_root: Path
) -> dict[str, Any]:
    config = load_motionbert_p6b_config(config_path)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    seed = int(config["seed"])
    _set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("MotionBERT B0 requires CUDA")
    device = torch.device("cuda")
    train = MotionBERTSkeletonDataset(config, partition="train")
    validation = MotionBERTSkeletonDataset(config, partition="validation")
    train_batch = _move_batch(
        _batch(train, _supported_indices(train, int(config["b0"]["train_rows"]))),
        device,
    )
    validation_batch = _move_batch(
        _batch(
            validation,
            _supported_indices(validation, int(config["b0"]["validation_rows"])),
        ),
        device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model, coverage = build_motionbert_lite_expert(config)
    model = model.to(device)
    set_motionbert_train_stage(model, "B1")
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        pretrained_embedding = model(
            train_batch["sequence"], train_batch["available"].bool()
        )["embedding"].float()
    _set_seed(seed + 1)
    random_model = _random_expert(config).to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        random_embedding = random_model(
            train_batch["sequence"], train_batch["available"].bool()
        )["embedding"].float()
    embedding_delta = float(
        (pretrained_embedding - random_embedding).abs().max().cpu()
    )
    del random_model, random_embedding
    torch.cuda.empty_cache()

    _set_seed(seed)
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-2,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(train_batch["sequence"], train_batch["available"].bool())
        loss = F.cross_entropy(output["logits"].float(), train_batch["label"].long())
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite = bool(torch.isfinite(loss)) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    optimizer.step()
    changed = {
        "backbone": any(
            not torch.equal(before[name], parameter.detach().cpu())
            for name, parameter in model.named_parameters()
            if name.startswith("backbone.")
        ),
        "head": any(
            not torch.equal(before[name], parameter.detach().cpu())
            for name, parameter in model.named_parameters()
            if not name.startswith("backbone.")
        ),
    }
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        validation_output = model(
            validation_batch["sequence"], validation_batch["available"].bool()
        )
    finite = finite and bool(torch.isfinite(validation_output["logits"]).all())

    head_state = {
        "head_norm": {
            name: value.detach().cpu().clone()
            for name, value in model.head_norm.state_dict().items()
        },
        "classifier": {
            name: value.detach().cpu().clone()
            for name, value in model.classifier.state_dict().items()
        },
    }
    head_path = output_root / "head_state.pt"
    _atomic_torch_save(head_path, head_state)
    reloaded, _ = build_motionbert_lite_expert(config)
    saved = torch.load(head_path, map_location="cpu", weights_only=True)
    reloaded.head_norm.load_state_dict(saved["head_norm"], strict=True)
    reloaded.classifier.load_state_dict(saved["classifier"], strict=True)
    exact = all(
        torch.equal(value, reloaded.head_norm.state_dict()[name])
        for name, value in head_state["head_norm"].items()
    ) and all(
        torch.equal(value, reloaded.classifier.state_dict()[name])
        for name, value in head_state["classifier"].items()
    )
    changed_groups = [name for name in ("backbone", "head") if changed[name]]
    report = {
        "stage": "P6-B0",
        "status": "smoke_passed" if finite and changed_groups == ["head"] and exact else "smoke_failed",
        "pretrained_element_coverage": coverage.element_fraction,
        "pretrained_missing_keys": list(coverage.missing_keys),
        "pretrained_unexpected_keys": list(coverage.unexpected_keys),
        "pretrained_shape_mismatches": list(coverage.shape_mismatches),
        "finite_forward_backward": finite,
        "loss": float(loss.detach().cpu()),
        "changed_parameter_groups": changed_groups,
        "pretrained_random_embedding_max_abs_delta": embedding_delta,
        "gradient_user_ids": [str(value) for value in train_batch["user_id"]],
        "validation_forward_rows": len(validation_batch["label"]),
        "head_reload_exact": exact,
        "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "seconds": time.perf_counter() - started,
        "checkpoint_sha256": str(config["checkpoint"]["sha256"]),
        "config_sha256": _sha256(config_path),
        "clean_view_sha256": _sha256(project_path(str(config["data"]["clean_view"]))),
        "projection_sha256": train.projection_sha256,
        "source_sha256": {
            "DSTformer.py": _sha256(project_path("third_party/motionbert/DSTformer.py")),
            "drop.py": _sha256(project_path("third_party/motionbert/drop.py")),
            "adapter": _sha256(project_path("src/models/motionbert_lite_skeleton.py")),
        },
    }
    _atomic_write(
        output_root / "resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )
    _atomic_write(
        output_root / "smoke_report.json", json.dumps(report, indent=2) + "\n"
    )
    if report["status"] != "smoke_passed":
        raise RuntimeError(f"MotionBERT smoke failed: {report}")
    return report


class CachedMotionHead(nn.Module):
    def __init__(self, *, dim: int = 512, classes: int = 40, dropout: float = 0.5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.norm(embeddings)))


def _class_prior(labels: np.ndarray, classes: int = 40) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=classes).astype(np.float64)
    probabilities = (counts + 1.0) / (counts.sum() + classes)
    return np.log(probabilities).astype(np.float32)


def cache_motionbert_embeddings(
    config_path: Path, *, output_path: Path
) -> dict[str, Any]:
    config = load_motionbert_p6b_config(config_path)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if not torch.cuda.is_available():
        raise RuntimeError("MotionBERT embedding cache requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model, coverage = build_motionbert_lite_expert(config)
    model = model.to(device).eval()
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "embeddings",
            "available",
            "labels",
            "sample_ids",
            "user_ids",
            "partition",
            "quality",
        )
    }
    for partition in ("train", "validation"):
        dataset = MotionBERTSkeletonDataset(config, partition=partition)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        for batch in loader:
            available = batch["available"].bool()
            embeddings = torch.zeros(len(available), 512, dtype=torch.float32)
            if bool(available.any()):
                sequence = batch["sequence"][available].to(device)
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    output = model(
                        sequence,
                        torch.ones(len(sequence), dtype=torch.bool, device=device),
                    )
                embeddings[available] = output["backbone_embedding"].float().cpu()
            arrays["embeddings"].append(embeddings.numpy())
            arrays["available"].append(available.numpy())
            arrays["labels"].append(batch["label"].numpy())
            arrays["sample_ids"].append(np.asarray(batch["sample_id"]).astype(str))
            arrays["user_ids"].append(np.asarray(batch["user_id"]).astype(str))
            arrays["partition"].append(
                np.asarray([partition] * len(available))
            )
            arrays["quality"].append(batch["quality"].numpy())
    merged = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output_path, **merged)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "embedding_shape": list(merged["embeddings"].shape),
        "available_rows": int(merged["available"].sum()),
        "train_rows": int((merged["partition"] == "train").sum()),
        "validation_rows": int((merged["partition"] == "validation").sum()),
        "pretrained_element_coverage": coverage.element_fraction,
        "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "seconds": time.perf_counter() - started,
    }


def _predict_cached(
    head: CachedMotionHead,
    embeddings: np.ndarray,
    available: np.ndarray,
    prior_logits: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    head.eval()
    output = np.repeat(prior_logits[None], len(embeddings), axis=0)
    indices = np.flatnonzero(available)
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            current = indices[start : start + 256]
            logits = head(torch.from_numpy(embeddings[current]).to(device))
            output[current] = logits.float().cpu().numpy()
    return output.astype(np.float32)


def run_motionbert_b1(
    config_path: Path,
    *,
    output_root: Path,
    cache_path: Path,
    visual_predictions_path: Path,
    expected_train_samples: int = 2039,
    expected_validation_samples: int = 388,
) -> dict[str, Any]:
    config = load_motionbert_p6b_config(config_path)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {
            "embeddings", "available", "labels", "sample_ids", "user_ids",
            "partition", "quality",
        }
        if not required.issubset(cache.files):
            raise ValueError("MotionBERT embedding cache schema changed")
        values = {name: cache[name].copy() for name in required}
    train_mask = values["partition"].astype(str) == "train"
    validation_mask = values["partition"].astype(str) == "validation"
    if train_mask.sum() != expected_train_samples or validation_mask.sum() != expected_validation_samples:
        raise ValueError("MotionBERT B1 fixed population changed")
    if values["embeddings"].shape != (expected_train_samples + expected_validation_samples, 512):
        raise ValueError("MotionBERT cached embedding shape changed")
    train_users = set(values["user_ids"][train_mask].astype(str).tolist())
    validation_users = set(values["user_ids"][validation_mask].astype(str).tolist())
    if train_users & validation_users:
        raise ValueError("MotionBERT B1 train/validation users overlap")

    _set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = CachedMotionHead(dropout=float(config["model"]["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["b1"]["learning_rate"]),
        weight_decay=float(config["b1"]["weight_decay"]),
    )
    supported_train = train_mask & values["available"].astype(bool)
    train_dataset = TensorDataset(
        torch.from_numpy(values["embeddings"][supported_train]).float(),
        torch.from_numpy(values["labels"][supported_train]).long(),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["b1"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history = []
    for epoch in range(1, int(config["b1"]["epochs"]) + 1):
        head.train()
        loss_total = 0.0
        rows = 0
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(embeddings)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(labels)
            rows += len(labels)
        history.append({"epoch": epoch, "mean_train_loss": loss_total / max(rows, 1)})
    prior = _class_prior(values["labels"][train_mask])
    train_logits = _predict_cached(
        head,
        values["embeddings"][train_mask],
        values["available"][train_mask].astype(bool),
        prior,
        device,
    )
    validation_logits = _predict_cached(
        head,
        values["embeddings"][validation_mask],
        values["available"][validation_mask].astype(bool),
        prior,
        device,
    )
    train_labels = values["labels"][train_mask].astype(np.int64)
    validation_labels = values["labels"][validation_mask].astype(np.int64)
    train_user_ids = values["user_ids"][train_mask].astype(str)
    validation_user_ids = values["user_ids"][validation_mask].astype(str)
    train_sample_ids = values["sample_ids"][train_mask].astype(str)
    validation_sample_ids = values["sample_ids"][validation_mask].astype(str)
    with np.load(visual_predictions_path, allow_pickle=False) as visual:
        if not np.array_equal(visual["sample_ids"].astype(str), validation_sample_ids):
            raise ValueError("visual/MotionBERT validation sample order changed")
        if not np.array_equal(visual["labels"].astype(np.int64), validation_labels):
            raise ValueError("visual/MotionBERT validation labels changed")
        visual_logits = visual["logits"].astype(np.float32)
    visual_correct = visual_logits.argmax(1) == validation_labels
    motion_correct = validation_logits.argmax(1) == validation_labels
    rescued = (~visual_correct) & motion_correct
    harmed = visual_correct & (~motion_correct)
    oracle = visual_correct | motion_correct
    rescues_by_user = {
        user: int((rescued & (validation_user_ids == user)).sum())
        for user in sorted(validation_users)
    }
    metrics = _metrics(validation_labels, validation_logits, validation_user_ids)
    gates_config = config["b1"]["gates"]
    gates = {
        "accuracy": float(metrics["accuracy"]) >= float(gates_config["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]) >= float(gates_config["macro_f1"]),
        "unique_rescues": int(rescued.sum()) >= int(gates_config["unique_rescues"]),
        "visual_oracle_accuracy": float(oracle.mean()) >= float(gates_config["visual_oracle_accuracy"]),
        "unique_rescue_each_user": all(
            value >= int(gates_config["unique_rescue_each_user"])
            for value in rescues_by_user.values()
        ),
    }
    output_root.mkdir(parents=True)
    train_predictions = output_root / "train_predictions.npz"
    validation_predictions = output_root / "validation_predictions.npz"
    _atomic_npz(
        train_predictions,
        sample_ids=train_sample_ids,
        user_ids=train_user_ids,
        labels=train_labels,
        logits=train_logits,
        available=values["available"][train_mask],
    )
    _atomic_npz(
        validation_predictions,
        sample_ids=validation_sample_ids,
        user_ids=validation_user_ids,
        labels=validation_labels,
        logits=validation_logits,
        available=values["available"][validation_mask],
    )
    head_path = output_root / "head_state.pt"
    _atomic_torch_save(head_path, head.state_dict())
    report = {
        "stage": "P6-B1",
        "status": "completed",
        "epochs_completed": int(config["b1"]["epochs"]),
        "history": history,
        "train_metrics": _metrics(train_labels, train_logits, train_user_ids),
        "validation_metrics": metrics,
        "validation_evaluation_count": 1,
        "train_sample_ids": train_sample_ids.tolist(),
        "validation_sample_ids": validation_sample_ids.tolist(),
        "validation_users_entered_training": False,
        "cache": {
            "path": str(cache_path.resolve()),
            "sha256": _sha256(cache_path),
            "embedding_shape": list(values["embeddings"].shape),
        },
        "visual_comparison": {
            "unique_rescues": int(rescued.sum()),
            "harms": int(harmed.sum()),
            "net": int(rescued.sum() - harmed.sum()),
            "oracle_accuracy": float(oracle.mean()),
            "rescues_by_user": rescues_by_user,
        },
        "gates": gates,
        "b1_passed": all(gates.values()),
        "head_state": str(head_path),
        "head_state_sha256": _sha256(head_path),
        "train_predictions": str(train_predictions),
        "validation_predictions": str(validation_predictions),
    }
    _atomic_write(
        output_root / "b1_decision.json", json.dumps(report, indent=2) + "\n"
    )
    return report
