from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from src.experiments.motion_attribute_config import load_motion_attribute_config
from src.models.motion_attribute_expert import MotionAttributeExpert
from src.training.motion_attribute_loss import motion_attribute_loss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MotionAttributeCacheDataset(Dataset[dict[str, object]]):
    def __init__(self, cache_path: Path, *, partition: str) -> None:
        with np.load(cache_path, allow_pickle=False) as cache:
            values = {name: cache[name].copy() for name in cache.files}
        selected = values["partition"].astype(str) == partition
        self.features = torch.from_numpy(values["features"][selected]).float()
        self.mask = torch.from_numpy(values["mask"][selected]).bool()
        self.attributes = torch.from_numpy(values["attributes"][selected]).float()
        self.families = torch.from_numpy(values["families"][selected]).float()
        self.available = torch.from_numpy(values["available"][selected]).bool()
        self.labels = torch.from_numpy(values["labels"][selected]).long()
        self.sample_ids = values["sample_ids"][selected].astype(str)
        self.user_ids = values["user_ids"][selected].astype(str)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "features": self.features[index],
            "mask": self.mask[index],
            "attributes": self.attributes[index],
            "families": self.families[index],
            "available": self.available[index],
            "label": self.labels[index],
            "sample_id": str(self.sample_ids[index]),
            "user_id": str(self.user_ids[index]),
        }


def _supported_batch(
    dataset: MotionAttributeCacheDataset, count: int
) -> dict[str, object]:
    indices = torch.nonzero(dataset.available).flatten()[:count]
    if len(indices) != count:
        raise ValueError("motion smoke lacks supported rows")
    return {
        "features": dataset.features[indices],
        "mask": dataset.mask[indices],
        "attributes": dataset.attributes[indices],
        "families": dataset.families[indices],
        "available": dataset.available[indices],
        "label": dataset.labels[indices],
        "sample_id": dataset.sample_ids[indices.numpy()].tolist(),
        "user_id": dataset.user_ids[indices.numpy()].tolist(),
    }


def _move(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _parameter_group(name: str) -> str:
    if name.startswith("family_head."):
        return "family_head"
    if name.startswith("attribute_head."):
        return "attribute_head"
    if name.startswith("action_head."):
        return "action_head"
    return "encoder"


def run_motion_attribute_smoke(
    config_path: Path, *, cache_path: Path, output_root: Path
) -> dict[str, Any]:
    config = load_motion_attribute_config(config_path)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if not torch.cuda.is_available():
        raise RuntimeError("motion attribute smoke requires CUDA")
    output_root.mkdir(parents=True)
    _set_seed(int(config["seed"]))
    train = MotionAttributeCacheDataset(cache_path, partition="train")
    validation = MotionAttributeCacheDataset(cache_path, partition="validation")
    train_batch = _move(_supported_batch(train, 2), torch.device("cuda"))
    validation_batch = _move(
        _supported_batch(validation, 2), torch.device("cuda")
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = MotionAttributeExpert(
        channels=tuple(config["model"]["channels"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(train_batch["features"], train_batch["mask"])
        losses = motion_attribute_loss(
            output,
            family_targets=train_batch["families"],
            attribute_targets=train_batch["attributes"],
            labels=train_batch["label"],
            available=train_batch["available"],
            weights=config["loss_weights"],
        )
    losses["loss"].backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    finite = bool(torch.isfinite(losses["loss"])) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["training"]["gradient_clip"])
    )
    optimizer.step()
    changed = sorted(
        {
            _parameter_group(name)
            for name, parameter in model.named_parameters()
            if not torch.equal(before[name], parameter.detach().cpu())
        },
        key=("encoder", "family_head", "attribute_head", "action_head").index,
    )
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        validation_output = model(
            validation_batch["features"], validation_batch["mask"]
        )
        unsupported = model(
            torch.zeros(1, 96, 17, 6, device=device),
            torch.zeros(1, 96, dtype=torch.bool, device=device),
        )
    finite = finite and all(
        bool(torch.isfinite(value).all())
        for value in validation_output.values()
        if isinstance(value, torch.Tensor)
    )
    fallback_finite = all(
        bool(torch.isfinite(value).all())
        for value in unsupported.values()
        if isinstance(value, torch.Tensor)
    )
    checkpoint = output_root / "smoke_checkpoint.pt"
    _atomic_torch_save(checkpoint, model.state_dict())
    reloaded = MotionAttributeExpert(
        channels=tuple(config["model"]["channels"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    reloaded.eval()
    with torch.inference_mode():
        first = model(validation_batch["features"], validation_batch["mask"])[
            "action_logits"
        ]
        second = reloaded(
            validation_batch["features"], validation_batch["mask"]
        )["action_logits"]
    reload_delta = float((first - second).abs().max().cpu())
    report = {
        "stage": "MOTION-ATTRIBUTE-SMOKE",
        "status": "smoke_passed"
        if finite
        and fallback_finite
        and changed
        == ["encoder", "family_head", "attribute_head", "action_head"]
        and reload_delta == 0.0
        else "smoke_failed",
        "finite_forward_backward": finite,
        "changed_parameter_groups": changed,
        "gradient_user_ids": train_batch["user_id"],
        "validation_forward_rows": len(validation_batch["label"]),
        "unsupported_fallback_finite": fallback_finite,
        "reload_max_abs_logit_delta": reload_delta,
        "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cache_sha256": _sha256(cache_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    _atomic_write(output_root / "smoke_report.json", report)
    _atomic_write(
        output_root / "resolved_config.json",
        json.loads(json.dumps(config)),
    )
    if report["status"] != "smoke_passed":
        raise RuntimeError(f"motion attribute smoke failed: {report}")
    return report
