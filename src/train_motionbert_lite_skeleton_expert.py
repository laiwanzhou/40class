from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

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
