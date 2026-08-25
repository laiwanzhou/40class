from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from scripts.cache_ir_depth_videomaev2_p2a import _atomic_write_text
from src.data.hierarchical_multimodal_dataset import make_midfusion_dataset
from src.experiments.hierarchical_midfusion_config import (
    load_midfusion_config,
    project_path,
)
from src.models.body_motion_segment_encoder import BodyMotionSegmentEncoder
from src.models.hierarchical_action_query_fusion import HierarchicalActionQueryFusion
from src.models.hierarchical_multimodal_teacher import (
    GroupDropout,
    HierarchicalMultimodalTeacher,
)
from src.models.ir_depth_videomaev2_teacher import (
    build_official_videomaev2_vit_b,
    sha256_file,
)
from src.models.structured_ir_depth_visual_encoder import (
    StructuredIRDepthVisualEncoder,
    VideoMAESegmentBackboneAdapter,
)
from src.training.hierarchical_multimodal_losses import hierarchical_teacher_loss


ModelFactory = Callable[[dict[str, Any]], HierarchicalMultimodalTeacher]
DatasetFactory = Callable[[dict[str, Any]], Dataset[dict[str, object]]]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_teacher(config: dict[str, Any]) -> HierarchicalMultimodalTeacher:
    model_config = config["model"]
    checkpoint = Path(str(model_config["visual_checkpoint"]))
    if sha256_file(checkpoint) != model_config["visual_checkpoint_sha256"]:
        raise RuntimeError("VideoMAE checkpoint provenance changed")
    backbone, _ = build_official_videomaev2_vit_b(
        checkpoint_path=checkpoint,
        num_classes=40,
        with_cp=True,
    )
    dim = int(model_config["teacher_dim"])
    visual = StructuredIRDepthVisualEncoder(
        backbone=VideoMAESegmentBackboneAdapter(
            backbone=backbone, frozen_prefix_blocks=8, segment_count=8
        ),
        output_dim=dim,
    )
    body = BodyMotionSegmentEncoder(
        output_dim=dim, heads=int(model_config["attention_heads"])
    )
    fusion = HierarchicalActionQueryFusion(
        dim=dim,
        classes=40,
        heads=int(model_config["attention_heads"]),
        layers=int(model_config["fusion_layers"]),
    )
    return HierarchicalMultimodalTeacher(
        visual_encoder=visual,
        body_encoder=body,
        fusion=fusion,
        dim=dim,
        classes=40,
    )


def _default_smoke_dataset(config: dict[str, Any]) -> Dataset[dict[str, object]]:
    clean_view = project_path(
        str(config["data"]["skeleton_clean_views"])
    ) / "selected_final/clean_view.csv"
    if not clean_view.is_file():
        raise FileNotFoundError(
            f"missing selected-final Skeleton clean view: {clean_view}"
        )
    dataset = make_midfusion_dataset(
        config,
        partition="train",
        metadata_only=False,
        skeleton_clean_view=clean_view,
        training=False,
    )
    complete = next(
        index
        for index, trial in enumerate(dataset.trials)
        if all(trial.availability[name] for name in ("ir", "depth_color", "skeleton", "imu"))
    )
    missing_imu = next(
        index
        for index, trial in enumerate(dataset.trials)
        if trial.availability["ir"]
        and trial.availability["depth_color"]
        and trial.availability["skeleton"]
        and not trial.availability["imu"]
    )
    return Subset(dataset, [complete, missing_imu])


def _parameter_groups(
    model: HierarchicalMultimodalTeacher,
) -> dict[str, list[torch.nn.Parameter]]:
    visual = [
        parameter
        for parameter in model.visual_encoder.parameters()
        if parameter.requires_grad
    ]
    skeleton = [
        *model.body_encoder.skeleton_projection.parameters(),
        *model.body_encoder.skeleton_temporal.parameters(),
    ]
    imu = [
        *model.body_encoder.imu_projection.parameters(),
        *model.body_encoder.imu_role_score.parameters(),
        *model.body_encoder.imu_temporal.parameters(),
    ]
    body_ids = {id(parameter) for parameter in skeleton + imu}
    fusion = [
        parameter
        for module in (
            model.body_encoder.cross_attention,
            model.body_encoder.cross_norm,
            model.fusion,
            model.context_head,
            model.wrist_head,
            model.body_head,
        )
        for parameter in module.parameters()
        if parameter.requires_grad and id(parameter) not in body_ids
    ]
    return {"visual": visual, "skeleton": skeleton, "imu": imu, "fusion": fusion}


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_smoke(
    config_path: Path,
    *,
    output_root: Path,
    model_factory: ModelFactory | None = None,
    dataset_factory: DatasetFactory | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    config = load_midfusion_config(config_path)
    _set_seed(int(config["training"]["seed"]))
    device = device or torch.device("cuda")
    model = (model_factory or build_teacher)(config).to(device)
    dataset = (dataset_factory or _default_smoke_dataset)(config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    groups = _parameter_groups(model)
    before = {
        name: [parameter.detach().cpu().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    finite_gradients = True
    users: set[str] = set()
    last_batch: dict[str, Any] | None = None
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        users.update(str(value) for value in batch["user_id"])
        if users & {"user6", "user7"}:
            raise RuntimeError("validation users entered smoke gradients")
        batch = _move_batch(batch, device)
        last_batch = batch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(batch, dropout_policy=GroupDropout.disabled())
            losses = hierarchical_teacher_loss(
                output,
                batch["label"].long(),
                epoch=1,
                natural_pattern=True,
            )
        losses["loss"].backward()
        finite_gradients &= all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise FloatingPointError("non-finite hierarchical teacher smoke gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    if last_batch is None:
        raise RuntimeError("empty hierarchical teacher smoke dataset")

    changed = []
    for name, parameters in groups.items():
        if any(
            not torch.equal(start, parameter.detach().cpu())
            for start, parameter in zip(before[name], parameters, strict=True)
        ):
            changed.append(name)

    model.eval()
    body_only = _clone_batch(last_batch)
    body_only["visual_view_availability"].zero_()
    no_core = _clone_batch(last_batch)
    no_core["visual_view_availability"].zero_()
    no_core["skeleton_mask"].zero_()
    no_core["imu_role_mask"].zero_()
    with torch.inference_mode():
        body_output = model(body_only, dropout_policy=GroupDropout.disabled())
        no_core_output = model(no_core, dropout_policy=GroupDropout.disabled())
    report = {
        "status": "smoke_passed",
        "sample_users_entered_gradient": sorted(users),
        "finite_gradients": finite_gradients,
        "changed_parameter_groups": changed,
        "body_only_finite": bool(torch.isfinite(body_output["logits"]).all()),
        "no_core_finite": bool(torch.isfinite(no_core_output["logits"]).all()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_mib": float(
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    output_root.mkdir(parents=True)
    _atomic_write_text(
        output_root / "resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )
    _atomic_write_text(
        output_root / "smoke_report.json", json.dumps(report, indent=2) + "\n"
    )
    return report
