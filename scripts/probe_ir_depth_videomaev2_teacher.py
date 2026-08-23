from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ir_depth_videomaev2_dataset import IRDepthVideoMAEV2Dataset
from src.models.ir_depth_videomaev2_teacher import (
    MODALITY_NAMES,
    VIEW_NAMES,
    IRDepthVideoMAEV2Teacher,
    build_official_videomaev2_vit_b,
    sequential_multiview_backward,
    sha256_file,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_depth_videomaev2_vit_b_p0.yaml"
P0_SOURCE_PATHS = (
    PROJECT_ROOT / "src/models/ir_depth_videomaev2_teacher.py",
    PROJECT_ROOT / "src/data/ir_depth_videomaev2_dataset.py",
    Path(__file__).resolve(),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_probe_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P0 config must be a mapping")
    checkpoint = config.get("checkpoint", {})
    inputs = config.get("input", {})
    runtime = config.get("runtime", {})
    gates = config.get("gates", {})
    _require(config.get("stage") == "P0", "P0 stage changed")
    _require(checkpoint.get("architecture") == "vit_base_patch16_224", "backbone changed")
    _require(inputs.get("frames") == 16, "P0 requires exactly 16 frames")
    _require(inputs.get("image_size") == 224, "P0 requires 224 pixel input")
    _require(inputs.get("modalities") == list(MODALITY_NAMES), "P0 modalities changed")
    _require(inputs.get("views") == list(VIEW_NAMES), "P0 views changed")
    _require(
        inputs.get("roi_temporal_policy") == "fixed_trial_level_all_views",
        "P0 ROI temporal policy changed",
    )
    _require(runtime.get("physical_batch_trials") == 1, "P0 physical batch must be one")
    _require(runtime.get("amp_dtype") == "bfloat16", "P0 dtype must be bfloat16")
    _require(runtime.get("activation_checkpointing") is True, "checkpointing is required")
    _require(runtime.get("sequential_multiview_backward") is True, "sequential backward is required")
    _require(gates.get("peak_allocated_mib_below") == 7300, "P0 memory gate changed")
    return config


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_probe(config_path: Path) -> dict[str, Any]:
    config = load_probe_config(config_path)
    device = torch.device(str(config["runtime"]["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P0 requires CUDA")
    checkpoint_path = Path(str(config["checkpoint"]["path"]))
    backbone, provenance = build_official_videomaev2_vit_b(
        checkpoint_path=checkpoint_path,
        num_classes=40,
        with_cp=True,
    )
    model = IRDepthVideoMAEV2Teacher(backbone=backbone, num_classes=40).to(device)
    model.train()
    data = config["data"]
    dataset = IRDepthVideoMAEV2Dataset(
        manifest_path=_resolve_project_path(str(data["manifest"])),
        split_path=_resolve_project_path(str(data["split"])),
        data_root=Path(str(data["root"])),
        pose_cache_path=Path(str(data["pose_cache"])),
        partition=str(data["partition"]),
        training=False,
        frames=16,
        image_size=224,
        temporal_jitter=0.0,
        interaction_config=dict(config["roi"]),
    )
    sample: dict[str, object] | None = None
    for index in range(min(len(dataset), 32)):
        candidate = dataset[index]
        if bool(torch.as_tensor(candidate["availability"]).all()):
            sample = candidate
            break
    if sample is None:
        raise RuntimeError("P0 could not find an all-view-valid train12 trial")
    clips = torch.as_tensor(sample["clips"]).unsqueeze(0).to(device)
    availability = torch.as_tensor(sample["availability"]).unsqueeze(0).to(device)
    labels = torch.tensor([int(sample["label"])], device=device)
    if clips.shape != (1, 2, 4, 3, 16, 224, 224):
        raise RuntimeError(f"unexpected real P0 shape: {tuple(clips.shape)}")

    runtime = config["runtime"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(runtime["backbone_learning_rate"])},
            {"params": [model.class_view_gate], "lr": float(runtime["fusion_learning_rate"])},
        ],
        weight_decay=float(runtime["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    head_before = model.backbone.head.weight.detach().clone()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    # The detached first pass and replay pass must not share autocast's casted-weight
    # cache; cached tensors created under no_grad would silently detach weight grads.
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        result = sequential_multiview_backward(
            model=model,
            clips=clips,
            availability=availability,
            labels=labels,
            label_smoothing=float(runtime["label_smoothing"]),
        )
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    finite_gradients = bool(
        gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(runtime["gradient_clip"]))
    optimizer.step()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak_allocated = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / 1024**2)
    optimizer_changed = not torch.equal(head_before, model.backbone.head.weight.detach())
    expected_trace = [
        f"{modality}:{view}" for modality in MODALITY_NAMES for view in VIEW_NAMES
    ]
    gate_values = {
        "peak_allocated_below_7300_mib": peak_allocated < 7300,
        "strict_checkpoint_load": bool(provenance["strict_backbone_load"]),
        "finite_loss_logits_gradients": bool(
            torch.isfinite(result["loss"])
            and torch.isfinite(result["logits"]).all()
            and finite_gradients
        ),
        "optimizer_state_changed": optimizer_changed,
        "exact_execution_trace": model.last_execution_trace == expected_trace,
    }
    gate_values["passed"] = all(gate_values.values())
    report = {
        "schema_version": 1,
        "stage": "P0",
        "status": "passed" if gate_values["passed"] else "failed",
        "config": str(config_path.resolve()),
        "integrity": {
            "config_sha256": sha256_file(config_path.resolve()),
            "source_sha256": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
                for path in P0_SOURCE_PATHS
            },
        },
        "checkpoint": provenance,
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "total_vram_mib": float(torch.cuda.get_device_properties(device).total_memory / 1024**2),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "real_sample": {
            "sample_id": str(sample["sample_id"]),
            "user_id": str(sample["user_id"]),
            "label": int(sample["label"]),
            "clips_shape": list(clips.shape),
            "availability": availability.cpu().tolist(),
            "sampled_indices": torch.as_tensor(sample["sampled_indices"]).tolist(),
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trial_logits_shape": list(result["logits"].shape),
            "view_logits_shape": list(result["view_logits"].shape),
            "class_view_weights_shape": list(result["class_view_weights"].shape),
            "execution_trace": model.last_execution_trace,
        },
        "backward": {
            "loss": float(result["loss"]),
            "seconds_per_optimizer_step": seconds,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "finite_gradients": finite_gradients,
            "optimizer_state_changed": optimizer_changed,
        },
        "gates": gate_values,
        "p1_status": "eligible_to_start" if gate_values["passed"] else "blocked_by_p0",
    }
    report_path = _resolve_project_path(str(config["report"]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the IR/depth VideoMAE V2-B teacher")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run_probe(args.config.resolve())
    print(json.dumps({"status": report["status"], "p1_status": report["p1_status"]}))


if __name__ == "__main__":
    main()
