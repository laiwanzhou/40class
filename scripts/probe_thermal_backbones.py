from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import platform
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.models.thermal_tsm import IFormerTSM, MobileNetV3SmallTSM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports/thermal_backbone_environment_probe.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/thermal_t1a_candidate_qualification.md"
INTERNAL_LIMIT_BYTES = 95_000_000
IFORMER_MINIMUM_HEADROOM_BYTES = 2_000_000
FUSION_CALIBRATION_RESERVE_BYTES = 3_000_000
SCIENTIFIC_BASELINE_SHA = "c42bb43091c79903e5fde5655c2846c87305895a"
PROBE_SEED = 20260715
IFORMER_REPOSITORY = "https://github.com/ChuanyangZheng/iFormer"
IFORMER_PAPER_TITLE = (
    "iFormer: Integrating ConvNet and Transformer for Mobile Application"
)
IFORMER_PAPER_URL = "https://arxiv.org/abs/2501.15369"
IFORMER_REVISION = "2a87540fcb345afe9d950a58d0eb3873b938c3dc"
IFORMER_SOURCE_URL = (
    "https://raw.githubusercontent.com/ChuanyangZheng/iFormer/"
    f"{IFORMER_REVISION}/models/iformer.py"
)
IFORMER_CONFIG_URL = (
    "https://raw.githubusercontent.com/ChuanyangZheng/iFormer/"
    f"{IFORMER_REVISION}/configs/iFormer_t.yaml"
)
IFORMER_SMALL_CONFIG_URL = (
    "https://raw.githubusercontent.com/ChuanyangZheng/iFormer/"
    f"{IFORMER_REVISION}/configs/iFormer_s.yaml"
)
IFORMER_LICENSE_URL = (
    "https://raw.githubusercontent.com/ChuanyangZheng/iFormer/"
    f"{IFORMER_REVISION}/LICENSE"
)
IFORMER_CHECKPOINT_URL = (
    "https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_t.pth"
)
IFORMER_CHECKPOINT_SHA256 = (
    "7cbd778e3604694eb1a0becbf2e6a22798586f6bb46610a5c22b39880efb967e"
)
IFORMER_SMALL_CHECKPOINT_URL = (
    "https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/iFormer_s.pth"
)
IFORMER_SMALL_CHECKPOINT_SHA256 = (
    "dba81d99d9b6491b18fccd022f795d2b43e31bcb75f7aeea59be37b00bc7d7a1"
)
MOBILENET_WEIGHT_SHA256 = (
    "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
)
MOBILENET_SOURCE_URL = (
    "https://github.com/pytorch/vision/blob/v0.22.0/torchvision/models/mobilenetv3.py"
)
MOBILENET_LICENSE_URL = "https://github.com/pytorch/vision/blob/v0.22.0/LICENSE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_sha256(path: Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"weight SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_official_iformer_checkpoint(
    path: Path, expected_sha256: str
) -> Mapping[str, torch.Tensor]:
    require_file_sha256(path, expected_sha256)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model"), Mapping
    ):
        raise ValueError("Official iFormer checkpoint does not contain model state")
    state = checkpoint["model"]
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Official iFormer model state must contain only tensors")
    return state


@dataclass(frozen=True)
class DeploymentAsset:
    name: str
    identity: str
    serialized_bytes: int
    sha256: str
    path: str | None = None
    status: str = "retained"

    @classmethod
    def from_file(cls, name: str, path: Path, *, status: str = "retained") -> "DeploymentAsset":
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        digest = sha256_file(resolved)
        return cls(
            name=name,
            identity=digest,
            serialized_bytes=resolved.stat().st_size,
            sha256=digest,
            path=str(resolved),
            status=status,
        )


def build_deployment_ledger(assets: list[DeploymentAsset]) -> dict[str, Any]:
    unique: dict[str, DeploymentAsset] = {}
    duplicate_names: list[str] = []
    for asset in assets:
        if asset.serialized_bytes < 0 or len(asset.sha256) != 64:
            raise ValueError(f"Invalid deployment asset record: {asset.name}")
        previous = unique.get(asset.identity)
        if previous is None:
            unique[asset.identity] = asset
            continue
        if (
            previous.serialized_bytes != asset.serialized_bytes
            or previous.sha256 != asset.sha256
        ):
            raise ValueError(f"Conflicting duplicate asset identity: {asset.identity}")
        duplicate_names.append(asset.name)
    total = sum(asset.serialized_bytes for asset in unique.values())
    return {
        "assets": [asdict(asset) for asset in unique.values()],
        "unique_asset_count": len(unique),
        "deduplicated_reference_count": len(duplicate_names),
        "deduplicated_reference_names": duplicate_names,
        "total_serialized_bytes": total,
        "internal_limit_bytes_exclusive": INTERNAL_LIMIT_BYTES,
        "headroom_bytes": INTERNAL_LIMIT_BYTES - total,
        "passes_strict_limit": total < INTERNAL_LIMIT_BYTES,
    }


def require_complete_pretrained_load(
    module: nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> dict[str, list[str]]:
    result = module.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            "incomplete pretrained state_dict: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    module.load_state_dict(state_dict, strict=True)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def validate_candidate(
    provenance: Mapping[str, Any],
    logits: torch.Tensor,
    num_classes: int,
    package_bytes: int,
    *,
    minimum_headroom_bytes: int = 0,
) -> None:
    required = ("source_url", "license", "weight_sha256")
    if any(not provenance.get(field) for field in required):
        raise ValueError("candidate provenance must record source, license, and weight hash")
    if num_classes != 40 or tuple(logits.shape) != (2, 40):
        raise ValueError("candidate must expose a 40-class [2,40] forward contract")
    if not torch.isfinite(logits).all():
        raise ValueError("candidate forward output must be finite")
    if package_bytes >= INTERNAL_LIMIT_BYTES:
        raise ValueError("complete package must be strictly below 95,000,000 bytes")
    if INTERNAL_LIMIT_BYTES - package_bytes < minimum_headroom_bytes:
        raise ValueError(
            f"candidate leaves insufficient headroom: require {minimum_headroom_bytes} bytes"
        )


def serialize_state_dict(module: nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getvalue()


def parameter_inventory(module: nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "fp32_parameter_bytes": sum(parameter.numel() * 4 for parameter in parameters),
    }


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        urllib.request.urlretrieve(url, destination)
    return destination


def _torch_checkpoint_path(url: str) -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]


def _load_iformer_source(source_path: Path, symbol: str) -> nn.Module:
    module_name = f"thermal_t1a_mobile_iformer_{sha256_file(source_path)[:12]}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import official iFormer source: {source_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    constructor = getattr(module, symbol, None)
    if constructor is None:
        raise AttributeError(f"Official iFormer source has no model symbol {symbol}")
    return constructor(pretrained=False)


def _replace_iformer_head(model: nn.Module, num_classes: int = 40) -> None:
    try:
        head = model.classifier.classifier.l
    except AttributeError as error:
        raise TypeError("Official mobile iFormer classifier layout changed") from error
    if not isinstance(head, nn.Linear):
        raise TypeError("Official mobile iFormer classifier head is not nn.Linear")
    model.classifier.classifier.l = nn.Linear(head.in_features, num_classes)
    model.num_classes = num_classes


def _benchmark(
    model: nn.Module,
    device: torch.device,
    *,
    warmup: int = 3,
    iterations: int = 10,
) -> dict[str, Any]:
    model = model.to(device).eval()
    clips = torch.zeros(1, 16, 3, 224, 224, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(clips)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(clips)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
    result = {
        "device": str(device),
        "batch_size_trials": 1,
        "segments": 16,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "peak_cuda_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _package_with_candidate(
    frozen_assets: list[DeploymentAsset], candidate: DeploymentAsset
) -> dict[str, Any]:
    reserve = DeploymentAsset(
        name="calibration_and_optional_residual_upper_bound",
        identity="reserved-calibration-fusion-v1",
        serialized_bytes=FUSION_CALIBRATION_RESERVE_BYTES,
        sha256="0" * 64,
        status="reserved_upper_bound_not_yet_serialized",
    )
    return build_deployment_ledger([*frozen_assets, candidate, reserve])


def run_probe(
    *,
    x3d_checkpoint: Path,
    yolo_checkpoint: Path,
    output: Path = REPORT_PATH,
    summary_output: Path = SUMMARY_PATH,
) -> dict[str, Any]:
    torch.manual_seed(PROBE_SEED)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    mobile_weight_path = _download(weights.url, _torch_checkpoint_path(weights.url))
    mobile_weight_sha256 = require_file_sha256(
        mobile_weight_path, MOBILENET_WEIGHT_SHA256
    )
    mobile_state = torch.load(mobile_weight_path, map_location="cpu", weights_only=True)
    mobile_base = mobilenet_v3_small(weights=None)
    mobile_load = require_complete_pretrained_load(mobile_base, mobile_state)
    mobile_model = MobileNetV3SmallTSM(
        weights=None,
        backbone=mobile_base,
        num_classes=40,
        num_segments=16,
        fold_div=8,
    ).eval()
    with torch.inference_mode():
        mobile_logits = mobile_model(torch.zeros(2, 16, 3, 224, 224))
    mobile_serialized = serialize_state_dict(mobile_model)
    mobile_asset = DeploymentAsset(
        name="thermal_mobilenetv3_small_tsm_40class",
        identity=hashlib.sha256(mobile_serialized).hexdigest(),
        serialized_bytes=len(mobile_serialized),
        sha256=hashlib.sha256(mobile_serialized).hexdigest(),
        status="provisional_untrained_serialization_probe",
    )

    frozen_assets = [
        DeploymentAsset.from_file("frozen_ir_x3d_s_expert", x3d_checkpoint),
        DeploymentAsset.from_file("shared_yolo11n_pose", yolo_checkpoint),
        DeploymentAsset.from_file("shared_yolo11n_pose_reference", yolo_checkpoint),
    ]
    frozen_ledger = build_deployment_ledger(frozen_assets)
    mobile_package = _package_with_candidate(frozen_assets, mobile_asset)
    mobile_provenance = {
        "source_url": MOBILENET_SOURCE_URL,
        "source_revision": "torchvision-v0.22.0",
        "license": "BSD-3-Clause",
        "license_scope": "torchvision source code",
        "license_url": MOBILENET_LICENSE_URL,
        "pretrained_weight_license": "not_separately_stated_by_torchvision",
        "pretrained_weight_terms_note": (
            "Torchvision states that pretrained weights may have terms derived "
            "from their training data and that users must confirm permission."
        ),
        "weight_url": weights.url,
        "weight_path": str(mobile_weight_path),
        "weight_sha256": mobile_weight_sha256,
        "weight_serialized_bytes": mobile_weight_path.stat().st_size,
        "pretraining": "ImageNet-1K; torchvision IMAGENET1K_V1",
        "loaded_before_classifier_replacement": True,
        **mobile_load,
    }
    validate_candidate(
        mobile_provenance,
        mobile_logits,
        mobile_model.num_classes,
        mobile_package["total_serialized_bytes"],
    )

    cache = Path(torch.hub.get_dir()) / "thermal_t1a_mobile_iformer" / IFORMER_REVISION
    source_path = _download(IFORMER_SOURCE_URL, cache / "iformer.py")
    config_path = _download(IFORMER_CONFIG_URL, cache / "iFormer_t.yaml")
    small_config_path = _download(
        IFORMER_SMALL_CONFIG_URL, cache / "iFormer_s.yaml"
    )
    license_path = _download(IFORMER_LICENSE_URL, cache / "LICENSE")
    iformer_t_weight_path = _download(
        IFORMER_CHECKPOINT_URL, _torch_checkpoint_path(IFORMER_CHECKPOINT_URL)
    )
    iformer_t_state = load_official_iformer_checkpoint(
        iformer_t_weight_path, IFORMER_CHECKPOINT_SHA256
    )
    iformer_t_base = _load_iformer_source(source_path, "iFormer_t")
    iformer_t_load = require_complete_pretrained_load(iformer_t_base, iformer_t_state)
    iformer_t_original_inventory = parameter_inventory(iformer_t_base)
    _replace_iformer_head(iformer_t_base)
    iformer_t_model = IFormerTSM(
        backbone=iformer_t_base,
        num_classes=40,
        num_segments=16,
        fold_div=8,
        shift_before_stages=(0, 1, 2, 3),
    ).eval()
    with torch.inference_mode():
        iformer_t_logits = iformer_t_model(torch.zeros(2, 16, 3, 224, 224))
    iformer_t_serialized = serialize_state_dict(iformer_t_model)
    iformer_t_asset = DeploymentAsset(
        name="thermal_iformer_t_tsm_40class",
        identity=hashlib.sha256(iformer_t_serialized).hexdigest(),
        serialized_bytes=len(iformer_t_serialized),
        sha256=hashlib.sha256(iformer_t_serialized).hexdigest(),
        status="provisional_untrained_serialization_probe",
    )
    iformer_t_package = _package_with_candidate(frozen_assets, iformer_t_asset)
    iformer_common_provenance = {
        "paper_title": IFORMER_PAPER_TITLE,
        "paper_url": IFORMER_PAPER_URL,
        "source_url": IFORMER_REPOSITORY,
        "source_file_url": IFORMER_SOURCE_URL,
        "source_revision": IFORMER_REVISION,
        "source_sha256": sha256_file(source_path),
        "license": "MIT",
        "license_scope": "official ChuanyangZheng/iFormer source code",
        "license_url": IFORMER_LICENSE_URL,
        "license_sha256": sha256_file(license_path),
        "pretrained_weight_license": "not_separately_stated_in_official_repository",
        "pretraining": "ImageNet-1K; official v0.9 release",
        "checkpoint_load_policy": (
            "verify fixed SHA-256, load trusted official legacy checkpoint with "
            "weights_only=False, extract model only, then require strict state load"
        ),
        "official_constructor_auto_downloads_weights": False,
        "loaded_before_classifier_replacement": True,
    }
    iformer_t_provenance = {
        **iformer_common_provenance,
        "model_symbol": "iFormer_t",
        "official_config_url": IFORMER_CONFIG_URL,
        "official_config_sha256": sha256_file(config_path),
        "weight_url": IFORMER_CHECKPOINT_URL,
        "weight_path": str(iformer_t_weight_path),
        "weight_sha256": IFORMER_CHECKPOINT_SHA256,
        "weight_serialized_bytes": iformer_t_weight_path.stat().st_size,
        "checkpoint_model_key_count": len(iformer_t_state),
        **iformer_t_load,
    }
    validate_candidate(
        iformer_t_provenance,
        iformer_t_logits,
        iformer_t_model.num_classes,
        iformer_t_package["total_serialized_bytes"],
    )

    iformer_s_weight_path = _download(
        IFORMER_SMALL_CHECKPOINT_URL,
        _torch_checkpoint_path(IFORMER_SMALL_CHECKPOINT_URL),
    )
    iformer_s_state = load_official_iformer_checkpoint(
        iformer_s_weight_path, IFORMER_SMALL_CHECKPOINT_SHA256
    )
    iformer_s_base = _load_iformer_source(source_path, "iFormer_s")
    iformer_s_load = require_complete_pretrained_load(iformer_s_base, iformer_s_state)
    iformer_s_original_inventory = parameter_inventory(iformer_s_base)
    _replace_iformer_head(iformer_s_base)
    iformer_s_model = IFormerTSM(backbone=iformer_s_base).eval()
    iformer_s_serialized = serialize_state_dict(iformer_s_model)
    iformer_s_asset = DeploymentAsset(
        name="thermal_iformer_s_tsm_40class_budget_proxy",
        identity=hashlib.sha256(iformer_s_serialized).hexdigest(),
        serialized_bytes=len(iformer_s_serialized),
        sha256=hashlib.sha256(iformer_s_serialized).hexdigest(),
        status="conditional_untrained_serialization_probe",
    )
    iformer_s_package = _package_with_candidate(frozen_assets, iformer_s_asset)
    iformer_s_budget_eligible = (
        iformer_s_package["passes_strict_limit"]
        and iformer_s_package["headroom_bytes"] >= IFORMER_MINIMUM_HEADROOM_BYTES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobile_benchmark = _benchmark(mobile_model, device)
    iformer_t_benchmark = _benchmark(iformer_t_model, device)
    result: dict[str, Any] = {
        "schema_version": 2,
        "stage": "Thermal T1-A corrective source/loading/forward/budget audit only",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training_performed": False,
        "labels_or_evidence_read": False,
        "scientific_baseline_commit": SCIENTIFIC_BASELINE_SHA,
        "probe_seed": PROBE_SEED,
        "source_identity_correction": {
            "incorrect_prior_repository": "https://github.com/sail-sg/iFormer",
            "incorrect_prior_paper": "Inception Transformer",
            "correction": (
                "The prior negative result applied to a homonymous project and does "
                "not apply to the intended mobile iFormer family."
            ),
            "authoritative_repository": IFORMER_REPOSITORY,
            "authoritative_paper": IFORMER_PAPER_TITLE,
        },
        "environment": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "timm": importlib.metadata.version("timm"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
        },
        "tsm_contract": {
            "num_segments": 16,
            "fold_div": 8,
            "mobile_shift_before_feature_blocks": [1, 3, 6, 9],
            "iformer_shift_after_downsample_before_native_stages": [0, 1, 2, 3],
            "input_shape": [2, 16, 3, 224, 224],
            "mobile_output_shape": list(mobile_logits.shape),
            "iformer_t_output_shape": list(iformer_t_logits.shape),
            "finite": bool(
                torch.isfinite(mobile_logits).all()
                and torch.isfinite(iformer_t_logits).all()
            ),
            "trial_boundary": "explicit_batch_axis_no_cross_trial_shift",
        },
        "candidates": {
            "iformer_t_tsm": {
                "qualification": "eligible_primary_t1a_after_human_approval",
                "pretrained_loading_audit_passed": True,
                "random_initialization_fallback_allowed": False,
                "provenance": iformer_t_provenance,
                "resources": {
                    "official_1000_class": iformer_t_original_inventory,
                    **parameter_inventory(iformer_t_model),
                    "actual_serialized_state_dict_bytes": len(iformer_t_serialized),
                    "actual_serialized_state_dict_sha256": iformer_t_asset.sha256,
                },
                "forward": {
                    "input_shape": [2, 16, 3, 224, 224],
                    "output_shape": list(iformer_t_logits.shape),
                    "finite": bool(torch.isfinite(iformer_t_logits).all()),
                },
                "benchmark": iformer_t_benchmark,
                "provisional_complete_package": iformer_t_package,
                "config_created": True,
                "training_performed": False,
            },
            "mobilenetv3_small_tsm": {
                "qualification": "eligible_matched_control_t1a_with_weight_terms_caveat",
                "provenance": mobile_provenance,
                "resources": {
                    **parameter_inventory(mobile_model),
                    "actual_serialized_state_dict_bytes": len(mobile_serialized),
                    "actual_serialized_state_dict_sha256": mobile_asset.sha256,
                },
                "forward": {
                    "input_shape": [2, 16, 3, 224, 224],
                    "output_shape": list(mobile_logits.shape),
                    "finite": bool(torch.isfinite(mobile_logits).all()),
                },
                "benchmark": mobile_benchmark,
                "provisional_complete_package": mobile_package,
                "config_created": True,
            },
            "iformer_s_tsm": {
                "qualification": (
                    "eligible_conditional_capacity_upgrade_after_primary_and_control"
                    if iformer_s_budget_eligible
                    else "ineligible_budget"
                ),
                "conditional_only": True,
                "pretrained_loading_audit_passed": True,
                "random_initialization_fallback_allowed": False,
                "tsm_integrated": True,
                "forward_run": False,
                "training_performed": False,
                "provenance": {
                    **iformer_common_provenance,
                    "model_symbol": "iFormer_s",
                    "official_config_url": IFORMER_SMALL_CONFIG_URL,
                    "official_config_sha256": sha256_file(small_config_path),
                    "weight_url": IFORMER_SMALL_CHECKPOINT_URL,
                    "weight_path": str(iformer_s_weight_path),
                    "weight_sha256": IFORMER_SMALL_CHECKPOINT_SHA256,
                    "weight_serialized_bytes": iformer_s_weight_path.stat().st_size,
                    "checkpoint_model_key_count": len(iformer_s_state),
                    **iformer_s_load,
                },
                "resources": {
                    "official_1000_class": iformer_s_original_inventory,
                    **parameter_inventory(iformer_s_model),
                    "actual_40class_serialized_state_dict_bytes": len(iformer_s_serialized),
                    "actual_40class_serialized_state_dict_sha256": iformer_s_asset.sha256,
                },
                "provisional_complete_package": iformer_s_package,
                "minimum_required_headroom_bytes": IFORMER_MINIMUM_HEADROOM_BYTES,
                "budget_eligible": iformer_s_budget_eligible,
                "config_created": iformer_s_budget_eligible,
            },
        },
        "deployment_ledger": {
            "current_retained_inference_assets": frozen_ledger,
            "retained_expert_scope": ["IR/X3D-S", "shared YOLO11n-pose"],
            "not_yet_retained_or_serialized": [
                "Skeleton expert",
                "IMU expert",
                "Radar expert",
                "Depth expert",
                "Thermal expert",
                "calibration/fusion weights",
            ],
            "calibration_fusion_reserved_upper_bound_bytes": FUSION_CALIBRATION_RESERVE_BYTES,
            "accounting_rule": "Every learned inference asset is keyed and counted once; prediction/evidence archives are excluded.",
            "complete_six_modal_gate_status": "not_yet_evaluable_until_other_experts_are_retained",
        },
        "decision": {
            "iformer_t": "technically_eligible_primary_after_human_approval",
            "mobilenetv3_small": "technically_eligible_after_human_approval_with_weight_terms_caveat",
            "iformer_s": (
                "conditional_upgrade_eligible_but_not_authorized"
                if iformer_s_budget_eligible
                else "stop_ineligible_budget_no_training"
            ),
            "next_action": (
                "stop_after_corrective_T1-A_and_request_human_authorization_before_T1-B_training"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_output.write_text(_render_summary(result), encoding="utf-8")
    return result


def _render_summary(result: Mapping[str, Any]) -> str:
    tiny = result["candidates"]["iformer_t_tsm"]
    mobile = result["candidates"]["mobilenetv3_small_tsm"]
    small = result["candidates"]["iformer_s_tsm"]
    current = result["deployment_ledger"]["current_retained_inference_assets"]
    return f"""# Thermal T1-A Candidate Qualification

## Scope

Corrective source identity, environment, source/license/hash, strict pretrained loading, TSM forward, and deployment-byte audit only. No training, labels, sealed data, competition test, or ExpertEvidence were read.

Scientific baseline remains `{SCIENTIFIC_BASELINE_SHA}`.

## Source identity correction

The earlier T1-A audit inspected Sail-SG's homonymous *Inception Transformer*. That finding was valid for that repository but irrelevant to the intended candidate. The authoritative source is Chuanyang Zheng's *{IFORMER_PAPER_TITLE}* at revision `{IFORMER_REVISION}`, where the official symbol is `iFormer_t` and the source-code license is MIT.

## Decision

- **iFormer-T + TSM: technically eligible primary candidate, pending human training approval.** The official `iFormer_t(pretrained=True)` constructor does not download or load weights. The probe therefore explicitly downloads the checkpoint, verifies its fixed SHA, extracts it from the legacy training bundle, and strictly loads it with zero missing/unexpected keys before replacing the ImageNet classifier. The TSM wrapper produced finite `[2,16,3,224,224] -> [2,40]` output. Random-initialization fallback is prohibited.
- **pretrained MobileNetV3-Small + TSM: technically eligible matched control, with a weight-terms caveat.** Strict pretrained load completed with zero missing/unexpected keys; `[2,16,3,224,224] -> [2,40]` is finite. Torchvision source is BSD-3-Clause, while its official model documentation says pretrained-weight permission remains the user's responsibility because training-data terms may apply.
- **iFormer-S + TSM: budget/load eligible only as the frozen conditional upgrade.** Its correct-family checkpoint also loads strictly. The 40-class TSM state-dict proxy is {small['resources']['actual_40class_serialized_state_dict_bytes']:,} bytes and the provisional package is {small['provisional_complete_package']['total_serialized_bytes']:,} bytes, leaving {small['provisional_complete_package']['headroom_bytes']:,} bytes. It may not train before the primary and matched-control comparison authorizes the upgrade gate.

## Resources

| Candidate | Parameters | FP32 parameter bytes | Serialized bytes | Peak CUDA allocated | Median trial latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| iFormer-T + TSM | {tiny['resources']['parameter_count']:,} | {tiny['resources']['fp32_parameter_bytes']:,} | {tiny['resources']['actual_serialized_state_dict_bytes']:,} | {tiny['benchmark']['peak_cuda_memory_allocated_bytes']:,} | {tiny['benchmark']['latency_ms_median']:.3f} ms |
| MobileNetV3-Small + TSM | {mobile['resources']['parameter_count']:,} | {mobile['resources']['fp32_parameter_bytes']:,} | {mobile['resources']['actual_serialized_state_dict_bytes']:,} | {mobile['benchmark']['peak_cuda_memory_allocated_bytes']:,} | {mobile['benchmark']['latency_ms_median']:.3f} ms |
| iFormer-S + TSM budget proxy | {small['resources']['parameter_count']:,} | {small['resources']['fp32_parameter_bytes']:,} | {small['resources']['actual_40class_serialized_state_dict_bytes']:,} | not run | not run |

## Deployment Ledger

Current retained learned assets are frozen IR/X3D-S and shared YOLO11n-pose: {current['total_serialized_bytes']:,} bytes after deduplicating the repeated YOLO reference. A 3,000,000-byte calibration/fusion upper-bound reserve is included in candidate projections. Skeleton, IMU, Radar, Depth, Thermal, and fusion assets are not yet retained, so a complete six-modal package pass is not claimed.

Corrective T1-A stops here. No optimizer, backward pass, epoch loop, or learned Thermal weight was created. T1-B training remains unauthorized until explicit human approval.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Thermal T1-A backbones without training")
    parser.add_argument(
        "--x3d-checkpoint",
        type=Path,
        default=Path(
            r"D:\work\2026.7.14_kaggle\40class-x3d-adaptive-multiclip\outputs"
            r"\x3d_s_ir_context_train12_val2_dev"
            r"\x3d_s_ir_context_train12_val2_user6_user7_partial2_seed20260715"
            r"\best_accuracy.pt"
        ),
    )
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=Path(r"D:\work\2026.7.14_kaggle\40class\yolo11n-pose.pt"),
    )
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(
        x3d_checkpoint=args.x3d_checkpoint,
        yolo_checkpoint=args.yolo_checkpoint,
        output=args.output,
        summary_output=args.summary_output,
    )
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
