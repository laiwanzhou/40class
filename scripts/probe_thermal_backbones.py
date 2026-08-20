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
import types
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.models.thermal_tsm import MobileNetV3SmallTSM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports/thermal_backbone_environment_probe.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/thermal_t1a_candidate_qualification.md"
INTERNAL_LIMIT_BYTES = 95_000_000
IFORMER_MINIMUM_HEADROOM_BYTES = 2_000_000
FUSION_CALIBRATION_RESERVE_BYTES = 3_000_000
IFORMER_REPOSITORY = "https://github.com/sail-sg/iFormer"
IFORMER_REVISION = "725d8e7f455b5e17be20788b9bcd6c6c505c4be0"
IFORMER_SOURCE_URL = (
    "https://raw.githubusercontent.com/sail-sg/iFormer/"
    f"{IFORMER_REVISION}/models/inception_transformer.py"
)
IFORMER_LICENSE_URL = (
    "https://raw.githubusercontent.com/sail-sg/iFormer/"
    f"{IFORMER_REVISION}/LICENSE"
)
IFORMER_SMALL_WEIGHT_URL = (
    "https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_small.pth"
)
IFORMER_SMALL_WEIGHT_SHA256 = (
    "b95bcc4ef2262d02b75b2dd81f5b837f3703dd324426508a6be6c5398becfa4e"
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


def _load_iformer_small_source(source_path: Path) -> nn.Module:
    from timm.layers import to_2tuple

    legacy_helpers = types.ModuleType("timm.models.layers.helpers")
    legacy_helpers.to_2tuple = to_2tuple
    sys.modules.setdefault("timm.models.layers.helpers", legacy_helpers)
    module_name = "thermal_t1a_official_iformer"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official iFormer source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.iformer_small(pretrained=False)


def _replace_iformer_head(model: nn.Module, num_classes: int = 40) -> None:
    if not isinstance(model.head, nn.Linear):
        raise TypeError("Official iFormer classifier is not nn.Linear")
    model.head = nn.Linear(model.head.in_features, num_classes)
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
    return {
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

    cache = Path(torch.hub.get_dir()) / "thermal_t1a_iformer" / IFORMER_REVISION
    source_path = _download(IFORMER_SOURCE_URL, cache / "inception_transformer.py")
    license_path = _download(IFORMER_LICENSE_URL, cache / "LICENSE")
    iformer_weight_path = _download(
        IFORMER_SMALL_WEIGHT_URL, _torch_checkpoint_path(IFORMER_SMALL_WEIGHT_URL)
    )
    iformer_weight_sha256 = require_file_sha256(
        iformer_weight_path, IFORMER_SMALL_WEIGHT_SHA256
    )
    iformer_model = _load_iformer_small_source(source_path)
    iformer_state = torch.load(iformer_weight_path, map_location="cpu", weights_only=True)
    iformer_load = require_complete_pretrained_load(iformer_model, iformer_state)
    iformer_original_inventory = parameter_inventory(iformer_model)
    _replace_iformer_head(iformer_model)
    iformer_serialized = serialize_state_dict(iformer_model)
    iformer_asset = DeploymentAsset(
        name="thermal_iformer_s_tsm_40class_budget_proxy",
        identity=hashlib.sha256(iformer_serialized).hexdigest(),
        serialized_bytes=len(iformer_serialized),
        sha256=hashlib.sha256(iformer_serialized).hexdigest(),
        status="conditional_budget_probe_no_tsm_runtime_no_training",
    )
    iformer_package = _package_with_candidate(frozen_assets, iformer_asset)
    iformer_headroom = iformer_package["headroom_bytes"]
    iformer_budget_eligible = (
        iformer_package["passes_strict_limit"]
        and iformer_headroom >= IFORMER_MINIMUM_HEADROOM_BYTES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark = _benchmark(mobile_model, device)
    result: dict[str, Any] = {
        "schema_version": 1,
        "stage": "Thermal T1-A environment/loading/forward/budget audit only",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training_performed": False,
        "labels_or_evidence_read": False,
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
            "input_shape": [2, 16, 3, 224, 224],
            "output_shape": list(mobile_logits.shape),
            "finite": bool(torch.isfinite(mobile_logits).all()),
            "trial_boundary": "explicit_batch_axis_no_cross_trial_shift",
        },
        "candidates": {
            "iformer_t_tsm": {
                "qualification": "ineligible_source_identity",
                "reason": (
                    "Official Sail-SG iFormer revision publishes only iFormer-S/B/L; "
                    "no official iFormer-T architecture or pretrained checkpoint exists."
                ),
                "source_url": IFORMER_REPOSITORY,
                "source_revision": IFORMER_REVISION,
                "official_family_code_license": "Apache-2.0",
                "weight_url": None,
                "weight_sha256": None,
                "published_variants": ["iFormer-S", "iFormer-B", "iFormer-L"],
                "pretrained_loading_audit_passed": False,
                "config_created": False,
                "forward_run": False,
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
                "benchmark": benchmark,
                "provisional_complete_package": mobile_package,
                "config_created": True,
            },
            "iformer_s_tsm": {
                "qualification": "ineligible_budget",
                "conditional_only": True,
                "pretrained_loading_audit_passed": True,
                "tsm_integrated": False,
                "forward_run": False,
                "training_performed": False,
                "provenance": {
                    "source_url": IFORMER_REPOSITORY,
                    "source_file_url": IFORMER_SOURCE_URL,
                    "source_revision": IFORMER_REVISION,
                    "source_sha256": sha256_file(source_path),
                    "license": "Apache-2.0",
                    "license_scope": "official iFormer repository source code",
                    "license_url": IFORMER_LICENSE_URL,
                    "license_sha256": sha256_file(license_path),
                    "pretrained_weight_license": "not_separately_stated_in_official_repository",
                    "weight_url": IFORMER_SMALL_WEIGHT_URL,
                    "weight_path": str(iformer_weight_path),
                    "weight_sha256": iformer_weight_sha256,
                    "weight_serialized_bytes": iformer_weight_path.stat().st_size,
                    "pretraining": "ImageNet-1K",
                    "loaded_before_classifier_replacement": True,
                    **iformer_load,
                },
                "resources": {
                    "official_1000_class": iformer_original_inventory,
                    **parameter_inventory(iformer_model),
                    "actual_40class_serialized_state_dict_bytes": len(iformer_serialized),
                    "actual_40class_serialized_state_dict_sha256": iformer_asset.sha256,
                },
                "provisional_complete_package": iformer_package,
                "minimum_required_headroom_bytes": IFORMER_MINIMUM_HEADROOM_BYTES,
                "budget_eligible": iformer_budget_eligible,
                "config_created": False,
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
            "iformer_t": "stop_blocked_no_official_candidate_identity",
            "mobilenetv3_small": "technically_eligible_after_human_approval_with_weight_terms_caveat",
            "iformer_s": "stop_ineligible_budget_no_config_no_training",
            "next_action": "stop_after_T1-A_and_request_human_decision_on_replacing_or_defining_iFormer-T",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_output.write_text(_render_summary(result), encoding="utf-8")
    return result


def _render_summary(result: Mapping[str, Any]) -> str:
    mobile = result["candidates"]["mobilenetv3_small_tsm"]
    small = result["candidates"]["iformer_s_tsm"]
    current = result["deployment_ledger"]["current_retained_inference_assets"]
    return f"""# Thermal T1-A Candidate Qualification

## Scope

Environment, source/license/hash, strict pretrained loading, TSM forward, and deployment-byte audit only. No training, labels, sealed data, competition test, or ExpertEvidence were read.

## Decision

- **iFormer-T + TSM: ineligible / blocked.** Official Sail-SG iFormer revision `{IFORMER_REVISION}` publishes S/B/L only. No official iFormer-T architecture or pretrained weight was found, so no loader, config, or forward was fabricated.
- **pretrained MobileNetV3-Small + TSM: technically eligible matched control, with a weight-terms caveat.** Strict pretrained load completed with zero missing/unexpected keys; `[2,16,3,224,224] -> [2,40]` is finite. Torchvision source is BSD-3-Clause, while its official model documentation says pretrained-weight permission remains the user's responsibility because training-data terms may apply.
- **iFormer-S + TSM: ineligible on budget.** Official pretrained loading is complete, but the 40-class state-dict proxy is {small['resources']['actual_40class_serialized_state_dict_bytes']:,} bytes and the provisional package is {small['provisional_complete_package']['total_serialized_bytes']:,} bytes. No config, TSM runtime, forward, or training was created.

## Resources

| Candidate | Parameters | FP32 parameter bytes | Serialized bytes | Peak CUDA allocated | Median trial latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3-Small + TSM | {mobile['resources']['parameter_count']:,} | {mobile['resources']['fp32_parameter_bytes']:,} | {mobile['resources']['actual_serialized_state_dict_bytes']:,} | {mobile['benchmark']['peak_cuda_memory_allocated_bytes']:,} | {mobile['benchmark']['latency_ms_median']:.3f} ms |
| iFormer-S 40-class budget proxy | {small['resources']['parameter_count']:,} | {small['resources']['fp32_parameter_bytes']:,} | {small['resources']['actual_40class_serialized_state_dict_bytes']:,} | not run | not run |

## Deployment Ledger

Current retained learned assets are frozen IR/X3D-S and shared YOLO11n-pose: {current['total_serialized_bytes']:,} bytes after deduplicating the repeated YOLO reference. A 3,000,000-byte calibration/fusion upper-bound reserve is included in candidate projections. Skeleton, IMU, Radar, Depth, Thermal, and fusion assets are not yet retained, so a complete six-modal package pass is not claimed.

T1-A stops here pending a human decision on the undefined iFormer-T candidate identity.
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
