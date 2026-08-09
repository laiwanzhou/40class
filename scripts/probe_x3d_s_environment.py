from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path
from typing import Any


INPUT_SHAPE = [1, 3, 13, 182, 182]
MODEL_NAME = "x3d_s"
PRETRAINING = "kinetics_400"
SOURCE_REVISION = "pytorchvideo==0.1.5"
INTERNAL_SIZE_LIMIT_BYTES = 95_000_000
X3D_CHECKPOINT_FILENAME = "X3D_S.pyth"
X3D_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth"
)


def aggregate_unique_artifact_bytes(paths: list[Path]) -> int:
    unique_paths = {path.resolve(strict=True) for path in paths}
    return sum(path.stat().st_size for path in unique_paths)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_weight_file(
    path: Path,
    *,
    parameter_count: int,
    fp32_parameter_bytes: int,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "parameter_count": parameter_count,
        "fp32_parameter_bytes": fp32_parameter_bytes,
        "serialized_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def enforce_submission_size_gate(
    aggregate_bytes: int,
    *,
    limit_bytes: int = INTERNAL_SIZE_LIMIT_BYTES,
) -> None:
    if aggregate_bytes >= limit_bytes:
        raise ValueError(
            f"Inference artifacts total {aggregate_bytes:,} bytes; "
            f"the internal limit is {limit_bytes:,} bytes."
        )


def estimate_custom_head_parameters(
    *,
    backbone_dim: int,
    embedding_dim: int,
    num_classes: int,
) -> int:
    embedding_linear = backbone_dim * embedding_dim + embedding_dim
    layer_norm = 2 * embedding_dim
    classifier = embedding_dim * num_classes + num_classes
    return embedding_linear + layer_norm + classifier


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_inventory() -> dict[str, Any]:
    import torch
    import torchvision

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pytorchvideo": _package_version("pytorchvideo"),
        "ultralytics": _package_version("ultralytics"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _git_common_directory(repo_root: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    common = Path(result.stdout.strip())
    return (repo_root / common).resolve() if not common.is_absolute() else common.resolve()


def resolve_yolo_weights(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve(strict=True)

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [repo_root / "yolo11n-pose.pt"]
    common = _git_common_directory(repo_root)
    if common is not None:
        candidates.append(common.parent / "yolo11n-pose.pt")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate yolo11n-pose.pt in the worktree or Git common repository."
    )


def load_pretrained_x3d_s() -> Any:
    from pytorchvideo.models.hub import x3d_s

    return x3d_s(pretrained=True, progress=True)


def _serialize_state_dict(state_dict: dict[str, Any]) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def _custom_head_inventory() -> dict[str, Any]:
    import torch
    from torch import nn

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        head = nn.ModuleDict(
            {
                "embedding_head": nn.Sequential(
                    nn.Linear(2048, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.25),
                ),
                "classifier": nn.Linear(256, 40),
            }
        )
    serialized = _serialize_state_dict(head.state_dict())
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    return {
        "architecture": "Linear(2048,256)+LayerNorm+GELU+Dropout+Linear(256,40)",
        "parameter_count": parameter_count,
        "fp32_parameter_bytes": parameter_count * 4,
        "serialized_bytes_estimate": len(serialized),
        "sha256_estimate": _sha256_bytes(serialized),
        "embedded_in_final_x3d_checkpoint": True,
    }


def read_text_with_detected_encoding(path: Path) -> str:
    content = path.resolve(strict=True).read_bytes()
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig")
    return content.decode("utf-8", errors="replace")


def _read_pip_log(path: Path | None, exit_code: int | None) -> dict[str, Any]:
    output = None
    if path is not None:
        output = read_text_with_detected_encoding(path)
    return {
        "command": (
            r"D:\Anaconda\envs\pyTorch2.7\python.exe -m pip install "
            "-r requirements-x3d.txt"
        ),
        "exit_code": exit_code,
        "output": output,
        "installed_version": _package_version("pytorchvideo"),
        "installation_observed_successful": _package_version("pytorchvideo") == "0.1.5",
    }


def build_probe_payload(
    *,
    run_forward: bool,
    yolo_weights: Path | None = None,
    pip_log_path: Path | None = None,
    pip_exit_code: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_shape": INPUT_SHAPE,
        "model_name": MODEL_NAME,
        "pretraining": PRETRAINING,
        "source_revision": SOURCE_REVISION,
        "source_url": X3D_CHECKPOINT_URL,
        "license": metadata("pytorchvideo").get("License"),
        "internal_size_limit_bytes": INTERNAL_SIZE_LIMIT_BYTES,
        "runtime": _runtime_inventory(),
        "forward_requested": run_forward,
        "pip_install": _read_pip_log(pip_log_path, pip_exit_code),
    }
    if not run_forward:
        return payload

    import torch
    from ultralytics import YOLO

    model = load_pretrained_x3d_s().eval()
    x3d_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    x3d_fp32_parameter_bytes = x3d_parameter_count * 4
    x3d_checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / X3D_CHECKPOINT_FILENAME
    x3d_inventory = inventory_weight_file(
        x3d_checkpoint,
        parameter_count=x3d_parameter_count,
        fp32_parameter_bytes=x3d_fp32_parameter_bytes,
    )
    inference_state = _serialize_state_dict(model.state_dict())
    x3d_inventory.update(
        {
            "license": metadata("pytorchvideo").get("License"),
            "pretraining_data": "Kinetics-400",
            "checkpoint_url": X3D_CHECKPOINT_URL,
            "inference_state_dict_serialized_bytes": len(inference_state),
            "inference_state_dict_sha256": _sha256_bytes(inference_state),
            "source_checkpoint_includes_optimizer_state": True,
        }
    )

    yolo_path = resolve_yolo_weights(yolo_weights)
    yolo_model = YOLO(str(yolo_path)).model
    yolo_parameter_count = sum(parameter.numel() for parameter in yolo_model.parameters())
    yolo_inventory = inventory_weight_file(
        yolo_path,
        parameter_count=yolo_parameter_count,
        fp32_parameter_bytes=yolo_parameter_count * 4,
    )
    yolo_inventory.update(
        {
            "license": metadata("ultralytics").get("License"),
            "pretraining_data": "COCO pose",
            "ultralytics_version": _package_version("ultralytics"),
        }
    )

    head_inventory = _custom_head_inventory()
    aggregate_bytes = (
        x3d_inventory["serialized_bytes"]
        + yolo_inventory["serialized_bytes"]
        + head_inventory["serialized_bytes_estimate"]
    )
    enforce_submission_size_gate(aggregate_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_input = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model(model_input)
    output_is_finite = bool(torch.isfinite(output).all().item())
    if list(output.shape) != [1, 400]:
        raise RuntimeError(f"Unexpected X3D-S output shape: {list(output.shape)}")
    if not output_is_finite:
        raise RuntimeError("X3D-S output contains non-finite values.")

    payload.update(
        {
            "device": str(device),
            "output_shape": list(output.shape),
            "output_is_finite": output_is_finite,
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "components": {
                "x3d_s": x3d_inventory,
                "x3d_custom_head": head_inventory,
                "yolo11n_pose": yolo_inventory,
                "videomae_s": {
                    "required_at_inference": False,
                    "status": "teacher_only_pending_written_confirmation",
                },
            },
            "aggregate_inference_serialized_bytes": aggregate_bytes,
            "size_gate_passed": True,
            "size_accounting_note": (
                "Conservative Phase 0 estimate uses the complete official X3D source "
                "checkpoint, estimated custom-head state, and YOLO file. The final "
                "deployment checkpoint must be remeasured without double counting."
            ),
        }
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path)
    parser.add_argument("--pip-log", type=Path)
    parser.add_argument("--pip-exit-code", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_probe_payload(
        run_forward=True,
        yolo_weights=args.yolo_weights,
        pip_log_path=args.pip_log,
        pip_exit_code=args.pip_exit_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
