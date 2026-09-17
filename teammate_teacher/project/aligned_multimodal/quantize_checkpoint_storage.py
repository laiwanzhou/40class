from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert floating tensors in a PyTorch checkpoint state dict to FP16 "
            "for storage. Models may still be instantiated and evaluated in FP32."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_state_dict(state_dict: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    converted: dict[str, Any] = {}
    floating_tensors = 0
    integer_tensors = 0
    parameters = 0
    max_roundtrip_abs_error = 0.0
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            converted[key] = value
            continue
        parameters += value.numel()
        if value.is_floating_point():
            floating_tensors += 1
            half_value = value.detach().cpu().to(torch.float16)
            converted[key] = half_value
            if value.numel():
                error = (half_value.float() - value.detach().cpu().float()).abs().max()
                max_roundtrip_abs_error = max(
                    max_roundtrip_abs_error, float(error.item())
                )
        else:
            integer_tensors += 1
            converted[key] = value.detach().cpu()
    return converted, {
        "tensor_count": floating_tensors + integer_tensors,
        "floating_tensor_count": floating_tensors,
        "integer_tensor_count": integer_tensors,
        "parameter_and_buffer_elements": parameters,
        "max_tensor_roundtrip_abs_error": max_roundtrip_abs_error,
    }


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("Input and output checkpoints must be different files.")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing model_state_dict.")

    converted_state, tensor_summary = convert_state_dict(
        checkpoint["model_state_dict"]
    )
    converted_checkpoint = dict(checkpoint)
    converted_checkpoint["model_state_dict"] = converted_state
    converted_checkpoint["storage_dtype"] = "float16"
    converted_checkpoint["storage_source"] = str(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_checkpoint, output)
    reloaded = torch.load(output, map_location="cpu", weights_only=False)
    if list(reloaded["model_state_dict"]) != list(checkpoint["model_state_dict"]):
        raise RuntimeError("State-dict keys changed after conversion.")
    for key, original in checkpoint["model_state_dict"].items():
        candidate = reloaded["model_state_dict"][key]
        if isinstance(original, torch.Tensor):
            if original.shape != candidate.shape:
                raise RuntimeError(f"Shape changed for {key}.")
            if original.is_floating_point() and candidate.dtype != torch.float16:
                raise RuntimeError(f"Floating tensor {key} was not stored as FP16.")
            if not original.is_floating_point() and not torch.equal(
                original.cpu(), candidate
            ):
                raise RuntimeError(f"Non-floating tensor {key} changed.")

    source_bytes = source.stat().st_size
    output_bytes = output.stat().st_size
    summary = {
        "method": (
            "Floating state-dict tensors are stored as IEEE FP16. The evaluator "
            "loads them into an FP32 model, so this changes storage precision, "
            "not the runtime architecture."
        ),
        "source": {
            "path": str(source),
            "bytes": source_bytes,
            "size_mib": source_bytes / 2**20,
            "sha256": sha256(source),
        },
        "output": {
            "path": str(output),
            "bytes": output_bytes,
            "size_mib": output_bytes / 2**20,
            "sha256": sha256(output),
        },
        "compression_ratio": output_bytes / source_bytes,
        **tensor_summary,
    }
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else output.with_suffix(output.suffix + ".json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
