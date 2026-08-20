from __future__ import annotations

from types import TracebackType
from typing import Any

import torch
from torch import nn


def classify_segment_source(energy_shares: list[float]) -> str:
    if len(energy_shares) != 16:
        raise ValueError("Thermal attribution requires exactly 16 segment shares")
    ordered = sorted((float(value) for value in energy_shares), reverse=True)
    if ordered[0] >= 0.5:
        return "single_segment"
    if sum(ordered[:3]) >= 0.7:
        return "few_segments"
    return "distributed_sequence"


def _energy_concentration(energy: torch.Tensor, *, top_fraction: float) -> dict[str, float]:
    values = energy.detach().float().flatten().clamp_min(0)
    total = float(values.sum())
    if total == 0.0:
        return {"top1_share": 0.0, "top_fraction_share": 0.0, "effective_fraction": 0.0}
    probabilities = values / total
    count = max(1, int(round(len(values) * top_fraction)))
    ordered = probabilities.sort(descending=True).values
    effective = 1.0 / float(probabilities.square().sum())
    return {
        "top1_share": float(ordered[0]),
        "top_fraction_share": float(ordered[:count].sum()),
        "effective_fraction": effective / len(values),
    }


def tensor_attribution(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    if values.ndim != 4 or values.shape[0] != 16:
        raise ValueError("Block attribution expects [16,C,H,W]")
    segment_energy = values.square().sum(dim=(1, 2, 3))
    total_energy = float(segment_energy.sum())
    shares = (
        (segment_energy / total_energy).tolist()
        if total_energy > 0.0
        else [0.0] * 16
    )
    ordered_shares = sorted(shares, reverse=True)
    per_segment: list[dict[str, Any]] = []
    for index, segment in enumerate(values):
        channel_energy = segment.square().sum(dim=(1, 2))
        spatial_energy = segment.square().sum(dim=0)
        channel = _energy_concentration(channel_energy, top_fraction=0.1)
        spatial = _energy_concentration(spatial_energy, top_fraction=0.01)
        per_segment.append(
            {
                "segment_index": index,
                "rms": float(segment.square().mean().sqrt()),
                "abs_max": float(segment.abs().max()),
                "energy_share": float(shares[index]),
                "channel_energy": {
                    "top1_share": channel["top1_share"],
                    "top10pct_share": channel["top_fraction_share"],
                    "effective_channel_fraction": channel["effective_fraction"],
                },
                "spatial_energy": {
                    "top1_share": spatial["top1_share"],
                    "top1pct_share": spatial["top_fraction_share"],
                    "effective_spatial_fraction": spatial["effective_fraction"],
                },
            }
        )
    return {
        "shape": list(values.shape),
        "finite": bool(torch.isfinite(values).all()),
        "rms": float(values.square().mean().sqrt()),
        "abs_max": float(values.abs().max()),
        "segment_energy": {
            "shares": [float(value) for value in shares],
            "top1_share": float(ordered_shares[0]),
            "top3_share": float(sum(ordered_shares[:3])),
            "effective_segment_fraction": (
                1.0 / sum(value * value for value in shares) / 16.0
                if total_energy > 0.0
                else 0.0
            ),
            "source": classify_segment_source([float(value) for value in shares]),
        },
        "per_segment": per_segment,
    }


def _cosine_relationship(skip: torch.Tensor, branch: torch.Tensor) -> dict[str, Any]:
    skip_rows = skip.detach().float().reshape(16, -1)
    branch_rows = branch.detach().float().reshape(16, -1)
    cosine = nn.functional.cosine_similarity(skip_rows, branch_rows, dim=1, eps=1e-12)
    return {
        "per_segment": [float(value) for value in cosine],
        "mean": float(cosine.mean()),
        "median": float(cosine.median()),
        "min": float(cosine.min()),
        "max": float(cosine.max()),
        "positive_fraction": float((cosine > 0).float().mean()),
    }


class ReadOnlyResidualAttributor:
    """Capture internal ConvBlock tensors without changing the residual forward."""

    def __init__(self, residual: nn.Module) -> None:
        if not hasattr(residual, "m") or not isinstance(residual.m, nn.Sequential):
            raise TypeError("Expected iFormer Residual with a Sequential branch")
        if len(residual.m) != 4:
            raise ValueError("Expected depthwise, expand, GELU, project branch")
        self.residual = residual
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captured: dict[str, torch.Tensor] = {}

    def _capture_input(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Residual input changed")
        self._captured["input"] = inputs[0].detach()

    def _capture_output(
        self, name: str, _module: nn.Module, _inputs: Any, output: Any
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Residual component output changed")
        self._captured[name] = output.detach()

    def __enter__(self) -> ReadOnlyResidualAttributor:
        self._handles.append(self.residual.register_forward_pre_hook(self._capture_input))
        names = ("depthwise_conv_bn", "expand_conv_bn", "gelu", "project_conv_bn")
        for name, module in zip(names, self.residual.m, strict=True):
            self._handles.append(
                module.register_forward_hook(
                    lambda module, inputs, output, point=name: self._capture_output(
                        point, module, inputs, output
                    )
                )
            )
        self._handles.append(
            self.residual.register_forward_hook(
                lambda module, inputs, output: self._capture_output(
                    "output", module, inputs, output
                )
            )
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def report(self) -> dict[str, Any]:
        expected = {
            "input",
            "depthwise_conv_bn",
            "expand_conv_bn",
            "gelu",
            "project_conv_bn",
            "output",
        }
        if set(self._captured) != expected:
            raise RuntimeError("Residual attribution capture is incomplete")
        captured = {name: value.float().cpu() for name, value in self._captured.items()}
        skip = captured["input"]
        raw_project = captured["project_conv_bn"]
        output = captured["output"]
        gamma = getattr(self.residual, "gamma", None)
        scaled_project = (
            raw_project
            if gamma is None
            else raw_project * gamma.detach().float().cpu()
        )
        actual_branch = output - skip
        identity_error = (output - (skip + scaled_project)).abs().max()
        branch_error = (actual_branch - scaled_project).abs().max()
        result = {
            name: tensor_attribution(value)
            for name, value in captured.items()
            if name != "output"
        }
        result["residual_branch"] = tensor_attribution(actual_branch)
        result["residual_add_output"] = tensor_attribution(output)
        result["skip_branch"] = result["input"]
        result["skip_branch_cosine"] = _cosine_relationship(skip, actual_branch)
        skip_rms = max(result["skip_branch"]["rms"], 1e-12)
        result["branch_to_skip_rms_ratio"] = result["residual_branch"]["rms"] / skip_rms
        result["output_to_skip_rms_ratio"] = result["residual_add_output"]["rms"] / skip_rms
        result["gamma_present"] = gamma is not None
        result["residual_identity_max_abs_error"] = float(identity_error)
        result["raw_project_to_actual_branch_max_abs_error"] = float(branch_error)
        return result
