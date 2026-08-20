from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from types import TracebackType
from typing import Any

import torch
from torch import nn


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError("Hook output contains no tensor")


def activation_summary(value: Any) -> dict[str, Any]:
    tensor = _first_tensor(value).detach()
    finite = bool(torch.isfinite(tensor).all())
    values = tensor.float()
    leading = values.shape[0] if values.ndim else 1
    row_rms = values.reshape(leading, -1).square().mean(dim=1).sqrt()
    quantile_points = torch.tensor(
        [0.0, 0.5, 0.95, 1.0], device=row_rms.device, dtype=row_rms.dtype
    )
    row_quantiles = torch.quantile(row_rms, quantile_points)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "abs_max": float(values.abs().max()) if tensor.numel() else 0.0,
        "rms": float(values.square().mean().sqrt()) if tensor.numel() else 0.0,
        "mean": float(values.mean()) if tensor.numel() else 0.0,
        "std": float(values.std(unbiased=False)) if tensor.numel() else 0.0,
        "leading_axis_rms": {
            "min": float(row_quantiles[0]),
            "median": float(row_quantiles[1]),
            "p95": float(row_quantiles[2]),
            "max": float(row_quantiles[3]),
        },
    }


def state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ReadOnlyActivationTracer:
    """Forward hooks that only retain scalar summaries of selected outputs."""

    def __init__(self, model: nn.Module, module_names: Sequence[str]) -> None:
        modules = dict(model.named_modules())
        missing = [name for name in module_names if name not in modules]
        if missing:
            raise ValueError(f"Trace modules not found: {missing}")
        self._selected = [(name, modules[name]) for name in module_names]
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._counts: dict[str, int] = {}
        self._first_record_index: dict[str, int] = {}
        self.records: list[dict[str, Any]] = []

    def _capture(self, name: str, _module: nn.Module, _inputs: Any, output: Any) -> None:
        call_index = self._counts.get(name, 0)
        self._counts[name] = call_index + 1
        if call_index == 0:
            point = name
            self._first_record_index[name] = len(self.records)
        else:
            if call_index == 1:
                first = self._first_record_index[name]
                self.records[first]["point"] = f"{name}#0"
            point = f"{name}#{call_index}"
        self.records.append({"point": point, **activation_summary(output)})

    def __enter__(self) -> ReadOnlyActivationTracer:
        for name, module in self._selected:
            handle = module.register_forward_hook(
                lambda module, inputs, output, point=name: self._capture(
                    point, module, inputs, output
                )
            )
            self._handles.append(handle)
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


def first_amplification(
    spike: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    *,
    ratio_threshold: float,
) -> dict[str, Any]:
    if ratio_threshold <= 1.0 or len(spike) != len(control) or not spike:
        raise ValueError("Aligned non-empty traces and ratio_threshold > 1 are required")
    previous_point: str | None = None
    previous_ratio: float | None = None
    ratios: list[dict[str, Any]] = []
    for spike_row, control_row in zip(spike, control, strict=True):
        if spike_row["point"] != control_row["point"]:
            raise ValueError("Spike and control trace points differ")
        denominator = max(float(control_row["rms"]), 1e-12)
        ratio = float(spike_row["rms"]) / denominator
        row = {"point": str(spike_row["point"]), "ratio": ratio}
        ratios.append(row)
        if ratio >= ratio_threshold:
            return {
                **row,
                "previous_point": previous_point,
                "previous_ratio": previous_ratio,
                "threshold": ratio_threshold,
                "all_ratios": ratios,
            }
        previous_point = str(spike_row["point"])
        previous_ratio = ratio
    return {
        "point": None,
        "ratio": None,
        "previous_point": previous_point,
        "previous_ratio": previous_ratio,
        "threshold": ratio_threshold,
        "all_ratios": ratios,
    }
