from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertOutput:
    main_logits: torch.Tensor
    embedding: torch.Tensor
    quality: torch.Tensor
    quality_mask: torch.Tensor
    availability: torch.Tensor


@dataclass(frozen=True)
class VisualExpertForward:
    expert: ExpertOutput
    small_gate_logits: torch.Tensor
    sequence_features: torch.Tensor
    short_attention: torch.Tensor
    long_attention: torch.Tensor
    ir_roi_attention: torch.Tensor
    depth_relation_gate: torch.Tensor
    depth_gate: torch.Tensor


@dataclass(frozen=True)
class ExpertBatchResult:
    sample_ids: tuple[str, ...]
    class_map_hash: str
    output: ExpertOutput
    small_gate_logits: torch.Tensor | None = None
    sequence_features: torch.Tensor | None = None
    temporal_mask: torch.Tensor | None = None
    timestamps: torch.Tensor | None = None

    def validate(self) -> None:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Expert batch contains duplicate sample IDs")
        rows = len(self.sample_ids)
        fixed_outputs = {
            "main_logits": self.output.main_logits,
            "embedding": self.output.embedding,
            "quality": self.output.quality,
            "quality_mask": self.output.quality_mask,
            "availability": self.output.availability,
        }
        for name, value in fixed_outputs.items():
            if value.ndim == 0 or value.shape[0] != rows:
                raise ValueError(f"Expert {name} rows do not match sample IDs")
        if self.output.quality_mask.shape != self.output.quality.shape:
            raise ValueError("Expert quality mask does not match quality")
        for name, value in (
            ("small_gate_logits", self.small_gate_logits),
            ("sequence_features", self.sequence_features),
            ("temporal_mask", self.temporal_mask),
            ("timestamps", self.timestamps),
        ):
            if value is not None and (value.ndim == 0 or value.shape[0] != rows):
                raise ValueError(f"Expert {name} rows do not match sample IDs")
        if not self.class_map_hash:
            raise ValueError("class_map_hash must be non-empty")


def align_expert_batch(reference: ExpertBatchResult, other: ExpertBatchResult) -> torch.Tensor:
    """Return indices that reorder ``other`` to ``reference`` after strict ID checks."""
    reference.validate()
    other.validate()
    if reference.class_map_hash != other.class_map_hash:
        raise ValueError("Expert class maps differ")
    if set(reference.sample_ids) != set(other.sample_ids):
        missing = sorted(set(reference.sample_ids) - set(other.sample_ids))
        extra = sorted(set(other.sample_ids) - set(reference.sample_ids))
        raise ValueError(f"Expert sample sets differ: missing={missing[:5]}, extra={extra[:5]}")
    lookup = {sample_id: index for index, sample_id in enumerate(other.sample_ids)}
    return torch.tensor([lookup[sample_id] for sample_id in reference.sample_ids], dtype=torch.long)


def calibrated_probability_mixture(
    visual_logits: torch.Tensor,
    sensor_logits: torch.Tensor,
    alpha: torch.Tensor | float,
    *,
    visual_temperature: float = 1.0,
    sensor_temperature: float = 1.0,
) -> torch.Tensor:
    """Predeclared future fusion primitive; alpha=0 exactly recovers visual probabilities."""
    if visual_logits.shape != sensor_logits.shape:
        raise ValueError("Expert logits must have identical shapes")
    if visual_temperature <= 0 or sensor_temperature <= 0:
        raise ValueError("Temperatures must be positive")
    weight = torch.as_tensor(alpha, dtype=visual_logits.dtype, device=visual_logits.device)
    if torch.any((weight < 0) | (weight > 1)):
        raise ValueError("alpha must lie in [0, 1]")
    while weight.ndim < visual_logits.ndim:
        weight = weight.unsqueeze(-1)
    visual = torch.softmax(visual_logits / visual_temperature, dim=-1)
    sensor = torch.softmax(sensor_logits / sensor_temperature, dim=-1)
    return (1.0 - weight) * visual + weight * sensor
