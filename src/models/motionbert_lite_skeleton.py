from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from scripts.fetch_motionbert_lite_checkpoint import verify_motionbert_checkpoint
from src.experiments.motionbert_p6b_config import project_path
from third_party.motionbert import DSTformer


@dataclass(frozen=True)
class CheckpointCoverage:
    loaded_elements: int
    total_elements: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def element_fraction(self) -> float:
        return self.loaded_elements / max(self.total_elements, 1)


class MotionBERTLiteSkeletonExpert(nn.Module):
    def __init__(
        self,
        *,
        backbone: DSTformer,
        dim_rep: int = 512,
        classes: int = 40,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_norm = nn.LayerNorm(dim_rep)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim_rep, classes)

    def forward(
        self, sequence: torch.Tensor, available: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if sequence.ndim != 4 or sequence.shape[2:] != (17, 3):
            raise ValueError("MotionBERT sequence must have shape [B,T,17,3]")
        if available.shape != (sequence.shape[0],) or available.dtype != torch.bool:
            raise ValueError("MotionBERT availability must be bool [B]")
        sequence_features = self.backbone.get_representation(sequence)
        embedding = self.head_norm(sequence_features.mean(dim=(1, 2)))
        mask = available[:, None].to(embedding.dtype)
        embedding = embedding * mask
        logits = self.classifier(self.dropout(embedding)) * mask
        sequence_features = sequence_features * available[:, None, None, None].to(
            sequence_features.dtype
        )
        return {
            "sequence_features": sequence_features,
            "embedding": embedding,
            "logits": logits,
            "available": available,
        }


def set_motionbert_train_stage(
    model: MotionBERTLiteSkeletonExpert, stage: Literal["B1", "B2"]
) -> None:
    if stage not in {"B1", "B2"}:
        raise ValueError("MotionBERT train stage must be B1 or B2")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.head_norm, model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if stage == "B2":
        for index in (3, 4):
            for module in (
                model.backbone.blocks_st[index],
                model.backbone.blocks_ts[index],
                model.backbone.ts_attn[index],
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for module in (model.backbone.norm, model.backbone.pre_logits):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_pos") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("MotionBERT checkpoint misses model_pos state")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("MotionBERT checkpoint state contains non-tensors")
    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
    }


def _load_backbone_state(
    backbone: DSTformer, state: dict[str, torch.Tensor]
) -> CheckpointCoverage:
    target = backbone.state_dict()
    missing = tuple(sorted(key for key in target if key not in state))
    unexpected = tuple(sorted(key for key in state if key not in target))
    mismatches = tuple(
        sorted(
            (
                key,
                tuple(target[key].shape),
                tuple(state[key].shape),
            )
            for key in target
            if key in state and target[key].shape != state[key].shape
        )
    )
    compatible = {
        key: state[key]
        for key in target
        if key in state and target[key].shape == state[key].shape
    }
    loaded_elements = sum(target[key].numel() for key in compatible)
    total_elements = sum(value.numel() for value in target.values())
    coverage = CheckpointCoverage(
        loaded_elements=loaded_elements,
        total_elements=total_elements,
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=mismatches,
    )
    backbone.load_state_dict(compatible, strict=False)
    return coverage


def build_motionbert_lite_expert(
    config: dict[str, Any],
) -> tuple[MotionBERTLiteSkeletonExpert, CheckpointCoverage]:
    model_config = config["model"]
    checkpoint_config = config["checkpoint"]
    checkpoint_path = project_path(str(checkpoint_config["path"])).resolve()
    verify_motionbert_checkpoint(
        checkpoint_path,
        expected_bytes=int(checkpoint_config["bytes"]),
        expected_sha256=str(checkpoint_config["sha256"]),
    )
    backbone = DSTformer(
        dim_in=int(model_config["dim_in"]),
        dim_out=3,
        dim_feat=int(model_config["dim_feat"]),
        dim_rep=int(model_config["dim_rep"]),
        depth=int(model_config["depth"]),
        num_heads=int(model_config["num_heads"]),
        mlp_ratio=int(model_config["mlp_ratio"]),
        num_joints=int(model_config["num_joints"]),
        maxlen=int(model_config["maxlen"]),
        att_fuse=bool(model_config["att_fuse"]),
    )
    coverage = _load_backbone_state(backbone, _checkpoint_state(checkpoint_path))
    if coverage.element_fraction < float(
        config["b0"]["minimum_pretrained_element_coverage"]
    ):
        raise RuntimeError(
            f"MotionBERT pretrained element coverage too low: {coverage.element_fraction}"
        )
    expert = MotionBERTLiteSkeletonExpert(
        backbone=backbone,
        dim_rep=int(model_config["dim_rep"]),
        classes=40,
        dropout=float(model_config["dropout"]),
    )
    return expert, coverage
