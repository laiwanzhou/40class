from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from scripts.probe_thermal_backbones import (
    MOBILENET_WEIGHT_SHA256,
    _download,
    _torch_checkpoint_path,
    require_complete_pretrained_load,
    require_file_sha256,
)
from src.models.expert_contract import ExpertOutput
from src.models.thermal_tsm import MobileNetV3SmallTSM


class BNLinearClassifier(nn.Module):
    """The same BN-to-Linear head structure used by the iFormer-T candidate."""

    def __init__(self, in_features: int, num_classes: int = 40) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.l = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.l(self.bn(inputs))


class ThermalMobileNetExpert(nn.Module):
    """Trial-level expert surface for the matched MobileNetV3-Small + TSM control."""

    def __init__(self, spatial: MobileNetV3SmallTSM) -> None:
        super().__init__()
        feature_dim = int(spatial.classifier[0].in_features)
        spatial.classifier = BNLinearClassifier(feature_dim, num_classes=40)
        self.spatial = spatial

    def forward(
        self,
        clips: torch.Tensor,
        quality: torch.Tensor,
        quality_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> ExpertOutput:
        rows = clips.shape[0]
        if quality.ndim != 2 or quality.shape[0] != rows:
            raise ValueError("quality must be [B,Q]")
        if quality_mask.shape != quality.shape or quality_mask.dtype != torch.bool:
            raise ValueError("quality_mask must be boolean and match quality")
        if availability.shape != (rows,) or availability.dtype != torch.bool:
            raise ValueError("availability must be boolean [B]")
        embedding = self.spatial.forward_features(clips)
        logits = self.spatial.classifier(embedding)
        if logits.shape != (rows, 40) or not torch.isfinite(logits).all():
            raise RuntimeError("Thermal expert must emit finite [B,40] logits")
        return ExpertOutput(
            main_logits=logits,
            embedding=embedding,
            quality=quality,
            quality_mask=quality_mask,
            availability=availability,
        )

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.spatial.classifier.parameters()

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        head_ids = {id(parameter) for parameter in self.head_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in head_ids:
                yield parameter


def build_pretrained_mobilenet_expert(
    *,
    num_segments: int = 16,
    fold_div: int = 8,
) -> ThermalMobileNetExpert:
    """Build the audited official model; hash or strict-load failure is fatal."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    checkpoint_path = _download(weights.url, _torch_checkpoint_path(weights.url))
    require_file_sha256(Path(checkpoint_path), MOBILENET_WEIGHT_SHA256)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone = mobilenet_v3_small(weights=None)
    require_complete_pretrained_load(backbone, state)
    return ThermalMobileNetExpert(
        MobileNetV3SmallTSM(
            weights=None,
            backbone=backbone,
            num_classes=40,
            num_segments=num_segments,
            fold_div=fold_div,
            shift_before_blocks=(1, 3, 6, 9),
        )
    )
