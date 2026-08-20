from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from scripts.probe_thermal_backbones import (
    IFORMER_CHECKPOINT_SHA256,
    IFORMER_CHECKPOINT_URL,
    IFORMER_REVISION,
    IFORMER_SOURCE_URL,
    _download,
    _load_iformer_source,
    _replace_iformer_head,
    _torch_checkpoint_path,
    load_official_iformer_checkpoint,
    require_complete_pretrained_load,
)
from src.models.expert_contract import ExpertOutput
from src.models.thermal_tsm import IFormerTSM


class ThermalIFormerExpert(nn.Module):
    """Trial-level common expert surface for mobile iFormer-T + TSM."""

    def __init__(self, spatial: IFormerTSM) -> None:
        super().__init__()
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
        logits = self.spatial.backbone.classifier(embedding)
        if isinstance(logits, tuple):
            raise RuntimeError("Distillation output is outside the Thermal contract")
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
        yield from self.spatial.backbone.classifier.parameters()

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        head_ids = {id(parameter) for parameter in self.head_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in head_ids:
                yield parameter


def build_pretrained_iformer_t_expert(
    *,
    num_segments: int = 16,
    fold_div: int = 8,
) -> ThermalIFormerExpert:
    """Build the audited official model; any source/hash/load failure is fatal."""
    cache = Path(torch.hub.get_dir()) / "thermal_t1a_mobile_iformer" / IFORMER_REVISION
    source_path = _download(IFORMER_SOURCE_URL, cache / "iformer.py")
    checkpoint_path = _download(IFORMER_CHECKPOINT_URL, _torch_checkpoint_path(IFORMER_CHECKPOINT_URL))
    backbone = _load_iformer_source(source_path, "iFormer_t")
    state = load_official_iformer_checkpoint(checkpoint_path, IFORMER_CHECKPOINT_SHA256)
    require_complete_pretrained_load(backbone, state)
    _replace_iformer_head(backbone, num_classes=40)
    return ThermalIFormerExpert(
        IFormerTSM(
            backbone=backbone,
            num_classes=40,
            num_segments=num_segments,
            fold_div=fold_div,
            shift_before_stages=(0, 1, 2, 3),
        )
    )
