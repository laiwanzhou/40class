from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegmentBatch:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    quality: torch.Tensor
    quality_mask: torch.Tensor

    def validate(self, *, batch: int, segments: int, streams: int, dim: int) -> None:
        if self.tokens.shape != (batch, segments, streams, dim):
            raise ValueError("segment token shape changed")
        if self.token_mask.shape != (batch, segments, streams):
            raise ValueError("segment token mask shape changed")
        if self.token_mask.dtype != torch.bool:
            raise ValueError("segment token mask must be bool")
        if self.quality.ndim != 4 or self.quality.shape[:3] != (
            batch,
            segments,
            streams,
        ):
            raise ValueError("segment quality shape changed")
        if self.quality_mask.shape != self.quality.shape:
            raise ValueError("segment quality mask shape changed")
        if self.quality_mask.dtype != torch.bool:
            raise ValueError("segment quality mask must be bool")
        if not bool(torch.isfinite(self.tokens[self.token_mask]).all()):
            raise ValueError("non-finite available segment token")
        if bool((self.tokens[~self.token_mask] != 0).any()):
            raise ValueError("masked segment tokens must be zero")
        if not bool(torch.isfinite(self.quality[self.quality_mask]).all()):
            raise ValueError("non-finite available segment quality")

