from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroupTokens:
    tokens: torch.Tensor
    mask: torch.Tensor
    quality: torch.Tensor
    quality_mask: torch.Tensor

    def validate(
        self, *, batch: int, segments: int, streams: int, dim: int
    ) -> None:
        if self.tokens.shape != (batch, segments, streams, dim):
            raise ValueError("group token shape changed")
        if self.mask.shape != (batch, segments, streams) or self.mask.dtype != torch.bool:
            raise ValueError("group token mask changed")
        if self.quality.ndim != 4 or self.quality.shape[:3] != self.mask.shape:
            raise ValueError("group quality shape changed")
        if self.quality_mask.shape != self.quality.shape or self.quality_mask.dtype != torch.bool:
            raise ValueError("group quality mask changed")
        if not bool(torch.isfinite(self.tokens[self.mask]).all()):
            raise ValueError("non-finite available group token")
        if bool((self.tokens[~self.mask] != 0).any()):
            raise ValueError("masked group tokens must be zero")

