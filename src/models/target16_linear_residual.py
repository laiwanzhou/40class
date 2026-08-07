from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Target16LinearResidual(nn.Module):
    """A zero-initialized conditional residual over frozen B2 logits."""

    def __init__(self, embedding_dim: int, target_class_ids: Sequence[int]) -> None:
        super().__init__()
        target_ids = tuple(int(value) for value in target_class_ids)
        if len(target_ids) != 16 or len(set(target_ids)) != 16:
            raise ValueError("Exactly 16 unique target class IDs are required.")
        if target_ids != tuple(sorted(target_ids)):
            raise ValueError("Target class IDs must use ascending dataset label order.")
        self.register_buffer("target_index", torch.tensor(target_ids, dtype=torch.long), persistent=True)
        self.residual_head = nn.Linear(embedding_dim, len(target_ids))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @property
    def target_class_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.target_index.tolist())

    def forward(self, embeddings: torch.Tensor, base_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.residual_head.in_features:
            raise ValueError("Unexpected B2 embedding shape.")
        if base_logits.ndim != 2 or base_logits.shape != (len(embeddings), 40):
            raise ValueError("B2 logits must have shape [N, 40].")
        base_target_logits = base_logits.index_select(1, self.target_index)
        delta = self.residual_head(embeddings)
        return {
            "logits": base_target_logits.detach() + delta,
            "base_target_logits": base_target_logits.detach(),
            "delta_logits": delta,
        }
