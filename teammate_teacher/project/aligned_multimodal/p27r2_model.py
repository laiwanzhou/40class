from __future__ import annotations

import torch
from torch import nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            TemporalResidualBlock(hidden, 1, dropout),
            TemporalResidualBlock(hidden, 2, dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.transpose(1, 2)).transpose(1, 2)


class P27R2ResidualModel(nn.Module):
    """Small locally aligned event encoder on top of frozen base logits.

    The three modalities retain 32 normalized-progress tokens. Interaction is
    restricted to same/neighboring time bins by the temporal convolutions; no
    free global cross-modal attention is used.
    """

    def __init__(
        self,
        skeleton_dim: int,
        imu_dim: int,
        visual_dim: int,
        event_count: int,
        hidden: int = 64,
        event_dim: int = 96,
        dropout: float = 0.20,
        num_classes: int = 40,
        alpha_limit: float = 0.15,
        delta_limit: float = 2.0,
    ) -> None:
        super().__init__()
        self.skeleton_encoder = TemporalEncoder(skeleton_dim, hidden, dropout)
        self.imu_encoder = TemporalEncoder(imu_dim, hidden, dropout)
        self.visual_encoder = TemporalEncoder(visual_dim, hidden, dropout)
        self.missing_embedding = nn.Parameter(torch.zeros(3, hidden))
        nn.init.normal_(self.missing_embedding, std=0.02)
        self.local_fusion = nn.Sequential(
            nn.Linear(hidden * 3 + 4, event_dim),
            nn.GELU(),
            nn.LayerNorm(event_dim),
            nn.Dropout(dropout),
        )
        self.event_temporal = nn.Sequential(
            nn.Conv1d(event_dim, event_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, event_dim),
            nn.GELU(),
            TemporalResidualBlock(event_dim, 2, dropout),
        )
        pooled_dim = event_dim * 2 + 4
        self.event_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, event_count),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.alpha_raw = nn.Parameter(torch.zeros(()))
        self.alpha_limit = float(alpha_limit)
        self.delta_limit = float(delta_limit)

    def _presence(
        self, token: torch.Tensor, present: torch.Tensor, modality: int
    ) -> torch.Tensor:
        mask = present[:, None, None]
        missing = self.missing_embedding[modality][None, None]
        return token * mask + missing * (1.0 - mask)

    def forward(
        self,
        skeleton: torch.Tensor,
        imu: torch.Tensor,
        visual: torch.Tensor,
        modality_mask: torch.Tensor,
        base_logits: torch.Tensor,
        event_shuffle: torch.Tensor | None = None,
        zero_event: bool = False,
    ) -> dict[str, torch.Tensor]:
        skeleton_token = self._presence(
            self.skeleton_encoder(skeleton), modality_mask[:, 0], 0
        )
        imu_token = self._presence(
            self.imu_encoder(imu), modality_mask[:, 1], 1
        )
        visual_present = torch.maximum(modality_mask[:, 2], modality_mask[:, 3])
        visual_token = self._presence(
            self.visual_encoder(visual), visual_present, 2
        )
        repeated_mask = modality_mask[:, None].expand(
            -1, skeleton_token.shape[1], -1
        )
        joint = self.local_fusion(
            torch.cat(
                [skeleton_token, imu_token, visual_token, repeated_mask], dim=2
            )
        )
        event_sequence = self.event_temporal(joint.transpose(1, 2)).transpose(1, 2)
        pooled = torch.cat(
            [
                event_sequence.mean(dim=1),
                event_sequence.amax(dim=1),
                modality_mask,
            ],
            dim=1,
        )
        if event_shuffle is not None:
            pooled = pooled[event_shuffle]
        if zero_event:
            pooled = torch.zeros_like(pooled)
        event_prediction = torch.sigmoid(self.event_head(pooled))
        raw_delta = self.delta_head(pooled)
        raw_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        alpha = self.alpha_limit * torch.tanh(self.alpha_raw)
        logits = base_logits + alpha * delta
        return {
            "logits": logits,
            "delta": delta,
            "alpha": alpha,
            "event_prediction": event_prediction,
            "event_representation": pooled,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
