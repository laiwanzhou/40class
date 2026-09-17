from __future__ import annotations

import torch
from torch import nn


class SelectiveEventCorrector(nn.Module):
    """A coherent event classifier used as a selective base correction.

    Unlike an unrestricted residual head, the only class-dependent correction
    is the centred output of an independently supervised event classifier.
    A second head predicts whether the frozen legal base is likely to be
    incorrect.  The global mixing scale is initialised to exactly zero, so the
    initial output is numerically identical to the base logits.
    """

    def __init__(
        self,
        modality_hidden: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        segment_count: int = 4,
        base_hidden: int = 64,
        representation_dim: int = 128,
        projection_dim: int = 64,
        dropout: float = 0.20,
        delta_limit: float = 4.0,
        scale_limit: float = 2.0,
        event_target_count: int = 9,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.segment_count = int(segment_count)
        self.delta_limit = float(delta_limit)
        self.scale_limit = float(scale_limit)
        self.skeleton = nn.Sequential(
            nn.Linear(16, modality_hidden),
            nn.LayerNorm(modality_hidden),
            nn.GELU(),
        )
        self.imu = nn.Sequential(
            nn.Linear(25, modality_hidden),
            nn.LayerNorm(modality_hidden),
            nn.GELU(),
        )
        self.visual = nn.Sequential(
            nn.Linear(10, modality_hidden),
            nn.LayerNorm(modality_hidden),
            nn.GELU(),
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(modality_hidden * 3 + 4, gru_hidden),
            nn.LayerNorm(gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sequence = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        recurrent_dim = 2 * gru_hidden
        self.representation = nn.Sequential(
            nn.LayerNorm(recurrent_dim * (segment_count + 1)),
            nn.Linear(recurrent_dim * (segment_count + 1), representation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_classifier = nn.Linear(representation_dim, num_classes)
        self.event_regressor = nn.Linear(representation_dim, event_target_count)
        self.metric_projector = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.GELU(),
            nn.Linear(representation_dim, projection_dim),
        )
        self.order_head = nn.Linear(representation_dim, 1)
        self.base = nn.Sequential(
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, base_hidden),
            nn.GELU(),
        )
        selector_input = base_hidden + representation_dim + 22
        self.error_selector = nn.Sequential(
            nn.LayerNorm(selector_input),
            nn.Linear(selector_input, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        self.raw_mix_scale = nn.Parameter(torch.zeros(()))

    def encode_event(
        self,
        event_sequence: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        skeleton = self.skeleton(event_sequence[:, :, :16])
        imu = self.imu(event_sequence[:, :, 16:41])
        visual = self.visual(event_sequence[:, :, 41:51])
        repeated_mask = modality_mask[:, None].expand(
            -1, event_sequence.shape[1], -1
        )
        fused = self.local_fusion(
            torch.cat([skeleton, imu, visual, repeated_mask], dim=2)
        )
        output, hidden = self.sequence(fused)
        segment_states = [
            segment.mean(dim=1)
            for segment in torch.chunk(output, self.segment_count, dim=1)
        ]
        terminal = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.representation(
            torch.cat([*segment_states, terminal], dim=1)
        )

    def forward(
        self,
        base_logits: torch.Tensor,
        event_sequence: torch.Tensor,
        modality_mask: torch.Tensor,
        event_quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.encode_event(event_sequence, modality_mask)
        event_logits = self.event_classifier(representation)
        centred = event_logits - event_logits.mean(dim=1, keepdim=True)
        event_delta = self.delta_limit * torch.tanh(
            centred / self.delta_limit
        )
        selector_input = torch.cat(
            [
                self.base(base_logits),
                representation,
                event_quality,
                modality_mask,
            ],
            dim=1,
        )
        base_error_logit = self.error_selector(selector_input)
        base_error_probability = torch.sigmoid(base_error_logit)
        mix_scale = self.scale_limit * torch.tanh(self.raw_mix_scale)
        logits = (
            base_logits
            + mix_scale * base_error_probability * event_delta
        )
        return {
            "logits": logits,
            "event_logits": event_logits,
            "event_targets": torch.sigmoid(
                self.event_regressor(representation)
            ),
            "metric_embedding": self.metric_projector(representation),
            "order_logit": self.order_head(representation).squeeze(1),
            "base_error_logit": base_error_logit.squeeze(1),
            "base_error_probability": base_error_probability,
            "mix_scale": mix_scale,
            "event_delta": event_delta,
            "event_representation": representation,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
