from __future__ import annotations

import torch
from torch import nn


class ConditionalEventSequenceModel(nn.Module):
    """Phase-preserving event correction over legal frozen base logits.

    The model keeps four ordered temporal segments and the terminal recurrent
    states. It does not reduce the event stream to only a global mean/maximum.
    The intervention gate is sample-adaptive rather than a single global alpha.
    """

    def __init__(
        self,
        modality_hidden: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        segment_count: int = 4,
        base_hidden: int = 96,
        joint_hidden: int = 160,
        dropout: float = 0.20,
        delta_limit: float = 4.0,
        gate_initial_bias: float = -2.5,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.segment_count = int(segment_count)
        self.delta_limit = float(delta_limit)
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
        sequence_dim = 2 * gru_hidden
        self.segment_project = nn.Sequential(
            nn.LayerNorm(sequence_dim * (segment_count + 1)),
            nn.Linear(sequence_dim * (segment_count + 1), joint_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base = nn.Sequential(
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, base_hidden),
            nn.GELU(),
        )
        gate_input = base_hidden + joint_hidden + 22
        self.joint = nn.Sequential(
            nn.LayerNorm(gate_input),
            nn.Linear(gate_input, joint_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(joint_hidden, num_classes)
        self.gate = nn.Linear(gate_input, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_initial_bias))

    def encode_sequence(
        self,
        event_sequence: torch.Tensor,
        modality_mask: torch.Tensor,
        disable_event: bool = False,
    ) -> torch.Tensor:
        skeleton = self.skeleton(event_sequence[:, :, :16])
        imu = self.imu(event_sequence[:, :, 16:41])
        visual = self.visual(event_sequence[:, :, 41:51])
        repeated_mask = modality_mask[:, None].expand(-1, event_sequence.shape[1], -1)
        fused = self.local_fusion(
            torch.cat([skeleton, imu, visual, repeated_mask], dim=2)
        )
        if disable_event:
            fused = torch.zeros_like(fused)
        output, hidden = self.sequence(fused)
        segments = torch.chunk(output, self.segment_count, dim=1)
        segment_states = [segment.mean(dim=1) for segment in segments]
        terminal = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.segment_project(torch.cat([*segment_states, terminal], dim=1))

    def forward(
        self,
        base_logits: torch.Tensor,
        event_sequence: torch.Tensor,
        modality_mask: torch.Tensor,
        event_quality: torch.Tensor,
        disable_event: bool = False,
    ) -> dict[str, torch.Tensor]:
        base_feature = self.base(base_logits)
        event_feature = self.encode_sequence(
            event_sequence, modality_mask, disable_event=disable_event
        )
        quality = torch.cat([event_quality, modality_mask], dim=1)
        gate_input = torch.cat([base_feature, event_feature, quality], dim=1)
        joint = self.joint(gate_input)
        raw_delta = self.delta(joint)
        raw_delta = raw_delta - raw_delta.mean(dim=1, keepdim=True)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        gate = torch.sigmoid(self.gate(gate_input))
        logits = base_logits + gate * delta
        return {
            "logits": logits,
            "gate": gate,
            "delta": delta,
            "event_feature": event_feature,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
