from __future__ import annotations

import torch
from torch import nn

from .tcn import TemporalClassifier


class SegmentAwareConv1d(nn.Conv1d):
    def forward(
        self, inputs: torch.Tensor, segment_ids: torch.Tensor, temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.groups != 1 or self.stride != (1,):
            raise ValueError("T1-v1 supports only the original ungrouped stride-1 convolutions")
        if inputs.ndim != 3 or segment_ids.shape != (inputs.shape[0], inputs.shape[2]):
            raise ValueError("Expected inputs [B,C,T] and segment IDs [B,T]")
        if temporal_mask.shape != segment_ids.shape:
            raise ValueError("temporal_mask must match segment IDs")
        output = inputs.new_zeros((inputs.shape[0], self.out_channels, inputs.shape[2]))
        time_steps = inputs.shape[2]
        center = self.padding[0] // self.dilation[0]
        for kernel_index in range(self.kernel_size[0]):
            offset = (kernel_index - center) * self.dilation[0]
            shifted = torch.roll(inputs, shifts=-offset, dims=2)
            shifted_segments = torch.roll(segment_ids, shifts=-offset, dims=1)
            shifted_mask = torch.roll(temporal_mask, shifts=-offset, dims=1)
            boundary = torch.ones_like(temporal_mask)
            if abs(offset) >= time_steps:
                boundary.fill_(False)
            elif offset > 0:
                boundary[:, time_steps - offset:] = False
            elif offset < 0:
                boundary[:, :-offset] = False
            valid = (
                boundary & temporal_mask & shifted_mask & (segment_ids == shifted_segments)
                & (segment_ids >= 0)
            )
            contribution = torch.einsum("bct,oc->bot", shifted, self.weight[:, :, kernel_index])
            output = output + contribution * valid[:, None].to(output.dtype)
        if self.bias is not None:
            output = output + self.bias[None, :, None]
        return output * temporal_mask[:, None].to(output.dtype)


class SegmentAwareResidualTemporalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SegmentAwareConv1d(
            input_channels, output_channels, kernel_size=5, padding=dilation * 2, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = SegmentAwareConv1d(
            output_channels, output_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.skip = nn.Identity() if input_channels == output_channels else nn.Conv1d(input_channels, output_channels, 1)

    @staticmethod
    def apply_mask(values: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        return values * temporal_mask[:, None].to(values.dtype)

    def forward(
        self, inputs: torch.Tensor, segment_ids: torch.Tensor, temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_inputs = self.apply_mask(inputs, temporal_mask)
        values = self.conv1(masked_inputs, segment_ids, temporal_mask)
        values = self.apply_mask(self.dropout1(self.activation1(self.bn1(values))), temporal_mask)
        values = self.conv2(values, segment_ids, temporal_mask)
        values = self.apply_mask(self.dropout2(self.activation2(self.bn2(values))), temporal_mask)
        return self.apply_mask(values + self.skip(masked_inputs), temporal_mask)


class SegmentAwareTemporalClassifier(nn.Module):
    def __init__(
        self, input_features: int, embedding_dim: int = 128, num_classes: int = 40,
        channels: tuple[int, ...] = (64, 128), dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks = []
        current = input_features
        for index, channel in enumerate(channels):
            blocks.append(SegmentAwareResidualTemporalBlock(current, channel, 2**index, dropout))
            current = channel
        self.encoder = nn.ModuleList(blocks)
        self.projection = nn.Sequential(nn.Linear(current, embedding_dim), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, inputs: dict[str, torch.Tensor], temporal_mask: torch.Tensor) -> torch.Tensor:
        features = inputs["features"]
        segment_ids = inputs["segment_ids"]
        if features.ndim != 3 or segment_ids.shape != features.shape[:2]:
            raise ValueError("Expected features [B,T,F] and segment IDs [B,T]")
        mask = temporal_mask.bool() & (segment_ids >= 0)
        encoded = features.transpose(1, 2)
        for block in self.encoder:
            encoded = block(encoded, segment_ids, mask)
        return encoded.transpose(1, 2)

    def forward(
        self, inputs: dict[str, torch.Tensor], temporal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        segment_ids = inputs["segment_ids"]
        if temporal_mask is None:
            temporal_mask = segment_ids >= 0
        mask = temporal_mask.bool() & (segment_ids >= 0)
        encoded = self.encode(inputs, mask)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        embedding = self.projection(pooled)
        return {"embedding": embedding, "logits": self.classifier(embedding)}

    def load_from_temporal_classifier(self, source: TemporalClassifier) -> None:
        if len(self.encoder) != len(source.encoder):
            raise ValueError("Encoder depth mismatch")
        for target, original in zip(self.encoder, source.encoder, strict=True):
            target.conv1.load_state_dict(original.net[0].state_dict())
            target.bn1.load_state_dict(original.net[1].state_dict())
            target.conv2.load_state_dict(original.net[4].state_dict())
            target.bn2.load_state_dict(original.net[5].state_dict())
            target.skip.load_state_dict(original.skip.state_dict())
        self.projection.load_state_dict(source.projection.state_dict())
        self.classifier.load_state_dict(source.classifier.state_dict())
