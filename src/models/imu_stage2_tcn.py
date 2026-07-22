from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch
from torch import nn


SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
CHECKPOINT_HASH_FIELDS = (
    "stage2_contract_sha256",
    "training_index_sha256",
    "normalization_contract_sha256",
    "normalization_file_sha256",
    "class_order_sha256",
    "submission_contract_sha256",
)


class _MaskedTemporalBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gated_inputs = inputs * mask
        hidden = self.dropout(self.activation(self.conv1(gated_inputs))) * mask
        hidden = self.dropout(self.activation(self.conv2(hidden))) * mask
        return (hidden + self.skip(gated_inputs)) * mask


class _SensorEncoder(nn.Module):
    def __init__(self, channels: Sequence[int], dropout: float) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = 16
        for index, output_channels in enumerate(channels):
            blocks.append(
                _MaskedTemporalBlock(
                    current,
                    int(output_channels),
                    dilation=2**index,
                    dropout=dropout,
                )
            )
            current = int(output_channels)
        self.blocks = nn.ModuleList(blocks)
        self.output_channels = current

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = values.transpose(1, 2)
        channel_mask = mask.to(hidden.dtype).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden, channel_mask)
        hidden = hidden.transpose(1, 2)
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class IMUStage2Classifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        embedding_dim: int = 128,
        channels: Sequence[int] = (64, 128),
        dropout: float = 0.2,
        modality_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive derived integer")
        if not channels or any(int(channel) < 1 for channel in channels):
            raise ValueError("channels must contain positive integers")
        if not 0.0 <= float(modality_dropout) <= 1.0:
            raise ValueError("modality_dropout must be between 0 and 1")
        self.num_classes = num_classes
        self.modality_dropout = float(modality_dropout)
        self.sensor_encoders = nn.ModuleList(
            [_SensorEncoder(channels, dropout) for _ in range(5)]
        )
        encoded_channels = self.sensor_encoders[0].output_channels
        self.projection = nn.Sequential(
            nn.Linear(encoded_channels, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.null_embedding = nn.Parameter(torch.zeros(embedding_dim))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(batch["values"])
        valid_mask = torch.as_tensor(batch["valid_mask"], dtype=torch.bool, device=values.device)
        sequence_mask = torch.as_tensor(
            batch["sequence_mask"], dtype=torch.bool, device=values.device
        )
        usable_sensor_mask = torch.as_tensor(
            batch["usable_sensor_mask"], dtype=torch.bool, device=values.device
        )
        modality_mask = torch.as_tensor(
            batch.get("imu_modality_mask", torch.ones(len(values), dtype=torch.bool)),
            dtype=torch.bool,
            device=values.device,
        )
        if values.ndim != 4 or values.shape[2:] != (5, 16):
            raise ValueError("values must have shape [B,T,5,16]")
        if valid_mask.shape != values.shape[:3]:
            raise ValueError("valid_mask must have shape [B,T,5]")
        if sequence_mask.shape != values.shape[:2]:
            raise ValueError("sequence_mask must have shape [B,T]")
        if usable_sensor_mask.shape != (values.shape[0], 5):
            raise ValueError("usable_sensor_mask must have shape [B,5]")
        if modality_mask.shape != (values.shape[0],):
            raise ValueError("imu_modality_mask must have shape [B]")
        if self.training and self.modality_dropout > 0.0:
            if self.modality_dropout == 1.0:
                retained_modality = torch.zeros_like(modality_mask)
            else:
                retained_modality = torch.rand(
                    modality_mask.shape,
                    device=modality_mask.device,
                ) >= self.modality_dropout
            modality_mask = modality_mask & retained_modality
        observation_mask = valid_mask & sequence_mask.unsqueeze(-1)
        gated_values = torch.where(
            observation_mask.unsqueeze(-1), values, torch.zeros_like(values)
        )
        sensor_embeddings = []
        for sensor_index, encoder in enumerate(self.sensor_encoders):
            sensor_embeddings.append(
                encoder(
                    gated_values[:, :, sensor_index, :],
                    observation_mask[:, :, sensor_index],
                )
            )
        encoded = torch.stack(sensor_embeddings, dim=1)
        derived_usable = observation_mask.any(dim=1)
        fusion_mask = usable_sensor_mask & derived_usable
        fusion_weights = fusion_mask.to(encoded.dtype).unsqueeze(-1)
        fused = (encoded * fusion_weights).sum(dim=1) / fusion_weights.sum(dim=1).clamp_min(1.0)
        projected = self.projection(fused)
        has_available_imu = modality_mask & fusion_mask.any(dim=1)
        embedding = torch.where(
            has_available_imu.unsqueeze(-1),
            projected,
            self.null_embedding.unsqueeze(0).expand_as(projected),
        )
        logits = self.classifier(embedding)
        return {"embedding": embedding, "logits": logits}


def predict_label_indices(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError("logits must have shape [B,num_classes]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return torch.argmax(logits, dim=1)


def build_checkpoint_metadata(
    *,
    stage2_contract_sha256: str,
    training_index_sha256: str,
    normalization_contract_sha256: str,
    normalization_file_sha256: str,
    class_order_sha256: str,
    submission_contract_sha256: str,
    num_classes: int,
) -> dict[str, object]:
    bindings = {
        "stage2_contract_sha256": stage2_contract_sha256,
        "training_index_sha256": training_index_sha256,
        "normalization_contract_sha256": normalization_contract_sha256,
        "normalization_file_sha256": normalization_file_sha256,
        "class_order_sha256": class_order_sha256,
        "submission_contract_sha256": submission_contract_sha256,
    }
    normalized: dict[str, object] = {"checkpoint_metadata_version": "imu-checkpoint-v1"}
    for field in CHECKPOINT_HASH_FIELDS:
        value = bindings[field]
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field} must be a 64-character SHA-256")
        normalized[field] = value.lower()
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive derived integer")
    normalized["num_classes"] = num_classes
    return normalized
