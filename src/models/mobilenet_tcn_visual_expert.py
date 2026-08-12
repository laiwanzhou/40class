from __future__ import annotations

import hashlib
import io

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .expert_contract import ExpertOutput
from .x3d_s_visual_expert import _NORMALIZATION_TYPES


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class MobileNetFrameBackbone(nn.Module):
    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.output_dim = int(model.classifier[0].in_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.avgpool(self.features(images)).flatten(1)


class MobileNetTCNVisualExpert(nn.Module):
    """Matched 2D-frame encoder plus TCN baseline over the same local clips."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        num_classes: int = 40,
        tcn_channels: int = 256,
        embedding_dim: int = 256,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.2,
        update_backbone_bn_running_stats: bool = False,
    ) -> None:
        super().__init__()
        backbone_dim = getattr(backbone, "output_dim", None)
        if not isinstance(backbone_dim, int) or backbone_dim <= 0:
            raise ValueError("backbone.output_dim must be a positive integer")
        if tcn_channels % 16:
            raise ValueError("tcn_channels must be divisible by 16")
        self.backbone = backbone
        self.frame_projection = nn.Sequential(
            nn.Linear(backbone_dim, tcn_channels), nn.GELU(), nn.Dropout(dropout)
        )
        self.tcn_blocks = nn.ModuleList(
            DilatedResidualBlock(tcn_channels, dilation, dropout) for dilation in dilations
        )
        self.temporal_attention = nn.Conv1d(tcn_channels, 1, 1)
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(tcn_channels * 2),
            nn.Linear(tcn_channels * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.update_backbone_bn_running_stats = update_backbone_bn_running_stats
        self._backbone_trainable = True
        self.pretrained_source = MobileNet_V3_Small_Weights.IMAGENET1K_V1.url
        self.pretraining_dataset = "imagenet_1k"
        self.source_revision = "torchvision==0.22.0"
        self.license = "BSD-3-Clause"
        source_buffer = io.BytesIO()
        torch.save(self.backbone.state_dict(), source_buffer)
        source_state = source_buffer.getvalue()
        self.source_weight_bytes = len(source_state)
        self.source_weight_sha256 = hashlib.sha256(source_state).hexdigest()

    def train(self, mode: bool = True) -> MobileNetTCNVisualExpert:
        super().train(mode)
        if mode and (not self._backbone_trainable or not self.update_backbone_bn_running_stats):
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def set_backbone_trainable(self, enabled: bool) -> None:
        self._backbone_trainable = enabled
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(enabled)
        self.train(self.training)

    def parameter_groups(
        self, backbone_lr: float, head_lr: float, weight_decay: float
    ) -> list[dict[str, object]]:
        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        norm_ids = {
            id(parameter)
            for module in self.modules()
            if isinstance(module, _NORMALIZATION_TYPES)
            for parameter in module.parameters(recurse=False)
        }
        grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
        for name, parameter in self.named_parameters():
            lr = backbone_lr if id(parameter) in backbone_ids else head_lr
            decay = 0.0 if name.endswith(".bias") or id(parameter) in norm_ids else weight_decay
            grouped.setdefault((lr, decay), []).append(parameter)
        return [
            {"params": parameters, "lr": lr, "weight_decay": decay}
            for (lr, decay), parameters in grouped.items()
        ]

    def forward(
        self,
        clips: torch.Tensor,
        *,
        clip_mask: torch.Tensor,
        quality: torch.Tensor,
        quality_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> ExpertOutput:
        if clips.ndim != 6 or clips.shape[2:4] != (3, 13):
            raise ValueError("clips must have shape [B,K,3,13,H,W]")
        if clip_mask.shape != clips.shape[:2] or clip_mask.dtype != torch.bool:
            raise ValueError("clip_mask must be boolean [B,K]")
        valid_counts = clip_mask.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError("Each trial requires at least one valid clip")
        valid = clips[clip_mask]
        clip_count, _, frames, height, width = valid.shape
        images = valid.permute(0, 2, 1, 3, 4).reshape(clip_count * frames, 3, height, width)
        frame_features = self.backbone(images).reshape(clip_count, frames, -1)
        temporal = self.frame_projection(frame_features).transpose(1, 2)
        for block in self.tcn_blocks:
            temporal = block(temporal)
        weights = torch.softmax(self.temporal_attention(temporal).squeeze(1), dim=1)
        attended = (temporal * weights.unsqueeze(1)).sum(dim=2)
        mean = temporal.mean(dim=2)
        clip_embeddings = self.embedding_head(torch.cat((attended, mean), dim=1))
        clip_logits = self.classifier(clip_embeddings)

        batch_size, max_clips = clip_mask.shape
        trial_indices = (
            torch.arange(batch_size, device=clips.device)
            .unsqueeze(1)
            .expand(batch_size, max_clips)[clip_mask]
        )
        probabilities = self._mean_by_trial(
            torch.softmax(clip_logits, dim=-1), trial_indices, batch_size, valid_counts
        )
        embeddings = self._mean_by_trial(
            clip_embeddings, trial_indices, batch_size, valid_counts
        )
        return ExpertOutput(
            main_logits=probabilities.clamp_min(1e-8).log(),
            embedding=F.normalize(embeddings, dim=-1),
            quality=quality,
            quality_mask=quality_mask,
            availability=availability,
        )

    @staticmethod
    def _mean_by_trial(
        values: torch.Tensor,
        trial_indices: torch.Tensor,
        batch_size: int,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        result = values.new_zeros((batch_size, values.shape[-1]))
        result.index_add_(0, trial_indices, values)
        return result / counts.to(values.dtype).unsqueeze(1)

    def non_backbone_state_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {key: value for key, value in self.state_dict().items() if not key.startswith("backbone.")},
            buffer,
        )
        return buffer.getvalue()

def build_mobilenet_tcn_visual_expert(
    *, pretrained: bool = True,
    num_classes: int = 40,
    tcn_channels: int = 256,
    embedding_dim: int = 256,
    dropout: float = 0.2,
    update_backbone_bn_running_stats: bool = False,
) -> MobileNetTCNVisualExpert:
    return MobileNetTCNVisualExpert(
        backbone=MobileNetFrameBackbone(pretrained=pretrained),
        num_classes=num_classes,
        tcn_channels=tcn_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
        update_backbone_bn_running_stats=update_backbone_bn_running_stats,
    )
