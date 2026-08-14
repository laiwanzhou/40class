from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .expert_contract import ExpertOutput


_NORMALIZATION_TYPES = (
    nn.modules.batchnorm._BatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.modules.instancenorm._InstanceNorm,
)


def build_x3d_s_feature_backbone(*, pretrained: bool = True) -> nn.Module:
    """Load official PyTorchVideo X3D-S and remove its Kinetics projection."""
    from pytorchvideo.models.hub import x3d_s

    backbone = x3d_s(pretrained=pretrained)
    head = backbone.blocks[-1]
    projection = getattr(head, "proj", None)
    if not isinstance(projection, nn.Linear):
        raise TypeError("Official X3D-S head does not expose the expected linear projection")
    head.proj = None
    head.activation = None
    backbone.output_dim = projection.in_features
    return backbone


class X3DSVisualExpert(nn.Module):
    """Aggregate local X3D clips into one trial-level IR expert output."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        num_classes: int = 40,
        embedding_dim: int = 256,
        dropout: float = 0.25,
        backbone_dim: int | None = None,
        update_backbone_bn_running_stats: bool = False,
    ) -> None:
        super().__init__()
        resolved_backbone_dim = backbone_dim or getattr(backbone, "output_dim", None)
        if not isinstance(resolved_backbone_dim, int) or resolved_backbone_dim <= 0:
            raise ValueError("backbone_dim must be positive or exposed as backbone.output_dim")
        if num_classes <= 0 or embedding_dim <= 0:
            raise ValueError("num_classes and embedding_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.backbone = backbone
        self.embedding_head = nn.Sequential(
            nn.Linear(resolved_backbone_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.update_backbone_bn_running_stats = update_backbone_bn_running_stats
        self._backbone_trainable = True
        self._frozen_backbone_blocks: tuple[nn.Module, ...] = ()

    def train(self, mode: bool = True) -> X3DSVisualExpert:
        super().train(mode)
        if mode and not self._backbone_trainable:
            self.backbone.eval()
        elif mode:
            for block in self._frozen_backbone_blocks:
                block.eval()
            if not self.update_backbone_bn_running_stats:
                self._set_backbone_batch_norm_eval()
        return self

    def set_backbone_trainable(
        self,
        enabled: bool,
        *,
        last_blocks: int | None = None,
    ) -> None:
        self._backbone_trainable = enabled
        self._frozen_backbone_blocks = ()
        if not enabled or last_blocks is None:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(enabled)
        else:
            blocks = getattr(self.backbone, "blocks", None)
            if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
                raise ValueError("last_blocks requires backbone.blocks to be an ordered module list")
            if last_blocks <= 0 or last_blocks > len(blocks):
                raise ValueError(f"last_blocks must lie in [1, {len(blocks)}]")
            split_index = len(blocks) - last_blocks
            frozen_blocks = tuple(blocks[:split_index])
            trainable_blocks = tuple(blocks[split_index:])
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            for block in trainable_blocks:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            self._frozen_backbone_blocks = frozen_blocks
        self.train(self.training)

    def parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float,
    ) -> list[dict[str, object]]:
        if backbone_lr <= 0 or head_lr <= 0:
            raise ValueError("learning rates must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        normalization_parameters = {
            id(parameter)
            for module in self.modules()
            if isinstance(module, _NORMALIZATION_TYPES)
            for parameter in module.parameters(recurse=False)
        }
        backbone_parameters = {id(parameter) for parameter in self.backbone.parameters()}
        grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
        for name, parameter in self.named_parameters():
            learning_rate = backbone_lr if id(parameter) in backbone_parameters else head_lr
            decay = 0.0 if name.endswith(".bias") or id(parameter) in normalization_parameters else weight_decay
            grouped.setdefault((learning_rate, decay), []).append(parameter)

        return [
            {"params": parameters, "lr": learning_rate, "weight_decay": decay}
            for (learning_rate, decay), parameters in grouped.items()
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
        self._validate_inputs(clips, clip_mask, quality, quality_mask, availability)
        valid_counts = clip_mask.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError("Each trial must contain at least one valid clip; found zero valid clips")

        valid_clips = clips[clip_mask]
        clip_features = self._pool_backbone_output(self.backbone(valid_clips))
        clip_embeddings = self.embedding_head(clip_features)
        clip_logits = self.classifier(clip_embeddings)

        batch_size, num_clips = clip_mask.shape
        trial_indices = (
            torch.arange(batch_size, device=clips.device).unsqueeze(1).expand(batch_size, num_clips)[clip_mask]
        )
        trial_probabilities = self._mean_by_trial(
            torch.softmax(clip_logits, dim=-1), trial_indices, batch_size, valid_counts
        )
        trial_embeddings = self._mean_by_trial(clip_embeddings, trial_indices, batch_size, valid_counts)

        return ExpertOutput(
            main_logits=trial_probabilities.clamp_min(1e-8).log(),
            embedding=F.normalize(trial_embeddings, dim=-1),
            quality=quality,
            quality_mask=quality_mask,
            availability=availability,
        )

    def _set_backbone_batch_norm_eval(self) -> None:
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    @staticmethod
    def _pool_backbone_output(features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor) or features.ndim < 2:
            raise ValueError("backbone must return a tensor with batch and feature dimensions")
        if features.ndim > 2:
            features = features.mean(dim=tuple(range(2, features.ndim)))
        return features

    @staticmethod
    def _mean_by_trial(
        values: torch.Tensor,
        trial_indices: torch.Tensor,
        batch_size: int,
        valid_counts: torch.Tensor,
    ) -> torch.Tensor:
        result = values.new_zeros((batch_size, values.shape[-1]))
        result.index_add_(0, trial_indices, values)
        return result / valid_counts.to(dtype=values.dtype).unsqueeze(1)

    @staticmethod
    def _validate_inputs(
        clips: torch.Tensor,
        clip_mask: torch.Tensor,
        quality: torch.Tensor,
        quality_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> None:
        if clips.ndim != 6:
            raise ValueError("clips must have shape [B,K,C,T,H,W]")
        batch_size, num_clips, channels, frames, _, _ = clips.shape
        if channels != 3 or frames != 13:
            raise ValueError("each local X3D clip must have shape [3,13,H,W]")
        if clip_mask.shape != (batch_size, num_clips) or clip_mask.dtype != torch.bool:
            raise ValueError("clip_mask must be boolean with shape [B,K]")
        if quality.ndim != 2 or quality.shape[0] != batch_size:
            raise ValueError("quality rows must match the trial batch")
        if quality_mask.shape != quality.shape or quality_mask.dtype != torch.bool:
            raise ValueError("quality_mask must be boolean and match quality")
        if availability.ndim != 2 or availability.shape[0] != batch_size:
            raise ValueError("availability rows must match the trial batch")
