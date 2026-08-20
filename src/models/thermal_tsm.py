from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class TemporalShift(nn.Module):
    """Parameter-free TSM on explicit [batch, time, channel, height, width]."""

    def __init__(self, num_segments: int = 16, fold_div: int = 8) -> None:
        super().__init__()
        if num_segments < 1 or fold_div < 1:
            raise ValueError("num_segments and fold_div must be positive")
        self.num_segments = num_segments
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("TemporalShift expects [B,T,C,H,W]")
        if x.shape[1] != self.num_segments:
            raise ValueError(
                f"Expected num_segments={self.num_segments}, received T={x.shape[1]}"
            )
        fold = x.shape[2] // self.fold_div
        if fold == 0:
            return x

        shifted = torch.zeros_like(x)
        shifted[:, :-1, :fold] = x[:, 1:, :fold]
        shifted[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        shifted[:, :, 2 * fold :] = x[:, :, 2 * fold :]
        return shifted


class MobileNetV3SmallTSM(nn.Module):
    """MobileNetV3-Small with TSM before selected spatial feature blocks."""

    def __init__(
        self,
        *,
        weights: MobileNet_V3_Small_Weights | None,
        num_classes: int = 40,
        num_segments: int = 16,
        fold_div: int = 8,
        shift_before_blocks: Iterable[int] = (1, 3, 6, 9),
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 40:
            raise ValueError("Thermal expert head must have exactly 40 classes")
        base = backbone if backbone is not None else mobilenet_v3_small(weights=weights)
        if not hasattr(base, "features") or not hasattr(base, "classifier"):
            raise TypeError("Expected a torchvision MobileNetV3 backbone")

        in_features = int(base.classifier[-1].in_features)
        base.classifier[-1] = nn.Linear(in_features, num_classes)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.shift_before_blocks = frozenset(int(index) for index in shift_before_blocks)
        invalid = self.shift_before_blocks.difference(range(len(self.features)))
        if invalid:
            raise ValueError(f"Invalid MobileNet feature block indices: {sorted(invalid)}")
        self.temporal_shift = TemporalShift(num_segments, fold_div)

    def _shift_flat_features(
        self, features: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        shape = features.shape
        explicit = features.reshape(
            batch_size, self.num_segments, shape[1], shape[2], shape[3]
        )
        return self.temporal_shift(explicit).reshape(shape)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[1:] != (
            self.num_segments,
            3,
            224,
            224,
        ):
            raise ValueError(
                "MobileNetV3SmallTSM expects [B,16,3,224,224] "
                f"for this contract, received {tuple(clips.shape)}"
            )
        batch_size = clips.shape[0]
        x = clips.reshape(batch_size * self.num_segments, 3, 224, 224)
        for index, block in enumerate(self.features):
            if index in self.shift_before_blocks:
                x = self._shift_flat_features(x, batch_size)
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.reshape(batch_size, self.num_segments, -1).mean(dim=1)
        logits = self.classifier(x)
        if logits.shape != (batch_size, self.num_classes):
            raise RuntimeError(f"Unexpected classifier output shape: {tuple(logits.shape)}")
        return logits


class IFormerTSM(nn.Module):
    """Mobile iFormer with TSM at native stage boundaries."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        num_classes: int = 40,
        num_segments: int = 16,
        fold_div: int = 8,
        shift_before_stages: Iterable[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        if num_classes != 40:
            raise ValueError("Thermal expert head must have exactly 40 classes")
        if not getattr(backbone, "use_bn", False):
            raise TypeError("Expected the official mobile iFormer use_bn=True backbone")
        if not hasattr(backbone, "downsample_layers") or not hasattr(
            backbone, "stages"
        ):
            raise TypeError("Expected official iFormer downsample_layers and stages")
        if len(backbone.downsample_layers) != 4 or len(backbone.stages) != 4:
            raise ValueError("Thermal iFormer contract requires four native stages")

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.shift_before_stages = frozenset(
            int(index) for index in shift_before_stages
        )
        invalid = self.shift_before_stages.difference(range(4))
        if invalid:
            raise ValueError(f"Invalid iFormer stage indices: {sorted(invalid)}")
        self.temporal_shift = TemporalShift(num_segments, fold_div)

    def _shift_flat_features(
        self, features: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        shape = features.shape
        explicit = features.reshape(
            batch_size, self.num_segments, shape[1], shape[2], shape[3]
        )
        return self.temporal_shift(explicit).reshape(shape)

    def forward_frame_features(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[1:] != (
            self.num_segments,
            3,
            224,
            224,
        ):
            raise ValueError(
                "IFormerTSM expects [B,16,3,224,224] for this contract, "
                f"received {tuple(clips.shape)}"
            )
        batch_size = clips.shape[0]
        x = clips.reshape(batch_size * self.num_segments, 3, 224, 224)
        for index in range(4):
            if isinstance(x, tuple):
                features, auxiliary = x
                features = self.backbone.downsample_layers[index](features)
                if index in self.shift_before_stages:
                    features = self._shift_flat_features(features, batch_size)
                x = (features, auxiliary)
            else:
                x = self.backbone.downsample_layers[index](x)
                if index in self.shift_before_stages:
                    x = self._shift_flat_features(x, batch_size)
            x = self.backbone.stages[index](x)
        if isinstance(x, tuple):
            x = x[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.backbone.last_proj:
            x = self.backbone.act(self.backbone.proj(x))
        return x.reshape(batch_size, self.num_segments, -1)

    def forward_features(self, clips: torch.Tensor) -> torch.Tensor:
        return self.forward_frame_features(clips).mean(dim=1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        logits = self.backbone.classifier(self.forward_features(clips))
        if isinstance(logits, tuple):
            raise RuntimeError("Distillation output is outside the Thermal contract")
        if logits.shape != (clips.shape[0], self.num_classes):
            raise RuntimeError(f"Unexpected classifier output shape: {tuple(logits.shape)}")
        return logits
