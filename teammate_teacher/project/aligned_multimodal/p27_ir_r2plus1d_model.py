from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from aligned_model import SkeletonEncoder
from p27_ir_s3d_model import KINETICS_MEAN, KINETICS_STD


class P27IRR2Plus1DSkeletonModel(nn.Module):
    """Kinetics-pretrained R(2+1)D teacher candidate plus Skeleton."""

    def __init__(
        self,
        *,
        dropout: float = 0.3,
        skeleton_input_dim: int = 10,
        kinetics_pretrained: bool = True,
    ) -> None:
        super().__init__()
        video = r2plus1d_18(
            weights=(
                R2Plus1D_18_Weights.KINETICS400_V1
                if kinetics_pretrained
                else None
            )
        )
        self.video_features = nn.Sequential(
            video.stem,
            video.layer1,
            video.layer2,
            video.layer3,
            video.layer4,
        )
        self.skeleton = SkeletonEncoder(dropout, skeleton_input_dim)
        self.video_project = nn.Sequential(
            nn.Linear(512, 384), nn.LayerNorm(384), nn.GELU()
        )
        self.skeleton_project = nn.Sequential(
            nn.Linear(512, 384), nn.LayerNorm(384), nn.GELU()
        )
        self.gate = nn.Linear(768, 384)
        self.classifier = nn.Sequential(
            nn.LayerNorm(384 * 3),
            nn.Dropout(dropout),
            nn.Linear(384 * 3, 40),
        )

    @staticmethod
    def _prepare_ir(ir: torch.Tensor) -> torch.Tensor:
        gray_mean = (0.485 + 0.456 + 0.406) / 3.0
        gray_std = (0.229 + 0.224 + 0.225) / 3.0
        unit = ir * gray_std + gray_mean
        rgb = unit.repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
        mean = rgb.new_tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
        std = rgb.new_tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
        return (rgb - mean) / std

    def encode_video(self, ir: torch.Tensor) -> torch.Tensor:
        encoded = self.video_features(self._prepare_ir(ir))
        return nn.functional.adaptive_avg_pool3d(encoded, 1).flatten(1)

    def forward(self, ir: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        video = self.video_project(self.encode_video(ir))
        body = self.skeleton_project(self.skeleton(skeleton))
        gate = torch.sigmoid(self.gate(torch.cat([video, body], dim=1)))
        fused = gate * video + (1.0 - gate) * body
        return self.classifier(torch.cat([video, body, fused], dim=1))
