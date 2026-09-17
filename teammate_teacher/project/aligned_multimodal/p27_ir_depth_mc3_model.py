from __future__ import annotations

import torch
from torch import nn

from p27_ir_mc3_model import P27IRMC3SkeletonModel
from p27_ir_s3d_model import KINETICS_MEAN, KINETICS_STD


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class P27IRDepthMC3ResidualModel(nn.Module):
    """Depth residual through the same frozen MC3 trunk as the safe IR base."""

    def __init__(
        self,
        *,
        dropout: float = 0.3,
        skeleton_input_dim: int = 10,
    ) -> None:
        super().__init__()
        self.base = P27IRMC3SkeletonModel(
            dropout=dropout,
            skeleton_input_dim=skeleton_input_dim,
            kinetics_pretrained=False,
        )
        self.depth_project = nn.Sequential(
            nn.Linear(512, 384), nn.LayerNorm(384), nn.GELU()
        )
        self.depth_residual = nn.Sequential(
            nn.Linear(384 * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        # No bias: a correction must depend on the current Depth representation.
        self.delta_classifier = nn.Linear(128, 40, bias=False)
        nn.init.zeros_(self.delta_classifier.weight)

    @staticmethod
    def _prepare_depth(depth: torch.Tensor) -> torch.Tensor:
        mean = depth.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = depth.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        unit = depth * std + mean
        rgb = unit.permute(0, 2, 1, 3, 4)
        kinetics_mean = rgb.new_tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
        kinetics_std = rgb.new_tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
        return (rgb - kinetics_mean) / kinetics_std

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        encoded = self.base.video_features(self._prepare_depth(depth))
        return nn.functional.adaptive_avg_pool3d(encoded, 1).flatten(1)

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def forward(
        self,
        ir: torch.Tensor,
        depth: torch.Tensor,
        skeleton: torch.Tensor,
        *,
        ablation: str | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base_logits, video, body = self.base.forward_with_features(ir, skeleton)
            depth_encoded = self.encode_depth(depth)
        depth_feature = self.depth_project(depth_encoded)
        if ablation == "depth_zero":
            depth_feature = torch.zeros_like(depth_feature)
        elif ablation == "depth_shuffle":
            depth_feature = depth_feature.roll(1, dims=0)
        elif ablation is not None:
            raise ValueError(f"unknown depth ablation: {ablation}")
        residual = self.depth_residual(
            torch.cat(
                [
                    depth_feature,
                    depth_feature - video,
                    depth_feature * video,
                    depth_feature * body,
                ],
                dim=1,
            )
        )
        delta_logits = self.delta_classifier(residual)
        return {
            "logits": base_logits + delta_logits,
            "base_logits": base_logits,
            "delta_logits": delta_logits,
            "depth_feature": depth_feature,
        }
