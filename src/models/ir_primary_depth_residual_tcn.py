from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.models.full_sequence_multiscale_tcn import FullSequenceMultiScaleTCN


def masked_softmax(scores: torch.Tensor, valid: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    adjusted = scores + torch.log(confidence.clamp_min(1e-4))
    adjusted = adjusted.masked_fill(~valid, torch.finfo(adjusted.dtype).min)
    weights = torch.softmax(adjusted, dim=-1).masked_fill(~valid, 0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class DepthwiseBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, stride=stride, padding=1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.Hardswish(),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LightweightDepthEncoder(nn.Module):
    output_dim = 96

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            DepthwiseBlock(16, 24, 2),
            DepthwiseBlock(24, 40, 2),
            DepthwiseBlock(40, 64, 2),
            DepthwiseBlock(64, self.output_dim, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(inputs)).flatten(1)


class IRPrimaryDepthResidualTCN(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        frame_feature_dim: int = 128,
        channels: int = 128,
        embedding_dim: int = 256,
        short_dilations: Sequence[int] = (1, 2, 4),
        long_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.25,
        pretrained: bool = True,
        initial_depth_gate: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_depth_gate < 1.0:
            raise ValueError("initial_depth_gate must be between zero and one")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        ir_backbone = mobilenet_v3_small(weights=weights)
        rgb_conv = ir_backbone.features[0][0]
        if not isinstance(rgb_conv, nn.Conv2d):
            raise TypeError("Unexpected MobileNetV3 stem")
        ir_conv = nn.Conv2d(
            1, rgb_conv.out_channels, rgb_conv.kernel_size, rgb_conv.stride, rgb_conv.padding,
            rgb_conv.dilation, rgb_conv.groups, rgb_conv.bias is not None, rgb_conv.padding_mode,
        )
        with torch.no_grad():
            ir_conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
            if rgb_conv.bias is not None:
                assert ir_conv.bias is not None
                ir_conv.bias.copy_(rgb_conv.bias)
        ir_backbone.features[0][0] = ir_conv
        self.ir_encoder = ir_backbone.features
        self.ir_pool = ir_backbone.avgpool
        ir_dim = ir_backbone.classifier[0].in_features
        self.ir_local_scorer = nn.Linear(ir_dim, 1)
        self.ir_projection = nn.Sequential(
            nn.Linear(ir_dim * 2, frame_feature_dim), nn.GELU(), nn.Dropout(dropout),
        )

        self.depth_encoder = LightweightDepthEncoder()
        depth_dim = self.depth_encoder.output_dim
        self.depth_relation_gate = nn.Linear(depth_dim * 2, 1)
        self.depth_projection = nn.Sequential(
            nn.Linear(depth_dim * 2, frame_feature_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.modality_gate = nn.Linear(frame_feature_dim * 2, 1)
        nn.init.zeros_(self.modality_gate.weight)
        nn.init.constant_(self.modality_gate.bias, math.log(initial_depth_gate / (1.0 - initial_depth_gate)))
        self.fusion_norm = nn.LayerNorm(frame_feature_dim)

        self.temporal_model = FullSequenceMultiScaleTCN(
            frame_feature_dim=frame_feature_dim,
            channels=channels,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            short_dilations=short_dilations,
            long_dilations=long_dilations,
            dropout=dropout,
        )
        self.route_head = nn.Linear(embedding_dim, 2)

    def encode_spatial(
        self,
        ir: torch.Tensor,
        depth: torch.Tensor,
        ir_valid: torch.Tensor,
        depth_valid: torch.Tensor,
        view_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if ir.ndim != 6 or ir.shape[2:4] != (4, 1):
            raise ValueError(f"Expected IR [B,T,4,1,H,W], got {tuple(ir.shape)}")
        if depth.ndim != 6 or depth.shape[2:4] != (2, 3) or depth.shape[:2] != ir.shape[:2]:
            raise ValueError(f"Expected Depth [B,T,2,3,H,W], got {tuple(depth.shape)}")
        if ir_valid.shape != ir.shape[:3] or depth_valid.shape != depth.shape[:3]:
            raise ValueError("View masks do not align with inputs")
        if view_confidence.shape != ir.shape[:3]:
            raise ValueError("View confidence does not align with IR views")
        batch, frames, _, _, height, width = ir.shape

        ir_flat = ir.reshape(batch * frames * 4, 1, height, width)
        ir_features = self.ir_pool(self.ir_encoder(ir_flat)).flatten(1).reshape(batch, frames, 4, -1)
        ir_context = ir_features[:, :, 0]
        local_features = ir_features[:, :, 1:]
        local_valid = ir_valid[:, :, 1:]
        local_confidence = view_confidence[:, :, 1:]
        local_scores = self.ir_local_scorer(local_features).squeeze(-1)
        ir_attention = masked_softmax(local_scores, local_valid, local_confidence)
        local_summary = (local_features * ir_attention.unsqueeze(-1)).sum(dim=2)
        ir_frame = self.ir_projection(torch.cat((ir_context, local_summary), dim=-1))

        depth_flat = depth.reshape(batch * frames * 2, 3, height, width)
        depth_features = self.depth_encoder(depth_flat).reshape(batch, frames, 2, -1)
        depth_context = depth_features[:, :, 0]
        depth_relation = depth_features[:, :, 1]
        relation_confidence = view_confidence[:, :, 3] * depth_valid[:, :, 1].to(view_confidence.dtype)
        relation_gate = torch.sigmoid(
            self.depth_relation_gate(torch.cat((depth_context, depth_relation), dim=-1)),
        ).squeeze(-1) * relation_confidence
        depth_frame = self.depth_projection(
            torch.cat((depth_context, depth_relation * relation_gate.unsqueeze(-1)), dim=-1),
        )

        depth_gate = torch.sigmoid(self.modality_gate(torch.cat((ir_frame, depth_frame), dim=-1))).squeeze(-1)
        fused = self.fusion_norm(ir_frame + depth_gate.unsqueeze(-1) * depth_frame)
        return {
            "frame_features": fused,
            "ir_roi_attention": ir_attention,
            "depth_relation_gate": relation_gate,
            "depth_gate": depth_gate,
            "ir_frame_features": ir_frame,
            "depth_frame_features": depth_frame,
        }

    def forward_cached(self, frame_features: torch.Tensor, temporal_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.temporal_model(frame_features, temporal_mask)
        output["route_logits"] = self.route_head(output["embedding"])
        return output

    def forward(
        self,
        ir: torch.Tensor,
        depth: torch.Tensor,
        ir_valid: torch.Tensor,
        depth_valid: torch.Tensor,
        view_confidence: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spatial = self.encode_spatial(ir, depth, ir_valid, depth_valid, view_confidence)
        output = self.forward_cached(spatial["frame_features"], temporal_mask)
        output.update(spatial)
        return output

    def freeze_spatial(self) -> None:
        for module in (
            self.ir_encoder, self.ir_pool, self.ir_local_scorer, self.ir_projection,
            self.depth_encoder, self.depth_relation_gate, self.depth_projection,
            self.modality_gate, self.fusion_norm,
        ):
            module.requires_grad_(False)
            module.eval()
