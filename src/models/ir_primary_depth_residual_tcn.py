from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.data.ir_primary_full_sequence_dataset import QUALITY_NAMES
from src.models.expert_contract import ExpertOutput, VisualExpertForward
from src.models.full_sequence_multiscale_tcn import FullSequenceMultiScaleTCN


@contextmanager
def _restore_running_stats_after_recompute(module: nn.Module):
    batch_norms = [child for child in module.modules() if isinstance(child, nn.modules.batchnorm._BatchNorm)]
    states = [
        (
            None if child.running_mean is None else child.running_mean.clone(),
            None if child.running_var is None else child.running_var.clone(),
            None if child.num_batches_tracked is None else child.num_batches_tracked.clone(),
        )
        for child in batch_norms
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for child, (mean, variance, batches) in zip(batch_norms, states, strict=True):
                if mean is not None:
                    child.running_mean.copy_(mean)
                if variance is not None:
                    child.running_var.copy_(variance)
                if batches is not None:
                    child.num_batches_tracked.copy_(batches)


def _checkpoint_contexts(module: nn.Module):
    return nullcontext(), _restore_running_stats_after_recompute(module)


def masked_softmax(scores: torch.Tensor, valid: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
    adjusted = scores + torch.log(reliability.clamp_min(1e-4))
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

    def forward(self, inputs: torch.Tensor, pixel_valid: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        mask = nn.functional.interpolate(pixel_valid.to(features.dtype), features.shape[-2:], mode="area")
        numerator = (features * mask).sum(dim=(-2, -1))
        denominator = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
        return numerator / denominator


def _encode_valid(
    inputs: torch.Tensor,
    valid: torch.Tensor,
    encoder: nn.Module,
    output_dim: int,
    pixel_valid: torch.Tensor | None = None,
    chunk_size: int = 32,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    flat_inputs = inputs.flatten(0, 2)
    flat_valid = valid.flatten()
    output = inputs.new_zeros((len(flat_inputs), output_dim))
    if flat_valid.any():
        selected = flat_inputs[flat_valid]
        selected_mask = None if pixel_valid is None else pixel_valid.flatten(0, 2)[flat_valid]
        pieces: list[torch.Tensor] = []
        for start in range(0, len(selected), chunk_size):
            images = selected[start : start + chunk_size]
            masks = None if selected_mask is None else selected_mask[start : start + chunk_size]
            if activation_checkpointing and encoder.training:
                encoded = (
                    checkpoint(
                        encoder, images, use_reentrant=False,
                        context_fn=lambda: _checkpoint_contexts(encoder),
                    )
                    if masks is None
                    else checkpoint(
                        encoder, images, masks, use_reentrant=False,
                        context_fn=lambda: _checkpoint_contexts(encoder),
                    )
                )
            else:
                encoded = encoder(images) if masks is None else encoder(images, masks)
            pieces.append(encoded)
        encoded = torch.cat(pieces)
        output[flat_valid] = encoded
    return output.reshape(*inputs.shape[:3], output_dim)


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
        activation_checkpointing: bool = False,
        spatial_view_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_depth_gate < 1.0:
            raise ValueError("initial_depth_gate must be between zero and one")
        if spatial_view_chunk_size <= 0:
            raise ValueError("spatial_view_chunk_size must be positive")
        self.spatial_view_chunk_size = int(spatial_view_chunk_size)
        self.activation_checkpointing = bool(activation_checkpointing)
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
        self.ir_encoder = nn.Sequential(ir_backbone.features, ir_backbone.avgpool, nn.Flatten(1))
        ir_dim = ir_backbone.classifier[0].in_features
        self.ir_dim = ir_dim
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
        self.reliability_projection = nn.Sequential(
            nn.Linear(frame_feature_dim + 6, frame_feature_dim), nn.GELU(), nn.Dropout(dropout),
        )

        self.temporal_model = FullSequenceMultiScaleTCN(
            frame_feature_dim=frame_feature_dim,
            channels=channels,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            short_dilations=short_dilations,
            long_dilations=long_dilations,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
        )
        self.small_gate_head = nn.Linear(embedding_dim, 2)

    def encode_spatial(
        self,
        ir: torch.Tensor,
        depth: torch.Tensor,
        depth_pixel_valid: torch.Tensor,
        view_valid: torch.Tensor,
        view_reliability: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if ir.ndim != 6 or ir.shape[2:4] != (4, 1):
            raise ValueError(f"Expected IR [B,T,4,1,H,W], got {tuple(ir.shape)}")
        if depth.ndim != 6 or depth.shape[2:4] != (2, 3) or depth.shape[:2] != ir.shape[:2]:
            raise ValueError(f"Expected Depth [B,T,2,3,H,W], got {tuple(depth.shape)}")
        if depth_pixel_valid.shape != (*depth.shape[:3], 1, *depth.shape[-2:]):
            raise ValueError("Depth pixel mask does not align with inputs")
        if view_valid.shape != (*ir.shape[:2], 6) or view_reliability.shape != view_valid.shape:
            raise ValueError("Six-view masks/reliability do not align with inputs")
        if temporal_mask.shape != ir.shape[:2]:
            raise ValueError("Temporal mask does not align with inputs")

        temporal_views = temporal_mask.unsqueeze(-1)
        ir_valid = view_valid[:, :, :4] & temporal_views
        depth_valid = view_valid[:, :, 4:] & temporal_views
        ir_features = _encode_valid(
            ir, ir_valid, self.ir_encoder, self.ir_dim,
            chunk_size=self.spatial_view_chunk_size,
            activation_checkpointing=self.activation_checkpointing,
        )
        ir_context = ir_features[:, :, 0]
        local_features = ir_features[:, :, 1:]
        local_valid = ir_valid[:, :, 1:]
        local_reliability = view_reliability[:, :, 1:4]
        local_scores = self.ir_local_scorer(local_features).squeeze(-1)
        ir_attention = masked_softmax(local_scores, local_valid, local_reliability)
        local_summary = (local_features * ir_attention.unsqueeze(-1)).sum(dim=2)
        ir_frame = self.ir_projection(torch.cat((ir_context, local_summary), dim=-1))
        ir_frame = ir_frame * ir_valid[:, :, 0].unsqueeze(-1)

        depth_features = _encode_valid(
            depth,
            depth_valid,
            self.depth_encoder,
            self.depth_encoder.output_dim,
            depth_pixel_valid,
            chunk_size=self.spatial_view_chunk_size,
            activation_checkpointing=self.activation_checkpointing,
        )
        depth_context = depth_features[:, :, 0]
        depth_relation = depth_features[:, :, 1]
        relation_reliability = view_reliability[:, :, 5] * depth_valid[:, :, 1]
        relation_gate = torch.sigmoid(
            self.depth_relation_gate(torch.cat((depth_context, depth_relation), dim=-1)),
        ).squeeze(-1) * relation_reliability
        depth_frame = self.depth_projection(
            torch.cat((depth_context, depth_relation * relation_gate.unsqueeze(-1)), dim=-1),
        ) * depth_valid[:, :, 0].unsqueeze(-1)

        depth_availability = view_reliability[:, :, 4:].amax(dim=-1)
        depth_gate = torch.sigmoid(
            self.modality_gate(torch.cat((ir_frame, depth_frame), dim=-1)),
        ).squeeze(-1) * depth_availability
        fused = self.fusion_norm(ir_frame + depth_gate.unsqueeze(-1) * depth_frame)
        fused = self.reliability_projection(torch.cat((fused, view_reliability), dim=-1))
        fused = fused * temporal_mask.unsqueeze(-1)
        return {
            "frame_features": fused,
            "ir_roi_attention": ir_attention,
            "depth_relation_gate": relation_gate,
            "depth_gate": depth_gate,
        }

    def forward(
        self,
        ir: torch.Tensor,
        depth: torch.Tensor,
        depth_pixel_valid: torch.Tensor,
        view_valid: torch.Tensor,
        view_reliability: torch.Tensor,
        temporal_mask: torch.Tensor,
        quality: torch.Tensor,
        quality_mask: torch.Tensor,
    ) -> VisualExpertForward:
        if quality.shape != (len(ir), len(QUALITY_NAMES)) or quality_mask.shape != quality.shape:
            raise ValueError("Quality tensors do not match the fixed visual quality schema")
        spatial = self.encode_spatial(
            ir, depth, depth_pixel_valid, view_valid, view_reliability, temporal_mask,
        )
        temporal = self.temporal_model(spatial["frame_features"], temporal_mask)
        expert = ExpertOutput(
            main_logits=temporal["logits"],
            embedding=temporal["embedding"],
            quality=quality,
            quality_mask=quality_mask,
            availability=temporal_mask.any(dim=1, keepdim=True),
        )
        return VisualExpertForward(
            expert=expert,
            small_gate_logits=self.small_gate_head(temporal["embedding"]),
            sequence_features=spatial["frame_features"],
            short_attention=temporal["short_attention"],
            long_attention=temporal["long_attention"],
            ir_roi_attention=spatial["ir_roi_attention"],
            depth_relation_gate=spatial["depth_relation_gate"],
            depth_gate=spatial["depth_gate"],
        )
