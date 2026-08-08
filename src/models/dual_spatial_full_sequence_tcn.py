from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.models.depth_ir_pose_roi_expert import DepthIRPoseROIExpert
from src.models.full_sequence_multiscale_tcn import FullSequenceMultiScaleTCN


class DualSpatialFullSequenceTCN(nn.Module):
    """Shared Depth/IR spatial encoder plus full-frame residual temporal fusion."""

    def __init__(
        self,
        num_classes: int = 40,
        frame_feature_dim: int = 128,
        channels: int = 128,
        embedding_dim: int = 256,
        short_dilations: Sequence[int] = (1, 2, 4),
        long_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_encoder = DepthIRPoseROIExpert(
            num_classes=num_classes,
            expected_views=4,
            embedding_dim=192,
            frame_feature_dim=frame_feature_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
        # The original GRU and classifier are not part of this architecture.
        self.spatial_encoder.temporal.requires_grad_(False)
        self.spatial_encoder.classifier.requires_grad_(False)
        spatial_dim = self.spatial_encoder.local_scorer.in_features
        self.global_adapter = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, frame_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frame_feature_dim, frame_feature_dim),
        )
        self.global_gate = nn.Linear(frame_feature_dim * 2, 1)
        nn.init.zeros_(self.global_adapter[-1].weight)
        nn.init.zeros_(self.global_adapter[-1].bias)
        nn.init.zeros_(self.global_gate.weight)
        nn.init.zeros_(self.global_gate.bias)
        self.temporal_model = FullSequenceMultiScaleTCN(
            frame_feature_dim=frame_feature_dim,
            channels=channels,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            short_dilations=short_dilations,
            long_dilations=long_dilations,
            dropout=dropout,
        )

    @property
    def raw_spatial_dim(self) -> int:
        return self.spatial_encoder.local_scorer.in_features

    def encode_spatial(
        self,
        person_depth: torch.Tensor,
        person_ir: torch.Tensor,
        global_depth: torch.Tensor,
        global_ir: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if global_depth.ndim != 5 or global_ir.ndim != 5:
            raise ValueError("Global inputs must have shape [B,T,C,H,W].")
        depth = torch.cat((global_depth.unsqueeze(2), person_depth), dim=2)
        ir = torch.cat((global_ir.unsqueeze(2), person_ir), dim=2)
        view_output = self.spatial_encoder.encode_view_features(
            {"depth_input": depth, "ir_input": ir},
        )
        views = view_output["view_features"]
        global_features = views[:, :, 0]
        person_context = views[:, :, 1]
        locals_ = views[:, :, 2:]
        roi_attention = torch.softmax(
            self.spatial_encoder.local_scorer(locals_).squeeze(-1), dim=-1,
        )
        local_summary = (locals_ * roi_attention.unsqueeze(-1)).sum(dim=2)
        interaction_features = self.spatial_encoder.frame_projection(
            torch.cat((person_context, local_summary), dim=-1),
        )
        return {
            "interaction_features": interaction_features,
            "global_features": global_features,
            "roi_attention": roi_attention,
            "modality_gate": view_output["modality_gate"],
        }

    def fuse_spatial(
        self,
        interaction_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_residual = self.global_adapter(global_features)
        gate = torch.sigmoid(
            self.global_gate(torch.cat((interaction_features, global_residual), dim=-1)),
        )
        fused = interaction_features + gate * global_residual
        return fused, gate.squeeze(-1), global_residual

    def forward_cached(
        self,
        interaction_features: torch.Tensor,
        global_features: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused, spatial_gate, global_residual = self.fuse_spatial(
            interaction_features, global_features,
        )
        output = self.temporal_model(fused, temporal_mask)
        output.update({
            "spatial_gate": spatial_gate,
            "global_residual": global_residual,
            "fused_frame_features": fused,
        })
        return output

    def forward(
        self,
        person_depth: torch.Tensor,
        person_ir: torch.Tensor,
        global_depth: torch.Tensor,
        global_ir: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spatial = self.encode_spatial(person_depth, person_ir, global_depth, global_ir)
        output = self.forward_cached(
            spatial["interaction_features"], spatial["global_features"], temporal_mask,
        )
        output.update({
            "roi_attention": spatial["roi_attention"],
            "modality_gate": spatial["modality_gate"],
        })
        return output

    def freeze_spatial(self) -> None:
        self.spatial_encoder.requires_grad_(False)
        self.spatial_encoder.eval()
