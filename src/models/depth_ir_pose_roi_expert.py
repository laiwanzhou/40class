from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class DepthIRPoseROIExpert(nn.Module):
    def __init__(
        self,
        num_classes: int,
        expected_views: int = 4,
        embedding_dim: int = 192,
        frame_feature_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.depth_stem = backbone.features[0]
        self.ir_stem = copy.deepcopy(self.depth_stem)
        rgb_conv = self.depth_stem[0]
        if not isinstance(rgb_conv, nn.Conv2d):
            raise TypeError("Unexpected MobileNetV3 stem layout.")
        ir_conv = nn.Conv2d(
            1,
            rgb_conv.out_channels,
            rgb_conv.kernel_size,
            rgb_conv.stride,
            rgb_conv.padding,
            rgb_conv.dilation,
            rgb_conv.groups,
            rgb_conv.bias is not None,
            rgb_conv.padding_mode,
        )
        with torch.no_grad():
            ir_conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
            if rgb_conv.bias is not None:
                assert ir_conv.bias is not None
                ir_conv.bias.copy_(rgb_conv.bias)
        self.ir_stem[0] = ir_conv
        stem_channels = rgb_conv.out_channels
        self.modality_gate = nn.Conv2d(stem_channels * 2, stem_channels, kernel_size=1)
        nn.init.zeros_(self.modality_gate.weight)
        nn.init.zeros_(self.modality_gate.bias)
        self.shared_body = nn.Sequential(*list(backbone.features.children())[1:])
        self.avgpool = backbone.avgpool
        feature_dim = backbone.classifier[0].in_features
        self.expected_views = expected_views
        self.local_scorer = nn.Linear(feature_dim, 1)
        self.frame_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, frame_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(frame_feature_dim, embedding_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode_frames(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        depth = inputs["depth_input"]
        ir = inputs["ir_input"]
        if depth.ndim != 6 or depth.shape[2] != self.expected_views or depth.shape[3] != 3:
            raise ValueError(f"Expected Depth [B,T,{self.expected_views},3,H,W], got {tuple(depth.shape)}")
        if ir.ndim != 6 or ir.shape[:3] != depth.shape[:3] or ir.shape[3] != 1 or ir.shape[-2:] != depth.shape[-2:]:
            raise ValueError(f"Expected aligned IR [B,T,{self.expected_views},1,H,W], got {tuple(ir.shape)}")
        batch, frames, views, _, height, width = depth.shape
        flattened = batch * frames * views
        depth_features = self.depth_stem(depth.reshape(flattened, 3, height, width))
        ir_features = self.ir_stem(ir.reshape(flattened, 1, height, width))
        gate = torch.sigmoid(self.modality_gate(torch.cat((depth_features, ir_features), dim=1)))
        fused = gate * depth_features + (1.0 - gate) * ir_features
        features = self.avgpool(self.shared_body(fused)).flatten(1)
        features = features.reshape(batch, frames, views, -1)
        gate_summary = gate.mean(dim=(1, 2, 3)).reshape(batch, frames, views)
        global_features = features[:, :, 0]
        locals_ = features[:, :, 1:]
        roi_attention = torch.softmax(self.local_scorer(locals_).squeeze(-1), dim=-1)
        local_summary = (locals_ * roi_attention.unsqueeze(-1)).sum(dim=2)
        frame_features = self.frame_projection(torch.cat((global_features, local_summary), dim=-1))
        return {
            "frame_features": frame_features,
            "roi_attention": roi_attention,
            "modality_gate": gate_summary,
        }

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        temporal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        frame_output = self.encode_frames(inputs)
        frame_features = frame_output["frame_features"]
        if temporal_mask is None:
            _, hidden = self.temporal(frame_features)
        else:
            lengths = temporal_mask.sum(dim=1).clamp_min(1).to(torch.int64).cpu()
            packed = pack_padded_sequence(frame_features, lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.temporal(packed)
        embedding = hidden[-1]
        return {
            "embedding": embedding,
            "logits": self.classifier(self.dropout(embedding)),
            "roi_attention": frame_output["roi_attention"],
            "modality_gate": frame_output["modality_gate"],
        }
