from __future__ import annotations

import copy

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class ObjectInteractionTCNExpert(nn.Module):
    def __init__(
        self,
        target_class_ids: list[int],
        expected_views: int = 5,
        frame_feature_dim: int = 256,
        tcn_channels: int = 256,
        embedding_dim: int = 256,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.depth_stem = backbone.features[0]
        self.ir_stem = copy.deepcopy(self.depth_stem)
        rgb_conv = self.depth_stem[0]
        if not isinstance(rgb_conv, nn.Conv2d):
            raise TypeError("Unexpected MobileNetV3 stem")
        self.ir_stem[0] = nn.Conv2d(
            1, rgb_conv.out_channels, rgb_conv.kernel_size, rgb_conv.stride, rgb_conv.padding,
            rgb_conv.dilation, rgb_conv.groups, rgb_conv.bias is not None, rgb_conv.padding_mode,
        )
        self.modality_gate = nn.Conv2d(rgb_conv.out_channels * 2, rgb_conv.out_channels, 1)
        self.shared_body = nn.Sequential(*list(backbone.features.children())[1:])
        self.avgpool = backbone.avgpool
        feature_dim = backbone.classifier[0].in_features
        self.expected_views = expected_views
        self.target_class_ids = tuple(int(value) for value in target_class_ids)
        self.register_buffer("target_index", torch.tensor(self.target_class_ids, dtype=torch.long), persistent=True)
        self.view_gate = nn.Sequential(nn.Linear(feature_dim, 96), nn.GELU(), nn.Linear(96, 1))
        self.frame_projection = nn.Sequential(
            nn.Linear(feature_dim, frame_feature_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(frame_feature_dim, tcn_channels),
        )
        self.tcn_blocks = nn.ModuleList(
            DilatedResidualBlock(tcn_channels, kernel_size, dilation, dropout) for dilation in dilations
        )
        self.skip_projection = nn.Conv1d(tcn_channels, tcn_channels, 1)
        self.attention_pool = nn.Conv1d(tcn_channels, 1, 1)
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(tcn_channels * 2),
            nn.Linear(tcn_channels * 2, embedding_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(embedding_dim, len(target_class_ids))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.dilations = tuple(dilations)
        self.kernel_size = kernel_size
        self.training_stage = "warmup"

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def initialize_encoder(self, base_state: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        prefixes = ("depth_stem.", "ir_stem.", "modality_gate.", "shared_body.")
        copied = 0
        for name, value in base_state.items():
            if name.startswith(prefixes) and name in own and own[name].shape == value.shape:
                own[name].copy_(value)
                copied += 1
        if copied < 100:
            raise ValueError(f"Only copied {copied} base encoder tensors")

    def set_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
        for module in (self.depth_stem, self.ir_stem, self.shared_body, self.modality_gate):
            for parameter in module.parameters():
                parameter.requires_grad = False
        if stage == "warmup":
            self.training_stage = stage
            return
        if stage != "finetune":
            raise ValueError(f"Unknown training stage {stage}")
        for parameter in self.modality_gate.parameters():
            parameter.requires_grad = True
        start = max(0, len(self.shared_body) - 4)
        for layer in list(self.shared_body.children())[start:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.training_stage = stage

    def enforce_frozen_encoder_eval(self) -> None:
        self.depth_stem.eval()
        self.ir_stem.eval()
        if self.training_stage == "warmup":
            self.shared_body.eval()
            self.modality_gate.eval()
            return
        start = max(0, len(self.shared_body) - 4)
        for layer in list(self.shared_body.children())[:start]:
            layer.eval()

    def parameter_groups(self, backbone_lr: float, modules_lr: float) -> list[dict[str, object]]:
        backbone_ids = {
            id(parameter) for parameter in self.shared_body.parameters() if parameter.requires_grad
        } | {id(parameter) for parameter in self.modality_gate.parameters() if parameter.requires_grad}
        backbone = [parameter for parameter in self.parameters() if parameter.requires_grad and id(parameter) in backbone_ids]
        new_modules = [parameter for parameter in self.parameters() if parameter.requires_grad and id(parameter) not in backbone_ids]
        groups: list[dict[str, object]] = []
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr, "name": "expert_last_backbone"})
        groups.append({"params": new_modules, "lr": modules_lr, "name": "expert_new_modules"})
        return groups

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        view_valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        depth = inputs["depth_input"]
        ir = inputs["ir_input"]
        if depth.ndim != 6 or depth.shape[2] != self.expected_views or depth.shape[3] != 3:
            raise ValueError(f"Expected Depth [B,T,{self.expected_views},3,H,W], got {tuple(depth.shape)}")
        if ir.ndim != 6 or ir.shape[:3] != depth.shape[:3] or ir.shape[3] != 1:
            raise ValueError(f"Expected aligned IR, got {tuple(ir.shape)}")
        batch, frames, views, _, height, width = depth.shape
        count = batch * frames * views
        depth_features = self.depth_stem(depth.reshape(count, 3, height, width))
        ir_features = self.ir_stem(ir.reshape(count, 1, height, width))
        modality_gate = torch.sigmoid(self.modality_gate(torch.cat((depth_features, ir_features), dim=1)))
        fused = modality_gate * depth_features + (1.0 - modality_gate) * ir_features
        tokens = self.avgpool(self.shared_body(fused)).flatten(1).reshape(batch, frames, views, -1)
        gate_logits = self.view_gate(tokens).squeeze(-1)
        safe_valid = view_valid_mask.clone()
        all_invalid = ~safe_valid.any(dim=-1)
        safe_valid[..., 0] |= all_invalid
        gate_logits = gate_logits.masked_fill(~safe_valid, torch.finfo(gate_logits.dtype).min)
        view_weights = torch.softmax(gate_logits, dim=-1)
        view_weights = view_weights.masked_fill(~view_valid_mask, 0.0)
        frame_tokens = (tokens * view_weights.unsqueeze(-1)).sum(dim=2)
        temporal = self.frame_projection(frame_tokens).transpose(1, 2)
        temporal = temporal * temporal_mask.unsqueeze(1)
        skip = torch.zeros_like(temporal)
        activation_norms = []
        for block in self.tcn_blocks:
            temporal = block(temporal) * temporal_mask.unsqueeze(1)
            skip = skip + temporal
            activation_norms.append(temporal.square().mean(dim=(1, 2)).sqrt())
        skip = self.skip_projection(skip / len(self.tcn_blocks)) * temporal_mask.unsqueeze(1)
        attention_logits = self.attention_pool(skip).squeeze(1)
        attention_logits = attention_logits.masked_fill(~temporal_mask, torch.finfo(attention_logits.dtype).min)
        temporal_attention = torch.softmax(attention_logits, dim=1).masked_fill(~temporal_mask, 0.0)
        attention_embedding = (skip * temporal_attention.unsqueeze(1)).sum(dim=2)
        denominator = temporal_mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean_embedding = skip.sum(dim=2) / denominator
        embedding = self.sequence_projection(torch.cat((attention_embedding, mean_embedding), dim=1))
        delta_target = self.residual_head(embedding)
        delta_40 = base_logits.new_zeros((batch, 40))
        delta_40.scatter_(
            1,
            self.target_index.unsqueeze(0).expand(batch, -1),
            delta_target.to(base_logits.dtype),
        )
        final_logits = base_logits.detach() + delta_40
        modality_summary = modality_gate.mean(dim=(1, 2, 3)).reshape(batch, frames, views)
        entropy = -(view_weights.clamp_min(1e-12).log() * view_weights).sum(dim=-1)
        return {
            "logits": final_logits,
            "base_logits": base_logits.detach(),
            "delta_logits_target": delta_target,
            "delta_logits_40": delta_40,
            "embedding": embedding,
            "view_weights": view_weights,
            "view_gate_entropy": entropy,
            "temporal_attention": temporal_attention,
            "modality_gate": modality_summary,
            "tcn_activation_norms": torch.stack(activation_norms, dim=1),
        }
