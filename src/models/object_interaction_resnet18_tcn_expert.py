from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.models.object_interaction_tcn_expert import DilatedResidualBlock


class ObjectInteractionResNet18TCNExpert(nn.Module):
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
        encoder_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        self.depth_conv1 = backbone.conv1
        self.depth_bn1 = backbone.bn1
        self.ir_conv1 = nn.Conv2d(
            1, self.depth_conv1.out_channels, self.depth_conv1.kernel_size, self.depth_conv1.stride,
            self.depth_conv1.padding, bias=False,
        )
        with torch.no_grad():
            self.ir_conv1.weight.copy_(self.depth_conv1.weight.mean(dim=1, keepdim=True))
        self.ir_bn1 = copy.deepcopy(self.depth_bn1)
        self.relu = backbone.relu
        self.modality_gate = nn.Conv2d(128, 64, 1)
        nn.init.zeros_(self.modality_gate.weight)
        nn.init.zeros_(self.modality_gate.bias)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.expected_views = expected_views
        self.encoder_chunk_size = int(encoder_chunk_size)
        self.target_class_ids = tuple(int(value) for value in target_class_ids)
        self.register_buffer("target_index", torch.tensor(self.target_class_ids, dtype=torch.long), persistent=True)
        self.view_gate = nn.Sequential(nn.Linear(512, 96), nn.GELU(), nn.Linear(96, 1))
        self.frame_projection = nn.Sequential(
            nn.Linear(512, frame_feature_dim), nn.GELU(), nn.Dropout(dropout),
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
        self.pretrained_weights_loaded = True
        self.pretrained_weights_name = "ResNet18_Weights.IMAGENET1K_V1"
        self.pretrained_weights_url = weights.url

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def initialize_encoder(self, base_state: dict[str, torch.Tensor]) -> None:
        if not base_state:
            raise ValueError("Frozen base checkpoint state is required")
        if not self.pretrained_weights_loaded:
            raise RuntimeError("ImageNet ResNet18 weights were not loaded")

    def _encode_chunk(self, depth: torch.Tensor, ir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth_features = self.relu(self.depth_bn1(self.depth_conv1(depth)))
        ir_features = self.relu(self.ir_bn1(self.ir_conv1(ir)))
        gate = torch.sigmoid(self.modality_gate(torch.cat((depth_features, ir_features), dim=1)))
        fused = gate * depth_features + (1.0 - gate) * ir_features
        shared = self.maxpool(fused)
        shared = self.layer1(shared)
        shared = self.layer2(shared)
        shared = self.layer3(shared)
        shared = self.layer4(shared)
        return self.avgpool(shared).flatten(1), gate.mean(dim=(1, 2, 3))

    def encode_flat(
        self, depth: torch.Tensor, ir: torch.Tensor, chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.encoder_chunk_size if chunk_size is None else int(chunk_size)
        if size <= 0 or size >= len(depth):
            return self._encode_chunk(depth, ir)
        tokens: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for start in range(0, len(depth), size):
            token, gate = self._encode_chunk(depth[start:start + size], ir[start:start + size])
            tokens.append(token)
            gates.append(gate)
        return torch.cat(tokens), torch.cat(gates)

    def set_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
        encoder_modules = (
            self.depth_conv1, self.depth_bn1, self.ir_conv1, self.ir_bn1, self.modality_gate,
            self.layer1, self.layer2, self.layer3, self.layer4,
        )
        for module in encoder_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        if stage == "warmup":
            self.training_stage = stage
            return
        if stage != "finetune":
            raise ValueError(f"Unknown training stage {stage}")
        for module in (self.modality_gate, self.layer4):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.training_stage = stage

    @staticmethod
    def _freeze_batch_norm_statistics(module: nn.Module) -> None:
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()

    def enforce_frozen_encoder_eval(self) -> None:
        for module in (
            self.depth_conv1, self.depth_bn1, self.ir_conv1, self.ir_bn1,
            self.layer1, self.layer2, self.layer3,
        ):
            module.eval()
        if self.training_stage == "warmup":
            self.modality_gate.eval()
            self.layer4.eval()
        else:
            self._freeze_batch_norm_statistics(self.layer4)

    def parameter_groups(self, backbone_lr: float, modules_lr: float) -> list[dict[str, object]]:
        backbone_ids = {
            id(parameter) for module in (self.layer4, self.modality_gate)
            for parameter in module.parameters() if parameter.requires_grad
        }
        backbone = [p for p in self.parameters() if p.requires_grad and id(p) in backbone_ids]
        new_modules = [p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids]
        groups: list[dict[str, object]] = []
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr, "name": "expert_last_backbone"})
        groups.append({"params": new_modules, "lr": modules_lr, "name": "expert_new_modules"})
        return groups

    def experiment_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "resnet18",
            "pretrained_weights": self.pretrained_weights_name,
            "pretrained_weights_url": self.pretrained_weights_url,
            "pretrained_weights_loaded": self.pretrained_weights_loaded,
            "encoder_chunk_size": self.encoder_chunk_size,
            "token_dimension": 512,
            "ir_input_channels": 1,
            "shared_resnet_body_count": 1,
            "finetune_batch_norm_running_statistics": "frozen",
            "expert_parameter_count": sum(p.numel() for p in self.parameters()),
            "expert_fp32_parameter_bytes": sum(p.numel() * p.element_size() for p in self.parameters()),
        }

    @torch.inference_mode()
    def architecture_probe(self, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        depth = inputs["depth_input"][:, :2].reshape(-1, 3, inputs["depth_input"].shape[-2], inputs["depth_input"].shape[-1])
        ir = inputs["ir_input"][:, :2].reshape(-1, 1, inputs["ir_input"].shape[-2], inputs["ir_input"].shape[-1])
        chunked_tokens, chunked_gate = self.encode_flat(depth, ir, chunk_size=3)
        full_tokens, full_gate = self.encode_flat(depth, ir, chunk_size=0)
        result = {
            **self.experiment_metadata(),
            "chunk_probe_items": len(depth),
            "chunk_vs_full_token_max_abs_error": float((chunked_tokens - full_tokens).abs().max()),
            "chunk_vs_full_gate_max_abs_error": float((chunked_gate - full_gate).abs().max()),
            "ir_rgb_mean_initialization_max_abs_error": float(
                (self.ir_conv1.weight - self.depth_conv1.weight.mean(dim=1, keepdim=True)).abs().max()
            ),
            "initial_modality_gate_mean": float(torch.sigmoid(self.modality_gate.bias).mean()),
            "depth_ir_bn_are_independent": self.depth_bn1 is not self.ir_bn1,
            "residual_head_weight_max_abs": float(self.residual_head.weight.abs().max()),
            "residual_head_bias_max_abs": float(self.residual_head.bias.abs().max()),
        }
        if was_training:
            self.train()
            self.enforce_frozen_encoder_eval()
        return result

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
            raise ValueError(f"Expected aligned native IR, got {tuple(ir.shape)}")
        batch, frames, views, _, height, width = depth.shape
        count = batch * frames * views
        tokens, modality_summary = self.encode_flat(
            depth.reshape(count, 3, height, width), ir.reshape(count, 1, height, width),
        )
        tokens = tokens.reshape(batch, frames, views, 512)
        modality_summary = modality_summary.reshape(batch, frames, views)
        gate_logits = self.view_gate(tokens).squeeze(-1)
        safe_valid = view_valid_mask.clone()
        all_invalid = ~safe_valid.any(dim=-1)
        safe_valid[..., 0] |= all_invalid
        gate_logits = gate_logits.masked_fill(~safe_valid, torch.finfo(gate_logits.dtype).min)
        view_weights = torch.softmax(gate_logits, dim=-1).masked_fill(~view_valid_mask, 0.0)
        frame_tokens = (tokens * view_weights.unsqueeze(-1)).sum(dim=2)
        temporal = self.frame_projection(frame_tokens).transpose(1, 2) * temporal_mask.unsqueeze(1)
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
        delta_40.scatter_(1, self.target_index.unsqueeze(0).expand(batch, -1), delta_target.to(base_logits.dtype))
        final_logits = base_logits.detach() + delta_40
        entropy = -(view_weights.clamp_min(1e-12).log() * view_weights).sum(dim=-1)
        return {
            "logits": final_logits, "base_logits": base_logits.detach(),
            "delta_logits_target": delta_target, "delta_logits_40": delta_40,
            "embedding": embedding, "view_weights": view_weights, "view_gate_entropy": entropy,
            "temporal_attention": temporal_attention, "modality_gate": modality_summary,
            "tcn_activation_norms": torch.stack(activation_norms, dim=1),
        }
