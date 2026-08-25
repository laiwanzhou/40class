from __future__ import annotations

import torch
from torch import nn
from torch.utils import checkpoint

from src.models.multimodal_token_contract import GroupTokens


class VideoMAESegmentBackboneAdapter(nn.Module):
    """Expose a VideoMAE backbone as aligned prefix and eight-segment tail seams."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        frozen_prefix_blocks: int = 8,
        segment_count: int = 8,
    ) -> None:
        super().__init__()
        if not 0 <= frozen_prefix_blocks < len(backbone.blocks):
            raise ValueError("invalid frozen VideoMAE prefix")
        if segment_count != 8:
            raise ValueError("Stage-1 VideoMAE segment count must be eight")
        self.backbone = backbone
        self.embed_dim = int(backbone.embed_dim)
        self.frozen_prefix_blocks = int(frozen_prefix_blocks)
        self.segment_count = int(segment_count)
        for parameter in backbone.patch_embed.parameters():
            parameter.requires_grad = False
        for block in backbone.blocks[: self.frozen_prefix_blocks]:
            for parameter in block.parameters():
                parameter.requires_grad = False
        for block in backbone.blocks[self.frozen_prefix_blocks :]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def encode_prefix(self, clips: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.backbone.patch_embed(clips)
            positional = self.backbone.pos_embed.to(
                device=tokens.device, dtype=tokens.dtype
            )
            if positional.shape[1] < tokens.shape[1]:
                raise ValueError("VideoMAE positional embedding is too short")
            tokens = self.backbone.pos_drop(tokens + positional[:, : tokens.shape[1]])
            for block in self.backbone.blocks[: self.frozen_prefix_blocks]:
                tokens = block(tokens)
        if tokens.shape[1] % self.segment_count:
            raise ValueError("VideoMAE token count is not divisible by eight segments")
        spatial_tokens = tokens.shape[1] // self.segment_count
        return tokens.detach().reshape(
            tokens.shape[0], self.segment_count, spatial_tokens, tokens.shape[2]
        )

    def encode_tail(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[1] != self.segment_count:
            raise ValueError("VideoMAE prefix tokens must be [B,8,P,D]")
        batch, segments, spatial, dim = tokens.shape
        values = tokens.reshape(batch, segments * spatial, dim)
        for block in self.backbone.blocks[self.frozen_prefix_blocks :]:
            if self.training and torch.is_grad_enabled():
                values = checkpoint.checkpoint(block, values, use_reentrant=False)
            else:
                values = block(values)
        values = values.reshape(batch, segments, spatial, dim).mean(dim=2)
        return self.backbone.fc_norm(values)


def _masked_pair_weights(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if scores.shape != mask.shape or scores.shape[-1] != 2:
        raise ValueError("pair scores and mask must have matching [...,2] shape")
    masked_scores = torch.where(mask, scores, torch.zeros_like(scores))
    maximum = masked_scores.masked_fill(~mask, torch.finfo(scores.dtype).min).max(
        dim=-1, keepdim=True
    ).values
    maximum = torch.where(mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
    shifted = torch.where(mask, masked_scores - maximum, torch.zeros_like(scores))
    exponent = torch.exp(shifted) * mask.to(scores.dtype)
    return exponent / exponent.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class StructuredIRDepthVisualEncoder(nn.Module):
    """Aligned IR/Depth features with persistent context and wrist group tokens."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        output_dim: int = 256,
        depth_hidden_dim: int = 128,
        depth_gate_initial_bias: float = -2.0,
    ) -> None:
        super().__init__()
        embedding_dim = int(getattr(backbone, "embed_dim", 0))
        if embedding_dim < 1 or output_dim < 1:
            raise ValueError("visual backbone and output dimensions must be positive")
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        joint_dim = embedding_dim * 3
        self.depth_adapter = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, depth_hidden_dim),
            nn.GELU(),
            nn.Linear(depth_hidden_dim, embedding_dim),
        )
        nn.init.zeros_(self.depth_adapter[-1].weight)
        nn.init.zeros_(self.depth_adapter[-1].bias)
        self.depth_gate = nn.Sequential(nn.LayerNorm(joint_dim), nn.Linear(joint_dim, 1))
        nn.init.zeros_(self.depth_gate[-1].weight)
        nn.init.constant_(self.depth_gate[-1].bias, depth_gate_initial_bias)
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.context_router = nn.Linear(output_dim, 1, bias=False)
        self.wrist_router = nn.Linear(output_dim, 1, bias=False)

    def _encode_view(
        self,
        ir: torch.Tensor,
        depth: torch.Tensor,
        ir_available: torch.Tensor,
        depth_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ir_tokens = self.backbone.encode_prefix(ir)
        depth_tokens = self.backbone.encode_prefix(depth)
        if ir_tokens.shape != depth_tokens.shape or ir_tokens.ndim != 4:
            raise ValueError("visual prefix must return aligned [B,8,P,D] tokens")
        joint = torch.cat((ir_tokens, depth_tokens, depth_tokens - ir_tokens), dim=-1)
        delta = self.depth_adapter(joint)
        gate = torch.sigmoid(self.depth_gate(joint))
        depth_mask = depth_available[:, None, None, None].to(gate.dtype)
        gate = gate * depth_mask
        fused = ir_tokens + gate * delta
        fused = torch.where(
            ir_available[:, None, None, None], fused, depth_tokens
        )
        available = ir_available | depth_available
        fused = fused * available[:, None, None, None].to(fused.dtype)
        encoded = self.backbone.encode_tail(fused)
        if encoded.ndim != 3 or encoded.shape[1] != 8:
            raise ValueError("visual tail must return [B,8,D]")
        encoded = self.output_projection(encoded)
        encoded = encoded * available[:, None, None].to(encoded.dtype)
        delta_norm = delta.square().mean(dim=(2, 3)).sqrt()
        gate_mean = gate.mean(dim=(2, 3))
        return encoded, delta_norm, gate_mean

    def _combine_pair(
        self,
        features: torch.Tensor,
        available: torch.Tensor,
        router: nn.Linear,
        delta_norm: torch.Tensor,
        gate_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features [B,8,2,D], available [B,2]
        mask = available[:, None].expand(-1, features.shape[1], -1)
        scores = router(features).squeeze(-1)
        weights = _masked_pair_weights(scores, mask)
        token = torch.einsum("bsv,bsvd->bsd", weights, features)
        group_mask = mask.any(dim=2)
        token = token * group_mask[:, :, None].to(token.dtype)
        weighted_delta = (weights * delta_norm).sum(dim=2)
        weighted_gate = (weights * gate_mean).sum(dim=2)
        quality = torch.stack(
            (weighted_delta, weighted_gate, weights[:, :, 0], weights[:, :, 1]),
            dim=2,
        )
        quality = quality * group_mask[:, :, None].to(quality.dtype)
        return token, group_mask, quality

    def forward(
        self,
        *,
        ir: torch.Tensor,
        depth: torch.Tensor,
        availability: torch.Tensor,
    ) -> GroupTokens:
        if ir.ndim != 6 or ir.shape[1:3] != (4, 3):
            raise ValueError("IR must have shape [B,4,3,T,H,W]")
        if depth.shape != ir.shape:
            raise ValueError("Depth must match IR")
        if availability.shape != (ir.shape[0], 2, 4) or availability.dtype != torch.bool:
            raise ValueError("visual availability must be bool [B,2,4]")
        view_features, delta_norms, gate_means = [], [], []
        for view in range(4):
            encoded, delta_norm, gate_mean = self._encode_view(
                ir[:, view],
                depth[:, view],
                availability[:, 0, view],
                availability[:, 1, view],
            )
            view_features.append(encoded)
            delta_norms.append(delta_norm)
            gate_means.append(gate_mean)
        features = torch.stack(view_features, dim=2)
        delta_norm = torch.stack(delta_norms, dim=2)
        gate_mean = torch.stack(gate_means, dim=2)
        view_available = availability.any(dim=1)
        context = self._combine_pair(
            features[:, :, :2],
            view_available[:, :2],
            self.context_router,
            delta_norm[:, :, :2],
            gate_mean[:, :, :2],
        )
        wrist = self._combine_pair(
            features[:, :, 2:],
            view_available[:, 2:],
            self.wrist_router,
            delta_norm[:, :, 2:],
            gate_mean[:, :, 2:],
        )
        tokens = torch.stack((context[0], wrist[0]), dim=2)
        mask = torch.stack((context[1], wrist[1]), dim=2)
        quality = torch.stack((context[2], wrist[2]), dim=2)
        result = GroupTokens(
            tokens=tokens,
            mask=mask,
            quality=quality,
            quality_mask=mask[:, :, :, None].expand_as(quality),
        )
        result.validate(
            batch=ir.shape[0], segments=8, streams=2, dim=tokens.shape[-1]
        )
        return result
