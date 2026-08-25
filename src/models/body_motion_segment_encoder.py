from __future__ import annotations

import torch
from torch import nn

from src.models.multimodal_token_contract import GroupTokens


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_scores = torch.where(mask, scores, torch.zeros_like(scores))
    maximum = masked_scores.masked_fill(~mask, torch.finfo(scores.dtype).min).max(
        dim=-1, keepdim=True
    ).values
    maximum = torch.where(mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
    shifted = torch.where(mask, masked_scores - maximum, torch.zeros_like(scores))
    exponent = torch.exp(shifted) * mask.to(scores.dtype)
    return exponent / exponent.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class TemporalDepthwiseResidual(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pointwise = nn.Conv1d(dim, dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.depthwise(values.transpose(1, 2))).transpose(1, 2)
        output = torch.nn.functional.gelu(self.norm(values + residual))
        return output * mask[:, :, None].to(output.dtype)


class BodyMotionSegmentEncoder(nn.Module):
    """Encode Skeleton and role-aware IMU before fusing their segment evidence."""

    def __init__(self, *, output_dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        if output_dim < 1 or heads < 1 or output_dim % heads:
            raise ValueError("body-motion dimensions are incompatible")
        self.skeleton_projection = nn.Linear(17 * 6, output_dim)
        self.skeleton_norm = nn.LayerNorm(output_dim)
        self.skeleton_temporal = TemporalDepthwiseResidual(output_dim)
        self.imu_projection = nn.Linear(16, output_dim)
        self.imu_norm = nn.LayerNorm(output_dim)
        self.imu_role_score = nn.Linear(output_dim, 1, bias=False)
        self.imu_temporal = TemporalDepthwiseResidual(output_dim)
        self.cross_attention = nn.MultiheadAttention(
            output_dim, heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        skeleton: torch.Tensor,
        imu: torch.Tensor,
        skeleton_mask: torch.Tensor,
        imu_role_mask: torch.Tensor,
    ) -> GroupTokens:
        if skeleton.ndim != 4 or skeleton.shape[1:] != (8, 17, 6):
            raise ValueError("Skeleton must have shape [B,8,17,6]")
        batch = skeleton.shape[0]
        if imu.shape != (batch, 8, 5, 16):
            raise ValueError("IMU must have shape [B,8,5,16]")
        if skeleton_mask.shape != (batch, 8) or skeleton_mask.dtype != torch.bool:
            raise ValueError("Skeleton mask must be bool [B,8]")
        if imu_role_mask.shape != (batch, 8, 5) or imu_role_mask.dtype != torch.bool:
            raise ValueError("IMU role mask must be bool [B,8,5]")

        skeleton_token = self.skeleton_projection(skeleton.flatten(2))
        skeleton_token = torch.nn.functional.gelu(self.skeleton_norm(skeleton_token))
        skeleton_token = self.skeleton_temporal(skeleton_token, skeleton_mask)

        imu_roles = self.imu_projection(imu)
        imu_roles = torch.nn.functional.gelu(self.imu_norm(imu_roles))
        role_scores = self.imu_role_score(imu_roles).squeeze(-1)
        role_weights = _masked_softmax(role_scores, imu_role_mask)
        imu_token = torch.einsum("bsr,bsrd->bsd", role_weights, imu_roles)
        imu_mask = imu_role_mask.any(dim=2)
        imu_token = self.imu_temporal(imu_token, imu_mask)

        pair = torch.stack((skeleton_token, imu_token), dim=2).reshape(
            batch * 8, 2, skeleton_token.shape[-1]
        )
        attended, _ = self.cross_attention(pair, pair, pair, need_weights=False)
        attended = self.cross_norm(attended + pair).mean(dim=1).reshape(
            batch, 8, -1
        )
        both = skeleton_mask & imu_mask
        skeleton_only = skeleton_mask & ~imu_mask
        imu_only = imu_mask & ~skeleton_mask
        fused = torch.zeros_like(skeleton_token)
        fused = torch.where(both[:, :, None], attended, fused)
        fused = torch.where(skeleton_only[:, :, None], skeleton_token, fused)
        fused = torch.where(imu_only[:, :, None], imu_token, fused)
        group_mask = skeleton_mask | imu_mask
        fused = fused * group_mask[:, :, None].to(fused.dtype)

        quality = torch.cat(
            (
                skeleton_mask[:, :, None].to(fused.dtype),
                imu_mask[:, :, None].to(fused.dtype),
                role_weights,
                skeleton_token.norm(dim=2, keepdim=True),
                imu_token.norm(dim=2, keepdim=True),
            ),
            dim=2,
        )[:, :, None]
        quality = quality * group_mask[:, :, None, None].to(quality.dtype)
        result = GroupTokens(
            tokens=fused[:, :, None],
            mask=group_mask[:, :, None],
            quality=quality,
            quality_mask=group_mask[:, :, None, None].expand_as(quality),
        )
        result.validate(
            batch=batch, segments=8, streams=1, dim=fused.shape[-1]
        )
        return result
