from __future__ import annotations

import torch
from torch import nn

from p31_skeleton_imu_preprocessing import H36M_PARENTS, PART_JOINTS


MOTION_PART_JOINTS = tuple(PART_JOINTS[index] for index in range(2, 7))


def skeleton_adjacency() -> torch.Tensor:
    adjacency = torch.eye(17, dtype=torch.float32)
    for child, parent in enumerate(H36M_PARENTS.tolist()):
        adjacency[child, parent] = 1.0
        adjacency[parent, child] = 1.0
    adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    count = weight.sum(dim=dimensions).clamp_min(1.0)
    return (values * weight).sum(dim=dimensions) / count


class SpatialTemporalAttentionBlock(nn.Module):
    """Fixed anatomical propagation plus sample-conditioned joint attention."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(width)
        self.graph_projection = nn.Linear(width, width, bias=False)
        self.joint_attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.spatial_gate = nn.Parameter(torch.tensor(-1.5))
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_depthwise = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.temporal_pointwise = nn.Conv1d(width, width, 1)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 3, width), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("adjacency", skeleton_adjacency(), persistent=True)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # values [B,T,J,D], mask [B,T,J]
        batch, steps, joints, width = values.shape
        spatial = self.spatial_norm(values)
        graph = torch.einsum("ij,btjd->btid", self.adjacency, spatial)
        graph = self.graph_projection(graph)
        flat = spatial.reshape(batch * steps, joints, width)
        flat_mask = mask.reshape(batch * steps, joints)
        safe = flat_mask.clone()
        empty = ~safe.any(dim=1)
        safe[empty, 0] = True
        attention, _ = self.joint_attention(flat, flat, flat, key_padding_mask=~safe, need_weights=False)
        attention = attention.reshape(batch, steps, joints, width)
        gate = torch.sigmoid(self.spatial_gate)
        values = values + self.dropout(graph + gate * attention)

        temporal = self.temporal_norm(values).permute(0, 2, 3, 1).reshape(batch * joints, width, steps)
        temporal = self.temporal_pointwise(torch.nn.functional.gelu(self.temporal_depthwise(temporal)))
        temporal = temporal.reshape(batch, joints, width, steps).permute(0, 3, 1, 2)
        values = values + self.dropout(temporal)
        values = values + self.ffn(self.ffn_norm(values))
        return values * mask.to(values.dtype).unsqueeze(-1)


class SkeletonStreamEncoder(nn.Module):
    def __init__(self, width: int = 48, dropout: float = 0.15) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, width), nn.GELU(), nn.Dropout(dropout))
        self.window_embedding = nn.Parameter(torch.zeros(2, width))
        nn.init.trunc_normal_(self.window_embedding, std=0.02)
        self.blocks = nn.ModuleList(SpatialTemporalAttentionBlock(width, dropout) for _ in range(3))
        temporal_layer = nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, 1)
        self.pool = nn.Sequential(nn.LayerNorm(width * 4), nn.Linear(width * 4, width * 2), nn.GELU(), nn.Dropout(dropout))

    def forward(self, xyz: torch.Tensor, mask: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        # Input keeps the early and late windows explicit until after embedding.
        batch, windows, steps, joints, _ = xyz.shape
        encoded = self.stem(torch.cat((xyz, confidence.unsqueeze(-1)), dim=-1))
        encoded = encoded + self.window_embedding[None, :, None, None, :]
        encoded = encoded.reshape(batch, windows * steps, joints, -1)
        flat_mask = mask.reshape(batch, windows * steps, joints)
        for block in self.blocks:
            encoded = block(encoded, flat_mask)

        global_mean = masked_mean(encoded, flat_mask, (1, 2))
        maximum = encoded.masked_fill(~flat_mask.unsqueeze(-1), -1e4).amax(dim=(1, 2))
        maximum = torch.where(flat_mask.any(dim=(1, 2)).unsqueeze(-1), maximum, 0.0)

        part_values = []
        for indices in MOTION_PART_JOINTS:
            part_values.append(masked_mean(encoded[:, :, list(indices)], flat_mask[:, :, list(indices)], (1, 2)))
        part_mean = torch.stack(part_values, dim=1).mean(dim=1)

        time_token = masked_mean(encoded, flat_mask, (2,))
        time_mask = flat_mask.any(dim=2)
        safe = time_mask.clone()
        empty = ~safe.any(dim=1)
        safe[empty, 0] = True
        time_token = self.temporal(time_token, src_key_padding_mask=~safe)
        temporal_mean = masked_mean(time_token, time_mask, (1,))
        return self.pool(torch.cat((global_mean, maximum, part_mean, temporal_mean), dim=-1))


class P89FourStreamSkeleton(nn.Module):
    STREAMS = ("joint", "bone", "joint_motion", "bone_motion")

    def __init__(self, width: int = 48, classes: int = 40, dropout: float = 0.15) -> None:
        super().__init__()
        self.parents = H36M_PARENTS.tolist()
        self.encoders = nn.ModuleDict({name: SkeletonStreamEncoder(width, dropout) for name in self.STREAMS})
        embedding_width = width * 2
        self.stream_heads = nn.ModuleDict({name: nn.Linear(embedding_width, classes) for name in self.STREAMS})
        self.stream_gate = nn.Sequential(
            nn.LayerNorm(embedding_width * 4), nn.Linear(embedding_width * 4, 4), nn.Softmax(dim=-1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(embedding_width * 4),
            nn.Linear(embedding_width * 4, embedding_width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embedding_width * 2, embedding_width), nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_width, classes)

    def stream_inputs(
        self, features: torch.Tensor, joint_mask: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        joint = features[..., :3]
        bone = features[..., 3:6]
        joint_motion = features[..., 6:9]
        bone_motion = torch.zeros_like(bone)
        bone_motion[:, :, 1:] = bone[:, :, 1:] - bone[:, :, :-1]
        bone_mask = joint_mask & joint_mask[..., self.parents]
        motion_mask = joint_mask.clone()
        motion_mask[:, :, 0] = False
        motion_mask[:, :, 1:] &= joint_mask[:, :, :-1]
        bone_motion_mask = bone_mask.clone()
        bone_motion_mask[:, :, 0] = False
        bone_motion_mask[:, :, 1:] &= bone_mask[:, :, :-1]
        confidence = features[..., 12].clamp(0.0, 1.0)
        return {
            "joint": (joint, joint_mask, confidence),
            "bone": (bone, bone_mask, confidence),
            "joint_motion": (joint_motion, motion_mask, confidence),
            "bone_motion": (bone_motion, bone_motion_mask, confidence),
        }

    def forward(self, features: torch.Tensor, joint_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = self.stream_inputs(features, joint_mask)
        embeddings = {name: self.encoders[name](*inputs[name]) for name in self.STREAMS}
        concatenated = torch.cat([embeddings[name] for name in self.STREAMS], dim=-1)
        gates = self.stream_gate(concatenated)
        stream_logits = torch.stack([self.stream_heads[name](embeddings[name]) for name in self.STREAMS], dim=1)
        fused_embedding = self.fusion(concatenated)
        logits = self.classifier(fused_embedding) + 0.25 * (stream_logits * gates.unsqueeze(-1)).sum(dim=1)
        return {
            "logits": logits,
            "embedding": fused_embedding,
            "stream_logits": stream_logits,
            "stream_gates": gates,
        }
