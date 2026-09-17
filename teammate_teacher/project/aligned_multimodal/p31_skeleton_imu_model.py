from __future__ import annotations

import math

import torch
from torch import nn

from p31_skeleton_imu_preprocessing import (
    COMMON_PART_NAMES,
    H36M_PARENTS,
    IMU_DEVICE_NAMES,
    PART_JOINTS,
    SKELETON_FEATURE_NAMES,
    SKELETON_RELATION_NAMES,
)


def normalized_adjacency() -> torch.Tensor:
    adjacency = torch.eye(17, dtype=torch.float32)
    for child, parent_value in enumerate(H36M_PARENTS.tolist()):
        parent = int(parent_value)
        adjacency[child, parent] = 1.0
        adjacency[parent, child] = 1.0
    return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)


def part_joint_matrix() -> torch.Tensor:
    matrix = torch.zeros(len(COMMON_PART_NAMES), 17, dtype=torch.float32)
    for part_index, joint_indices in enumerate(PART_JOINTS):
        matrix[part_index, list(joint_indices)] = 1.0
    return matrix


class GraphTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.self_project = nn.Linear(width, width)
        self.neighbor_project = nn.Linear(width, width, bias=False)
        self.spatial_norm = nn.LayerNorm(width)
        self.temporal_depthwise = nn.Conv2d(
            width,
            width,
            kernel_size=(3, 1),
            padding=(dilation, 0),
            dilation=(dilation, 1),
            groups=width,
            bias=False,
        )
        self.temporal_pointwise = nn.Conv2d(width, width, kernel_size=1, bias=False)
        self.temporal_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        source: torch.Tensor,
        adjacency: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbor = torch.einsum("vw,btwd->btvd", adjacency, source)
        spatial = self.self_project(source) + self.neighbor_project(neighbor)
        source = self.spatial_norm(source + self.dropout(torch.nn.functional.gelu(spatial)))
        source = source * joint_mask.unsqueeze(-1)
        temporal = self.temporal_depthwise(source.permute(0, 3, 1, 2))
        temporal = self.temporal_pointwise(torch.nn.functional.gelu(temporal))
        temporal = temporal.permute(0, 2, 3, 1)
        source = self.temporal_norm(source + self.dropout(temporal))
        return source * joint_mask.unsqueeze(-1)


class SkeletonPartEncoder(nn.Module):
    """Step 8: all-frame 3D joint graph/temporal encoder with eight part tokens."""

    def __init__(
        self,
        graph_width: int = 96,
        output_width: int = 256,
        graph_layers: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.graph_width = graph_width
        self.output_width = output_width
        self.register_buffer("adjacency", normalized_adjacency())
        self.register_buffer("part_matrix", part_joint_matrix())
        self.input_project = nn.Sequential(
            nn.LayerNorm(len(SKELETON_FEATURE_NAMES) * 2),
            nn.Linear(len(SKELETON_FEATURE_NAMES) * 2, graph_width),
            nn.GELU(),
        )
        self.joint_embedding = nn.Parameter(
            torch.randn(1, 1, 17, graph_width) / math.sqrt(graph_width)
        )
        dilations = (1, 2, 4)[:graph_layers]
        if len(dilations) != graph_layers:
            raise ValueError("graph_layers must be between 1 and 3")
        self.blocks = nn.ModuleList(
            GraphTemporalBlock(graph_width, dilation, dropout)
            for dilation in dilations
        )
        self.relation_project = nn.Sequential(
            nn.LayerNorm(len(SKELETON_RELATION_NAMES) * 2),
            nn.Linear(
                len(SKELETON_RELATION_NAMES) * 2,
                len(COMMON_PART_NAMES) * graph_width,
            ),
            nn.GELU(),
        )
        self.quality_project = nn.Sequential(
            nn.Linear(2, graph_width),
            nn.GELU(),
        )
        self.part_embedding = nn.Parameter(
            torch.randn(1, 1, len(COMMON_PART_NAMES), graph_width)
            / math.sqrt(graph_width)
        )
        self.output_project = nn.Sequential(
            nn.LayerNorm(graph_width),
            nn.Linear(graph_width, output_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        skeleton_features: torch.Tensor,
        skeleton_feature_mask: torch.Tensor,
        skeleton_joint_mask: torch.Tensor,
        skeleton_relations: torch.Tensor,
        skeleton_relation_mask: torch.Tensor,
        skeleton_frame_quality: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if skeleton_features.ndim != 4 or skeleton_features.shape[2:] != (
            17,
            len(SKELETON_FEATURE_NAMES),
        ):
            raise ValueError("skeleton_features must have shape [B,T,17,13]")
        joint_mask = skeleton_joint_mask & frame_mask.unsqueeze(-1)
        feature_input = torch.cat(
            (
                skeleton_features * skeleton_feature_mask,
                skeleton_feature_mask.to(skeleton_features.dtype),
            ),
            dim=-1,
        )
        source = self.input_project(feature_input) + self.joint_embedding
        source = source * joint_mask.unsqueeze(-1)
        for block in self.blocks:
            source = block(source, self.adjacency, joint_mask)

        part_numerator = torch.einsum("btjd,pj->btpd", source, self.part_matrix)
        part_count = torch.einsum(
            "btj,pj->btp", joint_mask.to(source.dtype), self.part_matrix
        )
        part_source = part_numerator / part_count.clamp_min(1.0).unsqueeze(-1)
        part_mask = (part_count > 0) & frame_mask.unsqueeze(-1)

        relation_input = torch.cat(
            (
                skeleton_relations * skeleton_relation_mask,
                skeleton_relation_mask.to(skeleton_relations.dtype),
            ),
            dim=-1,
        )
        relation = self.relation_project(relation_input).reshape(
            skeleton_features.shape[0],
            skeleton_features.shape[1],
            len(COMMON_PART_NAMES),
            self.graph_width,
        )
        valid_ratio = part_count / self.part_matrix.sum(dim=1).clamp_min(1.0)
        quality = torch.stack(
            (
                skeleton_frame_quality.unsqueeze(-1).expand_as(valid_ratio),
                valid_ratio,
            ),
            dim=-1,
        )
        part_source = (
            part_source
            + relation
            + self.quality_project(quality)
            + self.part_embedding
        )
        part_tokens = self.output_project(part_source) * part_mask.unsqueeze(-1)

        velocity = skeleton_features[..., 6:9]
        velocity_mask = skeleton_feature_mask[..., 6:9].all(dim=-1) & joint_mask
        joint_motion = torch.linalg.vector_norm(velocity, dim=-1) * velocity_mask
        motion_sum = torch.einsum("btj,pj->btp", joint_motion, self.part_matrix)
        motion_count = torch.einsum(
            "btj,pj->btp", velocity_mask.to(source.dtype), self.part_matrix
        )
        motion_energy = motion_sum / motion_count.clamp_min(1.0)
        motion_energy = motion_energy * part_mask
        return {
            "part_tokens": part_tokens,
            "part_mask": part_mask,
            "part_quality": valid_ratio * skeleton_frame_quality.unsqueeze(-1),
            "motion_energy": motion_energy,
            "joint_sequence": source,
        }


class SharedIMUPointEncoder(nn.Module):
    def __init__(self, input_width: int = 20, hidden_width: int = 128) -> None:
        super().__init__()
        self.first = nn.Conv1d(input_width, 64, kernel_size=5, padding=2)
        self.first_norm = nn.LayerNorm(64)
        self.second = nn.Conv1d(64, hidden_width, kernel_size=5, padding=2)
        self.second_norm = nn.LayerNorm(hidden_width)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        source = self.first(points.transpose(1, 2)).transpose(1, 2)
        source = self.first_norm(torch.nn.functional.gelu(source))
        source = source * mask.unsqueeze(-1)
        source = self.second(source.transpose(1, 2)).transpose(1, 2)
        source = self.second_norm(torch.nn.functional.gelu(source))
        return source * mask.unsqueeze(-1)


class IMUIntervalPartEncoder(nn.Module):
    """Step 9: shared five-device point CNN and exact frame-interval pooling."""

    DIRECT_COMMON_PART_INDEX = (2, 3, 4, 5, 6)

    def __init__(
        self,
        hidden_width: int = 128,
        output_width: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.output_width = output_width
        self.register_buffer(
            "physical_scale",
            torch.tensor(
                (2.0, 2.0, 2.0, 500.0, 500.0, 500.0, 1.0, 1.0, 1.0, 1.0),
                dtype=torch.float32,
            ),
        )
        self.point_encoder = SharedIMUPointEncoder(20, hidden_width)
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_width, hidden_width // 2),
            nn.GELU(),
            nn.Linear(hidden_width // 2, 1),
        )
        interval_input_width = hidden_width * 3 + 13
        self.interval_project = nn.Sequential(
            nn.LayerNorm(interval_input_width),
            nn.Linear(interval_input_width, output_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.device_embedding = nn.Parameter(
            torch.randn(len(IMU_DEVICE_NAMES), output_width) / math.sqrt(output_width)
        )

    def _build_point_features(
        self,
        values: torch.Tensor,
        point_times: torch.Tensor,
        point_frame_index: torch.Tensor,
        point_mask: torch.Tensor,
        frame_times: torch.Tensor,
    ) -> torch.Tensor:
        scaled = values / self.physical_scale
        difference = torch.zeros_like(scaled[..., :6])
        difference[:, :, 1:] = scaled[:, :, 1:, :6] - scaled[:, :, :-1, :6]
        difference_mask = point_mask.clone()
        difference_mask[:, :, 0] = False
        difference_mask[:, :, 1:] &= point_mask[:, :, :-1]
        difference = difference * difference_mask.unsqueeze(-1)

        delta_time = torch.zeros_like(point_times)
        delta_time[:, :, 1:] = point_times[:, :, 1:] - point_times[:, :, :-1]
        delta_time = torch.clamp(delta_time / 0.05, 0.0, 10.0)
        safe_index = point_frame_index.clamp(0, frame_times.shape[1] - 1)
        expanded_frame_times = frame_times.unsqueeze(1).expand(
            -1, len(IMU_DEVICE_NAMES), -1
        )
        assigned_frame_time = torch.gather(expanded_frame_times, 2, safe_index)
        frame_offset = torch.clamp((point_times - assigned_frame_time) / 0.10, -10.0, 10.0)
        acc_norm = torch.linalg.vector_norm(scaled[..., :3], dim=-1)
        gyro_norm = torch.linalg.vector_norm(scaled[..., 3:6], dim=-1)
        point_features = torch.cat(
            (
                scaled,
                difference,
                acc_norm.unsqueeze(-1),
                gyro_norm.unsqueeze(-1),
                delta_time.unsqueeze(-1),
                frame_offset.unsqueeze(-1),
            ),
            dim=-1,
        )
        return point_features * point_mask.unsqueeze(-1)

    def _pool_one_device(
        self,
        hidden: torch.Tensor,
        scaled_raw: torch.Tensor,
        frame_index: torch.Tensor,
        point_mask: torch.Tensor,
        time_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_hidden = hidden[point_mask]
        valid_raw = scaled_raw[point_mask, :6]
        valid_index = frame_index[point_mask]
        count = hidden.new_zeros(time_steps)
        if len(valid_hidden) == 0:
            return (
                hidden.new_zeros(time_steps, self.output_width),
                torch.zeros(time_steps, dtype=torch.bool, device=hidden.device),
                count,
            )

        ones = count.new_ones(len(valid_index))
        count.index_add_(0, valid_index, ones)
        summed = hidden.new_zeros(time_steps, self.hidden_width)
        summed.index_add_(0, valid_index, valid_hidden)
        mean = summed / count.clamp_min(1.0).unsqueeze(-1)

        expanded_index = valid_index.unsqueeze(-1).expand(-1, self.hidden_width)
        maximum = hidden.new_full((time_steps, self.hidden_width), -1e4)
        maximum.scatter_reduce_(0, expanded_index, valid_hidden, reduce="amax", include_self=True)

        attention_weight = torch.sigmoid(self.attention_score(valid_hidden)).squeeze(-1)
        attended_sum = hidden.new_zeros(time_steps, self.hidden_width)
        attended_sum.index_add_(
            0, valid_index, valid_hidden * attention_weight.unsqueeze(-1)
        )
        attention_denominator = torch.zeros(
            time_steps,
            dtype=attention_weight.dtype,
            device=attention_weight.device,
        )
        attention_denominator.index_add_(0, valid_index, attention_weight)
        attended = attended_sum / attention_denominator.clamp_min(1e-6).unsqueeze(-1)

        squared_sum = valid_raw.new_zeros(time_steps, 6)
        squared_sum.index_add_(0, valid_index, valid_raw.square())
        rms = torch.sqrt(squared_sum / count.clamp_min(1.0).unsqueeze(-1) + 1e-8)
        raw_index = valid_index.unsqueeze(-1).expand(-1, 6)
        peak = valid_raw.new_zeros(time_steps, 6)
        peak.scatter_reduce_(0, raw_index, valid_raw.abs(), reduce="amax", include_self=True)
        statistics = torch.cat((rms, peak, torch.log1p(count).unsqueeze(-1)), dim=-1)
        interval_valid = count > 0
        maximum = torch.where(interval_valid.unsqueeze(-1), maximum, 0.0)
        pooled = torch.cat((mean, maximum, attended, statistics), dim=-1)
        token = self.interval_project(pooled) * interval_valid.unsqueeze(-1)
        return token, interval_valid, count

    def _pool_all_devices(
        self,
        hidden: torch.Tensor,
        scaled_raw: torch.Tensor,
        frame_index: torch.Tensor,
        point_mask: torch.Tensor,
        time_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized equivalent of calling ``_pool_one_device`` B*D times.

        A unique segment id is assigned to every (batch, device, frame)
        interval. This preserves the exact mean/max/attention/RMS/peak features
        while replacing the Python B-by-5 loop with a handful of batched CUDA
        reductions.
        """
        batch_size, devices, points, hidden_width = hidden.shape
        segment_count = batch_size * devices * time_steps
        flat_hidden = hidden.reshape(-1, hidden_width)
        flat_raw = scaled_raw[..., :6].reshape(-1, 6)
        flat_mask = point_mask.reshape(-1)
        batch_device = (
            torch.arange(batch_size * devices, device=hidden.device)
            .unsqueeze(1)
            .expand(-1, points)
            .reshape(-1)
        )
        flat_frame = frame_index.reshape(-1).clamp(0, time_steps - 1)
        segment = batch_device * time_steps + flat_frame

        valid_hidden = flat_hidden[flat_mask]
        valid_raw = flat_raw[flat_mask]
        valid_segment = segment[flat_mask]
        count = hidden.new_zeros(segment_count)
        count.index_add_(0, valid_segment, count.new_ones(len(valid_segment)))

        summed = hidden.new_zeros(segment_count, hidden_width)
        summed.index_add_(0, valid_segment, valid_hidden)
        mean = summed / count.clamp_min(1.0).unsqueeze(-1)

        expanded_segment = valid_segment.unsqueeze(-1).expand(-1, hidden_width)
        maximum = hidden.new_full((segment_count, hidden_width), -1e4)
        maximum.scatter_reduce_(
            0, expanded_segment, valid_hidden, reduce="amax", include_self=True
        )

        attention_weight = torch.sigmoid(self.attention_score(valid_hidden)).squeeze(-1)
        attended_sum = hidden.new_zeros(segment_count, hidden_width)
        attended_sum.index_add_(
            0, valid_segment, valid_hidden * attention_weight.unsqueeze(-1)
        )
        attention_denominator = torch.zeros(
            segment_count,
            dtype=attention_weight.dtype,
            device=hidden.device,
        )
        attention_denominator.index_add_(0, valid_segment, attention_weight)
        attended = attended_sum / attention_denominator.clamp_min(1e-6).unsqueeze(-1)

        squared_sum = valid_raw.new_zeros(segment_count, 6)
        squared_sum.index_add_(0, valid_segment, valid_raw.square())
        rms = torch.sqrt(squared_sum / count.clamp_min(1.0).unsqueeze(-1) + 1e-8)
        raw_segment = valid_segment.unsqueeze(-1).expand(-1, 6)
        peak = valid_raw.new_zeros(segment_count, 6)
        peak.scatter_reduce_(
            0, raw_segment, valid_raw.abs(), reduce="amax", include_self=True
        )
        interval_valid = count > 0
        maximum = torch.where(interval_valid.unsqueeze(-1), maximum, 0.0)
        statistics = torch.cat((rms, peak, torch.log1p(count).unsqueeze(-1)), dim=-1)
        pooled = torch.cat((mean, maximum, attended, statistics), dim=-1)
        token = self.interval_project(pooled) * interval_valid.unsqueeze(-1)

        token = token.reshape(batch_size, devices, time_steps, self.output_width)
        interval_valid = interval_valid.reshape(batch_size, devices, time_steps)
        count = count.reshape(batch_size, devices, time_steps)
        return (
            token.permute(0, 2, 1, 3).contiguous(),
            interval_valid.permute(0, 2, 1).contiguous(),
            count.permute(0, 2, 1).contiguous(),
        )

    def forward(
        self,
        imu_values: torch.Tensor,
        imu_time_seconds: torch.Tensor,
        imu_frame_index: torch.Tensor,
        imu_point_mask: torch.Tensor,
        imu_device_mask: torch.Tensor,
        frame_time_seconds: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if imu_values.ndim != 4 or imu_values.shape[1] != len(IMU_DEVICE_NAMES):
            raise ValueError("imu_values must have shape [B,5,N,10]")
        batch_size, devices, points, _ = imu_values.shape
        time_steps = frame_mask.shape[1]
        point_mask = imu_point_mask & imu_device_mask.unsqueeze(-1)
        point_features = self._build_point_features(
            imu_values,
            imu_time_seconds,
            imu_frame_index,
            point_mask,
            frame_time_seconds,
        )
        hidden = self.point_encoder(
            point_features.reshape(batch_size * devices, points, -1),
            point_mask.reshape(batch_size * devices, points),
        ).reshape(batch_size, devices, points, self.hidden_width)
        scaled_raw = imu_values / self.physical_scale

        device_tokens, interval_mask, interval_count = self._pool_all_devices(
            hidden,
            scaled_raw,
            imu_frame_index,
            point_mask,
            time_steps,
        )
        interval_mask = interval_mask & frame_mask.unsqueeze(-1)
        device_tokens = device_tokens + self.device_embedding[None, None]
        device_tokens = device_tokens * interval_mask.unsqueeze(-1)

        part_tokens = imu_values.new_zeros(
            batch_size, time_steps, len(COMMON_PART_NAMES), self.output_width
        )
        part_mask = torch.zeros(
            batch_size,
            time_steps,
            len(COMMON_PART_NAMES),
            dtype=torch.bool,
            device=imu_values.device,
        )
        part_quality = imu_values.new_zeros(
            batch_size, time_steps, len(COMMON_PART_NAMES)
        )
        for device_index, part_index in enumerate(self.DIRECT_COMMON_PART_INDEX):
            part_tokens[:, :, part_index] = device_tokens[:, :, device_index]
            part_mask[:, :, part_index] = interval_mask[:, :, device_index]
            part_quality[:, :, part_index] = torch.log1p(
                interval_count[:, :, device_index]
            )

        device_weight = interval_mask.to(imu_values.dtype)
        global_count = device_weight.sum(dim=2)
        part_tokens[:, :, 0] = device_tokens.sum(dim=2) / global_count.clamp_min(
            1.0
        ).unsqueeze(-1)
        part_mask[:, :, 0] = global_count > 0
        part_quality[:, :, 0] = torch.log1p(interval_count.sum(dim=2))

        arm_tokens = device_tokens[:, :, 1:3]
        arm_mask = interval_mask[:, :, 1:3]
        arm_count = arm_mask.to(imu_values.dtype).sum(dim=2)
        part_tokens[:, :, 7] = arm_tokens.sum(dim=2) / arm_count.clamp_min(1.0).unsqueeze(-1)
        part_mask[:, :, 7] = arm_count > 0
        part_quality[:, :, 7] = torch.log1p(interval_count[:, :, 1:3].sum(dim=2))
        part_tokens = part_tokens * part_mask.unsqueeze(-1)
        part_quality = part_quality / math.log(9.0)
        return {
            "part_tokens": part_tokens,
            "part_mask": part_mask,
            "part_quality": part_quality,
            "device_tokens": device_tokens,
            "interval_mask": interval_mask,
            "interval_count": interval_count,
            "point_sequence": hidden,
        }


class P31SkeletonIMUPartEncoders(nn.Module):
    """Run Steps 8 and 9 in parallel; Step 13 will fuse these outputs with P30."""

    def __init__(self, output_width: int = 256, dropout: float = 0.10) -> None:
        super().__init__()
        self.skeleton = SkeletonPartEncoder(output_width=output_width, dropout=dropout)
        self.imu = IMUIntervalPartEncoder(output_width=output_width, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        skeleton = self.skeleton(
            batch["skeleton_features"],
            batch["skeleton_feature_mask"],
            batch["skeleton_joint_mask"],
            batch["skeleton_relations"],
            batch["skeleton_relation_mask"],
            batch["skeleton_frame_quality"],
            batch["frame_mask"],
        )
        imu = self.imu(
            batch["imu_values"],
            batch["imu_time_seconds"],
            batch["imu_frame_index"],
            batch["imu_point_mask"],
            batch["imu_device_mask"],
            batch["frame_time_seconds"],
            batch["frame_mask"],
        )
        return {
            "skeleton_part_tokens": skeleton["part_tokens"],
            "skeleton_part_mask": skeleton["part_mask"],
            "skeleton_part_quality": skeleton["part_quality"],
            "skeleton_motion_energy": skeleton["motion_energy"],
            "imu_part_tokens": imu["part_tokens"],
            "imu_part_mask": imu["part_mask"],
            "imu_part_quality": imu["part_quality"],
            "imu_device_tokens": imu["device_tokens"],
            "imu_interval_mask": imu["interval_mask"],
            "imu_interval_count": imu["interval_count"],
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def model_size_mib(module: nn.Module, bytes_per_parameter: int = 4) -> float:
    return parameter_count(module) * bytes_per_parameter / (1024**2)
