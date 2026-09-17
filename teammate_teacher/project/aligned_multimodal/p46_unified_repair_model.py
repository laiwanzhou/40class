from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from p31_skeleton_imu_preprocessing import COMMON_PART_NAMES, IMU_DEVICE_NAMES
from p46_event_model import (
    EVENT_PART_NAMES,
    CompleteEventTemporalEncoder,
    LocalVisualObjectEncoder,
    MotionAdapter,
    PartTemporalBlock,
    SamePartEventFusion,
)
from p46_step10_model import P46Step10Model
from p46r_event_bottleneck_model import (
    P46R_MOTION_PART_INDICES,
    P46R_OFFSET_FRACTIONS,
)


def _temporal_difference(source: torch.Tensor) -> torch.Tensor:
    difference = torch.zeros_like(source)
    if source.shape[1] > 1:
        difference[:, 1:] = source[:, 1:] - source[:, :-1]
    return difference


def _masked_distribution(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    safe = logits.float().masked_fill(~mask, -1e4)
    weight = torch.softmax(safe, dim=1) * mask
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)


class RawIMUResidualEncoder(nn.Module):
    """Pool raw-axis IMU inside the existing per-frame/per-part IMU path.

    This is deliberately not an independent classifier branch.  Raw coordinates
    are converted to one residual token for the same device/frame intervals used
    by the compensated IMU encoder, then fused into its event-part tokens.
    """

    DIRECT_COMMON_PART_INDEX = (2, 3, 4, 5, 6)

    def __init__(
        self,
        width: int,
        dropout: float,
        *,
        axis_rotation_degrees: float = 0.0,
        coordinate_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.axis_rotation_radians = math.radians(float(axis_rotation_degrees))
        self.coordinate_dropout = float(coordinate_dropout)
        self.register_buffer(
            "physical_scale",
            torch.tensor((2.0, 2.0, 2.0, 500.0, 500.0, 500.0)),
        )
        # raw vector, raw first difference, acceleration/gyro norm, and delta-t
        self.point_project = nn.Sequential(
            nn.LayerNorm(15),
            nn.Linear(15, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # encoded mean/max, raw RMS/peak, and log interval count
        self.interval_project = nn.Sequential(
            nn.LayerNorm(width * 2 + 13),
            nn.Linear(width * 2 + 13, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.device_embedding = nn.Parameter(
            torch.randn(len(IMU_DEVICE_NAMES), width) / math.sqrt(width)
        )

    def _augment_raw_coordinates(self, raw: torch.Tensor) -> torch.Tensor:
        """Remove fixed sensor-axis shortcuts while preserving IMU timing."""

        if not self.training:
            return raw
        batch, devices = raw.shape[:2]
        augmented = raw
        if self.axis_rotation_radians > 0.0:
            axis = torch.randn(
                batch, devices, 3, device=raw.device, dtype=torch.float32
            )
            axis = F.normalize(axis, dim=-1, eps=1e-6)
            angle = (
                torch.rand(batch, devices, device=raw.device, dtype=torch.float32)
                * 2.0
                - 1.0
            ) * self.axis_rotation_radians
            x, y, z = axis.unbind(dim=-1)
            zero = torch.zeros_like(x)
            skew = torch.stack(
                (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
            ).reshape(batch, devices, 3, 3)
            identity = torch.eye(3, device=raw.device, dtype=torch.float32)
            cosine = torch.cos(angle)[..., None, None]
            sine = torch.sin(angle)[..., None, None]
            outer = axis[..., :, None] * axis[..., None, :]
            rotation = cosine * identity + (1.0 - cosine) * outer + sine * skew
            acceleration = torch.einsum(
                "bdij,bdnj->bdni", rotation, raw[..., :3].float()
            )
            gyro = torch.einsum(
                "bdij,bdnj->bdni", rotation, raw[..., 3:6].float()
            )
            augmented = torch.cat((acceleration, gyro), dim=-1).to(raw.dtype)
        if self.coordinate_dropout > 0.0:
            keep = (
                torch.rand(batch, devices, 3, device=raw.device)
                >= self.coordinate_dropout
            )
            empty = ~keep.any(dim=-1)
            if empty.any():
                replacement = torch.randint(
                    0, 3, (int(empty.sum().item()),), device=raw.device
                )
                row = torch.nonzero(empty, as_tuple=False)
                keep[row[:, 0], row[:, 1], replacement] = True
            keep = keep.to(augmented.dtype).unsqueeze(2)
            augmented = torch.cat(
                (augmented[..., :3] * keep, augmented[..., 3:6] * keep), dim=-1
            )
        return augmented

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        raw = batch["imu_raw_vectors"]
        if raw.ndim != 4 or raw.shape[1:] and raw.shape[-1] != 6:
            raise ValueError("imu_raw_vectors must have shape [B,5,N,6]")
        raw = self._augment_raw_coordinates(raw)
        batch_size, devices, points, _ = raw.shape
        frame_mask = batch["frame_mask"]
        time_steps = frame_mask.shape[1]
        point_mask = batch["imu_point_mask"] & batch["imu_device_mask"].unsqueeze(-1)
        scaled = raw / self.physical_scale.to(raw.dtype)
        difference = torch.zeros_like(scaled)
        difference_mask = point_mask.clone()
        if points > 1:
            difference[:, :, 1:] = scaled[:, :, 1:] - scaled[:, :, :-1]
            difference_mask[:, :, 0] = False
            difference_mask[:, :, 1:] &= point_mask[:, :, :-1]
        difference = difference * difference_mask.unsqueeze(-1)
        delta_time = torch.zeros_like(batch["imu_time_seconds"])
        if points > 1:
            delta_time[:, :, 1:] = (
                batch["imu_time_seconds"][:, :, 1:]
                - batch["imu_time_seconds"][:, :, :-1]
            )
        delta_time = torch.log1p((delta_time / 0.01).clamp(0.0, 100.0))
        point_features = torch.cat(
            (
                scaled,
                difference,
                torch.linalg.vector_norm(scaled[..., :3], dim=-1, keepdim=True),
                torch.linalg.vector_norm(scaled[..., 3:6], dim=-1, keepdim=True),
                delta_time.unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded = self.point_project(point_features) * point_mask.unsqueeze(-1)

        segments = batch_size * devices * time_steps
        frame_index = batch["imu_frame_index"].clamp(0, time_steps - 1)
        batch_device = (
            torch.arange(batch_size * devices, device=raw.device)
            .unsqueeze(1)
            .expand(-1, points)
            .reshape(-1)
        )
        segment = batch_device * time_steps + frame_index.reshape(-1)
        flat_mask = point_mask.reshape(-1)
        valid_segment = segment[flat_mask]
        # Scatter reductions accumulate in fp32 even under bf16/fp16 autocast.
        # PyTorch requires source and destination dtypes to match exactly here.
        valid_encoded = encoded.reshape(-1, self.width)[flat_mask].float()
        valid_raw = scaled.reshape(-1, 6)[flat_mask]

        count = raw.new_zeros(segments)
        count.index_add_(0, valid_segment, count.new_ones(len(valid_segment)))
        summed = raw.new_zeros(segments, self.width)
        summed.index_add_(0, valid_segment, valid_encoded)
        mean = summed / count.clamp_min(1.0).unsqueeze(-1)
        expanded = valid_segment.unsqueeze(-1).expand(-1, self.width)
        maximum = raw.new_full((segments, self.width), -1e4)
        maximum.scatter_reduce_(
            0, expanded, valid_encoded, reduce="amax", include_self=True
        )
        interval_valid = count > 0
        maximum = torch.where(interval_valid.unsqueeze(-1), maximum, 0.0)

        squared = raw.new_zeros(segments, 6)
        squared.index_add_(0, valid_segment, valid_raw.square())
        rms = torch.sqrt(squared / count.clamp_min(1.0).unsqueeze(-1) + 1e-8)
        raw_expanded = valid_segment.unsqueeze(-1).expand(-1, 6)
        peak = raw.new_zeros(segments, 6)
        peak.scatter_reduce_(
            0, raw_expanded, valid_raw.abs(), reduce="amax", include_self=True
        )
        pooled = torch.cat(
            (mean, maximum, rms, peak, torch.log1p(count).unsqueeze(-1)), dim=-1
        )
        device_token = self.interval_project(pooled).reshape(
            batch_size, devices, time_steps, self.width
        )
        device_token = device_token.permute(0, 2, 1, 3).contiguous()
        interval_valid = interval_valid.reshape(batch_size, devices, time_steps)
        interval_valid = interval_valid.permute(0, 2, 1).contiguous()
        interval_valid &= frame_mask.unsqueeze(-1)
        device_token = (
            device_token + self.device_embedding[None, None]
        ) * interval_valid.unsqueeze(-1)

        common_token = raw.new_zeros(
            batch_size, time_steps, len(COMMON_PART_NAMES), self.width
        )
        common_mask = torch.zeros(
            batch_size,
            time_steps,
            len(COMMON_PART_NAMES),
            dtype=torch.bool,
            device=raw.device,
        )
        for device_index, part_index in enumerate(self.DIRECT_COMMON_PART_INDEX):
            common_token[:, :, part_index] = device_token[:, :, device_index]
            common_mask[:, :, part_index] = interval_valid[:, :, device_index]
        global_count = interval_valid.to(raw.dtype).sum(dim=2)
        common_token[:, :, 0] = device_token.sum(dim=2) / global_count.clamp_min(
            1.0
        ).unsqueeze(-1)
        common_mask[:, :, 0] = global_count > 0
        arm_count = interval_valid[:, :, 1:3].to(raw.dtype).sum(dim=2)
        common_token[:, :, 7] = device_token[:, :, 1:3].sum(dim=2) / arm_count.clamp_min(
            1.0
        ).unsqueeze(-1)
        common_mask[:, :, 7] = arm_count > 0
        common_token *= common_mask.unsqueeze(-1)
        event_token, event_mask, _ = MotionAdapter._expand_base(
            common_token,
            common_mask,
            common_mask.to(raw.dtype),
        )
        return {
            "event_tokens": event_token,
            "event_mask": event_mask,
            "interval_mask": interval_valid,
        }


class UnifiedMotionAdapter(MotionAdapter):
    def __init__(
        self,
        width: int,
        dropout: float,
        *,
        raw_axis_rotation_degrees: float = 0.0,
        raw_coordinate_dropout: float = 0.0,
    ) -> None:
        super().__init__(width=width, dropout=dropout)
        self.raw = RawIMUResidualEncoder(
            width=width,
            dropout=dropout,
            axis_rotation_degrees=raw_axis_rotation_degrees,
            coordinate_dropout=raw_coordinate_dropout,
        )
        self.dual_coordinate_fusion = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dual_coordinate_norm = nn.LayerNorm(width)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = super().forward(batch)
        raw = self.raw(batch)
        raw_mask = raw["event_mask"] & output["imu_mask"]
        correction = self.dual_coordinate_fusion(
            torch.cat((output["imu_tokens"], raw["event_tokens"]), dim=-1)
        )
        imu_token = self.dual_coordinate_norm(
            output["imu_tokens"] + correction * raw_mask.unsqueeze(-1)
        )
        imu_token *= output["imu_mask"].unsqueeze(-1)
        output["imu_tokens"] = imu_token
        output["imu_motion"] = self._token_motion(imu_token, output["imu_mask"])
        output["raw_imu_event_tokens"] = raw["event_tokens"]
        output["raw_imu_event_mask"] = raw_mask
        return output


class GeometryAwareLocalVisualEncoder(LocalVisualObjectEncoder):
    """Inject explicit ROI pose geometry and provenance into local visual tokens."""

    REGION_TO_EVENT = (3, 4, 5, 6, 9)

    def __init__(
        self, width: int, dropout: float, *, torso_relative_geometry: bool = False
    ) -> None:
        super().__init__(width=width, dropout=dropout)
        self.torso_relative_geometry_enabled = bool(torso_relative_geometry)
        # current geometry (6), temporal delta (6), and five reliability scalars
        self.roi_geometry_project = nn.Sequential(
            nn.LayerNorm(17),
            nn.Linear(17, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.roi_source_embedding = nn.Embedding(7, width)
        self.roi_metadata_norm = nn.LayerNorm(width)
        self.roi_reliability = nn.Sequential(
            nn.Linear(5, width // 4),
            nn.GELU(),
            nn.Linear(width // 4, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def torso_relative_geometry(
        geometry: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Remove absolute camera position and apparent person-size shortcuts."""

        centres = geometry[..., :2]
        sizes = geometry[..., 2:4]
        valid_float = valid.to(geometry.dtype)
        arm_valid = valid_float[..., :2]
        arm_count = arm_valid.sum(dim=-1, keepdim=True)
        arm_centre = (
            centres[..., :2, :] * arm_valid.unsqueeze(-1)
        ).sum(dim=-2) / arm_count.clamp_min(1.0)
        all_count = valid_float.sum(dim=-1, keepdim=True)
        fallback_centre = (
            centres * valid_float.unsqueeze(-1)
        ).sum(dim=-2) / all_count.clamp_min(1.0)
        reference = torch.where(arm_count > 0, arm_centre, fallback_centre)

        arm_separation = torch.linalg.vector_norm(
            centres[..., 0, :] - centres[..., 1, :], dim=-1, keepdim=True
        )
        both_arms = valid[..., 0] & valid[..., 1]
        box_scale = torch.linalg.vector_norm(sizes, dim=-1)
        mean_box_scale = (box_scale * valid_float).sum(dim=-1, keepdim=True) / (
            all_count.clamp_min(1.0)
        )
        scale = torch.where(
            both_arms.unsqueeze(-1),
            arm_separation + 0.5 * mean_box_scale,
            2.0 * mean_box_scale,
        ).clamp_min(0.04)
        relative_centre = (
            (centres - reference.unsqueeze(-2)) / scale.unsqueeze(-2)
        ).clamp(-4.0, 4.0)
        relative_size = (sizes / scale.unsqueeze(-2)).clamp(0.0, 4.0)
        relative = torch.cat(
            (relative_centre, relative_size, geometry[..., 4:6]), dim=-1
        )
        return relative * valid_float.unsqueeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = super().forward(batch)
        geometry = batch["oriented_roi_geometry"]
        if self.torso_relative_geometry_enabled:
            geometry = self.torso_relative_geometry(
                geometry,
                batch["local_roi_valid"] & batch["frame_mask"].unsqueeze(-1),
            )
        geometry_delta = _temporal_difference(geometry)
        angle_valid = batch["oriented_angle_valid"].to(geometry.dtype)
        roi_valid = batch["local_roi_valid"].to(geometry.dtype)
        unclipped = 1.0 - batch["local_roi_clipped_ratio"].clamp(0.0, 1.0)
        pose_quality = batch["pose_quality_factor"].unsqueeze(-1).expand_as(roi_valid)
        quality = batch["local_roi_quality"].clamp(0.0, 1.0)
        reliability_values = torch.stack(
            (quality, unclipped, pose_quality, angle_valid, roi_valid), dim=-1
        )
        metadata_values = torch.cat(
            (geometry, geometry_delta, reliability_values), dim=-1
        )
        source = batch["local_roi_source"].clamp(0, 6)
        metadata = self.roi_metadata_norm(
            self.roi_geometry_project(metadata_values)
            + self.roi_source_embedding(source)
        )
        reliability = self.roi_reliability(reliability_values).squeeze(-1)
        reliability = reliability * roi_valid * batch["frame_mask"].unsqueeze(-1)

        event_metadata = metadata.new_zeros(
            *metadata.shape[:2], len(EVENT_PART_NAMES), metadata.shape[-1]
        )
        event_reliability = reliability.new_zeros(
            *reliability.shape[:2], len(EVENT_PART_NAMES)
        )
        for region, event_part in enumerate(self.REGION_TO_EVENT):
            event_metadata[:, :, event_part] = metadata[:, :, region]
            event_reliability[:, :, event_part] = reliability[:, :, region]
        part_sources = output["part_sources"] + (
            event_metadata.unsqueeze(3)
            * event_reliability.unsqueeze(-1).unsqueeze(-1)
            * output["part_source_mask"].unsqueeze(-1)
        )
        visual_quality = output["visual_quality"]
        local = event_reliability > 0
        visual_quality = torch.where(
            local,
            0.5 * (visual_quality + event_reliability),
            visual_quality,
        )
        return {
            **output,
            "part_sources": part_sources,
            "visual_quality": visual_quality.clamp(0.0, 1.0),
            "roi_metadata_embedding": metadata,
            "roi_reliability": reliability,
        }


class TimeAwareCompleteEventTemporalEncoder(CompleteEventTemporalEncoder):
    """Give the sole long temporal trunk explicit phase, cadence, and duration."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__(width=width, dropout=dropout)
        if width % 2:
            raise ValueError("time-aware temporal width must be even")
        frequencies = torch.pow(2.0, torch.linspace(0.0, 7.0, width // 2)) * math.pi
        self.register_buffer("time_frequencies", frequencies)
        self.time_scalar_project = nn.Sequential(
            nn.Linear(3, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_norm = nn.LayerNorm(width)

    def add_time_context(
        self,
        frame_sequence: torch.Tensor,
        frame_mask: torch.Tensor,
        time_position: torch.Tensor | None,
        frame_time_seconds: torch.Tensor | None,
    ) -> torch.Tensor:
        if time_position is None or frame_time_seconds is None:
            raise KeyError(
                "unified P46 requires time_position and frame_time_seconds"
            )
        position = time_position.to(frame_sequence.dtype)
        angle = position.unsqueeze(-1) * self.time_frequencies.to(frame_sequence.dtype)
        fourier = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
        frame_time = frame_time_seconds.to(frame_sequence.dtype)
        delta = torch.zeros_like(frame_time)
        if frame_time.shape[1] > 1:
            delta[:, 1:] = (frame_time[:, 1:] - frame_time[:, :-1]).clamp_min(0.0)
        last_index = frame_mask.sum(dim=1).clamp_min(1) - 1
        last_time = frame_time.gather(1, last_index.unsqueeze(1)).squeeze(1)
        first_time = frame_time[:, 0]
        duration = (last_time - first_time).clamp_min(0.0)
        scalars = torch.stack(
            (
                position,
                torch.log1p((delta / 0.04).clamp(0.0, 100.0)),
                torch.log1p(duration).unsqueeze(1).expand_as(position),
            ),
            dim=-1,
        )
        context = fourier + self.time_scalar_project(scalars)
        return self.time_norm(frame_sequence + context) * frame_mask.unsqueeze(-1)


class SharedSameTimeRelationshipRefiner(nn.Module):
    """Explicit same-part/same-frame relation learning on shared P46 tokens."""

    def __init__(
        self,
        width: int,
        dropout: float,
        *,
        hidden_width: int | None = None,
        maximum_residual_scale: float | None = None,
        initial_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        hidden_width = int(hidden_width or width * 2)
        self.relationship_project = nn.Sequential(
            nn.LayerNorm(width * 6 + 4),
            nn.Linear(width * 6 + 4, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, width),
        )
        self.temporal = PartTemporalBlock(width, dilation=1, dropout=dropout)
        self.output_norm = nn.LayerNorm(width)
        self.maximum_residual_scale = maximum_residual_scale
        if maximum_residual_scale is not None:
            if not 0.0 < initial_residual_scale < maximum_residual_scale:
                raise ValueError("initial relationship scale must lie inside (0, maximum)")
            probability = initial_residual_scale / maximum_residual_scale
            initial_logit = math.log(probability / (1.0 - probability))
            self.residual_scale_logits = nn.Parameter(
                torch.full((len(EVENT_PART_NAMES),), initial_logit)
            )
        self.localizer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.motion_sync = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 64, bias=False)
        )
        self.visual_sync = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 64, bias=False)
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(len(P46R_OFFSET_FRACTIONS) * len(P46R_MOTION_PART_INDICES)),
            nn.Linear(
                len(P46R_OFFSET_FRACTIONS) * len(P46R_MOTION_PART_INDICES), 64
            ),
            nn.GELU(),
            nn.Linear(64, len(P46R_OFFSET_FRACTIONS)),
        )

    @staticmethod
    def _lag_correlations(
        motion: torch.Tensor,
        visual: torch.Tensor,
        motion_valid: torch.Tensor,
        visual_valid: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = motion.shape[0]
        correlations: list[torch.Tensor] = []
        for fraction in P46R_OFFSET_FRACTIONS:
            rows: list[torch.Tensor] = []
            for row in range(batch):
                length = int(frame_mask[row].sum().item())
                shift = int(round(fraction * length))
                if fraction != 0.0 and shift == 0 and length > 1:
                    shift = 1 if fraction > 0 else -1
                shifted_visual = torch.roll(visual[row, :length], -shift, dims=0)
                shifted_valid = torch.roll(visual_valid[row, :length], -shift, dims=0)
                pair = motion_valid[row, :length] & shifted_valid
                similarity = (motion[row, :length] * shifted_visual).sum(dim=-1)
                weight = pair.to(similarity.dtype)
                rows.append(
                    (similarity * weight).sum(dim=0)
                    / weight.sum(dim=0).clamp_min(1.0)
                )
            correlations.append(torch.stack(rows, dim=0))
        return torch.stack(correlations, dim=1).reshape(batch, -1)

    def synthetic_offset_logits(
        self,
        shared: dict[str, torch.Tensor],
        offset_index: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Train offset supervision by shifting already encoded visual tokens.

        Motion, visual, and relation encoders are each executed only once.  The
        synthetic mismatch is created at the shared-token level, preventing a
        hidden second feature-extraction branch and avoiding duplicate compute.
        """

        visual = shared["relationship_visual_sync"].clone()
        visual_mask = shared["relationship_visual_mask"].clone()
        for row, label in enumerate(offset_index.tolist()):
            length = int(frame_mask[row].sum().item())
            fraction = float(P46R_OFFSET_FRACTIONS[label])
            shift = int(round(fraction * length))
            if fraction != 0.0 and shift == 0 and length > 1:
                shift = 1 if fraction > 0 else -1
            if length > 1 and shift:
                visual[row, :length] = torch.roll(
                    visual[row, :length], shift, dims=0
                )
                visual_mask[row, :length] = torch.roll(
                    visual_mask[row, :length], shift, dims=0
                )
        correlations = self._lag_correlations(
            shared["relationship_motion_sync"],
            visual,
            shared["relationship_motion_mask"],
            visual_mask,
            frame_mask,
        )
        return self.offset_head(correlations)

    def forward(
        self,
        fusion: dict[str, torch.Tensor],
        motion: dict[str, torch.Tensor],
        visual: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        motion_token = fusion["motion_query"]
        visual_token = fusion["attended_visual"]
        motion_delta = _temporal_difference(motion_token)
        visual_delta = _temporal_difference(visual_token)
        time_position = batch["time_position"].to(motion_token.dtype)
        frame_time = batch["frame_time_seconds"].to(motion_token.dtype)
        delta_time = torch.zeros_like(frame_time)
        if frame_time.shape[1] > 1:
            delta_time[:, 1:] = (frame_time[:, 1:] - frame_time[:, :-1]).clamp_min(0.0)
        scalar = torch.stack(
            (
                time_position.unsqueeze(-1).expand_as(fusion["soft_event_gate"]),
                torch.log1p((delta_time / 0.04).clamp(0.0, 100.0))
                .unsqueeze(-1)
                .expand_as(fusion["soft_event_gate"]),
                fusion["soft_event_gate"],
                visual["contact_proxy"],
            ),
            dim=-1,
        )
        relationship = self.relationship_project(
            torch.cat(
                (
                    motion_token,
                    visual_token,
                    motion_token * visual_token,
                    (motion_token - visual_token).abs(),
                    motion_delta,
                    visual_delta,
                    scalar,
                ),
                dim=-1,
            )
        )
        visual_mask = visual["part_source_mask"][..., 1:].any(dim=-1)
        motion_mask = motion["skeleton_mask"] | motion["imu_mask"]
        same_time_mask = fusion["event_mask"] & motion_mask & visual_mask
        local_mask = torch.zeros_like(same_time_mask)
        local_mask[:, :, list(P46R_MOTION_PART_INDICES)] = same_time_mask[
            :, :, list(P46R_MOTION_PART_INDICES)
        ]
        relationship *= local_mask.unsqueeze(-1)
        relationship = self.temporal(relationship, local_mask)
        if self.maximum_residual_scale is None:
            residual_scale = relationship.new_ones(len(EVENT_PART_NAMES))
        else:
            residual_scale = self.maximum_residual_scale * torch.sigmoid(
                self.residual_scale_logits
            )
        refined = self.output_norm(
            fusion["event_tokens"]
            + relationship * residual_scale[None, None, :, None]
        ) * fusion["event_mask"].unsqueeze(-1)

        index = torch.tensor(P46R_MOTION_PART_INDICES, device=refined.device)
        selected_relation = relationship.index_select(2, index)
        selected_mask = local_mask.index_select(2, index)
        selected_motion_mask = motion_mask.index_select(2, index)
        selected_visual_mask = visual_mask.index_select(2, index)
        localizer_logits = self.localizer(selected_relation).squeeze(-1)
        localizer_logits = localizer_logits.masked_fill(~selected_mask, -1e4)
        energy = (
            torch.log1p(motion["skeleton_motion"].clamp_min(0.0))
            + torch.log1p(motion["imu_motion"].clamp_min(0.0))
            + torch.log1p(visual["visual_motion"].clamp_min(0.0))
        ).index_select(2, index)
        evidence_target = _masked_distribution(energy / 0.35, selected_mask).detach()

        sync_motion = F.normalize(
            self.motion_sync(motion_token.index_select(2, index)).float(),
            dim=-1,
            eps=1e-6,
        )
        sync_visual = F.normalize(
            self.visual_sync(visual_token.index_select(2, index)).float(),
            dim=-1,
            eps=1e-6,
        )
        offset_correlation = self._lag_correlations(
            sync_motion,
            sync_visual,
            selected_motion_mask,
            selected_visual_mask,
            batch["frame_mask"],
        )
        return {
            **fusion,
            "event_tokens": refined,
            "relationship_sequence": relationship,
            "relationship_residual_scale": residual_scale,
            "relationship_mask": local_mask,
            "relationship_event_localizer_logits": localizer_logits,
            "relationship_evidence_target": evidence_target,
            "relationship_evidence_mask": selected_mask,
            "relationship_offset_correlation": offset_correlation,
            "relationship_offset_logits": self.offset_head(offset_correlation),
            "relationship_motion_sync": sync_motion,
            "relationship_visual_sync": sync_visual,
            "relationship_motion_mask": selected_motion_mask,
            "relationship_visual_mask": selected_visual_mask,
        }


class P46UnifiedEventTokenEncoder(nn.Module):
    """One motion encoder, one visual encoder, one relation/temporal trunk."""

    def __init__(
        self,
        width: int = 192,
        dropout: float = 0.12,
        *,
        torso_relative_geometry: bool = False,
        raw_axis_rotation_degrees: float = 0.0,
        raw_coordinate_dropout: float = 0.0,
        relationship_hidden_width: int | None = None,
        relationship_maximum_scale: float | None = None,
        relationship_initial_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.motion = UnifiedMotionAdapter(
            width=width,
            dropout=dropout,
            raw_axis_rotation_degrees=raw_axis_rotation_degrees,
            raw_coordinate_dropout=raw_coordinate_dropout,
        )
        self.visual = GeometryAwareLocalVisualEncoder(
            width=width,
            dropout=dropout,
            torso_relative_geometry=torso_relative_geometry,
        )
        self.fusion = SamePartEventFusion(width=width, dropout=dropout)
        self.relationship = SharedSameTimeRelationshipRefiner(
            width=width,
            dropout=dropout,
            hidden_width=relationship_hidden_width,
            maximum_residual_scale=relationship_maximum_scale,
            initial_residual_scale=relationship_initial_scale,
        )
        self.temporal = TimeAwareCompleteEventTemporalEncoder(
            width=width, dropout=dropout
        )

    def encode_relationship(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        motion = self.motion(batch)
        visual = self.visual(batch)
        fusion = self.fusion(motion, visual, batch["frame_mask"])
        relationship = self.relationship(fusion, motion, visual, batch)
        return {
            **relationship,
            "contact_proxy": visual["contact_proxy"],
            "skeleton_motion": motion["skeleton_motion"],
            "imu_motion": motion["imu_motion"],
            "visual_motion": visual["visual_motion"],
            "imu_interval_count": motion["imu_interval_count"],
            "raw_imu_event_tokens": motion["raw_imu_event_tokens"],
            "raw_imu_event_mask": motion["raw_imu_event_mask"],
            "roi_metadata_embedding": visual["roi_metadata_embedding"],
            "roi_reliability": visual["roi_reliability"],
            "left_object_token": visual["left_object_token"],
            "right_object_token": visual["right_object_token"],
            "surface_token": visual["surface_token"],
            "left_object_quality": visual["left_object_quality"],
            "right_object_quality": visual["right_object_quality"],
            "surface_quality": visual["surface_quality"],
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        relationship = self.encode_relationship(batch)
        temporal = self.temporal(
            relationship["event_tokens"],
            relationship["event_mask"],
            relationship["soft_event_gate"],
            relationship["contact_proxy"],
            batch["frame_mask"],
            batch["time_position"],
            batch["frame_time_seconds"],
        )
        return {
            **relationship,
            **temporal,
            "event_part_names": EVENT_PART_NAMES,
        }


class P46UnifiedRepairModel(P46Step10Model):
    """P46 repaired inside one shared input/relationship/temporal trunk."""

    def __init__(
        self,
        *,
        width: int = 192,
        dropout: float = 0.12,
        subjects: int = 14,
        **encoder_kwargs: object,
    ) -> None:
        super().__init__(width=width, dropout=dropout, subjects=subjects)
        self.encoder = P46UnifiedEventTokenEncoder(
            width=width, dropout=dropout, **encoder_kwargs
        )

    def forward_relationship(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self.encoder.encode_relationship(batch)

    def synthetic_offset_logits(
        self,
        output: dict[str, torch.Tensor],
        offset_index: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder.relationship.synthetic_offset_logits(
            output, offset_index, frame_mask
        )


def relationship_localization_loss(
    output: dict[str, torch.Tensor],
    *,
    confidence_weighted: bool = False,
) -> torch.Tensor:
    target = output["relationship_evidence_target"]
    mask = output["relationship_evidence_mask"]
    logits = output["relationship_event_localizer_logits"].float().masked_fill(
        ~mask, -1e4
    )
    log_probability = F.log_softmax(logits, dim=1)
    valid_part = mask.any(dim=1)
    loss = -(target * log_probability).sum(dim=1)
    if confidence_weighted:
        count = mask.sum(dim=1).clamp_min(1).to(target.dtype)
        entropy = -(target * torch.log(target.clamp_min(1e-8))).sum(dim=1)
        confidence = (
            1.0 - entropy / torch.log(count).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        loss = loss * confidence.detach()
    if not valid_part.any():
        return output["trial_embedding"].sum() * 0.0
    return loss[valid_part].mean()


class P46UnifiedRepairV2Model(P46UnifiedRepairModel):
    """Capacity-controlled unified repair with shortcut-resistant inputs."""

    def __init__(
        self,
        *,
        width: int = 168,
        dropout: float = 0.18,
        subjects: int = 14,
        raw_axis_rotation_degrees: float = 20.0,
        raw_coordinate_dropout: float = 0.10,
        relationship_maximum_scale: float = 0.25,
        relationship_initial_scale: float = 0.05,
    ) -> None:
        super().__init__(
            width=width,
            dropout=dropout,
            subjects=subjects,
            torso_relative_geometry=True,
            raw_axis_rotation_degrees=raw_axis_rotation_degrees,
            raw_coordinate_dropout=raw_coordinate_dropout,
            relationship_hidden_width=width,
            relationship_maximum_scale=relationship_maximum_scale,
            relationship_initial_scale=relationship_initial_scale,
        )


class P46UnifiedRepairV3Model(P46UnifiedRepairModel):
    """Clean unified trunk: full inputs with moderate generic regularisation."""

    def __init__(
        self,
        *,
        width: int = 180,
        dropout: float = 0.14,
        subjects: int = 14,
        raw_axis_rotation_degrees: float = 10.0,
        raw_coordinate_dropout: float = 0.05,
        relationship_maximum_scale: float = 0.30,
        relationship_initial_scale: float = 0.10,
    ) -> None:
        super().__init__(
            width=width,
            dropout=dropout,
            subjects=subjects,
            torso_relative_geometry=True,
            raw_axis_rotation_degrees=raw_axis_rotation_degrees,
            raw_coordinate_dropout=raw_coordinate_dropout,
            relationship_hidden_width=width,
            relationship_maximum_scale=relationship_maximum_scale,
            relationship_initial_scale=relationship_initial_scale,
        )


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
