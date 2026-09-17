from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from p46_event_model import MotionAdapter


P46R_PART_NAMES = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "hand_workspace",
)
P46R_MOTION_PART_INDICES = (3, 4, 5, 6, 9)
P46R_OFFSET_FRACTIONS = (-0.25, -0.125, 0.0, 0.125, 0.25)
P46R_VISUAL_KEYS = (
    "arm_spatial_features",
    "detail_spatial_features",
    "local_geometry_features",
    "oriented_roi_geometry",
    "oriented_angle_valid",
    "local_roi_valid",
    "local_roi_quality",
    "local_roi_source",
    "local_roi_clipped_ratio",
    "pose_quality_factor",
    "context_features",
    "context_valid",
    "context_quality",
)


def _temporal_difference(source: torch.Tensor) -> torch.Tensor:
    difference = torch.zeros_like(source)
    if source.shape[1] > 1:
        difference[:, 1:] = (source[:, 1:] - source[:, :-1]).abs()
    return difference


def _masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe_mask = mask.clone()
    empty = ~safe_mask.any(dim=1)
    if empty.any():
        safe_mask[empty, 0] = True
    probability = torch.softmax(logits.masked_fill(~safe_mask, -1e4), dim=1)
    probability = probability * mask
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _normalise_energy(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    mean = (value * weight).sum(dim=1, keepdim=True) / weight.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    return (torch.log1p(value.clamp_min(0.0)) / torch.log1p(mean.clamp_min(1e-4))).clamp(
        0.0, 5.0
    ) * weight


def circular_shift_visual_batch(
    batch: dict[str, Any],
    offset_index: torch.Tensor,
    fractions: tuple[float, ...] = P46R_OFFSET_FRACTIONS,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Circularly move visual evidence inside each valid trial.

    Circular movement is intentional: zero-filled edges would make offset prediction
    solvable without comparing visual and motion evidence.
    """

    output: dict[str, Any] = dict(batch)
    frame_mask = batch["frame_mask"]
    shifts = torch.zeros(len(offset_index), dtype=torch.long, device=offset_index.device)
    for row in range(len(offset_index)):
        length = int(frame_mask[row].sum().item())
        fraction = float(fractions[int(offset_index[row].item())])
        shift = int(round(fraction * length))
        if fraction != 0.0 and shift == 0 and length > 1:
            shift = 1 if fraction > 0 else -1
        shifts[row] = shift

    for key in P46R_VISUAL_KEYS:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            continue
        shifted = value.clone()
        for row, shift in enumerate(shifts.tolist()):
            length = int(frame_mask[row].sum().item())
            if length > 1 and shift:
                shifted[row, :length] = torch.roll(value[row, :length], shift, dims=0)
        output[key] = shifted
    return output, shifts


class P46RVisualEvidenceEncoder(nn.Module):
    """Encode only body-centred local D/IR regions; no global scene bypass."""

    def __init__(self, width: int = 128, dropout: float = 0.12) -> None:
        super().__init__()
        # current D+IR (256), temporal D+IR difference (256), current/delta
        # six-channel geometry (12), and four quality values.
        self.project = nn.Sequential(
            nn.LayerNorm(528),
            nn.Linear(528, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.part_embedding = nn.Parameter(
            torch.randn(len(P46R_PART_NAMES), width) / math.sqrt(width)
        )
        self.output_norm = nn.LayerNorm(width)

    @staticmethod
    def _pool_modalities(features: torch.Tensor) -> torch.Tensor:
        # [B,T,modality,region,H,W,C] -> [B,T,region,modality*C]
        pooled = features.mean(dim=(4, 5)).permute(0, 1, 3, 2, 4)
        return pooled.flatten(start_dim=3)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        current_scale: float = 1.0,
        difference_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        arm = self._pool_modalities(batch["arm_spatial_features"])
        detail = self._pool_modalities(batch["detail_spatial_features"])
        current = torch.cat((arm, detail), dim=2)
        difference = _temporal_difference(current)

        geometry = batch["local_geometry_features"].mean(dim=(3, 4))
        geometry_difference = _temporal_difference(geometry)
        quality = torch.stack(
            (
                batch["local_roi_quality"],
                1.0 - batch["local_roi_clipped_ratio"].clamp(0.0, 1.0),
                batch["pose_quality_factor"].unsqueeze(-1).expand(-1, -1, 5),
                batch["local_roi_valid"].to(current.dtype),
            ),
            dim=-1,
        )
        source = torch.cat(
            (
                current * current_scale,
                difference * difference_scale,
                geometry * current_scale,
                geometry_difference * difference_scale,
                quality,
            ),
            dim=-1,
        )
        valid = batch["local_roi_valid"] & batch["frame_mask"].unsqueeze(-1)
        token = self.output_norm(self.project(source) + self.part_embedding[None, None])
        token = token * valid.unsqueeze(-1)
        modality_difference = difference.reshape(*difference.shape[:3], 2, 128)
        visual_motion = modality_difference.square().mean(dim=(-1, -2)).sqrt()
        visual_motion = visual_motion * difference_scale * valid
        return {
            "visual_token": token,
            "visual_mask": valid,
            "visual_motion": visual_motion,
            "visual_quality": quality[..., :3].mean(dim=-1) * valid,
        }


class P46RMotionEvidenceEncoder(nn.Module):
    def __init__(self, width: int = 128, dropout: float = 0.12) -> None:
        super().__init__()
        self.base = MotionAdapter(width=width, dropout=dropout)
        self.project = nn.Sequential(
            nn.LayerNorm(width * 3 + 4),
            nn.Linear(width * 3 + 4, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        skeleton_scale: float = 1.0,
        imu_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        base = self.base(batch)
        index = torch.tensor(P46R_MOTION_PART_INDICES, device=batch["frame_mask"].device)
        skeleton = base["skeleton_tokens"].index_select(2, index) * skeleton_scale
        imu = base["imu_tokens"].index_select(2, index) * imu_scale
        skeleton_mask = base["skeleton_mask"].index_select(2, index)
        imu_mask = base["imu_mask"].index_select(2, index)
        if skeleton_scale == 0.0:
            skeleton_mask = torch.zeros_like(skeleton_mask)
        if imu_scale == 0.0:
            imu_mask = torch.zeros_like(imu_mask)
        skeleton_quality = base["skeleton_quality"].index_select(2, index) * skeleton_mask
        imu_quality = base["imu_quality"].index_select(2, index) * imu_mask
        skeleton_motion = (
            base["skeleton_motion"].index_select(2, index) * skeleton_scale * skeleton_mask
        )
        imu_motion = base["imu_motion"].index_select(2, index) * imu_scale * imu_mask
        pair_difference = (skeleton - imu).abs()
        scalars = torch.stack(
            (
                skeleton_quality,
                imu_quality,
                torch.log1p(skeleton_motion.clamp_min(0.0)),
                torch.log1p(imu_motion.clamp_min(0.0)),
            ),
            dim=-1,
        )
        valid = (skeleton_mask | imu_mask) & batch["frame_mask"].unsqueeze(-1)
        token = self.output_norm(
            self.project(torch.cat((skeleton, imu, pair_difference, scalars), dim=-1))
        ) * valid.unsqueeze(-1)
        return {
            "motion_token": token,
            "motion_mask": valid,
            "skeleton_motion": skeleton_motion,
            "imu_motion": imu_motion,
            "skeleton_mask": skeleton_mask,
            "imu_mask": imu_mask,
        }


class P46RTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            groups=width,
            bias=False,
        )
        self.pointwise = nn.Conv1d(width, width * 2, kernel_size=1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, parts, width = source.shape
        flat = source.permute(0, 2, 3, 1).reshape(batch * parts, width, steps)
        temporal = self.pointwise(self.depthwise(flat)).transpose(1, 2)
        value, gate = temporal.chunk(2, dim=-1)
        temporal = value * torch.sigmoid(gate)
        temporal = temporal.reshape(batch, parts, steps, width).permute(0, 2, 1, 3)
        return self.norm(source + self.dropout(temporal)) * mask.unsqueeze(-1)


class P46REventBottleneck(nn.Module):
    """A classification bottleneck that exposes only ordered local event summaries."""

    def __init__(self, width: int = 128, dropout: float = 0.12) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.LayerNorm(width * 4 + 6),
            nn.Linear(width * 4 + 6, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.blocks = nn.ModuleList(
            P46RTemporalBlock(width, dilation, dropout) for dilation in (1, 2, 4)
        )
        self.event_localizer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1)
        )
        self.part_summary = nn.Sequential(
            nn.LayerNorm(width * 5),
            nn.Linear(width * 5, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embedding = nn.Sequential(
            nn.LayerNorm(width * len(P46R_PART_NAMES)),
            nn.Linear(width * len(P46R_PART_NAMES), 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _phase_weight(
        position: torch.Tensor,
        centre: torch.Tensor,
        offset: float,
        mask: torch.Tensor,
        width: float = 0.18,
    ) -> torch.Tensor:
        target = (centre + offset).clamp(0.0, 1.0)
        weight = torch.exp(-0.5 * ((position - target) / width).square()) * mask
        return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(
        self,
        motion: dict[str, torch.Tensor],
        visual: dict[str, torch.Tensor],
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        motion_token = motion["motion_token"]
        visual_token = visual["visual_token"]
        scalar = torch.stack(
            (
                torch.log1p(motion["skeleton_motion"].clamp_min(0.0)),
                torch.log1p(motion["imu_motion"].clamp_min(0.0)),
                torch.log1p(visual["visual_motion"].clamp_min(0.0)),
                motion["skeleton_mask"].to(motion_token.dtype),
                motion["imu_mask"].to(motion_token.dtype),
                visual["visual_quality"],
            ),
            dim=-1,
        )
        source = self.fusion(
            torch.cat(
                (
                    motion_token,
                    visual_token,
                    motion_token * visual_token,
                    (motion_token - visual_token).abs(),
                    scalar,
                ),
                dim=-1,
            )
        )
        mask = frame_mask.unsqueeze(-1) & (
            motion["motion_mask"] | visual["visual_mask"]
        )
        source = source * mask.unsqueeze(-1)
        for block in self.blocks:
            source = block(source, mask)

        localizer_logits = self.event_localizer(source).squeeze(-1)
        localizer_logits = localizer_logits.masked_fill(~mask, -1e4)
        during_weight = _masked_distribution(localizer_logits, mask)
        steps = source.shape[1]
        position = torch.linspace(
            0.0, 1.0, steps, device=source.device, dtype=source.dtype
        )[None, :, None]
        centre = (during_weight * position).sum(dim=1, keepdim=True)
        before_weight = self._phase_weight(position, centre, -0.20, mask)
        after_weight = self._phase_weight(position, centre, 0.20, mask)

        def pool(weight: torch.Tensor) -> torch.Tensor:
            return (source * weight.unsqueeze(-1)).sum(dim=1)

        before = pool(before_weight)
        during = pool(during_weight)
        after = pool(after_weight)
        part = self.part_summary(
            torch.cat((before, during, after, during - before, after - during), dim=-1)
        )
        embedding = self.embedding(part.flatten(start_dim=1))
        return {
            "embedding": embedding,
            "event_sequence": source,
            "event_mask": mask,
            "event_localizer_logits": localizer_logits,
            "event_during_weight": during_weight,
            "event_centre": centre.squeeze(1),
            "part_summary": part,
        }


class P46REventModel(nn.Module):
    def __init__(
        self,
        width: int = 128,
        dropout: float = 0.12,
        classes: int = 21,
    ) -> None:
        super().__init__()
        self.width = width
        self.motion = P46RMotionEvidenceEncoder(width=width, dropout=dropout)
        self.visual = P46RVisualEvidenceEncoder(width=width, dropout=dropout)
        self.bottleneck = P46REventBottleneck(width=width, dropout=dropout)
        self.detail_head = nn.Sequential(
            nn.LayerNorm(256), nn.Dropout(dropout), nn.Linear(256, classes)
        )
        self.contrast_head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128, bias=False)
        )
        self.motion_sync_project = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 64, bias=False)
        )
        self.visual_sync_project = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 64, bias=False)
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(len(P46R_OFFSET_FRACTIONS) * len(P46R_PART_NAMES)),
            nn.Linear(len(P46R_OFFSET_FRACTIONS) * len(P46R_PART_NAMES), 64),
            nn.GELU(),
            nn.Linear(64, len(P46R_OFFSET_FRACTIONS)),
        )

    @staticmethod
    def _lag_correlations(
        motion: torch.Tensor,
        visual: torch.Tensor,
        motion_mask: torch.Tensor,
        visual_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, parts, _ = motion.shape
        correlations: list[torch.Tensor] = []
        for fraction in P46R_OFFSET_FRACTIONS:
            per_sample: list[torch.Tensor] = []
            for row in range(batch):
                length = int(frame_mask[row].sum().item())
                shift = int(round(fraction * length))
                if fraction != 0.0 and shift == 0 and length > 1:
                    shift = 1 if fraction > 0 else -1
                shifted_visual = torch.roll(visual[row, :length], -shift, dims=0)
                shifted_mask = torch.roll(visual_mask[row, :length], -shift, dims=0)
                valid = motion_mask[row, :length] & shifted_mask
                similarity = (motion[row, :length] * shifted_visual).sum(dim=-1)
                weight = valid.to(similarity.dtype)
                value = (similarity * weight).sum(dim=0) / weight.sum(dim=0).clamp_min(1.0)
                per_sample.append(value)
            correlations.append(torch.stack(per_sample, dim=0))
        return torch.stack(correlations, dim=1).reshape(batch, -1)

    @staticmethod
    def _evidence_target(
        motion: dict[str, torch.Tensor], visual: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = (
            motion["skeleton_mask"],
            motion["imu_mask"],
            visual["visual_mask"],
        )
        values = (
            motion["skeleton_motion"],
            motion["imu_motion"],
            visual["visual_motion"],
        )
        numerator = values[0].new_zeros(values[0].shape)
        denominator = values[0].new_zeros(values[0].shape)
        for value, mask in zip(values, masks):
            numerator = numerator + _normalise_energy(value, mask)
            denominator = denominator + mask.to(value.dtype)
        evidence = numerator / denominator.clamp_min(1.0)
        valid = denominator > 0
        target = _masked_distribution(evidence / 0.35, valid).detach()
        return target, valid

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        visual_scale: float = 1.0,
        visual_difference_scale: float = 1.0,
        skeleton_scale: float = 1.0,
        imu_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        motion = self.motion(
            batch, skeleton_scale=skeleton_scale, imu_scale=imu_scale
        )
        visual = self.visual(
            batch,
            current_scale=visual_scale,
            difference_scale=visual_difference_scale,
        )
        if visual_scale == 0.0:
            visual["visual_mask"] = torch.zeros_like(visual["visual_mask"])
            visual["visual_quality"] = torch.zeros_like(visual["visual_quality"])
            visual["visual_motion"] = torch.zeros_like(visual["visual_motion"])
            visual["visual_token"] = torch.zeros_like(visual["visual_token"])
        event = self.bottleneck(motion, visual, batch["frame_mask"])
        motion_sync = F.normalize(
            self.motion_sync_project(motion["motion_token"]).float(), dim=-1, eps=1e-6
        )
        visual_sync = F.normalize(
            self.visual_sync_project(visual["visual_token"]).float(), dim=-1, eps=1e-6
        )
        correlation = self._lag_correlations(
            motion_sync,
            visual_sync,
            motion["motion_mask"],
            visual["visual_mask"],
            batch["frame_mask"],
        )
        evidence_target, evidence_mask = self._evidence_target(motion, visual)
        embedding = event["embedding"]
        return {
            **event,
            "detail_logits": self.detail_head(embedding),
            "offset_logits": self.offset_head(correlation),
            "offset_correlation": correlation,
            "contrast_embedding": F.normalize(
                self.contrast_head(embedding).float(), dim=-1, eps=1e-6
            ),
            "evidence_target": evidence_target,
            "evidence_mask": evidence_mask,
            "visual_motion": visual["visual_motion"],
            "skeleton_motion": motion["skeleton_motion"],
            "imu_motion": motion["imu_motion"],
        }


def event_localization_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    target = output["evidence_target"]
    mask = output["evidence_mask"]
    log_probability = F.log_softmax(
        output["event_localizer_logits"].float().masked_fill(~mask, -1e4), dim=1
    )
    valid_part = mask.any(dim=1)
    loss = -(target * log_probability).sum(dim=1)
    if not valid_part.any():
        return output["embedding"].sum() * 0.0
    return loss[valid_part].mean()


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())

