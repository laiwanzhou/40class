from __future__ import annotations

import math

import torch
from torch import nn

from p31_skeleton_imu_model import IMUIntervalPartEncoder, SkeletonPartEncoder
from p31_skeleton_imu_preprocessing import COMMON_PART_NAMES
from p46_event_preprocessing import P46_LOCAL_REGIONS


EVENT_PART_NAMES = (
    "global",
    "head",
    "torso",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "left_leg",
    "right_leg",
    "hand_workspace",
)
EVENT_PHASE_NAMES = ("approach", "contact", "manipulate", "release", "static_hold")


class MotionAdapter(nn.Module):
    """Expand P31 eight-part motion outputs to explicit ten-part event semantics."""

    BASE_TO_EVENT = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (6, 8), (7, 9))

    def __init__(self, width: int = 192, dropout: float = 0.12) -> None:
        super().__init__()
        self.width = width
        self.skeleton = SkeletonPartEncoder(output_width=width, dropout=dropout)
        self.imu = IMUIntervalPartEncoder(output_width=width, dropout=dropout)
        self.wrist_project = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, width), nn.GELU())
        self.orientation_project = nn.Sequential(
            nn.Linear(10, width), nn.GELU(), nn.Linear(width, width)
        )
        self.event_embedding = nn.Parameter(
            torch.randn(len(EVENT_PART_NAMES), width) / math.sqrt(width)
        )

    @staticmethod
    def _expand_base(
        values: torch.Tensor,
        masks: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (*values.shape[:2], len(EVENT_PART_NAMES), values.shape[-1])
        output = values.new_zeros(shape)
        output_mask = torch.zeros(
            *masks.shape[:2], len(EVENT_PART_NAMES), dtype=torch.bool, device=masks.device
        )
        output_quality = quality.new_zeros(*quality.shape[:2], len(EVENT_PART_NAMES))
        for base, event in MotionAdapter.BASE_TO_EVENT:
            output[:, :, event] = values[:, :, base]
            output_mask[:, :, event] = masks[:, :, base]
            output_quality[:, :, event] = quality[:, :, base]
        output[:, :, 5] = values[:, :, 3]
        output[:, :, 6] = values[:, :, 4]
        output_mask[:, :, 5] = masks[:, :, 3]
        output_mask[:, :, 6] = masks[:, :, 4]
        output_quality[:, :, 5] = quality[:, :, 3]
        output_quality[:, :, 6] = quality[:, :, 4]
        return output, output_mask, output_quality

    @staticmethod
    def _token_motion(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        energy = tokens.new_zeros(mask.shape)
        if tokens.shape[1] > 1:
            pair = mask[:, 1:] & mask[:, :-1]
            difference = torch.linalg.vector_norm(tokens[:, 1:] - tokens[:, :-1], dim=-1)
            energy[:, 1:] = difference / math.sqrt(tokens.shape[-1]) * pair
        return energy

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
        skeleton_token, skeleton_mask, skeleton_quality = self._expand_base(
            skeleton["part_tokens"], skeleton["part_mask"], skeleton["part_quality"]
        )
        imu_token, imu_mask, imu_quality = self._expand_base(
            imu["part_tokens"], imu["part_mask"], imu["part_quality"]
        )
        wrist_source = skeleton["joint_sequence"]
        left_wrist = self.wrist_project(wrist_source[:, :, 13])
        right_wrist = self.wrist_project(wrist_source[:, :, 16])
        left_valid = batch["skeleton_joint_mask"][:, :, 13] & batch["frame_mask"]
        right_valid = batch["skeleton_joint_mask"][:, :, 16] & batch["frame_mask"]
        skeleton_token[:, :, 5] = skeleton_token[:, :, 5] + left_wrist * left_valid.unsqueeze(-1)
        skeleton_token[:, :, 6] = skeleton_token[:, :, 6] + right_wrist * right_valid.unsqueeze(-1)
        skeleton_mask[:, :, 5] &= left_valid
        skeleton_mask[:, :, 6] &= right_valid

        orientation_input = torch.cat(
            (
                batch["body_axes_camera"].flatten(start_dim=2),
                batch["body_axes_raw_valid"].to(batch["body_axes_camera"].dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        orientation = self.orientation_project(orientation_input)
        skeleton_token = skeleton_token + orientation.unsqueeze(2)
        skeleton_token = skeleton_token + self.event_embedding[None, None]
        imu_token = imu_token + self.event_embedding[None, None]
        skeleton_token = skeleton_token * skeleton_mask.unsqueeze(-1)
        imu_token = imu_token * imu_mask.unsqueeze(-1)

        skeleton_motion = self._expand_base(
            skeleton["motion_energy"].unsqueeze(-1),
            skeleton["part_mask"],
            skeleton["part_quality"],
        )[0].squeeze(-1)
        skeleton_motion[:, :, 5] = torch.linalg.vector_norm(
            batch["skeleton_features"][:, :, 13, 6:9], dim=-1
        )
        skeleton_motion[:, :, 6] = torch.linalg.vector_norm(
            batch["skeleton_features"][:, :, 16, 6:9], dim=-1
        )
        imu_motion = self._token_motion(imu_token, imu_mask)
        return {
            "skeleton_tokens": skeleton_token,
            "skeleton_mask": skeleton_mask,
            "skeleton_quality": skeleton_quality.clamp(0.0, 1.0),
            "skeleton_motion": skeleton_motion,
            "imu_tokens": imu_token,
            "imu_mask": imu_mask,
            "imu_quality": imu_quality.clamp(0.0, 1.0),
            "imu_motion": imu_motion,
            "imu_interval_count": imu["interval_count"],
        }


class LocalVisualObjectEncoder(nn.Module):
    """Encode oriented D/IR cells and form soft object/surface tokens without labels."""

    def __init__(self, width: int = 192, dropout: float = 0.12) -> None:
        super().__init__()
        self.width = width
        self.local_project = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, width), nn.GELU())
        self.difference_project = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, width), nn.GELU()
        )
        self.geometry_project = nn.Sequential(
            nn.Linear(6, width // 2), nn.GELU(), nn.Linear(width // 2, width)
        )
        self.context_project = nn.Sequential(
            nn.LayerNorm(896), nn.Linear(896, width), nn.GELU()
        )
        scale = 1.0 / math.sqrt(width)
        self.modality_embedding = nn.Parameter(torch.randn(2, width) * scale)
        self.region_embedding = nn.Parameter(torch.randn(len(P46_LOCAL_REGIONS), width) * scale)
        self.context_region_embedding = nn.Parameter(torch.randn(2, width) * scale)
        self.row_embedding = nn.Parameter(torch.randn(5, width) * scale)
        self.column_embedding = nn.Parameter(torch.randn(5, width) * scale)
        self.object_queries = nn.Parameter(torch.randn(3, width) * scale)
        self.object_key = nn.Linear(width, width, bias=False)
        self.object_geometry_score = nn.Sequential(
            nn.Linear(6, width // 4), nn.GELU(), nn.Linear(width // 4, 1)
        )
        self.contact_head = nn.Sequential(
            nn.LayerNorm(width * 2 + 2),
            nn.Linear(width * 2 + 2, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        self.output_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.null_visual = nn.Parameter(torch.zeros(width))

    @staticmethod
    def _grid_coordinates(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coordinate = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
        row, column = torch.meshgrid(coordinate, coordinate, indexing="ij")
        return torch.stack((column, row), dim=-1).reshape(size * size, 2)

    def _encode_grid(
        self,
        features: torch.Tensor,
        geometry: torch.Tensor,
        region_offset: int,
        valid: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        # features [B,T,M,R,H,W,128], geometry [B,T,R,H,W,6]
        previous = torch.zeros_like(features)
        if features.shape[1] > 1:
            previous[:, 1:] = (features[:, 1:] - features[:, :-1]).abs()
        encoded = self.local_project(features) + self.difference_project(previous)
        height, width = features.shape[4:6]
        row_embedding = self.row_embedding[:height][None, None, None, None, :, None]
        column_embedding = self.column_embedding[:width][None, None, None, None, None, :]
        encoded = encoded + self.modality_embedding[None, None, :, None, None, None]
        encoded = encoded + row_embedding + column_embedding
        encoded = encoded + self.geometry_project(geometry).unsqueeze(2)
        region_tokens: list[torch.Tensor] = []
        region_masks: list[torch.Tensor] = []
        region_geometry: list[torch.Tensor] = []
        region_motion: list[torch.Tensor] = []
        for local_region in range(features.shape[3]):
            region = region_offset + local_region
            token = encoded[:, :, :, local_region] + self.region_embedding[region]
            token = token.reshape(*token.shape[:2], -1, self.width)
            mask = valid[:, :, region, None].expand(-1, -1, 2 * height * width)
            mask = mask & frame_mask.unsqueeze(-1)
            geom = geometry[:, :, local_region].reshape(
                *geometry.shape[:2], height * width, 6
            )
            geom = geom[:, :, None].expand(-1, -1, 2, -1, -1).reshape(
                *geom.shape[:2], 2 * height * width, 6
            )
            motion = previous[:, :, :, local_region].square().mean(dim=-1).sqrt()
            motion = motion.reshape(*motion.shape[:2], -1)
            region_tokens.append(token * mask.unsqueeze(-1))
            region_masks.append(mask)
            region_geometry.append(geom)
            region_motion.append(motion * mask)
        return region_tokens, region_masks, region_geometry, region_motion

    def _soft_query(
        self,
        query_index: int,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        geometry: torch.Tensor,
        surface: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.object_queries[query_index]
        score = torch.einsum("btkd,d->btk", self.object_key(tokens), query)
        score = score / math.sqrt(self.width)
        score = score + self.object_geometry_score(geometry).squeeze(-1)
        depth_valid = geometry[..., 2].clamp(0.0, 1.0)
        non_body = (1.0 - geometry[..., 5]).clamp(0.0, 1.0)
        prior = depth_valid if surface else depth_valid * non_body
        score = score + torch.log(prior.clamp_min(0.02))
        score = score.masked_fill(~masks, -1e4)
        weight = torch.softmax(score, dim=-1) * masks
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        token = torch.sum(tokens * weight.unsqueeze(-1), dim=-2)
        valid = masks.any(dim=-1)
        quality = torch.sum(prior * weight, dim=-1) * valid
        return self.output_norm(token) * valid.unsqueeze(-1), valid, quality

    @staticmethod
    def _pad_sources(
        sources: list[list[tuple[torch.Tensor, torch.Tensor]]],
        null_visual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _, width = sources[0][0][0].shape
        maximum = max(
            1 + sum(value.shape[2] for value, _ in part_sources)
            for part_sources in sources
        )
        token = null_visual.new_zeros(
            batch, steps, len(EVENT_PART_NAMES), maximum, width
        )
        mask = torch.zeros(
            batch,
            steps,
            len(EVENT_PART_NAMES),
            maximum,
            dtype=torch.bool,
            device=null_visual.device,
        )
        token[:, :, :, 0] = null_visual
        mask[:, :, :, 0] = True
        for part, part_sources in enumerate(sources):
            offset = 1
            for value, valid in part_sources:
                count = value.shape[2]
                token[:, :, part, offset : offset + count] = value
                mask[:, :, part, offset : offset + count] = valid
                offset += count
        return token, mask

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        geometry = batch["local_geometry_features"]
        # Arms use 3x3 visual grids; pool their 5x5 geometry to exactly 3x3.
        arm_geometry = geometry[:, :, :2].permute(0, 1, 2, 5, 3, 4)
        flat_arm_geometry = arm_geometry.reshape(-1, 6, 5, 5)
        arm_geometry = torch.nn.functional.adaptive_avg_pool2d(flat_arm_geometry, 3)
        arm_geometry = arm_geometry.reshape(*geometry.shape[:2], 2, 6, 3, 3)
        arm_geometry = arm_geometry.permute(0, 1, 2, 4, 5, 3)
        arms = self._encode_grid(
            batch["arm_spatial_features"],
            arm_geometry,
            0,
            batch["local_roi_valid"],
            batch["frame_mask"],
        )
        detail = self._encode_grid(
            batch["detail_spatial_features"],
            geometry[:, :, 2:],
            2,
            batch["local_roi_valid"],
            batch["frame_mask"],
        )
        region_tokens = arms[0] + detail[0]
        region_masks = arms[1] + detail[1]
        region_geometry = arms[2] + detail[2]
        region_motion = arms[3] + detail[3]

        context = self.context_project(batch["context_features"])
        context = context + self.modality_embedding[None, None, :, None]
        context = context + self.context_region_embedding[None, None, None]
        context = context.reshape(*context.shape[:2], 4, self.width)
        context_mask = batch["context_valid"][:, :, None].expand(-1, -1, 2, -1)
        context_mask = context_mask.reshape(*context_mask.shape[:2], 4)
        context_mask = context_mask & batch["frame_mask"].unsqueeze(-1)
        context = context * context_mask.unsqueeze(-1)

        left_candidates = torch.cat((region_tokens[2], region_tokens[4]), dim=2)
        left_masks = torch.cat((region_masks[2], region_masks[4]), dim=2)
        left_geometry = torch.cat((region_geometry[2], region_geometry[4]), dim=2)
        right_candidates = torch.cat((region_tokens[3], region_tokens[4]), dim=2)
        right_masks = torch.cat((region_masks[3], region_masks[4]), dim=2)
        right_geometry = torch.cat((region_geometry[3], region_geometry[4]), dim=2)
        workspace_candidates = region_tokens[4]
        workspace_masks = region_masks[4]
        workspace_geometry = region_geometry[4]
        left_object = self._soft_query(
            0, left_candidates, left_masks, left_geometry
        )
        right_object = self._soft_query(
            1, right_candidates, right_masks, right_geometry
        )
        surface = self._soft_query(
            2, workspace_candidates, workspace_masks, workspace_geometry, surface=True
        )
        left_hand_pool = region_tokens[2].sum(dim=2) / region_masks[2].sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        right_hand_pool = region_tokens[3].sum(dim=2) / region_masks[3].sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        left_contact = torch.sigmoid(
            self.contact_head(
                torch.cat(
                    (
                        left_hand_pool,
                        left_object[0],
                        left_object[2].unsqueeze(-1),
                        left_object[1].to(left_object[0].dtype).unsqueeze(-1),
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
        ) * left_object[1]
        right_contact = torch.sigmoid(
            self.contact_head(
                torch.cat(
                    (
                        right_hand_pool,
                        right_object[0],
                        right_object[2].unsqueeze(-1),
                        right_object[1].to(right_object[0].dtype).unsqueeze(-1),
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
        ) * right_object[1]

        left_object_token = left_object[0].unsqueeze(2)
        right_object_token = right_object[0].unsqueeze(2)
        surface_token = surface[0].unsqueeze(2)
        left_object_mask = left_object[1].unsqueeze(2)
        right_object_mask = right_object[1].unsqueeze(2)
        surface_mask = surface[1].unsqueeze(2)
        # Every part receives low-resolution context. Local parts additionally
        # receive only semantically corresponding cells and soft object tokens.
        sources: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [(context, context_mask)] for _ in EVENT_PART_NAMES
        ]
        sources[2] += [(region_tokens[0], region_masks[0]), (region_tokens[1], region_masks[1])]
        sources[3] += [(region_tokens[0], region_masks[0]), (region_tokens[2], region_masks[2]), (left_object_token, left_object_mask)]
        sources[4] += [(region_tokens[1], region_masks[1]), (region_tokens[3], region_masks[3]), (right_object_token, right_object_mask)]
        sources[5] += [(region_tokens[2], region_masks[2]), (region_tokens[4], region_masks[4]), (left_object_token, left_object_mask)]
        sources[6] += [(region_tokens[3], region_masks[3]), (region_tokens[4], region_masks[4]), (right_object_token, right_object_mask)]
        sources[9] += [
            (region_tokens[2], region_masks[2]),
            (region_tokens[3], region_masks[3]),
            (region_tokens[4], region_masks[4]),
            (left_object_token, left_object_mask),
            (right_object_token, right_object_mask),
            (surface_token, surface_mask),
        ]
        part_sources, part_source_mask = self._pad_sources(sources, self.null_visual)

        visual_motion = batch["local_roi_quality"].new_zeros(
            *batch["frame_mask"].shape, len(EVENT_PART_NAMES)
        )
        region_motion_mean = [
            value.sum(dim=2) / mask.sum(dim=2).clamp_min(1)
            for value, mask in zip(region_motion, region_masks)
        ]
        visual_motion[:, :, 3] = region_motion_mean[0]
        visual_motion[:, :, 4] = region_motion_mean[1]
        visual_motion[:, :, 5] = region_motion_mean[2]
        visual_motion[:, :, 6] = region_motion_mean[3]
        visual_motion[:, :, 9] = region_motion_mean[4]
        visual_quality = batch["local_roi_quality"].new_zeros(visual_motion.shape)
        visual_quality[:, :, 0:3] = batch["context_quality"].mean(dim=2, keepdim=True)
        visual_quality[:, :, 7:9] = batch["context_quality"].mean(dim=2, keepdim=True)
        visual_quality[:, :, 3] = batch["local_roi_quality"][:, :, 0]
        visual_quality[:, :, 4] = batch["local_roi_quality"][:, :, 1]
        visual_quality[:, :, 5] = batch["local_roi_quality"][:, :, 2]
        visual_quality[:, :, 6] = batch["local_roi_quality"][:, :, 3]
        visual_quality[:, :, 9] = batch["local_roi_quality"][:, :, 4]
        contact = visual_motion.new_zeros(visual_motion.shape)
        contact[:, :, 3] = left_contact
        contact[:, :, 4] = right_contact
        contact[:, :, 5] = left_contact
        contact[:, :, 6] = right_contact
        contact[:, :, 9] = torch.maximum(left_contact, right_contact)
        return {
            "part_sources": self.dropout(part_sources),
            "part_source_mask": part_source_mask,
            "visual_motion": visual_motion,
            "visual_quality": visual_quality.clamp(0.0, 1.0),
            "contact_proxy": contact,
            "left_object_token": left_object[0],
            "right_object_token": right_object[0],
            "surface_token": surface[0],
            "left_object_quality": left_object[2],
            "right_object_quality": right_object[2],
            "surface_quality": surface[2],
        }


class SamePartEventFusion(nn.Module):
    """[6]-[8]: soft motion/contact gate and same-frame same-part cross-attention."""

    def __init__(self, width: int = 192, dropout: float = 0.12) -> None:
        super().__init__()
        self.width = width
        self.skeleton_project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.imu_project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.motion_gate = nn.Sequential(
            nn.Linear(7, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        self.query_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, num_heads=6, dropout=dropout, batch_first=True
        )
        self.dynamic_project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.output = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_embedding = nn.Parameter(
            torch.randn(len(EVENT_PART_NAMES), width) / math.sqrt(width)
        )

    def forward(
        self,
        motion: dict[str, torch.Tensor],
        visual: dict[str, torch.Tensor],
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        skeleton = self.skeleton_project(motion["skeleton_tokens"])
        imu = self.imu_project(motion["imu_tokens"])
        skeleton_weight = motion["skeleton_quality"] * motion["skeleton_mask"]
        imu_weight = motion["imu_quality"] * motion["imu_mask"]
        denominator = (skeleton_weight + imu_weight).clamp_min(1e-6)
        query = (
            skeleton * skeleton_weight.unsqueeze(-1)
            + imu * imu_weight.unsqueeze(-1)
        ) / denominator.unsqueeze(-1)
        query_mask = (motion["skeleton_mask"] | motion["imu_mask"]) & frame_mask.unsqueeze(-1)
        query = self.query_norm(query + self.event_embedding[None, None])
        query = query * query_mask.unsqueeze(-1)

        gate_input = torch.stack(
            (
                torch.log1p(motion["skeleton_motion"].clamp_min(0.0)),
                torch.log1p(motion["imu_motion"].clamp_min(0.0)),
                torch.log1p(visual["visual_motion"].clamp_min(0.0)),
                motion["skeleton_quality"],
                motion["imu_quality"],
                visual["visual_quality"],
                visual["contact_proxy"],
            ),
            dim=-1,
        )
        soft_gate = torch.sigmoid(self.motion_gate(gate_input).squeeze(-1))
        soft_gate = soft_gate * frame_mask.unsqueeze(-1)
        source = visual["part_sources"]
        source_mask = visual["part_source_mask"]
        batch, steps, parts, candidates, width = source.shape
        flat_query = query.reshape(batch * steps * parts, 1, width)
        flat_source = source.reshape(batch * steps * parts, candidates, width)
        flat_mask = source_mask.reshape(batch * steps * parts, candidates)
        attended, attention = self.cross_attention(
            flat_query,
            flat_source,
            flat_source,
            key_padding_mask=~flat_mask,
            need_weights=True,
        )
        attended = attended.reshape(batch, steps, parts, width)
        attention = attention.reshape(batch, steps, parts, candidates)
        dynamic = torch.zeros_like(attended)
        if steps > 1:
            dynamic[:, 1:] = attended[:, 1:] - attended[:, :-1]
        dynamic = self.dynamic_project(dynamic) * soft_gate.unsqueeze(-1)
        event = self.output(torch.cat((query, attended, dynamic), dim=-1))
        event_mask = frame_mask.unsqueeze(-1) & (
            query_mask | source_mask[..., 1:].any(dim=-1)
        )
        event = event * event_mask.unsqueeze(-1)
        return {
            "event_tokens": event,
            "event_mask": event_mask,
            "soft_event_gate": soft_gate,
            "visual_cross_attention": attention,
            "motion_query": query,
            "attended_visual": attended,
        }


class PartTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
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


class CompleteEventTemporalEncoder(nn.Module):
    """[9] Model short local phases and the complete variable-length event sequence."""

    def __init__(self, width: int = 192, dropout: float = 0.12) -> None:
        super().__init__()
        self.short_blocks = nn.ModuleList(
            PartTemporalBlock(width, dilation, dropout) for dilation in (1, 2, 4)
        )
        part_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=6,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.part_encoder = nn.TransformerEncoder(part_layer, num_layers=1)
        self.part_attention = nn.Sequential(
            nn.LayerNorm(width + 2), nn.Linear(width + 2, width // 2), nn.Tanh(), nn.Linear(width // 2, 1)
        )
        self.frame_project = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU()
        )
        long_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=6,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.long_temporal = nn.TransformerEncoder(long_layer, num_layers=2)
        self.phase_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(EVENT_PHASE_NAMES)))
        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1)
        )
        self.embedding = nn.Sequential(
            nn.LayerNorm(width * 3 + 12),
            nn.Linear(width * 3 + 12, 384),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def add_time_context(
        self,
        frame_sequence: torch.Tensor,
        frame_mask: torch.Tensor,
        time_position: torch.Tensor | None,
        frame_time_seconds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Hook for explicit time-aware descendants.

        The original P46 contract intentionally remains unchanged: callers that
        do not provide explicit time coordinates receive the exact historical
        frame sequence.  The unified-repair encoder overrides this hook so the
        long temporal model sees phase, cadence, and duration information.
        """

        del frame_mask, time_position, frame_time_seconds
        return frame_sequence

    def forward(
        self,
        event_tokens: torch.Tensor,
        event_mask: torch.Tensor,
        soft_gate: torch.Tensor,
        contact_proxy: torch.Tensor,
        frame_mask: torch.Tensor,
        time_position: torch.Tensor | None = None,
        frame_time_seconds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source = event_tokens
        for block in self.short_blocks:
            source = block(source, event_mask)
        batch, steps, parts, width = source.shape
        flat = source.reshape(batch * steps, parts, width)
        flat_mask = event_mask.reshape(batch * steps, parts)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        encoded = self.part_encoder(flat, src_key_padding_mask=~safe_mask)
        encoded = encoded.reshape(batch, steps, parts, width) * event_mask.unsqueeze(-1)
        part_score = self.part_attention(
            torch.cat((encoded, soft_gate.unsqueeze(-1), contact_proxy.unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        part_score = part_score.masked_fill(~event_mask, -1e4)
        part_weight = torch.softmax(part_score, dim=2) * event_mask
        part_weight = part_weight / part_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        attended_part = torch.sum(encoded * part_weight.unsqueeze(-1), dim=2)
        global_part = encoded[:, :, 0]
        frame_sequence = self.frame_project(torch.cat((attended_part, global_part), dim=-1))
        frame_sequence = self.add_time_context(
            frame_sequence,
            frame_mask,
            time_position,
            frame_time_seconds,
        )
        frame_sequence = frame_sequence * frame_mask.unsqueeze(-1)
        temporal = self.long_temporal(frame_sequence, src_key_padding_mask=~frame_mask)
        temporal = temporal * frame_mask.unsqueeze(-1)
        phase_logits = self.phase_head(temporal) * frame_mask.unsqueeze(-1)

        temporal_score = self.temporal_attention(temporal).squeeze(-1)
        temporal_score = temporal_score.masked_fill(~frame_mask, -1e4)
        temporal_weight = torch.softmax(temporal_score, dim=1) * frame_mask
        temporal_weight = temporal_weight / temporal_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attended = torch.sum(temporal * temporal_weight.unsqueeze(-1), dim=1)
        maximum = temporal.masked_fill(~frame_mask.unsqueeze(-1), -1e4).amax(dim=1)
        mean = (temporal * frame_mask.unsqueeze(-1)).sum(dim=1) / frame_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)

        frame_float = frame_mask.to(soft_gate.dtype)
        gate_frame = soft_gate.mean(dim=2)
        contact_frame = contact_proxy.mean(dim=2)
        event_change = event_tokens.new_zeros(batch, steps)
        if steps > 1:
            event_change[:, 1:] = torch.linalg.vector_norm(
                event_tokens[:, 1:] - event_tokens[:, :-1], dim=-1
            ).mean(dim=2) / math.sqrt(width)
        statistics: list[torch.Tensor] = []
        for value in (gate_frame, contact_frame, event_change):
            valid_value = value * frame_float
            statistics.extend(
                (
                    valid_value.sum(dim=1) / frame_float.sum(dim=1).clamp_min(1),
                    value.masked_fill(~frame_mask, -1e4).amax(dim=1),
                    torch.sqrt(
                        ((value - valid_value.sum(dim=1, keepdim=True) / frame_float.sum(dim=1, keepdim=True).clamp_min(1)).square() * frame_float).sum(dim=1)
                        / frame_float.sum(dim=1).clamp_min(1)
                        + 1e-8
                    ),
                    (valid_value[:, 1:] * valid_value[:, :-1]).sum(dim=1)
                    / (frame_float[:, 1:] * frame_float[:, :-1]).sum(dim=1).clamp_min(1),
                )
            )
        explicit_statistics = torch.stack(statistics, dim=1)
        trial_embedding = self.embedding(
            torch.cat((attended, maximum, mean, explicit_statistics), dim=-1)
        )
        return {
            "trial_embedding": trial_embedding,
            "event_temporal_sequence": temporal,
            "phase_logits": phase_logits,
            "part_attention": part_weight,
            "temporal_attention": temporal_weight,
            "explicit_event_statistics": explicit_statistics,
        }


class P46EventTokenEncoder(nn.Module):
    """Steps [7]-[9]. Step [10] losses and all classification heads are external."""

    def __init__(self, width: int = 192, dropout: float = 0.12) -> None:
        super().__init__()
        self.motion = MotionAdapter(width=width, dropout=dropout)
        self.visual = LocalVisualObjectEncoder(width=width, dropout=dropout)
        self.fusion = SamePartEventFusion(width=width, dropout=dropout)
        self.temporal = CompleteEventTemporalEncoder(width=width, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        motion = self.motion(batch)
        visual = self.visual(batch)
        fusion = self.fusion(motion, visual, batch["frame_mask"])
        temporal = self.temporal(
            fusion["event_tokens"],
            fusion["event_mask"],
            fusion["soft_event_gate"],
            visual["contact_proxy"],
            batch["frame_mask"],
        )
        return {
            **temporal,
            **fusion,
            "event_part_names": EVENT_PART_NAMES,
            "event_phase_names": EVENT_PHASE_NAMES,
            "contact_proxy": visual["contact_proxy"],
            "left_object_token": visual["left_object_token"],
            "right_object_token": visual["right_object_token"],
            "surface_token": visual["surface_token"],
            "left_object_quality": visual["left_object_quality"],
            "right_object_quality": visual["right_object_quality"],
            "surface_quality": visual["surface_quality"],
            "skeleton_motion": motion["skeleton_motion"],
            "imu_motion": motion["imu_motion"],
            "visual_motion": visual["visual_motion"],
            "imu_interval_count": motion["imu_interval_count"],
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def model_size_mib(module: nn.Module, bytes_per_parameter: int = 4) -> float:
    return parameter_count(module) * bytes_per_parameter / (1024**2)
