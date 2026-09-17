from __future__ import annotations

import math

import torch
from torch import nn

from p86_mc3_visual_model import P86MC3VisualStudent


class P86V2TemporalDedupStudent(P86MC3VisualStudent):
    """MC3 visual student that does not count repeated source frames as new evidence.

    P86-v1 samples two overlapping 16-slot windows. Short trials therefore contain
    many rounded duplicate source frames, and temporal augmentation can add more.
    This model masks later copies within each window and uses a zero-initialized,
    continuous source-time projection. With unique time positions and the projection
    at initialization, it is functionally identical to the P86-v1 temporal head.
    """

    def __init__(
        self,
        classes: int = 40,
        width: int = 512,
        dropout: float = 0.18,
        fusion_mode: str = "gated",
        enable_distillation_projection: bool = False,
        frames: int = 16,
        kinetics_pretrained: bool = True,
        duplicate_tolerance: float = 1e-6,
    ) -> None:
        super().__init__(
            classes=classes,
            width=width,
            dropout=dropout,
            fusion_mode=fusion_mode,
            enable_distillation_projection=enable_distillation_projection,
            frames=frames,
            kinetics_pretrained=kinetics_pretrained,
            temporal_modeling=True,
            exact_time_modeling=False,
        )
        self.duplicate_tolerance = float(duplicate_tolerance)
        rng_state = torch.random.get_rng_state()
        self.continuous_time_projection = nn.Linear(9, width)
        nn.init.zeros_(self.continuous_time_projection.weight)
        nn.init.zeros_(self.continuous_time_projection.bias)
        # Preserve the common data-order/dropout RNG stream for the paired H0/H1
        # experiment. Candidate-only parameter initialization must not change it.
        torch.random.set_rng_state(rng_state)
        # Existing training utilities pass source time only when this attribute is true.
        self.exact_time_modeling = True

    def unique_time_mask(self, global_time_position: torch.Tensor) -> torch.Tensor:
        if global_time_position.ndim != 3 or global_time_position.shape[1:] != (
            2,
            self.frames,
        ):
            raise ValueError(
                f"global_time_position must be [B,2,{self.frames}], "
                f"got {tuple(global_time_position.shape)}"
            )
        difference = (
            global_time_position.unsqueeze(-1) - global_time_position.unsqueeze(-2)
        ).abs()
        same = difference <= self.duplicate_tolerance
        previous = torch.tril(
            torch.ones(
                self.frames,
                self.frames,
                dtype=torch.bool,
                device=global_time_position.device,
            ),
            diagonal=-1,
        )
        duplicate = (same & previous).any(dim=-1)
        return ~duplicate

    def encode_clips_from_backbone_sequence(
        self,
        backbone_sequence: torch.Tensor,
        view_valid: torch.Tensor | None = None,
        global_time_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if backbone_sequence.ndim != 5 or backbone_sequence.shape[1:] != (
            2,
            3,
            self.frames,
            512,
        ):
            raise ValueError(
                "backbone_sequence must have shape "
                f"[B,2,3,{self.frames},512], got {tuple(backbone_sequence.shape)}"
            )
        batch = backbone_sequence.shape[0]
        if view_valid is None or global_time_position is None:
            raise ValueError("P86-v2 temporal de-duplication requires validity and source time")
        if view_valid.shape != (batch, 2, self.frames, 3):
            raise ValueError("view_valid has the wrong shape")
        assert self.temporal_encoder is not None and self.temporal_fusion is not None

        unique = self.unique_time_mask(global_time_position)
        sequence = backbone_sequence.reshape(batch * 6, self.frames, 512)
        time_mask = view_valid.permute(0, 1, 3, 2) & unique.unsqueeze(2)
        time_mask = time_mask.reshape(batch * 6, self.frames)
        safe_mask = time_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True

        frequencies = global_time_position.new_tensor((1.0, 2.0, 4.0, 8.0))
        phase = math.pi * global_time_position.unsqueeze(-1) * frequencies
        exact_features = torch.cat(
            (global_time_position.unsqueeze(-1), phase.sin(), phase.cos()), dim=-1
        )
        continuous_time = self.continuous_time_projection(exact_features)
        continuous_time = continuous_time.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        temporal_input = (
            sequence
            + self.time_position
            + continuous_time.reshape(batch * 6, self.frames, 512)
        )
        encoded = self.temporal_encoder(
            temporal_input,
            src_key_padding_mask=~safe_mask,
        )
        whole = self._masked_mean(encoded, time_mask)
        midpoint = max(self.frames // 2, 1)
        early = self._masked_mean(encoded[:, :midpoint], time_mask[:, :midpoint])
        late = self._masked_mean(encoded[:, midpoint:], time_mask[:, midpoint:])
        ordered = self.temporal_fusion(
            torch.cat((whole, early, late, late - early), dim=1)
        )
        return (whole + ordered).reshape(batch, 2, 3, 512)


class P86V2Layer3SpatialDedupStudent(P86V2TemporalDedupStudent):
    """De-duplicated temporal model with a cheap pre-layer4 spatial residual.

    A fixed 2x2 pyramid extracts global, vertical, horizontal and diagonal layout
    from layer3 before layer4 discards fine spatial detail. The final projection is
    zero initialized, so this class starts exactly at the temporal de-dup model and
    adds about 0.4M parameters rather than another 512-wide Transformer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rng_state = torch.random.get_rng_state()
        self.layer3_spatial_projection = nn.Sequential(
            nn.LayerNorm(256 * 4),
            nn.Linear(256 * 4, 256),
            nn.GELU(),
            nn.Linear(256, 512),
        )
        nn.init.zeros_(self.layer3_spatial_projection[3].weight)
        nn.init.zeros_(self.layer3_spatial_projection[3].bias)
        torch.random.set_rng_state(rng_state)

    def encode_backbone_sequence(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6 or images.shape[1:4] != (2, self.frames, 3):
            raise ValueError(
                f"expected [B,2,{self.frames},3,H,W], got {tuple(images.shape)}"
            )
        batch, windows, steps, views, height, width = images.shape
        clips = images.permute(0, 1, 3, 2, 4, 5).reshape(
            batch * windows * views, steps, 1, height, width
        )
        unit = clips.float().div_(255.0)
        rgb = unit.repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
        # Keep normalization identical to the inherited MC3 implementation.
        mean = rgb.new_tensor((0.43216, 0.394666, 0.37645)).view(1, 3, 1, 1, 1)
        std = rgb.new_tensor((0.22803, 0.22145, 0.216989)).view(1, 3, 1, 1, 1)
        value = (rgb - mean) / std
        value = self.stem(value)
        value = self.layer1(value)
        value = self.layer2(value)
        layer3 = self.layer3(value)
        layer4 = self.layer4(layer3)

        pooled3 = nn.functional.adaptive_avg_pool3d(
            layer3, output_size=(self.frames, 2, 2)
        )
        regions = pooled3.flatten(-2).permute(0, 2, 3, 1)
        top = 0.5 * (regions[:, :, 0] + regions[:, :, 1])
        bottom = 0.5 * (regions[:, :, 2] + regions[:, :, 3])
        left = 0.5 * (regions[:, :, 0] + regions[:, :, 2])
        right = 0.5 * (regions[:, :, 1] + regions[:, :, 3])
        diagonal = 0.5 * (
            regions[:, :, 0] + regions[:, :, 3]
            - regions[:, :, 1] - regions[:, :, 2]
        )
        global3 = regions.mean(dim=2)
        spatial = self.layer3_spatial_projection(
            torch.cat((global3, bottom - top, right - left, diagonal), dim=-1)
        )

        sequence = layer4.mean(dim=(-1, -2)).permute(0, 2, 1)
        if sequence.shape[1] != self.frames:
            sequence = nn.functional.interpolate(
                sequence.transpose(1, 2), self.frames, mode="linear", align_corners=False
            ).transpose(1, 2)
        sequence = sequence + spatial
        return sequence.reshape(batch, windows, views, self.frames, 512)
