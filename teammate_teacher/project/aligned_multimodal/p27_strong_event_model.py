from __future__ import annotations

import torch
from torch import nn

from aligned_model import AttentionPool, TemporalBlock
from p27_model import IMUTemporalEncoder


class P27StrongEventModule(nn.Module):
    """Small event learner on fold-pure IR/Skeleton sequences and raw IMU."""

    def __init__(
        self,
        imu_mean: torch.Tensor,
        imu_std: torch.Tensor,
        event_target_count: int = 9,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.visual_project = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.skeleton_project = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.imu = IMUTemporalEncoder(64)
        self.imu_key_project = nn.Linear(64, 128)
        self.query = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.imu_attention = nn.MultiheadAttention(
            128, 4, dropout=dropout * 0.5, batch_first=True
        )
        self.imu_attention_norm = nn.LayerNorm(128)
        self.event_input = nn.Sequential(
            nn.Linear(128 * 5 + 3, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.Sequential(
            TemporalBlock(192, 1, dropout * 0.5),
            TemporalBlock(192, 2, dropout * 0.5),
            TemporalBlock(192, 4, dropout * 0.5),
        )
        self.pool = AttentionPool(192, attention_dim=96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 40),
        )
        if event_target_count != 9:
            raise ValueError("the audited target layout contains exactly 9 targets")
        self.skeleton_event_head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 5)
        )
        self.imu_event_head = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 4)
        )
        self.register_buffer("imu_mean", imu_mean.float().clone())
        self.register_buffer("imu_std", imu_std.float().clone())
        window = torch.ones(12, 12 * 5, dtype=torch.bool)
        for query_time in range(12):
            for key_time in range(max(0, query_time - 1), min(12, query_time + 2)):
                window[query_time, key_time * 5 : (key_time + 1) * 5] = False
        self.register_buffer("imu_window_mask", window)

    def forward(
        self,
        visual_sequence: torch.Tensor,
        skeleton_sequence: torch.Tensor,
        imu: torch.Tensor,
        imu_time_mask: torch.Tensor,
        imu_device_mask: torch.Tensor,
        *,
        ablation: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if ablation == "time_reverse":
            visual_sequence = visual_sequence.flip(1)
            skeleton_sequence = skeleton_sequence.flip(1)
            imu = imu.flip(2)
            imu_time_mask = imu_time_mask.flip(2)
            ablation = None
        elif ablation == "time_permute":
            generator = torch.Generator(device=visual_sequence.device)
            generator.manual_seed(27083)
            visual_order = torch.randperm(
                visual_sequence.shape[1],
                generator=generator,
                device=visual_sequence.device,
            )
            imu_order = torch.randperm(
                imu.shape[2], generator=generator, device=imu.device
            )
            visual_sequence = visual_sequence[:, visual_order]
            skeleton_sequence = skeleton_sequence[:, visual_order]
            imu = imu[:, :, imu_order]
            imu_time_mask = imu_time_mask[:, :, imu_order]
            ablation = None
        visual = self.visual_project(visual_sequence)
        skeleton = self.skeleton_project(skeleton_sequence)
        imu_devices, imu_mask = self.imu(
            imu,
            imu_time_mask,
            imu_device_mask,
            self.imu_mean,
            self.imu_std,
            output_steps=12,
        )
        query = self.query(torch.cat([visual, skeleton], dim=-1))
        batch_size = len(query)
        keys = self.imu_key_project(imu_devices).reshape(batch_size, 12 * 5, 128)
        key_padding_mask = imu_mask.reshape(batch_size, 12 * 5) <= 0
        all_missing = key_padding_mask.all(dim=1)
        if bool(all_missing.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_missing, 0] = False
            keys = keys.clone()
            keys[all_missing, 0] = 0
        attended, _ = self.imu_attention(
            query,
            keys,
            keys,
            attn_mask=self.imu_window_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended = self.imu_attention_norm(attended)
        imu_present = (imu_device_mask.sum(dim=1, keepdim=True) > 0).float()
        attended = attended * imu_present.unsqueeze(1)
        presence = torch.cat(
            [
                torch.ones_like(imu_present),
                torch.ones_like(imu_present),
                imu_present,
            ],
            dim=1,
        ).unsqueeze(1).expand(batch_size, 12, 3)
        event = self.event_input(
            torch.cat(
                [
                    visual,
                    skeleton,
                    attended,
                    visual * skeleton,
                    skeleton * attended,
                    presence,
                ],
                dim=-1,
            )
        )
        event = self.temporal(event.transpose(1, 2)).transpose(1, 2)
        if ablation == "imu_zero":
            return self.forward(
                visual_sequence,
                skeleton_sequence,
                torch.zeros_like(imu),
                torch.zeros_like(imu_time_mask),
                torch.zeros_like(imu_device_mask),
                ablation=None,
            )
        elif ablation not in {None, "event_zero", "event_shuffle"}:
            raise ValueError(f"unknown ablation: {ablation}")
        pooled = self.pool(event)
        if ablation == "event_zero":
            pooled = torch.zeros_like(pooled)
        elif ablation == "event_shuffle":
            pooled = pooled.roll(1, dims=0)
        skeleton_summary = torch.cat(
            [skeleton.mean(dim=1), skeleton.amax(dim=1)], dim=1
        )
        expanded_imu_mask = imu_mask.unsqueeze(-1)
        imu_mean = (imu_devices * expanded_imu_mask).sum(dim=(1, 2)) / (
            expanded_imu_mask.sum(dim=(1, 2)).clamp_min(1.0)
        )
        imu_max = imu_devices.masked_fill(
            expanded_imu_mask <= 0, torch.finfo(imu_devices.dtype).min
        ).amax(dim=(1, 2))
        imu_max = torch.where(
            imu_present > 0, imu_max, torch.zeros_like(imu_max)
        )
        imu_predictions = self.imu_event_head(
            torch.cat([imu_mean, imu_max], dim=1)
        )
        skeleton_predictions = self.skeleton_event_head(skeleton_summary)
        event_predictions = torch.cat(
            [
                imu_predictions[:, :1],
                skeleton_predictions,
                imu_predictions[:, 1:],
            ],
            dim=1,
        )
        return {
            "logits": self.classifier(pooled),
            "event_predictions": event_predictions,
            "event_sequence": event,
            "event_pooled": pooled,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
