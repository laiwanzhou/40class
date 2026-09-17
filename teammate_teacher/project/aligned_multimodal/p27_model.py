from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.ops import roi_align

from aligned_model import AttentionPool, SkeletonEncoder, TemporalBlock, temporal_shift


PROJECT_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SharedDepthIRVisualEncoder(nn.Module):
    """One ResNet trunk with modality-specific stems and feature-space Local ROI.

    Depth initializes the shared trunk. IR has its own stem and is injected by a
    1x1 fusion initialized at low gain. Full and Local therefore add no duplicate
    backbone parameters.
    """

    def __init__(self, dropout: float = 0.20) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.depth_stem = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.ir_stem = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.stem_fuse = nn.Conv2d(128, 64, 1, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.global_temporal = nn.Sequential(
            TemporalBlock(512, 1, dropout * 0.5),
            TemporalBlock(512, 2, dropout * 0.5),
        )
        self.local_project = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.local_temporal = nn.Sequential(
            TemporalBlock(128, 1, dropout * 0.5),
            TemporalBlock(128, 2, dropout * 0.5),
        )
        self.global_project = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.reset_fusion()

    def reset_fusion(self) -> None:
        with torch.no_grad():
            self.stem_fuse.weight.zero_()
            for channel in range(64):
                self.stem_fuse.weight[channel, channel, 0, 0] = 1.0
                self.stem_fuse.weight[channel, 64 + channel, 0, 0] = 0.10

    @staticmethod
    def _roi_pool(
        features: torch.Tensor,
        boxes: torch.Tensor,
        batch_size: int,
        time_steps: int,
    ) -> torch.Tensor:
        _, channels, height, width = features.shape
        repeated = boxes[:, None].expand(batch_size, time_steps, 4).reshape(-1, 4)
        scaled = repeated.clone()
        scaled[:, [0, 2]] *= float(width)
        scaled[:, [1, 3]] *= float(height)
        batch_indices = torch.arange(
            batch_size * time_steps, device=features.device, dtype=features.dtype
        ).unsqueeze(1)
        rois = torch.cat([batch_indices, scaled], dim=1)
        pooled = roi_align(
            features,
            rois,
            output_size=(2, 3),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        ).mean(dim=(-2, -1))
        return pooled.reshape(batch_size, time_steps, channels)

    def forward(
        self,
        depth: torch.Tensor,
        ir: torch.Tensor,
        roi: torch.Tensor,
        depth_present: torch.Tensor,
        ir_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps = depth.shape[:2]
        depth_flat = depth.reshape(batch_size * time_steps, 3, *depth.shape[-2:])
        ir_flat = ir.reshape(batch_size * time_steps, 1, *ir.shape[-2:])
        depth_mask = depth_present[:, None].expand(batch_size, time_steps, 1).reshape(-1, 1, 1, 1)
        ir_mask = ir_present[:, None].expand(batch_size, time_steps, 1).reshape(-1, 1, 1, 1)
        depth_features = self.depth_stem(depth_flat) * depth_mask
        ir_features = self.ir_stem(ir_flat) * ir_mask
        x = self.stem_fuse(torch.cat([depth_features, ir_features], dim=1))
        x = self.maxpool(self.relu(self.bn1(x)))
        x = self.layer1(temporal_shift(x, batch_size, time_steps))
        x = self.layer2(temporal_shift(x, batch_size, time_steps))
        x = self.layer3(temporal_shift(x, batch_size, time_steps))
        local = self._roi_pool(x, roi, batch_size, time_steps)
        local = self.local_project(local)
        local = self.local_temporal(local.transpose(1, 2)).transpose(1, 2)
        x = self.layer4(temporal_shift(x, batch_size, time_steps))
        global_sequence = x.mean(dim=(-2, -1)).reshape(batch_size, time_steps, 512)
        global_sequence = self.global_temporal(
            global_sequence.transpose(1, 2)
        ).transpose(1, 2)
        return self.global_project(global_sequence), local


class IMUTemporalEncoder(nn.Module):
    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(10, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, output_dim, 1, bias=False),
            nn.GroupNorm(8, output_dim),
            nn.GELU(),
        )
        self.device_embedding = nn.Parameter(torch.zeros(1, 5, 1, output_dim))
        nn.init.normal_(self.device_embedding, std=0.02)

    def forward(
        self,
        imu: torch.Tensor,
        time_mask: torch.Tensor,
        device_mask: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        output_steps: int = 12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, devices, source_steps, channels = imu.shape
        normalized = (imu - mean.view(1, 1, 1, channels)) / std.view(1, 1, 1, channels)
        normalized = normalized * time_mask.unsqueeze(-1)
        encoded = self.encoder(
            normalized.reshape(batch_size * devices, source_steps, channels).transpose(1, 2)
        )
        encoded = nn.functional.interpolate(
            encoded, size=output_steps, mode="linear", align_corners=False
        )
        encoded = encoded.transpose(1, 2).reshape(
            batch_size, devices, output_steps, self.output_dim
        )
        mask = nn.functional.interpolate(
            time_mask.reshape(batch_size * devices, 1, source_steps),
            size=output_steps,
            mode="nearest",
        ).reshape(batch_size, devices, output_steps)
        mask = mask * device_mask.unsqueeze(-1)
        encoded = encoded * mask.unsqueeze(-1)
        encoded = encoded + self.device_embedding * mask.unsqueeze(-1)
        return encoded.permute(0, 2, 1, 3), mask.permute(0, 2, 1)


class P27EventModel(nn.Module):
    def __init__(
        self,
        imu_mean: torch.Tensor,
        imu_std: torch.Tensor,
        num_classes: int = 40,
        event_dim: int = 192,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.visual = SharedDepthIRVisualEncoder(dropout)
        self.skeleton = SkeletonEncoder(dropout, input_dim=4)
        self.skeleton_project = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.imu = IMUTemporalEncoder(64)
        self.imu_key_project = nn.Linear(64, 128)
        self.imu_query = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.imu_attention = nn.MultiheadAttention(
            128, num_heads=4, dropout=dropout * 0.5, batch_first=True
        )
        self.imu_attention_norm = nn.LayerNorm(128)
        self.event_input = nn.Sequential(
            nn.Linear(128 * 5 + 4, event_dim),
            nn.LayerNorm(event_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_temporal = nn.Sequential(
            TemporalBlock(event_dim, 1, dropout * 0.5),
            TemporalBlock(event_dim, 2, dropout * 0.5),
        )
        self.event_pool = AttentionPool(event_dim, attention_dim=96)
        self.global_context = nn.Sequential(
            nn.Linear(128, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(event_dim * 2 + 32),
            nn.Dropout(dropout),
            nn.Linear(event_dim * 2 + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
        self.skeleton_event_head = nn.Linear(128, 10)
        self.visual_event_head = nn.Linear(256, 2)
        self.imu_event_head = nn.Linear(64, 6)
        self.shared_motion_head = nn.Linear(event_dim, 1)
        self.clip_event_head = nn.Sequential(
            nn.LayerNorm(event_dim * 2), nn.Linear(event_dim * 2, 4)
        )
        self.visual_mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.skeleton_mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.imu_mask_token = nn.Parameter(torch.zeros(1, 1, 1, 64))
        self.register_buffer("imu_mean", imu_mean.float().clone())
        self.register_buffer("imu_std", imu_std.float().clone())
        window_mask = torch.ones(12, 12 * 5, dtype=torch.bool)
        for query_time in range(12):
            for key_time in range(max(0, query_time - 1), min(12, query_time + 2)):
                window_mask[query_time, key_time * 5 : (key_time + 1) * 5] = False
        self.register_buffer("imu_window_mask", window_mask)

    def apply_temporal_mask(
        self,
        visual_global: torch.Tensor,
        visual_local: torch.Tensor,
        skeleton: torch.Tensor,
        imu: torch.Tensor,
        probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or probability <= 0:
            return visual_global, visual_local, skeleton, imu
        batch_size, time_steps = visual_global.shape[:2]
        visual_mask = torch.rand(batch_size, time_steps, 1, device=visual_global.device) < probability
        skeleton_mask = torch.rand(batch_size, time_steps, 1, device=skeleton.device) < probability
        imu_mask = torch.rand(batch_size, time_steps, 1, 1, device=imu.device) < probability
        visual_global = torch.where(visual_mask, self.visual_mask_token, visual_global)
        visual_local = torch.where(visual_mask, self.visual_mask_token, visual_local)
        skeleton = torch.where(skeleton_mask, self.skeleton_mask_token, skeleton)
        imu = torch.where(imu_mask, self.imu_mask_token, imu)
        return visual_global, visual_local, skeleton, imu

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        temporal_mask_probability: float = 0.0,
        ablation: str | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_global, visual_local = self.visual(
            batch["depth"],
            batch["ir"],
            batch["roi"],
            batch["depth_present"],
            batch["ir_present"],
        )
        skeleton = self.skeleton_project(
            self.skeleton.encode_sequence(batch["skeleton"])
        )
        imu_devices, imu_mask = self.imu(
            batch["imu"],
            batch["imu_time_mask"],
            batch["imu_device_mask"],
            self.imu_mean,
            self.imu_std,
            output_steps=12,
        )
        visual_global, visual_local, skeleton, imu_devices = self.apply_temporal_mask(
            visual_global,
            visual_local,
            skeleton,
            imu_devices,
            temporal_mask_probability,
        )
        batch_size, time_steps = skeleton.shape[:2]
        query = self.imu_query(torch.cat([visual_local, skeleton], dim=-1))
        imu_keys = self.imu_key_project(imu_devices).reshape(batch_size, time_steps * 5, 128)
        attended, _ = self.imu_attention(
            query,
            imu_keys,
            imu_keys,
            attn_mask=self.imu_window_mask,
            need_weights=False,
        )
        attended = self.imu_attention_norm(attended)
        attended = attended * batch["imu_present"].unsqueeze(1)
        presence = torch.cat(
            [
                batch["depth_present"],
                batch["ir_present"],
                batch["skeleton_present"],
                batch["imu_present"],
            ],
            dim=1,
        ).unsqueeze(1).expand(batch_size, time_steps, 4)
        event_inputs = torch.cat(
            [
                visual_local,
                visual_global - visual_local,
                skeleton,
                attended,
                visual_local * skeleton,
                presence,
            ],
            dim=-1,
        )
        event_sequence = self.event_input(event_inputs)
        event_sequence = self.event_temporal(
            event_sequence.transpose(1, 2)
        ).transpose(1, 2)
        event_for_classifier = event_sequence
        global_sequence = visual_global
        if ablation == "event_zero":
            event_for_classifier = torch.zeros_like(event_sequence)
        elif ablation == "event_shuffle":
            event_for_classifier = event_sequence.roll(1, dims=0)
        elif ablation == "global_shuffle":
            global_sequence = visual_global.roll(1, dims=0)
        elif ablation not in {None, "imu_zero", "ir_zero"}:
            raise ValueError(f"unknown ablation: {ablation}")
        event_pooled = self.event_pool(event_for_classifier)
        visual_present = torch.maximum(batch["depth_present"], batch["ir_present"])
        global_context = self.global_context(global_sequence.mean(dim=1)) * visual_present
        logits = self.classifier(torch.cat([event_pooled, global_context], dim=1))
        imu_pooled = (imu_devices * imu_mask.unsqueeze(-1)).sum(dim=2) / (
            imu_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        )
        return {
            "logits": logits,
            "event_sequence": event_sequence,
            "event_pooled": event_pooled,
            "global_context": global_context,
            "skeleton_events": self.skeleton_event_head(skeleton),
            "visual_events": self.visual_event_head(
                torch.cat([visual_global, visual_local], dim=-1)
            ),
            "imu_events": self.imu_event_head(imu_pooled),
            "shared_motion": self.shared_motion_head(event_sequence),
            "clip_events": self.clip_event_head(self.event_pool(event_sequence)),
        }


def initialise_p27_from_fold(
    model: P27EventModel,
    fold: int,
    project_dir: Path = PROJECT_DIR,
) -> dict[str, dict[str, object]]:
    paths = {
        "depth": project_dir / "runs" / "p5_depth_imagenet" / f"fold_{fold}" / "last.pt",
        "ir": project_dir / "runs" / "p8_ir_imagenet_sum" / f"fold_{fold}" / "last.pt",
        "skeleton": project_dir
        / "runs"
        / "p0_tracking"
        / f"fold_{fold}_skeleton_first"
        / "last.pt",
    }
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    depth_state = checkpoints["depth"]["model_state_dict"]
    ir_state = checkpoints["ir"]["model_state_dict"]
    skeleton_state = checkpoints["skeleton"]["model_state_dict"]
    model.visual.depth_stem.weight.data.copy_(depth_state["visual.depth_stem.weight"])
    model.visual.ir_stem.weight.data.copy_(ir_state["visual.ir_stem.weight"])
    for module_name in (
        "bn1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "global_temporal",
    ):
        source_name = "temporal" if module_name == "global_temporal" else module_name
        module = getattr(model.visual, module_name)
        prefix = f"visual.{source_name}."
        state = {
            key[len(prefix) :]: value
            for key, value in depth_state.items()
            if key.startswith(prefix)
        }
        module.load_state_dict(state, strict=True)
    skeleton_prefix = "skeleton."
    model.skeleton.load_state_dict(
        {
            key[len(skeleton_prefix) :]: value
            for key, value in skeleton_state.items()
            if key.startswith(skeleton_prefix)
        },
        strict=True,
    )
    model.visual.reset_fusion()
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "epoch": int(checkpoints[name]["epoch"]),
        }
        for name, path in paths.items()
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fp16_size_mib(model: nn.Module) -> float:
    return parameter_count(model) * 2 / (1024**2)


def _vector_norm(value: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return torch.linalg.vector_norm(value, dim=-1).clamp_min(epsilon)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dim=-1) / (_vector_norm(left) * _vector_norm(right))


def compute_event_targets(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    skeleton = batch["skeleton"]
    left = skeleton[:, :, 13, :3]
    right = skeleton[:, :, 16, :3]
    head = skeleton[:, :, 10, :3]
    torso = skeleton[:, :, 8, :3]
    left_head = _vector_norm(left - head)
    right_head = _vector_norm(right - head)
    hands = _vector_norm(left - right)
    left_torso = _vector_norm(left - torso)
    right_torso = _vector_norm(right - torso)
    left_velocity = torch.diff(left, dim=1, prepend=left[:, :1])
    right_velocity = torch.diff(right, dim=1, prepend=right[:, :1])
    left_speed = _vector_norm(left_velocity)
    right_speed = _vector_norm(right_velocity)
    left_approach = torch.diff(left_head, dim=1, prepend=left_head[:, :1])
    right_approach = torch.diff(right_head, dim=1, prepend=right_head[:, :1])
    coordination = _cosine(left_velocity, right_velocity)
    skeleton_targets = torch.stack(
        [
            (left_head / 2.0).clamp(0, 1),
            (right_head / 2.0).clamp(0, 1),
            (hands / 2.0).clamp(0, 1),
            (left_torso / 2.0).clamp(0, 1),
            (right_torso / 2.0).clamp(0, 1),
            torch.tanh(left_speed * 5.0),
            torch.tanh(right_speed * 5.0),
            torch.tanh(left_approach * 5.0),
            torch.tanh(right_approach * 5.0),
            coordination.clamp(-1, 1),
        ],
        dim=-1,
    )

    depth_gray = batch["depth"].mean(dim=2)
    ir_gray = batch["ir"].squeeze(2)
    depth_motion = torch.diff(depth_gray, dim=1, prepend=depth_gray[:, :1]).abs()
    ir_motion = torch.diff(ir_gray, dim=1, prepend=ir_gray[:, :1]).abs()
    depth_present = batch["depth_present"].view(-1, 1, 1, 1)
    ir_present = batch["ir_present"].view(-1, 1, 1, 1)
    denom = (depth_present + ir_present).clamp_min(1.0)
    motion = (depth_motion * depth_present + ir_motion * ir_present) / denom
    global_motion = torch.tanh(motion.mean(dim=(-2, -1)))
    batch_size, time_steps, height, width = motion.shape
    repeated = batch["roi"][:, None].expand(batch_size, time_steps, 4).reshape(-1, 4).clone()
    repeated[:, [0, 2]] *= float(width)
    repeated[:, [1, 3]] *= float(height)
    indices = torch.arange(
        batch_size * time_steps, device=motion.device, dtype=motion.dtype
    ).unsqueeze(1)
    rois = torch.cat([indices, repeated], dim=1)
    local_motion = roi_align(
        motion.reshape(batch_size * time_steps, 1, height, width),
        rois,
        output_size=(8, 8),
        spatial_scale=1.0,
        sampling_ratio=1,
        aligned=True,
    ).mean(dim=(1, 2, 3)).reshape(batch_size, time_steps)
    local_motion = torch.tanh(local_motion)
    visual_targets = torch.stack([global_motion, local_motion], dim=-1)

    imu = batch["imu"]
    imu_mask = batch["imu_time_mask"]
    gyro = _vector_norm(imu[:, :, :, 3:6])
    acceleration = imu[:, :, :, :3]
    acceleration_delta = _vector_norm(
        torch.diff(acceleration, dim=2, prepend=acceleration[:, :, :1])
    )
    gyro_scaled = torch.log1p(gyro.clamp_min(0.0)) / math.log(721.0)
    acc_scaled = torch.log1p(acceleration_delta.clamp_min(0.0)) / math.log(6.0)
    source = torch.stack(
        [
            gyro_scaled[:, 1],
            gyro_scaled[:, 2],
            gyro_scaled[:, 0],
            acc_scaled[:, 1],
            acc_scaled[:, 2],
            (gyro_scaled[:, 1] - gyro_scaled[:, 2]).abs(),
        ],
        dim=1,
    )
    imu_targets = nn.functional.interpolate(
        source, size=12, mode="linear", align_corners=False
    ).transpose(1, 2).clamp(0, 1)
    imu_motion = imu_targets[:, :, :3].mean(dim=-1)
    shared_motion = 0.5 * (global_motion + imu_motion)

    arm_gyro = gyro_scaled[:, 1:3] * imu_mask[:, 1:3]
    periodicity: list[torch.Tensor] = []
    for arm in range(2):
        signal = arm_gyro[:, arm]
        signal = signal - signal.mean(dim=1, keepdim=True)
        denominator = signal.square().mean(dim=1).clamp_min(1e-6)
        correlations = []
        for lag in range(2, 9):
            correlations.append(
                (signal[:, :-lag] * signal[:, lag:]).mean(dim=1) / denominator
            )
        periodicity.append(torch.stack(correlations, dim=1).amax(dim=1).clamp(0, 1))
    left_centered = arm_gyro[:, 0] - arm_gyro[:, 0].mean(dim=1, keepdim=True)
    right_centered = arm_gyro[:, 1] - arm_gyro[:, 1].mean(dim=1, keepdim=True)
    bilateral = (
        (left_centered * right_centered).sum(dim=1)
        / (
            torch.linalg.vector_norm(left_centered, dim=1)
            * torch.linalg.vector_norm(right_centered, dim=1)
        ).clamp_min(1e-6)
    ).clamp(-1, 1)
    stillness = 1.0 - arm_gyro.mean(dim=(1, 2)).clamp(0, 1)
    clip_targets = torch.stack(
        [periodicity[0], periodicity[1], stillness, bilateral], dim=-1
    )
    return {
        "skeleton": skeleton_targets.detach(),
        "visual": visual_targets.detach(),
        "imu": imu_targets.detach(),
        "shared_motion": shared_motion.unsqueeze(-1).detach(),
        "clip": clip_targets.detach(),
        "skeleton_mask": batch["skeleton_present"].detach(),
        "visual_mask": torch.maximum(
            batch["depth_present"], batch["ir_present"]
        ).detach(),
        "imu_mask": batch["imu_present"].detach(),
    }
