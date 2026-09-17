"""One-fold hierarchical token fusion teacher for P91.

The model directly classifies 40 activities from IR, Depth, Thermal, dynamic
hand, MotionBERT, Skeleton-statistic, IMU-statistic and cross-relation tokens.
P90 is used only as a training-time distillation target and evaluation baseline;
it is not the structural output shell of this model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from p90_teacher_fusion_audit import align
from p91_unrestricted_fusion_teacher import build_cohorts, concatenate


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p91_hierarchical_multimodal_h3_v1"
DEPTH = PROJECT / "runs/p91_videomaev2_depth_fold0_v1/complete_features.npz"
THERMAL = PROJECT / "runs/p91_videomaev2_thermal_fold0_v1/complete_features.npz"
HAND = PROJECT / "runs/p91_videomaev2_hand_fold0_v1/complete_features.npz"
ROUTER = PROJECT / "runs/p90_crossuser_visual_router_v1/full_predictions.npz"
MOTION_CACHE = HERE / "runs/p86_motion_window_cache_t16_v1"
HDGCN_CACHE = PROJECT / "runs/p91_hdgcn_ntu60_xsub_fold0_v1/complete_features.npz"
FAMILIES = np.asarray(
    [
        0, 0, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 1,
        4, 4, 5, 5, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 7, 7, 7,
    ],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.22)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--statistics-dim", type=int, default=64)
    parser.add_argument("--seeds", default="17,43,71")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


@dataclass
class FusionData:
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    teacher_prediction: np.ndarray
    expert_probability: np.ndarray
    streams: dict[str, np.ndarray]
    statistics: dict[str, np.ndarray]
    skeleton_sequence: np.ndarray
    skeleton_mask: np.ndarray
    imu_sequence: np.ndarray
    imu_mask: np.ndarray
    thermal_available: np.ndarray
    boundaries: dict[str, np.ndarray]


def build_data() -> FusionData:
    cohorts = build_cohorts()
    order = ["H1_selection", "H2_confirmation", "E0_p87_sequence_source", "H3_independent_fold0"]
    missing = [name for name in order if name not in cohorts]
    if missing:
        raise ValueError(f"missing cohorts: {missing}")
    sizes = {name: len(cohorts[name].labels) for name in order}
    merged = concatenate("P91_all", [cohorts[name] for name in order])
    offsets = np.cumsum([0] + [sizes[name] for name in order])
    boundaries = {
        name: np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
        for index, name in enumerate(order)
    }
    ids = merged.sample_ids.astype(str)
    depth = load_npz(DEPTH)
    thermal = load_npz(THERMAL)
    hand = load_npz(HAND)
    hdgcn = load_npz(HDGCN_CACHE)
    with (MOTION_CACHE / "rows.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        motion_ids = np.asarray([row["sample_id"] for row in csv.DictReader(handle)], dtype=str)
    skeleton_sequence = np.asarray(
        np.load(MOTION_CACHE / "skeleton_features.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(len(motion_ids), 32, 17, 13)
    skeleton_mask = np.asarray(
        np.load(MOTION_CACHE / "skeleton_joint_mask.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(len(motion_ids), 32, 17)
    imu_sequence = np.asarray(
        np.load(MOTION_CACHE / "imu_sequences.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(len(motion_ids), 32, 5, 4, 16)
    imu_mask = np.asarray(
        np.load(MOTION_CACHE / "imu_sequence_mask.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(len(motion_ids), 32, 5, 4)
    streams = {
        "vmae": merged.vmae_tokens.astype(np.float32),
        "iv2": merged.iv2_tokens.astype(np.float32),
        "depth": align(depth["sample_ids"].astype(str), depth["features"], ids).reshape(
            len(ids), -1, 768
        ).astype(np.float32),
        "thermal": align(
            thermal["sample_ids"].astype(str), thermal["features"], ids
        ).reshape(len(ids), -1, 768).astype(np.float32),
        "hand": align(hand["sample_ids"].astype(str), hand["features"], ids).reshape(
            len(ids), -1, 768
        ).astype(np.float32),
        "motionbert": merged.motionbert_tokens.astype(np.float32),
        "hdgcn": np.tile(
            align(hdgcn["sample_ids"].astype(str), hdgcn["features"], ids).astype(
                np.float32
            ),
            (1, 1, 3),
        ),
    }
    thermal_available = align(
        thermal["sample_ids"].astype(str), thermal["modality_available"], ids
    ).astype(np.float32)
    router = load_npz(ROUTER)
    teacher_prediction = merged.safe_prediction.copy().astype(np.int64)
    for name in ("H1_selection", "H2_confirmation", "H3_independent_fold0"):
        lookup = {
            str(sample_id): int(prediction)
            for sample_id, prediction in zip(
                router[f"{name}_sample_ids"], router[f"{name}_router_prediction"]
            )
        }
        rows = boundaries[name]
        teacher_prediction[rows] = np.asarray([lookup[str(ids[row])] for row in rows])
    return FusionData(
        sample_ids=ids,
        labels=merged.labels.astype(np.int64),
        users=merged.users.astype(str),
        teacher_prediction=teacher_prediction,
        expert_probability=merged.expert_probability.astype(np.float32),
        streams=streams,
        statistics={
            "skeleton": merged.skeleton_statistics.astype(np.float32),
            "imu": merged.imu_statistics.astype(np.float32),
            "relation": merged.relation_statistics.astype(np.float32),
        },
        skeleton_sequence=align(motion_ids, skeleton_sequence, ids).astype(np.float32),
        skeleton_mask=align(motion_ids, skeleton_mask, ids).astype(np.float32),
        imu_sequence=align(motion_ids, imu_sequence, ids).astype(np.float32),
        imu_mask=align(motion_ids, imu_mask, ids).astype(np.float32),
        thermal_available=thermal_available,
        boundaries=boundaries,
    )


class Preprocessor:
    def __init__(self, statistics_dim: int, seed: int) -> None:
        self.statistics_dim = statistics_dim
        self.seed = seed
        self.token_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.sequence_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}

    def fit(self, data: FusionData, indices: np.ndarray) -> "Preprocessor":
        for name, stream in data.streams.items():
            values = stream[indices].reshape(-1, stream.shape[-1]).astype(np.float64)
            mean = values.mean(axis=0).astype(np.float32)
            std = np.maximum(values.std(axis=0), 1e-4).astype(np.float32)
            self.token_stats[name] = (mean, std)
        for name, values in data.statistics.items():
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values[indices].astype(np.float64))
            dimensions = min(self.statistics_dim, len(indices) - 1, scaled.shape[1])
            pca = PCA(
                n_components=dimensions,
                whiten=True,
                svd_solver="randomized",
                random_state=self.seed,
            )
            pca.fit(scaled)
            self.scalers[name] = scaler
            self.pcas[name] = pca
        skeleton_values = data.skeleton_sequence[indices]
        skeleton_valid = data.skeleton_mask[indices].astype(bool)
        skeleton_mean = np.zeros(13, dtype=np.float32)
        skeleton_std = np.ones(13, dtype=np.float32)
        for channel in range(13):
            values = skeleton_values[..., channel][skeleton_valid]
            if len(values):
                skeleton_mean[channel] = float(values.mean())
                skeleton_std[channel] = max(float(values.std()), 1e-4)
        self.sequence_stats["skeleton"] = (skeleton_mean, skeleton_std)
        imu_values = data.imu_sequence[indices]
        imu_valid = data.imu_mask[indices].astype(bool)
        imu_mean = np.zeros(4, dtype=np.float32)
        imu_std = np.ones(4, dtype=np.float32)
        for channel in range(4):
            values = imu_values[..., channel, :][imu_valid[..., channel]]
            if len(values):
                imu_mean[channel] = float(values.mean())
                imu_std[channel] = max(float(values.std()), 1e-4)
        self.sequence_stats["imu"] = (imu_mean, imu_std)
        return self

    def transform(self, data: FusionData) -> FusionData:
        streams = {
            name: ((values - self.token_stats[name][0]) / self.token_stats[name][1]).astype(
                np.float32
            )
            for name, values in data.streams.items()
        }
        statistics = {
            name: self.pcas[name]
            .transform(self.scalers[name].transform(values.astype(np.float64)))
            .astype(np.float32)
            for name, values in data.statistics.items()
        }
        skeleton_mean, skeleton_std = self.sequence_stats["skeleton"]
        skeleton_sequence = (
            (data.skeleton_sequence - skeleton_mean[None, None, None, :])
            / skeleton_std[None, None, None, :]
        ).astype(np.float32)
        skeleton_sequence *= data.skeleton_mask[..., None]
        imu_mean, imu_std = self.sequence_stats["imu"]
        imu_sequence = (
            (data.imu_sequence - imu_mean[None, None, None, :, None])
            / imu_std[None, None, None, :, None]
        ).astype(np.float32)
        imu_sequence *= data.imu_mask[..., None]
        return FusionData(
            sample_ids=data.sample_ids,
            labels=data.labels,
            users=data.users,
            teacher_prediction=data.teacher_prediction,
            expert_probability=data.expert_probability,
            streams=streams,
            statistics=statistics,
            skeleton_sequence=skeleton_sequence,
            skeleton_mask=data.skeleton_mask,
            imu_sequence=imu_sequence,
            imu_mask=data.imu_mask,
            thermal_available=data.thermal_available,
            boundaries=data.boundaries,
        )

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "input_dim": int(self.pcas[name].n_features_in_),
                "output_dim": int(self.pcas[name].n_components_),
                "explained_variance": float(self.pcas[name].explained_variance_ratio_.sum()),
            }
            for name in self.pcas
        }


def sample_weights(data: FusionData, indices: np.ndarray) -> np.ndarray:
    counts = np.bincount(data.labels[indices], minlength=40).astype(np.float64)
    class_weight = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    user_values, user_counts = np.unique(data.users[indices], return_counts=True)
    user_weight = {user: 1.0 / math.sqrt(count) for user, count in zip(user_values, user_counts)}
    output = np.ones(len(data.labels), dtype=np.float32)
    output[indices] = np.asarray(
        [class_weight[data.labels[row]] * user_weight[data.users[row]] for row in indices],
        dtype=np.float32,
    )
    output[indices] /= output[indices].mean()
    output[indices] *= np.where(
        data.teacher_prediction[indices] == data.labels[indices], 1.0, 1.30
    ).astype(np.float32)
    return output


class FusionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, data: FusionData, indices: np.ndarray, weights: np.ndarray) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = weights

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        item = {name: torch.from_numpy(values[row]) for name, values in self.data.streams.items()}
        item.update(
            {name: torch.from_numpy(values[row]) for name, values in self.data.statistics.items()}
        )
        item.update(
            {
                "skeleton_sequence": torch.from_numpy(self.data.skeleton_sequence[row]),
                "skeleton_mask": torch.from_numpy(self.data.skeleton_mask[row]),
                "imu_sequence": torch.from_numpy(self.data.imu_sequence[row]),
                "imu_mask": torch.from_numpy(self.data.imu_mask[row]),
                "thermal_available": torch.tensor(
                    self.data.thermal_available[row], dtype=torch.float32
                ),
                "label": torch.tensor(self.data.labels[row], dtype=torch.long),
                "family": torch.tensor(FAMILIES[self.data.labels[row]], dtype=torch.long),
                "teacher": torch.tensor(
                    self.data.teacher_prediction[row], dtype=torch.long
                ),
                "expert_probability": torch.from_numpy(
                    self.data.expert_probability[row]
                ),
                "weight": torch.tensor(self.weights[row], dtype=torch.float32),
                "row": torch.tensor(row, dtype=torch.long),
            }
        )
        return item


def h36m_adjacency() -> torch.Tensor:
    joints = 17
    edges = (
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
        (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    )
    identity = torch.eye(joints)
    inward = torch.zeros(joints, joints)
    outward = torch.zeros(joints, joints)
    for parent, child in edges:
        inward[child, parent] = 1.0
        outward[parent, child] = 1.0
    for value in (inward, outward):
        degree = value.sum(dim=0, keepdim=True).clamp_min(1.0)
        value.div_(degree)
    return torch.stack((identity, inward, outward))


class AdaptiveGraphTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.register_buffer("base_adjacency", h36m_adjacency())
        self.adaptive_adjacency = nn.Parameter(torch.zeros(3, 17, 17))
        self.spatial = nn.Conv2d(in_channels, out_channels * 3, kernel_size=1, bias=False)
        inner = max(out_channels // 8, 8)
        self.query = nn.Conv2d(in_channels, inner, kernel_size=1, bias=False)
        self.key = nn.Conv2d(in_channels, inner, kernel_size=1, bias=False)
        self.dynamic_scale = nn.Parameter(torch.tensor(0.0))
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, time, joints = x.shape
        value = self.spatial(x).reshape(batch, 3, -1, time, joints)
        adjacency = self.base_adjacency + self.adaptive_adjacency
        static = torch.einsum("bkctv,kvw->bctw", value, adjacency)
        query = self.query(x).mean(dim=2)
        key = self.key(x).mean(dim=2)
        dynamic = torch.softmax(
            torch.einsum("bcv,bcw->bvw", query, key) / math.sqrt(query.shape[1]),
            dim=-1,
        )
        dynamic_value = value.mean(dim=1)
        dynamic_output = torch.einsum("bctv,bvw->bctw", dynamic_value, dynamic)
        spatial = static + torch.tanh(self.dynamic_scale) * dynamic_output
        return F.gelu(self.temporal(spatial) + self.residual(x))


class CTRSkeletonEncoder(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            AdaptiveGraphTemporalBlock(13, 64),
            AdaptiveGraphTemporalBlock(64, 96, stride=2),
            AdaptiveGraphTemporalBlock(96, 128, stride=2),
        )
        self.output = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, model_dim))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        value = sequence.permute(0, 3, 1, 2).contiguous()
        value = self.blocks(value).mean(dim=3).transpose(1, 2)
        return self.output(value)


class DevicewiseIMUEncoder(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        hidden = max(model_dim // 2, 64)
        self.input = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.device_embedding = nn.Parameter(torch.randn(5, hidden) * 0.02)
        self.device_attention = nn.MultiheadAttention(
            hidden, num_heads=max(1, min(heads, hidden // 16)), dropout=dropout, batch_first=True
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, model_dim, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, time, devices = sequence.shape[:3]
        value = sequence.reshape(batch, time, devices, 64)
        value = self.input(value) + self.device_embedding[None, None]
        flat = value.reshape(batch * time, devices, -1)
        attended, _ = self.device_attention(flat, flat, flat, need_weights=False)
        attended = attended.reshape(batch, time, devices, -1)
        valid = (mask.sum(dim=-1) > 0).to(attended.dtype)
        pooled = (attended * valid[..., None]).sum(dim=2)
        pooled /= valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        return self.temporal(pooled.transpose(1, 2)).transpose(1, 2)


class HierarchicalTeacher(nn.Module):
    STREAM_NAMES = (
        "vmae", "iv2", "depth", "thermal", "hand", "motionbert", "hdgcn"
    )
    STAT_NAMES = ("skeleton", "imu", "relation")

    def __init__(
        self,
        statistics_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        stream_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        stream_dims = stream_dims or {name: 768 for name in self.STREAM_NAMES}
        missing_dims = set(self.STREAM_NAMES) - set(stream_dims)
        if missing_dims:
            raise ValueError(f"missing stream dimensions: {sorted(missing_dims)}")
        self.stream_project = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(stream_dims[name]), nn.Linear(stream_dims[name], model_dim)
                )
                for name in self.STREAM_NAMES
            }
        )
        self.stat_project = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(statistics_dim), nn.Linear(statistics_dim, model_dim), nn.GELU()
                )
                for name in self.STAT_NAMES
            }
        )
        self.ctr_skeleton = CTRSkeletonEncoder(model_dim)
        self.devicewise_imu = DevicewiseIMUEncoder(model_dim, heads, dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        type_count = len(self.STREAM_NAMES) + 2 + len(self.STAT_NAMES)
        self.type_embedding = nn.Parameter(torch.randn(type_count, model_dim) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(12, model_dim) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.main_head = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, 40),
        )
        self.family_head = nn.Linear(model_dim, 8)
        self.reliability_head = nn.Linear(model_dim, 1)
        self.global_visual_head = nn.Linear(model_dim, 40)
        self.local_visual_head = nn.Linear(model_dim, 40)
        self.sensor_head = nn.Linear(model_dim, 40)

    def maybe_drop(self, value: torch.Tensor, probability: float) -> torch.Tensor:
        if not self.training or probability <= 0:
            return value
        keep = (torch.rand(len(value), 1, 1, device=value.device) >= probability).to(value.dtype)
        return value * keep

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens: list[torch.Tensor] = []
        pools: dict[str, torch.Tensor] = {}
        for type_id, name in enumerate(self.STREAM_NAMES):
            value = self.stream_project[name](batch[name])
            value = value + self.type_embedding[type_id] + self.position_embedding[: value.shape[1]]
            if name == "thermal":
                value = value * batch["thermal_available"][:, None, None]
            if name in {"depth", "thermal", "hand", "motionbert", "hdgcn"}:
                value = self.maybe_drop(value, 0.10)
            pools[name] = value.mean(dim=1)
            tokens.append(value)
        skeleton_type = len(self.STREAM_NAMES)
        ctr_skeleton = self.ctr_skeleton(batch["skeleton_sequence"])
        ctr_skeleton = ctr_skeleton + self.type_embedding[skeleton_type] + self.position_embedding[: ctr_skeleton.shape[1]]
        ctr_skeleton = self.maybe_drop(ctr_skeleton, 0.10)
        pools["ctr_skeleton"] = ctr_skeleton.mean(dim=1)
        tokens.append(ctr_skeleton)
        raw_imu = self.devicewise_imu(batch["imu_sequence"], batch["imu_mask"])
        raw_imu = raw_imu + self.type_embedding[skeleton_type + 1] + self.position_embedding[: raw_imu.shape[1]]
        raw_imu = self.maybe_drop(raw_imu, 0.10)
        pools["raw_imu"] = raw_imu.mean(dim=1)
        tokens.append(raw_imu)
        for offset, name in enumerate(self.STAT_NAMES, start=skeleton_type + 2):
            value = self.stat_project[name](batch[name]).unsqueeze(1)
            value = value + self.type_embedding[offset]
            value = self.maybe_drop(value, 0.10 if name != "relation" else 0.06)
            pools[name] = value[:, 0]
            tokens.append(value)
        cls = self.cls.expand(len(batch["label"]), -1, -1)
        encoded = self.encoder(torch.cat((cls, *tokens), dim=1))
        representation = self.norm(encoded[:, 0])
        global_visual_names = ["vmae", "iv2"]
        if "vjepa" in pools:
            global_visual_names.append("vjepa")
        global_visual = sum(pools[name] for name in global_visual_names) / len(
            global_visual_names
        )
        local_visual = (pools["depth"] + pools["thermal"] + pools["hand"]) / 3.0
        sensor_names = [
            name
            for name in (
                "motionbert", "hdgcn", "ctr_skeleton", "raw_imu",
                "skeleton", "imu", "relation",
            )
            if name in pools
        ]
        sensor = sum(pools[name] for name in sensor_names) / len(sensor_names)
        return {
            "logits": self.main_head(representation),
            "representation": representation,
            "family_logits": self.family_head(representation),
            "reliability_logits": self.reliability_head(representation).squeeze(1),
            "global_visual_logits": self.global_visual_head(global_visual),
            "local_visual_logits": self.local_visual_head(local_visual),
            "sensor_logits": self.sensor_head(sensor),
        }


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def infer_full(
    model: HierarchicalTeacher,
    data: FusionData,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        FusionDataset(data, indices, np.ones(len(data.labels), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    logits = []
    reliability = []
    rows = []
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(batch)
        logits.append(output["logits"].float().cpu().numpy())
        reliability.append(output["reliability_logits"].float().cpu().numpy())
        rows.append(batch["row"].cpu().numpy())
    order = np.concatenate(rows)
    if not np.array_equal(order, indices):
        raise ValueError("inference order changed")
    return np.concatenate(logits), np.concatenate(reliability)


def infer(
    model: HierarchicalTeacher,
    data: FusionData,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    return infer_full(model, data, indices, device, batch_size)[0]


def train_model(
    args: argparse.Namespace,
    data: FusionData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    seed: int,
    fixed_epochs: int | None = None,
    model_type: type[HierarchicalTeacher] = HierarchicalTeacher,
) -> tuple[
    HierarchicalTeacher,
    dict[str, Any],
    np.ndarray | None,
    np.ndarray | None,
]:
    set_seed(seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model_type(
        statistics_dim=data.statistics["skeleton"].shape[1],
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    epochs = fixed_epochs or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weights = sample_weights(data, train_indices)
    loader = DataLoader(
        FusionDataset(data, train_indices, weights),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_logits: np.ndarray | None = None
    best_reliability: np.ndarray | None = None
    best_accuracy = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(batch)
                main = F.cross_entropy(
                    output["logits"], batch["label"], reduction="none", label_smoothing=0.025
                )
                main = (main * batch["weight"]).mean()
                family = F.cross_entropy(output["family_logits"], batch["family"])
                global_visual = F.cross_entropy(output["global_visual_logits"], batch["label"])
                local_visual = F.cross_entropy(output["local_visual_logits"], batch["label"])
                sensor = F.cross_entropy(output["sensor_logits"], batch["label"])
                teacher_ce = F.cross_entropy(
                    output["logits"], batch["teacher"], reduction="none"
                )
                teacher_weight = torch.where(
                    batch["teacher"] == batch["label"], 0.20, 0.025
                )
                distill = (teacher_ce * teacher_weight).mean()
                base_wrong = (batch["teacher"] != batch["label"]).float()
                reliability = F.binary_cross_entropy_with_logits(
                    output["reliability_logits"], base_wrong
                )
                loss = (
                    main
                    + 0.18 * family
                    + 0.08 * global_visual
                    + 0.10 * local_visual
                    + 0.07 * sensor
                    + distill
                    + 0.08 * reliability
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        record: dict[str, float] = {"epoch": epoch, "loss": float(np.mean(losses))}
        if validation_indices is not None:
            logits, reliability_logits = infer_full(
                model, data, validation_indices, device, args.batch_size
            )
            accuracy = float(np.mean(logits.argmax(axis=1) == data.labels[validation_indices]))
            record["validation_accuracy"] = accuracy
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = accuracy
                best_epoch = epoch
                best_logits = logits
                best_reliability = reliability_logits
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                history.append(record)
                break
        history.append(record)
    if validation_indices is not None:
        if best_state is None:
            raise RuntimeError("no validation checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    audit = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "epochs_ran": len(history),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }
    return model, audit, best_logits, best_reliability


def blend_prediction(
    logits: np.ndarray, teacher_prediction: np.ndarray, weight: float
) -> np.ndarray:
    neural = softmax(logits, axis=1)
    teacher = np.full((len(logits), 40), 0.06 / 39.0, dtype=np.float64)
    teacher[np.arange(len(logits)), teacher_prediction] = 0.94
    score = weight * np.log(np.clip(neural, 1e-8, 1.0))
    score += (1.0 - weight) * np.log(np.clip(teacher, 1e-8, 1.0))
    return score.argmax(axis=1)


def variable_blend_prediction(
    logits: np.ndarray,
    teacher_prediction: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    neural = softmax(logits, axis=1)
    teacher = np.full((len(logits), 40), 0.06 / 39.0, dtype=np.float64)
    teacher[np.arange(len(logits)), teacher_prediction] = 0.94
    weights = np.asarray(weights, dtype=np.float64)[:, None]
    score = weights * np.log(np.clip(neural, 1e-8, 1.0))
    score += (1.0 - weights) * np.log(np.clip(teacher, 1e-8, 1.0))
    return score.argmax(axis=1)


def select_adaptive_blend(
    logits: np.ndarray,
    reliability_logits: np.ndarray,
    labels: np.ndarray,
    teacher_prediction: np.ndarray,
    users: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reliability = 1.0 / (1.0 + np.exp(-np.clip(reliability_logits, -20, 20)))
    grid = []
    for base_weight in np.arange(0.40, 0.676, 0.025):
        for scale in np.arange(0.0, 0.501, 0.05):
            weights = np.clip(base_weight + scale * (2.0 * reliability - 1.0), 0.0, 1.0)
            prediction = variable_blend_prediction(logits, teacher_prediction, weights)
            row = {
                "base_weight": float(base_weight),
                "scale": float(scale),
                "mean_weight": float(weights.mean()),
                **audit(labels, teacher_prediction, prediction),
            }
            user_nets = []
            for user in np.unique(users):
                selected = users == user
                user_nets.append(
                    int(np.sum(prediction[selected] == labels[selected]))
                    - int(np.sum(teacher_prediction[selected] == labels[selected]))
                )
            row["negative_users"] = int(np.sum(np.asarray(user_nets) < 0))
            row["worst_user_net"] = int(min(user_nets, default=0))
            grid.append(row)
    eligible = [
        row for row in grid if row["negative_users"] <= 1 and row["worst_user_net"] >= -1
    ]
    selected = max(
        eligible or grid,
        key=lambda row: (
            row["correct"],
            -row["harm"],
            -row["negative_users"],
            row["worst_user_net"],
            -row["scale"],
        ),
    )
    return selected, grid


def apply_adaptive_blend(
    logits: np.ndarray,
    reliability_logits: np.ndarray,
    teacher_prediction: np.ndarray,
    selection: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    reliability = 1.0 / (1.0 + np.exp(-np.clip(reliability_logits, -20, 20)))
    weights = np.clip(
        float(selection["base_weight"])
        + float(selection["scale"]) * (2.0 * reliability - 1.0),
        0.0,
        1.0,
    )
    return variable_blend_prediction(logits, teacher_prediction, weights), weights


def audit(labels: np.ndarray, base: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result = {
        "accuracy": float(np.mean(prediction == labels)),
        "correct": int(np.sum(prediction == labels)),
        "base_accuracy": float(np.mean(base == labels)),
        "base_correct": int(np.sum(base == labels)),
        "rescue": int(np.sum((base != labels) & (prediction == labels))),
        "harm": int(np.sum((base == labels) & (prediction != labels))),
        "changed": int(np.sum(base != prediction)),
    }
    result["net"] = result["correct"] - result["base_correct"]
    return result


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = build_data()
    h1 = raw.boundaries["H1_selection"]
    h2 = raw.boundaries["H2_confirmation"]
    embargo = raw.boundaries["E0_p87_sequence_source"]
    h3 = raw.boundaries["H3_independent_fold0"]
    inner_train = np.concatenate((h1, embargo))
    final_train = np.concatenate((h1, h2, embargo))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    inner_pre = Preprocessor(args.statistics_dim, seeds[0]).fit(raw, inner_train)
    inner = inner_pre.transform(raw)
    inner_logits = []
    inner_reliability = []
    inner_audits = []
    selected_epochs = []
    for seed in seeds:
        print(f"  inner seed={seed}", flush=True)
        _, model_audit, logits, reliability_logits = train_model(
            args, inner, inner_train, h2, seed
        )
        if logits is None or reliability_logits is None:
            raise RuntimeError("missing inner outputs")
        inner_logits.append(logits)
        inner_reliability.append(reliability_logits)
        inner_audits.append(model_audit)
        selected_epochs.append(int(model_audit["best_epoch"]))
    inner_ensemble = np.mean(inner_logits, axis=0)
    inner_reliability_ensemble = np.mean(inner_reliability, axis=0)
    blend_grid = []
    for weight in np.linspace(0.0, 1.0, 41):
        prediction = blend_prediction(
            inner_ensemble, inner.teacher_prediction[h2], float(weight)
        )
        blend_grid.append(
            {"weight": float(weight), **audit(inner.labels[h2], inner.teacher_prediction[h2], prediction)}
        )
    selected_blend = max(
        blend_grid, key=lambda row: (row["correct"], -row["harm"], -row["weight"])
    )
    selected_adaptive, adaptive_grid = select_adaptive_blend(
        inner_ensemble,
        inner_reliability_ensemble,
        inner.labels[h2],
        inner.teacher_prediction[h2],
        inner.users[h2],
    )
    selected_method = (
        "adaptive"
        if (
            selected_adaptive["correct"],
            -selected_adaptive["harm"],
            -selected_adaptive["negative_users"],
        )
        > (selected_blend["correct"], -selected_blend["harm"], -99)
        else "constant"
    )
    fixed_epochs = max(1, int(round(float(np.median(selected_epochs)))))
    print(
        f"  selected epochs={selected_epochs} median={fixed_epochs} "
        f"blend={selected_blend['weight']} adaptive={selected_adaptive['base_weight']}+"
        f"{selected_adaptive['scale']} selected={selected_method}",
        flush=True,
    )

    final_pre = Preprocessor(args.statistics_dim, seeds[0] + 1000).fit(raw, final_train)
    final = final_pre.transform(raw)
    target_logits = []
    target_reliability = []
    final_audits = []
    for seed in seeds:
        print(f"  final seed={seed} epochs={fixed_epochs}", flush=True)
        model, model_audit, _, _ = train_model(
            args, final, final_train, None, seed + 10000, fixed_epochs=fixed_epochs
        )
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        logits, reliability_logits = infer_full(
            model, final, h3, device, args.batch_size
        )
        target_logits.append(logits)
        target_reliability.append(reliability_logits)
        final_audits.append(model_audit)
        torch.save(
            {"state_dict": model.state_dict(), "audit": model_audit},
            output / f"seed_{seed}_teacher.pt",
        )
    target_ensemble = np.mean(target_logits, axis=0)
    target_reliability_ensemble = np.mean(target_reliability, axis=0)
    direct_prediction = target_ensemble.argmax(axis=1)
    blended_prediction = blend_prediction(
        target_ensemble, final.teacher_prediction[h3], float(selected_blend["weight"])
    )
    adaptive_prediction, adaptive_weights = apply_adaptive_blend(
        target_ensemble,
        target_reliability_ensemble,
        final.teacher_prediction[h3],
        selected_adaptive,
    )
    selected_prediction = (
        adaptive_prediction if selected_method == "adaptive" else blended_prediction
    )
    report = {
        "protocol": (
            "H1+embargo train / H2 model-and-blend selection; refit H1+H2+embargo; "
            "H3 evaluated once; no size constraint."
        ),
        "tokens": {
            name: list(values.shape[1:]) for name, values in raw.streams.items()
        },
        "statistics": {name: list(values.shape[1:]) for name, values in raw.statistics.items()},
        "inner": {
            "train_samples": int(len(inner_train)),
            "validation_samples": int(len(h2)),
            "seed_audits": inner_audits,
            "ensemble_direct": audit(
                inner.labels[h2],
                inner.teacher_prediction[h2],
                inner_ensemble.argmax(axis=1),
            ),
            "selected_blend": selected_blend,
            "selected_adaptive": selected_adaptive,
            "selected_method": selected_method,
            "top_blends": sorted(blend_grid, key=lambda row: row["correct"], reverse=True)[:10],
            "top_adaptive": sorted(
                adaptive_grid,
                key=lambda row: (
                    row["correct"], -row["harm"], -row["negative_users"]
                ),
                reverse=True,
            )[:10],
        },
        "final": {
            "train_samples": int(len(final_train)),
            "target_samples": int(len(h3)),
            "fixed_epochs": fixed_epochs,
            "seed_audits": final_audits,
            "direct": audit(
                final.labels[h3], final.teacher_prediction[h3], direct_prediction
            ),
            "source_selected_blend": audit(
                final.labels[h3], final.teacher_prediction[h3], blended_prediction
            ),
            "source_selected_adaptive": audit(
                final.labels[h3], final.teacher_prediction[h3], adaptive_prediction
            ),
            "selected_method": selected_method,
            "selected_prediction": audit(
                final.labels[h3], final.teacher_prediction[h3], selected_prediction
            ),
        },
        "preprocessor": final_pre.summary(),
    }
    np.savez_compressed(
        output / "inner_predictions.npz",
        sample_ids=inner.sample_ids[h2],
        labels=inner.labels[h2],
        users=inner.users[h2],
        teacher_prediction=inner.teacher_prediction[h2],
        direct_logits=inner_ensemble.astype(np.float32),
        reliability_logits=inner_reliability_ensemble.astype(np.float32),
        selected_constant_weight=np.asarray(selected_blend["weight"], dtype=np.float32),
    )
    np.savez_compressed(
        output / "predictions.npz",
        sample_ids=final.sample_ids[h3],
        labels=final.labels[h3],
        base_prediction=final.teacher_prediction[h3],
        direct_logits=target_ensemble.astype(np.float32),
        direct_prediction=direct_prediction,
        blended_prediction=blended_prediction,
        adaptive_prediction=adaptive_prediction,
        selected_prediction=selected_prediction,
        reliability_logits=target_reliability_ensemble.astype(np.float32),
        adaptive_weights=adaptive_weights.astype(np.float32),
        selected_blend_weight=np.asarray(selected_blend["weight"], dtype=np.float32),
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
