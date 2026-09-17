"""P116 source-safe high-resolution visual capability audit for fixed hard pairs.

Candidate classes intervene through cross-attention over every spatial token,
before any trial-level spatial or temporal pooling.  The fixed pair set is
24<->26, 19<->24, 6<->37, and 21<->22.  Layer2, layer3, a staged
layer2->layer3 scorer, the P115-style layer4 grid, and the P108 pooled Visual
descriptor are compared under identical subject splits and optimizer steps.

This is a capability audit only.  It does not modify P89 or produce a final
replacement decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from audit_p112_multimodal_pair_tree import OUTER_USERS, _align
from p115_full_token_candidate_elimination import (
    SEED,
    FrameDataset,
    TokenGrid,
    _open_memmap,
    bootstrap_mean_ci,
    build_protocol,
    load_p108_pooled_visual,
    read_csv,
    write_csv,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PIXELS = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_HIGHRES = HERE / "runs/p116_resnet18_layer2_layer3_tokens_v1"
DEFAULT_LAYER4 = HERE / "runs/p115_resnet18_full_tokens_v1"
DEFAULT_VJEPA = HERE.parent / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
DEFAULT_OUTPUT = HERE / "runs/p116_highres_pair_capability_v1"

PROTOCOL_VERSION = "p116_highres_pair_capability_v1"
PAIRS = ((24, 26), (19, 24), (6, 37), (21, 22))
PAIR_NAMES = tuple(f"{left}<->{right}" for left, right in PAIRS)
HIERARCHICAL_VARIANT = "hierarchical_layer2_layer3"
WORKSPACE_VARIANT = "workspace_hierarchical_layer2_layer3"
HIGHRES_VARIANTS = (
    "layer2_20x20",
    "layer3_10x10",
    "layer2_then_layer3",
    HIERARCHICAL_VARIANT,
    WORKSPACE_VARIANT,
)
CONTROL_VARIANTS = ("layer4_5x5", "pooled_vlit_vhpd_vwpd")
# Keep the first-round indices stable in cached run seeds.  The hierarchical
# variant is the single source-motivated iteration and is appended.
VARIANTS = (
    "layer2_20x20",
    "layer3_10x10",
    "layer2_then_layer3",
    "layer4_5x5",
    "pooled_vlit_vhpd_vwpd",
    HIERARCHICAL_VARIANT,
    WORKSPACE_VARIANT,
)
PERTURBATIONS = ("aligned", "shuffle", "zero")
INNER_FOLDS = 3


@dataclass(frozen=True)
class Grid:
    windows: int
    times: int
    views: int
    height: int
    width: int
    channels: int

    @property
    def token_count(self) -> int:
        return self.windows * self.times * self.views * self.height * self.width

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            self.windows,
            self.times,
            self.views,
            self.height,
            self.width,
            self.channels,
        )


LAYER2_GRID = Grid(2, 16, 3, 20, 20, 128)
LAYER3_GRID = Grid(2, 16, 3, 10, 10, 256)
LAYER4_GRID = Grid(2, 16, 3, 5, 5, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "tokens", "audit"), default="all")
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--highres-cache", type=Path, default=DEFAULT_HIGHRES)
    parser.add_argument("--layer4-cache", type=Path, default=DEFAULT_LAYER4)
    parser.add_argument("--vjepa-root", type=Path, default=DEFAULT_VJEPA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", default=",".join(OUTER_USERS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--extract-batch-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Layer2Layer3Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        children = list(backbone.children())
        self.through_layer2 = nn.Sequential(*children[:6])
        self.layer3 = children[6]

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layer2 = self.through_layer2(value)
        return layer2, self.layer3(layer2)


@torch.inference_mode()
def build_highres_cache(args: argparse.Namespace) -> dict[str, Any]:
    output = args.highres_cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.pixel_cache / "rows.csv")
    frame_count = len(rows) * 2 * 16 * 3
    layer2 = _open_memmap(
        output / "layer2_tokens.npy", np.dtype(np.float16), (len(rows), *LAYER2_GRID.shape)
    )
    layer3 = _open_memmap(
        output / "layer3_tokens.npy", np.dtype(np.float16), (len(rows), *LAYER3_GRID.shape)
    )
    completed = _open_memmap(
        output / "completed_frames.npy", np.dtype(np.bool_), (frame_count,)
    )
    dataset = FrameDataset(args.pixel_cache / "images.npy")
    if len(dataset) != frame_count:
        raise RuntimeError("P116 frame inventory changed")
    pending = np.flatnonzero(~np.asarray(completed, dtype=bool))
    encoder = Layer2Layer3Encoder().to(args.device).eval()
    loader = DataLoader(
        torch.utils.data.Subset(dataset, pending.tolist()),
        batch_size=args.extract_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    flat2 = layer2.reshape(frame_count, 20, 20, 128)
    flat3 = layer3.reshape(frame_count, 10, 10, 256)
    processed = 0
    for images, indices in loader:
        images = images.to(args.device, non_blocking=True)
        with torch.autocast(
            device_type="cuda" if args.device.startswith("cuda") else "cpu",
            dtype=torch.float16,
            enabled=args.device.startswith("cuda"),
        ):
            value2, value3 = encoder(images)
        if tuple(value2.shape[1:]) != (128, 20, 20):
            raise RuntimeError(f"P116 layer2 shape changed: {tuple(value2.shape)}")
        if tuple(value3.shape[1:]) != (256, 10, 10):
            raise RuntimeError(f"P116 layer3 shape changed: {tuple(value3.shape)}")
        selected = indices.numpy().astype(np.int64)
        flat2[selected] = value2.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)
        flat3[selected] = value3.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)
        completed[selected] = True
        processed += len(selected)
        if processed % (args.extract_batch_size * 25) < args.extract_batch_size:
            layer2.flush()
            layer3.flush()
            completed.flush()
            print(
                json.dumps(
                    {"p116_frames": int(np.sum(completed)), "total": frame_count}
                ),
                flush=True,
            )
    layer2.flush()
    layer3.flush()
    completed.flush()
    if not np.asarray(completed, dtype=bool).all():
        raise RuntimeError("P116 high-resolution extraction incomplete")
    summary = {
        "protocol": PROTOCOL_VERSION,
        "complete": True,
        "label_free_extraction": True,
        "backbone": "torchvision ResNet18 ImageNet1K V1 through layer3",
        "input_views": ["scene", "person", "workspace_hand_interaction"],
        "input_windows": ["early", "late"],
        "frames_per_window": 16,
        "global_spatial_pooling_before_cache": False,
        "temporal_pooling_before_cache": False,
        "layer2_grid": asdict(LAYER2_GRID),
        "layer3_grid": asdict(LAYER3_GRID),
        "layer2_tokens_per_trial": LAYER2_GRID.token_count,
        "layer3_tokens_per_trial": LAYER3_GRID.token_count,
        "rows": len(rows),
        "cache_bytes": int(
            (output / "layer2_tokens.npy").stat().st_size
            + (output / "layer3_tokens.npy").stat().st_size
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


class GridPosition(nn.Module):
    def __init__(self, grid: Grid, dim: int) -> None:
        super().__init__()
        self.grid = grid
        self.window = nn.Embedding(grid.windows, dim)
        self.time = nn.Embedding(grid.times, dim)
        self.view = nn.Embedding(grid.views, dim)
        self.row = nn.Embedding(grid.height, dim)
        self.column = nn.Embedding(grid.width, dim)
        for embedding in (self.window, self.time, self.view, self.row, self.column):
            nn.init.trunc_normal_(embedding.weight, std=0.02)

    def forward(self, device: torch.device) -> torch.Tensor:
        grid = self.grid
        value = (
            self.window(torch.arange(grid.windows, device=device))[:, None, None, None, None]
            + self.time(torch.arange(grid.times, device=device))[None, :, None, None, None]
            + self.view(torch.arange(grid.views, device=device))[None, None, :, None, None]
            + self.row(torch.arange(grid.height, device=device))[None, None, None, :, None]
            + self.column(torch.arange(grid.width, device=device))[None, None, None, None, :]
        )
        return value.reshape(grid.token_count, -1)


class CandidateTokenEvidence(nn.Module):
    """Candidate-conditioned aggregation over an unpooled token grid."""

    def __init__(self, channels: int, width: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.content = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, width))
        self.query_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 3, width),
        )
        self.scale = width**-0.5

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        position: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        content = self.content(tokens) + position[None]
        affinity = torch.einsum(
            "bkd,bnd->bkn", self.query_norm(query), self.token_norm(content)
        ) * self.scale
        attention = affinity.softmax(dim=-1)
        attended = torch.einsum("bkn,bnd->bkd", attention, content)
        state = query + attended
        state = state + self.feed_forward(state)
        return state, attention if return_attention else None


class SpatialPairScorer(nn.Module):
    def __init__(self, stages: Sequence[tuple[str, Grid]], width: int = 64) -> None:
        super().__init__()
        self.stage_names = tuple(name for name, _ in stages)
        self.grids = {name: grid for name, grid in stages}
        self.class_query = nn.Embedding(40, width)
        self.positions = nn.ModuleDict(
            {name: GridPosition(grid, width) for name, grid in stages}
        )
        self.evidence = nn.ModuleDict(
            {
                name: CandidateTokenEvidence(grid.channels, width)
                for name, grid in stages
            }
        )
        self.score = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )
        nn.init.trunc_normal_(self.class_query.weight, std=0.02)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        query = self.class_query(candidates)
        audit: dict[str, torch.Tensor | None] = {}
        for name in self.stage_names:
            value = batch[name]
            grid = self.grids[name]
            if tuple(value.shape[1:]) != grid.shape:
                raise ValueError(f"{name} token grid changed: {tuple(value.shape[1:])}")
            query, attention = self.evidence[name](
                query,
                value.reshape(len(value), grid.token_count, grid.channels),
                self.positions[name](value.device),
                return_attention,
            )
            audit[name] = attention
        return self.score(query).squeeze(-1), audit


class HierarchicalSpatialTemporalPairScorer(nn.Module):
    """Localize per frame/view first, then aggregate candidate-specific time evidence."""

    def __init__(
        self, width: int = 64, view_indices: tuple[int, ...] = (0, 1, 2)
    ) -> None:
        super().__init__()
        if not view_indices or not set(view_indices) <= {0, 1, 2}:
            raise ValueError("invalid hierarchical view indices")
        self.view_indices = view_indices
        self.class_query = nn.Embedding(40, width)
        self.layer2_position = GridPosition(Grid(1, 1, 1, 20, 20, 128), width)
        self.layer3_position = GridPosition(Grid(1, 1, 1, 10, 10, 256), width)
        self.layer2_evidence = CandidateTokenEvidence(128, width)
        self.layer3_evidence = CandidateTokenEvidence(256, width)
        self.window = nn.Embedding(2, width)
        self.time = nn.Embedding(16, width)
        self.view = nn.Embedding(3, width)
        self.temporal_query_norm = nn.LayerNorm(width)
        self.frame_norm = nn.LayerNorm(width)
        self.temporal_feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 3),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(width * 3, width),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.scale = width**-0.5
        for embedding in (self.class_query, self.window, self.time, self.view):
            nn.init.trunc_normal_(embedding.weight, std=0.02)

    def frame_position(self, device: torch.device) -> torch.Tensor:
        view_ids = torch.as_tensor(self.view_indices, device=device)
        value = (
            self.window(torch.arange(2, device=device))[:, None, None]
            + self.time(torch.arange(16, device=device))[None, :, None]
            + self.view(view_ids)[None, None, :]
        )
        return value.reshape(2 * 16 * len(self.view_indices), -1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        layer2 = batch["layer2"]
        layer3 = batch["layer3"]
        if tuple(layer2.shape[1:]) != LAYER2_GRID.shape:
            raise ValueError("hierarchical layer2 grid changed")
        if tuple(layer3.shape[1:]) != LAYER3_GRID.shape:
            raise ValueError("hierarchical layer3 grid changed")
        layer2 = layer2[:, :, :, list(self.view_indices)]
        layer3 = layer3[:, :, :, list(self.view_indices)]
        batch_size, candidate_count = candidates.shape
        frame_count = 2 * 16 * len(self.view_indices)
        base_query = self.class_query(candidates)
        frame_query = (
            base_query[:, None]
            .expand(-1, frame_count, -1, -1)
            .reshape(batch_size * frame_count, candidate_count, -1)
        )
        frame_query, layer2_attention = self.layer2_evidence(
            frame_query,
            layer2.reshape(batch_size * frame_count, 400, 128),
            self.layer2_position(layer2.device),
            return_attention,
        )
        frame_query, layer3_attention = self.layer3_evidence(
            frame_query,
            layer3.reshape(batch_size * frame_count, 100, 256),
            self.layer3_position(layer2.device),
            return_attention,
        )
        frame_state = frame_query.reshape(
            batch_size, frame_count, candidate_count, -1
        ).permute(0, 2, 1, 3)
        frame_state = frame_state + self.frame_position(layer2.device)[None, None]
        affinity = torch.einsum(
            "bkd,bkfd->bkf",
            self.temporal_query_norm(base_query),
            self.frame_norm(frame_state),
        ) * self.scale
        temporal_attention = affinity.softmax(dim=-1)
        attended = torch.einsum("bkf,bkfd->bkd", temporal_attention, frame_state)
        state = base_query + attended
        state = state + self.temporal_feed_forward(state)
        return self.score(state).squeeze(-1), {
            "layer2_spatial": layer2_attention,
            "layer3_spatial": layer3_attention,
            "frame_temporal": temporal_attention if return_attention else None,
        }


class PooledPairScorer(nn.Module):
    def __init__(self, input_dim: int, width: int = 64) -> None:
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU())
        self.query = nn.Embedding(40, width)
        self.score = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.trunc_normal_(self.query.weight, std=0.02)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, None]]:
        if return_attention:
            raise ValueError("pooled control has no spatial attention")
        visual = self.visual(batch["pooled"])[:, None].expand(-1, candidates.shape[1], -1)
        query = self.query(candidates)
        interaction = torch.cat(
            (visual, query, visual * query, torch.abs(visual - query)), dim=-1
        )
        return self.score(interaction).squeeze(-1), {}


@dataclass(frozen=True)
class PairEntry:
    row: int
    pair_id: int
    target: int


@dataclass
class Sources:
    ids: np.ndarray
    users: np.ndarray
    labels: np.ndarray
    layer2: np.ndarray
    layer3: np.ndarray
    layer4: np.ndarray
    pooled: np.ndarray


def load_sources(args: argparse.Namespace) -> Sources:
    rows = read_csv(args.pixel_cache / "rows.csv")
    ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
    users = np.asarray([row["user_id"] for row in rows], dtype=str)
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    high_summary = json.loads((args.highres_cache / "summary.json").read_text(encoding="utf-8"))
    layer4_summary = json.loads((args.layer4_cache / "summary.json").read_text(encoding="utf-8"))
    if not high_summary.get("complete") or high_summary.get("global_spatial_pooling_before_cache"):
        raise RuntimeError("P116 high-resolution cache contract failed")
    if not layer4_summary.get("complete") or layer4_summary.get("global_spatial_pooling_before_cache"):
        raise RuntimeError("P116 layer4 cache contract failed")
    layer2 = np.load(args.highres_cache / "layer2_tokens.npy", mmap_mode="r")
    layer3 = np.load(args.highres_cache / "layer3_tokens.npy", mmap_mode="r")
    layer4 = np.load(args.layer4_cache / "full_tokens.npy", mmap_mode="r")
    pooled = load_p108_pooled_visual(args.vjepa_root, ids)
    if layer2.shape != (len(rows), *LAYER2_GRID.shape):
        raise RuntimeError("P116 layer2 shape differs")
    if layer3.shape != (len(rows), *LAYER3_GRID.shape):
        raise RuntimeError("P116 layer3 shape differs")
    if layer4.shape != (len(rows), *LAYER4_GRID.shape):
        raise RuntimeError("P116 layer4 shape differs")
    return Sources(ids, users, labels, layer2, layer3, layer4, pooled)


def build_entries(labels: np.ndarray, selected: np.ndarray) -> list[PairEntry]:
    entries: list[PairEntry] = []
    for pair_id, pair in enumerate(PAIRS):
        rows = np.flatnonzero(selected & np.isin(labels, pair))
        entries.extend(
            PairEntry(int(row), pair_id, int(labels[row] == pair[1])) for row in rows
        )
    return entries


def shuffled_input_rows(
    entries: Sequence[PairEntry], users: np.ndarray
) -> np.ndarray:
    output = np.asarray([entry.row for entry in entries], dtype=np.int64)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        groups[(str(users[entry.row]), entry.pair_id)].append(index)
    for indices in groups.values():
        if len(indices) > 1:
            source_rows = output[indices].copy()
            output[indices] = np.roll(source_rows, 1)
    return output


class PairDataset(Dataset):
    def __init__(
        self,
        sources: Sources,
        variant: str,
        entries: Sequence[PairEntry],
        perturbation: str = "aligned",
    ) -> None:
        self.sources = sources
        self.variant = variant
        self.entries = list(entries)
        if perturbation not in PERTURBATIONS:
            raise ValueError(f"unknown perturbation: {perturbation}")
        self.perturbation = perturbation
        self.input_rows = (
            shuffled_input_rows(self.entries, sources.users)
            if perturbation == "shuffle"
            else np.asarray([entry.row for entry in self.entries], dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, int]:
        entry = self.entries[index]
        row = int(self.input_rows[index])
        pair = torch.tensor(PAIRS[entry.pair_id], dtype=torch.long)

        def tensor(values: np.ndarray) -> torch.Tensor:
            value = torch.from_numpy(np.array(values[row], dtype=np.float32, copy=True))
            return torch.zeros_like(value) if self.perturbation == "zero" else value

        if self.variant == "layer2_20x20":
            batch = {"layer2": tensor(self.sources.layer2)}
        elif self.variant == "layer3_10x10":
            batch = {"layer3": tensor(self.sources.layer3)}
        elif self.variant in (
            "layer2_then_layer3",
            HIERARCHICAL_VARIANT,
            WORKSPACE_VARIANT,
        ):
            batch = {
                "layer2": tensor(self.sources.layer2),
                "layer3": tensor(self.sources.layer3),
            }
        elif self.variant == "layer4_5x5":
            batch = {"layer4": tensor(self.sources.layer4)}
        else:
            batch = {"pooled": tensor(self.sources.pooled)}
        return batch, pair, entry.target, index


def make_model(variant: str, sources: Sources, width: int) -> nn.Module:
    if variant == "layer2_20x20":
        return SpatialPairScorer((("layer2", LAYER2_GRID),), width)
    if variant == "layer3_10x10":
        return SpatialPairScorer((("layer3", LAYER3_GRID),), width)
    if variant == "layer2_then_layer3":
        return SpatialPairScorer(
            (("layer2", LAYER2_GRID), ("layer3", LAYER3_GRID)), width
        )
    if variant == "layer4_5x5":
        return SpatialPairScorer((("layer4", LAYER4_GRID),), width)
    if variant == HIERARCHICAL_VARIANT:
        return HierarchicalSpatialTemporalPairScorer(width)
    if variant == WORKSPACE_VARIANT:
        return HierarchicalSpatialTemporalPairScorer(width, view_indices=(2,))
    return PooledPairScorer(sources.pooled.shape[1], width)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def predict_entries(
    model: nn.Module,
    sources: Sources,
    variant: str,
    entries: Sequence[PairEntry],
    perturbation: str,
    args: argparse.Namespace,
) -> np.ndarray:
    dataset = PairDataset(sources, variant, entries, perturbation)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    output = np.empty(len(entries), dtype=np.int64)
    model.eval()
    for batch, candidates, _, indices in loader:
        batch = move_batch(batch, args.device)
        candidates = candidates.to(args.device, non_blocking=True)
        with torch.autocast(
            device_type="cuda" if args.device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda"),
        ):
            logits, _ = model(batch, candidates)
        output[indices.numpy()] = logits.argmax(1).cpu().numpy()
    return output


def fit_model(
    variant: str,
    sources: Sources,
    entries: Sequence[PairEntry],
    args: argparse.Namespace,
    seed: int,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    seed_everything(seed)
    model = make_model(variant, sources, args.width).to(args.device)
    loader = DataLoader(
        PairDataset(sources, variant, entries, "aligned"),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        correct = total = 0
        for batch, candidates, targets, _ in loader:
            batch = move_batch(batch, args.device)
            candidates = candidates.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if args.device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                logits, _ = model(batch, candidates)
                loss = F.cross_entropy(logits, targets, label_smoothing=0.03)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(1) == targets).sum().detach().cpu())
            total += len(targets)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "training_pair_accuracy": correct / max(total, 1),
        }
        history.append(record)
        print(json.dumps({"p116": variant, **record}), flush=True)
    return model, history


def metric_bundle(
    entries: Sequence[PairEntry],
    prediction: np.ndarray,
    sources: Sources,
) -> dict[str, Any]:
    target = np.asarray([entry.target for entry in entries], dtype=np.int64)
    pair_values: dict[str, float] = {}
    for pair_id, name in enumerate(PAIR_NAMES):
        selected = np.asarray([entry.pair_id == pair_id for entry in entries], dtype=bool)
        pair_values[name] = float(np.mean(prediction[selected] == target[selected]))
    subject_values: dict[str, float] = {}
    entry_users = np.asarray([sources.users[entry.row] for entry in entries], dtype=str)
    for subject in sorted(set(entry_users.tolist())):
        selected = entry_users == subject
        subject_values[subject] = float(np.mean(prediction[selected] == target[selected]))
    return {
        "rows": len(entries),
        "accuracy": float(np.mean(prediction == target)),
        "macro_pair_accuracy": float(np.mean(list(pair_values.values()))),
        "worst_pair_accuracy": float(min(pair_values.values())),
        "worst_subject_accuracy": float(min(subject_values.values())),
        "pair_accuracy": pair_values,
        "subject_accuracy": subject_values,
    }


def run_signature(args: argparse.Namespace, outer: str, variant: str) -> str:
    return json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "outer": outer,
            "variant": variant,
            "pairs": PAIRS,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "inner_folds": INNER_FOLDS,
        },
        sort_keys=True,
    )


def inner_groups(users: np.ndarray) -> tuple[tuple[str, ...], ...]:
    ordered = sorted(set(users.tolist()))
    return tuple(tuple(ordered[offset::INNER_FOLDS]) for offset in range(INNER_FOLDS))


def run_source_inner_variant(
    outer: str,
    variant: str,
    sources: Sources,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cache_path = args.output_dir / f"{outer}__{variant}_source_inner.json"
    signature = run_signature(args, outer, variant)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") != signature:
            raise RuntimeError(f"P116 stale source-inner result: {cache_path}")
        return cached["result"]
    outer_users = set(OUTER_USERS[outer])
    source_users = sources.users[~np.isin(sources.users, list(outer_users))]
    all_entries = build_entries(sources.labels, ~np.isin(sources.users, list(outer_users)))
    predictions = {
        perturbation: np.full(len(all_entries), -1, dtype=np.int64)
        for perturbation in PERTURBATIONS
    }
    audit: list[dict[str, Any]] = []
    for inner, group in enumerate(inner_groups(source_users)):
        train_mask = ~np.isin(sources.users, [*outer_users, *group])
        eval_mask = np.isin(sources.users, list(group)) & ~np.isin(
            sources.users, list(outer_users)
        )
        train_entries = build_entries(sources.labels, train_mask)
        eval_entries = build_entries(sources.labels, eval_mask)
        model, history = fit_model(
            variant,
            sources,
            train_entries,
            args,
            SEED + list(OUTER_USERS).index(outer) * 1000 + VARIANTS.index(variant) * 100 + inner,
        )
        eval_lookup = {
            (entry.row, entry.pair_id): index for index, entry in enumerate(all_entries)
        }
        target_indices = np.asarray(
            [eval_lookup[(entry.row, entry.pair_id)] for entry in eval_entries], dtype=np.int64
        )
        for perturbation in PERTURBATIONS:
            predictions[perturbation][target_indices] = predict_entries(
                model, sources, variant, eval_entries, perturbation, args
            )
        audit.append(
            {
                "inner_fold": inner,
                "held_users": list(group),
                "train_entries": len(train_entries),
                "eval_entries": len(eval_entries),
                "last_epoch": history[-1],
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if any(np.any(value < 0) for value in predictions.values()):
        raise RuntimeError("P116 source inner OOF coverage incomplete")
    metrics = {
        perturbation: metric_bundle(all_entries, prediction, sources)
        for perturbation, prediction in predictions.items()
    }
    aligned = metrics["aligned"]
    gap = aligned["macro_pair_accuracy"] - max(
        metrics["shuffle"]["macro_pair_accuracy"],
        metrics["zero"]["macro_pair_accuracy"],
    )
    result = {
        "variant": variant,
        "metrics": metrics,
        "sample_specific_gap": float(gap),
        "inner_audit": audit,
    }
    cache_path.write_text(
        json.dumps({"signature": signature, "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def source_selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    aligned = result["metrics"]["aligned"]
    gap = float(result["sample_specific_gap"])
    return (
        float(aligned["macro_pair_accuracy"]) + 0.5 * max(-0.20, min(0.20, gap)),
        gap,
        float(aligned["macro_pair_accuracy"]),
        float(aligned["worst_subject_accuracy"]),
    )


def fit_outer_variant(
    outer: str,
    variant: str,
    sources: Sources,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = args.output_dir
    path = output / f"{outer}__{variant}_held_predictions.npz"
    signature = run_signature(args, outer, variant)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["signature"].item()) != signature:
                raise RuntimeError(f"P116 stale outer result: {path}")
            return {key: np.asarray(archive[key]) for key in archive.files}
    held_users = set(OUTER_USERS[outer])
    train_entries = build_entries(
        sources.labels, ~np.isin(sources.users, list(held_users))
    )
    held_entries = build_entries(
        sources.labels, np.isin(sources.users, list(held_users))
    )
    model, history = fit_model(
        variant,
        sources,
        train_entries,
        args,
        SEED + list(OUTER_USERS).index(outer) * 1000 + VARIANTS.index(variant) * 100 + 99,
    )
    predictions = {
        perturbation: predict_entries(
            model, sources, variant, held_entries, perturbation, args
        )
        for perturbation in PERTURBATIONS
    }
    checkpoint = output / f"{outer}__{variant}_final.pt"
    torch.save(
        {
            "protocol": PROTOCOL_VERSION,
            "outer": outer,
            "variant": variant,
            "model_state": model.state_dict(),
            "history": history,
            "pairs": PAIRS,
        },
        checkpoint,
    )
    payload: dict[str, Any] = {
        "signature": np.asarray(signature),
        "rows": np.asarray([entry.row for entry in held_entries], dtype=np.int64),
        "pair_ids": np.asarray([entry.pair_id for entry in held_entries], dtype=np.int64),
        "targets": np.asarray([entry.target for entry in held_entries], dtype=np.int64),
        "aligned": predictions["aligned"],
        "shuffle": predictions["shuffle"],
        "zero": predictions["zero"],
        "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        "checkpoint_bytes": np.asarray(checkpoint.stat().st_size),
    }
    np.savez_compressed(path, **payload)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def entries_from_payload(payload: dict[str, Any]) -> list[PairEntry]:
    return [
        PairEntry(int(row), int(pair_id), int(target))
        for row, pair_id, target in zip(
            payload["rows"], payload["pair_ids"], payload["targets"]
        )
    ]


def sample_rows(
    outer: str,
    variant: str,
    payload: dict[str, Any],
    sources: Sources,
    source_selected_highres: str,
) -> list[dict[str, Any]]:
    entries = entries_from_payload(payload)
    output: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        for perturbation in PERTURBATIONS:
            prediction = int(payload[perturbation][index])
            output.append(
                {
                    "sample_id": sources.ids[entry.row],
                    "subject": sources.users[entry.row],
                    "outer_fold": outer,
                    "variant": variant,
                    "source_selected_highres": int(variant == source_selected_highres),
                    "pair": PAIR_NAMES[entry.pair_id],
                    "true_label": int(sources.labels[entry.row]),
                    "perturbation": perturbation,
                    "prediction": int(PAIRS[entry.pair_id][prediction]),
                    "correct": int(prediction == entry.target),
                }
            )
    return output


def aggregate_sample_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    accuracy = float(np.mean([int(row["correct"]) for row in rows]))
    pair_accuracy = {
        pair: float(
            np.mean([int(row["correct"]) for row in rows if row["pair"] == pair])
        )
        for pair in PAIR_NAMES
    }
    subject_accuracy = {
        subject: float(
            np.mean([int(row["correct"]) for row in rows if row["subject"] == subject])
        )
        for subject in sorted({str(row["subject"]) for row in rows})
    }
    return {
        "rows": len(rows),
        "accuracy": accuracy,
        "macro_pair_accuracy": float(np.mean(list(pair_accuracy.values()))),
        "worst_pair_accuracy": float(min(pair_accuracy.values())),
        "worst_subject_accuracy": float(min(subject_accuracy.values())),
        "pair_accuracy": pair_accuracy,
        "subject_accuracy": subject_accuracy,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol()
    sources = load_sources(args)
    order = _align(sources.ids, protocol.ids)
    if int(np.sum(protocol.safe == protocol.labels)) != 2117:
        raise RuntimeError("P89 changed")
    if not np.array_equal(sources.labels[order], protocol.labels):
        raise RuntimeError("P116/P89 label alignment differs")
    folds = tuple(value.strip() for value in args.folds.split(",") if value.strip())
    if not set(folds) <= set(OUTER_USERS):
        raise ValueError("unknown P116 outer fold")

    # Freeze every source-only recipe before any outer-held model is evaluated.
    source_results: dict[str, dict[str, dict[str, Any]]] = {}
    selection_rows: list[dict[str, Any]] = []
    selected_highres: dict[str, str] = {}
    for outer in folds:
        source_results[outer] = {}
        for variant in VARIANTS:
            print(json.dumps({"p116_source_inner": outer, "variant": variant}), flush=True)
            result = run_source_inner_variant(outer, variant, sources, args)
            source_results[outer][variant] = result
            for perturbation in PERTURBATIONS:
                metric = result["metrics"][perturbation]
                selection_rows.append(
                    {
                        "outer_fold": outer,
                        "variant": variant,
                        "perturbation": perturbation,
                        "macro_pair_accuracy": metric["macro_pair_accuracy"],
                        "worst_pair_accuracy": metric["worst_pair_accuracy"],
                        "worst_subject_accuracy": metric["worst_subject_accuracy"],
                        "sample_specific_gap": result["sample_specific_gap"],
                    }
                )
        selected_highres[outer] = max(
            (source_results[outer][variant] for variant in HIGHRES_VARIANTS),
            key=source_selection_key,
        )["variant"]
        for row in selection_rows:
            if row["outer_fold"] == outer:
                row["source_selected_highres"] = int(
                    row["variant"] == selected_highres[outer]
                )

    all_samples: list[dict[str, Any]] = []
    for outer in folds:
        for variant in VARIANTS:
            print(json.dumps({"p116_outer": outer, "variant": variant}), flush=True)
            payload = fit_outer_variant(outer, variant, sources, args)
            all_samples.extend(
                sample_rows(
                    outer,
                    variant,
                    payload,
                    sources,
                    selected_highres[outer],
                )
            )

    held_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for perturbation in PERTURBATIONS:
            selected = [
                row
                for row in all_samples
                if row["variant"] == variant and row["perturbation"] == perturbation
            ]
            metric = aggregate_sample_metrics(selected)
            held_rows.append(
                {
                    "scope": "aggregate",
                    "outer_fold": "ALL",
                    "variant": variant,
                    "perturbation": perturbation,
                    **{key: value for key, value in metric.items() if not isinstance(value, dict)},
                }
            )
            for outer in folds:
                fold_selected = [row for row in selected if row["outer_fold"] == outer]
                fold_metric = aggregate_sample_metrics(fold_selected)
                held_rows.append(
                    {
                        "scope": "fold",
                        "outer_fold": outer,
                        "variant": variant,
                        "perturbation": perturbation,
                        **{
                            key: value
                            for key, value in fold_metric.items()
                            if not isinstance(value, dict)
                        },
                    }
                )
            if perturbation == "aligned":
                for subject, accuracy in metric["subject_accuracy"].items():
                    subject_rows.append(
                        {"variant": variant, "subject": subject, "accuracy": accuracy}
                    )
                for pair, accuracy in metric["pair_accuracy"].items():
                    pair_rows.append(
                        {"variant": variant, "pair": pair, "accuracy": accuracy}
                    )

    selected_samples = [
        row
        for row in all_samples
        if row["variant"] == selected_highres[row["outer_fold"]]
    ]
    selected_metrics = {
        perturbation: aggregate_sample_metrics(
            [row for row in selected_samples if row["perturbation"] == perturbation]
        )
        for perturbation in PERTURBATIONS
    }
    selected_fold_metrics = {
        outer: {
            perturbation: aggregate_sample_metrics(
                [
                    row
                    for row in selected_samples
                    if row["outer_fold"] == outer
                    and row["perturbation"] == perturbation
                ]
            )
            for perturbation in PERTURBATIONS
        }
        for outer in folds
    }
    control_metrics = {
        variant: {
            perturbation: aggregate_sample_metrics(
                [
                    row
                    for row in all_samples
                    if row["variant"] == variant
                    and row["perturbation"] == perturbation
                ]
            )
            for perturbation in PERTURBATIONS
        }
        for variant in CONTROL_VARIANTS
    }
    aligned = selected_metrics["aligned"]
    aligned_gap = aligned["macro_pair_accuracy"] - max(
        selected_metrics["shuffle"]["macro_pair_accuracy"],
        selected_metrics["zero"]["macro_pair_accuracy"],
    )
    best_control = max(
        control_metrics,
        key=lambda variant: control_metrics[variant]["aligned"]["macro_pair_accuracy"],
    )
    control_gap = (
        aligned["macro_pair_accuracy"]
        - control_metrics[best_control]["aligned"]["macro_pair_accuracy"]
    )
    subject_control_comparisons = []
    for variant in CONTROL_VARIANTS:
        shared_subjects = sorted(
            set(aligned["subject_accuracy"])
            & set(control_metrics[variant]["aligned"]["subject_accuracy"])
        )
        differences = np.asarray(
            [
                aligned["subject_accuracy"][subject]
                - control_metrics[variant]["aligned"]["subject_accuracy"][subject]
                for subject in shared_subjects
            ],
            dtype=np.float64,
        )
        subject_control_comparisons.append(
            {
                "control": variant,
                "subjects": len(shared_subjects),
                "mean_subject_accuracy_gain": float(differences.mean()),
                "subject_bootstrap_95ci": bootstrap_mean_ci(
                    differences, SEED + CONTROL_VARIANTS.index(variant)
                ),
            }
        )
    selected_fold_aligned = [
        aggregate_sample_metrics(
            [
                row
                for row in selected_samples
                if row["outer_fold"] == outer and row["perturbation"] == "aligned"
            ]
        )
        for outer in folds
    ]
    stable_strong = bool(
        aligned["macro_pair_accuracy"] >= 0.65
        and aligned_gap >= 0.05
        and control_gap >= 0.05
        and all(metric["macro_pair_accuracy"] >= 0.60 for metric in selected_fold_aligned)
        and all(
            metric["macro_pair_accuracy"]
            > aggregate_sample_metrics(
                [
                    row
                    for row in selected_samples
                    if row["outer_fold"] == outer and row["perturbation"] == "shuffle"
                ]
            )["macro_pair_accuracy"]
            for outer, metric in zip(folds, selected_fold_aligned)
        )
    )
    pair_verdicts = {}
    for pair in PAIR_NAMES:
        highres_value = aligned["pair_accuracy"][pair]
        control_value = max(
            control_metrics[variant]["aligned"]["pair_accuracy"][pair]
            for variant in CONTROL_VARIANTS
        )
        shuffle_value = selected_metrics["shuffle"]["pair_accuracy"][pair]
        fold_values = {
            outer: selected_fold_metrics[outer]["aligned"]["pair_accuracy"][pair]
            for outer in folds
        }
        pair_verdicts[pair] = {
            "highres_accuracy": highres_value,
            "best_control_accuracy": control_value,
            "shuffle_accuracy": shuffle_value,
            "highres_gain": highres_value - control_value,
            "sample_specific_gap": highres_value - shuffle_value,
            "fold_accuracy": fold_values,
            "stable_evidence": bool(
                highres_value >= 0.65
                and highres_value - control_value >= 0.05
                and highres_value - shuffle_value >= 0.05
            ),
        }
    complete = set(folds) == set(OUTER_USERS)
    summary = {
        "stage": "P116_highres_pair_capability",
        "status": "complete" if complete else "partial",
        "decision": (
            "HIGHRES_SAMPLE_SPECIFIC_EVIDENCE_CONFIRMED"
            if stable_strong
            else "STOP_INTERNAL_VISUAL_MINING_USE_EXTERNAL_HAND_OBJECT_PRETRAINING"
        ),
        "p89": {
            "frozen": True,
            "correct": 2117,
            "rows": 2470,
            "accuracy": 2117 / 2470,
            "modified": False,
        },
        "fixed_pairs": list(PAIR_NAMES),
        "representation": {
            "layer2": "2x16x3x20x20 = 38400 tokens; candidate query before pooling",
            "layer3": "2x16x3x10x10 = 9600 tokens; candidate query before pooling",
            "workspace_view": "P86 hand_workspace ROI crop at scale 1.40",
            "controls": list(CONTROL_VARIANTS),
        },
        "protocol": {
            "source_inner_selection_only": True,
            "held_variant_selection": False,
            "aligned_shuffle_zero": True,
            "epochs": args.epochs,
            "batch_size_all_variants": args.batch_size,
            "seed_sweep": False,
            "threshold_sweep": False,
            "test_rows_loaded": False,
            "final_replacement": False,
        },
        "source_selected_highres_by_fold": selected_highres,
        "selected_highres_metrics": selected_metrics,
        "selected_highres_fold_metrics": selected_fold_metrics,
        "control_metrics": control_metrics,
        "subject_control_comparisons": subject_control_comparisons,
        "selected_highres_sample_specific_gap": aligned_gap,
        "selected_highres_gain_over_best_control": control_gap,
        "pair_verdicts": pair_verdicts,
        "stable_strong_evidence": stable_strong,
        "next_step": (
            "source-safe ROI/temporal refinement before Top-5 audit"
            if stable_strong
            else "public hand-object / egocentric interaction pretraining"
        ),
        "final_decision_metrics": {"rescue": None, "harm": None, "net": None},
    }
    selected_metric_rows = []
    for perturbation, metric in selected_metrics.items():
        selected_metric_rows.append(
            {
                "scope": "aggregate",
                "outer_fold": "ALL",
                "selected_variant": "source_crossfit",
                "perturbation": perturbation,
                "macro_pair_accuracy": metric["macro_pair_accuracy"],
                "worst_pair_accuracy": metric["worst_pair_accuracy"],
                "worst_subject_accuracy": metric["worst_subject_accuracy"],
            }
        )
    for outer in folds:
        for perturbation, metric in selected_fold_metrics[outer].items():
            selected_metric_rows.append(
                {
                    "scope": "fold",
                    "outer_fold": outer,
                    "selected_variant": selected_highres[outer],
                    "perturbation": perturbation,
                    "macro_pair_accuracy": metric["macro_pair_accuracy"],
                    "worst_pair_accuracy": metric["worst_pair_accuracy"],
                    "worst_subject_accuracy": metric["worst_subject_accuracy"],
                }
            )
    write_csv(args.output_dir / "source_inner_metrics.csv", selection_rows)
    write_csv(args.output_dir / "held_metrics.csv", held_rows)
    write_csv(args.output_dir / "held_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / "held_pair_metrics.csv", pair_rows)
    write_csv(args.output_dir / "selected_highres_metrics.csv", selected_metric_rows)
    write_csv(args.output_dir / "sample_predictions.csv", all_samples)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    for name in ("pixel_cache", "highres_cache", "layer4_cache", "vjepa_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.stage in ("all", "tokens"):
        build_highres_cache(args)
    if args.stage in ("all", "audit"):
        run_audit(args)


if __name__ == "__main__":
    main()
