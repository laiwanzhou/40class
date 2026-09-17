"""P115 full-token visual candidate-elimination cascade over frozen P89.

The primary model keeps every ResNet18 layer-4 spatial token from all 32
frames and all scene/person/workspace views until a candidate-class query
attends to the token sequence.  It only removes low-compatibility candidates;
it never emits a replacement for the frozen P89 Top-1 decision.

Two pooled controls use the same outer/inner subject protocol and the same
fixed conformal threshold rule: a same-backbone mean/std/phase descriptor and
the existing P108 VLIT/VHPD/VWPD V-JEPA2 descriptors.  No held label selects a
threshold, model, epoch, seed, or variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
from p103_local_feature_data import MASTER_MANIFEST
from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_PIXELS = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_TOKENS = HERE / "runs/p115_resnet18_full_tokens_v1"
DEFAULT_VJEPA = PROJECT / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
DEFAULT_OUTPUT = HERE / "runs/p115_full_token_candidate_elimination_v2"

SEED = 20260824
CLASSES = 40
INNER_FOLDS = 3
HIGH_CONFIDENCE_QUANTILE = 0.75
TRUE_CANDIDATE_ALPHA = 0.01
MINIMUM_SURVIVORS = 2
VARIANTS = (
    "full_token",
    "same_backbone_pooled",
    "pooled_vlit_vhpd_vwpd",
)
PROTOCOL_VERSION = "p115_full_token_candidate_elimination_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "tokens", "audit"), default="all")
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--vjepa-root", type=Path, default=DEFAULT_VJEPA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", default=",".join(OUTER_USERS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=6)
    # Equal batch size gives all variants the same optimizer-step budget per
    # epoch; a larger pooled batch would be an under-trained control.
    parser.add_argument("--pooled-batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


@dataclass(frozen=True)
class TokenGrid:
    windows: int = 2
    times: int = 16
    views: int = 3
    height: int = 5
    width: int = 5
    channels: int = 512

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


class FullTokenCandidateScorer(nn.Module):
    """Candidate query aggregates evidence only after all W×T×V×H×W tokens exist."""

    def __init__(
        self,
        grid: TokenGrid = TokenGrid(),
        width: int = 96,
        heads: int = 4,
        classes: int = CLASSES,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must divide attention heads")
        self.grid = grid
        self.classes = classes
        self.content = nn.Sequential(nn.LayerNorm(grid.channels), nn.Linear(grid.channels, width))
        self.window_embedding = nn.Embedding(grid.windows, width)
        self.time_embedding = nn.Embedding(grid.times, width)
        self.view_embedding = nn.Embedding(grid.views, width)
        self.row_embedding = nn.Embedding(grid.height, width)
        self.column_embedding = nn.Embedding(grid.width, width)
        self.class_query = nn.Embedding(classes, width)
        self.query_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 3, width),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )
        for embedding in (
            self.window_embedding,
            self.time_embedding,
            self.view_embedding,
            self.row_embedding,
            self.column_embedding,
            self.class_query,
        ):
            nn.init.trunc_normal_(embedding.weight, std=0.02)

    def _position(self, device: torch.device) -> torch.Tensor:
        g = self.grid
        value = (
            self.window_embedding(torch.arange(g.windows, device=device))[:, None, None, None, None]
            + self.time_embedding(torch.arange(g.times, device=device))[None, :, None, None, None]
            + self.view_embedding(torch.arange(g.views, device=device))[None, None, :, None, None]
            + self.row_embedding(torch.arange(g.height, device=device))[None, None, None, :, None]
            + self.column_embedding(torch.arange(g.width, device=device))[None, None, None, None, :]
        )
        return value.reshape(g.token_count, -1)

    def forward(
        self,
        tokens: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        expected = self.grid.shape
        if tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"token grid changed: {tuple(tokens.shape[1:])} != {expected}")
        batch = len(tokens)
        if candidate_ids is None:
            candidate_ids = torch.arange(self.classes, device=tokens.device)[None].expand(batch, -1)
        content = self.content(tokens).reshape(batch, self.grid.token_count, -1)
        content = content + self._position(tokens.device)[None]
        query = self.class_query(candidate_ids)
        attended, attention = self.attention(
            self.query_norm(query),
            self.token_norm(content),
            self.token_norm(content),
            need_weights=return_attention,
            average_attn_weights=False,
        )
        state = query + attended
        state = state + self.feed_forward(state)
        logits = self.score(state).squeeze(-1)
        return logits, attention


class PooledCandidateScorer(nn.Module):
    """Shared candidate-conditioned control after an explicitly pooled descriptor."""

    def __init__(
        self,
        input_dim: int,
        width: int = 96,
        classes: int = CLASSES,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.visual = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU())
        self.class_query = nn.Embedding(classes, width)
        self.score = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, 1),
        )
        nn.init.trunc_normal_(self.class_query.weight, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        candidate_ids: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if return_attention:
            raise ValueError("pooled control has no token attention")
        batch = len(values)
        if candidate_ids is None:
            candidate_ids = torch.arange(self.classes, device=values.device)[None].expand(batch, -1)
        visual = self.visual(values)[:, None].expand(-1, candidate_ids.shape[1], -1)
        query = self.class_query(candidate_ids)
        interaction = torch.cat((visual, query, visual * query, torch.abs(visual - query)), dim=-1)
        return self.score(interaction).squeeze(-1), None


class FrameDataset(Dataset):
    def __init__(self, images_path: Path) -> None:
        self.images = np.load(images_path, mmap_mode="r")
        if tuple(self.images.shape[1:]) != (2, 16, 3, 160, 160):
            raise RuntimeError(f"P115 pixel cache changed: {self.images.shape}")
        self.per_row = 2 * 16 * 3
        self.mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]

    def __len__(self) -> int:
        return len(self.images) * self.per_row

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row, offset = divmod(index, self.per_row)
        window, offset = divmod(offset, 16 * 3)
        time, view = divmod(offset, 3)
        raw = np.array(self.images[row, window, time, view], copy=True)
        tensor = torch.from_numpy(raw).float().div_(255.0)[None].repeat(3, 1, 1)
        return (tensor - self.mean) / self.std, index


def _open_memmap(path: Path, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.memmap:
    if path.exists():
        value = np.lib.format.open_memmap(path, mode="r+")
        if value.dtype != dtype or value.shape != shape:
            raise RuntimeError(f"incompatible cache {path}: {value.shape}/{value.dtype}")
        return value
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value[...] = False if dtype == np.dtype(np.bool_) else 0
    value.flush()
    return value


@torch.inference_mode()
def build_token_cache(args: argparse.Namespace) -> dict[str, Any]:
    output = args.token_cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pixel_rows = read_csv(args.pixel_cache / "rows.csv")
    grid = TokenGrid()
    row_count = len(pixel_rows)
    token_shape = (row_count, *grid.shape)
    features = _open_memmap(output / "full_tokens.npy", np.dtype(np.float16), token_shape)
    completed = _open_memmap(
        output / "completed_frames.npy", np.dtype(np.bool_), (row_count * 2 * 16 * 3,)
    )
    dataset = FrameDataset(args.pixel_cache / "images.npy")
    if len(dataset) != len(completed):
        raise RuntimeError("frame inventory and completion bitmap differ")
    pending = np.flatnonzero(~np.asarray(completed, dtype=bool))
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(backbone.children())[:-2]).to(args.device).eval()
    loader = DataLoader(
        torch.utils.data.Subset(dataset, pending.tolist()),
        batch_size=args.extract_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    flat = features.reshape(len(completed), grid.height, grid.width, grid.channels)
    processed = 0
    for images, indices in loader:
        images = images.to(args.device, non_blocking=True)
        with torch.autocast(
            device_type="cuda" if args.device.startswith("cuda") else "cpu",
            dtype=torch.float16,
            enabled=args.device.startswith("cuda"),
        ):
            value = encoder(images)
        if tuple(value.shape[1:]) != (grid.channels, grid.height, grid.width):
            raise RuntimeError(f"ResNet spatial token shape changed: {tuple(value.shape)}")
        value = value.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)
        selected = indices.numpy().astype(np.int64)
        flat[selected] = value
        completed[selected] = True
        processed += len(selected)
        if processed % (args.extract_batch_size * 50) < args.extract_batch_size:
            features.flush()
            completed.flush()
            print(
                json.dumps(
                    {"p115_tokens": int(np.sum(completed)), "total": len(completed)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    features.flush()
    completed.flush()
    if not np.asarray(completed, dtype=bool).all():
        raise RuntimeError("P115 full-token extraction incomplete")

    pooled_shape = (row_count, 3 * 512 * 4)
    pooled = _open_memmap(output / "same_backbone_pooled.npy", np.dtype(np.float16), pooled_shape)
    for start in range(0, row_count, 16):
        stop = min(start + 16, row_count)
        value = np.asarray(features[start:stop], dtype=np.float32)
        mean = value.mean(axis=(1, 2, 4, 5))
        std = value.std(axis=(1, 2, 4, 5))
        early = value[:, 0].mean(axis=(1, 3, 4))
        late = value[:, 1].mean(axis=(1, 3, 4))
        pooled[start:stop] = np.concatenate((mean, std, early, late - early), axis=1).reshape(
            stop - start, -1
        )
    pooled.flush()
    summary = {
        "protocol": PROTOCOL_VERSION,
        "complete": True,
        "label_free_extraction": True,
        "backbone": "torchvision ResNet18 ImageNet1K V1 frozen through layer4",
        "input_views": ["scene", "person", "workspace"],
        "input_windows": ["early", "late"],
        "frames_per_window": 16,
        "global_spatial_pooling_before_cache": False,
        "temporal_pooling_before_cache": False,
        "token_grid": asdict(grid),
        "token_count_per_trial": grid.token_count,
        "shape": list(token_shape),
        "dtype": "float16",
        "same_backbone_pooled_shape": list(pooled_shape),
        "rows": row_count,
        "cache_bytes": int((output / "full_tokens.npy").stat().st_size),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


@dataclass(frozen=True)
class Protocol:
    ids: np.ndarray
    users: np.ndarray
    labels: np.ndarray
    safe: np.ndarray
    probability: np.ndarray
    fold_names: np.ndarray


def build_protocol() -> Protocol:
    splits = load_splits()
    names = tuple(OUTER_USERS)
    protocol = Protocol(
        ids=np.concatenate([splits[name].sample_ids.astype(str) for name in names]),
        users=np.concatenate([splits[name].users.astype(str) for name in names]),
        labels=np.concatenate([splits[name].labels.astype(np.int64) for name in names]),
        safe=np.concatenate([splits[name].safe_prediction.astype(np.int64) for name in names]),
        probability=np.concatenate([splits[name].safe_probability.astype(np.float64) for name in names]),
        fold_names=np.concatenate([np.repeat(name, len(splits[name].labels)) for name in names]).astype(str),
    )
    if len(protocol.ids) != 2470 or int(np.sum(protocol.safe == protocol.labels)) != 2117:
        raise RuntimeError("frozen P89 baseline changed from 2117/2470")
    if len(np.unique(protocol.ids)) != len(protocol.ids):
        raise RuntimeError("duplicate P89 OOF sample IDs")
    return protocol


def anchored_topk(probability: np.ndarray, anchor: np.ndarray, k: int) -> np.ndarray:
    if k < 1 or probability.ndim != 2 or len(anchor) != len(probability):
        raise ValueError("invalid anchored Top-k inputs")
    order = np.argsort(-probability, axis=1, kind="stable")
    output = np.empty((len(probability), k), dtype=np.int64)
    for row in range(len(probability)):
        values = [int(anchor[row])]
        values.extend(int(value) for value in order[row] if int(value) != int(anchor[row]))
        output[row] = values[:k]
    return output


def fixed_quantile(values: np.ndarray, quantile: float) -> float:
    source = np.asarray(values, dtype=np.float64)
    if not len(source) or not np.isfinite(source).all():
        raise ValueError("quantile source is empty or non-finite")
    try:
        return float(np.quantile(source, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(source, quantile, interpolation="higher"))


def conformal_lower_threshold(values: np.ndarray, alpha: float = TRUE_CANDIDATE_ALPHA) -> float:
    """Fixed lower-tail conformal cutoff; strict '<' controls empirical false kill."""
    source = np.sort(np.asarray(values, dtype=np.float64))
    if not len(source) or not np.isfinite(source).all() or not 0.0 < alpha < 1.0:
        raise ValueError("invalid conformal threshold inputs")
    rank = max(1, int(math.floor(alpha * (len(source) + 1))))
    rank = min(rank, len(source))
    return float(source[rank - 1])


def recursive_eliminate(
    candidates: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    anchor: int,
    minimum_survivors: int = MINIMUM_SURVIVORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove incompatible candidates from weakest upward, protecting P89 and a leaf floor."""
    candidate = np.asarray(candidates, dtype=np.int64)
    compatibility = np.asarray(scores, dtype=np.float64)
    if candidate.ndim != 1 or compatibility.shape != candidate.shape:
        raise ValueError("candidate and score shapes differ")
    if int(anchor) not in set(candidate.tolist()):
        raise ValueError("P89 anchor missing from candidate set")
    keep = np.ones(len(candidate), dtype=bool)
    for index in np.argsort(compatibility, kind="stable"):
        if int(candidate[index]) == int(anchor):
            continue
        if compatibility[index] >= threshold or int(keep.sum()) <= minimum_survivors:
            continue
        keep[index] = False
    return candidate[keep], candidate[~keep]


def load_p108_pooled_visual(vjepa_root: Path, pixel_ids: np.ndarray) -> np.ndarray:
    summary = json.loads((vjepa_root / "cache_summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete") or not summary.get("label_free_extraction"):
        raise RuntimeError("P108 V-JEPA2 cache is incomplete or not label-free")
    view_names = tuple(map(str, summary["view_names"]))
    with MASTER_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_ids = np.asarray([row["sample_id"] for row in csv.DictReader(handle)], dtype=str)
    order = _align(manifest_ids, pixel_ids)
    feature_memmap = np.load(vjepa_root / "features.npy", mmap_mode="r")
    action_memmap = np.load(vjepa_root / "ssv2_logits.npy", mmap_mode="r")
    view_lookup = {name: index for index, name in enumerate(view_names)}

    interaction = [f"hand_{phase}_interaction" for phase in ("full", "early", "late", "motion_peak")]
    hand_names = [f"hand_{phase}_{part}" for part in ("left", "right", "interaction") for phase in ("full", "early", "late", "motion_peak")]
    workspace = [f"global_{phase}_workspace" for phase in ("full", "early", "middle", "late")]
    required = tuple(dict.fromkeys([*interaction, *hand_names, *workspace]))
    indices = np.asarray([view_lookup[name] for name in required], dtype=np.int64)
    features = np.asarray(feature_memmap[order[:, None], indices[None]], dtype=np.float32)
    actions = np.asarray(action_memmap[order[:, None], indices[None]], dtype=np.float32)
    selected = {name: index for index, name in enumerate(required)}

    def take(values: np.ndarray, names: Sequence[str]) -> np.ndarray:
        return values[:, [selected[name] for name in names]]

    blocks: list[np.ndarray] = [take(features, interaction), take(actions, interaction)]
    for values in (features, actions):
        deltas = []
        for part in ("left", "right", "interaction"):
            deltas.extend(
                (
                    take(values, [f"hand_late_{part}"]) - take(values, [f"hand_early_{part}"]),
                    take(values, [f"hand_motion_peak_{part}"]) - take(values, [f"hand_full_{part}"]),
                )
            )
        blocks.extend(deltas)
    for values in (features, actions):
        blocks.extend(
            (
                take(values, workspace),
                take(values, ["global_late_workspace"]) - take(values, ["global_early_workspace"]),
                take(values, ["global_middle_workspace"]) - take(values, ["global_early_workspace"]),
            )
        )
    return np.concatenate([value.reshape(len(pixel_ids), -1) for value in blocks], axis=1).astype(
        np.float32, copy=False
    )


class TrainingDataset(Dataset):
    def __init__(self, values: np.ndarray, rows: np.ndarray, labels: np.ndarray) -> None:
        self.values = values
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = int(self.rows[index])
        value = torch.from_numpy(np.array(self.values[row], dtype=np.float32, copy=True))
        return value, int(self.labels[row])


class PredictionDataset(Dataset):
    def __init__(self, values: np.ndarray, rows: np.ndarray) -> None:
        self.values = values
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = int(self.rows[index])
        return torch.from_numpy(np.array(self.values[row], dtype=np.float32, copy=True)), index


@dataclass
class FeatureSources:
    pixel_ids: np.ndarray
    pixel_users: np.ndarray
    pixel_labels: np.ndarray
    values: dict[str, np.ndarray]


def load_feature_sources(args: argparse.Namespace) -> FeatureSources:
    rows = read_csv(args.pixel_cache / "rows.csv")
    pixel_ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
    pixel_users = np.asarray([row["user_id"] for row in rows], dtype=str)
    pixel_labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    token_summary = json.loads((args.token_cache / "summary.json").read_text(encoding="utf-8"))
    if not token_summary.get("complete") or token_summary.get(
        "global_spatial_pooling_before_cache"
    ):
        raise RuntimeError("P115 full-token cache contract failed")
    values = {
        "full_token": np.load(args.token_cache / "full_tokens.npy", mmap_mode="r"),
        "same_backbone_pooled": np.load(args.token_cache / "same_backbone_pooled.npy", mmap_mode="r"),
        "pooled_vlit_vhpd_vwpd": load_p108_pooled_visual(args.vjepa_root, pixel_ids),
    }
    if values["full_token"].shape != (len(rows), *TokenGrid().shape):
        raise RuntimeError("full-token tensor shape changed")
    if any(len(value) != len(rows) for value in values.values()):
        raise RuntimeError("visual source row counts differ")
    return FeatureSources(pixel_ids, pixel_users, pixel_labels, values)


def make_model(variant: str, values: np.ndarray, args: argparse.Namespace) -> nn.Module:
    if variant == "full_token":
        return FullTokenCandidateScorer(TokenGrid(), args.width, args.heads)
    return PooledCandidateScorer(int(np.prod(values.shape[1:])), args.width)


@torch.inference_mode()
def predict_scores(
    model: nn.Module,
    values: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    workers: int,
    device: str,
) -> np.ndarray:
    dataset = PredictionDataset(values, rows)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
    )
    output = np.empty((len(rows), CLASSES), dtype=np.float32)
    model.eval()
    for batch, indices in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            logits, _ = model(batch)
        output[indices.numpy()] = torch.sigmoid(logits.float()).cpu().numpy()
    return output


def fit_and_predict(
    variant: str,
    values: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    checkpoint: Path | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    seed_everything(seed)
    model = make_model(variant, values, args).to(args.device)
    batch_size = args.batch_size if variant == "full_token" else args.pooled_batch_size
    dataset = TrainingDataset(values, train_rows, labels)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    positive_weight = torch.full((CLASSES,), CLASSES - 1.0, device=args.device)
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        correct = total = 0
        for batch, target in loader:
            batch = batch.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if args.device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                logits, _ = model(batch)
                binary = torch.zeros_like(logits)
                binary.scatter_(1, target[:, None], 1.0)
                loss = F.binary_cross_entropy_with_logits(
                    logits, binary, pos_weight=positive_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(1) == target).sum().detach().cpu())
            total += len(target)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "training_top1": correct / max(total, 1),
        }
        history.append(record)
        print(json.dumps({"p115": variant, **record}), flush=True)
    scores = predict_scores(model, values, predict_rows, batch_size, args.workers, args.device)
    checkpoint_bytes = 0
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol": PROTOCOL_VERSION,
                "variant": variant,
                "model_state": model.state_dict(),
                "width": args.width,
                "heads": args.heads,
                "epochs": args.epochs,
                "seed": seed,
                "train_rows": len(train_rows),
            },
            checkpoint,
        )
        checkpoint_bytes = checkpoint.stat().st_size
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores, history, checkpoint_bytes


def inner_user_groups(source_users: np.ndarray, count: int = INNER_FOLDS) -> tuple[tuple[str, ...], ...]:
    users = sorted(set(map(str, source_users.tolist())))
    groups = tuple(tuple(users[offset::count]) for offset in range(count))
    if any(not group for group in groups) or set().union(*map(set, groups)) != set(users):
        raise RuntimeError("invalid source inner-user partition")
    return groups


def _result_path(output: Path, outer: str, variant: str) -> Path:
    return output / f"{outer}__{variant}_scores.npz"


def run_signature(args: argparse.Namespace, variant: str) -> str:
    return json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "variant": variant,
            "epochs": args.epochs,
            "batch_size": args.batch_size if variant == "full_token" else args.pooled_batch_size,
            "width": args.width,
            "heads": args.heads,
            "learning_rate": args.learning_rate,
            "inner_folds": INNER_FOLDS,
        },
        sort_keys=True,
    )


def train_outer_variant(
    outer: str,
    variant: str,
    protocol: Protocol,
    features: FeatureSources,
    protocol_order: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = args.output_dir.resolve()
    result_path = _result_path(output, outer, variant)
    if result_path.exists():
        with np.load(result_path, allow_pickle=False) as archive:
            if str(archive["protocol"].item()) != PROTOCOL_VERSION:
                raise RuntimeError(f"stale P115 result: {result_path}")
            if str(archive["run_signature"].item()) != run_signature(args, variant):
                raise RuntimeError(f"P115 run signature differs: {result_path}")
            return {key: np.asarray(archive[key]) for key in archive.files}

    held = protocol.fold_names == outer
    source = ~held
    held_users = set(OUTER_USERS[outer])
    values = features.values[variant]
    source_oof = np.full((len(protocol.ids), CLASSES), np.nan, dtype=np.float32)
    groups = inner_user_groups(protocol.users[source])
    inner_audit: list[dict[str, Any]] = []
    for inner, group in enumerate(groups):
        eval_protocol = source & np.isin(protocol.users, list(group))
        train_full = ~np.isin(features.pixel_users, [*held_users, *group])
        if set(features.pixel_users[train_full].tolist()) & (held_users | set(group)):
            raise RuntimeError("inner source mask leaked a held subject")
        scores, history, _ = fit_and_predict(
            variant,
            values,
            features.pixel_labels,
            np.flatnonzero(train_full),
            protocol_order[eval_protocol],
            args,
            seed=SEED + list(OUTER_USERS).index(outer) * 1000 + VARIANTS.index(variant) * 100 + inner,
        )
        source_oof[eval_protocol] = scores
        inner_audit.append(
            {
                "inner_fold": inner,
                "held_users": list(group),
                "train_rows": int(train_full.sum()),
                "eval_rows": int(eval_protocol.sum()),
                "last_epoch": history[-1],
            }
        )
    if not np.isfinite(source_oof[source]).all() or np.isfinite(source_oof[held]).any():
        raise RuntimeError("source crossfit score coverage failed")

    safe_confidence = protocol.probability[np.arange(len(protocol.ids)), protocol.safe]
    high_threshold = fixed_quantile(safe_confidence[source], HIGH_CONFIDENCE_QUANTILE)
    source_low = source & (safe_confidence < high_threshold)
    true_scores = source_oof[np.arange(len(protocol.ids)), protocol.labels]
    candidate_threshold = conformal_lower_threshold(true_scores[source_low])
    calibration_kill = float(np.mean(true_scores[source_low] < candidate_threshold))

    final_train = ~np.isin(features.pixel_users, list(held_users))
    if set(features.pixel_users[final_train].tolist()) & held_users:
        raise RuntimeError("outer source mask leaked a held subject")
    checkpoint = output / f"{outer}__{variant}_final.pt"
    held_scores, history, checkpoint_bytes = fit_and_predict(
        variant,
        values,
        features.pixel_labels,
        np.flatnonzero(final_train),
        protocol_order[held],
        args,
        seed=SEED + list(OUTER_USERS).index(outer) * 1000 + VARIANTS.index(variant) * 100 + 99,
        checkpoint=checkpoint,
    )
    outer_scores = np.full((len(protocol.ids), CLASSES), np.nan, dtype=np.float32)
    outer_scores[held] = held_scores
    payload: dict[str, Any] = {
        "protocol": np.asarray(PROTOCOL_VERSION),
        "run_signature": np.asarray(run_signature(args, variant)),
        "source_oof_scores": source_oof.astype(np.float16),
        "outer_scores": outer_scores.astype(np.float16),
        "high_threshold": np.asarray(high_threshold),
        "candidate_threshold": np.asarray(candidate_threshold),
        "source_low_rows": np.asarray(source_low.sum()),
        "source_calibration_true_kill_rate": np.asarray(calibration_kill),
        "checkpoint_bytes": np.asarray(checkpoint_bytes),
        "inner_audit_json": np.asarray(json.dumps(inner_audit, ensure_ascii=False)),
        "final_history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
    }
    np.savez_compressed(result_path, **payload)
    return payload


def _as_scalar(value: Any) -> float:
    return float(np.asarray(value).item())


def evaluate_fold_variant(
    outer: str,
    variant: str,
    result: dict[str, Any],
    protocol: Protocol,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    held = protocol.fold_names == outer
    scores = np.asarray(result["outer_scores"], dtype=np.float32)
    high_threshold = _as_scalar(result["high_threshold"])
    candidate_threshold = _as_scalar(result["candidate_threshold"])
    safe_confidence = protocol.probability[np.arange(len(protocol.ids)), protocol.safe]
    candidate_sets = {k: anchored_topk(protocol.probability, protocol.safe, k) for k in (5, 10)}
    sample_rows: list[dict[str, Any]] = []
    threshold_rows = [
        {
            "outer_fold": outer,
            "variant": variant,
            "high_confidence_quantile": HIGH_CONFIDENCE_QUANTILE,
            "p89_high_confidence_threshold": high_threshold,
            "true_candidate_alpha": TRUE_CANDIDATE_ALPHA,
            "candidate_elimination_threshold": candidate_threshold,
            "source_low_rows": int(_as_scalar(result["source_low_rows"])),
            "source_calibration_true_kill_rate": _as_scalar(
                result["source_calibration_true_kill_rate"]
            ),
            "held_labels_used_for_threshold": 0,
        }
    ]
    for row in np.flatnonzero(held):
        high = bool(safe_confidence[row] >= high_threshold)
        for k, candidates in candidate_sets.items():
            candidate = candidates[row]
            compatible = scores[row, candidate]
            if high:
                survivors = np.asarray([protocol.safe[row]], dtype=np.int64)
                eliminated = np.asarray([], dtype=np.int64)
                raw_killed = False
            else:
                survivors, eliminated = recursive_eliminate(
                    candidate,
                    compatible,
                    candidate_threshold,
                    int(protocol.safe[row]),
                )
                raw_killed = bool(
                    int(protocol.labels[row]) in set(candidate.tolist())
                    and compatible[np.flatnonzero(candidate == protocol.labels[row])[0]] < candidate_threshold
                )
            true_present = int(protocol.labels[row]) in set(candidate.tolist())
            true_killed = bool((not high) and true_present and int(protocol.labels[row]) not in set(survivors.tolist()))
            wrong = candidate != protocol.labels[row]
            wrong_eliminated = int(np.sum(np.isin(candidate[wrong], eliminated))) if not high else 0
            sample_rows.append(
                {
                    "sample_id": protocol.ids[row],
                    "subject": protocol.users[row],
                    "outer_fold": outer,
                    "variant": variant,
                    "top_k": k,
                    "route": "HIGH_CONF_P89_PASS" if high else "LOW_CONF_VISUAL_ELIMINATION",
                    "true_label": int(protocol.labels[row]),
                    "p89_prediction": int(protocol.safe[row]),
                    "p89_correct": int(protocol.safe[row] == protocol.labels[row]),
                    "p89_confidence": float(safe_confidence[row]),
                    "candidates": "|".join(map(str, candidate.tolist())),
                    "compatibility": "|".join(f"{float(value):.7f}" for value in compatible.tolist()),
                    "survivors": "|".join(map(str, survivors.tolist())),
                    "eliminated": "|".join(map(str, eliminated.tolist())),
                    "candidate_count_after": len(survivors),
                    "true_candidate_present": int(true_present),
                    "true_candidate_killed": int(true_killed),
                    "raw_true_candidate_below_threshold": int((not high) and raw_killed),
                    "wrong_candidates": int(wrong.sum()) if not high else 0,
                    "wrong_candidates_eliminated": wrong_eliminated,
                    "p89_anchor_below_threshold": int(
                        (not high)
                        and compatible[np.flatnonzero(candidate == protocol.safe[row])[0]] < candidate_threshold
                    ),
                }
            )
    return sample_rows, threshold_rows


def metric_from_samples(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    low = [row for row in rows if row["route"] == "LOW_CONF_VISUAL_ELIMINATION"]
    high = [row for row in rows if row["route"] == "HIGH_CONF_P89_PASS"]
    eligible = sum(int(row["true_candidate_present"]) for row in low)
    true_killed = sum(int(row["true_candidate_killed"]) for row in low)
    raw_killed = sum(int(row["raw_true_candidate_below_threshold"]) for row in low)
    wrong_total = sum(int(row["wrong_candidates"]) for row in low)
    wrong_removed = sum(int(row["wrong_candidates_eliminated"]) for row in low)
    after_sum = sum(int(row["candidate_count_after"]) for row in low)
    return {
        "rows": len(rows),
        "high_confidence_pass_rows": len(high),
        "high_confidence_p89_precision": (
            sum(int(row["p89_correct"]) for row in high) / len(high) if high else None
        ),
        "low_confidence_rows": len(low),
        "true_candidate_eligible_rows": eligible,
        "true_candidate_coverage": eligible / len(low) if low else None,
        "true_candidate_killed": true_killed,
        "true_candidate_mis_kill_rate": true_killed / eligible if eligible else None,
        "raw_true_candidate_below_threshold": raw_killed,
        "raw_true_candidate_below_threshold_rate": raw_killed / eligible if eligible else None,
        "wrong_candidates": wrong_total,
        "wrong_candidates_eliminated": wrong_removed,
        "wrong_candidate_elimination_rate": wrong_removed / wrong_total if wrong_total else None,
        "candidate_count_after_sum": after_sum,
        "average_candidates_after": after_sum / len(low) if low else None,
        "compressed_to_at_most_2": (
            sum(int(row["candidate_count_after"] <= 2) for row in low) / len(low) if low else None
        ),
        "compressed_to_at_most_3": (
            sum(int(row["candidate_count_after"] <= 3) for row in low) / len(low) if low else None
        ),
        "p89_anchor_below_threshold": sum(int(row["p89_anchor_below_threshold"]) for row in low),
        "final_decision_formed": False,
        "rescue": None,
        "harm": None,
        "net": None,
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int, repeats: int = 20000) -> list[float]:
    source = np.asarray(values, dtype=np.float64)
    if not len(source):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 1000):
        stop = min(start + 1000, repeats)
        sample = rng.integers(0, len(source), size=(stop - start, len(source)))
        means[start:stop] = source[sample].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def subject_metric_map(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for subject in sorted({str(row["subject"]) for row in rows}):
        output[subject] = metric_from_samples([row for row in rows if row["subject"] == subject])
    return output


def compare_to_controls(
    samples: Sequence[dict[str, Any]], top_k: int = 5
) -> list[dict[str, Any]]:
    selected = [row for row in samples if int(row["top_k"]) == top_k]
    by_variant = {
        variant: subject_metric_map([row for row in selected if row["variant"] == variant])
        for variant in VARIANTS
    }
    comparisons: list[dict[str, Any]] = []
    full = by_variant["full_token"]
    for control in VARIANTS[1:]:
        shared = sorted(set(full) & set(by_variant[control]))
        size_gain = np.asarray(
            [
                by_variant[control][subject]["average_candidates_after"]
                - full[subject]["average_candidates_after"]
                for subject in shared
            ],
            dtype=np.float64,
        )
        wrong_gain = np.asarray(
            [
                full[subject]["wrong_candidate_elimination_rate"]
                - by_variant[control][subject]["wrong_candidate_elimination_rate"]
                for subject in shared
            ],
            dtype=np.float64,
        )
        size_ci = bootstrap_mean_ci(size_gain, SEED + VARIANTS.index(control) * 10)
        wrong_ci = bootstrap_mean_ci(wrong_gain, SEED + VARIANTS.index(control) * 10 + 1)
        comparisons.append(
            {
                "control": control,
                "subjects": len(shared),
                "mean_candidate_count_improvement": float(size_gain.mean()),
                "mean_candidate_count_improvement_subject_bootstrap_95ci": size_ci,
                "wrong_elimination_rate_improvement": float(wrong_gain.mean()),
                "wrong_elimination_rate_improvement_subject_bootstrap_95ci": wrong_ci,
                "full_token_significantly_better": bool(size_ci[0] > 0.0 and wrong_ci[0] > 0.0),
            }
        )
    return comparisons


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol()
    features = load_feature_sources(args)
    protocol_order = _align(features.pixel_ids, protocol.ids)
    if not np.array_equal(features.pixel_users[protocol_order], protocol.users):
        raise RuntimeError("P115 P89/full-token user alignment differs")
    if not np.array_equal(features.pixel_labels[protocol_order], protocol.labels):
        raise RuntimeError("P115 P89/full-token label alignment differs")

    requested_folds = tuple(value.strip() for value in args.folds.split(",") if value.strip())
    requested_variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if not set(requested_folds) <= set(OUTER_USERS):
        raise ValueError(f"unknown outer folds: {requested_folds}")
    if not set(requested_variants) <= set(VARIANTS):
        raise ValueError(f"unknown variants: {requested_variants}")

    all_samples: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for outer in requested_folds:
        for variant in requested_variants:
            print(json.dumps({"p115_outer": outer, "variant": variant}), flush=True)
            result = train_outer_variant(
                outer, variant, protocol, features, protocol_order, args
            )
            sample_rows, threshold_rows = evaluate_fold_variant(
                outer, variant, result, protocol
            )
            all_samples.extend(sample_rows)
            thresholds.extend(threshold_rows)

    metric_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for variant in requested_variants:
        for top_k in (5, 10):
            variant_k = [
                row for row in all_samples if row["variant"] == variant and row["top_k"] == top_k
            ]
            if not variant_k:
                continue
            metric_rows.append(
                {
                    "scope": "aggregate",
                    "outer_fold": "ALL",
                    "variant": variant,
                    "top_k": top_k,
                    **metric_from_samples(variant_k),
                }
            )
            for outer in requested_folds:
                rows = [row for row in variant_k if row["outer_fold"] == outer]
                metric_rows.append(
                    {
                        "scope": "fold",
                        "outer_fold": outer,
                        "variant": variant,
                        "top_k": top_k,
                        **metric_from_samples(rows),
                    }
                )
            for subject, metric in subject_metric_map(variant_k).items():
                subject_rows.append(
                    {
                        "subject": subject,
                        "variant": variant,
                        "top_k": top_k,
                        **metric,
                    }
                )

    complete = set(requested_folds) == set(OUTER_USERS) and set(requested_variants) == set(VARIANTS)
    comparisons = compare_to_controls(all_samples) if complete else []
    full_top5 = next(
        (
            row
            for row in metric_rows
            if row["scope"] == "aggregate"
            and row["variant"] == "full_token"
            and row["top_k"] == 5
        ),
        None,
    )
    full_fold5 = [
        row
        for row in metric_rows
        if row["scope"] == "fold" and row["variant"] == "full_token" and row["top_k"] == 5
    ]
    target_met = bool(
        complete
        and full_top5 is not None
        and full_top5["average_candidates_after"] <= 3.0
        and full_top5["true_candidate_mis_kill_rate"] <= 0.02
        and full_top5["wrong_candidate_elimination_rate"] >= 0.40
        and all(row["average_candidates_after"] <= 3.25 for row in full_fold5)
        and all(row["true_candidate_mis_kill_rate"] <= 0.05 for row in full_fold5)
    )
    significant = bool(comparisons and all(row["full_token_significantly_better"] for row in comparisons))
    advance = target_met and significant
    decision = (
        "ADVANCE_TO_SKELETON_IMU_LEAF_CONFIRMATION"
        if advance
        else "STOP_FULL_TOKEN_VISUAL_ROUTE_NEGATIVE_OR_NON_DOMINANT"
    )
    summary = {
        "stage": "P115_full_token_visual_candidate_elimination",
        "status": "complete" if complete else "partial",
        "decision": decision,
        "p89": {"frozen": True, "correct": 2117, "rows": 2470, "accuracy": 2117 / 2470},
        "protocol": {
            "full_token": (
                "all 2x16x3x5x5 ResNet18 layer4 tokens retained until candidate-query cross-attention"
            ),
            "candidate_scorer": "one shared 40-candidate compatible/incompatible scorer",
            "high_confidence_gate": (
                f"fixed q={HIGH_CONFIDENCE_QUANTILE} of source P89 OOF confidence"
            ),
            "candidate_threshold": (
                f"fixed lower conformal alpha={TRUE_CANDIDATE_ALPHA} on source inner-subject OOF true-candidate scores"
            ),
            "candidate_sets": "P89-anchored Top-5; Top-10 audited under the same threshold",
            "cascade": (
                f"remove scores below threshold weakest-first; protect P89 Top-1 and at least {MINIMUM_SURVIVORS} survivors"
            ),
            "final_replacement": False,
            "threshold_sweep": False,
            "pair_specialists": False,
            "held_labels_used_for_training_or_thresholds": False,
            "test_rows_loaded": False,
        },
        "variants": {
            "full_token": "candidate query before any spatial/temporal pooling",
            "same_backbone_pooled": "same ResNet18 tokens pooled to mean/std/early/late-early",
            "pooled_vlit_vhpd_vwpd": "existing P108 V-JEPA2 VLIT/VHPD/VWPD pooled descriptors",
        },
        "thresholds": thresholds,
        "metrics": metric_rows,
        "pooled_control_comparisons_top5": comparisons,
        "advance_gate": {
            "target_met": target_met,
            "full_token_significantly_better_than_both_pooled_controls": significant,
            "required": {
                "aggregate_top5_average_candidates_after_max": 3.0,
                "aggregate_true_candidate_mis_kill_rate_max": 0.02,
                "aggregate_wrong_candidate_elimination_rate_min": 0.40,
                "each_fold_average_candidates_after_max": 3.25,
                "each_fold_true_candidate_mis_kill_rate_max": 0.05,
                "paired_subject_bootstrap_improvement_lower_bound": ">0 for candidate count and wrong elimination",
            },
        },
        "final_decision_metrics": {"rescue": None, "harm": None, "net": None},
    }
    write_csv(output / "thresholds.csv", thresholds)
    write_csv(output / "fold_metrics.csv", metric_rows)
    write_csv(output / "subject_metrics.csv", subject_rows)
    write_csv(output / "sample_candidate_cascades.csv", all_samples)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    args.pixel_cache = args.pixel_cache.resolve()
    args.token_cache = args.token_cache.resolve()
    args.vjepa_root = args.vjepa_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.stage in ("all", "tokens"):
        build_token_cache(args)
    if args.stage in ("all", "audit"):
        run_audit(args)


if __name__ == "__main__":
    main()
