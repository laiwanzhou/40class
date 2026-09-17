"""Label-free cached P12 depth adapter for the P434 source-only rebuild."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_PROJECT_DIR = Path(__file__).resolve().parent


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (_PROJECT_DIR / value).resolve()


class P434DepthDataset(Dataset):
    """Read-only Jet-RGB depth clips with caller-supplied labels only.

    The adapter intentionally does not instantiate ``AlignedMultimodalDataset``
    because that constructor parses the task manifest and opens other modality
    caches.  Temporal sampling and augmentation mirror its depth-only path.
    """

    num_frames = 12
    image_height = 144
    image_width = 192

    def __init__(self, cache_dir: str | Path, sample_ids: Sequence[str],
                 indices: Sequence[int], labels: Sequence[int] | None = None,
                 augment: bool = False):
        self.cache_dir = resolve_project_path(cache_dir)
        self.sample_ids = np.asarray(sample_ids)
        if (self.sample_ids.ndim != 1 or not len(self.sample_ids)
                or self.sample_ids.dtype.kind not in "OUS"
                or any(not isinstance(x, (str, np.str_)) or not str(x).strip() for x in self.sample_ids)
                or len(set(self.sample_ids.astype(str).tolist())) != len(self.sample_ids)):
            raise ValueError("sample_ids must be unique nonblank strings")
        raw_indices = np.asarray(indices)
        if (raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu"
                or raw_indices.dtype.kind == "b" or len(set(raw_indices.tolist())) != len(raw_indices)
                or np.any(raw_indices < 0) or np.any(raw_indices >= len(self.sample_ids))):
            raise ValueError("indices must be unique in-range integer values")
        self.indices = tuple(int(x) for x in raw_indices.tolist())
        self.labels = None
        if labels is not None:
            raw_labels = np.asarray(labels)
            if (raw_labels.ndim != 1 or raw_labels.shape != raw_indices.shape
                    or raw_labels.dtype.kind not in "iu" or raw_labels.dtype.kind == "b"
                    or np.any((raw_labels < 0) | (raw_labels >= 40))):
                raise ValueError("labels must be supplied integer class IDs in [0,39]")
            self.labels = tuple(int(x) for x in raw_labels.tolist())
        self.augment = bool(augment)
        metadata_path = self.cache_dir / "metadata.json"
        depth_path = self.cache_dir / "depth_uint8.npy"
        if not metadata_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(metadata_path if not metadata_path.is_file() else depth_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid depth cache metadata") from exc
        metadata_ids = metadata.get("sample_ids")
        metadata_offsets = metadata.get("offsets")
        if (not isinstance(metadata_ids, list) or not isinstance(metadata_offsets, list)
                or len(metadata_ids) != len(metadata_offsets) or not metadata_ids):
            raise ValueError("metadata must contain aligned sample_ids and offsets")
        if any(not isinstance(x, str) or not x.strip() for x in metadata_ids):
            raise ValueError("metadata sample_ids must be nonblank strings")
        metadata_ids = tuple(metadata_ids)
        if len(set(metadata_ids)) != len(metadata_ids):
            raise ValueError("metadata sample_ids must be unique")
        depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(depth, np.memmap) or depth.dtype != np.dtype(np.uint8):
            raise ValueError("depth cache must be a uint8 read-only memmap")
        if depth.ndim != 4 or tuple(depth.shape[1:]) != (self.image_height, self.image_width, 3):
            raise ValueError(f"depth cache schema differs: {depth.shape}")
        total_frames = int(depth.shape[0])
        offsets: list[tuple[int, int]] = []
        for item in metadata_offsets:
            if (not isinstance(item, (list, tuple)) or len(item) != 2
                    or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in item)):
                raise ValueError("metadata offsets must be integer [offset,length] pairs")
            offset, length = int(item[0]), int(item[1])
            if offset < 0 or length <= 0 or offset + length > total_frames:
                raise ValueError("metadata offset out of bounds")
            offsets.append((offset, length))
        if len(metadata_ids) != len(self.sample_ids):
            raise ValueError("depth metadata universe size differs from supplied IDs")
        if set(metadata_ids) != set(self.sample_ids.astype(str).tolist()):
            raise ValueError("depth metadata IDs differ from supplied universe")
        if metadata.get("total_frames", total_frames) != total_frames:
            raise ValueError("metadata total_frames differs from depth cache")
        if (metadata.get("num_samples") not in (None, len(metadata_ids))
                or metadata.get("image_height", self.image_height) != self.image_height
                or metadata.get("image_width", self.image_width) != self.image_width):
            raise ValueError("depth metadata dimensions/sample count differ")
        cursor = 0
        for offset, length in offsets:
            if offset != cursor:
                raise ValueError("depth metadata offsets are not contiguous")
            cursor += length
        if cursor != total_frames:
            raise ValueError("depth metadata offsets do not cover the depth cache")
        metadata_rows = {sample_id: row for row, sample_id in enumerate(metadata_ids)}
        selected_ids = tuple(self.sample_ids[index].__str__() for index in self.indices)
        if any(sample_id not in metadata_rows for sample_id in selected_ids):
            raise KeyError("requested sample ID missing from depth cache metadata")
        self._depth = depth
        self._offsets = tuple(offsets)
        self._selected_ids = selected_ids
        self._cache_rows = tuple(metadata_rows[sample_id] for sample_id in selected_ids)

    def __len__(self) -> int:
        return len(self.indices)

    def _sample_positions(self, length: int) -> list[int]:
        if length <= self.num_frames:
            return torch.linspace(0, length - 1, self.num_frames).round().long().tolist()
        boundaries = torch.linspace(0, length, self.num_frames + 1).floor().long()
        positions: list[int] = []
        for i in range(self.num_frames):
            start = int(boundaries[i]); end = max(start + 1, int(boundaries[i + 1]))
            if self.augment:
                positions.append(int(torch.randint(start, end, (1,)).item()))
            else:
                positions.append((start + end - 1) // 2)
        return [min(length - 1, position) for position in positions]

    def __getitem__(self, index: int) -> dict[str, object]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cache_row = self._cache_rows[index]
        offset, length = self._offsets[cache_row]
        positions = np.asarray(self._sample_positions(length), dtype=np.int64) + offset
        flip = bool(torch.rand(1).item() < 0.5) if self.augment else False
        depth = np.asarray(self._depth[positions])
        if flip:
            depth = np.flip(depth, axis=2)
        tensor = torch.from_numpy(np.ascontiguousarray(depth)).permute(0, 3, 1, 2).float()
        # P12's depth_imagenet config uses ImageNet-initialized weights but the
        # dataset default remains the legacy [-1,1] normalization.
        tensor = tensor.div(127.5).sub(1.0)
        result: dict[str, object] = {"depth": tensor, "sample_id": self._selected_ids[index],
                                     "label": -1 if self.labels is None else self.labels[index]}
        return result


__all__ = ["P434DepthDataset"]
