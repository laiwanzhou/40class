from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import train_shared_full_local_oof as baseline


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PEAK_CACHE = PROJECT_DIR / "runs" / "p18_peak_local_depth_cache"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p18_peak_sampling_shared_oof"


class PeakSamplingFullLocalDataset(baseline.Dataset):
    def __init__(
        self,
        full_cache: Path,
        local_cache: Path,
        fold_csv: Path,
        held_fold: int,
        split: str,
        augment: bool,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        metadata = json.loads(
            (full_cache / "metadata.json").read_text(encoding="utf-8")
        )
        self.full_location = {
            str(sample_id): (int(offset), int(length))
            for sample_id, (offset, length) in zip(
                metadata["sample_ids"],
                metadata["offsets"],
                strict=True,
            )
        }
        self.full_path = full_cache / "depth_uint8.npy"
        peak_ids = np.load(
            local_cache / "sample_ids.npy",
            allow_pickle=False,
        ).astype(str)
        self.local_location = {
            sample_id: index for index, sample_id in enumerate(peak_ids)
        }
        self.local_path = local_cache / "local_peak_candidates_uint8.npy"
        self.uniform_positions = np.load(
            local_cache / "uniform_positions.npy",
            allow_pickle=False,
        )
        self.peak_positions = np.load(
            local_cache / "peak_positions.npy",
            allow_pickle=False,
        )
        self.peak_candidates = np.load(
            local_cache / "peak_candidates.npy",
            allow_pickle=False,
        )
        self._full: np.ndarray | None = None
        self._local: np.ndarray | None = None
        self.samples = [
            row
            for row in baseline.read_csv(fold_csv)
            if row["split"] == split
        ]
        self.samples.sort(key=lambda row: row["sample_id"])
        for row in self.samples:
            sample_id = row["sample_id"]
            if sample_id not in self.full_location:
                raise ValueError(f"Full cache missing {sample_id}")
            if sample_id not in self.local_location:
                raise ValueError(f"Peak Local cache missing {sample_id}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._full is None:
            self._full = np.load(self.full_path, mmap_mode="r")
        if self._local is None:
            self._local = np.load(self.local_path, mmap_mode="r")
        return self._full, self._local

    def selected_positions(self, local_index: int) -> list[int]:
        uniform = self.uniform_positions[local_index].astype(int).tolist()
        used = set(uniform)
        selected = list(uniform)
        for peak_index in range(self.peak_candidates.shape[1]):
            candidates = (
                self.peak_candidates[local_index, peak_index]
                .astype(int)
                .tolist()
            )
            if self.augment:
                order = torch.randperm(len(candidates)).tolist()
            else:
                order = [1, 0, 2]
            chosen = None
            for candidate_index in order:
                candidate = candidates[candidate_index]
                if candidate not in used:
                    chosen = candidate
                    break
            if chosen is None:
                chosen = int(self.peak_positions[local_index, peak_index])
            selected.append(chosen)
            used.add(chosen)
        return sorted(selected)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.samples[index]
        sample_id = row["sample_id"]
        local_index = self.local_location[sample_id]
        positions = self.selected_positions(local_index)
        if len(positions) != 12:
            raise RuntimeError(f"{sample_id}: expected 12 positions")
        full_cache, local_cache = self.arrays()
        offset, length = self.full_location[sample_id]
        if min(positions) < 0 or max(positions) >= length:
            raise RuntimeError(f"{sample_id}: peak position outside length={length}")
        full = np.asarray(
            full_cache[offset + np.asarray(positions, dtype=np.int64)]
        ).copy()
        cache_positions = np.concatenate(
            [
                self.uniform_positions[local_index].reshape(-1),
                self.peak_candidates[local_index].reshape(-1),
            ]
        ).astype(int)
        position_to_cache = {}
        for cache_index, position in enumerate(cache_positions.tolist()):
            position_to_cache.setdefault(position, cache_index)
        local_indices = [position_to_cache[position] for position in positions]
        local = np.asarray(local_cache[local_index, local_indices]).copy()
        full_tensor = (
            torch.from_numpy(full).permute(0, 3, 1, 2).float().div_(255.0)
        )
        local_tensor = (
            torch.from_numpy(local).permute(0, 3, 1, 2).float().div_(255.0)
        )
        if self.augment and bool(torch.rand(1).item() < 0.5):
            full_tensor = torch.flip(full_tensor, dims=(3,))
            local_tensor = torch.flip(local_tensor, dims=(3,))
        full_tensor = (
            full_tensor - baseline.IMAGENET_MEAN[None, :, None, None]
        ) / baseline.IMAGENET_STD[None, :, None, None]
        local_tensor = (
            local_tensor - baseline.IMAGENET_MEAN[None, :, None, None]
        ) / baseline.IMAGENET_STD[None, :, None, None]
        return {
            "depth_full": full_tensor,
            "depth_local": local_tensor,
            "label": int(row["class_id"]),
            "sample_id": sample_id,
        }


def main() -> None:
    baseline.DEFAULT_LOCAL_CACHE = DEFAULT_PEAK_CACHE
    baseline.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    baseline.FullLocalCacheDataset = PeakSamplingFullLocalDataset
    baseline.main()
    summary_path = DEFAULT_OUTPUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["temporal_sampling"] = {
            "experiment": "8_uniform_plus_4_local_depth_motion_peaks",
            "training": "peak candidates jittered by -1/0/+1",
            "validation": "fixed peak centers",
            "full_local_frame_alignment": True,
            "cache": str(DEFAULT_PEAK_CACHE.resolve()),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
