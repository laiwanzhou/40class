from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import train_shared_full_local_oof as baseline
from aligned_model import VisualEncoder
from local_roi_data import sample_positions


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EXTRA_CACHE = PROJECT_DIR / "runs" / "p19_local_depth_difference_cache"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p19_local_depth_difference_shared_oof"
DIFFERENCE_SCALE = 16.0
BaselineSharedFullLocalModel = baseline.SharedFullLocalModel
_ALIGNED_RAM_CACHE: tuple[
    np.ndarray,
    np.ndarray,
    dict[str, int],
] | None = None


def preload_aligned_ram_cache(
    full_cache: Path,
    local_cache: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    global _ALIGNED_RAM_CACHE
    if _ALIGNED_RAM_CACHE is not None:
        return _ALIGNED_RAM_CACHE
    fallback_paths = [
        local_cache / f"fallback_fold_{fold}_uint8.npy" for fold in range(3)
    ]
    if not all(os.path.samefile(fallback_paths[0], path) for path in fallback_paths[1:]):
        raise ValueError(
            "RAM sharing across folds requires the oracle fallback caches to be hard-linked"
        )
    started = time.perf_counter()
    extra_ids = np.load(
        DEFAULT_EXTRA_CACHE / "sample_ids.npy",
        allow_pickle=False,
    ).astype(str)
    aligned_location = {
        sample_id: index for index, sample_id in enumerate(extra_ids)
    }
    metadata = json.loads(
        (full_cache / "metadata.json").read_text(encoding="utf-8")
    )
    full_location = {
        str(sample_id): (int(offset), int(length))
        for sample_id, (offset, length) in zip(
            metadata["sample_ids"],
            metadata["offsets"],
            strict=True,
        )
    }
    full_source = np.load(full_cache / "depth_uint8.npy", mmap_mode="r")
    full_aligned = np.empty(
        (len(extra_ids), 12, 144, 192, 3),
        dtype=np.uint8,
    )
    for index, sample_id in enumerate(extra_ids):
        offset, length = full_location[sample_id]
        positions = np.asarray(
            sample_positions(length, 12, augment=False),
            dtype=np.int64,
        )
        full_aligned[index] = full_source[offset + positions]
        if (index + 1) % 500 == 0:
            print(
                f"RAM preload Full {index + 1}/{len(extra_ids)}",
                flush=True,
            )
    del full_source

    nonfallback_ids = np.load(
        local_cache / "nonfallback_sample_ids.npy",
        allow_pickle=False,
    ).astype(str)
    fallback_ids = np.load(
        local_cache / "fallback_sample_ids.npy",
        allow_pickle=False,
    ).astype(str)
    nonfallback_source = np.load(
        local_cache / "nonfallback_uint8.npy",
        mmap_mode="r",
    )
    fallback_source = np.load(fallback_paths[0], mmap_mode="r")
    local_aligned = np.empty(
        (len(extra_ids), 12, 144, 192, 5),
        dtype=np.uint8,
    )
    for index, sample_id in enumerate(nonfallback_ids):
        local_aligned[aligned_location[sample_id], ..., :3] = (
            nonfallback_source[index]
        )
    for index, sample_id in enumerate(fallback_ids):
        local_aligned[aligned_location[sample_id], ..., :3] = (
            fallback_source[index]
        )
    del nonfallback_source, fallback_source
    extra_source = np.load(
        DEFAULT_EXTRA_CACHE / "extra_channels_uint8.npy",
        allow_pickle=False,
    )
    local_aligned[..., 3:] = extra_source
    del extra_source
    if (
        len(full_aligned) != 2914
        or full_aligned.shape[:2] != (2914, 12)
        or local_aligned.shape != (2914, 12, 144, 192, 5)
    ):
        raise ValueError("Aligned RAM cache shape mismatch")
    _ALIGNED_RAM_CACHE = (
        full_aligned,
        local_aligned,
        aligned_location,
    )
    print(
        "RAM preload complete "
        f"bytes={full_aligned.nbytes + local_aligned.nbytes} "
        f"seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return _ALIGNED_RAM_CACHE


class DifferenceFullLocalDataset(baseline.Dataset):
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
        nonfallback_ids = np.load(
            local_cache / "nonfallback_sample_ids.npy",
            allow_pickle=False,
        ).astype(str)
        fallback_ids = np.load(
            local_cache / "fallback_sample_ids.npy",
            allow_pickle=False,
        ).astype(str)
        self.local_location = {
            sample_id: ("nonfallback", index)
            for index, sample_id in enumerate(nonfallback_ids)
        }
        self.local_location.update(
            {
                sample_id: ("fallback", index)
                for index, sample_id in enumerate(fallback_ids)
            }
        )
        self.local_nonfallback_path = local_cache / "nonfallback_uint8.npy"
        self.local_fallback_path = (
            local_cache / f"fallback_fold_{held_fold}_uint8.npy"
        )
        extra_ids = np.load(
            DEFAULT_EXTRA_CACHE / "sample_ids.npy",
            allow_pickle=False,
        ).astype(str)
        self.extra_location = {
            sample_id: index for index, sample_id in enumerate(extra_ids)
        }
        self.extra_path = DEFAULT_EXTRA_CACHE / "extra_channels_uint8.npy"
        (
            self._full_aligned,
            self._local_aligned,
            self._aligned_location,
        ) = preload_aligned_ram_cache(full_cache, local_cache)
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
                raise ValueError(f"Local cache missing {sample_id}")
            if sample_id not in self.extra_location:
                raise ValueError(f"Difference cache missing {sample_id}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.samples[index]
        sample_id = row["sample_id"]
        aligned_index = self._aligned_location[sample_id]
        full_rgb = self._full_aligned[aligned_index]
        local_five = self._local_aligned[aligned_index]
        full_tensor = torch.from_numpy(full_rgb).permute(0, 3, 1, 2)
        local_tensor = torch.from_numpy(local_five).permute(0, 3, 1, 2)
        if self.augment and bool(torch.rand(1).item() < 0.5):
            full_tensor = torch.flip(full_tensor, dims=(3,))
            local_tensor = torch.flip(local_tensor, dims=(3,))
        return {
            "depth_full": full_tensor,
            "depth_local": local_tensor,
            "label": int(row["class_id"]),
            "sample_id": sample_id,
        }


class DifferenceSharedFullLocalModel(BaselineSharedFullLocalModel):
    def __init__(self, dropout: float = 0.3, num_classes: int = 40) -> None:
        super().__init__(dropout=dropout, num_classes=num_classes)
        self.visual = VisualEncoder(
            ("depth",),
            dropout=dropout,
            imagenet_pretrained=True,
            depth_input_channels=5,
        )
        self.register_buffer(
            "_imagenet_mean",
            baseline.IMAGENET_MEAN[None, None, :, None, None].clone(),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            baseline.IMAGENET_STD[None, None, :, None, None].clone(),
            persistent=False,
        )

    def forward(
        self,
        full: torch.Tensor,
        local: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        full_rgb = full.float().div_(255.0)
        full_rgb = (full_rgb - self._imagenet_mean) / self._imagenet_std
        full_extra = torch.zeros(
            (*full_rgb.shape[:2], 2, *full_rgb.shape[-2:]),
            dtype=full_rgb.dtype,
            device=full_rgb.device,
        )
        local_float = local.float()
        local_rgb = local_float[:, :, :3].div_(255.0)
        local_rgb = (local_rgb - self._imagenet_mean) / self._imagenet_std
        local_difference = (
            local_float[:, :, 3:4].div_(DIFFERENCE_SCALE).clamp_(0.0, 1.0)
        )
        local_valid = local_float[:, :, 4:5].div_(255.0)
        return super().forward(
            torch.cat([full_rgb, full_extra], dim=2),
            torch.cat([local_rgb, local_difference, local_valid], dim=2),
        )


def load_difference_initialization(
    model: DifferenceSharedFullLocalModel,
    checkpoint_path: Path,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["model_state_dict"]
    visual_state = {
        key.removeprefix("visual."): value
        for key, value in state.items()
        if key.startswith("visual.")
    }
    source_stem = visual_state["depth_stem.weight"]
    target_stem = model.visual.depth_stem.weight.detach().clone()
    if source_stem.shape[1] != 3 or target_stem.shape[1] != 5:
        raise ValueError(
            f"Unexpected stem shapes source={source_stem.shape} "
            f"target={target_stem.shape}"
        )
    target_stem[:, :3].copy_(source_stem)
    target_stem[:, 3:].zero_()
    visual_state["depth_stem.weight"] = target_stem
    model.visual.load_state_dict(visual_state, strict=True)


def main() -> None:
    baseline.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    baseline.FullLocalCacheDataset = DifferenceFullLocalDataset
    baseline.SharedFullLocalModel = DifferenceSharedFullLocalModel
    baseline.load_full_initialization = load_difference_initialization
    baseline.main()
    summary_path = DEFAULT_OUTPUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["input_channels"] = {
            "shared_stem_channels": 5,
            "full": "ImageNet-normalized JET RGB + two zero channels",
            "local": (
                "ImageNet-normalized JET RGB + decoded depth absolute "
                "difference clipped after /16 + valid-intersection mask/255"
            ),
            "initialization": (
                "copy checkpoint stem channels 0:3 exactly; initialize "
                "channels 3:5 to zero"
            ),
            "sampling": "original fixed 12 segment midpoints",
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
