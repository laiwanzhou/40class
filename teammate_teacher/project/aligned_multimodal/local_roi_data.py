from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aligned_data import frame_map


IMAGE_WIDTH = 192
IMAGE_HEIGHT = 144
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_positions(
    length: int,
    count: int,
    augment: bool,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Mirror AlignedMultimodalDataset._sample_positions exactly."""
    if length <= 0:
        raise ValueError("Frame sequence must not be empty")
    if length <= count:
        return (
            torch.linspace(0, length - 1, count).round().long().tolist()
        )
    boundaries = torch.linspace(0, length, count + 1).floor().long()
    positions: list[int] = []
    for index in range(count):
        start = int(boundaries[index])
        end = max(start + 1, int(boundaries[index + 1]))
        if augment:
            positions.append(
                int(torch.randint(start, end, (1,), generator=generator).item())
            )
        else:
            positions.append(min(length - 1, (start + end - 1) // 2))
    return positions


def standardize_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    context: float = 0.15,
    target_ratio: float = 4.0 / 3.0,
) -> tuple[int, int, int, int]:
    """Add context, convert to 4:3, and clamp without distorting coordinates."""
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        raise ValueError(f"Invalid ROI {box} for {width}x{height}")
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    box_width = (x1 - x0 + 1) * (1.0 + 2.0 * context)
    box_height = (y1 - y0 + 1) * (1.0 + 2.0 * context)
    if box_width / box_height < target_ratio:
        box_width = box_height * target_ratio
    else:
        box_height = box_width / target_ratio
    box_width = min(float(width), box_width)
    box_height = min(float(height), box_height)
    left = min(max(0.0, center_x - box_width / 2.0), width - box_width)
    top = min(max(0.0, center_y - box_height / 2.0), height - box_height)
    right = left + box_width
    bottom = top + box_height
    return (
        max(0, int(np.floor(left))),
        max(0, int(np.floor(top))),
        min(width, int(np.ceil(right))),
        min(height, int(np.ceil(bottom))),
    )


@dataclass(frozen=True)
class FoldPureBox:
    sample_id: str
    held_fold: int
    locator_training_folds: tuple[int, int]
    x0: float
    y0: float
    x1: float
    y1: float
    raw_width: int
    raw_height: int


def parse_training_folds(value: str) -> tuple[int, int]:
    stripped = value.strip()
    try:
        decoded = json.loads(stripped)
        if isinstance(decoded, list):
            folds = tuple(sorted(int(item) for item in decoded))
        else:
            raise ValueError
    except (json.JSONDecodeError, ValueError, TypeError):
        folds = tuple(
            sorted(int(item.strip()) for item in stripped.split(",") if item.strip())
        )
    if len(folds) != 2 or len(set(folds)) != 2:
        raise ValueError(f"locator_training_folds must contain two folds: {value}")
    return folds


def load_fold_pure_boxes(
    path: Path,
    held_fold: int,
    required_sample_ids: set[str],
) -> dict[str, FoldPureBox]:
    """Load locator predictions and reject any fold-contaminated box table.

    Required CSV columns:
      sample_id, held_fold, locator_training_folds,
      x0, y0, x1, y1, raw_width, raw_height

    The table must contain predictions made by one locator trained only on the
    other two folds. Human boxes from the held fold are not an accepted input.
    """
    expected_training_folds = tuple(fold for fold in range(3) if fold != held_fold)
    selected: dict[str, FoldPureBox] = {}
    for row in read_csv(path):
        if int(row["held_fold"]) != held_fold:
            continue
        training_folds = parse_training_folds(row["locator_training_folds"])
        if training_folds != expected_training_folds:
            raise ValueError(
                f"{row['sample_id']}: held_fold={held_fold} must use locator "
                f"training folds {expected_training_folds}, got {training_folds}"
            )
        sample_id = row["sample_id"]
        if sample_id in selected:
            raise ValueError(
                f"Duplicate fold-{held_fold} locator prediction: {sample_id}"
            )
        selected[sample_id] = FoldPureBox(
            sample_id=sample_id,
            held_fold=held_fold,
            locator_training_folds=training_folds,
            x0=float(row["x0"]),
            y0=float(row["y0"]),
            x1=float(row["x1"]),
            y1=float(row["y1"]),
            raw_width=int(row["raw_width"]),
            raw_height=int(row["raw_height"]),
        )
    missing = sorted(required_sample_ids - set(selected))
    extra = sorted(set(selected) - required_sample_ids)
    if missing or extra:
        raise ValueError(
            f"Fold-{held_fold} locator coverage mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    return selected


class FoldPureLocalDepthDataset(Dataset):
    """Local Depth data whose ROI predictions obey subject-fold isolation.

    A separate instance is constructed for every held fold. Its locator box
    table must cover both train and validation samples with predictions from the
    same locator trained only on the other two folds. This makes crop generation,
    Local classification, and later routing auditable as one cross-fitted chain.
    """

    def __init__(
        self,
        manifest_path: Path,
        subject_fold_path: Path,
        locator_predictions_path: Path,
        held_fold: int,
        split: str,
        *,
        num_frames: int = 12,
        augment: bool = False,
        context: float = 0.15,
    ) -> None:
        if held_fold not in (0, 1, 2):
            raise ValueError("held_fold must be 0, 1, or 2")
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        manifest = {row["sample_id"]: row for row in read_csv(manifest_path)}
        fold_rows = read_csv(subject_fold_path)
        fold_by_id = {row["sample_id"]: row for row in fold_rows}
        if set(manifest) != set(fold_by_id):
            raise ValueError("Manifest and subject-fold manifest have different samples")
        samples = [
            manifest[sample_id]
            for sample_id, fold_row in fold_by_id.items()
            if fold_row["split"] == split
        ]
        samples.sort(key=lambda row: row["sample_id"])
        boxes = load_fold_pure_boxes(
            locator_predictions_path,
            held_fold,
            set(manifest),
        )
        self.samples = samples
        self.boxes = boxes
        self.held_fold = held_fold
        self.split = split
        self.num_frames = num_frames
        self.augment = augment
        self.context = context

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.samples[index]
        sample_id = row["sample_id"]
        maps = {
            "depth": frame_map(Path(row["depth_dir"]), "depth"),
            "ir": frame_map(Path(row["ir_dir"]), "ir"),
            "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
        }
        common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
        positions = sample_positions(
            len(common_ids),
            self.num_frames,
            self.augment,
        )
        frame_ids = [common_ids[position] for position in positions]
        record = self.boxes[sample_id]
        crop_box = standardize_box(
            (record.x0, record.y0, record.x1, record.y1),
            record.raw_width,
            record.raw_height,
            self.context,
        )
        frames: list[torch.Tensor] = []
        for frame_id in frame_ids:
            with Image.open(maps["depth"][frame_id]) as image:
                image = image.convert("RGB")
                if image.size != (record.raw_width, record.raw_height):
                    raise ValueError(
                        f"{sample_id}: Depth {image.size} != locator coordinate "
                        f"space {(record.raw_width, record.raw_height)}"
                    )
                image = image.crop(crop_box).resize(
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    Image.Resampling.BILINEAR,
                )
                array = np.asarray(image, dtype=np.uint8).copy()
            frames.append(
                torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            )
        tensor = torch.stack(frames)
        if self.augment and bool(torch.rand(1).item() < 0.5):
            tensor = torch.flip(tensor, dims=(3,))
        tensor = (tensor - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[
            None, :, None, None
        ]
        return {
            "depth_local": tensor,
            "label": int(row["class_id"]),
            "sample_id": sample_id,
            "held_fold": self.held_fold,
        }
