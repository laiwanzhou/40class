from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from aligned_data import (
    IMAGENET_GRAY_MEAN,
    IMAGENET_GRAY_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LEFT_RIGHT_PAIRS,
    load_skeleton_sequence,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_UNION_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_FOLD_SUMMARY = PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json"
DEFAULT_ALIGNED_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
DEFAULT_IMU_CACHE = PROJECT_DIR / "cache" / "imu_32"
DEFAULT_LOCATOR_DIR = PROJECT_DIR / "runs" / "p12_fold_pure_locator_predictions"

FRAME_COUNTER_RE = re.compile(r"(\d{8})$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_sample_id(row: dict[str, str]) -> str:
    return f"train__c{int(row['class_id']):02d}__{row['user_id']}__{row['trial_id']}"


def canonical_frame_key(path: Path, modality: str) -> str:
    """Return the legacy full frame key after removing modality affixes."""
    stem = path.stem
    if modality == "depth":
        if not stem.startswith("Depth_") or not stem.endswith("_Color"):
            raise ValueError(f"unexpected Depth filename: {path.name}")
        stem = stem[len("Depth_") : -len("_Color")]
    elif modality == "ir":
        if not stem.startswith("IR_"):
            raise ValueError(f"unexpected IR filename: {path.name}")
        stem = stem[len("IR_") :]
    elif modality == "skeleton":
        if not stem.startswith("Color_"):
            raise ValueError(f"unexpected Skeleton filename: {path.name}")
        stem = stem[len("Color_") :]
    else:
        raise ValueError(modality)
    return stem


def terminal_frame_counter(key: str) -> str | None:
    match = FRAME_COUNTER_RE.search(key)
    return match.group(1) if match else None


def frame_map(path: str | Path, modality: str) -> dict[str, Path]:
    trial_dir = Path(path) if path else Path("__missing__")
    if not trial_dir.is_dir():
        return {}
    if modality == "skeleton":
        prediction_dir = trial_dir / "predictions"
        files: Iterable[Path] = prediction_dir.glob("*.json") if prediction_dir.is_dir() else ()
    else:
        files = (
            child
            for child in trial_dir.iterdir()
            if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    output: dict[str, Path] = {}
    for file_path in files:
        try:
            key = canonical_frame_key(file_path, modality)
        except ValueError:
            continue
        if key in output:
            raise ValueError(f"duplicate {modality} full frame key {key}: {trial_dir}")
        output[key] = file_path
    return output


def align_frame_maps(
    maps: dict[str, dict[str, Path]]
) -> tuple[dict[str, dict[str, Path]], list[str], str]:
    """Align present modalities with a validated two-level key.

    Full timestamp keys remain authoritative. Only the 17 known filename-schema
    exceptions have no full-key intersection. For those trials we fall back to
    the terminal frame counter after proving it is unique in every present
    modality. A normal trial with repeated counters therefore never silently
    collapses two frames.
    """

    present = {name: mapping for name, mapping in maps.items() if mapping}
    if not present:
        return maps, [], "none"
    full_common = sorted(set.intersection(*(set(mapping) for mapping in present.values())))
    if full_common:
        return maps, full_common, "full"
    counter_maps: dict[str, dict[str, Path]] = {}
    for modality, mapping in present.items():
        converted: dict[str, Path] = {}
        for full_key, path in mapping.items():
            counter = terminal_frame_counter(full_key)
            if counter is None:
                return maps, [], "unresolved"
            if counter in converted:
                return maps, [], "ambiguous_counter"
            converted[counter] = path
        counter_maps[modality] = converted
    counter_common = sorted(
        set.intersection(*(set(mapping) for mapping in counter_maps.values()))
    )
    if not counter_common:
        return maps, [], "unresolved"
    aligned = dict(maps)
    aligned.update(counter_maps)
    return aligned, counter_common, "counter_fallback"


def subject_to_fold(summary_path: Path = DEFAULT_FOLD_SUMMARY) -> dict[str, int]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mapping: dict[str, int] = {}
    for fold in summary["folds"]:
        fold_id = int(fold["fold"])
        for subject in fold["val_users"]:
            if subject in mapping:
                raise ValueError(f"subject appears in multiple held folds: {subject}")
            mapping[subject] = fold_id
    return mapping


def build_p27_manifest_rows(
    union_manifest: Path = DEFAULT_UNION_MANIFEST,
    imu_cache_dir: Path = DEFAULT_IMU_CACHE,
    fold_summary: Path = DEFAULT_FOLD_SUMMARY,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    union_rows = read_csv(union_manifest)
    imu_index = {
        row["sample_id"]: row
        for row in read_csv(imu_cache_dir / "index.csv")
        if row["split"] == "train"
    }
    fold_by_subject = subject_to_fold(fold_summary)
    rows: list[dict[str, object]] = []
    excluded = 0
    for source in union_rows:
        sample_id = canonical_sample_id(source)
        raw_maps = {
            "depth": frame_map(source["depth_color_path"], "depth"),
            "ir": frame_map(source["ir_path"], "ir"),
            "skeleton": frame_map(source["skeleton_path"], "skeleton"),
        }
        maps, common_keys, alignment_mode = align_frame_maps(raw_maps)
        visual_present = {name: int(bool(mapping)) for name, mapping in raw_maps.items()}
        imu_row = imu_index.get(sample_id)
        imu_usable = int(imu_row is not None and int(imu_row["usable"]) == 1)
        eligible = bool(common_keys) or bool(imu_usable)
        if not eligible:
            excluded += 1
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "source_sample_id": source["sample_id"],
                "class_id": int(source["class_id"]),
                "class_name": source["class_name"],
                "user_id": source["user_id"],
                "trial_id": source["trial_id"],
                "subject_fold": fold_by_subject[source["user_id"]],
                "depth_dir": source["depth_color_path"],
                "ir_dir": source["ir_path"],
                "skeleton_dir": source["skeleton_path"],
                "imu_dir": source["imu_path"],
                "depth_usable": visual_present["depth"],
                "ir_usable": visual_present["ir"],
                "skeleton_usable": visual_present["skeleton"],
                "imu_usable": imu_usable,
                "aligned_frame_count": len(common_keys),
                "alignment_mode": alignment_mode,
                "depth_frame_count": len(raw_maps["depth"]),
                "ir_frame_count": len(raw_maps["ir"]),
                "skeleton_frame_count": len(raw_maps["skeleton"]),
                "imu_cache_index": int(imu_row["cache_index"]) if imu_row is not None else -1,
                "imu_device_count": int(imu_row["device_count"]) if imu_row is not None else 0,
            }
        )
    rows.sort(key=lambda row: str(row["sample_id"]))
    summary = {
        "source_union": len(union_rows),
        "eligible": len(rows),
        "excluded_no_p27_modality": excluded,
        "depth_usable": sum(int(row["depth_usable"]) for row in rows),
        "ir_usable": sum(int(row["ir_usable"]) for row in rows),
        "skeleton_usable": sum(int(row["skeleton_usable"]) for row in rows),
        "imu_usable": sum(int(row["imu_usable"]) for row in rows),
        "aligned_depth_ir_skeleton": sum(
            int(row["depth_usable"])
            * int(row["ir_usable"])
            * int(row["skeleton_usable"])
            * int(int(row["aligned_frame_count"]) > 0)
            for row in rows
        ),
        "counter_fallback": sum(
            int(row["alignment_mode"] == "counter_fallback") for row in rows
        ),
    }
    return rows, summary


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class P27Sample:
    row: dict[str, str]

    @property
    def sample_id(self) -> str:
        return self.row["sample_id"]

    @property
    def class_id(self) -> int:
        return int(self.row["class_id"])

    @property
    def user_id(self) -> str:
        return self.row["user_id"]


class P27Dataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        outer_fold: int,
        train: bool,
        num_frames: int = 12,
        image_height: int = 144,
        image_width: int = 192,
        aligned_cache_dir: str | Path = DEFAULT_ALIGNED_CACHE,
        imu_cache_dir: str | Path = DEFAULT_IMU_CACHE,
        locator_dir: str | Path = DEFAULT_LOCATOR_DIR,
        roi_padding: float = 0.30,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.outer_fold = int(outer_fold)
        self.train = bool(train)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.aligned_cache_dir = Path(aligned_cache_dir).resolve()
        self.imu_cache_dir = Path(imu_cache_dir).resolve()
        self.roi_padding = float(roi_padding)
        all_rows = read_csv(self.manifest_path)
        selected = [
            P27Sample(row)
            for row in all_rows
            if (int(row["subject_fold"]) != self.outer_fold) == self.train
        ]
        if not selected:
            raise RuntimeError(f"no P27 samples for fold={outer_fold}, train={train}")
        self.samples = selected

        metadata = json.loads((self.aligned_cache_dir / "metadata.json").read_text(encoding="utf-8"))
        self.cache_locations = {
            sample_id: (int(offset), int(length))
            for sample_id, (offset, length) in zip(
                metadata["sample_ids"], metadata["offsets"], strict=True
            )
        }
        self._arrays: dict[str, np.ndarray] = {}
        locator_path = Path(locator_dir).resolve() / f"fold_{self.outer_fold}_locator_predictions.csv"
        self.locators = {
            row["sample_id"]: np.asarray(
                [
                    float(row["x0"]) / float(row["raw_width"]),
                    float(row["y0"]) / float(row["raw_height"]),
                    float(row["x1"]) / float(row["raw_width"]),
                    float(row["y1"]) / float(row["raw_height"]),
                ],
                dtype=np.float32,
            )
            for row in read_csv(locator_path)
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def _array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            paths = {
                "depth": self.aligned_cache_dir / "depth_uint8.npy",
                "ir": self.aligned_cache_dir / "ir_uint8.npy",
                "skeleton": self.aligned_cache_dir / "skeleton_float32.npy",
                "imu": self.imu_cache_dir / "imu_float32.npy",
                "imu_time_mask": self.imu_cache_dir / "time_mask_uint8.npy",
                "imu_device_mask": self.imu_cache_dir / "device_mask_uint8.npy",
            }
            self._arrays[name] = np.load(paths[name], mmap_mode="r", allow_pickle=False)
        return self._arrays[name]

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_positions(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros(self.num_frames, dtype=np.int64)
        if length <= self.num_frames:
            return np.rint(np.linspace(0, length - 1, self.num_frames)).astype(np.int64)
        boundaries = np.floor(np.linspace(0, length, self.num_frames + 1)).astype(np.int64)
        positions: list[int] = []
        for index in range(self.num_frames):
            start = int(boundaries[index])
            stop = max(start + 1, int(boundaries[index + 1]))
            if self.train:
                positions.append(int(torch.randint(start, stop, (1,)).item()))
            else:
                positions.append(min(length - 1, (start + stop - 1) // 2))
        return np.asarray(positions, dtype=np.int64)

    @staticmethod
    def _normalise_depth(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).float().div_(255.0)
        mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return tensor.sub_(mean).div_(std)

    @staticmethod
    def _normalise_ir(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(1).float().div_(255.0)
        return tensor.sub_(IMAGENET_GRAY_MEAN).div_(IMAGENET_GRAY_STD)

    def _raw_visual(
        self, row: dict[str, str], flip: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_maps = {
            "depth": frame_map(row["depth_dir"], "depth"),
            "ir": frame_map(row["ir_dir"], "ir"),
            "skeleton": frame_map(row["skeleton_dir"], "skeleton"),
        }
        maps, keys, _ = align_frame_maps(raw_maps)
        if not keys:
            return (
                torch.zeros(self.num_frames, 3, self.image_height, self.image_width),
                torch.zeros(self.num_frames, 1, self.image_height, self.image_width),
                torch.zeros(self.num_frames, 17, 4),
            )
        positions = self._sample_positions(len(keys))
        selected = [keys[index] for index in positions]
        depth_frames: list[np.ndarray] = []
        ir_frames: list[np.ndarray] = []
        for key in selected:
            if maps["depth"]:
                try:
                    with Image.open(maps["depth"][key]) as image:
                        image = image.convert("RGB").resize(
                            (self.image_width, self.image_height), Image.Resampling.BILINEAR
                        )
                        array = np.asarray(image, dtype=np.uint8)
                except (OSError, UnidentifiedImageError):
                    array = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            else:
                array = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            depth_frames.append(np.flip(array, axis=1).copy() if flip else array)
            if maps["ir"]:
                try:
                    with Image.open(maps["ir"][key]) as image:
                        image = image.convert("L").resize(
                            (self.image_width, self.image_height), Image.Resampling.BILINEAR
                        )
                        array_ir = np.asarray(image, dtype=np.uint8)
                except (OSError, UnidentifiedImageError):
                    array_ir = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
            else:
                array_ir = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
            ir_frames.append(np.flip(array_ir, axis=1).copy() if flip else array_ir)
        if maps["skeleton"]:
            skeleton = load_skeleton_sequence(
                [maps["skeleton"][key] for key in keys], flip=flip, strategy="first"
            )[torch.from_numpy(positions)]
        else:
            skeleton = torch.zeros(self.num_frames, 17, 4)
        return (
            self._normalise_depth(np.stack(depth_frames)),
            self._normalise_ir(np.stack(ir_frames)),
            skeleton,
        )

    def _visual(self, sample: P27Sample, flip: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample.sample_id not in self.cache_locations:
            return self._raw_visual(sample.row, flip)
        offset, length = self.cache_locations[sample.sample_id]
        positions = self._sample_positions(length) + offset
        depth = np.asarray(self._array("depth")[positions])
        ir = np.asarray(self._array("ir")[positions])
        skeleton = np.array(self._array("skeleton")[positions], dtype=np.float32, copy=True)
        if flip:
            depth = np.flip(depth, axis=2)
            ir = np.flip(ir, axis=2)
            skeleton[:, :, 0] *= -1.0
            for left, right in LEFT_RIGHT_PAIRS:
                skeleton[:, [left, right]] = skeleton[:, [right, left]]
        return (
            self._normalise_depth(depth),
            self._normalise_ir(ir),
            torch.from_numpy(np.ascontiguousarray(skeleton)),
        )

    def _roi(self, sample_id: str, flip: bool) -> torch.Tensor:
        box = self.locators.get(
            sample_id, np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        ).copy()
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        box[0] -= self.roi_padding * width
        box[2] += self.roi_padding * width
        box[1] -= self.roi_padding * height
        box[3] += self.roi_padding * height
        box = np.clip(box, 0.0, 1.0)
        if flip:
            box[[0, 2]] = 1.0 - box[[2, 0]]
        return torch.from_numpy(box.astype(np.float32))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample = self.samples[index]
        row = sample.row
        flip = bool(self.train and torch.rand(1).item() < 0.5)
        depth, ir, skeleton = self._visual(sample, flip)
        depth_present = float(int(row["depth_usable"]))
        ir_present = float(int(row["ir_usable"]))
        skeleton_present = float(int(row["skeleton_usable"]))
        if not depth_present:
            depth.zero_()
        if not ir_present:
            ir.zero_()
        if not skeleton_present:
            skeleton.zero_()
        imu_index = int(row["imu_cache_index"])
        if imu_index >= 0:
            imu = torch.from_numpy(
                np.array(self._array("imu")[imu_index], dtype=np.float32, copy=True)
            )
            imu_time_mask = torch.from_numpy(
                np.array(self._array("imu_time_mask")[imu_index], dtype=np.float32, copy=True)
            )
            imu_device_mask = torch.from_numpy(
                np.array(self._array("imu_device_mask")[imu_index], dtype=np.float32, copy=True)
            )
        else:
            imu = torch.zeros(5, 32, 10)
            imu_time_mask = torch.zeros(5, 32)
            imu_device_mask = torch.zeros(5)
        return {
            "label": sample.class_id,
            "sample_id": sample.sample_id,
            "subject": sample.user_id,
            "depth": depth,
            "ir": ir,
            "skeleton": skeleton,
            "imu": imu,
            "imu_time_mask": imu_time_mask,
            "imu_device_mask": imu_device_mask,
            "depth_present": torch.tensor([depth_present], dtype=torch.float32),
            "ir_present": torch.tensor([ir_present], dtype=torch.float32),
            "skeleton_present": torch.tensor([skeleton_present], dtype=torch.float32),
            "imu_present": torch.tensor([float(int(row["imu_usable"]))], dtype=torch.float32),
            "roi": self._roi(sample.sample_id, flip),
        }


def compute_fold_imu_stats(
    manifest_path: str | Path,
    outer_fold: int,
    imu_cache_dir: str | Path = DEFAULT_IMU_CACHE,
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        row
        for row in read_csv(Path(manifest_path))
        if int(row["subject_fold"]) != int(outer_fold) and int(row["imu_usable"]) == 1
    ]
    values = np.load(Path(imu_cache_dir) / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(Path(imu_cache_dir) / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    count = np.zeros(10, dtype=np.float64)
    total = np.zeros(10, dtype=np.float64)
    total_sq = np.zeros(10, dtype=np.float64)
    for row in rows:
        index = int(row["imu_cache_index"])
        mask = np.asarray(masks[index], dtype=bool)
        array = np.asarray(values[index], dtype=np.float64)
        valid = array[mask]
        if len(valid):
            count += len(valid)
            total += valid.sum(axis=0)
            total_sq += np.square(valid).sum(axis=0)
    mean = total / np.maximum(count, 1.0)
    variance = total_sq / np.maximum(count, 1.0) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)
