from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from scripts.probe_ir_depth_videomaev2_teacher import load_probe_config
from src.data.canonical_multimodal_index import (
    CORE_MODALITIES,
    CanonicalTrial,
    build_canonical_trials,
    normalized_segment_bounds,
)
from src.data.clean_skeleton_segments import load_skeleton_segments
from src.data.common import sorted_files
from src.data.ir_depth_videomaev2_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IRDepthVideoMAEV2Dataset,
    uniform_trial_indices,
)
from src.data.raw_imu_segments import load_raw_imu_segments
from src.experiments.hierarchical_midfusion_config import project_path


ModalityLoader = Callable[[CanonicalTrial | None], dict[str, torch.Tensor]]


class EmptyModalityLoader:
    @staticmethod
    def visual() -> ModalityLoader:
        def load(_: CanonicalTrial | None) -> dict[str, torch.Tensor]:
            return {
                "values": torch.zeros(2, 4, 3, 16, 224, 224),
                "availability": torch.zeros(2, 4, dtype=torch.bool),
                "modality_usable": torch.zeros(2, dtype=torch.bool),
            }

        return load

    @staticmethod
    def skeleton() -> ModalityLoader:
        def load(_: CanonicalTrial | None) -> dict[str, torch.Tensor]:
            return {
                "values": torch.zeros(8, 17, 6),
                "mask": torch.zeros(8, dtype=torch.bool),
                "quality": torch.zeros(8, 4),
                "modality_usable": torch.tensor(False),
            }

        return load

    @staticmethod
    def imu() -> ModalityLoader:
        def load(_: CanonicalTrial | None) -> dict[str, torch.Tensor]:
            return {
                "values": torch.zeros(8, 5, 16),
                "role_mask": torch.zeros(8, 5, dtype=torch.bool),
                "quality": torch.zeros(8, 5, 3),
                "modality_usable": torch.tensor(False),
            }

        return load


class StrictVisualLoader:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        partition: str,
        training: bool,
    ) -> None:
        data = config["data"]
        p0 = load_probe_config(project_path(str(data["p0_config"])))
        dataset = IRDepthVideoMAEV2Dataset(
            manifest_path=project_path(str(config["population"]["manifest"])),
            split_path=project_path(str(config["population"]["development_split"])),
            data_root=Path(str(data["root"])),
            pose_cache_path=Path(str(data["pose_cache"])),
            pairing_audit_path=project_path(str(data["pairing_audit"])),
            partition=partition,
            training=training,
            frames=16,
            image_size=int(data["image_size"]),
            temporal_jitter=0.0,
            interaction_config=dict(p0["roi"]),
        )
        self.dataset = dataset
        self.lookup = {
            str(row["sample_id"]): index for index, row in enumerate(dataset.samples)
        }
        self.image_size = int(data["image_size"])

    def _ir_global_fallback(self, trial: CanonicalTrial) -> dict[str, torch.Tensor]:
        output = EmptyModalityLoader.visual()(trial)
        ir_path = trial.paths["ir"]
        if ir_path is None:
            return output
        paths = sorted_files(ir_path, {".png", ".jpg", ".jpeg", ".bmp"})
        if not paths:
            return output
        indices = uniform_trial_indices(len(paths), 16, jitter=0.0)
        frames = []
        for index in indices:
            with Image.open(paths[int(index)]) as image:
                resized = TF.resize(
                    image.convert("L"),
                    [self.image_size, self.image_size],
                    antialias=True,
                )
                tensor = TF.to_tensor(resized).repeat(3, 1, 1)
                frames.append((tensor - IMAGENET_MEAN) / IMAGENET_STD)
        output["values"][0, 0] = torch.stack(frames, dim=1)
        output["availability"][0, 0] = True
        output["modality_usable"][0] = True
        return output

    def __call__(self, trial: CanonicalTrial | None) -> dict[str, torch.Tensor]:
        if trial is None:
            return EmptyModalityLoader.visual()(trial)
        index = self.lookup.get(trial.sample_id)
        if index is None:
            return self._ir_global_fallback(trial)
        item = self.dataset[index]
        availability = item["availability"].bool()
        return {
            "values": item["clips"].float(),
            "availability": availability,
            "modality_usable": availability.any(dim=1),
        }


class CleanSkeletonLoader:
    def __init__(self, clean_view: Path, data_root: Path) -> None:
        frame = pd.read_csv(clean_view, encoding="utf-8-sig", dtype={"sample_id": str})
        self.lookup = {
            str(sample_id): group.reset_index(drop=True)
            for sample_id, group in frame.groupby("sample_id", sort=False)
        }
        self.data_root = data_root

    def __call__(self, trial: CanonicalTrial | None) -> dict[str, torch.Tensor]:
        if trial is None or trial.sample_id not in self.lookup:
            return EmptyModalityLoader.skeleton()(trial)
        result = load_skeleton_segments(
            self.lookup[trial.sample_id], data_root=self.data_root, segment_count=8
        )
        return {
            "values": result.features,
            "mask": result.mask,
            "quality": result.quality,
            "modality_usable": result.mask.any(),
        }


class RawIMULoader:
    def __call__(self, trial: CanonicalTrial | None) -> dict[str, torch.Tensor]:
        if trial is None or trial.paths["imu"] is None:
            return EmptyModalityLoader.imu()(trial)
        result = load_raw_imu_segments(trial.paths["imu"], segment_count=8)
        return {
            "values": result.features,
            "role_mask": result.role_mask,
            "quality": result.quality,
            "modality_usable": result.role_mask.any(),
        }


class HierarchicalMultimodalDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        trials: list[CanonicalTrial],
        *,
        visual_loader: ModalityLoader,
        skeleton_loader: ModalityLoader,
        imu_loader: ModalityLoader,
        metadata_only: bool = False,
    ) -> None:
        self.trials = list(trials)
        self.visual_loader = visual_loader
        self.skeleton_loader = skeleton_loader
        self.imu_loader = imu_loader
        self.metadata_only = bool(metadata_only)

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index: int) -> dict[str, object]:
        trial = self.trials[index]
        present = torch.tensor(
            [trial.availability[name] for name in CORE_MODALITIES], dtype=torch.bool
        )
        if self.metadata_only:
            visual = EmptyModalityLoader.visual()(trial)
            skeleton = EmptyModalityLoader.skeleton()(trial)
            imu = EmptyModalityLoader.imu()(trial)
            core_available = present.any()
        else:
            visual = self.visual_loader(trial)
            skeleton = self.skeleton_loader(trial)
            imu = self.imu_loader(trial)
            core_available = torch.stack(
                (
                    visual["modality_usable"][0],
                    visual["modality_usable"][1],
                    skeleton["modality_usable"],
                    imu["modality_usable"],
                )
            ).any()
        usable = torch.stack(
            (
                visual["modality_usable"][0],
                visual["modality_usable"][1],
                skeleton["modality_usable"],
                imu["modality_usable"],
            )
        ).bool()
        return {
            "visual": visual["values"],
            "visual_view_availability": visual["availability"],
            "visual_segment_indices": torch.from_numpy(
                normalized_segment_bounds(16, segments=8)
            ),
            "skeleton": skeleton["values"],
            "skeleton_mask": skeleton["mask"],
            "skeleton_quality": skeleton["quality"],
            "imu": imu["values"],
            "imu_role_mask": imu["role_mask"],
            "imu_quality": imu["quality"],
            "present": present,
            "usable": usable,
            "availability": usable,
            "core_available": core_available.bool(),
            "sample_id": trial.sample_id,
            "user_id": trial.user_id,
            "label": trial.class_id,
        }

    @classmethod
    def from_patterns_for_test(
        cls,
        config: dict[str, Any],
        *,
        patterns: list[str],
        visual_loader: ModalityLoader,
        skeleton_loader: ModalityLoader,
        imu_loader: ModalityLoader,
    ) -> "HierarchicalMultimodalDataset":
        availability_by_pattern = {
            "complete": [True, True, True, True],
            "missing_imu": [True, True, True, False],
            "thermal_only": [False, False, False, False],
        }
        trials = []
        for index, pattern in enumerate(patterns):
            values = availability_by_pattern[pattern]
            availability = dict(zip(CORE_MODALITIES, values, strict=True))
            availability.update({"radar": False, "thermal": pattern == "thermal_only"})
            paths = {
                name: Path(f"{name}/{index}") if is_present else None
                for name, is_present in availability.items()
            }
            trials.append(
                CanonicalTrial(
                    sample_id=f"sample_{index}",
                    user_id=str(config["population"]["train_user_ids"][0]),
                    class_id=index,
                    paths=paths,
                    availability=availability,
                )
            )
        return cls(
            trials,
            visual_loader=visual_loader,
            skeleton_loader=skeleton_loader,
            imu_loader=imu_loader,
            metadata_only=True,
        )


def make_midfusion_dataset(
    config: dict[str, Any],
    *,
    partition: str,
    metadata_only: bool = False,
    skeleton_clean_view: Path | None = None,
    training: bool = False,
) -> HierarchicalMultimodalDataset:
    data = config["data"]
    trials = build_canonical_trials(
        project_path(str(config["population"]["manifest"])),
        project_path(str(config["population"]["development_split"])),
        Path(str(data["root"])),
        partition=partition,
    )
    if metadata_only:
        visual_loader = EmptyModalityLoader.visual()
        skeleton_loader = EmptyModalityLoader.skeleton()
        imu_loader = EmptyModalityLoader.imu()
    else:
        visual_loader = StrictVisualLoader(
            config, partition=partition, training=training
        )
        skeleton_loader = (
            CleanSkeletonLoader(skeleton_clean_view, Path(str(data["root"])))
            if skeleton_clean_view is not None
            else EmptyModalityLoader.skeleton()
        )
        imu_loader = RawIMULoader()
    return HierarchicalMultimodalDataset(
        trials,
        visual_loader=visual_loader,
        skeleton_loader=skeleton_loader,
        imu_loader=imu_loader,
        metadata_only=metadata_only,
    )
