from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
import torch

from src.data.hierarchical_multimodal_dataset import (
    EmptyModalityLoader,
    HierarchicalMultimodalDataset,
    make_midfusion_dataset,
    RawIMULoader,
    StrictVisualLoader,
)
from src.data.canonical_multimodal_index import CanonicalTrial
from src.experiments.hierarchical_midfusion_config import load_midfusion_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml"


def test_dataset_keeps_natural_missing_patterns_with_injected_loaders() -> None:
    config = load_midfusion_config(CONFIG)
    dataset = HierarchicalMultimodalDataset.from_patterns_for_test(
        config,
        patterns=["complete", "missing_imu", "thermal_only"],
        visual_loader=EmptyModalityLoader.visual(),
        skeleton_loader=EmptyModalityLoader.skeleton(),
        imu_loader=EmptyModalityLoader.imu(),
    )

    complete, missing_imu, thermal_only = dataset[0], dataset[1], dataset[2]

    assert complete["present"].tolist() == [True, True, True, True]
    assert missing_imu["present"].tolist() == [True, True, True, False]
    assert thermal_only["present"].tolist() == [False, False, False, False]
    assert bool(thermal_only["core_available"]) is False
    assert len(dataset) == 3


def test_real_population_counts_are_not_filtered() -> None:
    config = load_midfusion_config(CONFIG)
    train = make_midfusion_dataset(config, partition="train", metadata_only=True)
    validation = make_midfusion_dataset(
        config, partition="validation", metadata_only=True
    )

    assert len(train) == 2039
    assert len(validation) == 388
    assert sum(bool(validation[index]["core_available"]) for index in range(len(validation))) == 385


def test_metadata_mode_does_not_claim_present_modalities_are_decoded() -> None:
    config = load_midfusion_config(CONFIG)
    dataset = make_midfusion_dataset(config, partition="validation", metadata_only=True)
    item = next(dataset[index] for index in range(len(dataset)) if dataset[index]["present"].all())

    assert not item["usable"].any()
    assert torch.count_nonzero(item["visual"]) == 0
    assert torch.count_nonzero(item["skeleton"]) == 0
    assert torch.count_nonzero(item["imu"]) == 0


def test_empty_loader_returns_exact_shapes() -> None:
    visual = EmptyModalityLoader.visual()(None)
    skeleton = EmptyModalityLoader.skeleton()(None)
    imu = EmptyModalityLoader.imu()(None)

    assert visual["values"].shape == (2, 4, 3, 16, 224, 224)
    assert visual["availability"].shape == (2, 4)
    assert skeleton["values"].shape == (8, 17, 6)
    assert imu["values"].shape == (8, 5, 16)


def test_imu_manifest_path_missing_is_present_but_unusable(tmp_path: Path) -> None:
    missing = tmp_path / "missing_imu_trial"
    trial = CanonicalTrial(
        sample_id="sample_missing_imu",
        user_id="user1",
        class_id=0,
        paths={
            "ir": None,
            "depth_color": None,
            "skeleton": None,
            "imu": missing,
            "radar": None,
            "thermal": None,
        },
        availability={
            "ir": False,
            "depth_color": False,
            "skeleton": False,
            "imu": True,
            "radar": False,
            "thermal": False,
        },
    )

    result = RawIMULoader()(trial)

    assert not bool(result["modality_usable"])
    assert not result["role_mask"].any()
    assert result["failure_reason"] == "manifest_path_missing"

    dataset = HierarchicalMultimodalDataset(
        [trial],
        visual_loader=EmptyModalityLoader.visual(),
        skeleton_loader=EmptyModalityLoader.skeleton(),
        imu_loader=RawIMULoader(),
    )
    item = dataset[0]

    assert bool(item["present"][3]) is True
    assert bool(item["usable"][3]) is False
    assert item["failure_reasons"][3] == "manifest_path_missing"


class FailingPoseDataset:
    def __init__(self, message: str) -> None:
        self.message = message

    def __getitem__(self, index: int) -> dict[str, object]:
        raise ValueError(self.message)


def _visual_trial(tmp_path: Path) -> CanonicalTrial:
    ir = tmp_path / "ir"
    depth = tmp_path / "depth"
    ir.mkdir()
    depth.mkdir()
    timestamp = "2025-01-01_00-00-00.000_00000001"
    Image.new("L", (16, 16), color=128).save(ir / f"IR_{timestamp}.png")
    Image.new("RGB", (16, 16), color=(64, 96, 128)).save(
        depth / f"Depth_{timestamp}_Color.png"
    )
    paths = {name: None for name in ("ir", "depth_color", "skeleton", "imu", "radar", "thermal")}
    paths.update({"ir": ir, "depth_color": depth})
    return CanonicalTrial(
        sample_id="no_pose_sample",
        user_id="user1",
        class_id=0,
        paths=paths,
        availability={name: paths[name] is not None for name in paths},
    )


def test_visual_loader_degrades_only_exact_no_person_pose_failure(
    tmp_path: Path,
) -> None:
    trial = _visual_trial(tmp_path)
    loader = object.__new__(StrictVisualLoader)
    loader.lookup = {trial.sample_id: 0}
    loader.image_size = 224
    loader.dataset = FailingPoseDataset(
        "fixed trial context has no valid pose probe; full-frame fallback forbidden"
    )

    result = loader(trial)

    assert result["availability"].tolist() == [
        [True, False, False, False],
        [True, False, False, False],
    ]
    assert result["failure_reasons"] == (
        "no_valid_yolo_person_pose_global_only",
        "no_valid_yolo_person_pose_global_only",
    )
    assert torch.count_nonzero(result["values"][:, 0]) > 0
    assert torch.count_nonzero(result["values"][:, 1:]) == 0

    loader.dataset = FailingPoseDataset("person_boxes must have shape [T,4]")
    with pytest.raises(ValueError, match="shape"):
        loader(trial)
