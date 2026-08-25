from __future__ import annotations

from pathlib import Path

import torch

from src.data.hierarchical_multimodal_dataset import (
    EmptyModalityLoader,
    HierarchicalMultimodalDataset,
    make_midfusion_dataset,
)
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
