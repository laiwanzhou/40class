from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.models import TemporalClassifier
from src.train_skeleton_t96_tcn_strict_oof import dataset_for, model_for
from src.train_skeleton_c0_c1_strict_oof import resolve


def test_t2_v1_config_differs_from_c1_only_in_sequence_length_and_outputs() -> None:
    c1 = yaml.safe_load(Path("configs/experiments/skeleton_c0_c1_strict_oof.yaml").read_text(encoding="utf-8"))
    t2 = yaml.safe_load(Path("configs/experiments/skeleton_t96_tcn_strict_oof.yaml").read_text(encoding="utf-8"))
    ignored = {"output_root", "report_dir", "scale_policy", "input_features", "device"}

    for key, value in c1.items():
        if key in ignored or key == "representations":
            continue
        assert t2[key] == (96 if key == "sequence_length" else value)
    assert t2["scale_policy"] == "per_frame"
    assert t2["input_features"] == 102
    assert t2["sequence_length"] == 96


def test_t2_uses_original_c1_temporal_classifier() -> None:
    config = yaml.safe_load(Path("configs/experiments/skeleton_t96_tcn_strict_oof.yaml").read_text(encoding="utf-8"))
    model = model_for(config)

    assert type(model) is TemporalClassifier
    assert sum(parameter.numel() for parameter in model.parameters()) == 172_776


def test_t2_real_dataset_returns_96_by_102_with_gap_mask() -> None:
    config = yaml.safe_load(Path("configs/experiments/skeleton_t96_tcn_strict_oof.yaml").read_text(encoding="utf-8"))
    assignment = json.loads(Path("metadata/splits/train14_oof_3fold.json").read_text(encoding="utf-8"))
    fold = assignment["folds"][0]
    dataset = dataset_for(
        resolve(config["strict_views_root"]) / "fold_0/inner_selection/clean_view.csv",
        resolve(config["data_root"]), list(fold["epoch_selection"]["fit_user_ids"]), config,
    )
    item = dataset[0]

    assert item["input"].shape == (96, 102)
    assert item["temporal_mask"].shape == (96,)
    assert torch.all(item["input"][~item["temporal_mask"]] == 0)
