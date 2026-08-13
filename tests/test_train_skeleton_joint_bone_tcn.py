from __future__ import annotations

from pathlib import Path

import yaml

from src.train_skeleton_joint_bone_tcn_strict_oof import model_for


def test_d2_v1_config_freezes_c1_training_contract() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/skeleton_joint_bone_tcn_strict_oof.yaml").read_text(encoding="utf-8")
    )

    assert config["input_features"] == 198
    assert config["sequence_length"] == 64
    assert config["scale_policy"] == "per_frame"
    assert config["tcn_channels"] == [64, 128]
    assert config["embedding_dim"] == 128
    assert config["seed"] == 20260812
    assert sum(parameter.numel() for parameter in model_for(config).parameters()) == 209_640
