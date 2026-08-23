from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.probe_ir_depth_videomaev2_teacher import load_probe_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/ir_depth_videomaev2_vit_b_p0.yaml"


def test_p0_config_freezes_real_probe_contract() -> None:
    config = load_probe_config(CONFIG)

    assert config["input"]["frames"] == 16
    assert config["input"]["image_size"] == 224
    assert config["input"]["modalities"] == ["ir", "depth"]
    assert config["input"]["views"] == [
        "global",
        "person_context",
        "left_hand_object",
        "right_hand_object",
    ]
    assert config["runtime"]["physical_batch_trials"] == 1
    assert config["runtime"]["amp_dtype"] == "bfloat16"
    assert config["runtime"]["sequential_multiview_backward"] is True
    assert config["gates"]["peak_allocated_mib_below"] == 7300


def test_p0_config_rejects_silent_frame_or_view_reduction(tmp_path: Path) -> None:
    import yaml

    config = load_probe_config(CONFIG)
    changed = copy.deepcopy(config)
    changed["input"]["frames"] = 8
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="16 frames"):
        load_probe_config(path)
