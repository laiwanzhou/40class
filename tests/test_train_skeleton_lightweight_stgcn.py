from __future__ import annotations

from pathlib import Path

import yaml


def test_d1_v1_config_freezes_canonical_seed_and_kernel() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/skeleton_lightweight_stgcn_strict_oof.yaml").read_text(encoding="utf-8")
    )

    assert config["seed"] == 20260812
    assert config["temporal_kernel_size"] == 5
