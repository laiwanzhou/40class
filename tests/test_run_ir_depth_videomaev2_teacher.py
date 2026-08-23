from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_ir_depth_videomaev2_teacher import (
    build_inverse_frequency_sampler,
    cosine_warmup_factor,
    load_training_config,
    require_training_authorization,
    validate_p0_binding,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/ir_depth_videomaev2_vit_b_p1.yaml"


def test_p1_config_requires_passed_p0_and_freezes_one_run() -> None:
    config = load_training_config(CONFIG)

    assert config["p0_report"] == "reports/ir_depth_videomaev2_vit_b_p0_probe.json"
    assert config["training"]["epochs"] == 20
    assert config["training"]["gradient_accumulation"] == 8
    assert config["training"]["unfrozen_backbone_blocks"] == 12
    assert config["training"]["sampler"] == "inverse_frequency_replacement"
    assert config["training"]["early_stopping_patience"] == 6


def test_p1_requires_exact_user_authorization_token() -> None:
    config = load_training_config(CONFIG)

    with pytest.raises(PermissionError, match="authorization"):
        require_training_authorization(config, token="wrong")
    require_training_authorization(
        config, token=str(config["authorization"]["token"])
    )


def test_inverse_frequency_sampler_gives_each_class_equal_mass() -> None:
    class_ids = [0, 0, 0, 1, 2, 2]
    _, audit = build_inverse_frequency_sampler(class_ids, seed=9)

    assert audit["class_counts"] == {"0": 3, "1": 1, "2": 2}
    assert np.allclose(list(audit["class_probability_mass"].values()), [1 / 3] * 3)


def test_warmup_then_cosine_schedule_matches_frozen_recipe() -> None:
    assert cosine_warmup_factor(0, warmup_steps=20, total_steps=100) == pytest.approx(0.05)
    assert cosine_warmup_factor(19, warmup_steps=20, total_steps=100) == pytest.approx(1.0)
    assert cosine_warmup_factor(100, warmup_steps=20, total_steps=100) == pytest.approx(0.0)


def test_p0_report_is_bound_to_current_config_sources_and_checkpoint() -> None:
    import json

    config = load_training_config(CONFIG)
    report = json.loads((ROOT / str(config["p0_report"])).read_text(encoding="utf-8"))

    validate_p0_binding(config, report)
