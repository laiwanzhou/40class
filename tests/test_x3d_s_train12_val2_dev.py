from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.run_x3d_s_train12_val2_dev import (
    DEV_OUTPUT_ROOT,
    resolve_effective_config,
    validate_split_contract,
)


TRAIN_USERS = [
    "user1", "user2", "user3", "user5", "user6", "user7",
    "user8", "user9", "user16", "user18", "user19", "user20",
]
VAL_USERS = ["user21", "user22"]


def _split() -> dict[str, object]:
    return {
        "development_only": True,
        "train_user_ids": TRAIN_USERS,
        "validation_user_ids": VAL_USERS,
        "metric_policy": {"num_classes": 40},
    }


def test_validate_split_contract_freezes_exact_train12_val2_users() -> None:
    train, validation = validate_split_contract(_split())

    assert train == tuple(TRAIN_USERS)
    assert validation == tuple(VAL_USERS)
    assert set(train).isdisjoint(validation)


def test_validate_split_contract_rejects_overlap() -> None:
    split = _split()
    split["validation_user_ids"] = ["user20", "user22"]

    with pytest.raises(ValueError, match="exact frozen"):
        validate_split_contract(split)


def test_resolve_effective_config_replaces_stale_partition_and_output() -> None:
    config = {
        "output_root": str(DEV_OUTPUT_ROOT),
        "development_partition": {
            "train_user_ids": ["stale"],
            "validation_user_ids": ["stale"],
        },
    }

    resolved = resolve_effective_config(config, split=_split(), seed=20260715)

    assert resolved["development_partition"] == {
        "train_user_ids": TRAIN_USERS,
        "validation_user_ids": VAL_USERS,
    }
    assert resolved["seed"] == 20260715


def test_resolve_effective_config_rejects_unprotected_output_root() -> None:
    config = {"output_root": "outputs/x3d_s_ir_context_oof"}

    with pytest.raises(ValueError, match="protected output root"):
        resolve_effective_config(config, split=_split(), seed=20260715)


def test_partial2_config_keeps_full_temporal_coverage() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/x3d_s_ir_context_train12_val2_partial2.yaml")
        .read_text(encoding="utf-8")
    )

    assert "train_clip_keep_fraction" not in config["temporal"]
    assert config["training"]["unfrozen_backbone_blocks"] == 2
    assert config["training"]["warmup_epochs"] == 2
    assert config["optimizer"]["backbone_lr"] == pytest.approx(3e-5)
    assert config["augmentation"]["brightness"] == [0.9, 1.1]
    assert config["training"]["label_smoothing"] == 0.0


def test_partial1_changes_only_unfrozen_backbone_block_count() -> None:
    partial2 = yaml.safe_load(
        Path("configs/experiments/x3d_s_ir_context_train12_val2_partial2.yaml")
        .read_text(encoding="utf-8")
    )
    partial1 = yaml.safe_load(
        Path("configs/experiments/x3d_s_ir_context_train12_val2_partial1.yaml")
        .read_text(encoding="utf-8")
    )

    assert partial2["training"]["unfrozen_backbone_blocks"] == 2
    assert partial1["training"]["unfrozen_backbone_blocks"] == 1
    partial2["training"]["unfrozen_backbone_blocks"] = 1
    assert partial1 == partial2


def test_layerwise_lr1_changes_only_block_specific_learning_rates() -> None:
    partial2 = yaml.safe_load(
        Path("configs/experiments/x3d_s_ir_context_train12_val2_partial2.yaml")
        .read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path("configs/experiments/x3d_s_ir_context_train12_val2_layerwise_lr1.yaml")
        .read_text(encoding="utf-8")
    )

    assert candidate["optimizer"]["backbone_block_lrs"] == {4: 3e-6, 5: 1e-5}
    candidate["optimizer"].pop("backbone_block_lrs")
    assert candidate == partial2
