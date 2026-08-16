from __future__ import annotations

from pathlib import Path
import json

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
NEW_TRAIN_USERS = [
    "user1", "user2", "user3", "user5", "user8", "user9",
    "user16", "user18", "user19", "user20", "user21", "user22",
]
NEW_VAL_USERS = ["user6", "user7"]
HELDOUT_USERS = ["user4", "user17", "user23", "user24"]


def _split() -> dict[str, object]:
    return {
        "name": "train12_val2_user21_user22",
        "development_only": True,
        "train_user_ids": TRAIN_USERS,
        "validation_user_ids": VAL_USERS,
        "ir_audit": {
            "train_usable_trials": 1996,
            "validation_usable_trials": 324,
            "train_class_count": 40,
            "validation_class_count": 36,
            "validation_missing_class_ids": [25, 26, 33, 35],
        },
        "metric_policy": {"num_classes": 40},
    }


def _new_split() -> dict[str, object]:
    return {
        "name": "train12_val2_user6_user7",
        "development_only": True,
        "train_user_ids": NEW_TRAIN_USERS,
        "validation_user_ids": NEW_VAL_USERS,
        "heldout_user_ids": HELDOUT_USERS,
        "ir_audit": {
            "train_usable_trials": 1935,
            "validation_usable_trials": 385,
            "train_class_count": 40,
            "validation_class_count": 40,
            "validation_missing_class_ids": [],
            "validation_minimum_class_support": 2,
        },
        "metric_policy": {
            "num_classes": 40,
            "worst_user_population": NEW_VAL_USERS,
        },
    }


def test_validate_split_contract_freezes_exact_train12_val2_users() -> None:
    train, validation = validate_split_contract(_split())

    assert train == tuple(TRAIN_USERS)
    assert validation == tuple(VAL_USERS)
    assert set(train).isdisjoint(validation)


def test_validate_split_contract_accepts_named_user6_user7_profile() -> None:
    train, validation = validate_split_contract(_new_split())

    assert train == tuple(NEW_TRAIN_USERS)
    assert validation == tuple(NEW_VAL_USERS)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("heldout_user_ids", HELDOUT_USERS[:-1], "heldout"),
        ("train_user_ids", NEW_TRAIN_USERS[:-1] + ["user4"], "exact frozen"),
        ("validation_user_ids", ["user6", "user8"], "exact frozen"),
        (
            "ir_audit",
            {
                "train_usable_trials": 1935,
                "validation_usable_trials": 384,
                "train_class_count": 40,
                "validation_class_count": 40,
                "validation_missing_class_ids": [],
                "validation_minimum_class_support": 2,
            },
            "IR audit",
        ),
    ],
)
def test_validate_new_profile_rejects_contract_drift(
    field: str, value: object, message: str
) -> None:
    split = _new_split()
    split[field] = value

    with pytest.raises(ValueError, match=message):
        validate_split_contract(split)


def test_validate_split_contract_rejects_unknown_profile() -> None:
    split = _new_split()
    split["name"] = "train12_val2_unregistered"

    with pytest.raises(ValueError, match="registered named profile"):
        validate_split_contract(split)


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


def test_resolve_effective_config_records_new_split_identity() -> None:
    config = {"output_root": str(DEV_OUTPUT_ROOT)}

    resolved = resolve_effective_config(config, split=_new_split(), seed=20260715)

    assert resolved["development_split_name"] == "train12_val2_user6_user7"
    assert resolved["development_partition"] == {
        "train_user_ids": NEW_TRAIN_USERS,
        "validation_user_ids": NEW_VAL_USERS,
    }


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


def test_user6_user7_partial2_changes_only_development_partition() -> None:
    historical = yaml.safe_load(
        Path("configs/experiments/x3d_s_ir_context_train12_val2_partial2.yaml")
        .read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_context_train12_val2_user6_user7_partial2.yaml"
        ).read_text(encoding="utf-8")
    )

    assert candidate["development_partition"] == {
        "train_user_ids": NEW_TRAIN_USERS,
        "validation_user_ids": NEW_VAL_USERS,
    }
    candidate["development_partition"] = historical["development_partition"]
    assert candidate == historical


def test_user6_user7_partial2_preregistration_freezes_generation_r() -> None:
    preregistration = json.loads(
        Path("reports/x3d_s_train12_val2_user6_user7_partial2_preregistration.json")
        .read_text(encoding="utf-8")
    )

    assert preregistration["generation"] == "R"
    assert preregistration["candidate"] == "train12_val2_user6_user7_partial2"
    assert preregistration["development_split"] == (
        "metadata/splits/train12_val2_user6_user7_development.json"
    )
    assert preregistration["run_id"] == (
        "x3d_s_ir_context_train12_val2_user6_user7_partial2_seed20260715"
    )
    assert preregistration["metric_contract"] == {
        "accuracy": "385 usable-IR trials from user6/user7",
        "macro_f1": "fixed labels 0..39; all 40 classes observed",
        "worst_user_accuracy": "minimum of user6 and user7 Accuracy",
        "checkpoint_objective": "Accuracy then Macro-F1 then earlier epoch",
    }
    assert preregistration["direct_head_implementation_permitted"] is False
    assert preregistration["direct_head_result_inspected"] is False
    assert set(preregistration["bound_sha256"]) == {
        "config",
        "parent_partial2_config",
        "development_split",
        "canonical_assignment",
        "canonical_phase5_oof_evidence",
        "direct_head_design_spec",
    }
    assert all(
        len(value) == 64 for value in preregistration["bound_sha256"].values()
    )


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
