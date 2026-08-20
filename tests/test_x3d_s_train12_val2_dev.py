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


def test_user6_user7_single13_changes_only_temporal_sampling_mode() -> None:
    reference = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_context_train12_val2_user6_user7_partial2.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_context_train12_val2_user6_user7_single13_global.yaml"
        ).read_text(encoding="utf-8")
    )

    assert candidate["temporal"]["sampling_mode"] == "global_single_clip"
    candidate["temporal"].pop("sampling_mode")
    assert candidate == reference


def test_user6_user7_single13_fixed_context_changes_only_spatial_input() -> None:
    reference = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_context_train12_val2_user6_user7_single13_global.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_context_train12_val2_user6_user7_single13_fixed_context.yaml"
        ).read_text(encoding="utf-8")
    )

    assert candidate["spatial_input"] == {
        "crop_mode": "fixed_trial_person_context",
        "pose_cache": (
            "D:\\work\\2026.7.14_kaggle\\40class\\outputs\\"
            "depth_ir_person_crop_40class_fold0\\person_crop_pose_tracks.npz"
        ),
        "detection_frames": 8,
        "crop_margin": 1.4,
        "minimum_side_fraction": 0.35,
        "full_frame_fallback": False,
    }
    candidate.pop("spatial_input")
    assert candidate == reference


def test_ir_depth4_candidate_retains_single13_fixed_context_and_changes_input_only() -> None:
    reference = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_context_train12_val2_user6_user7_single13_fixed_context.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_depth4_train12_val2_user6_user7_single13_fixed_context.yaml"
        ).read_text(encoding="utf-8")
    )

    assert candidate["temporal"]["sampling_mode"] == "global_single_clip"
    assert candidate["spatial_input"] == reference["spatial_input"]
    assert candidate["input_view"] == "depth_color_rgb_plus_ir_gray"
    assert candidate["input_channels"] == 4
    assert candidate["early_fusion"]["channel_order"] == [
        "depth_r", "depth_g", "depth_b", "ir_gray"
    ]
    candidate["input_view"] = reference["input_view"]
    candidate.pop("input_channels")
    candidate.pop("early_fusion")
    assert candidate == reference


def test_ir_depth4_workers4_config_is_loader_only_operational_amendment() -> None:
    reference = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_depth4_train12_val2_user6_user7_single13_fixed_context.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_depth4_train12_val2_user6_user7_single13_fixed_context_workers4.yaml"
        ).read_text(encoding="utf-8")
    )

    assert candidate["loader"] == {
        "max_trials_per_batch": 2,
        "max_valid_clips_per_batch": 8,
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "multiprocessing_context": "spawn",
        "worker_torch_threads": 1,
    }
    candidate["loader"] = reference["loader"]
    assert candidate == reference


def test_ir_anchored_adapter_config_changes_only_registered_model_intervention() -> None:
    reference = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_depth4_train12_val2_user6_user7_single13_fixed_context_workers4.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_anchored_depth_adapter_train12_val2_user6_user7_"
            "single13_fixed_context_workers4.yaml"
        ).read_text(encoding="utf-8")
    )

    assert candidate["fusion_strategy"] == "ir_anchored_depth_residual"
    assert candidate["early_fusion"]["stem_initialization"] == (
        "standard_k400_rgb_after_ir_anchor_zero_depth_residual"
    )
    assert candidate["optimizer"]["input_adapter_lr"] == pytest.approx(3e-4)
    candidate.pop("fusion_strategy")
    candidate["early_fusion"]["stem_initialization"] = (
        reference["early_fusion"]["stem_initialization"]
    )
    candidate["optimizer"].pop("input_adapter_lr")
    assert candidate == reference


def test_ordinal_motion_adapter_changes_only_depth_representation() -> None:
    reference = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_anchored_depth_adapter_train12_val2_user6_user7_"
            "single13_fixed_context_workers4.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = yaml.safe_load(
        Path(
            "configs/experiments/"
            "x3d_s_ir_ordinal_motion_adapter_train12_val2_user6_user7_"
            "single13_fixed_context_workers4.yaml"
        ).read_text(encoding="utf-8")
    )

    assert candidate["input_view"] == "depth_ordinal_motion_plus_ir_gray"
    assert candidate["early_fusion"]["channel_order"] == [
        "depth_relative_displacement",
        "depth_time_normalized_velocity",
        "depth_motion_magnitude",
        "ir_gray",
    ]
    assert candidate["depth_motion"] == {
        "source": "inverse_opencv_jet_ordinal",
        "reference": "per_pixel_temporal_median",
        "displacement_scale": "selected_clip_valid_ordinal_iqr_min_8",
        "velocity_scale": "8_ordinal_levels_per_source_frame",
        "velocity_timebase": "source_frame_index",
        "clip_range": [-1.0, 1.0],
        "invalid_policy": "strict_temporal_intersection_zero_fill",
    }
    candidate["input_view"] = reference["input_view"]
    candidate["early_fusion"]["channel_order"] = reference["early_fusion"][
        "channel_order"
    ]
    candidate.pop("depth_motion")
    assert candidate == reference


def test_ordinal_motion_preregistration_freezes_one_run_and_review_floor() -> None:
    preregistration = json.loads(
        Path(
            "reports/"
            "x3d_s_train12_val2_user6_user7_single13_fixed_context_"
            "ordinal_motion_adapter_preregistration.json"
        ).read_text(encoding="utf-8")
    )

    assert preregistration["run_id"] == (
        "x3d_s_ir_ordinal_motion_adapter_train12_val2_user6_user7_"
        "single13_fixed_context_workers4_seed20260715"
    )
    assert preregistration["population"]["validation_user_ids"] == ["user6", "user7"]
    assert preregistration["metric_contract"]["human_review_accuracy_floor"] == pytest.approx(
        0.528051948051948
    )
    assert preregistration["stability_work_automatically_authorized"] is False
    assert "heldout4_evaluation" in preregistration["forbidden"]
    assert len(preregistration["bound_sha256"]) == 11
    assert all(
        len(value) == 64 for value in preregistration["bound_sha256"].values()
    )


def test_user6_user7_single13_preregistration_freezes_single_variable() -> None:
    preregistration = json.loads(
        Path(
            "reports/"
            "x3d_s_train12_val2_user6_user7_single13_global_preregistration.json"
        ).read_text(encoding="utf-8")
    )

    assert preregistration["approved_post_freeze_exception"] is True
    assert preregistration["development_split"] == (
        "metadata/splits/train12_val2_user6_user7_development.json"
    )
    assert preregistration["sole_intervention"] == {
        "field": "temporal.sampling_mode",
        "reference": "adaptive_local_windows",
        "candidate": "global_single_clip",
        "candidate_definition": (
            "one [0,T) window with 13 equal bins; random within-bin training "
            "sample and deterministic validation midpoint"
        ),
        "effective_clips_per_trial": 1,
    }
    assert preregistration["matched_reference"]["accuracy"] == pytest.approx(
        0.5324675324675324
    )
    assert preregistration["population"]["validation_class_count"] == 40
    assert "heldout4_access" in preregistration["forbidden"]
    assert all(
        len(value) == 64 for value in preregistration["bound_sha256"].values()
    )


def test_user6_user7_fixed_context_preregistration_freezes_spatial_only() -> None:
    preregistration = json.loads(
        Path(
            "reports/"
            "x3d_s_train12_val2_user6_user7_single13_fixed_context_preregistration.json"
        ).read_text(encoding="utf-8")
    )

    assert preregistration["approved_formal_training_ordinal"] == 1
    assert preregistration["approved_formal_training_total"] == 2
    assert preregistration["sole_intervention"]["field"] == "spatial_input.crop_mode"
    assert preregistration["sole_intervention"]["endpoint_uniform_probe_count"] == 8
    assert preregistration["sole_intervention"]["full_frame_fallback"] is False
    assert preregistration["pretraining_spatial_audit"]["fallback_count"] == 0
    assert preregistration["matched_reference"]["accuracy"] == pytest.approx(
        0.522077922077922
    )
    assert "automatic_second_formal_training_launch" in preregistration["forbidden"]
    assert all(len(value) == 64 for value in preregistration["bound_sha256"].values())


def test_ir_depth4_preregistration_locks_point63_stability_gate() -> None:
    preregistration = json.loads(
        Path(
            "reports/"
            "x3d_s_train12_val2_user6_user7_single13_fixed_context_ir_depth4_preregistration.json"
        ).read_text(encoding="utf-8")
    )

    assert preregistration["approved_formal_training_ordinal"] == 2
    assert preregistration["approved_formal_training_total"] == 2
    assert preregistration["stability_review_accuracy_gate"] == pytest.approx(0.63)
    assert preregistration["stability_work_automatically_authorized"] is False
    assert preregistration["sole_intervention"]["channel_order"] == [
        "depth_r", "depth_g", "depth_b", "ir_gray"
    ]
    assert preregistration["fixed_conditions"] == {
        "temporal_sampling": "one globally stratified 13-frame clip",
        "spatial_crop": "one fixed trial person-context box",
    }
    assert "three_fold_training" in preregistration["forbidden"]
    assert "additional_seed" in preregistration["forbidden"]
    assert all(len(value) == 64 for value in preregistration["bound_sha256"].values())


def test_ir_anchored_adapter_preregistration_preserves_ir_anchor() -> None:
    preregistration = json.loads(
        Path(
            "reports/"
            "x3d_s_train12_val2_user6_user7_single13_fixed_context_"
            "ir_anchored_depth_adapter_preregistration.json"
        ).read_text(encoding="utf-8")
    )

    assert preregistration["approved_post_freeze_exception"] is True
    assert preregistration["seed"] == 20260715
    assert preregistration["sole_intervention"] == {
        "reference": "expanded four-channel K400 input stem",
        "candidate": "immutable repeated-IR anchor plus zero-initialized Depth residual",
        "adapter": "bias-free Conv3d 3-to-3 kernel-1",
        "adapter_parameters": 9,
        "adapter_lr": 0.0003,
        "initial_depth_sensitivity": 0,
    }
    assert preregistration["stability_review_accuracy_gate"] == pytest.approx(0.63)
    assert preregistration["stability_work_automatically_authorized"] is False
    assert "three_fold_training" in preregistration["forbidden"]
    assert "additional_seed" in preregistration["forbidden"]
    assert all(len(value) == 64 for value in preregistration["bound_sha256"].values())


def test_ir_depth4_workers4_amendment_is_manual_and_forbids_automatic_fallback() -> None:
    amendment = json.loads(
        Path(
            "reports/"
            "x3d_s_train12_val2_user6_user7_single13_fixed_context_ir_depth4_"
            "loader_amendment.json"
        ).read_text(encoding="utf-8")
    )

    assert amendment["manual_user_authorization"] is True
    assert amendment["preferred_num_workers"] == 4
    assert amendment["fallback_num_workers"] == 2
    assert amendment["fallback_requires_failed_memory_smoke"] is True
    assert amendment["scientific_intervention_changed"] is False
    assert amendment["aborted_workers0_run"]["completed_epochs"] == 1
    assert amendment["memory_smoke_gate"]["minimum_system_available_gib"] == 4.0
    assert amendment["formal_training_launch_authorized_after_passing_smoke"] is True

    smoke = json.loads(
        Path(
            "reports/"
            "x3d_s_train12_val2_user6_user7_single13_fixed_context_ir_depth4_"
            "workers4_memory_smoke.json"
        ).read_text(encoding="utf-8")
    )
    assert smoke["decision"] == "pass_workers4_nonpersistent"
    assert smoke["exit_code"] == 0
    assert smoke["minimum_system_available_gib"] >= 4.0
    assert smoke["fallback_to_workers2_required"] is False


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
