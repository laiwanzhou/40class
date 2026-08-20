from __future__ import annotations

import pytest

from scripts.report_x3d_s_ir_anchored_depth_adapter import (
    EXPECTED_RUN_ID,
    decision,
    validate_candidate_identity,
)


def _config() -> dict[str, object]:
    return {
        "input_view": "depth_color_rgb_plus_ir_gray",
        "input_channels": 4,
        "fusion_strategy": "ir_anchored_depth_residual",
        "early_fusion": {
            "channel_order": ["depth_r", "depth_g", "depth_b", "ir_gray"],
            "stem_initialization": (
                "standard_k400_rgb_after_ir_anchor_zero_depth_residual"
            ),
            "synchronized_geometric_transform": True,
            "depth_photometric_augmentation": False,
            "ir_photometric_augmentation": True,
        },
        "optimizer": {"input_adapter_lr": 3e-4},
        "temporal": {"sampling_mode": "global_single_clip", "local_frames": 13},
        "spatial_input": {"crop_mode": "fixed_trial_person_context"},
        "loader": {"num_workers": 4, "persistent_workers": False},
    }


def test_identity_contract_accepts_only_registered_adapter_run() -> None:
    validate_candidate_identity(
        run_id=EXPECTED_RUN_ID,
        config=_config(),
        summary={"resolved_config_sha256": "a" * 64},
        provenance={
            "resolved_config_sha256": "a" * 64,
            "development_split_name": "train12_val2_user6_user7",
            "seed": 20260715,
            "smoke_test": False,
        },
        computed_config_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "wrong-run", "run ID"),
        ("fusion_strategy", "expanded_input_stem", "fusion strategy"),
        ("input_channels", 3, "four input channels"),
    ],
)
def test_identity_contract_rejects_wrong_candidate(
    field: str, value: object, message: str
) -> None:
    config = _config()
    run_id = EXPECTED_RUN_ID
    if field == "run_id":
        run_id = str(value)
    else:
        config[field] = value

    with pytest.raises(ValueError, match=message):
        validate_candidate_identity(
            run_id=run_id,
            config=config,
            summary={"resolved_config_sha256": "a" * 64},
            provenance={
                "resolved_config_sha256": "a" * 64,
                "development_split_name": "train12_val2_user6_user7",
                "seed": 20260715,
                "smoke_test": False,
            },
            computed_config_sha256="a" * 64,
        )


def test_adapter_decision_contract() -> None:
    assert decision({"accuracy": 0.64}) == "eligible_for_manual_stability_review"
    assert decision({"accuracy": 0.54}) == "improved_but_below_stability_gate"
    assert decision({"accuracy": 0.51}) == "human_review_regression"

