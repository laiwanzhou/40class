from __future__ import annotations

import pytest

from scripts.report_x3d_s_ordinal_motion_adapter import (
    CHANNEL_ORDER,
    DEPTH_MOTION_CONTRACT,
    EXPECTED_RUN_ID,
    decision,
    validate_candidate_identity,
)


def _config() -> dict[str, object]:
    return {
        "input_view": "depth_ordinal_motion_plus_ir_gray",
        "input_channels": 4,
        "fusion_strategy": "ir_anchored_depth_residual",
        "early_fusion": {
            "channel_order": list(CHANNEL_ORDER),
            "stem_initialization": (
                "standard_k400_rgb_after_ir_anchor_zero_depth_residual"
            ),
        },
        "depth_motion": dict(DEPTH_MOTION_CONTRACT),
        "optimizer": {"input_adapter_lr": 3e-4},
        "temporal": {"sampling_mode": "global_single_clip", "local_frames": 13},
        "spatial_input": {"crop_mode": "fixed_trial_person_context"},
        "loader": {"num_workers": 4, "persistent_workers": False},
    }


def _validate(config: dict[str, object], *, run_id: str = EXPECTED_RUN_ID) -> None:
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


def test_identity_contract_accepts_registered_ordinal_motion_run() -> None:
    _validate(_config())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("run_id", "run ID"),
        ("input_view", "input view"),
        ("channel_order", "channel order"),
        ("depth_motion", "motion contract"),
    ],
)
def test_identity_contract_rejects_wrong_candidate(
    mutation: str, message: str
) -> None:
    config = _config()
    run_id = EXPECTED_RUN_ID
    if mutation == "run_id":
        run_id = "wrong-run"
    elif mutation == "input_view":
        config["input_view"] = "depth_color_rgb_plus_ir_gray"
    elif mutation == "channel_order":
        config["early_fusion"]["channel_order"] = ["wrong"]  # type: ignore[index]
    else:
        config["depth_motion"] = {"source": "wrong"}
    with pytest.raises(ValueError, match=message):
        _validate(config, run_id=run_id)


def test_decision_uses_current_anchor_and_preserves_regressions() -> None:
    assert decision({"accuracy": 0.64}) == "eligible_for_manual_stability_review"
    assert decision({"accuracy": 0.55}) == "improved_but_below_stability_gate"
    assert decision({"accuracy": 0.54}) == "non_winning_ablation"
    assert decision({"accuracy": 0.52}) == "human_review_regression"
