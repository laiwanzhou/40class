from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from audit_p106_hard_confusion_forensics import selection_key, verdict
from p106_forensic_features import FINE_BLOCKS


def test_fine_block_panel_is_frozen_in_protocol_order() -> None:
    assert FINE_BLOCKS == (
        "local_interaction_temporal",
        "local_hand_phase_delta",
        "global_visual_phase_delta",
        "skeleton_wrist_arm_temporal",
        "imu_arm_phase_direction",
        "depth_person_workspace_geometry",
        "thermal_person_workspace_interaction",
    )


def test_source_selection_tie_uses_fixed_block_order() -> None:
    metrics = {
        "balanced_accuracy": 0.75,
        "macro_f1": 0.74,
        "accuracy": 0.76,
    }
    first = {"block": FINE_BLOCKS[0], "source_inner_specialist": metrics}
    second = {"block": FINE_BLOCKS[1], "source_inner_specialist": metrics}

    assert selection_key(first) > selection_key(second)


def test_candidate_verdict_requires_causal_and_stability_gates() -> None:
    aggregate = {
        "selected_fold_count": 3,
        "variants": {"aligned": {"net": 4}},
        "aligned_minus_shuffle_correct": 2,
        "aligned_minus_zero_correct": 5,
        "positive_folds": 2,
        "positive_subjects": 3,
        "negative_subjects": 2,
        "deployable_trigger_net": 0,
    }

    assert verdict(aggregate) == "FORENSIC_SPECIALIST_CANDIDATE"
    aggregate["aligned_minus_shuffle_correct"] = 0
    assert verdict(aggregate) == "NO_CURRENT_FINE_FEATURE_EVIDENCE"
    aggregate["aligned_minus_shuffle_correct"] = 2
    aggregate["positive_folds"] = 1
    assert verdict(aggregate) == "MECHANISM_ONLY"
