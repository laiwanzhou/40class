from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_p109_visual_first_tree import (  # noqa: E402
    HARD_CLASSES,
    LEAF_RECIPES,
    ROOT_GROUPS,
    SUPPORT_CLASSES,
    VISUAL_BLOCKS,
    deployable_entry,
    first_failure,
    hard_class_paths,
    validate_tree,
)


def test_tree_partition_and_hard_coverage() -> None:
    validate_tree()
    assert len(SUPPORT_CLASSES) == 26
    assert len(HARD_CLASSES) == 10
    assert set(HARD_CLASSES) <= set(SUPPORT_CLASSES)
    assert len(ROOT_GROUPS) == 4
    paths = hard_class_paths()
    assert paths[37] == ("HAND_OBJECT_ORAL_TABLE", "ORAL_INTAKE")
    assert paths[10] == ("HAND_OBJECT_ORAL_TABLE", "TABLEWARE_MANIPULATION")
    assert paths[22] == ("DOCUMENT_HAND", "DOCUMENT_ACTIVITY")
    assert paths[24] == ("PERSONAL_DEVICE_BODY", "PHONE_DEVICE")
    assert paths[39] == ("PERSONAL_DEVICE_BODY", "BODY_CONTACT_DEVICE")
    assert paths[34] == ("POSTURE_TRANSITION", "POSTURE_MOTION")


def test_visual_is_mandatory_and_auxiliary_is_leaf_only() -> None:
    assert VISUAL_BLOCKS == ("VLIT", "VHPD", "VWPD")
    assert len(LEAF_RECIPES) == 12
    for recipe in LEAF_RECIPES:
        assert recipe[0] in VISUAL_BLOCKS
        assert sum(value in VISUAL_BLOCKS for value in recipe) == 1
        assert set(recipe[1:]) <= {"SWT", "IAPD"}


def test_failure_localization_is_earliest_layer() -> None:
    assert first_failure(0, 1, 0, 1, 7, 8) == "ROOT_VISUAL_FAIL"
    assert first_failure(0, 0, 0, 1, 7, 8) == "FAMILY_VISUAL_FAIL"
    assert first_failure(0, 0, 0, 0, 7, 8) == "LEAF_CLASS_FAIL"
    assert first_failure(0, 0, 0, 0, 7, 7) == "TREE_CORRECT"


def test_deployable_entry_requires_top1_and_second_support_member() -> None:
    probability = np.zeros((3, 40), dtype=np.float64)
    probability[0, [24, 26, 27, 1, 2]] = [0.5, 0.2, 0.1, 0.05, 0.04]
    probability[1, [0, 24, 26, 27, 2]] = [0.5, 0.2, 0.1, 0.05, 0.04]
    probability[2, [24, 0, 2, 3, 4]] = [0.5, 0.2, 0.1, 0.05, 0.04]
    assert deployable_entry(probability).tolist() == [True, False, False]
