from __future__ import annotations

import numpy as np

from src.roi.ir_primary_input_builder import IRPrimaryInputROIBuilder


def synthetic_pose(frames: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    person = np.tile(np.asarray([210.0, 40.0, 430.0, 470.0], dtype=np.float32), (frames, 1))
    keypoints = np.full((frames, 17, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((frames, 17), dtype=np.float32)
    values = {
        0: (320.0, 85.0), 5: (275.0, 155.0), 6: (365.0, 155.0),
        7: (260.0, 230.0), 8: (380.0, 230.0),
        9: (300.0, 295.0), 10: (340.0, 295.0),
        11: (290.0, 320.0), 12: (350.0, 320.0),
    }
    for index, value in values.items():
        keypoints[:, index] = value
        confidence[:, index] = 0.95
    return person, keypoints, confidence


def test_context_contains_person_and_valid_keypoints() -> None:
    person, keypoints, confidence = synthetic_pose()
    result = IRPrimaryInputROIBuilder().build(person, keypoints, confidence, 640, 480)
    for frame, context in enumerate(result.boxes[:, 0]):
        assert context[0] <= person[frame, 0]
        assert context[1] <= person[frame, 1]
        assert context[2] >= person[frame, 2]
        assert context[3] >= person[frame, 3]
        valid = confidence[frame] >= 0.25
        assert np.all(keypoints[frame, valid, 0] >= context[0])
        assert np.all(keypoints[frame, valid, 0] <= context[2])
        assert np.all(keypoints[frame, valid, 1] >= context[1])
        assert np.all(keypoints[frame, valid, 1] <= context[3])


def test_close_hands_create_adaptive_view_and_suppress_duplicate() -> None:
    person, keypoints, confidence = synthetic_pose()
    result = IRPrimaryInputROIBuilder(duplicate_iou_threshold=0.0).build(
        person, keypoints, confidence, 640, 480,
    )
    assert result.valid_mask[:, 3].all()
    assert not result.valid_mask[:, 2].any()
    assert set(result.sources[:, 3]) <= {
        "hand_head_union", "two_hand_table_context", "two_hand_relation", "single_hand_context",
    }
