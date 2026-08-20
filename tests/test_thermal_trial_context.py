from __future__ import annotations

from src.roi.thermal_trial_context import build_trial_context


def test_union_context_is_expanded_squared_and_clamped() -> None:
    context = build_trial_context(
        sample_id="train__c00__user1__1-1-1",
        frame_size=(320, 240),
        frame_count=101,
        detections_by_index={
            0: [(50, 30, 100, 180, 0.8), (0, 0, 10, 10, 0.4)],
            14: [(80, 40, 150, 190, 0.7)],
        },
    )

    assert context.available
    assert context.probe_indices == (0, 14, 29, 43, 57, 71, 86, 100)
    assert len(context.accepted_detections) == 2
    assert context.bbox_xyxy is not None
    x1, y1, x2, y2 = context.bbox_xyxy
    assert x2 - x1 == y2 - y1
    assert x2 - x1 >= 112
    assert 0 <= x1 < x2 <= 320
    assert 0 <= y1 < y2 <= 240
    assert context.fallback_reason is None


def test_highest_confidence_person_is_selected_per_probe() -> None:
    context = build_trial_context(
        sample_id="sample",
        frame_size=(320, 240),
        frame_count=2,
        detections_by_index={
            0: [(1, 1, 20, 30, 0.3), (10, 10, 100, 200, 0.9)],
            1: [(20, 20, 110, 210, 0.8)],
        },
    )

    assert [row[-1] for row in context.accepted_detections] == [0.9, 0.8]


def test_non_singleton_requires_two_hits() -> None:
    context = build_trial_context(
        sample_id="sample",
        frame_size=(320, 240),
        frame_count=20,
        detections_by_index={0: [(10, 10, 100, 200, 0.9)]},
    )

    assert not context.available
    assert context.bbox_xyxy is None
    assert context.fallback_reason == "insufficient_detection_hits"


def test_singleton_accepts_one_hit_and_rejects_no_hit() -> None:
    accepted = build_trial_context(
        sample_id="single",
        frame_size=(320, 240),
        frame_count=1,
        detections_by_index={0: [(10, 10, 100, 200, 0.9)]},
    )
    rejected = build_trial_context(
        sample_id="single",
        frame_size=(320, 240),
        frame_count=1,
        detections_by_index={},
    )

    assert accepted.available
    assert not rejected.available
    assert rejected.fallback_reason == "no_detection"


def test_invalid_and_low_confidence_boxes_are_rejected() -> None:
    context = build_trial_context(
        sample_id="sample",
        frame_size=(320, 240),
        frame_count=10,
        detections_by_index={
            0: [(10, 10, 5, 20, 0.9)],
            1: [(10, 10, 100, 200, 0.2)],
        },
    )

    assert not context.available
    assert context.accepted_detections == ()
    assert context.fallback_reason == "no_detection"
