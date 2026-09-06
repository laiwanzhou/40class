from __future__ import annotations

import numpy as np
import pytest

from src.data.visual90_dataset import (
    prepare_roi_track,
    require_geometry,
    select_continuous_clips,
)


def test_source_gap_is_cut_before_sampling_and_equal_segments_choose_earliest():
    times = np.arange(16, dtype=float) * 10
    times[2:] += 1000
    result = select_continuous_clips(times)
    assert set(result.indices[0]) == {0, 1}
    assert result.bounds.tolist() == [[0, 2], [4, 8], [8, 12], [12, 16]]
    assert result.source_breaks.tolist() == [0, 2, 16]


def test_empty_bins_do_not_borrow_and_single_frame_repeats_real_input():
    result = select_continuous_clips(np.array([100.0]))
    assert result.valid.tolist() == [False, False, False, True]
    assert (result.indices[:3] == -1).all()
    assert (result.indices[3] == 0).all()
    assert result.unique_fraction[3] == 1 / 16


def test_counter_reset_splits_even_with_regular_timestamps():
    result = select_continuous_clips(
        np.arange(16, dtype=float),
        frame_ids=np.array([0, 1, 0, 1] + list(range(2, 14))),
        verified_counter_step=1,
    )
    assert result.bounds[0].tolist() == [0, 2]


def test_unknown_continuity_is_not_assumed_from_list_order():
    with pytest.raises(ValueError, match='continuity_unverified'):
        select_continuous_clips(np.full(16, np.nan))


@pytest.mark.parametrize('gap', [1, 3])
def test_bounded_short_roi_gaps_interpolate_and_preserve_evidence(gap):
    boxes = np.tile([10., 10., 30., 30.], (8, 1))
    boxes[2:2 + gap] = np.nan
    result = prepare_roi_track(boxes, 100, 100)
    assert result.eligible
    assert result.interpolated.sum() == gap
    np.testing.assert_allclose(result.boxes, np.tile([10., 10., 30., 30.], (8, 1)))


@pytest.mark.parametrize('indices', [[0], [7], [2, 3, 4, 5]])
def test_endpoint_or_long_roi_gap_rejects_whole_clip(indices):
    boxes = np.tile([10., 10., 30., 30.], (8, 1))
    boxes[indices] = np.nan
    result = prepare_roi_track(boxes, 100, 100)
    assert not result.eligible
    assert (result.boxes == 0).all()


def test_smoothed_boxes_still_cover_each_original_box():
    boxes = np.array([[5., 5., 25., 25.], [50., 5., 70., 25.], [5., 5., 25., 25.]])
    result = prepare_roi_track(boxes, 100, 100)
    assert result.eligible
    assert (result.boxes[:, :2] <= boxes[:, :2] + 1e-6).all()
    assert (result.boxes[:, 2:] >= boxes[:, 2:] - 1e-6).all()


def test_geometry_requires_evidence_not_just_a_boolean():
    with pytest.raises(ValueError, match='geometry_unverified'):
        require_geometry({'status': 'geometry_unverified'})
    with pytest.raises(ValueError, match='geometry_unverified'):
        require_geometry({'status': 'verified'})
