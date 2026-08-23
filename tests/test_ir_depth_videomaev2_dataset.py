from __future__ import annotations

import numpy as np

from src.data.ir_depth_videomaev2_dataset import uniform_trial_indices


def test_fixed_sixteen_frame_sampler_spans_the_complete_trial() -> None:
    indices = uniform_trial_indices(101, 16)

    assert indices.shape == (16,)
    assert indices[0] == 0
    assert indices[-1] == 100
    assert np.all(indices[1:] >= indices[:-1])
    assert len(np.unique(indices)) == 16


def test_short_trial_repeats_frames_without_changing_fixed_budget() -> None:
    indices = uniform_trial_indices(5, 16)

    assert indices.shape == (16,)
    assert indices[0] == 0
    assert indices[-1] == 4
    assert set(indices) == set(range(5))
