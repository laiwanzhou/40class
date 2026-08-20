from __future__ import annotations

import numpy as np

from scripts.fit_thermal_v2_normalization import (
    ChannelMoments,
    normalization_frame_indices,
)


def test_normalization_uses_three_window_centers() -> None:
    assert normalization_frame_indices(101) == (25, 50, 75)
    assert normalization_frame_indices(1) == (0,)


def test_channel_moments_compute_population_mean_and_std() -> None:
    moments = ChannelMoments()
    moments.update(np.asarray([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]))

    mean, std = moments.finalize()

    assert np.allclose(mean, [0.5, 0.0, 0.5])
    assert np.allclose(std, [0.5, 0.0, 0.5])
    assert moments.pixel_count == 2
