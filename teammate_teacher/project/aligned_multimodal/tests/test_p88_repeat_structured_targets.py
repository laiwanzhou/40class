from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audit_p87_sequence_decoder import DecoderConfig
from p88_aligned_repeat_holdout import AlignedRepeatConfig
from p88_build_repeat_structured_targets import repeat_consensus_targets


def test_repeat_consensus_targets_aligns_a_missing_middle_action() -> None:
    # Two takes: [0, 1, 2] and [0, 2].  The timestamps create two short sessions
    # inside one medium recording block.
    grouping = np.full((5, 4), 0.01, dtype=np.float64)
    grouping[[0, 1, 2], [0, 1, 2]] = 0.97
    grouping[[3, 4], [0, 2]] = 0.97
    source = grouping.copy()
    source[3] = [0.70, 0.01, 0.01, 0.28]
    source[4] = [0.01, 0.01, 0.70, 0.28]
    metadata = SimpleNamespace(
        dates=np.asarray(["d"] * 5),
        starts=np.asarray([0.0, 1.0, 2.0, 50.0, 51.0]),
    )
    decoder = DecoderConfig(10.0, 0.0, 1.0, 10)
    config = AlignedRepeatConfig(90.0, 0.5, 0.8, 0.3, 0.2, 3)

    adjusted, audit = repeat_consensus_targets(
        source,
        grouping,
        source.argmax(axis=1),
        np.arange(5),
        metadata,
        decoder,
        config,
    )

    np.testing.assert_allclose(adjusted.sum(axis=1), 1.0)
    assert audit["grouped_sessions"] == 2
    assert audit["aligned_pairs"] == 2
    assert audit["changed_target_rows"] == 4
