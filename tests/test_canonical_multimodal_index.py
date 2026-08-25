from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.canonical_multimodal_index import (
    build_canonical_trials,
    normalized_segment_bounds,
)
from src.data.multimodal_segment_contract import SegmentBatch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata/manifest.csv"
SPLIT = ROOT / "metadata/splits/train12_val2_user6_user7_development.json"
DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")


def test_normalized_segment_bounds_cover_complete_sequence() -> None:
    bounds = normalized_segment_bounds(17, segments=8)

    assert bounds.tolist() == [
        [0, 2], [2, 4], [4, 6], [6, 8],
        [8, 11], [11, 13], [13, 15], [15, 17],
    ]
    assert bounds[0, 0] == 0 and bounds[-1, 1] == 17


def test_canonical_index_retains_thermal_only_rows_as_core_unavailable() -> None:
    rows = build_canonical_trials(
        MANIFEST, SPLIT, DATA_ROOT, partition="validation"
    )

    assert len(rows) == 388
    assert sum(row.core_available for row in rows) == 385
    assert sum(not row.core_available for row in rows) == 3
    assert {row.class_id for row in rows} == set(range(40))
    assert {row.user_id for row in rows} == {"user6", "user7"}


def test_real_train_canonical_index_preserves_all_2039_rows() -> None:
    rows = build_canonical_trials(MANIFEST, SPLIT, DATA_ROOT, partition="train")

    assert len(rows) == 2039
    assert len({row.sample_id for row in rows}) == 2039
    assert sum(row.core_available for row in rows) == 1957
    assert sum(not row.core_available for row in rows) == 82


def test_segment_batch_rejects_unmasked_nonfinite_token() -> None:
    batch = SegmentBatch(
        tokens=torch.zeros(1, 8, 2, 4),
        token_mask=torch.ones(1, 8, 2, dtype=torch.bool),
        quality=torch.zeros(1, 8, 2, 3),
        quality_mask=torch.ones(1, 8, 2, 3, dtype=torch.bool),
    )
    batch.tokens[0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="non-finite"):
        batch.validate(batch=1, segments=8, streams=2, dim=4)


def test_segment_batch_allows_zeroed_masked_token() -> None:
    mask = torch.ones(1, 8, 2, dtype=torch.bool)
    mask[:, 3, 1] = False
    tokens = torch.ones(1, 8, 2, 4)
    tokens[:, 3, 1] = 0
    batch = SegmentBatch(
        tokens=tokens,
        token_mask=mask,
        quality=torch.zeros(1, 8, 2, 3),
        quality_mask=torch.zeros(1, 8, 2, 3, dtype=torch.bool),
    )

    batch.validate(batch=1, segments=8, streams=2, dim=4)
