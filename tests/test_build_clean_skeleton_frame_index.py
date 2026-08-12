from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_clean_skeleton_frame_index.py"
SPEC = importlib.util.spec_from_file_location("build_clean_skeleton_frame_index", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
INDEXER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEXER)


def test_parse_frame_supports_timestamped_and_plain_names() -> None:
    timestamped = Path("Color_2025-05-31_17-45-41.716_00000048.json")
    plain = Path("Color_00000048.json")

    assert INDEXER.parse_frame(timestamped) == (
        48, "2025-05-31_17-45-41.716", "2025-05-31_17-45-41.716_00000048"
    )
    assert INDEXER.parse_frame(plain) == (48, None, None)


def test_canonical_file_prefers_timestamped_duplicate() -> None:
    plain = Path("Color_00000048.json")
    timestamped = Path("Color_2025-05-31_17-45-41.716_00000048.json")

    assert INDEXER.canonical_file([plain, timestamped]) == timestamped


def test_retained_segments_break_at_ambiguous_frame() -> None:
    frame = pd.DataFrame({
        "sample_id": ["a", "a", "a", "a"],
        "frame_id": [1, 2, 3, 4],
        "use_for_frame_training": [True, True, False, True],
    })

    segments = INDEXER.assign_retained_segments(frame)

    assert segments.iloc[0] == 0
    assert segments.iloc[1] == 0
    assert pd.isna(segments.iloc[2])
    assert segments.iloc[3] == 1
