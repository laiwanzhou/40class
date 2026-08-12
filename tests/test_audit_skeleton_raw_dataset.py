from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_skeleton_raw_dataset.py"
SPEC = importlib.util.spec_from_file_location("audit_skeleton_raw_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_frame_pattern_accepts_both_observed_filename_schemes() -> None:
    plain = AUDIT.FRAME_RE.fullmatch("Color_42")
    timestamped = AUDIT.FRAME_RE.fullmatch("Color_2026-01-02_03-04-05.600_42")

    assert plain is not None
    assert plain.group("timestamp") is None
    assert plain.group("frame") == "42"
    assert timestamped is not None
    assert timestamped.group("timestamp") == "2026-01-02_03-04-05.600"
    assert timestamped.group("frame") == "42"


def test_select_person_uses_highest_mean_score() -> None:
    low = {"keypoints": np.zeros((17, 3)).tolist(), "keypoint_scores": [0.2] * 17}
    high = {"keypoints": np.ones((17, 3)).tolist(), "keypoint_scores": [0.9] * 17}

    selected, count, error = AUDIT.select_person([low, high])

    assert selected is high
    assert count == 2
    assert error is None


def test_select_person_tie_is_stable_and_returns_first_person() -> None:
    first = {"keypoints": np.zeros((17, 3)).tolist(), "keypoint_scores": [1.0] * 17}
    second = {"keypoints": np.ones((17, 3)).tolist(), "keypoint_scores": [1.0] * 17}

    selected, count, error = AUDIT.select_person([first, second])

    assert selected is first
    assert count == 2
    assert error is None


def test_describe_reports_expected_quantiles() -> None:
    stats = AUDIT.describe(np.array([1.0, 2.0, 3.0]))

    assert stats["min"] == 1.0
    assert stats["median"] == 2.0
    assert stats["mean"] == 2.0
    assert stats["max"] == 3.0


def test_consecutive_runs_deduplicates_and_splits_gaps() -> None:
    assert AUDIT.consecutive_runs([8, 7, 7, 3, 2, 10]) == [[2, 3], [7, 8], [10]]
