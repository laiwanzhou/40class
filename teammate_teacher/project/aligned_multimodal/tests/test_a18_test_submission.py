from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ALIGNED = HERE.parent
if str(ALIGNED) not in sys.path:
    sys.path.insert(0, str(ALIGNED))

from predict_a18_test import (  # noqa: E402
    audit_submission,
    mean_filled_alignment,
    official_id,
    write_csv,
)


def test_mean_filled_alignment_marks_missing_rows_without_fake_evidence() -> None:
    source_ids = np.asarray(["b", "a"])
    values = np.asarray(
        [
            [[20.0, 21.0], [22.0, 23.0]],
            [[10.0, 11.0], [12.0, 13.0]],
        ],
        dtype=np.float32,
    )
    target_ids = np.asarray(["a", "missing", "b"])
    mean = np.asarray([1.5, 2.5], dtype=np.float32)
    aligned, available = mean_filled_alignment(
        source_ids, values, target_ids, mean
    )
    assert available.tolist() == [True, False, True]
    assert np.array_equal(aligned[0], values[1])
    assert np.array_equal(aligned[2], values[0])
    assert np.array_equal(aligned[1], np.asarray([[1.5, 2.5], [1.5, 2.5]]))


def test_official_id_accepts_kaggle_path_variants() -> None:
    assert official_id("small_model_track_test/SM_test_0001/") == "SM_test_0001"
    assert official_id("small_model_track_test\\SM_test_0405\\") == "SM_test_0405"


def test_submission_audit_requires_exact_order_and_range(tmp_path: Path) -> None:
    official = [
        {"path": f"small_model_track_test/SM_test_{index:04d}/", "prediction": ""}
        for index in range(1, 406)
    ]
    path = tmp_path / "submission.csv"
    write_csv(
        path,
        [
            {"path": row["path"], "prediction": index % 40}
            for index, row in enumerate(official)
        ],
    )
    result = audit_submission(path, official)
    assert all(result["checks"].values())
