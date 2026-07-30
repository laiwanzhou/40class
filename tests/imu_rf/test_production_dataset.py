from __future__ import annotations

import pandas as pd
import pytest

from src.training.imu_rf_production import validate_production_split


def _formal_rows() -> pd.DataFrame:
    rows = []
    for index in range(2757):
        rows.append(
            {
                "sample_id": f"sample-{index:04d}",
                "user_id": f"user-{index % 9}",
                "label_index": index % 40,
                "selected_for_run": True,
                "split": "train" if index < 2184 else "validation",
                "stage2_npz_relpath": f"class/action-{index}/imu_stage2.npz",
                "status": "success",
            }
        )
    return pd.DataFrame(rows)


def test_production_split_is_exact_disjoint_selected_2757_union() -> None:
    frame = _formal_rows()
    selected_ids = set(frame["sample_id"])
    result = validate_production_split(frame, selected_ids=selected_ids)
    assert result["train_count"] == 2184
    assert result["validation_count"] == 573
    assert result["union_count"] == 2757
    assert result["overlap_count"] == 0
    assert result["excluded_samples"] == 0
    assert len(result["class_counts"]) == 40


def test_production_split_rejects_duplicate_or_nonselected_sample() -> None:
    frame = _formal_rows()
    frame.loc[2184, "sample_id"] = frame.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_production_split(frame, selected_ids=set(_formal_rows()["sample_id"]))
    frame = _formal_rows()
    selected = set(frame["sample_id"])
    selected.remove("sample-0000")
    with pytest.raises(ValueError, match="selected2757"):
        validate_production_split(frame, selected_ids=selected)


def test_production_split_rejects_test_unlabeled_or_wrong_counts() -> None:
    frame = _formal_rows()
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="train and validation"):
        validate_production_split(frame, selected_ids=set(frame["sample_id"]))
    frame = _formal_rows().iloc[:-1].copy()
    with pytest.raises(ValueError, match="count"):
        validate_production_split(frame, selected_ids=set(frame["sample_id"]))
