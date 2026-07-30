from __future__ import annotations

import numpy as np
import pytest

from src.training.imu_rf_production import (
    apply_production_imputer,
    fit_production_imputer,
)


def test_production_imputer_fits_all_labeled_raw_rows_not_train_only() -> None:
    raw = np.asarray([[1.0, np.nan], [3.0, 8.0], [100.0, 10.0]], dtype=np.float64)
    imputer = fit_production_imputer(raw, ["a", "b", "c"], ["first", "second"])
    assert imputer["fit_scope"] == "all_labeled"
    assert imputer["fit_sample_count"] == 3
    assert imputer["feature_count"] == 2
    assert imputer["medians"] == [3.0, 9.0]
    transformed = apply_production_imputer(raw, imputer, ["first", "second"])
    assert np.isfinite(transformed).all()
    assert transformed[0, 1] == 9.0


def test_production_imputer_rejects_train_only_or_feature_order_mismatch() -> None:
    raw = np.asarray([[1.0, np.nan], [3.0, 8.0]], dtype=np.float64)
    imputer = fit_production_imputer(raw, ["a", "b"], ["first", "second"])
    changed = dict(imputer)
    changed["fit_scope"] = "train"
    with pytest.raises(ValueError, match="scope"):
        apply_production_imputer(raw, changed, ["first", "second"])
    with pytest.raises(ValueError, match="feature names"):
        apply_production_imputer(raw, imputer, ["second", "first"])


def test_production_imputer_uses_zero_only_for_all_missing_feature() -> None:
    raw = np.asarray([[np.nan, 2.0], [np.nan, 4.0]], dtype=np.float64)
    imputer = fit_production_imputer(raw, ["a", "b"], ["missing", "observed"])
    assert imputer["medians"] == [0.0, 3.0]
    assert imputer["all_missing_features"] == ["missing"]

