from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.features.imu_rf_features import (
    FEATURE_SCHEMA_VERSION,
    apply_median_imputer,
    build_feature_schema,
    extract_summary_features,
    fit_median_imputer,
)


def _sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full((4, 5, 16), np.nan, dtype=np.float32)
    valid = np.zeros((4, 5), dtype=bool)
    valid[[0, 2, 3], 0] = True
    for channel in range(16):
        values[[0, 2, 3], 0, channel] = np.asarray(
            [1.0 + channel, 3.0 + channel, 5.0 + channel], dtype=np.float32
        )
    return values, valid, np.asarray([0, 100, 200, 300], dtype=np.int64)


def test_summary_schema_is_fixed_and_excludes_leakage_fields() -> None:
    schema = build_feature_schema()
    assert schema["schema_version"] == FEATURE_SCHEMA_VERSION
    names = schema["feature_names"]
    assert len(names) == len(set(names))
    assert names[:4] == [
        "LL__acc_x_g__mean",
        "LL__acc_x_g__mean__missing",
        "LL__acc_x_g__std",
        "LL__acc_x_g__std__missing",
    ]
    forbidden = ("sample_id", "user_id", "class_name", "label", "path")
    assert not any(token in name.lower() for name in names for token in forbidden)


def test_masked_and_invalid_values_do_not_affect_summary_features() -> None:
    values, valid, timestamps = _sequence()
    first = extract_summary_features(values, valid, timestamps)
    mutated = values.copy()
    mutated[~valid] = np.float32(3.4e38)
    second = extract_summary_features(mutated, valid, timestamps)
    assert np.array_equal(np.isnan(first), np.isnan(second))
    np.testing.assert_array_equal(first[~np.isnan(first)], second[~np.isnan(second)])
    schema = build_feature_schema()["feature_names"]
    lookup = dict(zip(schema, first, strict=True))
    assert lookup["LL__acc_x_g__mean"] == pytest.approx(3.0)
    assert lookup["LL__acc_x_g__std"] == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert lookup["LL__acc_x_g__first_last_delta"] == pytest.approx(4.0)
    assert lookup["LL__acc_x_g__valid_count"] == 3.0
    assert lookup["LL__acc_x_g__valid_ratio"] == pytest.approx(0.75)
    assert lookup["LL__valid_segment_count"] == 2.0
    assert lookup["LL__longest_invalid_run"] == 1.0


def test_missing_sensor_is_explicit_and_train_only_imputer_makes_finite_matrix() -> None:
    values, valid, timestamps = _sequence()
    present = extract_summary_features(values, valid, timestamps)
    missing_values = np.full_like(values, np.nan)
    missing_valid = np.zeros_like(valid)
    missing = extract_summary_features(missing_values, missing_valid, timestamps)
    raw_train = np.stack([present, present])
    train_ids = ["train-b", "train-a"]
    imputer = fit_median_imputer(raw_train, train_ids)
    expected_sha = hashlib.sha256(b"train-a\ntrain-b\n").hexdigest()
    assert imputer["fit_split"] == "train"
    assert imputer["fit_sample_id_sha256"] == expected_sha
    transformed = apply_median_imputer(np.stack([present, missing]), imputer)
    assert transformed.shape == (2, len(build_feature_schema()["feature_names"]))
    assert np.isfinite(transformed).all()
    names = build_feature_schema()["feature_names"]
    lookup = dict(zip(names, transformed[1], strict=True))
    assert lookup["RL__acc_x_g__mean__missing"] == 1.0
    assert lookup["sample__usable_sensor_count"] == 0.0


def test_validation_cannot_change_train_fitted_medians() -> None:
    values, valid, timestamps = _sequence()
    train = extract_summary_features(values, valid, timestamps)
    validation = train.copy()
    validation[0] = 999999.0
    imputer = fit_median_imputer(np.stack([train, train]), ["a", "b"])
    original = list(imputer["medians"])
    apply_median_imputer(validation[None, :], imputer)
    assert imputer["medians"] == original


@pytest.mark.parametrize(
    ("values_shape", "mask_shape", "time_shape"),
    [((3, 5, 16), (2, 5), (3,)), ((3, 5, 15), (3, 5), (3,)), ((3, 5, 16), (3, 5), (2,))],
)
def test_summary_rejects_shape_contract_mismatches(
    values_shape: tuple[int, ...], mask_shape: tuple[int, ...], time_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        extract_summary_features(
            np.zeros(values_shape, dtype=np.float32),
            np.zeros(mask_shape, dtype=bool),
            np.zeros(time_shape, dtype=np.int64),
        )
