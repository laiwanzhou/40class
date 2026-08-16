from __future__ import annotations

import numpy as np
import pytest

from scripts.report_x3d_s_train12_val2_dev import (
    validate_prediction_population,
)


def test_prediction_population_requires_exact_validation_users_and_count() -> None:
    users = np.asarray(["user21"] * 160 + ["user22"] * 164)
    sample_ids = np.asarray([f"trial_{index}" for index in range(324)])

    validate_prediction_population(
        users, sample_ids,
        expected_validation_users=("user21", "user22"),
        expected_validation_trials=324,
    )

    assert tuple(sorted(set(users.tolist()))) == ("user21", "user22")


def test_prediction_population_accepts_user6_user7_profile() -> None:
    users = np.asarray(["user6"] * 190 + ["user7"] * 195)
    sample_ids = np.asarray([f"trial_{index}" for index in range(385)])

    validate_prediction_population(
        users, sample_ids,
        expected_validation_users=("user6", "user7"),
        expected_validation_trials=385,
    )


def test_prediction_population_rejects_wrong_user() -> None:
    users = np.asarray(["user21"] * 160 + ["user23"] * 164)
    sample_ids = np.asarray([f"trial_{index}" for index in range(324)])

    with pytest.raises(ValueError, match="exact frozen validation users"):
        validate_prediction_population(
            users, sample_ids,
            expected_validation_users=("user21", "user22"),
            expected_validation_trials=324,
        )


def test_prediction_population_rejects_duplicate_or_wrong_count() -> None:
    users = np.asarray(["user21"] * 160 + ["user22"] * 163)
    sample_ids = np.asarray([f"trial_{index}" for index in range(323)])

    with pytest.raises(ValueError, match="324"):
        validate_prediction_population(
            users, sample_ids,
            expected_validation_users=("user21", "user22"),
            expected_validation_trials=324,
        )

    users = np.asarray(["user21"] * 160 + ["user22"] * 164)
    sample_ids = np.asarray(["duplicate"] * 324)
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_population(
            users, sample_ids,
            expected_validation_users=("user21", "user22"),
            expected_validation_trials=324,
        )
