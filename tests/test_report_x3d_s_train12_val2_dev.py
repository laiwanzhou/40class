from __future__ import annotations

import numpy as np
import pytest

from scripts.report_x3d_s_train12_val2_dev import (
    EXPECTED_VALIDATION_USERS,
    validate_prediction_population,
)


def test_prediction_population_requires_exact_validation_users_and_count() -> None:
    users = np.asarray(["user21"] * 160 + ["user22"] * 164)
    sample_ids = np.asarray([f"trial_{index}" for index in range(324)])

    validate_prediction_population(users, sample_ids)

    assert tuple(sorted(set(users.tolist()))) == EXPECTED_VALIDATION_USERS


def test_prediction_population_rejects_wrong_user() -> None:
    users = np.asarray(["user21"] * 160 + ["user23"] * 164)
    sample_ids = np.asarray([f"trial_{index}" for index in range(324)])

    with pytest.raises(ValueError, match="exact frozen validation users"):
        validate_prediction_population(users, sample_ids)


def test_prediction_population_rejects_duplicate_or_wrong_count() -> None:
    users = np.asarray(["user21"] * 160 + ["user22"] * 163)
    sample_ids = np.asarray([f"trial_{index}" for index in range(323)])

    with pytest.raises(ValueError, match="324"):
        validate_prediction_population(users, sample_ids)

    users = np.asarray(["user21"] * 160 + ["user22"] * 164)
    sample_ids = np.asarray(["duplicate"] * 324)
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_population(users, sample_ids)
