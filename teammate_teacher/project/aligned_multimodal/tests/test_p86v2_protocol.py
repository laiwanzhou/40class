from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p86v2_metrics import emission_metrics, rescue_harm
from p86v2_protocol import assert_no_forbidden_path, build_split, load_protocol, read_rows


def test_frozen_protocol_is_subject_disjoint_and_embargoed() -> None:
    protocol = load_protocol()
    rows = read_rows(PROJECT_DIR / "runs/p86_visual_pixel_cache_t16_r160_v12/rows.csv")
    development = build_split(rows, "development", protocol)
    confirmation = build_split(rows, "confirmation", protocol)

    assert len(development.holdout_indices) == 679
    assert len(confirmation.holdout_indices) == 818
    assert len(development.embargo_indices) == 444
    assert len(development.training_indices) == 1791
    assert len(confirmation.training_indices) == 1652
    assert set(development.training_subjects).isdisjoint(development.holdout_subjects)
    assert set(confirmation.training_subjects).isdisjoint(confirmation.holdout_subjects)
    assert set(development.holdout_subjects).isdisjoint(confirmation.holdout_subjects)
    assert development.sample_fingerprint == confirmation.sample_fingerprint


def test_forbidden_resource_guard() -> None:
    with pytest.raises(RuntimeError):
        assert_no_forbidden_path(Path("runs/p46_test_submission.csv"))
    assert_no_forbidden_path(Path("runs/p85_train_only_teacher/targets.npz"))


def test_emission_metrics_and_rescue_harm() -> None:
    labels = np.array([0, 1, 2, 3])
    users = ["a", "a", "b", "b"]
    baseline = np.full((4, 40), -4.0)
    baseline[np.arange(4), [0, 0, 2, 3]] = 4.0
    candidate = baseline.copy()
    candidate[1, 0] = -4.0
    candidate[1, 1] = 4.0
    candidate[2, 2] = -4.0
    candidate[2, 1] = 4.0

    metrics = emission_metrics(candidate, labels, users)
    delta = rescue_harm(candidate, baseline, labels, users)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["negative_log_likelihood"] > 0.0
    assert 0.0 <= metrics["ece_15"] <= 1.0
    assert delta["rescued"] == 1
    assert delta["harmed"] == 1
    assert delta["net"] == 0
