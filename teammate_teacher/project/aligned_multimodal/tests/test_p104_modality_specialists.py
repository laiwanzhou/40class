from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from analyze_p104_single_modalities import pair_complementarity
from analyze_p104_final_audit import finish_subjects
from analyze_p104_thermal_extension import source_metric_key
from p104_modality_data import _thermal
from train_p104_pair_specialists_oof import selected_plans
from train_p104_modality_specialists_oof import (
    Projection,
    cyclic_shuffle_source,
    deployable_trigger,
    intervention_metrics,
)


def _source_predictions(correct_rows: set[int]) -> dict[tuple[int, str, int], tuple[int, int]]:
    return {
        (0, "1__2", row): (
            row % 2 + 1,
            row % 2 + 1 if row in correct_rows else 2 - row % 2,
        )
        for row in range(10)
    }


def test_pair_complementarity_requires_two_unique_corrections_each_and_three_points() -> None:
    first = _source_predictions({0, 1, 2, 3, 4, 5})
    second = _source_predictions({0, 1, 2, 3, 6, 7})
    result = pair_complementarity(first, second, 0, "1__2", best_single_accuracy=0.6)

    assert result["unique_first"] == 2
    assert result["unique_second"] == 2
    assert result["oracle_union_accuracy"] == 0.8
    assert result["eligible"] is True


def test_pair_complementarity_rejects_single_unique_correction() -> None:
    first = _source_predictions({0, 1, 2, 3, 4, 5})
    second = _source_predictions({0, 1, 2, 3, 6})
    result = pair_complementarity(first, second, 0, "1__2", best_single_accuracy=0.6)

    assert result["unique_second"] == 1
    assert result["eligible"] is False


def test_selected_pair_plans_do_not_create_unselected_work() -> None:
    archive = {
        "plans": [
            {"outer_fold": 1, "family_key": "3__5", "selected_pair": None},
            {
                "outer_fold": 0,
                "family_key": "21__22",
                "selected_pair": {"modalities": ["LocalV", "Skeleton"]},
            },
            {
                "outer_fold": 2,
                "family_key": "8__9",
                "selected_pair": {"modalities": ["LocalV", "Depth"]},
            },
        ]
    }

    assert [value["family_key"] for value in selected_plans(archive, [0, 1], False)] == [
        "21__22"
    ]
    assert len(selected_plans(archive, [0, 2], True)) == 1


def test_finish_subjects_counts_stability_and_worst_net() -> None:
    result = finish_subjects(
        {
            "user_a": {"rows": 5, "a_correct": 2, "specialist_correct": 4},
            "user_b": {"rows": 4, "a_correct": 4, "specialist_correct": 3},
            "user_c": {"rows": 2, "a_correct": 2, "specialist_correct": 2},
        }
    )

    assert result["positive_subjects"] == 1
    assert result["neutral_subjects"] == 1
    assert result["negative_subjects"] == 1
    assert result["worst_subject"]["subject"] == "user_b"


def test_thermal_loader_selects_h1_rows_without_requesting_historical_labels(tmp_path: Path) -> None:
    path = tmp_path / "thermal.npz"
    np.savez(
        path,
        sample_ids=np.asarray(["b", "h3", "a"]),
        users=np.asarray(["user2", "user3", "user1"]),
        features=np.arange(18, dtype=np.float32).reshape(3, 2, 3),
        action_logits=np.arange(12, dtype=np.float32).reshape(3, 2, 2),
        modality_available=np.asarray([1, 0, 1], dtype=np.uint8),
        modality=np.asarray("thermal"),
        labels=np.asarray(["must_not_be_loaded"], dtype=object),
        fold_id=np.asarray([99], dtype=object),
    )
    data = SimpleNamespace(
        sample_ids=np.asarray(["a", "b"]),
        users=np.asarray(["user1", "user2"]),
    )

    result = _thermal(data, path)

    assert result.aligned.shape == (2, 10)
    assert result.available.tolist() == [True, True]
    assert result.audit["historical_label_field_requested"] is False
    assert result.audit["historical_fold_field_requested"] is False
    assert result.audit["h3_rows_selected"] == 0


def test_thermal_extension_source_selection_prioritizes_balanced_accuracy() -> None:
    higher_balanced = {
        "balanced_accuracy": 0.71,
        "macro_f1": 0.60,
        "accuracy": 0.60,
    }
    higher_accuracy = {
        "balanced_accuracy": 0.70,
        "macro_f1": 0.90,
        "accuracy": 0.90,
    }

    assert source_metric_key(higher_balanced) > source_metric_key(higher_accuracy)


def test_cyclic_shuffle_is_stable_and_never_crosses_subject() -> None:
    ids = np.asarray(["b", "a", "d", "c", "x"])
    users = np.asarray(["u1", "u1", "u2", "u2", "other"])
    selected = np.asarray([True, True, True, True, False])

    mapping = cyclic_shuffle_source(ids, users, selected)

    assert mapping.tolist() == [1, 0, 3, 2, 4]
    assert np.array_equal(users[mapping[selected]], users[selected])


def test_deployable_trigger_uses_only_a_top3_membership() -> None:
    probability = np.asarray(
        [
            [0.60, 0.30, 0.10, 0.00],
            [0.60, 0.10, 0.30, 0.00],
            [0.10, 0.60, 0.30, 0.00],
            [0.10, 0.20, 0.60, 0.10],
        ]
    )

    trigger = deployable_trigger(probability, [0, 1])

    assert trigger.tolist() == [True, True, True, False]


def test_projection_zero_and_intervention_metrics() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(30, 12)).astype(np.float32)
    projection = Projection.fit(values, np.arange(24), components=5)

    zero = projection.zero(3)
    audit = intervention_metrics(
        np.asarray([0, 1, 1, 0]),
        np.asarray([1, 1, 0, 0]),
        np.asarray([0, 0, 1, 1]),
    )

    assert zero.shape == (3, 5)
    assert np.isfinite(zero).all()
    assert audit["rescue"] == 2
    assert audit["harm"] == 2
    assert audit["net"] == 0
