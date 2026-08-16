from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.models import TemporalClassifier
from src.train_skeleton_c1_seed_ensemble_strict_oof import ensemble_predictions, model_for


def test_e1_config_preserves_c1_contract_and_freezes_three_seeds() -> None:
    c1 = yaml.safe_load(Path("configs/experiments/skeleton_c0_c1_strict_oof.yaml").read_text(encoding="utf-8"))
    e1 = yaml.safe_load(
        Path("configs/experiments/skeleton_c1_seed_ensemble_strict_oof.yaml").read_text(encoding="utf-8")
    )
    ignored = {"output_root", "report_dir", "seeds", "scale_policy", "input_features", "device"}

    for key, value in c1.items():
        if key in ignored or key in {"seed", "representations"}:
            continue
        assert e1[key] == value
    assert e1["scale_policy"] == "per_frame"
    assert e1["input_features"] == 102
    assert e1["sequence_length"] == 64
    assert e1["seeds"] == [20260812, 20260912, 20261012]
    assert len(set(e1["seeds"])) == 3


def test_e1_uses_unchanged_c1_temporal_classifier() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/skeleton_c1_seed_ensemble_strict_oof.yaml").read_text(encoding="utf-8")
    )
    model = model_for(config)

    assert type(model) is TemporalClassifier
    assert sum(parameter.numel() for parameter in model.parameters()) == 172_776


def _member(logits: np.ndarray, seed: int) -> pd.DataFrame:
    frame = pd.DataFrame({
        "representation": "E1_member",
        "fold": [0, 0],
        "sample_id": ["sample-a", "sample-b"],
        "user_id": ["user1", "user1"],
        "label": [0, 1],
        "prediction": logits.argmax(axis=1),
        "seed": seed,
    })
    for class_id in range(logits.shape[1]):
        frame[f"logit_{class_id:02d}"] = logits[:, class_id]
    return frame


def test_ensemble_predictions_equal_mean_member_softmax_probabilities() -> None:
    logits_a = np.array([[2.0, 0.0], [0.2, 0.8]])
    logits_b = np.array([[0.0, 1.0], [1.5, 0.1]])
    logits_c = np.array([[1.0, 0.5], [0.0, 2.0]])

    result = ensemble_predictions([
        _member(logits_a, 20260812),
        _member(logits_b, 20260912),
        _member(logits_c, 20261012),
    ], num_classes=2)
    expected = torch.stack([
        torch.softmax(torch.tensor(logits_a), dim=1),
        torch.softmax(torch.tensor(logits_b), dim=1),
        torch.softmax(torch.tensor(logits_c), dim=1),
    ]).mean(dim=0).numpy()

    np.testing.assert_allclose(result[["probability_00", "probability_01"]], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result["prediction"], expected.argmax(axis=1))
    assert (result["ensemble_members"] == 3).all()


def test_ensemble_predictions_reject_mismatched_member_identity() -> None:
    first = _member(np.array([[2.0, 0.0], [0.2, 0.8]]), 20260812)
    second = _member(np.array([[0.0, 1.0], [1.5, 0.1]]), 20260912)
    second.loc[1, "label"] = 0

    with pytest.raises(ValueError, match="not exactly paired"):
        ensemble_predictions([first, second], num_classes=2)
