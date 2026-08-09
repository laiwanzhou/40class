from __future__ import annotations

import pytest
import torch

from src.models.expert_contract import (
    ExpertBatchResult,
    ExpertOutput,
    align_expert_batch,
    calibrated_probability_mixture,
)


def result(sample_ids: tuple[str, ...], class_hash: str = "map") -> ExpertBatchResult:
    rows = len(sample_ids)
    return ExpertBatchResult(
        sample_ids=sample_ids,
        class_map_hash=class_hash,
        output=ExpertOutput(
            main_logits=torch.randn(rows, 40),
            embedding=torch.randn(rows, 8),
            quality=torch.ones(rows, 3),
            quality_mask=torch.ones(rows, 3, dtype=torch.bool),
            availability=torch.ones(rows, 1, dtype=torch.bool),
        ),
    )


def test_alignment_uses_sample_ids_not_array_order() -> None:
    reference = result(("a", "b", "c"))
    other = result(("c", "a", "b"))
    assert align_expert_batch(reference, other).tolist() == [1, 2, 0]


def test_alignment_rejects_missing_samples_and_class_map_changes() -> None:
    with pytest.raises(ValueError, match="sample sets differ"):
        align_expert_batch(result(("a", "b")), result(("a", "c")))
    with pytest.raises(ValueError, match="class maps differ"):
        align_expert_batch(result(("a",), "one"), result(("a",), "two"))


def test_alpha_zero_exactly_recovers_visual_probabilities() -> None:
    visual = torch.randn(4, 40)
    sensor = torch.randn(4, 40)
    fused = calibrated_probability_mixture(visual, sensor, 0.0)
    torch.testing.assert_close(fused, torch.softmax(visual, dim=-1), atol=0.0, rtol=0.0)


def test_validation_checks_every_fixed_output_row() -> None:
    malformed = ExpertBatchResult(
        sample_ids=("a", "b"),
        class_map_hash="map",
        output=ExpertOutput(
            main_logits=torch.randn(2, 40),
            embedding=torch.randn(1, 8),
            quality=torch.ones(2, 3),
            quality_mask=torch.ones(2, 3, dtype=torch.bool),
            availability=torch.ones(2, 1, dtype=torch.bool),
        ),
    )
    with pytest.raises(ValueError, match="embedding rows"):
        malformed.validate()
