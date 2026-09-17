from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p103_rich_candidate_model import (
    P103RichCandidateTeacher,
    RichCandidateConfig,
    token_topology,
)
from train_p103_b2_rich_visual_oof import (
    build_hard_population,
    candidate_context,
    candidate_list_loss,
)


def synthetic_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    return {
        "vmae_temporal": torch.randn(batch_size, 2, 3, 8, 768, generator=generator),
        "iv2_temporal": torch.randn(batch_size, 2, 3, 8, 768, generator=generator),
        "vmae_pooled": torch.randn(batch_size, 2, 3, 768, generator=generator),
        "iv2_pooled": torch.randn(batch_size, 2, 3, 768, generator=generator),
        "vmae_action": torch.randn(batch_size, 2, 3, 710, generator=generator),
        "iv2_action": torch.randn(batch_size, 2, 3, 400, generator=generator),
        "candidate_ids": torch.tensor([[0, 1, 2, -1]] * batch_size),
        "a_context": torch.zeros(batch_size, 4, 7),
        "visual_scale": torch.ones(batch_size),
    }


def test_rich_token_topology_preserves_encoder_view_time_and_type() -> None:
    topology = token_topology()
    assert {key: value.shape for key, value in topology.items()} == {
        "encoder": (120,),
        "window": (120,),
        "view": (120,),
        "time": (120,),
        "token_type": (120,),
    }
    assert np.sum(topology["token_type"] == 0) == 96
    assert np.sum(topology["token_type"] == 1) == 12
    assert np.sum(topology["token_type"] == 2) == 12
    assert np.sum(topology["view"] == 2) == 40


def test_candidate_query_changes_pre_score_evidence_attention() -> None:
    torch.manual_seed(8)
    model = P103RichCandidateTeacher(RichCandidateConfig(dropout=0.0)).eval()
    output = model(synthetic_batch(1), return_attention=True)
    assert output["candidate_scores"].shape == (1, 4)
    assert output["token_attention"].shape == (1, 4, 4, 120)
    assert output["attention_groups"].shape == (1, 4, 18)
    assert output["candidate_scores"][0, 3] < -1000
    first = output["token_attention"][0, :, 0]
    second = output["token_attention"][0, :, 1]
    assert not torch.allclose(first, second)
    assert not hasattr(model, "classifier")


def test_zero_visual_removes_sample_specific_content() -> None:
    torch.manual_seed(9)
    model = P103RichCandidateTeacher(RichCandidateConfig(dropout=0.0)).eval()
    batch = synthetic_batch(2)
    batch["visual_scale"] = torch.zeros(2)
    output = model(batch)
    assert torch.allclose(
        output["candidate_scores"][0], output["candidate_scores"][1], atol=1e-6
    )


def test_candidate_list_loss_compares_true_to_valid_hard_negatives() -> None:
    scores = torch.tensor([[0.2, 0.1, -0.3, -1e4]], requires_grad=True)
    candidates = torch.tensor([[4, 5, 6, -1]])
    target = torch.tensor([1])
    loss, audit = candidate_list_loss(
        scores, candidates, target, torch.tensor([4.0])
    )
    loss.backward()
    assert loss.item() > 0
    assert audit["pair"] > 0
    assert scores.grad is not None
    assert scores.grad[0, 1] < 0


def test_hard_population_prioritizes_errors_and_source_only_protection() -> None:
    rows = 8
    probability = np.full((rows, 40), 1e-6, dtype=np.float64)
    labels = np.arange(rows) % 2
    for row in range(rows):
        if row in (0, 1):
            probability[row, 1 - labels[row]] = 0.55
            probability[row, labels[row]] = 0.44
        else:
            probability[row, labels[row]] = 0.55 + 0.04 * row
            probability[row, 1 - labels[row]] = 0.44 - 0.04 * row
    probability /= probability.sum(axis=1, keepdims=True)
    candidates = np.full((rows, 8), -1, dtype=np.int64)
    candidates[:, :2] = np.asarray([[0, 1]])
    selected, weight, audit = build_hard_population(
        np.asarray([f"s{row}" for row in range(rows)]),
        labels,
        np.asarray([True] * 6 + [False] * 2),
        probability,
        candidates,
    )
    assert {0, 1} <= set(selected.tolist())
    assert weight[0] == weight[1] == 4.0
    assert audit["core_hard_rows"] == 2
    assert audit["selected_rows"] <= 6


def test_a_prior_is_context_not_fixed_score_addition() -> None:
    probability = np.full((1, 40), 1e-6)
    probability[0, 3] = 0.7
    probability[0, 5] = 0.2
    probability /= probability.sum(axis=1, keepdims=True)
    candidates = np.asarray([[3, 5, -1]])
    context = candidate_context(probability, candidates)
    assert context.shape == (1, 3, 7)
    assert context[0, 0, 1] > context[0, 1, 1]
    assert context[0, 0, 4] == 1.0
    assert np.all(context[0, 2] == 0.0)
