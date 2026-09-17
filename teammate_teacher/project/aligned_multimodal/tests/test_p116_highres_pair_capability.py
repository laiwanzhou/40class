from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p116_highres_pair_capability import (  # noqa: E402
    PAIRS,
    CandidateTokenEvidence,
    Grid,
    GridPosition,
    HierarchicalSpatialTemporalPairScorer,
    PairDataset,
    PairEntry,
    Sources,
    SpatialPairScorer,
    build_entries,
    shuffled_input_rows,
    source_selection_key,
)


def test_fixed_pairs_match_requested_high_value_boundaries() -> None:
    assert PAIRS == ((24, 26), (19, 24), (6, 37), (21, 22))


def test_candidate_attention_reads_every_unpooled_token() -> None:
    module = CandidateTokenEvidence(channels=4, width=8, dropout=0.0)
    tokens = torch.randn(2, 30, 4, requires_grad=True)
    query = torch.randn(2, 2, 8, requires_grad=True)
    state, attention = module(query, tokens, torch.randn(30, 8), True)
    assert state.shape == (2, 2, 8)
    assert attention is not None and attention.shape == (2, 2, 30)
    assert torch.all(attention > 0)
    state.sum().backward()
    assert torch.all(tokens.grad.abs().sum(dim=2) > 0)


def test_staged_scorer_keeps_candidate_query_before_each_grid() -> None:
    first = Grid(1, 2, 2, 3, 3, 4)
    second = Grid(1, 2, 2, 2, 2, 6)
    model = SpatialPairScorer((("layer2", first), ("layer3", second)), width=8)
    batch = {
        "layer2": torch.randn(2, *first.shape),
        "layer3": torch.randn(2, *second.shape),
    }
    candidates = torch.tensor([[24, 26], [6, 37]])
    logits, attention = model(batch, candidates, return_attention=True)
    assert logits.shape == (2, 2)
    assert attention["layer2"].shape == (2, 2, first.token_count)
    assert attention["layer3"].shape == (2, 2, second.token_count)
    assert not any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.modules())


def test_hierarchical_scorer_localizes_before_temporal_aggregation() -> None:
    model = HierarchicalSpatialTemporalPairScorer(width=8)
    batch = {
        "layer2": torch.randn(1, 2, 16, 3, 20, 20, 128),
        "layer3": torch.randn(1, 2, 16, 3, 10, 10, 256),
    }
    logits, attention = model(batch, torch.tensor([[24, 26]]), True)
    assert logits.shape == (1, 2)
    assert attention["layer2_spatial"].shape == (96, 2, 400)
    assert attention["layer3_spatial"].shape == (96, 2, 100)
    assert attention["frame_temporal"].shape == (1, 2, 96)

    workspace = HierarchicalSpatialTemporalPairScorer(width=8, view_indices=(2,))
    _, workspace_attention = workspace(batch, torch.tensor([[24, 26]]), True)
    assert workspace_attention["layer2_spatial"].shape == (32, 2, 400)
    assert workspace_attention["layer3_spatial"].shape == (32, 2, 100)
    assert workspace_attention["frame_temporal"].shape == (1, 2, 32)


def test_shuffle_is_within_subject_and_pair_and_zero_removes_content() -> None:
    ids = np.asarray(["a", "b", "c", "d"])
    users = np.asarray(["u1", "u1", "u2", "u2"])
    labels = np.asarray([24, 26, 24, 26])
    pooled = np.arange(16, dtype=np.float32).reshape(4, 4)
    dummy = np.zeros((4, 1), dtype=np.float32)
    sources = Sources(ids, users, labels, dummy, dummy, dummy, pooled)
    entries = [PairEntry(0, 0, 0), PairEntry(1, 0, 1), PairEntry(2, 0, 0), PairEntry(3, 0, 1)]
    shuffled = shuffled_input_rows(entries, users)
    assert shuffled.tolist() == [1, 0, 3, 2]
    dataset = PairDataset(sources, "pooled_vlit_vhpd_vwpd", entries, "zero")
    batch, candidates, target, _ = dataset[0]
    assert candidates.tolist() == [24, 26]
    assert target == 0
    assert torch.count_nonzero(batch["pooled"]) == 0


def test_pair_entries_repeat_class24_for_both_relevant_boundaries() -> None:
    labels = np.asarray([19, 21, 22, 24, 26, 37, 6])
    entries = build_entries(labels, np.ones(len(labels), dtype=bool))
    class24 = [entry for entry in entries if entry.row == 3]
    assert sorted(entry.pair_id for entry in class24) == [0, 1]


def test_source_selection_prefers_sample_specific_and_accurate_recipe() -> None:
    def result(name: str, aligned: float, gap: float, worst: float) -> dict:
        return {
            "variant": name,
            "sample_specific_gap": gap,
            "metrics": {"aligned": {"macro_pair_accuracy": aligned, "worst_subject_accuracy": worst}},
        }

    strong = result("layer2_20x20", 0.75, 0.10, 0.60)
    shortcut = result("layer3_10x10", 0.80, -0.02, 0.70)
    assert source_selection_key(strong) > source_selection_key(shortcut)
