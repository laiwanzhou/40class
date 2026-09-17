from __future__ import annotations

import numpy as np
import torch

from aligned_multimodal.train_p25_subject_invariant_adapters import (
    SubjectInvariantAdapter,
    balanced_batches,
    class_subject_index,
    cross_subject_supervised_contrastive_loss,
    gradient_reverse,
)


def test_adapter_shape_and_l2_norm() -> None:
    model = SubjectInvariantAdapter(raw_dim=16, subject_count=3)
    logits, subject_logits, representation = model(
        torch.randn(7, 16), grl_strength=0.05, use_subject_head=True
    )
    assert logits.shape == (7, 40)
    assert subject_logits is not None and subject_logits.shape == (7, 3)
    assert representation.shape == (7, 64)
    torch.testing.assert_close(
        representation.norm(dim=1), torch.ones(7), atol=1e-6, rtol=1e-6
    )


def test_gradient_reversal_exact_strength() -> None:
    values = torch.tensor([[1.0, -2.0]], requires_grad=True)
    gradient_reverse(values, 0.05).sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, -0.05))


def test_supcon_excludes_same_subject_positive() -> None:
    representation = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    labels = torch.tensor([0, 0, 0, 1])
    subjects = torch.tensor([0, 0, 1, 1])
    loss, stats = cross_subject_supervised_contrastive_loss(
        representation, labels, subjects, temperature=0.1
    )
    assert torch.isfinite(loss)
    assert stats["valid_anchors"] == 3
    assert stats["positive_directed_pairs"] == 4


def test_balanced_sampler_produces_cross_subject_class_blocks() -> None:
    labels = np.repeat(np.arange(40), 8)
    subjects = np.tile(np.repeat(np.asarray(["a", "b", "c", "d"]), 2), 40)
    indices = np.arange(len(labels))
    lookup = class_subject_index(indices, labels, subjects)
    batches, stats = balanced_batches(
        lookup,
        steps=2,
        classes_per_batch=32,
        subject_slots_per_class=4,
        seed=20260729,
    )
    assert all(len(batch) == 128 for batch in batches)
    for batch in batches:
        for class_id in np.unique(labels[batch]):
            assert len(np.unique(subjects[batch][labels[batch] == class_id])) == 4
    assert stats["fallback_repeated_subject_slots"] == 0


def test_balanced_sampler_preserves_two_subjects_in_sparse_class() -> None:
    labels = np.repeat(np.arange(40), 2)
    subjects = np.tile(np.asarray(["a", "b"]), 40)
    lookup = class_subject_index(np.arange(len(labels)), labels, subjects)
    batches, stats = balanced_batches(
        lookup,
        steps=1,
        classes_per_batch=32,
        subject_slots_per_class=4,
        seed=20260729,
    )
    batch = batches[0]
    for class_id in np.unique(labels[batch]):
        assert set(subjects[batch][labels[batch] == class_id]) == {"a", "b"}
    assert stats["fallback_repeated_subject_slots"] == 64
