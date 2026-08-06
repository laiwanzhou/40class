from __future__ import annotations

import torch

from src.data.cross_user_batch_sampler import CrossUserActionBatchSampler
from src.models.cross_user_supcon import cross_user_supcon_loss


class FakeDataset:
    def __init__(self) -> None:
        self.samples = [
            {"label": label, "user_id": user, "sample_id": f"{label}-{user}-{trial}"}
            for label in range(4)
            for user in ("u1", "u2", "u3")
            for trial in range(2)
        ]

    def __len__(self) -> int:
        return len(self.samples)


def test_sampler_guarantees_cross_user_positives_and_same_user_negatives() -> None:
    dataset = FakeDataset()
    sampler = CrossUserActionBatchSampler.from_dataset(dataset, batch_size=4, seed=17)
    audit = sampler.audit(epochs=3)
    assert audit["cross_user_positive_anchor_rate"] == 1.0
    assert audit["same_user_negative_anchor_rate"] == 1.0
    assert audit["class_anchor_count_max"] - audit["class_anchor_count_min"] <= 3
    for batch in sampler:
        assert len(batch) == len(set(batch)) == 4
        labels = [dataset.samples[index]["label"] for index in batch]
        users = [dataset.samples[index]["user_id"] for index in batch]
        assert len(set(labels)) == 2
        assert len(set(users)) == 2
        assert all(labels.count(label) == 2 for label in set(labels))
        assert all(users.count(user) == 2 for user in set(users))


def test_cross_user_supcon_loss_is_finite_and_differentiable() -> None:
    projection = torch.randn(4, 8, requires_grad=True)
    projection = torch.nn.functional.normalize(projection, dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    users = torch.tensor([0, 1, 0, 1])
    loss = cross_user_supcon_loss(projection, labels, users, temperature=0.1, same_user_negative_weight=2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert projection.grad_fn is not None


def test_cross_user_supcon_rejects_missing_cross_user_positive() -> None:
    projection = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    users = torch.tensor([0, 0, 1, 1])
    try:
        cross_user_supcon_loss(projection, labels, users, temperature=0.1, same_user_negative_weight=2.0)
    except ValueError as error:
        assert "cross-user positive" in str(error)
    else:
        raise AssertionError("Expected missing cross-user positives to fail")
