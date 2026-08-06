from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sized
import math
from typing import Any

import numpy as np
from torch.utils.data import Sampler


class CrossUserActionBatchSampler(Sampler[list[int]]):
    """Build 2-action x 2-user batches for cross-user supervised contrast."""

    def __init__(self, dataset: Sized, batch_size: int, seed: int) -> None:
        if batch_size != 4:
            raise ValueError("CrossUserActionBatchSampler requires batch_size=4")
        samples = getattr(dataset, "samples", None)
        if not isinstance(samples, list) or not samples:
            raise TypeError("Dataset must expose a non-empty samples list")
        self.seed = int(seed)
        self.epoch = 0
        self.batch_count = math.ceil(len(samples) / batch_size)
        grouped: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, sample in enumerate(samples):
            grouped[int(sample["label"])][str(sample["user_id"])].append(index)
        self.grouped = {label: dict(users) for label, users in grouped.items()}
        self.labels = sorted(self.grouped)
        self.label_support = {
            label: sum(len(indices) for indices in self.grouped[label].values()) for label in self.labels
        }
        self.sample_count = len(samples)
        self.compatible: dict[int, list[int]] = {}
        for first in self.labels:
            first_users = set(self.grouped[first])
            self.compatible[first] = [
                second for second in self.labels
                if second != first and len(first_users & set(self.grouped[second])) >= 2
            ]
            if not self.compatible[first]:
                raise ValueError(f"Class {first} has no cross-user, cross-action batch partner")

    def __len__(self) -> int:
        return self.batch_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        total_slots = self.batch_count * 2
        ideal = {
            label: self.label_support[label] * total_slots / self.sample_count for label in self.labels
        }
        slot_counts = {label: int(np.floor(value)) for label, value in ideal.items()}
        remainder = total_slots - sum(slot_counts.values())
        tie_break = {label: float(rng.random()) for label in self.labels}
        for label in sorted(
            self.labels, key=lambda value: (ideal[value] - slot_counts[value], tie_break[value]), reverse=True,
        )[:remainder]:
            slot_counts[label] += 1
        slots = np.concatenate([
            np.full(slot_counts[label], label, dtype=np.int64) for label in self.labels
        ])
        rng.shuffle(slots)
        remaining = {label: int(np.sum(slots == label)) for label in self.labels}
        pairs: list[tuple[int, int]] = []
        while sum(remaining.values()):
            maximum = max(remaining.values())
            first = int(rng.choice([label for label, count in remaining.items() if count == maximum]))
            remaining[first] -= 1
            candidates = [label for label in self.compatible[first] if remaining[label] > 0]
            if not candidates:
                raise RuntimeError("Could not balance compatible action pairs")
            candidate_maximum = max(remaining[label] for label in candidates)
            second = int(rng.choice([label for label in candidates if remaining[label] == candidate_maximum]))
            remaining[second] -= 1
            pairs.append((first, second))
        rng.shuffle(pairs)
        pools = {
            (label, user): rng.permutation(indices).tolist()
            for label, users in self.grouped.items()
            for user, indices in users.items()
        }
        for first, second in pairs:
            common_users = sorted(set(self.grouped[first]) & set(self.grouped[second]))
            user_pairs = [
                (common_users[left], common_users[right])
                for left in range(len(common_users)) for right in range(left + 1, len(common_users))
            ]
            scores = [sum(bool(pools[(label, user)]) for label in (first, second) for user in pair) for pair in user_pairs]
            maximum_score = max(scores)
            selected_users = list(user_pairs[int(rng.choice(np.flatnonzero(np.asarray(scores) == maximum_score)))])
            for label in (first, second):
                for user in selected_users:
                    if not pools[(label, user)]:
                        pools[(label, user)] = rng.permutation(self.grouped[label][user]).tolist()
            batch = [
                int(pools[(label, str(user_id))].pop())
                for label in (first, second)
                for user_id in selected_users
            ]
            rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def audit(self, epochs: int = 1) -> dict[str, Any]:
        original_epoch = self.epoch
        batches = anchors = cross_user_positives = same_user_negatives = 0
        class_counts = np.zeros(len(self.labels), dtype=np.int64)
        unique_rates: list[float] = []
        for epoch in range(epochs):
            self.set_epoch(epoch)
            epoch_indices: set[int] = set()
            for batch in self:
                epoch_indices.update(batch)
                batches += 1
                labels = [int(getattr(self, "_audit_samples")[index]["label"]) for index in batch]
                users = [str(getattr(self, "_audit_samples")[index]["user_id"]) for index in batch]
                for anchor in range(len(batch)):
                    anchors += 1
                    class_counts[labels[anchor]] += 1
                    cross_user_positives += int(any(
                        labels[other] == labels[anchor] and users[other] != users[anchor]
                        for other in range(len(batch)) if other != anchor
                    ))
                    same_user_negatives += int(any(
                        labels[other] != labels[anchor] and users[other] == users[anchor]
                        for other in range(len(batch)) if other != anchor
                    ))
            unique_rates.append(len(epoch_indices) / self.sample_count)
        self.epoch = original_epoch
        return {
            "batches": batches,
            "anchors": anchors,
            "cross_user_positive_anchor_rate": cross_user_positives / anchors,
            "same_user_negative_anchor_rate": same_user_negatives / anchors,
            "class_anchor_count_min": int(class_counts.min()),
            "class_anchor_count_max": int(class_counts.max()),
            "unique_sample_rate_min": float(min(unique_rates)),
        }

    @classmethod
    def from_dataset(cls, dataset: Sized, batch_size: int, seed: int) -> "CrossUserActionBatchSampler":
        sampler = cls(dataset, batch_size, seed)
        sampler._audit_samples = getattr(dataset, "samples")
        return sampler
