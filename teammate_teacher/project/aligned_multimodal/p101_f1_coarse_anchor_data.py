"""Merged coarse-P100 and fine-P101 data contract for the P101-F1 Teacher."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from p100a_global_teacher_data import (
    CANONICAL_VARIANTS as P100_VARIANTS,
    FoldNormalizer,
    P100AData,
    P100ADataset,
)
from p101_finegrained_teacher_data import (
    CANONICAL_VARIANTS as P101_VARIANTS,
    P101Data,
    P101Dataset,
)


class P101F1Dataset(Dataset[dict[str, torch.Tensor]]):
    """One row contains frozen coarse VS inputs plus fine local V/S/I inputs."""

    def __init__(
        self,
        coarse: P100AData,
        fine: P101Data,
        indices: np.ndarray,
        normalizer: FoldNormalizer,
        sample_weights: np.ndarray | None = None,
        nested_anchor_error: np.ndarray | None = None,
        nested_anchor_uncertainty: np.ndarray | None = None,
        skeleton_source: np.ndarray | None = None,
        imu_source: np.ndarray | None = None,
        zero_modalities: Iterable[str] = (),
        reverse_modalities: Iterable[str] = (),
        negative_imu_source: np.ndarray | None = None,
    ) -> None:
        if not np.array_equal(coarse.sample_ids, fine.sample_ids):
            raise RuntimeError("P100 coarse and P101 fine sample row orders differ")
        if not np.array_equal(coarse.labels, fine.labels):
            raise RuntimeError("P100 coarse and P101 fine labels differ")
        if not np.array_equal(coarse.fold_ids, fine.fold_ids):
            raise RuntimeError("P100 coarse and P101 fine folds differ")
        self.indices = np.asarray(indices, dtype=np.int64)
        zero = frozenset(zero_modalities)
        self.coarse_dataset = P100ADataset(
            coarse,
            self.indices,
            normalizer,
            P100_VARIANTS["VS"],
            sample_weights=sample_weights,
            skeleton_source=skeleton_source,
            zero_modalities=("skeleton",) if "skeleton" in zero else (),
        )
        self.fine_dataset = P101Dataset(
            fine,
            self.indices,
            P101_VARIANTS["VSI"],
            skeleton_source=skeleton_source,
            imu_source=imu_source,
            zero_modalities=zero,
            reverse_modalities=reverse_modalities,
            sample_weights=sample_weights,
            nested_vs_error=nested_anchor_error,
            negative_imu_source=negative_imu_source,
        )
        self.nested_anchor_uncertainty = nested_anchor_uncertainty

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        coarse = self.coarse_dataset[index]
        fine = self.fine_dataset[index]
        for key in ("row", "label"):
            if not torch.equal(coarse[key], fine[key]):
                raise RuntimeError(f"P101-F1 merged row field differs: {key}")
        # Fine local fields use explicit names and do not collide with P100's
        # coarse pooled fields.  Shared row/label/weight values are identical.
        output = dict(coarse)
        for key, value in fine.items():
            # P100 and P101 both use action-logit names, but P100 expects a
            # flattened six-token grid while P101 retains [2,3].  The F1 local
            # encoder consumes only temporal visual tokens, so preserve every
            # coarse field on collision and add only genuinely fine fields.
            if key not in output:
                output[key] = value
        output["nested_anchor_error"] = fine["nested_vs_error"]
        if self.nested_anchor_uncertainty is not None:
            row = int(output["row"])
            output["nested_anchor_uncertainty"] = torch.tensor(
                float(self.nested_anchor_uncertainty[row]), dtype=torch.float32
            )
        return output
