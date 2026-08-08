from __future__ import annotations

import numpy as np
import torch

from src.data.full_sequence_person_crop_dataset import FullSequencePersonCropPoseROIDataset
from src.models.full_sequence_multiscale_tcn import FullSequenceMultiScaleTCN


def test_full_sequence_indices_cover_both_clip_ends() -> None:
    dataset = object.__new__(FullSequencePersonCropPoseROIDataset)
    dataset.num_frames = 96
    indices, mask = dataset._window(135)
    assert len(indices) == 96
    assert indices[0] == 0 and indices[-1] == 134
    assert len(np.unique(indices)) == 96
    assert mask.all()


def test_short_sequences_are_padded_and_masked() -> None:
    dataset = object.__new__(FullSequencePersonCropPoseROIDataset)
    dataset.num_frames = 96
    indices, mask = dataset._window(23)
    assert np.array_equal(indices[:23], np.arange(23))
    assert np.all(indices[23:] == 22)
    assert int(mask.sum()) == 23


def test_multiscale_tcn_shapes_masks_and_receptive_fields() -> None:
    model = FullSequenceMultiScaleTCN(dropout=0.0)
    features = torch.randn(3, 96, 128)
    mask = torch.zeros(3, 96, dtype=torch.bool)
    mask[0, :23] = True
    mask[1, :69] = True
    mask[2] = True
    output = model(features, mask)
    assert output["logits"].shape == (3, 40)
    assert output["embedding"].shape == (3, 256)
    assert model.short_branch.receptive_field == 29
    assert model.long_branch.receptive_field == 253
    assert torch.allclose(output["short_attention"].sum(1), torch.ones(3))
    assert torch.allclose(output["long_attention"].sum(1), torch.ones(3))
    assert torch.count_nonzero(output["short_attention"][0, 23:]) == 0
