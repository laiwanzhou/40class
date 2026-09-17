from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p103_local_candidate_model import (
    LocalCandidateConfig,
    P103LocalCandidateTeacher,
    local_token_topology,
)
from p103_local_feature_data import VJEPA_VIEW_INDICES, VJEPA_VIEW_NAMES, VMAE_VIEW_NAMES


def synthetic_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    return {
        "vmae_features": torch.randn(batch_size, 6, 768, generator=generator),
        "vmae_actions": torch.randn(batch_size, 6, 710, generator=generator),
        "vjepa_features": torch.randn(batch_size, 16, 1024, generator=generator),
        "vjepa_actions": torch.randn(batch_size, 16, 174, generator=generator),
        "candidate_ids": torch.tensor([[0, 1, 2, -1]] * batch_size),
        "a_context": torch.zeros(batch_size, 4, 7),
        "local_scale": torch.ones(batch_size),
    }


def test_local_topology_preserves_encoder_roi_window_and_type() -> None:
    topology = local_token_topology()
    assert {name: value.shape for name, value in topology.items()} == {
        "encoder": (44,),
        "roi": (44,),
        "window": (44,),
        "token_type": (44,),
    }
    assert np.sum(topology["encoder"] == 0) == 12
    assert np.sum(topology["encoder"] == 1) == 32
    assert np.sum(topology["roi"] == 0) == 8
    assert np.sum(topology["roi"] == 3) == 12
    assert np.sum(topology["token_type"] == 0) == 22
    assert np.sum(topology["token_type"] == 1) == 22


def test_local_view_contract_is_explicit_and_non_global() -> None:
    assert len(VMAE_VIEW_NAMES) == 6
    assert len(VJEPA_VIEW_NAMES) == len(VJEPA_VIEW_INDICES) == 16
    assert set(VJEPA_VIEW_INDICES.tolist()) == {2, 5, 8, 11, *range(12, 24)}
    assert all("scene" not in name and "person" not in name for name in VJEPA_VIEW_NAMES)
    assert "hand_motion_peak_interaction" in VJEPA_VIEW_NAMES


def test_candidate_identity_changes_local_attention_before_score() -> None:
    torch.manual_seed(12)
    model = P103LocalCandidateTeacher(LocalCandidateConfig(dropout=0.0)).eval()
    output = model(synthetic_batch(1), return_attention=True)
    assert output["candidate_scores"].shape == (1, 4)
    assert output["token_attention"].shape == (1, 4, 4, 44)
    assert output["attention_groups"].shape == (1, 4, 13)
    assert output["candidate_scores"][0, 3] < -1000
    assert not torch.allclose(
        output["token_attention"][0, :, 0], output["token_attention"][0, :, 1]
    )
    assert not hasattr(model, "classifier")


def test_zero_local_removes_sample_specific_content() -> None:
    torch.manual_seed(13)
    model = P103LocalCandidateTeacher(LocalCandidateConfig(dropout=0.0)).eval()
    batch = synthetic_batch(2)
    batch["local_scale"] = torch.zeros(2)
    output = model(batch)
    assert torch.allclose(
        output["candidate_scores"][0], output["candidate_scores"][1], atol=1e-6
    )
