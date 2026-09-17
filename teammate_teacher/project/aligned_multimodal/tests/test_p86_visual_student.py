from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p86_visual_student_data import window_indices
from p86_visual_student_model import P86VisualStudent, model_size_mib
from train_p86_visual_student_oof import relation_loss


def test_window_indices_match_p85_contract() -> None:
    early = window_indices(101, 0.0, 0.70)
    late = window_indices(101, 0.30, 1.0)
    assert len(early) == len(late) == 16
    assert early[0] == 0 and early[-1] == 70
    assert late[0] == 30 and late[-1] == 100
    assert np.all(np.diff(early) >= 0) and np.all(np.diff(late) >= 0)


def test_visual_student_shapes_gradients_and_budget() -> None:
    model = P86VisualStudent(dropout=0.0)
    features = torch.randn(2, 2, 16, 3, 896)
    mask = torch.ones(2, 2, 16, 3, dtype=torch.bool)
    mask[1, :, :, 2] = False
    quality = mask.float() * 0.8
    position = torch.stack(
        (
            torch.linspace(0.0, 0.7, 16).repeat(2, 1),
            torch.linspace(0.3, 1.0, 16).repeat(2, 1),
        ),
        dim=1,
    )
    output = model(features, mask, quality, position)
    assert output["logits"].shape == (2, 40)
    assert output["visual_embedding"].shape == (2, 256)
    assert output["clip_embeddings"].shape == (2, 2, 3, 192)
    assert output["frame_sequence"].shape == (2, 2, 16, 192)
    assert torch.isfinite(output["logits"]).all()
    output["logits"].sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert model_size_mib(model, 4) < 20.0


def test_relation_loss_zero_for_matching_relations() -> None:
    values = torch.randn(3, 2, 3, 12)
    mask = torch.ones(3, 2, 3, dtype=torch.bool)
    assert torch.allclose(relation_loss(values, values, mask), torch.tensor(0.0), atol=1e-7)
