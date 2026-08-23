from __future__ import annotations

import copy

import torch
from torch import nn

from src.models.ir_depth_videomaev2_teacher import (
    IRDepthVideoMAEV2Teacher,
    sequential_multiview_backward,
)


class TinyVideoBackbone(nn.Module):
    def __init__(self, num_classes: int = 40) -> None:
        super().__init__()
        self.projection = nn.Linear(3, num_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.projection(clips.mean(dim=(2, 3, 4)))


def test_teacher_fuses_all_available_modality_views_into_trial_logits() -> None:
    model = IRDepthVideoMAEV2Teacher(backbone=TinyVideoBackbone(), num_classes=40)
    clips = torch.randn(2, 2, 4, 3, 4, 8, 8)
    availability = torch.ones(2, 2, 4, dtype=torch.bool)
    availability[0, 1, 3] = False

    output = model(clips=clips, availability=availability)

    assert output["logits"].shape == (2, 40)
    assert output["view_logits"].shape == (2, 2, 4, 40)
    assert output["class_view_weights"].shape == (2, 40, 2, 4)
    assert torch.allclose(
        output["class_view_weights"][0, :, 1, 3], torch.zeros(40)
    )
    assert torch.allclose(
        output["class_view_weights"].sum(dim=(2, 3)), torch.ones(2, 40)
    )
    assert model.last_execution_trace == [
        "ir:global",
        "ir:person_context",
        "ir:left_hand_object",
        "ir:right_hand_object",
        "depth:global",
        "depth:person_context",
        "depth:left_hand_object",
        "depth:right_hand_object",
    ]


def test_sequential_backward_matches_joint_multiview_gradient() -> None:
    torch.manual_seed(7)
    joint = IRDepthVideoMAEV2Teacher(backbone=TinyVideoBackbone(), num_classes=40)
    sequential = copy.deepcopy(joint)
    clips = torch.randn(1, 2, 4, 3, 4, 8, 8)
    availability = torch.ones(1, 2, 4, dtype=torch.bool)
    labels = torch.tensor([13])

    joint_loss = nn.functional.cross_entropy(
        joint(clips=clips, availability=availability)["logits"], labels
    )
    joint_loss.backward()
    result = sequential_multiview_backward(
        model=sequential,
        clips=clips,
        availability=availability,
        labels=labels,
        label_smoothing=0.0,
    )

    assert torch.allclose(result["loss"], joint_loss.detach(), atol=1e-6)
    assert result["logits"].shape == (1, 40)
    for joint_parameter, sequential_parameter in zip(
        joint.parameters(), sequential.parameters(), strict=True
    ):
        assert joint_parameter.grad is not None
        assert sequential_parameter.grad is not None
        assert torch.allclose(
            sequential_parameter.grad, joint_parameter.grad, atol=1e-6, rtol=1e-5
        )
