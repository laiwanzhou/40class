from __future__ import annotations

import numpy as np
import torch

from src.models.target16_conditional_expert import Target16ConditionalExpert
from src.target16_hierarchical_fusion import fusion_metrics, hierarchical_predictions


TARGETS = np.arange(16, dtype=np.int64)


def test_b2_initialization_maps_target_head_and_freezes_early_layers() -> None:
    source = Target16ConditionalExpert(TARGETS, pretrained=False)
    state = source.state_dict()
    state["classifier.weight"] = torch.arange(40 * 192, dtype=torch.float32).reshape(40, 192)
    state["classifier.bias"] = torch.arange(40, dtype=torch.float32)

    model = Target16ConditionalExpert(TARGETS, pretrained=False)
    model.initialize_from_b2({"model_state_dict": state})
    assert torch.equal(model.classifier.weight, state["classifier.weight"][:16])
    assert torch.equal(model.classifier.bias, state["classifier.bias"][:16])
    audit = model.freeze_b2_early_layers()
    model.train()
    assert audit["trainable_parameters"] < audit["total_parameters"]
    assert not model.depth_stem.training
    assert not model.shared_body[0].training
    assert model.shared_body[-1].training


def test_alpha_zero_reproduces_b2_and_gate_protects_external_predictions() -> None:
    b2 = np.full((4, 40), 1e-4, dtype=np.float64)
    b2[0, 2], b2[0, 30] = 0.7, 0.1
    b2[1, 3], b2[1, 31] = 0.6, 0.2
    b2[2, 32], b2[2, 4] = 0.8, 0.1
    b2[3, 33], b2[3, 5] = 0.7, 0.2
    b2 /= b2.sum(axis=1, keepdims=True)
    e2 = np.full((4, 16), 1 / 16, dtype=np.float64)
    predictions, gate = hierarchical_predictions(b2, e2, TARGETS, alpha=0.0)
    assert np.array_equal(predictions, b2.argmax(axis=1))
    assert np.array_equal(gate, np.array([True, True, False, False]))


def test_correct_delta_equals_rescued_minus_harmed() -> None:
    labels = np.array([1, 2, 32, 33])
    base = np.array([2, 2, 4, 33])
    fused = np.array([1, 3, 5, 33])
    gate = np.array([True, True, True, False])
    metrics = fusion_metrics(labels, base, fused, gate, TARGETS)
    assert metrics["rescued"] == 1
    assert metrics["harmed"] == 1
    assert metrics["net_rescue"] == 0
