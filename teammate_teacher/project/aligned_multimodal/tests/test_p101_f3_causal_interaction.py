from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p100a_global_teacher_data import (  # noqa: E402
    CANONICAL_VARIANTS as P100_VARIANTS,
    FoldNormalizer,
    load_p100a_data,
)
from p100a_global_teacher_model import (  # noqa: E402
    P100AGlobalTeacher,
    P100AModelConfig,
)
from p101_f1_coarse_anchor_data import P101F1Dataset  # noqa: E402
from p101_f1_coarse_anchor_model import P101F1Config  # noqa: E402
from p101_f3_causal_interaction_model import (  # noqa: E402
    P101F3CausalInteractionTeacher,
)
from p101_finegrained_teacher_data import (  # noqa: E402
    load_p101_data,
    within_subject_wrong_label_source,
)


def small_anchor() -> P100AGlobalTeacher:
    return P100AGlobalTeacher(
        P100AModelConfig(
            modalities=P100_VARIANTS["VS"],
            model_dim=64,
            heads=4,
            modality_layers=1,
            fusion_layers=1,
            fusion_latents=4,
            dropout=0.0,
            evidence_dropout=0.0,
        )
    )


def small_model() -> P101F3CausalInteractionTeacher:
    return P101F3CausalInteractionTeacher(
        small_anchor(),
        P101F1Config(
            model_dim=64,
            motion_dim=32,
            heads=4,
            dropout=0.0,
            evidence_layers=1,
        ),
    )


def real_batch(
    zero_imu: bool = False, negative: bool = False
) -> dict[str, torch.Tensor]:
    coarse = load_p100a_data()
    fine = load_p101_data()
    train, held = coarse.indices_for_fold(0)
    rows = held[:2] if not negative else train[:2]
    negative_source = (
        within_subject_wrong_label_source(fine, train, 123) if negative else None
    )
    dataset = P101F1Dataset(
        coarse,
        fine,
        rows,
        FoldNormalizer.fit(coarse, train),
        zero_modalities=("imu",) if zero_imu else (),
        negative_imu_source=negative_source,
    )
    return {
        key: torch.stack([dataset[0][key], dataset[1][key]])
        for key in dataset[0]
    }


def test_f3_exact_anchor_initialization_and_one_classifier() -> None:
    model = small_model().eval()
    batch = real_batch()
    with torch.no_grad():
        output = model(batch)
    torch.testing.assert_close(output["logits"], output["anchor_logits"], rtol=0, atol=0)
    assert torch.all(output["fine_residual_rms"] == 0)
    forty_class = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Linear) and module.out_features == 40
    ]
    assert len(forty_class) == 1


def test_f3_identical_vsi_and_zero_evidence_cancel_algebraically() -> None:
    model = small_model().eval()
    with torch.no_grad():
        model.residual_projection.network[4].weight.normal_(std=0.1)
        model.residual_projection.network[4].bias.normal_(std=0.1)
        anchor = torch.randn(3, 64)
        tokens = torch.randn(3, 8, 64)
        residual, difference = model._centered_residual(anchor, tokens, tokens)
    torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0, atol=0)
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)


def test_f3_zero_imu_exact_anchor_after_nonzero_projection() -> None:
    model = small_model().eval()
    with torch.no_grad():
        model.residual_projection.network[4].weight.normal_(std=0.1)
        model.residual_projection.network[4].bias.normal_(std=0.1)
        output = model(real_batch(zero_imu=True))
    torch.testing.assert_close(output["logits"], output["anchor_logits"], rtol=0, atol=0)


def test_f3_correspondence_gate_is_in_classification_gradient_path() -> None:
    model = small_model().train()
    with torch.no_grad():
        model.residual_projection.network[4].weight.normal_(std=0.1)
    output = model(real_batch())
    output["logits"].square().sum().backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.local.correspondence.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0


def test_f3_training_negative_branch_produces_anchor_preservable_logits() -> None:
    model = small_model().train()
    output = model(real_batch(negative=True))
    assert output["negative_logits"].shape == (2, 40)
    assert output["negative_pair_gate"].shape == (2,)
    assert torch.isfinite(output["negative_logits"]).all()
