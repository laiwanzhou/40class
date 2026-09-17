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
from p101_f1_coarse_anchor_model import (  # noqa: E402
    P101F1CoarseAnchoredTeacher,
    P101F1Config,
    select_f1_trainable_parameters,
)
from p101_finegrained_teacher_data import load_p101_data  # noqa: E402
from p101_finegrained_teacher_model import (  # noqa: E402
    P101FineGrainedTeacher,
    P101ModelConfig,
)
from train_p101_f1_coarse_anchor_oof import (  # noqa: E402
    rebase_to_canonical_anchor,
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


def small_config() -> P101F1Config:
    return P101F1Config(
        model_dim=64,
        motion_dim=32,
        heads=4,
        dropout=0.0,
        evidence_layers=1,
    )


def real_batch(zero_imu: bool = False) -> dict[str, torch.Tensor]:
    coarse = load_p100a_data()
    fine = load_p101_data()
    train, held = coarse.indices_for_fold(0)
    dataset = P101F1Dataset(
        coarse,
        fine,
        held[:2],
        FoldNormalizer.fit(coarse, train),
        zero_modalities=("imu",) if zero_imu else (),
    )
    return {
        key: torch.stack([dataset[0][key], dataset[1][key]])
        for key in dataset[0]
    }


def test_f1_merged_real_dataset_is_aligned_and_h3_free() -> None:
    batch = real_batch()
    assert batch["visual_vmae"].shape == (2, 6, 768)
    assert batch["visual_vmae_temporal"].shape == (2, 2, 3, 8, 768)
    assert batch["skeleton_motionbert"].shape == (2, 12, 768)
    assert batch["skeleton_features"].shape == (2, 2, 16, 17, 13)
    assert batch["imu_sequences"].shape == (2, 2, 16, 5, 4, 16)
    assert torch.equal(batch["nested_anchor_error"], torch.zeros(2))


def test_f1_has_one_classifier_and_exact_anchor_initialization() -> None:
    anchor = small_anchor().eval()
    model = P101F1CoarseAnchoredTeacher(anchor, small_config()).eval()
    batch = real_batch()
    with torch.no_grad():
        anchor_logits = anchor(batch)["logits"]
        output = model(batch)
    torch.testing.assert_close(output["logits"], anchor_logits, rtol=0, atol=0)
    assert torch.all(output["fine_residual_rms"] == 0)
    forty_class = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Linear) and module.out_features == 40
    ]
    assert len(forty_class) == 1


def test_zero_imu_exactly_returns_anchor_even_after_residual_is_nonzero() -> None:
    anchor = small_anchor().eval()
    model = P101F1CoarseAnchoredTeacher(anchor, small_config()).eval()
    with torch.no_grad():
        model.residual_projection.network[4].weight.normal_(std=0.1)
        model.residual_projection.network[4].bias.normal_(std=0.1)
    batch = real_batch(zero_imu=True)
    with torch.no_grad():
        expected = anchor(batch)["logits"]
        output = model(batch)
    torch.testing.assert_close(output["logits"], expected, rtol=0, atol=0)
    assert torch.all(output["fine_residual_rms"] == 0)


def test_f1_loads_outer_safe_f0_local_state_and_freezes_anchor() -> None:
    f0 = P101FineGrainedTeacher(
        P101ModelConfig(
            modalities=("visual", "skeleton", "imu"),
            model_dim=64,
            motion_dim=32,
            heads=4,
            global_layers=1,
            dropout=0.0,
        )
    )
    model = P101F1CoarseAnchoredTeacher(small_anchor(), small_config())
    audit = model.local.load_f0_state(f0.state_dict())
    assert audit["loaded_tensors"] > 0
    parameters, trainable, frozen = select_f1_trainable_parameters(model)
    assert parameters
    assert any(name.startswith("local.imu_encoder.") for name in trainable)
    assert "anchor.classifier.0.weight" in frozen
    assert "anchor.classifier.3.weight" in frozen
    assert not any(name.startswith("anchor.") for name in trainable)


def test_canonical_anchor_rebase_preserves_learned_delta() -> None:
    reconstructed = np.array([[0.001, -0.001, 0.0]], dtype=np.float32)
    canonical = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    delta = np.array([[0.2, -0.1, -0.1]], dtype=np.float32)
    arrays = {
        "direct_anchor_logits": reconstructed,
        "direct_logits": reconstructed + delta,
        "local_reverse_skeleton_logits": reconstructed + 2 * delta,
        "zero_imu_logits": reconstructed.copy(),
        "reverse_imu_logits": reconstructed - delta,
        "shuffle_imu_logits": reconstructed + 0.5 * delta,
        "zero_skeleton_logits": np.array([[4.0, 3.0, 2.0]], dtype=np.float32),
        "direct_probability": np.zeros((1, 3), dtype=np.float32),
    }
    rebased, audit = rebase_to_canonical_anchor(arrays, canonical)
    np.testing.assert_array_equal(rebased["direct_anchor_logits"], canonical)
    np.testing.assert_allclose(rebased["direct_logits"] - canonical, delta)
    np.testing.assert_allclose(
        rebased["local_reverse_skeleton_logits"] - canonical, 2 * delta
    )
    np.testing.assert_array_equal(
        rebased["zero_skeleton_logits"], arrays["zero_skeleton_logits"]
    )
    assert audit["max_probability_drift_before_rebase"] > 0
