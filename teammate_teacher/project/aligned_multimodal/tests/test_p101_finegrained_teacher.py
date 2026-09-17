from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p101_finegrained_teacher_data import (  # noqa: E402
    CANONICAL_VARIANTS,
    DEV_USER_SET,
    FOLD_ROW_COUNTS,
    P101Dataset,
    load_p101_data,
    within_subject_permutation,
    within_subject_wrong_label_source,
)
from p101_finegrained_teacher_model import (  # noqa: E402
    P101FineGrainedTeacher,
    P101ModelConfig,
    trainable_imu_adapter_parameters,
)
from train_p101_finegrained_teacher_oof import build_causal_base_model  # noqa: E402


def synthetic_batch(batch: int = 2, present: bool = True) -> dict[str, torch.Tensor]:
    joint_mask = torch.full((batch, 2, 16, 17), present)
    bin_mask = torch.full((batch, 2, 16, 5), present)
    return {
        "visual_vmae_temporal": torch.randn(batch, 2, 3, 8, 768),
        "visual_iv2_temporal": torch.randn(batch, 2, 3, 8, 768),
        "visual_vmae_pooled": torch.randn(batch, 2, 3, 768),
        "visual_iv2_pooled": torch.randn(batch, 2, 3, 768),
        "visual_vmae_action": torch.randn(batch, 2, 3, 710),
        "visual_iv2_action": torch.randn(batch, 2, 3, 400),
        "visual_time": torch.tensor(
            np.stack(
                (
                    np.linspace(0.0, 0.7, 8),
                    np.linspace(0.3, 1.0, 8),
                )
            ),
            dtype=torch.float32,
        ).expand(batch, -1, -1),
        "motion_time": torch.tensor(
            np.stack(
                (
                    np.linspace(0.0, 0.7, 16),
                    np.linspace(0.3, 1.0, 16),
                )
            ),
            dtype=torch.float32,
        ).expand(batch, -1, -1),
        "skeleton_features": torch.randn(batch, 2, 16, 17, 13),
        "skeleton_feature_mask": joint_mask[..., None].expand(-1, -1, -1, -1, 13),
        "skeleton_joint_mask": joint_mask,
        "skeleton_relations": torch.randn(batch, 2, 16, 18),
        "skeleton_relation_mask": torch.full((batch, 2, 16, 18), present),
        "skeleton_frame_quality": torch.ones(batch, 2, 16) * float(present),
        "imu_sequences": torch.randn(batch, 2, 16, 5, 4, 16),
        "imu_sequence_mask": bin_mask[..., None].expand(-1, -1, -1, -1, 4),
        "imu_bin_statistics": torch.randn(batch, 2, 16, 5, 52),
        "imu_bin_mask": bin_mask,
        "imu_global_statistics": torch.randn(batch, 5, 48),
        "imu_global_mask": torch.full((batch, 5, 2), present),
    }


def small_config(modalities: tuple[str, ...]) -> P101ModelConfig:
    return P101ModelConfig(
        modalities=modalities,
        model_dim=64,
        motion_dim=32,
        heads=4,
        global_layers=1,
        dropout=0.0,
    )


def test_real_p101_contract_is_1941_subject_disjoint_and_h3_free() -> None:
    data = load_p101_data()
    assert len(data.sample_ids) == 1941
    assert set(data.users.tolist()) == DEV_USER_SET
    assert tuple(int((data.fold_ids == fold).sum()) for fold in range(4)) == FOLD_ROW_COUNTS
    for fold in range(4):
        train, held = data.indices_for_fold(fold)
        assert not set(data.users[train]) & set(data.users[held])


def test_real_dataset_retains_visual_time_and_p86_part_device_grids() -> None:
    data = load_p101_data()
    _, held = data.indices_for_fold(0)
    item = P101Dataset(data, held[:1], CANONICAL_VARIANTS["VSI"])[0]
    assert item["visual_vmae_temporal"].shape == (2, 3, 8, 768)
    assert item["skeleton_features"].shape == (2, 16, 17, 13)
    assert item["imu_sequences"].shape == (2, 16, 5, 4, 16)
    assert torch.all(item["visual_time"][:, 1:] > item["visual_time"][:, :-1])
    assert torch.isfinite(item["visual_iv2_action"]).all()


def test_all_variants_have_only_one_final_40class_linear() -> None:
    batch = synthetic_batch()
    for modalities in CANONICAL_VARIANTS.values():
        model = P101FineGrainedTeacher(small_config(modalities)).eval()
        with torch.no_grad():
            output = model(batch)
        assert output["logits"].shape == (2, 40)
        heads = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.Linear) and module.out_features == 40
        ]
        assert len(heads) == 1


def test_vsi_zero_point_exactly_matches_loaded_vs_anchor() -> None:
    torch.manual_seed(11)
    anchor = P101FineGrainedTeacher(small_config(CANONICAL_VARIANTS["VS"])).eval()
    candidate = P101FineGrainedTeacher(small_config(CANONICAL_VARIANTS["VSI"])).eval()
    incompatible = candidate.load_state_dict(anchor.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    parameters, trainable, frozen = trainable_imu_adapter_parameters(candidate)
    assert parameters
    assert any(name.startswith("imu_encoder.") for name in trainable)
    assert "classifier.weight" in frozen
    batch = synthetic_batch()
    with torch.no_grad():
        anchor_logits = anchor(batch)["logits"]
        candidate_output = candidate(batch)
    torch.testing.assert_close(candidate_output["logits"], anchor_logits, rtol=0, atol=0)
    assert float(candidate_output["imu_residual_rms"]) == 0.0


def test_zero_motion_masks_remove_effect_before_classifier() -> None:
    model = P101FineGrainedTeacher(small_config(CANONICAL_VARIANTS["VSI"])).eval()
    first = synthetic_batch(present=False)
    second = {key: value.clone() for key, value in first.items()}
    second["skeleton_features"] = torch.randn_like(second["skeleton_features"]) * 100
    second["imu_sequences"] = torch.randn_like(second["imu_sequences"]) * 100
    with torch.no_grad():
        left = model(first)["logits"]
        right = model(second)["logits"]
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_within_subject_sources_never_cross_subject() -> None:
    data = load_p101_data()
    train, _ = data.indices_for_fold(0)
    shuffled = within_subject_permutation(data, train, "imu", seed=17)
    negative = within_subject_wrong_label_source(data, train, seed=19)
    assert np.array_equal(data.users[shuffled[train]], data.users[train])
    assert np.array_equal(data.users[negative[train]], data.users[train])
    changed = train[negative[train] != train]
    assert np.all(data.labels[negative[changed]] != data.labels[changed])


def test_causal_variants_share_exact_visual_global_classifier_start() -> None:
    config = {
        "model": {
            "model_dim": 64,
            "motion_dim": 32,
            "heads": 4,
            "global_layers": 1,
            "dropout": 0.0,
            "local_radius": 0.18,
            "workspace_arm_prior": 0.35,
            "maximum_imu_scale": 0.25,
        }
    }
    visual = build_causal_base_model(config, "V", 23, torch.device("cpu"))
    for variant in ("VS", "VI"):
        candidate = build_causal_base_model(config, variant, 23, torch.device("cpu"))
        candidate_state = candidate.state_dict()
        for name, value in visual.state_dict().items():
            torch.testing.assert_close(candidate_state[name], value, rtol=0, atol=0)
