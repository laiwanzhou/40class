from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p100a_global_teacher_data import (  # noqa: E402
    CANONICAL_VARIANTS,
    DEV_USER_SET,
    FOLD_ROW_COUNTS,
    FoldNormalizer,
    P100ADataset,
    load_p100a_data,
    within_subject_permutation,
)
from p100a_global_teacher_model import (  # noqa: E402
    P100AGlobalTeacher,
    P100AModelConfig,
)
from train_p100a_global_teacher_oof import paired_comparison  # noqa: E402
from train_p100a_a1_protected_imu_oof import (  # noqa: E402
    build_anchor_and_candidate,
    exact_anchor_error,
    make_loader,
)


def test_real_cache_contract_is_exact_and_h3_free() -> None:
    data = load_p100a_data()
    assert len(data.sample_ids) == 1941
    assert set(data.users) == DEV_USER_SET
    assert tuple(int((data.fold_ids == fold).sum()) for fold in range(4)) == FOLD_ROW_COUNTS
    for fold in range(4):
        train, held = data.indices_for_fold(fold)
        assert not (set(data.users[train]) & set(data.users[held]))


def test_fold_normalizer_and_dataset_shapes() -> None:
    data = load_p100a_data()
    train, held = data.indices_for_fold(0)
    normalizer = FoldNormalizer.fit(data, train)
    dataset = P100ADataset(
        data, held[:2], normalizer, CANONICAL_VARIANTS["VSI"]
    )
    item = dataset[0]
    assert item["visual_vmae"].shape == (6, 768)
    assert item["skeleton_hdgcn"].shape == (6, 16, 256)
    assert item["skeleton_sequence"].shape == (32, 17, 13)
    assert item["imu_sequence"].shape == (32, 5, 4, 16)
    assert item["cross_statistics"].shape == (400,)
    assert torch.isfinite(item["imu_statistics"]).all()


def test_all_variants_have_one_post_fusion_classifier() -> None:
    data = load_p100a_data()
    train, held = data.indices_for_fold(0)
    normalizer = FoldNormalizer.fit(data, train)
    for variant, modalities in CANONICAL_VARIANTS.items():
        dataset = P100ADataset(data, held[:2], normalizer, modalities)
        batch = {
            name: torch.stack([dataset[0][name], dataset[1][name]])
            for name in dataset[0]
        }
        model = P100AGlobalTeacher(
            P100AModelConfig(
                modalities=modalities,
                model_dim=64,
                heads=4,
                modality_layers=1,
                fusion_layers=1,
                fusion_latents=4,
                dropout=0.0,
                evidence_dropout=0.0,
            )
        ).eval()
        with torch.no_grad():
            output = model(batch)
        assert output["logits"].shape == (2, 40), variant
        forty_class_linear = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.Linear) and module.out_features == 40
        ]
        assert len(forty_class_linear) == 1


def test_zero_evidence_closes_gate_before_classification() -> None:
    data = load_p100a_data()
    train, held = data.indices_for_fold(0)
    normalizer = FoldNormalizer.fit(data, train)
    modalities = CANONICAL_VARIANTS["VSI"]
    direct = P100ADataset(data, held[:2], normalizer, modalities)
    zero = P100ADataset(
        data,
        held[:2],
        normalizer,
        modalities,
        zero_modalities=("skeleton", "imu"),
    )
    direct_batch = {
        name: torch.stack([direct[0][name], direct[1][name]]) for name in direct[0]
    }
    zero_batch = {
        name: torch.stack([zero[0][name], zero[1][name]]) for name in zero[0]
    }
    model = P100AGlobalTeacher(
        P100AModelConfig(
            modalities=modalities,
            model_dim=64,
            heads=4,
            modality_layers=1,
            fusion_layers=1,
            fusion_latents=4,
            dropout=0.0,
            evidence_dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        direct_output = model(direct_batch)
        zero_output = model(zero_batch)
    assert torch.all(zero_output["reliability"]["skeleton"] == 0)
    assert torch.all(zero_output["reliability"]["imu"] == 0)
    assert torch.all(zero_output["reliability"]["cross"] == 0)
    assert not torch.allclose(direct_output["logits"], zero_output["logits"])


def test_within_subject_shuffle_preserves_subject_and_availability() -> None:
    data = load_p100a_data()
    _, held = data.indices_for_fold(2)
    source = within_subject_permutation(data, held, "imu", seed=7)
    assert np.array_equal(data.users[source[held]], data.users[held])
    assert np.array_equal(data.imu_available[source[held]], data.imu_available[held])
    assert np.any(source[held] != held)


def test_paired_comparison_counts_rescue_and_harm() -> None:
    labels = np.asarray([0, 1, 2, 3])
    control_prediction = np.asarray([0, 0, 2, 3])
    candidate_prediction = np.asarray([0, 1, 0, 3])
    control = np.eye(40)[control_prediction]
    candidate = np.eye(40)[candidate_prediction]
    users = np.asarray(["u1", "u1", "u2", "u2"])
    result = paired_comparison(candidate, control, labels, users)
    assert result["rescue"] == 1
    assert result["harm"] == 1
    assert result["net"] == 0


def test_protected_imu_zero_point_exactly_matches_vs_anchor() -> None:
    data = load_p100a_data()
    train, held = data.indices_for_fold(0)
    normalizer = FoldNormalizer.fit(data, train)
    shared = {
        "model_dim": 64,
        "heads": 4,
        "modality_layers": 1,
        "fusion_layers": 2,
        "fusion_latents": 4,
        "dropout": 0.0,
        "evidence_dropout": 0.0,
    }
    anchor_config = P100AModelConfig(
        modalities=CANONICAL_VARIANTS["VS"], **shared
    )
    source = P100AGlobalTeacher(anchor_config)
    checkpoint = {
        "state_dict": source.state_dict(),
        "model_config": anchor_config.__dict__,
    }
    config = {
        "model": {
            **shared,
            "protected_imu_residual": True,
            "protected_imu_max_scale": 0.5,
        }
    }
    anchor, candidate, trainable, frozen = build_anchor_and_candidate(
        checkpoint, config, torch.device("cpu")
    )
    dataset = P100ADataset(
        data, held[:3], normalizer, CANONICAL_VARIANTS["VSI"]
    )
    batch = next(iter(make_loader(dataset, 3, False, seed=7)))
    assert exact_anchor_error(anchor, candidate, batch, torch.device("cpu")) <= 1e-7
    assert any("imu_encoder" in name for name in trainable)
    assert any("protected_scale" in name for name in trainable)
    assert any("visual_encoder" in name for name in frozen)
    assert any("classifier" in name for name in frozen)
    with torch.no_grad():
        output = candidate(batch)
    assert torch.all(output["evidence_scale"]["imu"] == 0)
    assert torch.all(output["evidence_scale"]["cross"] == 0)
