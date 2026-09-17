from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p98_four_modal_teacher_data import (  # noqa: E402
    FourModalData,
    FourModalPreprocessor,
    counterfactual_data,
    parse_sample_identity,
    validate_modalities,
)
from p98_four_modal_teacher_model import (  # noqa: E402
    FourModalTeacher,
    FourModalTeacherConfig,
)
from train_p98_four_modal_teacher import FourModalDataset  # noqa: E402


def synthetic_data(modalities: tuple[str, ...]) -> FourModalData:
    generator = np.random.default_rng(17)
    samples = 14
    streams: dict[str, np.ndarray] = {}
    stream_groups: dict[str, str] = {}
    statistics: dict[str, np.ndarray] = {}
    statistic_groups: dict[str, str] = {}
    if "ir" in modalities:
        streams["ir_vmae"] = generator.normal(size=(samples, 6, 16)).astype(np.float32)
        streams["ir_iv2"] = generator.normal(size=(samples, 6, 16)).astype(np.float32)
        stream_groups.update({"ir_vmae": "ir", "ir_iv2": "ir"})
    if "depth" in modalities:
        streams["depth"] = generator.normal(size=(samples, 3, 16)).astype(np.float32)
        stream_groups["depth"] = "depth"
    if "skeleton" in modalities:
        streams["skeleton_motionbert"] = generator.normal(
            size=(samples, 12, 16)
        ).astype(np.float32)
        stream_groups["skeleton_motionbert"] = "skeleton"
        statistics["skeleton_statistics"] = generator.normal(size=(samples, 24)).astype(
            np.float32
        )
        statistic_groups["skeleton_statistics"] = "skeleton"
    if "imu" in modalities:
        statistics["imu_statistics"] = generator.normal(size=(samples, 28)).astype(
            np.float32
        )
        statistic_groups["imu_statistics"] = "imu"
    if "skeleton" in modalities and "imu" in modalities:
        statistics["cross_relation_statistics"] = generator.normal(
            size=(samples, 12)
        ).astype(np.float32)
        statistic_groups["cross_relation_statistics"] = "cross"
    sample_ids = np.asarray(
        [f"train__c{index % 10:02d}__user{1 + index % 5}__1-1-{index}" for index in range(samples)]
    )
    labels = np.asarray([index % 10 for index in range(samples)], dtype=np.int64)
    users = np.asarray([f"user{1 + index % 5}" for index in range(samples)])
    expert_values = [
        np.eye(40, dtype=np.float32)[labels] * 0.90 + 0.10 / 40.0,
        np.full((samples, 40), 1.0 / 40.0, dtype=np.float32),
    ]
    expert_names = ["source_safe_decoded", "p87s_emission"]
    expert_groups = ["anchor", "anchor"]
    for modality in modalities:
        values = generator.uniform(0.01, 1.0, size=(samples, 40)).astype(np.float32)
        values /= values.sum(axis=1, keepdims=True)
        expert_values.append(values)
        expert_names.append(f"{modality}_expert")
        expert_groups.append(modality)
    skeleton_mask = np.ones((samples, 32, 17), dtype=np.float32)
    imu_mask = np.ones((samples, 32, 5, 4), dtype=np.float32)
    data = FourModalData(
        sample_ids=sample_ids,
        labels=labels,
        users=users,
        base_prediction=labels.copy(),
        expert_probability=np.stack(expert_values, axis=1),
        expert_names=tuple(expert_names),
        expert_groups=tuple(expert_groups),
        streams=streams,
        stream_groups=stream_groups,
        statistics=statistics,
        statistic_groups=statistic_groups,
        skeleton_sequence=generator.normal(size=(samples, 32, 17, 13)).astype(np.float32),
        skeleton_mask=skeleton_mask,
        imu_sequence=generator.normal(size=(samples, 32, 5, 4, 16)).astype(np.float32),
        imu_mask=imu_mask,
        modality_available={
            name: np.ones(samples, dtype=np.float32) for name in modalities
        },
        boundaries={
            "H1_selection": np.arange(0, 6),
            "E0_source_only": np.arange(6, 10),
            "H2_confirmation": np.arange(10, 14),
        },
        active_modalities=modalities,
    )
    data.validate()
    return data


def batch_from_data(data: FourModalData, rows: np.ndarray) -> dict[str, torch.Tensor]:
    dataset = FourModalDataset(data, rows, np.ones(len(data.labels), dtype=np.float32))
    items = [dataset[index] for index in range(len(dataset))]
    return {
        key: torch.stack([item[key] for item in items])
        for key in items[0]
    }


def model_from_data(data: FourModalData) -> FourModalTeacher:
    return FourModalTeacher(
        FourModalTeacherConfig(
            modalities=data.active_modalities,
            model_dim=32,
            layers=1,
            heads=4,
            dropout=0.0,
            modality_dropout=0.0,
        ),
        stream_dims={name: values.shape[-1] for name, values in data.streams.items()},
        stream_groups=data.stream_groups,
        statistic_dims={name: values.shape[-1] for name, values in data.statistics.items()},
        statistic_groups=data.statistic_groups,
        expert_names=data.expert_names,
        expert_groups=data.expert_groups,
    )


def test_source_identity_and_modality_contract() -> None:
    assert parse_sample_identity("train__c24__user19__5-1-2") == (24, "user19")
    assert validate_modalities(("imu", "ir", "imu", "depth")) == (
        "ir",
        "depth",
        "imu",
    )


def test_four_modal_preprocessor_and_model_forward_backward() -> None:
    raw = synthetic_data(("ir", "depth", "skeleton", "imu"))
    preprocessor = FourModalPreprocessor(statistics_dim=6, seed=17).fit(
        raw, np.arange(0, 10)
    )
    data = preprocessor.transform(raw)
    model = model_from_data(data)
    batch = batch_from_data(data, np.arange(10, 14))
    output = model(batch)
    assert output["logits"].shape == (4, 40)
    assert output["representation"].shape == (4, 32)
    assert output["modality_importance"].shape == (4, 4)
    assert set(output["modality_logits"]) == {"ir", "depth", "skeleton", "imu"}
    assert torch.allclose(
        output["modality_importance"].sum(dim=1), torch.ones(4), atol=1e-6
    )
    output["logits"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_visual_geometry_ablation_builds_without_motion_encoders() -> None:
    raw = synthetic_data(("ir", "depth"))
    data = FourModalPreprocessor(statistics_dim=4, seed=17).fit(
        raw, np.arange(0, 10)
    ).transform(raw)
    model = model_from_data(data)
    assert model.skeleton_encoder is None
    assert model.imu_encoder is None
    output = model(batch_from_data(data, np.arange(10, 14)))
    assert set(output["modality_logits"]) == {"ir", "depth"}
    assert output["modality_importance"].shape == (4, 2)


def test_zero_and_shuffle_counterfactuals_change_only_selected_modality() -> None:
    data = synthetic_data(("ir", "depth", "skeleton", "imu"))
    zero = counterfactual_data(data, "skeleton", "zero", seed=3)
    assert np.array_equal(zero.labels, data.labels)
    assert np.array_equal(zero.streams["ir_vmae"], data.streams["ir_vmae"])
    assert not zero.streams["skeleton_motionbert"].any()
    assert not zero.statistics["skeleton_statistics"].any()
    assert not zero.statistics["cross_relation_statistics"].any()
    assert not zero.skeleton_sequence.any()
    assert not zero.skeleton_mask.any()
    assert not zero.modality_available["skeleton"].any()
    skeleton_expert = data.expert_groups.index("skeleton")
    assert np.allclose(zero.expert_probability[:, skeleton_expert], 1.0 / 40.0)

    shuffled = counterfactual_data(data, "imu", "shuffle", seed=5)
    assert np.array_equal(shuffled.labels, data.labels)
    assert np.array_equal(shuffled.streams["depth"], data.streams["depth"])
    assert not np.array_equal(shuffled.imu_sequence, data.imu_sequence)
    imu_expert = data.expert_groups.index("imu")
    assert not np.array_equal(
        shuffled.expert_probability[:, imu_expert],
        data.expert_probability[:, imu_expert],
    )
    assert not np.array_equal(
        shuffled.statistics["cross_relation_statistics"],
        data.statistics["cross_relation_statistics"],
    )


def test_expert_residual_initializes_to_exact_safe_fallback() -> None:
    data = synthetic_data(("ir", "depth", "skeleton", "imu"))
    model = FourModalTeacher(
        FourModalTeacherConfig(
            modalities=data.active_modalities,
            model_dim=32,
            layers=1,
            heads=4,
            dropout=0.0,
            modality_dropout=0.0,
            expert_residual=True,
            anchor_margin=0.15,
            residual_scale=1.25,
        ),
        stream_dims={name: values.shape[-1] for name, values in data.streams.items()},
        stream_groups=data.stream_groups,
        statistic_dims={name: values.shape[-1] for name, values in data.statistics.items()},
        statistic_groups=data.statistic_groups,
        expert_names=data.expert_names,
        expert_groups=data.expert_groups,
    ).eval()
    batch = batch_from_data(data, np.arange(10, 14))
    with torch.no_grad():
        output = model(batch)
    assert torch.equal(output["logits"].argmax(dim=1), batch["base_prediction"])
    assert torch.count_nonzero(output["residual_logits"]) == 0


def test_expert_mixture_masks_unavailable_modality_and_normalizes_gate() -> None:
    raw = synthetic_data(("ir", "depth", "skeleton", "imu"))
    data = counterfactual_data(raw, "ir", "zero", seed=5)
    model = FourModalTeacher(
        FourModalTeacherConfig(
            modalities=data.active_modalities,
            model_dim=32,
            layers=1,
            heads=4,
            dropout=0.0,
            modality_dropout=0.0,
            expert_mixture=True,
        ),
        stream_dims={name: values.shape[-1] for name, values in data.streams.items()},
        stream_groups=data.stream_groups,
        statistic_dims={name: values.shape[-1] for name, values in data.statistics.items()},
        statistic_groups=data.statistic_groups,
        expert_names=data.expert_names,
        expert_groups=data.expert_groups,
    ).eval()
    with torch.no_grad():
        output = model(batch_from_data(data, np.arange(10, 14)))
    assert output["expert_gate"].shape == (4, len(data.expert_names))
    assert torch.allclose(output["expert_gate"].sum(dim=1), torch.ones(4), atol=1e-6)
    ir_expert = data.expert_groups.index("ir")
    assert torch.count_nonzero(output["expert_gate"][:, ir_expert]) == 0
