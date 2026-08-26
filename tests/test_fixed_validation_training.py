from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
import yaml

from scripts.report_hierarchical_multimodal_teacher import (
    build_fixed_validation_report,
)

from src.data.body_normalization_state import (
    apply_body_normalization_state,
    fit_body_normalization_state,
)
from src.data.canonical_multimodal_index import CanonicalTrial
from src.experiments.hierarchical_midfusion_config import load_midfusion_config
from src.models.body_motion_segment_encoder import BodyMotionSegmentEncoder
from src.models.hierarchical_action_query_fusion import HierarchicalActionQueryFusion
from src.models.hierarchical_multimodal_teacher import HierarchicalMultimodalTeacher
from src.models.structured_ir_depth_visual_encoder import StructuredIRDepthVisualEncoder
from src.train_hierarchical_multimodal_teacher import (
    FIXED_VALIDATION_CANDIDATES,
    run_fixed_validation,
    _diagnostic_tensor_to_numpy,
    train_candidate_split,
)


MODALITIES = ("ir", "depth_color", "skeleton", "imu", "radar", "thermal")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml"


class ConstantSkeletonLoader:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.normalization = None
        self.decode_calls = 0

    def __call__(self, trial: CanonicalTrial) -> dict[str, torch.Tensor]:
        self.decode_calls += 1
        return {
            "values": torch.full((8, 17, 6), self.value),
            "mask": torch.ones(8, dtype=torch.bool),
            "quality": torch.ones(8, 4),
            "modality_usable": torch.tensor(True),
        }

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.normalization = (mean.copy(), std.copy())


class ConstantIMULoader:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.normalization = None
        self.decode_calls = 0

    def __call__(self, trial: CanonicalTrial) -> dict[str, torch.Tensor]:
        self.decode_calls += 1
        return {
            "values": torch.full((8, 5, 16), self.value),
            "role_mask": torch.ones(8, 5, dtype=torch.bool),
            "quality": torch.ones(8, 5, 3),
            "modality_usable": torch.tensor(True),
        }

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.normalization = (mean.copy(), std.copy())


class BodyDataset:
    def __init__(self, users: tuple[str, ...], value: float) -> None:
        self.trials = [
            CanonicalTrial(
                sample_id=f"{user}_sample",
                user_id=user,
                class_id=index,
                paths={name: Path(f"{name}/{user}") for name in MODALITIES},
                availability={name: name in {"skeleton", "imu"} for name in MODALITIES},
            )
            for index, user in enumerate(users)
        ]
        self.skeleton_loader = ConstantSkeletonLoader(value)
        self.imu_loader = ConstantIMULoader(value)


def test_normalization_state_fits_train_and_applies_to_validation() -> None:
    train = BodyDataset(("user1", "user2"), value=2.0)
    validation = BodyDataset(("user6", "user7"), value=100.0)

    state = fit_body_normalization_state(train, np.arange(len(train.trials)))
    apply_body_normalization_state(train, state)
    apply_body_normalization_state(validation, state)

    assert state.fit_user_ids == ("user1", "user2")
    assert set(state.fit_user_ids).isdisjoint({"user6", "user7"})
    assert validation.skeleton_loader.decode_calls == 0
    assert validation.imu_loader.decode_calls == 0
    assert np.allclose(state.skeleton_mean, 2.0)
    assert np.allclose(state.imu_mean, 2.0)
    assert np.allclose(train.imu_loader.normalization[0], 2.0)
    assert np.allclose(validation.imu_loader.normalization[0], 2.0)


def test_bfloat16_diagnostics_are_archived_as_float32_numpy() -> None:
    values = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    archived = _diagnostic_tensor_to_numpy(values)

    assert archived.dtype == np.float32
    np.testing.assert_array_equal(archived, [[1.0, 2.0]])


class TinyTemporalBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 12
        self.proj = nn.Conv3d(3, 12, (2, 1, 1), stride=(2, 1, 1))
        self.tail = nn.Linear(12, 12)

    def encode_prefix(self, clips: torch.Tensor) -> torch.Tensor:
        return self.proj(clips).mean((3, 4)).transpose(1, 2)[:, :, None]

    def encode_tail(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tail(tokens.mean(2))


def tiny_model(_: dict) -> HierarchicalMultimodalTeacher:
    dim = 32
    return HierarchicalMultimodalTeacher(
        visual_encoder=StructuredIRDepthVisualEncoder(
            backbone=TinyTemporalBackbone(), output_dim=dim
        ),
        body_encoder=BodyMotionSegmentEncoder(output_dim=dim, heads=4),
        fusion=HierarchicalActionQueryFusion(
            dim=dim, classes=40, heads=4, layers=2
        ),
        dim=dim,
        classes=40,
    )


class TinySplitDataset(Dataset[dict[str, object]]):
    def __init__(self, *, prefix: str, users: tuple[str, ...]) -> None:
        self.user_ids = np.repeat(np.asarray(users), 2)
        self.labels = np.tile(np.asarray([0, 1]), len(users))
        self.sample_ids = np.asarray(
            [f"{prefix}_{index}" for index in range(len(self.labels))]
        )
        self.trials = [
            CanonicalTrial(
                sample_id=str(sample_id),
                user_id=str(user_id),
                class_id=int(label),
                paths={name: Path(f"{name}/{sample_id}") for name in MODALITIES},
                availability={name: name in {"ir", "depth_color", "skeleton", "imu"} for name in MODALITIES},
            )
            for sample_id, user_id, label in zip(
                self.sample_ids, self.user_ids, self.labels, strict=True
            )
        ]
        normalization_value = 2.0 if prefix == "train" else 100.0
        self.skeleton_loader = ConstantSkeletonLoader(normalization_value)
        self.imu_loader = ConstantIMULoader(normalization_value)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(900 + index)
        return {
            "visual": torch.randn(2, 4, 3, 16, 4, 4, generator=generator),
            "visual_view_availability": torch.ones(2, 4, dtype=torch.bool),
            "skeleton": torch.randn(8, 17, 6, generator=generator),
            "skeleton_mask": torch.ones(8, dtype=torch.bool),
            "skeleton_quality": torch.ones(8, 4),
            "imu": torch.randn(8, 5, 16, generator=generator),
            "imu_role_mask": torch.ones(8, 5, dtype=torch.bool),
            "imu_quality": torch.ones(8, 5, 3),
            "availability": torch.ones(4, dtype=torch.bool),
            "core_available": torch.tensor(True),
            "sample_id": str(self.sample_ids[index]),
            "user_id": str(self.user_ids[index]),
            "label": int(self.labels[index]),
        }


def test_candidate_split_isolates_user67_and_evaluates_each_scope_once(
    tmp_path: Path,
) -> None:
    config = load_midfusion_config(CONFIG)
    config["training"] = {**config["training"], "fixed_epochs": 1}
    train = TinySplitDataset(prefix="train", users=("user1", "user2"))
    validation = TinySplitDataset(prefix="validation", users=("user6", "user7"))

    result = train_candidate_split(
        config=config,
        candidate="visual_skeleton_imu",
        train_dataset=train,
        validation_dataset=validation,
        run_dir=tmp_path / "candidate",
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )

    assert result["fit_user_ids"] == ["user1", "user2"]
    assert result["validation_user_ids"] == ["user6", "user7"]
    assert result["train_evaluation_count"] == 1
    assert result["validation_evaluation_count"] == 1
    assert set(result["fit_sample_ids"]).isdisjoint(
        result["validation_sample_ids"]
    )
    assert (tmp_path / "candidate/train_predictions.npz").is_file()
    assert (tmp_path / "candidate/validation_predictions.npz").is_file()
    with np.load(
        tmp_path / "candidate/validation_predictions.npz", allow_pickle=False
    ) as archive:
        assert archive["logits"].shape == (4, 40)
        assert archive["group_attention"].shape == (4, 40, 3)
        assert archive["segment_attention"].shape == (4, 40, 8)
        assert archive["context_logits"].shape == (4, 40)
        assert archive["wrist_logits"].shape == (4, 40)
        assert archive["body_logits"].shape == (4, 40)
        assert archive["effective_group_mask"].shape == (4, 3)
        assert archive["availability"].shape == (4, 4)
        assert archive["skeleton_quality"].shape == (4, 8, 4)
        assert archive["imu_quality"].shape == (4, 8, 5, 3)


def test_candidate_split_resume_rejects_changed_validation_samples(
    tmp_path: Path,
) -> None:
    config = load_midfusion_config(CONFIG)
    config["training"] = {**config["training"], "fixed_epochs": 1}
    train = TinySplitDataset(prefix="train", users=("user1", "user2"))
    validation = TinySplitDataset(prefix="validation", users=("user6", "user7"))
    run_dir = tmp_path / "resume_candidate"
    train_candidate_split(
        config=config,
        candidate="visual_skeleton_imu",
        train_dataset=train,
        validation_dataset=validation,
        run_dir=run_dir,
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )
    (run_dir / "summary.json").unlink()
    validation.sample_ids[0] = "changed_validation_sample"

    with pytest.raises(RuntimeError, match="validation samples changed"):
        train_candidate_split(
            config=config,
            candidate="visual_skeleton_imu",
            train_dataset=train,
            validation_dataset=validation,
            run_dir=run_dir,
            model_factory=tiny_model,
            device=torch.device("cpu"),
        )


def test_completed_checkpoint_allows_explicit_evaluation_only_recovery(
    tmp_path: Path,
) -> None:
    config = load_midfusion_config(CONFIG)
    config["training"] = {**config["training"], "fixed_epochs": 1}
    config["recovery"] = {}
    train = TinySplitDataset(prefix="train", users=("user1", "user2"))
    validation = TinySplitDataset(prefix="validation", users=("user6", "user7"))
    run_dir = tmp_path / "evaluation_recovery"
    first = train_candidate_split(
        config=config,
        candidate="visual_only",
        train_dataset=train,
        validation_dataset=validation,
        run_dir=run_dir,
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )
    (run_dir / "summary.json").unlink()
    (run_dir / "train_predictions.npz").unlink()
    (run_dir / "validation_predictions.npz").unlink()
    config["recovery"] = {
        "evaluation_only_checkpoint": {
            "candidate": "visual_only",
            "config_sha256": first["config_sha256"],
            "completed_epoch": 1,
        }
    }

    recovered = train_candidate_split(
        config=config,
        candidate="visual_only",
        train_dataset=train,
        validation_dataset=validation,
        run_dir=run_dir,
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )

    assert recovered["evaluation_only_recovery"] is True
    assert recovered["training_config_sha256"] == first["config_sha256"]
    assert recovered["evaluation_config_sha256"] != first["config_sha256"]
    assert recovered["history"] == first["history"]


def test_fixed_runner_orchestrates_three_candidates_without_grouped_cv(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["training"]["fixed_epochs"] = 1
    config_path = tmp_path / "tiny_fixed.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def dataset_factory(_: dict, partition: str) -> TinySplitDataset:
        if partition == "train":
            return TinySplitDataset(prefix="train", users=("user1", "user2"))
        return TinySplitDataset(
            prefix="validation", users=("user6", "user7")
        )

    report = run_fixed_validation(
        config_path,
        output_root=tmp_path / "fixed_run",
        dataset_factory=dataset_factory,
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )

    assert report["evaluation_protocol"] == "fixed_user6_user7"
    assert report["train_population_samples"] == 4
    assert report["validation_population_samples"] == 4
    assert list(report["candidate_results"]) == list(FIXED_VALIDATION_CANDIDATES)
    assert report["validation_users_entered_training"] is False
    assert report["development_validation"] is True
    assert report["independent_final_test"] is False
    assert report["research_category"] in {
        "reject",
        "promising",
        "full_teacher_worthy",
        "teacher_target_reached",
    }
    assert report["student_planning_authorized"] == (
        report["research_category"]
        in {"full_teacher_worthy", "teacher_target_reached"}
    )
    assert len(report["provenance"]["git_commit"]) == 40
    assert len(report["provenance"]["config_file_sha256"]) == 64
    assert len(report["provenance"]["trainer_source_sha256"]) == 64
    assert {
        result["config_sha256"]
        for result in report["candidate_results"].values()
    } == {report["config_sha256"]}
    assert set(report["normalization"]["fit_user_ids"]) == {"user1", "user2"}
    assert (tmp_path / "fixed_run/normalization_state.npz").is_file()
    assert (tmp_path / "fixed_run/fixed_validation_report.json").is_file()
    assert (tmp_path / "fixed_run/fixed_validation_report.md").is_file()
    report_path = tmp_path / "fixed_run/fixed_validation_report.json"
    verified = build_fixed_validation_report(report_path)
    assert verified["selected_candidate"] == report["selected_candidate"]

    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["candidate_results"]["visual_only"]["validation_metrics"][
        "accuracy"
    ] += 0.25
    report_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metric mismatch"):
        build_fixed_validation_report(report_path)
