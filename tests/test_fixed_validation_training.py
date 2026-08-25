from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

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
from src.train_hierarchical_multimodal_teacher import train_candidate_split


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
