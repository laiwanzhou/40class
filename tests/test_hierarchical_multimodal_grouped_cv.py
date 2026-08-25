from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.experiments.hierarchical_midfusion_config import load_midfusion_config
from src.models.body_motion_segment_encoder import BodyMotionSegmentEncoder
from src.models.hierarchical_action_query_fusion import HierarchicalActionQueryFusion
from src.models.hierarchical_multimodal_teacher import HierarchicalMultimodalTeacher
from src.models.structured_ir_depth_visual_encoder import StructuredIRDepthVisualEncoder
from src.train_hierarchical_multimodal_teacher import (
    CANDIDATE_MODALITIES,
    EpochClassUserBalancedSampler,
    final_evaluation_candidates,
    fit_body_normalization,
    fit_class_prior,
    partition_fold_indices,
    pool_fold_predictions,
    select_grouped_candidate,
    train_candidate_fold,
)
from src.data.canonical_multimodal_index import CanonicalTrial
from src.data.hierarchical_multimodal_dataset import (
    EmptyModalityLoader,
    HierarchicalMultimodalDataset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml"


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


class TinyGroupedDataset(Dataset[dict[str, object]]):
    def __init__(self) -> None:
        self.user_ids = np.asarray(["a"] * 8 + ["b"] * 8 + ["c"] * 8)
        self.labels = np.tile(np.arange(8), 3)
        self.sample_ids = np.asarray([f"sample_{index}" for index in range(24)])

    def __len__(self) -> int:
        return 24

    def __getitem__(self, index: int) -> dict[str, object]:
        torch.manual_seed(500 + index)
        return {
            "visual": torch.randn(2, 4, 3, 16, 4, 4),
            "visual_view_availability": torch.ones(2, 4, dtype=torch.bool),
            "skeleton": torch.randn(8, 17, 6),
            "skeleton_mask": torch.ones(8, dtype=torch.bool),
            "imu": torch.randn(8, 5, 16),
            "imu_role_mask": torch.ones(8, 5, dtype=torch.bool),
            "availability": torch.ones(4, dtype=torch.bool),
            "core_available": torch.tensor(True),
            "sample_id": str(self.sample_ids[index]),
            "user_id": str(self.user_ids[index]),
            "label": int(self.labels[index]),
        }


def test_sampler_changes_sequence_each_epoch_and_balances_classes() -> None:
    dataset = TinyGroupedDataset()
    sampler = EpochClassUserBalancedSampler(
        labels=dataset.labels,
        users=dataset.user_ids,
        samples=2400,
        seed=17,
    )
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert first != second
    counts = np.bincount(dataset.labels[first], minlength=8)
    assert counts.max() - counts.min() < 100


def test_partition_fold_indices_never_puts_validation_user_in_fit() -> None:
    users = np.asarray(["a", "a", "b", "b", "c", "c"])
    fit, validation = partition_fold_indices(users, validation_users={"b"})

    assert set(users[fit]) == {"a", "c"}
    assert set(users[validation]) == {"b"}
    assert not set(fit) & set(validation)


def test_candidate_modalities_and_final_selection_are_frozen() -> None:
    assert CANDIDATE_MODALITIES == {
        "visual_only": ("ir", "depth_color"),
        "visual_skeleton": ("ir", "depth_color", "skeleton"),
        "visual_imu": ("ir", "depth_color", "imu"),
        "visual_skeleton_imu": ("ir", "depth_color", "skeleton", "imu"),
    }
    assert final_evaluation_candidates("visual_skeleton_imu") == (
        "visual_skeleton_imu",
    )


def test_train_only_class_prior_is_finite_and_normalized() -> None:
    prior = fit_class_prior(np.asarray([0, 0, 1, 2]), classes=4)

    assert torch.isfinite(prior).all()
    assert torch.allclose(prior.exp().sum(), torch.tensor(1.0))
    assert prior.argmax().item() == 0


def test_tiny_fold_runs_fixed_epochs_and_validates_once(tmp_path: Path) -> None:
    config = load_midfusion_config(CONFIG)
    config["training"] = {**config["training"], "fixed_epochs": 2}
    dataset = TinyGroupedDataset()
    fit, validation = partition_fold_indices(
        dataset.user_ids, validation_users={"c"}
    )

    result = train_candidate_fold(
        config=config,
        candidate="visual_skeleton_imu",
        dataset=dataset,
        fit_indices=fit,
        validation_indices=validation,
        run_dir=tmp_path / "fold",
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )

    assert result["epochs_completed"] == 2
    assert result["validation_evaluation_count"] == 1
    assert result["validation_sample_ids"] == dataset.sample_ids[validation].tolist()
    assert all("validation" not in row for row in result["history"])
    assert (tmp_path / "fold/latest_checkpoint.pt").is_file()


def test_tiny_fold_resumes_model_optimizer_and_history(tmp_path: Path) -> None:
    config = load_midfusion_config(CONFIG)
    config["training"] = {**config["training"], "fixed_epochs": 1}
    dataset = TinyGroupedDataset()
    fit, validation = partition_fold_indices(
        dataset.user_ids, validation_users={"c"}
    )
    run_dir = tmp_path / "resume_fold"
    first = train_candidate_fold(
        config=config,
        candidate="visual_skeleton_imu",
        dataset=dataset,
        fit_indices=fit,
        validation_indices=validation,
        run_dir=run_dir,
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )
    (run_dir / "summary.json").unlink()
    (run_dir / "validation_predictions.npz").unlink()
    config["training"] = {**config["training"], "fixed_epochs": 2}

    resumed = train_candidate_fold(
        config=config,
        candidate="visual_skeleton_imu",
        dataset=dataset,
        fit_indices=fit,
        validation_indices=validation,
        run_dir=run_dir,
        model_factory=tiny_model,
        device=torch.device("cpu"),
    )

    assert first["epochs_completed"] == 1
    assert resumed["epochs_completed"] == 2
    assert [row["epoch"] for row in resumed["history"]] == [1, 2]


class SkeletonNormalizationLoader:
    def __init__(self) -> None:
        self.normalization = None

    def __call__(self, trial):
        value = float(trial.class_id)
        return {
            "values": torch.full((8, 17, 6), value),
            "mask": torch.ones(8, dtype=torch.bool),
            "quality": torch.zeros(8, 4),
            "modality_usable": torch.tensor(True),
        }

    def set_normalization(self, mean, std):
        self.normalization = (mean, std)


class IMUNormalizationLoader:
    def __init__(self) -> None:
        self.normalization = None

    def __call__(self, trial):
        value = float(trial.class_id)
        return {
            "values": torch.full((8, 5, 16), value),
            "role_mask": torch.ones(8, 5, dtype=torch.bool),
            "quality": torch.zeros(8, 5, 3),
            "modality_usable": torch.tensor(True),
        }

    def set_normalization(self, mean, std):
        self.normalization = (mean, std)


def test_body_normalization_fits_only_supplied_indices() -> None:
    trials = [
        CanonicalTrial(
            sample_id=f"sample_{index}",
            user_id=f"u{index}",
            class_id=value,
            paths={name: None for name in ("ir", "depth_color", "skeleton", "imu", "radar", "thermal")},
            availability={
                "ir": False,
                "depth_color": False,
                "skeleton": True,
                "imu": True,
                "radar": False,
                "thermal": False,
            },
        )
        for index, value in enumerate((0, 2, 100))
    ]
    skeleton_loader = SkeletonNormalizationLoader()
    imu_loader = IMUNormalizationLoader()
    dataset = HierarchicalMultimodalDataset(
        trials,
        visual_loader=EmptyModalityLoader.visual(),
        skeleton_loader=skeleton_loader,
        imu_loader=imu_loader,
    )

    provenance = fit_body_normalization(dataset, np.asarray([0, 1]))

    assert provenance["fit_sample_ids"] == ["sample_0", "sample_1"]
    assert np.allclose(skeleton_loader.normalization[0], 1.0)
    assert np.allclose(imu_loader.normalization[0], 1.0)


def test_pool_fold_predictions_requires_exact_single_ownership() -> None:
    pooled = pool_fold_predictions(
        sample_count=4,
        classes=3,
        folds=[
            (np.asarray([0, 2]), np.asarray([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])),
            (np.asarray([1, 3]), np.asarray([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])),
        ],
    )

    assert pooled.shape == (4, 3)
    assert np.isfinite(pooled).all()


def test_grouped_candidate_selection_uses_frozen_metric_order() -> None:
    metrics = {
        "visual_only": {"accuracy": 0.7, "macro_f1": 0.6, "worst_user_accuracy": 0.5, "nll": 1.0},
        "visual_skeleton": {"accuracy": 0.7, "macro_f1": 0.61, "worst_user_accuracy": 0.4, "nll": 0.9},
        "visual_imu": {"accuracy": 0.7, "macro_f1": 0.61, "worst_user_accuracy": 0.45, "nll": 1.1},
        "visual_skeleton_imu": {"accuracy": 0.7, "macro_f1": 0.61, "worst_user_accuracy": 0.45, "nll": 1.1},
    }

    assert select_grouped_candidate(metrics) == "visual_imu"
