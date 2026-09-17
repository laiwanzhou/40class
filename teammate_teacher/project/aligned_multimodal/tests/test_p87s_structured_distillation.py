from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from adapt_p87s_structured_student import (  # noqa: E402
    LabelFreePseudoDataset,
    model_build_args,
    tempered_probability,
)
from build_p87s_structured_targets import (  # noqa: E402
    backed_off_structured_probability,
    normalized_entropy,
)
from build_p162_p150_student_targets import smoothed_one_hot  # noqa: E402
from build_p87s_test_structured_targets import config_from_nested_summary  # noqa: E402
from p87s_test_data import collate_p87s_test  # noqa: E402
from p87s_deploy_model import (  # noqa: E402
    build_p87s_deploy_model,
    deployment_model_config,
)
from predict_p87s_test_student import (  # noqa: E402
    changed_pair_histogram,
    class_histogram,
    entropy_from_log_probability,
    log_softmax_numpy,
)


class StubSupervisedDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        assert index == 0
        return {
            "sample_id": "held-sample",
            "label": torch.tensor(7, dtype=torch.long),
            "feature": torch.ones(2),
        }


def test_label_free_pseudo_dataset_deletes_ground_truth() -> None:
    probability = np.asarray([0.1, 0.9], dtype=np.float32)
    dataset = LabelFreePseudoDataset(
        StubSupervisedDataset(),
        {"held-sample": probability},
        {"held-sample": 0.75},
    )
    item = dataset[0]
    assert "label" not in item
    assert torch.allclose(item["pseudo_probability"], torch.from_numpy(probability))
    assert float(item["pseudo_confidence"]) == 0.75


def test_curriculum_dataset_exposes_both_targets_without_label() -> None:
    structured = np.asarray([0.9, 0.1], dtype=np.float32)
    emission = np.asarray([0.6, 0.4], dtype=np.float32)
    dataset = LabelFreePseudoDataset(
        StubSupervisedDataset(),
        {"held-sample": structured},
        {"held-sample": 0.8},
        secondary_probability_by_id={"held-sample": emission},
    )
    item = dataset[0]
    assert "label" not in item
    assert torch.allclose(
        item["pseudo_emission_probability"], torch.from_numpy(emission)
    )


class StubAlreadyLabelFreeDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        assert index == 0
        return {
            "sample_id": "SM_test_0001",
            "feature": torch.ones(2),
        }


def test_pseudo_wrapper_accepts_already_label_free_test_dataset() -> None:
    probability = np.asarray([0.2, 0.8], dtype=np.float32)
    dataset = LabelFreePseudoDataset(
        StubAlreadyLabelFreeDataset(),
        {"SM_test_0001": probability},
        {"SM_test_0001": 0.7},
    )
    item = dataset[0]
    assert "label" not in item
    assert torch.allclose(item["pseudo_probability"], torch.from_numpy(probability))


def test_test_collate_rejects_label_or_teacher_fields() -> None:
    clean = {
        "sample_id": "SM_test_0001",
        "user_id": "anonymous",
        "cache_index": 0,
        "feature": torch.ones(2),
    }
    batch = collate_p87s_test([clean])
    assert batch["sample_id"] == ["SM_test_0001"]
    for forbidden in (
        {**clean, "label": torch.tensor(0)},
        {**clean, "teacher_logits": torch.ones(2)},
    ):
        try:
            collate_p87s_test([forbidden])
        except RuntimeError:
            pass
        else:
            raise AssertionError("label/teacher Test field was not rejected")


def test_p87s_deployment_model_is_self_contained_and_under_budget() -> None:
    visual = {
        "backbone": "mc3_18_temporal",
        "classes": 40,
        "width": 512,
        "dropout": 0.18,
        "fusion_mode": "gated",
        "enable_distillation_projection": False,
        "frames": 16,
        "exact_time_modeling": False,
    }
    motion = {
        "width": 96,
        "alignment_width": 64,
        "classes": 40,
        "dropout": 0.12,
        "imu_instance_normalization": False,
        "imu_event_feature_width": 0,
        "skeleton_time_position": False,
        "skeleton_multistream": False,
        "skeleton_multistream_strength": 0.1,
        "skeleton_adaptive_graph": False,
        "domain_classes": 0,
        "domain_reversal_scale": 1.0,
    }
    config = deployment_model_config(
        visual,
        motion,
        "separate",
        {
            "initial_residual_strength": 0.25,
            "separate_modality_dropout": 0.0,
            "global_fusion_mode": "additive",
            "reliability_groups": 1,
        },
    )
    model = build_p87s_deploy_model(config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 23_560_564
    assert parameters * 4 / 1024**2 < 100.0


def test_tempered_probability_remains_normalized() -> None:
    probability = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
    softened = tempered_probability(probability, temperature=2.0)
    assert torch.allclose(softened.sum(dim=1), torch.ones(2))
    assert softened[0, 0] < probability[0, 0]


def test_p150_smoothed_one_hot_contract() -> None:
    probability = smoothed_one_hot(np.asarray([0, 39]), smoothing=0.02)
    assert probability.shape == (2, 40)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.array_equal(probability.argmax(axis=1), [0, 39])
    assert np.allclose(probability.max(axis=1), 0.9805)


def test_legacy_p87_summary_defaults_to_clip_fusion(tmp_path: Path) -> None:
    summary = {
        "config": {
            "visual_checkpoint": str(tmp_path / "visual.pt"),
            "pretrain_checkpoint": str(tmp_path / "motion.pt"),
            "initial_residual_strength": 0.25,
        },
        "modality": "separate",
    }
    args = model_build_args(tmp_path / "student.pt", summary)
    assert args.fusion_position == "clip"
    assert args.temporal_radius == 1
    assert args.temporal_residual_budget == 0.10
    assert args.temporal_attention_logit_limit == 1.0
    assert args.spatial_grid == 5
    assert args.spatial_attention_logit_limit == 1.0


def test_normalized_entropy_bounds() -> None:
    probability = np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    entropy = normalized_entropy(probability)
    assert np.allclose(entropy, [0.0, 1.0])


def test_structured_backoff_preserves_emission_support() -> None:
    emission = np.asarray([[0.25, 0.75]], dtype=np.float64)
    structured = np.asarray([[1.0, 0.0]], dtype=np.float64)
    probability, weight = backed_off_structured_probability(
        emission, structured, beam_width=50
    )
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert probability[0, 1] > 0.0
    assert np.allclose(weight, [0.98])


def test_test_config_requires_exact_nested_consensus(tmp_path: Path) -> None:
    fold_results = [
        {
            "selected": {
                "gap_seconds": gap,
                "transition_weight": weight,
                "trigram_backoff": backoff,
            }
        }
        for gap, weight, backoff in (
            (45.0, 0.30, 5.0),
            (30.0, 0.25, 5.0),
            (30.0, 0.25, 5.0),
        )
    ]
    summary = {
        "folds": fold_results,
        "test_config_from_outer_medians": {
            "gap_seconds": 30.0,
            "transition_weight": 0.25,
            "trigram_backoff": 5.0,
            "beam_width": 50,
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(__import__("json").dumps(summary), encoding="utf-8")
    config = config_from_nested_summary(path)
    assert config.gap_seconds == 30.0
    assert config.transition_weight == 0.25
    assert config.trigram_backoff == 5.0


def test_final_unlabeled_audit_probability_helpers() -> None:
    logits = np.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    log_probability = log_softmax_numpy(logits)
    assert np.allclose(np.exp(log_probability).sum(axis=1), 1.0)
    entropy = entropy_from_log_probability(log_probability)
    assert entropy[0] < entropy[1]


def test_final_unlabeled_audit_histograms_are_deterministic() -> None:
    before = np.asarray([0, 0, 2, 3, 3], dtype=np.int64)
    after = np.asarray([0, 1, 1, 3, 1], dtype=np.int64)
    assert class_histogram(after) == {"0": 1, "1": 3, "3": 1}
    assert changed_pair_histogram(before, after) == {
        "0->1": 1,
        "2->1": 1,
        "3->1": 1,
    }
