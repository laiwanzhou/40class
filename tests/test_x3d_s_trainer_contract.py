from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
import src.train_x3d_s_visual_expert as trainer_module

from src.models.expert_contract import ExpertOutput
from src.data.ir_primary_full_sequence_dataset import class_map_hash
from src.models.x3d_s_visual_expert import X3DSVisualExpert
from scripts.audit_x3d_s_run import audit_smoke_run
from src.train_x3d_s_visual_expert import (
    ClipBudgetBatchSampler,
    TrialPredictionResult,
    aggregate_clip_predictions,
    apply_runtime_overrides,
    build_arg_parser,
    finalize_train14,
    prepare_finalize_manifest,
    is_better_checkpoint,
    prepare_run_directory,
    resolved_config_sha256,
    run_model_epoch,
    save_prediction_archive,
    train_partition,
    train_strict_oof_partition,
    _trial_nll_loss,
    refit_strict_oof_partition,
    validate_config,
    validate_oof_assignment,
    validate_user_partition,
    _formal_refit_config,
    _length_bucket,
    _learning_rates_by_scope,
    _maximum_learning_rate,
    _warmup_cosine_multiplier,
)


class TinyTrainerBackbone(nn.Module):
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=1, stride=(1, 4, 4), bias=False)
        self.bn = nn.BatchNorm3d(4)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.bn(self.conv(clips)))).flatten(1)


class WideTrainerBackbone(nn.Module):
    output_dim = 2048

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, self.output_dim))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        pooled = clips.mean(dim=(1, 2, 3, 4)).unsqueeze(1)
        return pooled * self.scale.unsqueeze(0)


class BlockTrainerBackbone(nn.Module):
    output_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(3, 3, kernel_size=1, bias=False),
                    nn.BatchNorm3d(3),
                    nn.ReLU(),
                )
                for _ in range(4)
            ]
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            clips = block(clips)
        return self.pool(clips).flatten(1)


def tiny_trainer_model() -> X3DSVisualExpert:
    return X3DSVisualExpert(
        backbone=TinyTrainerBackbone(),
        num_classes=40,
        embedding_dim=8,
        dropout=0.0,
        update_backbone_bn_running_stats=False,
    )


def direct_trainer_model() -> X3DSVisualExpert:
    return X3DSVisualExpert(
        backbone=WideTrainerBackbone(),
        num_classes=40,
        embedding_dim=2048,
        dropout=0.25,
        head_type="direct",
        update_backbone_bn_running_stats=False,
    )


def model_batch(prefix: str = "sample") -> dict[str, object]:
    return {
        "clips": torch.randn(2, 2, 1, 3, 13, 16, 16),
        "clip_mask": torch.tensor([[True, True], [True, False]]),
        "labels": torch.tensor([0, 1]),
        "sample_ids": (f"{prefix}-a", f"{prefix}-b"),
        "user_ids": ("u1", "u2"),
        "class_map_hash": "class-map",
        "quality": torch.ones(2, 6),
        "quality_mask": torch.ones(2, 6, dtype=torch.bool),
        "availability": torch.ones(2, 1, dtype=torch.bool),
        "num_frames": torch.tensor([33, 13]),
        "num_clips": torch.tensor([2, 1]),
    }


class TinyTrialDataset(Dataset[dict[str, object]]):
    class_map_hash = "class-map"
    class_names = [f"class-{index}" for index in range(40)]

    def __init__(self, prefix: str, rows: int) -> None:
        self.prefix = prefix
        self.rows = rows
        self.num_clips = [1 + index % 2 for index in range(rows)]
        self.epoch = -1

    def __len__(self) -> int:
        return self.rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        count = self.num_clips[index]
        return {
            "clips": torch.randn(count, 1, 3, 13, 16, 16),
            "clip_mask": torch.ones(count, dtype=torch.bool),
            "num_frames": 13 + index,
            "num_clips": count,
            "label": index % 2,
            "sample_id": f"{self.prefix}-{index}",
            "user_id": f"u{index % 2}",
            "class_map_hash": self.class_map_hash,
            "quality": torch.ones(6),
            "quality_mask": torch.ones(6, dtype=torch.bool),
            "availability": torch.ones(1, dtype=torch.bool),
            "source_indices": torch.zeros(count, 1, 13, dtype=torch.long),
            "window_bounds": torch.zeros(count, 2, dtype=torch.long),
            "clip_unique_frame_fraction": torch.ones(count),
        }


class TrialLinearExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 40, bias=False)

    def forward(
        self,
        clips: torch.Tensor,
        *,
        clip_mask: torch.Tensor,
        quality: torch.Tensor,
        quality_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> ExpertOutput:
        del clip_mask
        feature = clips[:, 0, 0, 0, 0, 0].unsqueeze(1)
        logits = torch.log_softmax(self.classifier(feature), dim=-1)
        return ExpertOutput(
            main_logits=logits,
            embedding=torch.nn.functional.normalize(torch.cat((feature, feature), dim=1)),
            quality=quality,
            quality_mask=quality_mask,
            availability=availability,
        )


def trial_linear_batch(features: list[float], *, offset: int) -> dict[str, object]:
    rows = len(features)
    clips = torch.zeros(rows, 1, 1, 3, 13, 1, 1)
    clips[:, 0, 0, 0, 0, 0, 0] = torch.tensor(features)
    return {
        "clips": clips,
        "clip_mask": torch.ones(rows, 1, dtype=torch.bool),
        "labels": torch.tensor([(offset + index) % 3 for index in range(rows)]),
        "sample_ids": tuple(f"weighted-{offset + index}" for index in range(rows)),
        "user_ids": tuple("u1" for _ in range(rows)),
        "class_map_hash": "class-map",
        "quality": torch.ones(rows, 1),
        "quality_mask": torch.ones(rows, 1, dtype=torch.bool),
        "availability": torch.ones(rows, 1, dtype=torch.bool),
        "num_frames": torch.tensor([65 if rows == 1 else 13] * rows),
        "num_clips": torch.ones(rows, dtype=torch.long),
    }


def fixed_config(tmp_path: Path) -> dict[str, object]:
    return {
        "input_manifest": "manifest.csv",
        "output_root": str(tmp_path),
        "input_view": "ir_context_path",
        "num_classes": 40,
        "embedding_dim": 256,
        "seed": 20260715,
        "device": "cpu",
        "pretrained": False,
        "temporal": {
            "local_frames": 13,
            "target_window_frames": 32,
            "max_clips": 8,
            "train_views_per_window": 1,
            "val_views_per_window": 1,
            "aggregation": "mean_probability",
        },
        "loader": {
            "max_trials_per_batch": 2,
            "max_valid_clips_per_batch": 8,
            "num_workers": 0,
        },
        "backbone_bn": {
            "update_running_stats": False,
            "train_affine_after_unfreeze": True,
        },
        "optimizer": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "weight_decay": 0.05,
            "gradient_accumulation": 4,
            "gradient_clip": 1.0,
        },
        "training": {"epochs": 30, "warmup_epochs": 2, "patience": 8},
        "amp": {"enabled": True, "dtype": "bfloat16"},
        "size_gate": {"internal_limit_bytes": 95_000_000},
    }


def test_validate_config_accepts_bounded_spawn_workers(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["loader"] = {
        "max_trials_per_batch": 2,
        "max_valid_clips_per_batch": 8,
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "multiprocessing_context": "spawn",
        "worker_torch_threads": 1,
    }

    validate_config(config)


def test_validate_config_rejects_unsafe_multiworker_settings(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["loader"] = {
        "max_trials_per_batch": 2,
        "max_valid_clips_per_batch": 8,
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "multiprocessing_context": "spawn",
        "worker_torch_threads": 2,
    }

    with pytest.raises(ValueError, match="one Torch CPU thread"):
        validate_config(config)


def fixture_prediction_result() -> TrialPredictionResult:
    rows = 3
    return TrialPredictionResult(
        sample_ids=("a", "b", "c"),
        user_ids=("u1", "u1", "u2"),
        labels=torch.tensor([0, 1, 2]),
        class_map_hash="class-map",
        output=ExpertOutput(
            main_logits=torch.log_softmax(torch.randn(rows, 40), dim=-1),
            embedding=torch.randn(rows, 256),
            quality=torch.ones(rows, 6),
            quality_mask=torch.ones(rows, 6, dtype=torch.bool),
            availability=torch.ones(rows, 1, dtype=torch.bool),
        ),
        num_frames=torch.tensor([13, 33, 236]),
        num_clips=torch.tensor([1, 2, 8]),
    )


def test_validation_aggregates_valid_clips_with_one_view() -> None:
    clip_view_logits = torch.tensor([[[[4.0, 0.0]], [[0.0, 9.0]]]])

    probabilities = aggregate_clip_predictions(
        clip_view_logits,
        clip_mask=torch.tensor([[True, False]]),
    )

    assert probabilities.shape == (1, 2)
    assert probabilities.argmax(dim=1).item() == 0


def test_trial_nll_loss_matches_standard_nll_without_smoothing() -> None:
    logits = torch.log_softmax(torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]), dim=1)
    labels = torch.tensor([0, 2])

    actual = _trial_nll_loss(logits, labels, label_smoothing=0.0, reduction="sum")
    expected = torch.nn.functional.nll_loss(logits, labels, reduction="sum")

    torch.testing.assert_close(actual, expected)


def test_trial_nll_loss_uses_uniform_label_smoothing_per_trial() -> None:
    logits = torch.log_softmax(torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]), dim=1)
    labels = torch.tensor([0, 2])
    rows = torch.arange(labels.shape[0])

    actual = _trial_nll_loss(logits, labels, label_smoothing=0.1, reduction="sum")
    expected = -(
        0.9 * logits[rows, labels]
        + 0.1 * logits.mean(dim=1)
    ).sum()

    torch.testing.assert_close(actual, expected)


def test_prediction_archive_contains_fusion_contract_fields(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"

    save_prediction_archive(path, fixture_prediction_result())

    with np.load(path, allow_pickle=False) as data:
        assert {
            "sample_ids",
            "user_ids",
            "labels",
            "logits",
            "embeddings",
            "quality",
            "quality_mask",
            "availability",
            "class_map_hash",
            "num_frames",
            "num_clips",
        } <= set(data.files)
        assert data["logits"].shape == (3, 40)
        assert data["embeddings"].shape == (3, 256)
        assert data["head_type"].item() == "projected"
        assert int(data["embedding_dim"].item()) == 256


def test_direct_prediction_archive_records_head_and_embedding_dimension(
    tmp_path: Path,
) -> None:
    path = tmp_path / "direct_predictions.npz"
    result = fixture_prediction_result()
    direct_result = TrialPredictionResult(
        **{
            **result.__dict__,
            "output": ExpertOutput(
                **{
                    **result.output.__dict__,
                    "embedding": torch.randn(3, 2048),
                }
            ),
        }
    )

    save_prediction_archive(path, direct_result, head_type="direct")

    with np.load(path, allow_pickle=False) as data:
        assert data["head_type"].item() == "direct"
        assert int(data["embedding_dim"].item()) == 2048
        assert data["embeddings"].shape == (3, 2048)


def test_prediction_result_rejects_duplicate_sample_ids() -> None:
    result = fixture_prediction_result()
    malformed = TrialPredictionResult(**{**result.__dict__, "sample_ids": ("a", "a", "c")})

    with pytest.raises(ValueError, match="duplicate sample"):
        malformed.validate()


def test_config_freezes_first_run_temporal_and_bn_policy(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    validate_config(config)
    config["backbone_bn"] = {
        "update_running_stats": True,
        "train_affine_after_unfreeze": True,
    }

    with pytest.raises(ValueError, match="running stats"):
        validate_config(config)


def test_head_type_defaults_to_projected_and_direct_requires_x3d_2048(
    tmp_path: Path,
) -> None:
    config = fixed_config(tmp_path)
    validate_config(config)
    assert trainer_module._resolved_head_type(config) == "projected"

    direct = fixed_config(tmp_path)
    direct["head_type"] = "direct"
    direct["embedding_dim"] = 2048
    validate_config(direct)
    assert trainer_module._resolved_head_type(direct) == "direct"

    wrong_dimension = {**direct, "embedding_dim": 256}
    with pytest.raises(ValueError, match="embedding_dim must be 2048"):
        validate_config(wrong_dimension)

    wrong_family = {**direct, "model_family": "mobilenet_v3_small_tcn"}
    with pytest.raises(ValueError, match="only supported for x3d_s"):
        validate_config(wrong_family)

    unknown = {**config, "head_type": "unknown"}
    with pytest.raises(ValueError, match="head_type"):
        validate_config(unknown)


def test_build_model_passes_direct_head_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["head_type"] = "direct"
    config["embedding_dim"] = 2048
    config["dropout"] = 0.25
    monkeypatch.setattr(
        trainer_module,
        "build_x3d_s_feature_backbone",
        lambda *, pretrained: WideTrainerBackbone(),
    )

    model = trainer_module._build_model(config)

    assert isinstance(model, X3DSVisualExpert)
    assert model.head_type == "direct"
    assert model.output_embedding_dim == 2048


def test_build_model_uses_standard_rgb_backbone_for_ir_anchored_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = fixed_config(tmp_path)
    config.update(
        input_view="depth_color_rgb_plus_ir_gray",
        input_channels=4,
        fusion_strategy="ir_anchored_depth_residual",
        dropout=0.25,
    )
    calls: list[dict[str, object]] = []

    def build_backbone(**kwargs: object) -> TinyTrainerBackbone:
        calls.append(dict(kwargs))
        return TinyTrainerBackbone()

    monkeypatch.setattr(trainer_module, "build_x3d_s_feature_backbone", build_backbone)

    model = trainer_module._build_model(config)

    assert isinstance(model, X3DSVisualExpert)
    assert model.input_channels == 4
    assert model.input_adapter_mode == "ir_anchored_depth_residual"
    assert calls == [{"pretrained": False}]


def test_config_accepts_bounded_partial_unfreeze_policy(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["training"] = {
        "epochs": 20,
        "scheduler_horizon_epochs": 20,
        "warmup_epochs": 5,
        "unfrozen_backbone_blocks": 2,
        "label_smoothing": 0.1,
        "patience": 8,
        "early_stopping_enabled": True,
    }

    validate_config(config)

    config["training"]["unfrozen_backbone_blocks"] = 0
    with pytest.raises(ValueError, match="unfrozen_backbone_blocks"):
        validate_config(config)


def test_config_accepts_exact_lrs_for_unfrozen_x3d_blocks(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["training"]["unfrozen_backbone_blocks"] = 2
    config["optimizer"]["backbone_block_lrs"] = {4: 3e-6, 5: 1e-5}

    validate_config(config)

    config["optimizer"]["backbone_block_lrs"] = {5: 1e-5}
    with pytest.raises(ValueError, match="exactly match"):
        validate_config(config)

    config["optimizer"]["backbone_block_lrs"] = {4: -1.0, 5: 1e-5}
    with pytest.raises(ValueError, match="positive"):
        validate_config(config)


def test_learning_rate_audit_distinguishes_active_and_frozen_scopes() -> None:
    frozen = nn.Parameter(torch.ones(1), requires_grad=False)
    block4 = nn.Parameter(torch.ones(1))
    block5 = nn.Parameter(torch.ones(1))
    head = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [
            {"params": [frozen], "lr": 3e-5, "group_name": "backbone_default"},
            {"params": [block4], "lr": 3e-6, "group_name": "backbone_block_4"},
            {"params": [block5], "lr": 1e-5, "group_name": "backbone_block_5"},
            {"params": [head], "lr": 3e-4, "group_name": "custom_head"},
        ]
    )

    assert _learning_rates_by_scope(optimizer, active_only=True) == {
        "backbone_block_4": pytest.approx(3e-6),
        "backbone_block_5": pytest.approx(1e-5),
        "custom_head": pytest.approx(3e-4),
    }
    assert _maximum_learning_rate(optimizer, scope_prefix="backbone_") == pytest.approx(
        1e-5
    )


def test_user_partition_rejects_heldout_and_overlap() -> None:
    official_train = {"u1", "u2", "u3"}
    heldout = {"u4"}
    with pytest.raises(ValueError, match="held-out"):
        validate_user_partition(("u1", "u4"), ("u2",), official_train, heldout)
    with pytest.raises(ValueError, match="overlap"):
        validate_user_partition(("u1", "u2"), ("u2", "u3"), official_train, heldout)


def test_oof_assignment_reuses_only_allowed_disjoint_users() -> None:
    assignment = {
        "folds": [
            {"fold": 0, "train_user_ids": ["u1", "u2"], "validation_user_ids": ["u3"]},
            {"fold": 1, "train_user_ids": ["u1", "u3"], "validation_user_ids": ["u2"]},
            {"fold": 2, "train_user_ids": ["u2", "u3"], "validation_user_ids": ["u1"]},
        ]
    }

    folds = validate_oof_assignment(assignment, allowed_users={"u1", "u2", "u3"})

    assert [fold.fold for fold in folds] == [0, 1, 2]
    assert {user for fold in folds for user in fold.validation_user_ids} == {"u1", "u2", "u3"}


def test_clip_budget_sampler_preserves_every_trial_within_limits() -> None:
    sampler = ClipBudgetBatchSampler(
        num_clips=[8, 1, 7, 2, 3],
        max_trials_per_batch=2,
        max_valid_clips_per_batch=8,
        shuffle=False,
        seed=7,
    )

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(5))
    assert all(len(batch) <= 2 for batch in batches)
    assert all(sum(sampler.num_clips[index] for index in batch) <= 8 for batch in batches)
    assert batches[0] == [0]


def test_accuracy_checkpoint_tie_breaks_by_macro_then_earlier_epoch() -> None:
    incumbent = (0.5, 0.4, 5)
    assert is_better_checkpoint((0.5, 0.41, 8), incumbent, objective="accuracy")
    assert not is_better_checkpoint((0.5, 0.4, 6), incumbent, objective="accuracy")


def test_cli_and_run_directory_reject_overwrite(tmp_path: Path) -> None:
    options = {action.dest for action in build_arg_parser()._actions}
    assert {
        "config",
        "smoke_test",
        "run_id",
        "epochs",
        "seed",
        "max_train_batches",
        "max_val_batches",
        "train_user_ids",
        "validation_user_ids",
        "oof_fold_assignment",
        "oof_role",
    } <= options
    prepare_run_directory(tmp_path, "new-run")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_run_directory(tmp_path, "new-run")


def test_smoke_runtime_reaches_backbone_unfreeze_with_one_microbatch() -> None:
    config = fixed_config(Path("."))
    args = argparse.Namespace(
        epochs=None,
        smoke_test=True,
        max_train_batches=None,
        max_val_batches=None,
        seed=None,
    )

    resolved, max_train_batches, max_val_batches = apply_runtime_overrides(config, args)

    assert resolved["training"]["epochs"] == 3
    assert max_train_batches == 1
    assert max_val_batches == 1


def test_cli_seed_overrides_yaml_and_changes_resolved_provenance_hash() -> None:
    config = fixed_config(Path("."))
    first_args = argparse.Namespace(
        epochs=None,
        smoke_test=False,
        max_train_batches=None,
        max_val_batches=None,
        seed=20260716,
    )
    second_args = argparse.Namespace(**{**vars(first_args), "seed": 20260717})

    first, _, _ = apply_runtime_overrides(config, first_args)
    second, _, _ = apply_runtime_overrides(config, second_args)

    assert first["seed"] == 20260716
    assert second["seed"] == 20260717
    assert resolved_config_sha256(first) != resolved_config_sha256(second)


def test_cosine_schedule_keeps_nonzero_lr_for_first_unfrozen_epoch() -> None:
    assert _warmup_cosine_multiplier(
        0, scheduler_horizon_epochs=3, warmup_epochs=2
    ) == pytest.approx(0.5)
    assert _warmup_cosine_multiplier(
        1, scheduler_horizon_epochs=3, warmup_epochs=2
    ) == pytest.approx(1.0)
    assert _warmup_cosine_multiplier(
        2, scheduler_horizon_epochs=3, warmup_epochs=2
    ) == pytest.approx(1.0)


def test_formal_refit_preserves_the_full_cosine_schedule_prefix(tmp_path: Path) -> None:
    selection_config = fixed_config(tmp_path)
    selection_config["training"] = {
        "epochs": 30,
        "scheduler_horizon_epochs": 30,
        "warmup_epochs": 2,
        "patience": 8,
        "early_stopping_enabled": False,
    }

    formal_config = _formal_refit_config(selection_config, selected_epoch=12)

    assert formal_config["training"]["epochs"] == 12
    assert formal_config["training"]["scheduler_horizon_epochs"] == 30
    selection_prefix = [
        _warmup_cosine_multiplier(index, scheduler_horizon_epochs=30, warmup_epochs=2)
        for index in range(12)
    ]
    formal_prefix = [
        _warmup_cosine_multiplier(
            index,
            scheduler_horizon_epochs=formal_config["training"]["scheduler_horizon_epochs"],
            warmup_epochs=2,
        )
        for index in range(12)
    ]
    assert formal_prefix == pytest.approx(selection_prefix)


def test_disabled_early_stopping_runs_the_full_inner_epoch_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixed_config(tmp_path)
    config["training"] = {
        "epochs": 3,
        "scheduler_horizon_epochs": 30,
        "warmup_epochs": 2,
        "patience": 1,
        "early_stopping_enabled": False,
    }
    config["amp"] = {"enabled": False, "dtype": "bfloat16"}
    config["deployment_artifacts"] = {"yolo_checkpoint": None}
    run_directory = tmp_path / "full-inner-search"
    run_directory.mkdir()
    monkeypatch.setattr(
        trainer_module,
        "is_better_checkpoint",
        lambda candidate, current, *, objective: current is None,
    )

    summary = train_partition(
        model=tiny_trainer_model(),
        train_dataset=TinyTrialDataset("train", 2),
        validation_dataset=TinyTrialDataset("validation", 2),
        config=config,
        run_directory=run_directory,
        device=torch.device("cpu"),
        max_train_batches=1,
        max_val_batches=1,
    )

    assert summary["epochs_completed"] == 3
    assert len(pd.read_csv(run_directory / "history.csv")) == 3


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    [(1, "<=13"), (13, "<=13"), (14, "14-32"), (32, "14-32"),
     (33, "33-64"), (64, "33-64"), (65, ">64"), (236, ">64")],
)
def test_length_bucket_matches_the_phase4_reporting_contract(
    num_frames: int, expected: str
) -> None:
    assert _length_bucket(num_frames) == expected


def test_eval_epoch_emits_one_prediction_per_trial() -> None:
    model = tiny_trainer_model()

    outcome = run_model_epoch(
        model,
        [model_batch()],
        device=torch.device("cpu"),
        optimizer=None,
        gradient_accumulation=1,
        gradient_clip=1.0,
        amp_enabled=False,
    )

    assert outcome.predictions.sample_ids == ("sample-a", "sample-b")
    assert outcome.predictions.output.main_logits.shape == (2, 40)
    torch.testing.assert_close(
        outcome.predictions.output.main_logits.exp().sum(dim=1),
        torch.ones(2),
    )
    assert outcome.metrics["sample_count"] == 2
    assert outcome.metrics["class_coverage"] == 2


def test_gradient_accumulation_weights_trials_equally_across_variable_microbatches() -> None:
    torch.manual_seed(7)
    accumulated_model = TrialLinearExpert()
    full_batch_model = TrialLinearExpert()
    full_batch_model.load_state_dict(accumulated_model.state_dict())
    microbatches = [
        trial_linear_batch([0.5], offset=0),
        trial_linear_batch([1.0, 1.5], offset=1),
        trial_linear_batch([2.0, 2.5], offset=3),
        trial_linear_batch([3.0, 3.5], offset=5),
    ]
    full_batch = trial_linear_batch([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], offset=0)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    full_batch_optimizer = torch.optim.SGD(full_batch_model.parameters(), lr=0.1)

    accumulated_outcome = run_model_epoch(
        accumulated_model,
        microbatches,
        device=torch.device("cpu"),
        optimizer=accumulated_optimizer,
        gradient_accumulation=4,
        gradient_clip=1e6,
        amp_enabled=False,
    )
    full_batch_outcome = run_model_epoch(
        full_batch_model,
        [full_batch],
        device=torch.device("cpu"),
        optimizer=full_batch_optimizer,
        gradient_accumulation=1,
        gradient_clip=1e6,
        amp_enabled=False,
    )

    assert torch.allclose(
        accumulated_model.classifier.weight,
        full_batch_model.classifier.weight,
        atol=1e-7,
        rtol=0.0,
    )
    assert accumulated_outcome.metrics["loss"] == pytest.approx(
        full_batch_outcome.metrics["loss"], abs=1e-6
    )
def test_train_epoch_updates_head_without_updating_bn_running_stats() -> None:
    model = tiny_trainer_model()
    model.set_backbone_trainable(True)
    optimizer = torch.optim.AdamW(model.parameter_groups(3e-5, 3e-4, 0.05))
    classifier_before = model.classifier.weight.detach().clone()
    backbone = model.backbone
    assert isinstance(backbone, TinyTrainerBackbone)
    running_mean_before = backbone.bn.running_mean.detach().clone()

    outcome = run_model_epoch(
        model,
        [model_batch("one"), model_batch("two")],
        device=torch.device("cpu"),
        optimizer=optimizer,
        gradient_accumulation=2,
        gradient_clip=1.0,
        amp_enabled=False,
    )

    assert outcome.metrics["sample_count"] == 4
    assert outcome.metrics["backbone_received_finite_gradient"] is True
    assert outcome.metrics["head_received_finite_gradient"] is True
    assert not torch.equal(model.classifier.weight, classifier_before)
    torch.testing.assert_close(backbone.bn.running_mean, running_mean_before, atol=0.0, rtol=0.0)


def test_train_epoch_records_scoped_block_and_classifier_gradients() -> None:
    model = X3DSVisualExpert(
        backbone=BlockTrainerBackbone(),
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
    )
    model.set_backbone_trainable(True, last_blocks=2)
    optimizer = torch.optim.AdamW(model.parameter_groups(3e-5, 3e-4, 0.05))

    outcome = run_model_epoch(
        model,
        [model_batch("scoped")],
        device=torch.device("cpu"),
        optimizer=optimizer,
        gradient_accumulation=1,
        gradient_clip=1.0,
        amp_enabled=False,
    )

    scopes = outcome.metrics["gradient_scopes_with_finite_nonzero"]
    assert scopes["backbone_block_2"] is True
    assert scopes["backbone_block_3"] is True
    assert scopes["classifier"] is True


def test_train_epoch_records_ir_anchored_adapter_gradient() -> None:
    model = X3DSVisualExpert(
        backbone=TinyTrainerBackbone(),
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
        input_channels=4,
        input_adapter_mode="ir_anchored_depth_residual",
    )
    optimizer = torch.optim.AdamW(
        model.parameter_groups(3e-5, 3e-4, 0.05, input_adapter_lr=3e-4)
    )
    batch = model_batch("adapter")
    batch["clips"] = torch.randn(2, 2, 1, 4, 13, 8, 8)

    outcome = run_model_epoch(
        model,
        [batch],
        device=torch.device("cpu"),
        optimizer=optimizer,
        gradient_accumulation=1,
        gradient_clip=1.0,
        amp_enabled=False,
    )

    assert outcome.metrics["gradient_scopes_with_finite_nonzero"]["input_adapter"] is True


def test_adapter_gradient_does_not_masquerade_as_head_gradient() -> None:
    model = X3DSVisualExpert(
        backbone=TinyTrainerBackbone(),
        num_classes=40,
        embedding_dim=16,
        dropout=0.0,
        input_channels=4,
        input_adapter_mode="ir_anchored_depth_residual",
    )
    model.set_backbone_trainable(False)
    for parameter in model.embedding_head.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(3e-5, 3e-4, 0.05, input_adapter_lr=3e-4)
    )
    batch = model_batch("adapter-only")
    batch["clips"] = torch.randn(2, 2, 1, 4, 13, 8, 8)

    outcome = run_model_epoch(
        model,
        [batch],
        device=torch.device("cpu"),
        optimizer=optimizer,
        gradient_accumulation=1,
        gradient_clip=1.0,
        amp_enabled=False,
    )

    assert outcome.metrics["head_received_finite_gradient"] is False
    assert outcome.metrics["gradient_scopes_with_finite_nonzero"]["input_adapter"] is True


def test_train_partition_writes_checkpoints_archives_and_summary(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["training"] = {"epochs": 1, "warmup_epochs": 2, "patience": 8}
    config["amp"] = {"enabled": False, "dtype": "bfloat16"}
    config["deployment_artifacts"] = {"yolo_checkpoint": None}
    run_directory = tmp_path / "partition"
    run_directory.mkdir()
    train_dataset = TinyTrialDataset("train", 4)
    val_dataset = TinyTrialDataset("val", 2)

    summary = train_partition(
        model=tiny_trainer_model(),
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        config=config,
        run_directory=run_directory,
        device=torch.device("cpu"),
        max_train_batches=None,
        max_val_batches=None,
    )

    assert train_dataset.epoch == 1
    for name in (
        "best_accuracy.pt",
        "best_macro_f1.pt",
        "val_predictions_best_accuracy.npz",
        "val_predictions_best_macro_f1.npz",
        "history.csv",
        "per_class_best_accuracy.csv",
        "per_class_best_macro_f1.csv",
        "run_summary.json",
    ):
        assert (run_directory / name).is_file(), name
    saved_summary = json.loads((run_directory / "run_summary.json").read_text(encoding="utf-8"))
    assert saved_summary == summary
    checkpoint = torch.load(
        run_directory / "best_accuracy.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["head_type"] == "projected"
    assert checkpoint["embedding_dim"] == 256
    assert summary["head_type"] == "projected"
    assert summary["embedding_dim"] == 8
    assert summary["prediction_archive_bytes"]["best_accuracy"] > 0
    assert summary["ir_route_provisional_size_gate_passed"] is True
    assert summary["best_accuracy"]["epoch"] == 1


def test_direct_partition_records_checkpoint_archive_and_resource_metadata(
    tmp_path: Path,
) -> None:
    config = fixed_config(tmp_path)
    config["head_type"] = "direct"
    config["embedding_dim"] = 2048
    config["training"] = {"epochs": 1, "warmup_epochs": 2, "patience": 8}
    config["amp"] = {"enabled": False, "dtype": "bfloat16"}
    config["deployment_artifacts"] = {"yolo_checkpoint": None}
    run_directory = tmp_path / "direct-partition"
    run_directory.mkdir()

    summary = train_partition(
        model=direct_trainer_model(),
        train_dataset=TinyTrialDataset("direct-train", 2),
        validation_dataset=TinyTrialDataset("direct-val", 2),
        config=config,
        run_directory=run_directory,
        device=torch.device("cpu"),
        max_train_batches=1,
        max_val_batches=1,
    )

    checkpoint = torch.load(
        run_directory / "best_accuracy.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["head_type"] == "direct"
    assert checkpoint["embedding_dim"] == 2048
    assert summary["head_type"] == "direct"
    assert summary["embedding_dim"] == 2048
    assert summary["custom_head_parameter_count"] == 2048 * 40 + 40
    assert summary["prediction_archive_bytes"]["best_accuracy"] > 0
    history = pd.read_csv(run_directory / "history.csv")
    scopes = json.loads(history.iloc[0]["train_gradient_scopes_with_finite_nonzero"])
    assert scopes["classifier"] is True
    assert summary["gradient_scopes_with_finite_nonzero"]["classifier"] is True
    with np.load(run_directory / "val_predictions_best_accuracy.npz") as archive:
        assert archive["head_type"].item() == "direct"
        assert int(archive["embedding_dim"].item()) == 2048


def test_train_partition_records_backbone_gradient_after_epoch_three(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["training"] = {"epochs": 3, "warmup_epochs": 2, "patience": 8}
    config["amp"] = {"enabled": False, "dtype": "bfloat16"}
    config["deployment_artifacts"] = {"yolo_checkpoint": None}
    run_directory = tmp_path / "gradient-smoke"
    run_directory.mkdir()

    summary = train_partition(
        model=tiny_trainer_model(),
        train_dataset=TinyTrialDataset("train", 2),
        validation_dataset=TinyTrialDataset("val", 2),
        config=config,
        run_directory=run_directory,
        device=torch.device("cpu"),
        max_train_batches=1,
        max_val_batches=1,
    )

    assert summary["head_gradient_verified"] is True
    assert summary["backbone_gradient_verified_after_unfreeze"] is True


def test_strict_oof_freezes_refit_checkpoint_before_outer_validation(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["training"] = {
        "epochs": 2,
        "scheduler_horizon_epochs": 30,
        "warmup_epochs": 2,
        "patience": 8,
        "early_stopping_enabled": False,
    }
    config["amp"] = {"enabled": False, "dtype": "bfloat16"}
    config["deployment_artifacts"] = {"yolo_checkpoint": None}
    run_directory = tmp_path / "strict-oof"
    run_directory.mkdir()

    class GuardedOuterValidation(TinyTrialDataset):
        def __getitem__(self, index: int) -> dict[str, object]:
            assert (run_directory / "formal_outer_refit.pt").is_file()
            return super().__getitem__(index)

    summary = train_strict_oof_partition(
        model_factory=tiny_trainer_model,
        inner_fit_dataset=TinyTrialDataset("inner-fit", 2),
        inner_validation_dataset=TinyTrialDataset("inner-validation", 2),
        outer_train_dataset=TinyTrialDataset("outer-train", 2),
        outer_validation_dataset=GuardedOuterValidation("outer-validation", 2),
        config=config,
        run_directory=run_directory,
        device=torch.device("cpu"),
        fold_provenance={
            "outer_fold": 0,
            "inner_fit_user_ids": ["u1"],
            "inner_validation_user_ids": ["u2"],
            "outer_train_user_ids": ["u1", "u2"],
            "outer_validation_user_ids": ["u3"],
            "assignment_sha256": "a" * 64,
        },
        max_train_batches=1,
        max_val_batches=1,
    )

    assert summary["selection_labels_from_outer_validation"] is False
    assert summary["formal_checkpoint"] == "formal_outer_refit.pt"
    assert summary["selected_epoch"] in {1, 2}
    assert summary["scheduler_horizon_epochs"] == 30
    formal_checkpoint = torch.load(
        run_directory / "formal_outer_refit.pt", map_location="cpu", weights_only=False
    )
    assert formal_checkpoint["strict_oof_provenance"]["scheduler_horizon_epochs"] == 30
    assert (run_directory / "formal_outer_predictions.npz").is_file()


def test_manual_selected_epoch_refit_is_frozen_before_outer_validation(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["training"] = {
        "epochs": 30,
        "scheduler_horizon_epochs": 30,
        "warmup_epochs": 2,
        "patience": 8,
        "early_stopping_enabled": False,
    }
    config["amp"] = {"enabled": False, "dtype": "bfloat16"}
    config["deployment_artifacts"] = {"yolo_checkpoint": None}
    run_directory = tmp_path / "manual-refit"
    run_directory.mkdir()

    class GuardedOuterValidation(TinyTrialDataset):
        def __getitem__(self, index: int) -> dict[str, object]:
            assert (run_directory / "formal_outer_refit.pt").is_file()
            return super().__getitem__(index)

    summary = refit_strict_oof_partition(
        model_factory=tiny_trainer_model,
        outer_train_dataset=TinyTrialDataset("outer-train", 2),
        outer_validation_dataset=GuardedOuterValidation("outer-validation", 2),
        config=config,
        run_directory=run_directory,
        device=torch.device("cpu"),
        selected_epoch=1,
        fold_provenance={"outer_fold": 1, "selection_policy": "manual_selected_epoch"},
        selection_summary={"status": "incomplete", "epochs_completed": 28},
        max_train_batches=1,
        max_val_batches=1,
    )

    assert summary["selected_epoch"] == 1
    checkpoint = torch.load(
        run_directory / "formal_outer_refit.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["strict_oof_provenance"]["selection_policy"] == "manual_selected_epoch"


def test_finalize_train14_trains_fixed_epochs_without_validation(tmp_path: Path) -> None:
    config = fixed_config(tmp_path)
    config["training"] = {"epochs": 1, "warmup_epochs": 2, "patience": 8}
    config["amp"] = {"enabled": False, "dtype": "bfloat16"}
    run_directory = tmp_path / "finalize"
    run_directory.mkdir()

    summary = finalize_train14(
        model=tiny_trainer_model(),
        train_dataset=TinyTrialDataset("train", 4),
        config=config,
        run_directory=run_directory,
        device=torch.device("cpu"),
        max_train_batches=None,
    )

    assert summary["role"] == "finalize_train14"
    assert summary["epochs_completed"] == 1
    assert (run_directory / "final_train14.pt").is_file()
    assert (run_directory / "finalize_history.csv").is_file()
    assert not list(run_directory.glob("val_predictions_*.npz"))


def test_finalize_manifest_requires_exactly_all_official_train_users() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "user_id": ["u1", "u2", "u3"],
            "split": ["ignored", "ignored", "ignored"],
        }
    )

    selected = prepare_finalize_manifest(frame, official_train_users={"u1", "u2", "u3"})

    assert set(selected["user_id"]) == {"u1", "u2", "u3"}
    assert set(selected["split"]) == {"train"}
    with pytest.raises(ValueError, match="missing official train-14"):
        prepare_finalize_manifest(frame.iloc[:2], official_train_users={"u1", "u2", "u3"})


def test_smoke_audit_counts_deployable_files_once_and_checks_fusion(tmp_path: Path) -> None:
    class_rows = pd.DataFrame(
        {"class_id": range(40), "action_name": [f"class-{index}" for index in range(40)]}
    )
    manifest = tmp_path / "manifest.csv"
    class_rows.assign(
        split="val",
        sample_id=[f"sample-{index}" for index in range(40)],
        user_id="u",
        source_frame_index=0,
    ).to_csv(manifest, index=False, encoding="utf-8-sig")
    expected_hash = class_map_hash(class_rows)
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    torch.save(
        {
            "model_state_dict": {
                "backbone.weight": torch.ones(3, 3),
                "embedding_head.weight": torch.ones(4, 3),
                "classifier.weight": torch.ones(40, 4),
            }
        },
        smoke / "best_accuracy.pt",
    )
    sample_ids = np.asarray(["sample-0", "sample-1"])
    logits = np.random.default_rng(7).normal(size=(2, 40)).astype(np.float32)
    archive = {
        "sample_ids": sample_ids,
        "user_ids": np.asarray(["u", "u"]),
        "labels": np.asarray([0, 1]),
        "logits": logits,
        "embeddings": np.ones((2, 256), dtype=np.float32),
        "quality": np.ones((2, 6), dtype=np.float32),
        "quality_mask": np.ones((2, 6), dtype=bool),
        "availability": np.ones((2, 1), dtype=bool),
        "class_map_hash": np.asarray(expected_hash),
        "num_frames": np.asarray([13, 33]),
        "num_clips": np.asarray([1, 2]),
    }
    for objective in ("best_accuracy", "best_macro_f1"):
        np.savez_compressed(smoke / f"val_predictions_{objective}.npz", **archive)
    (smoke / "run_summary.json").write_text(
        json.dumps({"val_samples_evaluated_last_epoch": 2}), encoding="utf-8"
    )
    yolo = tmp_path / "yolo.pt"
    yolo.write_bytes(b"pose-weights")
    config = {
        "input_manifest": str(manifest),
        "deployment_artifacts": {"yolo_checkpoint": str(yolo)},
        "size_gate": {"internal_limit_bytes": 95_000_000},
    }

    report = audit_smoke_run(config, smoke)

    assert report["status"] == "passed"
    assert report["deployable_file_count"] == 2
    assert report["custom_head_counted_twice"] is False
    assert report["validation_archive_complete"] is True
    assert report["alignment_gate_passed"] is True
    assert report["alpha_zero_gate_passed"] is True
    assert report["class_map_hash"] == expected_hash
