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
    run_model_epoch,
    save_prediction_archive,
    train_partition,
    validate_config,
    validate_oof_assignment,
    validate_user_partition,
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


def tiny_trainer_model() -> X3DSVisualExpert:
    return X3DSVisualExpert(
        backbone=TinyTrainerBackbone(),
        num_classes=40,
        embedding_dim=8,
        dropout=0.0,
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
    )

    resolved, max_train_batches, max_val_batches = apply_runtime_overrides(config, args)

    assert resolved["training"]["epochs"] == 3
    assert max_train_batches == 1
    assert max_val_batches == 1


def test_cosine_schedule_keeps_nonzero_lr_for_first_unfrozen_epoch() -> None:
    assert _warmup_cosine_multiplier(0, epochs=3, warmup_epochs=2) == pytest.approx(0.5)
    assert _warmup_cosine_multiplier(1, epochs=3, warmup_epochs=2) == pytest.approx(1.0)
    assert _warmup_cosine_multiplier(2, epochs=3, warmup_epochs=2) == pytest.approx(1.0)


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
    assert summary["ir_route_provisional_size_gate_passed"] is True
    assert summary["best_accuracy"]["epoch"] == 1


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
