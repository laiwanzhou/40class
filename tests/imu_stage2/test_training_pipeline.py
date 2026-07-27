from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG = REPOSITORY_ROOT / "configs" / "imu_stage2_tcn_fold0.json"


def _model_config() -> dict[str, object]:
    return {
        "embedding_dim": 16,
        "tcn_channels": [8, 12],
        "dropout": 0.0,
        "imu_modality_dropout": 0.0,
    }


def _batch(*, finite: bool = True) -> dict[str, object]:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn((4, 6, 5, 16), generator=generator)
    if not finite:
        values[0, 0, 0, 0] = float("nan")
    valid_mask = torch.ones((4, 6, 5), dtype=torch.bool)
    return {
        "values": values,
        "valid_mask": valid_mask,
        "sequence_mask": torch.ones((4, 6), dtype=torch.bool),
        "sensor_mask": torch.ones((4, 5), dtype=torch.bool),
        "usable_sensor_mask": torch.ones((4, 5), dtype=torch.bool),
        "timestamps_ms": torch.arange(6).repeat(4, 1) * 100,
        "lengths": torch.full((4,), 6, dtype=torch.int64),
        "sample_id": ["s0", "s1", "s2", "s3"],
        "imu_modality_mask": torch.ones(4, dtype=torch.bool),
        "labels": torch.tensor([0, 1, 2, 1], dtype=torch.int64),
    }


def _training_metadata() -> dict[str, object]:
    from src.models.imu_stage2_tcn import build_training_checkpoint_metadata

    return build_training_checkpoint_metadata(
        stage2_contract_sha256="a" * 64,
        training_index_sha256="b" * 64,
        normalization_contract_sha256="c" * 64,
        normalization_file_sha256="d" * 64,
        class_order_sha256="e" * 64,
        num_classes=3,
    )


def test_training_metadata_is_strictly_separate_from_submission_metadata() -> None:
    from src.models.imu_stage2_tcn import (
        build_checkpoint_metadata,
        build_training_checkpoint_metadata,
    )

    metadata = _training_metadata()

    assert metadata == {
        "checkpoint_metadata_version": "imu-training-checkpoint-v1",
        "stage2_contract_sha256": "a" * 64,
        "training_index_sha256": "b" * 64,
        "normalization_contract_sha256": "c" * 64,
        "normalization_file_sha256": "d" * 64,
        "class_order_sha256": "e" * 64,
        "num_classes": 3,
    }
    assert "submission_contract_sha256" not in metadata
    with pytest.raises(TypeError):
        build_training_checkpoint_metadata(  # type: ignore[call-arg]
            **metadata,
            submission_contract_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="submission_contract_sha256"):
        build_checkpoint_metadata(
            stage2_contract_sha256="a" * 64,
            training_index_sha256="b" * 64,
            normalization_contract_sha256="c" * 64,
            normalization_file_sha256="d" * 64,
            class_order_sha256="e" * 64,
            submission_contract_sha256="",
            num_classes=3,
        )


def test_training_config_is_exact_and_rejects_unknown_or_changed_fixed_values(
    tmp_path: Path,
) -> None:
    from src.training.imu_stage2_trainer import load_training_config

    payload = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
    assert load_training_config(TRAINING_CONFIG) == payload

    for name, value in (("unknown", 1), ("num_classes", 39), ("optimizer", "Adam")):
        changed = dict(payload)
        changed[name] = value
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=name):
            load_training_config(path)


def test_training_artifact_bindings_reject_cross_contract_or_class_order_mix() -> None:
    from src.training.imu_stage2_trainer import validate_training_artifact_bindings

    metadata = {
        "stage2_contract_sha256": "a" * 64,
        "class_order_sha256": "b" * 64,
        "num_classes": 40,
    }
    validate_training_artifact_bindings(
        metadata,
        stage2_contract_sha256="a" * 64,
        class_order_sha256="b" * 64,
        num_classes=40,
    )
    for field, value in (
        ("stage2_contract_sha256", "c" * 64),
        ("class_order_sha256", "c" * 64),
        ("num_classes", 39),
    ):
        changed = dict(metadata)
        changed[field] = value
        with pytest.raises(ValueError, match=field):
            validate_training_artifact_bindings(
                changed,
                stage2_contract_sha256="a" * 64,
                class_order_sha256="b" * 64,
                num_classes=40,
            )


def test_metrics_handle_zero_denominators_and_use_true_rows_predicted_columns() -> None:
    from src.training.imu_stage2_trainer import classification_metrics

    result = classification_metrics(
        labels=np.array([0, 0, 1], dtype=np.int64),
        predictions=np.array([0, 1, 1], dtype=np.int64),
        num_classes=3,
    )

    np.testing.assert_array_equal(
        result["confusion_matrix"],
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64),
    )
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_precision"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert result["per_class"][2] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)


def test_train_one_epoch_updates_parameters_and_reports_finite_values() -> None:
    from src.models.imu_stage2_tcn import build_imu_stage2_model
    from src.training.imu_stage2_trainer import train_one_epoch

    torch.manual_seed(3)
    model = build_imu_stage2_model(_model_config(), num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    batch = _batch()
    model.train()
    with torch.no_grad():
        expected_accuracy = float(
            (model(batch)["logits"].argmax(dim=1) == batch["labels"])
            .to(torch.float32)
            .mean()
        )

    result = train_one_epoch(
        model,
        [batch],
        optimizer,
        device=torch.device("cpu"),
        label_smoothing=0.05,
        gradient_clip_norm=1.0,
        fail_fast_first_batch=True,
    )

    assert np.isfinite(result["loss"])
    assert np.isfinite(result["gradient_norm"])
    assert result["accuracy"] == pytest.approx(expected_accuracy)
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters())
    )


@pytest.mark.parametrize("bad_kind", ["input", "loss", "gradient"])
def test_first_batch_non_finite_values_fail_fast(
    bad_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.models.imu_stage2_tcn import build_imu_stage2_model
    from src.training import imu_stage2_trainer

    model = build_imu_stage2_model(_model_config(), num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _batch(finite=bad_kind != "input")
    if bad_kind == "loss":
        monkeypatch.setattr(
            imu_stage2_trainer,
            "training_loss",
            lambda *args, **kwargs: torch.tensor(float("inf"), requires_grad=True),
        )
    elif bad_kind == "gradient":
        parameter = next(model.parameters())
        parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))

    with pytest.raises(FloatingPointError, match=bad_kind):
        imu_stage2_trainer.train_one_epoch(
            model,
            [batch],
            optimizer,
            device=torch.device("cpu"),
            label_smoothing=0.05,
            gradient_clip_norm=1.0,
            fail_fast_first_batch=True,
        )


def test_evaluate_preserves_parameters_and_exports_finite_logits_and_embeddings() -> None:
    from src.models.imu_stage2_tcn import build_imu_stage2_model
    from src.training.imu_stage2_trainer import evaluate

    model = build_imu_stage2_model(_model_config(), num_classes=3)
    before = copy.deepcopy(model.state_dict())

    result = evaluate(model, [_batch()], device=torch.device("cpu"), num_classes=3)

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert result["logits"].shape == (4, 3)
    assert result["embeddings"].shape == (4, 16)
    assert np.isfinite(result["logits"]).all()
    assert np.isfinite(result["embeddings"]).all()
    assert result["sample_ids"] == ["s0", "s1", "s2", "s3"]


def test_best_order_and_early_stopping_are_deterministic() -> None:
    from src.training.imu_stage2_trainer import EarlyStopping, is_better_validation

    base = {"macro_f1": 0.5, "accuracy": 0.6, "loss": 1.0, "epoch": 3}
    assert is_better_validation({**base, "macro_f1": 0.6, "epoch": 9}, base)
    assert is_better_validation({**base, "accuracy": 0.7, "epoch": 9}, base)
    assert is_better_validation({**base, "loss": 0.9, "epoch": 9}, base)
    assert is_better_validation({**base, "epoch": 2}, base)
    assert not is_better_validation({**base, "epoch": 4}, base)

    stopper = EarlyStopping(patience=2)
    assert not stopper.observe(base)
    assert not stopper.observe({**base, "epoch": 4})
    assert stopper.observe({**base, "epoch": 5})


def test_checkpoint_round_trip_validates_training_metadata(tmp_path: Path) -> None:
    from src.models.imu_stage2_tcn import build_imu_stage2_model
    from src.training.imu_stage2_trainer import load_checkpoint, save_checkpoint

    model = build_imu_stage2_model(_model_config(), num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    path = tmp_path / "checkpoint.pt"
    metadata = _training_metadata()
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        metrics={"macro_f1": 0.5},
        metadata=metadata,
        config={"seed": 7},
    )

    restored = build_imu_stage2_model(_model_config(), num_classes=3)
    payload = load_checkpoint(path, model=restored, expected_metadata=metadata)
    assert payload["epoch"] == 1
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)

    invalid = dict(metadata)
    invalid["training_index_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(path, model=restored, expected_metadata=invalid)


def test_validation_output_export_has_exact_arrays(tmp_path: Path) -> None:
    from src.training.imu_stage2_trainer import write_validation_outputs

    path = tmp_path / "validation_outputs.npz"
    write_validation_outputs(
        path,
        sample_ids=["s0", "s1"],
        labels=np.array([0, 1], dtype=np.int64),
        logits=np.ones((2, 3), dtype=np.float32),
        embeddings=np.ones((2, 128), dtype=np.float32),
    )

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"sample_ids", "labels", "logits", "embeddings"}
        assert archive["sample_ids"].tolist() == ["s0", "s1"]
        assert archive["labels"].dtype == np.int64
        assert archive["logits"].shape == (2, 3)
        assert archive["embeddings"].shape == (2, 128)


def test_output_transaction_publishes_once_and_cleans_failure(tmp_path: Path) -> None:
    from src.training.imu_stage2_trainer import staged_output_directory

    output = tmp_path / "run"
    with staged_output_directory(output) as staging:
        (staging / "marker.txt").write_text("ok", encoding="utf-8")
    assert (output / "marker.txt").read_text(encoding="utf-8") == "ok"
    assert not list(tmp_path.glob(".run.staging-*"))

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        with staged_output_directory(failed) as staging:
            (staging / "partial.txt").write_text("bad", encoding="utf-8")
            raise RuntimeError("injected")
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))


def test_fit_model_publishes_best_last_metrics_and_validation_outputs(
    tmp_path: Path,
) -> None:
    from src.models.imu_stage2_tcn import build_imu_stage2_model
    from src.training.imu_stage2_trainer import fit_model

    model_config = _model_config()
    model_config["embedding_dim"] = 128
    model = build_imu_stage2_model(model_config, num_classes=3)
    config = {
        "seed": 7,
        "maximum_epochs": 2,
        "early_stopping_patience": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "gradient_clip_norm": 1.0,
    }
    output = tmp_path / "run"

    summary = fit_model(
        model=model,
        train_loader=[_batch()],
        validation_loader=[_batch()],
        output_dir=output,
        config=config,
        metadata=_training_metadata(),
        device=torch.device("cpu"),
    )

    assert summary["status"] == "success"
    assert set(path.name for path in output.iterdir()) == {
        "best.pt",
        "last.pt",
        "metrics.json",
        "validation_outputs.npz",
    }
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_epoch"] in {1, 2}
    with np.load(output / "validation_outputs.npz", allow_pickle=False) as archive:
        assert archive["logits"].shape == (4, 3)
        assert archive["embeddings"].shape == (4, 128)


def test_cli_help_works_outside_repository_and_preflight_creates_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = REPOSITORY_ROOT / "scripts" / "train_imu_stage2.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={key: value for key, value in dict(__import__("os").environ).items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--preflight-only" in help_result.stdout

    from scripts import train_imu_stage2

    output = tmp_path / "must-not-exist"
    expected = {"status": "preflight_ok", "train_samples": 2, "validation_samples": 1}
    monkeypatch.setattr(train_imu_stage2, "preflight_training", lambda **kwargs: expected)
    code = train_imu_stage2.main(
        [
            "--config", str(TRAINING_CONFIG),
            "--stage2-root", str(tmp_path / "stage2"),
            "--training-index-dir", str(tmp_path / "index"),
            "--normalization-dir", str(tmp_path / "normalization"),
            "--output-dir", str(output),
            "--preflight-only",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert not output.exists()
