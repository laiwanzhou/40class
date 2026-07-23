from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.preprocess_imu_stage1 as stage1
from scripts.build_imu_training_index import build_class_order
from scripts.compute_imu_normalization import write_normalization_artifacts
from src.data.imu_stage2_contracts import (
    DataStatus,
    ImuPathNotDirectoryError,
    InferenceSample,
    MissingImuDirectoryError,
    NoRecognizableImuCsvError,
    NoUsableGridCellsError,
    NoValidStage1RecordsError,
    SequenceLengthSafetyError,
    Stage1DataValidationError,
    Stage2ActionResult,
    canonical_json_bytes,
    contract_sha256,
    sha256_file,
)
from src.data.imu_stage2_io import build_stage2_schema
from src.models.imu_stage2_tcn import build_checkpoint_metadata


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_raw_csv(path: Path, timestamp: str, base: float) -> None:
    row = [timestamp, "WTLL(device)", *[base + index for index in range(16)]]
    pd.DataFrame([row], columns=stage1.REQUIRED_SOURCE_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def _make_result(sample_id: str = "SM_test_0001") -> Stage2ActionResult:
    valid = np.asarray([[True, False, False, False, False], [True] * 5], dtype=bool)
    values = np.full((2, 5, 16), np.nan, dtype=np.float32)
    values[valid] = 1.0
    return Stage2ActionResult(
        sample_id=sample_id,
        values=values,
        sensor_mask=np.ones(5, dtype=bool),
        valid_mask=valid,
        timestamps_ms=np.asarray([0, 100], dtype=np.int64),
        qc={},
        status=DataStatus.SUCCESS,
    )


def test_discovery_keeps_missing_imu_samples_and_audits_ignored_entries(
    tmp_path: Path,
) -> None:
    from src.inference.imu_stage2_pipeline import discover_test_samples

    root = tmp_path / "test"
    root.mkdir()
    (root / "SM_test_0002").mkdir()
    (root / "SM_test_0001").mkdir()
    (root / ".claude").mkdir()
    (root / "notes.txt").write_text("ignored", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "SM_test_9999").mkdir()

    discovery = discover_test_samples(root)

    assert [sample.sample_id for sample in discovery.samples] == [
        "SM_test_0001",
        "SM_test_0002",
    ]
    assert all(sample.source_relative_path == Path(sample.sample_id) for sample in discovery.samples)
    assert discovery.ignored_entries == (".claude", "nested", "notes.txt")


def test_discovery_records_deterministic_relative_ignored_names_inside_samples(
    tmp_path: Path,
) -> None:
    from src.inference.imu_stage2_pipeline import discover_test_samples

    root = tmp_path / "test"
    imu = root / "SM_test_0001" / "IMU"
    imu.mkdir(parents=True)
    (root / ".claude").mkdir()
    (root / "SM_test_0001" / "notes.txt").write_text("ignored", encoding="utf-8")
    (imu / "readme.md").write_text("ignored", encoding="utf-8")
    (imu / "nested").mkdir()

    discovery = discover_test_samples(root)

    assert discovery.ignored_entries == (".claude",)
    assert discovery.sample_ignored_entries == (
        "SM_test_0001/IMU/nested",
        "SM_test_0001/IMU/readme.md",
        "SM_test_0001/notes.txt",
    )
    assert all("\\" not in name and not Path(name).is_absolute() for name in (
        *discovery.ignored_entries,
        *discovery.sample_ignored_entries,
    ))


def test_raw_adapter_naturally_sorts_direct_csv_before_stage1_ranking(
    tmp_path: Path,
) -> None:
    from src.data.imu_stage1_bridge import process_raw_imu_source
    from src.inference.imu_stage2_pipeline import adapt_raw_imu_source, discover_test_samples

    root = tmp_path / "test"
    imu = root / "SM_test_0001" / "IMU"
    imu.mkdir(parents=True)
    part10 = imu / "part10.csv"
    part2 = imu / "part2.csv"
    _write_raw_csv(part10, "2025-01-01 00:00:00.000000000", 10.0)
    _write_raw_csv(part2, "2025-01-01 00:00:00.000000000", 20.0)
    nested = imu / "nested"
    nested.mkdir()
    _write_raw_csv(nested / "part1.csv", "2025-01-01 00:00:00.000000000", 30.0)

    descriptor = discover_test_samples(root).samples[0]
    source = adapt_raw_imu_source(descriptor)

    assert [path.name for path in source.input_csv_files] == ["part2.csv", "part10.csv"]
    stage1_data = process_raw_imu_source(source)
    ranks = dict(
        zip(
            stage1_data.dataframe["source_file"].astype(str),
            stage1_data.dataframe["_source_file_rank"].astype(int),
            strict=True,
        )
    )
    assert ranks == {"part10.csv": 1, "part2.csv": 0}


def test_raw_adapter_deduplicates_repeated_discovery_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference.imu_stage2_pipeline import adapt_raw_imu_source, discover_test_samples

    root = tmp_path / "test"
    imu = root / "SM_test_0001" / "IMU"
    imu.mkdir(parents=True)
    csv_path = imu / "part2.csv"
    _write_raw_csv(csv_path, "2025-01-01 00:00:00.000000000", 10.0)
    original_iterdir = Path.iterdir

    def repeated_iterdir(path: Path):
        if path == imu:
            return iter((csv_path, csv_path))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", repeated_iterdir)
    descriptor = discover_test_samples(root).samples[0]

    source = adapt_raw_imu_source(descriptor)

    assert source.input_csv_files == (csv_path,)


def test_raw_adapter_reports_missing_non_directory_and_empty_imu(tmp_path: Path) -> None:
    from src.inference.imu_stage2_pipeline import adapt_raw_imu_source, discover_test_samples

    root = tmp_path / "test"
    missing = root / "SM_test_0001"
    not_directory = root / "SM_test_0002"
    empty = root / "SM_test_0003"
    missing.mkdir(parents=True)
    not_directory.mkdir()
    (not_directory / "IMU").write_text("not a directory", encoding="utf-8")
    (empty / "IMU").mkdir(parents=True)
    descriptors = {item.sample_id: item for item in discover_test_samples(root).samples}

    with pytest.raises(MissingImuDirectoryError):
        adapt_raw_imu_source(descriptors["SM_test_0001"])
    with pytest.raises(ImuPathNotDirectoryError):
        adapt_raw_imu_source(descriptors["SM_test_0002"])
    with pytest.raises(NoRecognizableImuCsvError):
        adapt_raw_imu_source(descriptors["SM_test_0003"])


def test_preprocess_runs_shared_stage1_and_stage2_cores_for_available_imu(
    tmp_path: Path,
) -> None:
    from src.inference.imu_stage2_pipeline import (
        discover_test_samples,
        preprocess_inference_sample,
    )

    root = tmp_path / "test"
    imu = root / "SM_test_0001" / "IMU"
    imu.mkdir(parents=True)
    _write_raw_csv(
        imu / "part2.csv",
        "2025-01-01 00:00:00.000000000",
        10.0,
    )

    sample = preprocess_inference_sample(discover_test_samples(root).samples[0])

    assert sample.sample_id == "SM_test_0001"
    assert sample.imu_available
    assert sample.modality_mask
    assert sample.imu_result is not None
    assert sample.imu_result.imu_usable


DEGRADABLE_ERRORS = (
    MissingImuDirectoryError,
    ImuPathNotDirectoryError,
    NoRecognizableImuCsvError,
    NoValidStage1RecordsError,
    Stage1DataValidationError,
    NoUsableGridCellsError,
    SequenceLengthSafetyError,
)


@pytest.mark.parametrize("error_type", DEGRADABLE_ERRORS)
def test_only_allowlisted_typed_errors_degrade_one_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))

    def fail(_descriptor: object) -> NoReturn:
        raise error_type("SM_test_0001", "safe failure")

    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", fail)
    sample = pipeline.preprocess_inference_sample(descriptor)

    assert sample == InferenceSample(
        sample_id="SM_test_0001",
        imu_result=None,
        imu_available=False,
        modality_mask=False,
    )


@pytest.mark.parametrize(
    "error",
    [
        AssertionError("assert"),
        IndexError("index"),
        KeyError("key"),
        MemoryError("memory"),
        ValueError("unknown value"),
        RuntimeError("unknown runtime"),
        Exception("unknown"),
    ],
)
def test_unknown_errors_escape_as_global_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))

    def fail(_descriptor: object) -> NoReturn:
        raise error

    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", fail)
    with pytest.raises(type(error), match=str(error).replace("unknown", "unknown")):
        pipeline.preprocess_inference_sample(descriptor)


def test_preprocess_converts_no_usable_stage2_result_to_typed_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))
    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", lambda _: object())
    monkeypatch.setattr(
        pipeline,
        "process_raw_imu_source",
        lambda _: SimpleNamespace(
            relative_time_ns=np.asarray([0, 100_000_000], dtype=np.int64)
        ),
    )
    result = _make_result()
    result.valid_mask[:] = False
    result.values[:] = np.nan
    monkeypatch.setattr(pipeline, "process_stage2_action", lambda *_args, **_kwargs: result)

    sample = pipeline.preprocess_inference_sample(descriptor)

    assert not sample.imu_available
    assert sample.imu_result is None
    assert not sample.modality_mask


@pytest.mark.parametrize(
    "error_type",
    [NoUsableGridCellsError, SequenceLengthSafetyError],
)
def test_stage2_degradation_preserves_successful_stage1_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))
    stage1_result = SimpleNamespace(
        relative_time_ns=np.asarray([0, 100_000_000], dtype=np.int64)
    )
    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", lambda _: object())
    monkeypatch.setattr(pipeline, "process_raw_imu_source", lambda _: stage1_result)

    def fail_stage2(*_args: object, **_kwargs: object) -> NoReturn:
        raise error_type("SM_test_0001", "typed stage 2 degradation")

    monkeypatch.setattr(pipeline, "process_stage2_action", fail_stage2)

    diagnostics = pipeline.preprocess_inference_sample_with_diagnostics(descriptor)

    assert diagnostics.sample == InferenceSample(
        sample_id="SM_test_0001",
        imu_result=None,
        imu_available=False,
        modality_mask=False,
    )
    assert diagnostics.source_status == "available"
    assert diagnostics.stage1_status == "success"
    assert diagnostics.stage2_status == "degraded"
    assert diagnostics.stage1_result is stage1_result
    assert diagnostics.stage2_result is None
    assert isinstance(diagnostics.degradation_error, error_type)
    assert diagnostics.degradation_error.safe_message == "typed stage 2 degradation"


def test_stage1_degradation_does_not_claim_stage_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))

    def fail_source(_descriptor: object) -> NoReturn:
        raise MissingImuDirectoryError("SM_test_0001", "missing")

    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", fail_source)
    diagnostics = pipeline.preprocess_inference_sample_with_diagnostics(descriptor)

    assert diagnostics.source_status == "unavailable"
    assert diagnostics.stage1_status == "unavailable"
    assert diagnostics.stage2_status == "unavailable"
    assert diagnostics.stage1_result is None
    assert diagnostics.stage2_result is None
    assert isinstance(diagnostics.degradation_error, MissingImuDirectoryError)


def test_stage2_invariant_failure_is_not_hidden_by_no_usable_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))
    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", lambda _: object())
    monkeypatch.setattr(
        pipeline,
        "process_raw_imu_source",
        lambda _: SimpleNamespace(
            relative_time_ns=np.asarray([0, 100_000_000], dtype=np.int64)
        ),
    )
    result = _make_result()
    result.valid_mask[:] = False
    result.values[:] = np.nan
    result.timestamps_ms[0] = 1
    monkeypatch.setattr(pipeline, "process_stage2_action", lambda *_args, **_kwargs: result)

    with pytest.raises(ValueError, match="timestamps_ms must start at 0"):
        pipeline.preprocess_inference_sample(descriptor)


def test_stage2_sample_identity_mismatch_is_a_global_contract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))
    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", lambda _: object())
    monkeypatch.setattr(
        pipeline,
        "process_raw_imu_source",
        lambda _: SimpleNamespace(
            relative_time_ns=np.asarray([0, 100_000_000], dtype=np.int64)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "process_stage2_action",
        lambda *_args, **_kwargs: _make_result("SM_test_9999"),
    )

    with pytest.raises(ValueError, match="sample ID"):
        pipeline.preprocess_inference_sample(descriptor)


def test_inference_plan_is_lightweight_and_materialized_length_must_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))
    stage1_result = SimpleNamespace(
        relative_time_ns=np.asarray([0], dtype=np.int64)
    )
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", lambda _: object())
    monkeypatch.setattr(pipeline, "process_raw_imu_source", lambda _: stage1_result)

    def materialize(*_args: object, **_kwargs: object) -> Stage2ActionResult:
        calls.append("stage2")
        return _make_result()

    monkeypatch.setattr(pipeline, "process_stage2_action", materialize)

    plan = pipeline.plan_inference_sample(descriptor)

    assert plan.planned_t == 1
    assert plan.stage1_result is stage1_result
    assert calls == []
    with pytest.raises(ValueError, match="length disagrees"):
        pipeline.materialize_inference_plan(plan)
    assert calls == ["stage2"]


def test_inference_plan_applies_hard_limit_before_stage2_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline
    from src.data.imu_stage2_contracts import TestSampleDescriptor as SampleDescriptor

    descriptor = SampleDescriptor("SM_test_0001", tmp_path, Path("SM_test_0001"))
    stage1_result = SimpleNamespace(
        relative_time_ns=np.asarray([0, 200_000_000], dtype=np.int64)
    )
    monkeypatch.setattr(pipeline, "adapt_raw_imu_source", lambda _: object())
    monkeypatch.setattr(pipeline, "process_raw_imu_source", lambda _: stage1_result)

    def forbidden_stage2(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("Stage 2 tensor materialization must not run")

    monkeypatch.setattr(pipeline, "process_stage2_action", forbidden_stage2)

    plan = pipeline.plan_inference_sample(descriptor, hard_safety_limit_t=2)
    diagnostics = pipeline.materialize_inference_plan(
        plan,
        hard_safety_limit_t=2,
    )

    assert plan.planned_t == 0
    assert isinstance(plan.degradation_error, SequenceLengthSafetyError)
    assert diagnostics.stage1_status == "success"
    assert diagnostics.stage2_status == "degraded"
    assert isinstance(diagnostics.degradation_error, SequenceLengthSafetyError)
    assert diagnostics.stage2_result is None


def test_inference_collate_uses_nonpersistent_zero_length_placeholder() -> None:
    from src.inference.imu_stage2_pipeline import collate_inference_samples

    available = InferenceSample("SM_test_0001", _make_result(), True, True)
    unavailable = InferenceSample("SM_test_0002", None, False, False)

    batch = collate_inference_samples([available, unavailable])

    assert batch["values"].shape == (2, 2, 5, 16)
    assert batch["lengths"].tolist() == [2, 0]
    assert batch["sequence_mask"].tolist() == [[True, True], [False, False]]
    assert batch["imu_modality_mask"].tolist() == [True, False]
    assert not batch["valid_mask"][1].any()
    assert not batch["sensor_mask"][1].any()
    assert not batch["usable_sensor_mask"][1].any()
    assert torch.equal(batch["timestamps_ms"][1], torch.full((2,), -1, dtype=torch.int64))
    assert torch.equal(batch["values"][1], torch.zeros((2, 5, 16), dtype=torch.float32))


BUNDLE_ROLES = (
    "checkpoint",
    "model_config",
    "stage2_schema",
    "normalization_npz",
    "normalization_json",
    "class_order",
    "submission_contract",
    "inference_config",
)


def _bundle_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    schema = build_stage2_schema(
        {
            "implementation_version": "imu-stage2-v1",
            "generator_script": "scripts/preprocess_imu_stage2.py",
            "git_commit": "a" * 40,
            "created_at": "2026-07-22T00:00:00+00:00",
            "source_stage1_manifest": "manifest.csv",
            "source_stage1_manifest_sha256": "b" * 64,
        }
    )

    training_index_sha256 = "c" * 64
    statistics = {
        "count": np.ones((5, 16), dtype=np.int64),
        "mean": np.zeros((5, 16), dtype=np.float64),
        "raw_std": np.ones((5, 16), dtype=np.float64),
        "applied_scale": np.ones((5, 16), dtype=np.float64),
        "near_constant_mask": np.zeros((5, 16), dtype=bool),
        "minimum": np.zeros((5, 16), dtype=np.float64),
        "maximum": np.ones((5, 16), dtype=np.float64),
    }
    normalization = write_normalization_artifacts(
        statistics,
        bundle,
        stage2_contract_sha256=str(schema["contract_sha256"]),
        training_index_sha256=training_index_sha256,
        train_sample_id_sha256="d" * 64,
        fold=0,
        train_users=["user01"],
        source_stage2_manifest_sha256="e" * 64,
    )
    _write_json(bundle / "schema.json", schema)

    class_order = build_class_order(
        pd.DataFrame(
            [
                {"class_id": 2, "class_name": "two"},
                {"class_id": 10, "class_name": "ten"},
            ]
        )
    )
    _write_json(bundle / "class_order.json", class_order.to_payload())
    submission_contract = {
        "contract": {
            "submission_contract_version": "imu-submission-v1",
            "columns": ["sample_id", "label_index"],
        }
    }
    submission_contract["submission_contract_sha256"] = contract_sha256(
        submission_contract["contract"]
    )
    _write_json(bundle / "submission_contract.json", submission_contract)

    model_config = {
        "embedding_dim": 8,
        "tcn_channels": [8],
        "dropout": 0.0,
        "imu_modality_dropout": 0.1,
    }
    (bundle / "model_config.yaml").write_text(
        yaml.safe_dump(model_config, sort_keys=True), encoding="utf-8"
    )
    inference_config = {
        "inference_config_version": "imu-stage2-inference-v1",
        "hard_safety_limit_t": 10_000,
        "inference_seed": 20260715,
    }
    (bundle / "inference_config.yaml").write_text(
        yaml.safe_dump(inference_config, sort_keys=True), encoding="utf-8"
    )
    checkpoint_metadata = build_checkpoint_metadata(
        stage2_contract_sha256=str(schema["contract_sha256"]),
        training_index_sha256=training_index_sha256,
        normalization_contract_sha256=str(normalization["normalization_contract_sha256"]),
        normalization_file_sha256=str(normalization["normalization_file_sha256"]),
        class_order_sha256=class_order.class_order_sha256,
        submission_contract_sha256=str(
            submission_contract["submission_contract_sha256"]
        ),
        num_classes=class_order.num_classes,
    )
    torch.save(
        {"checkpoint_metadata": checkpoint_metadata, "model_state_dict": {}},
        bundle / "checkpoint.pt",
    )
    role_paths = {
        "checkpoint": "checkpoint.pt",
        "model_config": "model_config.yaml",
        "stage2_schema": "schema.json",
        "normalization_npz": "imu_normalization.npz",
        "normalization_json": "imu_normalization.json",
        "class_order": "class_order.json",
        "submission_contract": "submission_contract.json",
        "inference_config": "inference_config.yaml",
    }
    manifest: dict[str, object] = {
        "bundle_manifest_version": "imu-inference-bundle-v1",
        "files": {
            role: {"path": relative, "sha256": sha256_file(bundle / relative)}
            for role, relative in role_paths.items()
        },
    }
    _write_json(bundle / "inference_bundle_manifest.json", manifest)
    return bundle, manifest


def _refresh_bundle_hash(bundle: Path, manifest: dict[str, object], role: str) -> None:
    files = manifest["files"]
    assert isinstance(files, dict)
    entry = files[role]
    assert isinstance(entry, dict)
    entry["sha256"] = sha256_file(bundle / str(entry["path"]))
    _write_json(bundle / "inference_bundle_manifest.json", manifest)


def test_bundle_validation_succeeds_and_returns_verified_bindings(tmp_path: Path) -> None:
    from src.inference.imu_stage2_pipeline import load_inference_bundle

    bundle, _ = _bundle_fixture(tmp_path)

    loaded = load_inference_bundle(bundle)

    assert loaded.root == bundle.resolve()
    assert tuple(loaded.paths) == BUNDLE_ROLES
    assert loaded.checkpoint_metadata["num_classes"] == 2
    assert loaded.model_config["embedding_dim"] == 8
    assert loaded.inference_config["hard_safety_limit_t"] == 10_000
    assert loaded.normalization_arrays["mean"].shape == (5, 16)


def test_bundle_file_role_json_order_is_not_contractual(tmp_path: Path) -> None:
    from src.inference.imu_stage2_pipeline import load_inference_bundle

    bundle, manifest = _bundle_fixture(tmp_path)
    files = manifest["files"]
    assert isinstance(files, dict)
    manifest["files"] = dict(reversed(list(files.items())))
    _write_json(bundle / "inference_bundle_manifest.json", manifest)

    loaded = load_inference_bundle(bundle)

    assert tuple(loaded.paths) == BUNDLE_ROLES


@pytest.mark.parametrize("role", BUNDLE_ROLES)
def test_bundle_rejects_any_file_byte_tamper_before_checkpoint_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline

    bundle, manifest = _bundle_fixture(tmp_path)
    files = manifest["files"]
    assert isinstance(files, dict) and isinstance(files[role], dict)
    path = bundle / str(files[role]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")

    def must_not_load(_path: Path) -> NoReturn:
        raise AssertionError("checkpoint loaded before all bundle hashes passed")

    monkeypatch.setattr(pipeline, "_load_checkpoint_metadata", must_not_load)
    with pytest.raises(ValueError, match="SHA-256"):
        pipeline.load_inference_bundle(bundle)


def test_bundle_rejects_escaping_relative_path(tmp_path: Path) -> None:
    from src.inference.imu_stage2_pipeline import load_inference_bundle

    bundle, manifest = _bundle_fixture(tmp_path)
    files = manifest["files"]
    assert isinstance(files, dict) and isinstance(files["model_config"], dict)
    files["model_config"]["path"] = "../model_config.yaml"
    _write_json(bundle / "inference_bundle_manifest.json", manifest)

    with pytest.raises(ValueError, match="relative POSIX"):
        load_inference_bundle(bundle)


@pytest.mark.parametrize(
    "binding",
    [
        "stage2_contract",
        "normalization_file",
        "class_order",
        "submission_contract",
        "training_index",
    ],
)
def test_bundle_rejects_internal_binding_tamper(
    tmp_path: Path,
    binding: str,
) -> None:
    from src.inference.imu_stage2_pipeline import load_inference_bundle

    bundle, manifest = _bundle_fixture(tmp_path)
    if binding == "stage2_contract":
        payload = json.loads((bundle / "schema.json").read_text(encoding="utf-8"))
        payload["contract"]["grid_frequency_hz"] = 20
        _write_json(bundle / "schema.json", payload)
        _refresh_bundle_hash(bundle, manifest, "stage2_schema")
    elif binding == "normalization_file":
        payload = json.loads(
            (bundle / "imu_normalization.json").read_text(encoding="utf-8")
        )
        payload["normalization_file_sha256"] = "0" * 64
        _write_json(bundle / "imu_normalization.json", payload)
        _refresh_bundle_hash(bundle, manifest, "normalization_json")
    elif binding == "class_order":
        payload = json.loads((bundle / "class_order.json").read_text(encoding="utf-8"))
        payload["num_classes"] = 3
        _write_json(bundle / "class_order.json", payload)
        _refresh_bundle_hash(bundle, manifest, "class_order")
    elif binding == "submission_contract":
        payload = json.loads(
            (bundle / "submission_contract.json").read_text(encoding="utf-8")
        )
        payload["contract"]["columns"] = ["wrong"]
        _write_json(bundle / "submission_contract.json", payload)
        _refresh_bundle_hash(bundle, manifest, "submission_contract")
    else:
        checkpoint = torch.load(bundle / "checkpoint.pt", map_location="cpu", weights_only=True)
        checkpoint["checkpoint_metadata"]["training_index_sha256"] = "0" * 64
        torch.save(checkpoint, bundle / "checkpoint.pt")
        _refresh_bundle_hash(bundle, manifest, "checkpoint")

    with pytest.raises(ValueError, match="mismatch|incompatible|does not match"):
        load_inference_bundle(bundle)
