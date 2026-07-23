from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import weakref
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.preprocess_imu_stage1 as stage1
from scripts.build_imu_training_index import build_class_order
from scripts.compute_imu_normalization import write_normalization_artifacts
from src.data.imu_stage2_contracts import contract_sha256, sha256_file
from src.data.imu_stage2_io import build_stage2_schema
from src.models import build_checkpoint_metadata, build_imu_stage2_model


EXPECTED_INFERENCE_CONFIG = {
    "config_version": "imu-stage2-inference-v1",
    "hard_safety_limit_t": 10_000,
    "inference_seed": 20260715,
    "deterministic_algorithms": True,
    "batch_feature_budget": 327680,
    "maximum_batch_size": 16,
    "model_output_type": "logits",
    "prediction_rule": "argmax",
    "imu_unavailable_policy": "packaged_null_embedding",
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_raw_csv(path: Path, timestamp: str, base: float) -> None:
    row = [timestamp, "WTLL(device)", *[base + index for index in range(16)]]
    pd.DataFrame([row], columns=stage1.REQUIRED_SOURCE_COLUMNS).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _write_raw_csv_rows(path: Path, timestamps: list[str], base: float) -> None:
    rows = [
        [timestamp, "WTLL(device)", *[base + index for index in range(16)]]
        for timestamp in timestamps
    ]
    pd.DataFrame(rows, columns=stage1.REQUIRED_SOURCE_COLUMNS).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _source_artifacts(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    source = tmp_path / "source"
    source.mkdir()
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
        source,
        stage2_contract_sha256=str(schema["contract_sha256"]),
        training_index_sha256=training_index_sha256,
        train_sample_id_sha256="d" * 64,
        fold=0,
        train_users=["user01"],
        source_stage2_manifest_sha256="e" * 64,
    )
    _write_json(source / "schema.json", schema)
    class_order = build_class_order(
        pd.DataFrame(
            [
                {"class_id": 2, "class_name": "two"},
                {"class_id": 10, "class_name": "ten"},
            ]
        )
    )
    _write_json(source / "class_order.json", class_order.to_payload())
    model_config = {
        "embedding_dim": 8,
        "tcn_channels": [8],
        "dropout": 0.0,
        "imu_modality_dropout": 0.1,
    }
    (source / "model_config.yaml").write_text(
        yaml.safe_dump(model_config, sort_keys=True), encoding="utf-8"
    )
    (source / "inference_config.yaml").write_text(
        yaml.safe_dump(EXPECTED_INFERENCE_CONFIG, sort_keys=False), encoding="utf-8"
    )
    sample_submission = source / "sample_submission.csv"
    sample_submission.write_text(
        "sample_id,class_id\nSM_test_0001,0\nSM_test_0002,0\n", encoding="utf-8"
    )
    from src.inference.imu_stage2_pipeline import derive_submission_contract

    submission = derive_submission_contract(sample_submission)
    model = build_imu_stage2_model(model_config, num_classes=class_order.num_classes)
    metadata = build_checkpoint_metadata(
        stage2_contract_sha256=str(schema["contract_sha256"]),
        training_index_sha256=training_index_sha256,
        normalization_contract_sha256=str(normalization["normalization_contract_sha256"]),
        normalization_file_sha256=str(normalization["normalization_file_sha256"]),
        class_order_sha256=class_order.class_order_sha256,
        submission_contract_sha256=str(submission["submission_contract_sha256"]),
        num_classes=class_order.num_classes,
    )
    torch.save(
        {"checkpoint_metadata": metadata, "model_state_dict": model.state_dict()},
        source / "checkpoint.pt",
    )
    return {
        "checkpoint": source / "checkpoint.pt",
        "model_config": source / "model_config.yaml",
        "stage2_schema": source / "schema.json",
        "normalization_npz": source / "imu_normalization.npz",
        "normalization_json": source / "imu_normalization.json",
        "class_order": source / "class_order.json",
        "inference_config": source / "inference_config.yaml",
    }, sample_submission


def _build_bundle(tmp_path: Path) -> Path:
    from scripts.build_imu_inference_bundle import main

    artifacts, sample_submission = _source_artifacts(tmp_path)
    output = tmp_path / "bundle"
    argv: list[str] = []
    for role, path in artifacts.items():
        argv.extend(["--" + role.replace("_", "-"), str(path)])
    argv.extend(["--sample-submission", str(sample_submission), "--output-dir", str(output)])
    assert main(argv) == 0
    return output


def test_tracked_inference_config_is_exact_and_path_free() -> None:
    path = Path("configs/task03/imu_stage2_inference_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload == EXPECTED_INFERENCE_CONFIG
    assert all(type(payload[key]) is type(value) for key, value in EXPECTED_INFERENCE_CONFIG.items())
    assert not any("path" in key or "root" in key for key in payload)


def test_submission_contract_logits_and_decoding(tmp_path: Path) -> None:
    from src.inference.imu_stage2_pipeline import (
        decode_predictions,
        derive_submission_contract,
        validate_logits,
    )

    template = tmp_path / "sample_submission.csv"
    template.write_text("sample_id,class_id\nb,0\na,0\n", encoding="utf-8")
    payload = derive_submission_contract(template)
    contract = payload["contract"]
    assert contract == {
        "submission_contract_version": "imu-submission-v1",
        "columns": ["sample_id", "class_id"],
        "sample_id_column": "sample_id",
        "prediction_column": "class_id",
        "encoding": "utf-8",
        "header": True,
        "row_order": "sample_submission",
        "sample_ids": ["b", "a"],
        "prediction_representation": "class_id",
    }
    assert payload["submission_contract_sha256"] == contract_sha256(contract)

    logits = torch.tensor([[1.0, 1.0], [-1.0, 2.0]])
    validated = validate_logits(logits, batch_size=2, num_classes=2)
    class_order = build_class_order(
        pd.DataFrame([{"class_id": 2, "class_name": "two"}, {"class_id": 10, "class_name": "ten"}])
    )
    assert decode_predictions(validated, class_order, contract) == [2, 10]
    with pytest.raises(ValueError, match="shape"):
        validate_logits(torch.ones(2, 2, 1), batch_size=2, num_classes=2)
    with pytest.raises(ValueError, match="finite"):
        validate_logits(torch.tensor([[float("nan"), 0.0]]), batch_size=1, num_classes=2)
    with pytest.raises(ValueError, match="floating-point"):
        validate_logits(torch.tensor([[1, 0]]), batch_size=1, num_classes=2)


def test_bundle_builder_copies_eight_managed_files_and_reopens(tmp_path: Path) -> None:
    from scripts.build_imu_inference_bundle import main
    from src.inference.imu_stage2_pipeline import BUNDLE_ROLES, load_inference_bundle

    artifacts, sample_submission = _source_artifacts(tmp_path)
    output = tmp_path / "bundle"
    argv: list[str] = []
    for role, path in artifacts.items():
        argv.extend(["--" + role.replace("_", "-"), str(path)])
    argv.extend(["--sample-submission", str(sample_submission), "--output-dir", str(output)])
    assert main(argv) == 0
    loaded = load_inference_bundle(output)
    assert tuple(loaded.paths) == BUNDLE_ROLES
    assert len(loaded.manifest["files"]) == 8
    for entry in loaded.manifest["files"].values():
        assert "\\" not in entry["path"]
        assert len(entry["sha256"]) == 64
    assert not (output / sample_submission.name).exists()
    assert main(argv) == 2


def test_bundle_builder_rejects_type_changed_inference_contract(tmp_path: Path) -> None:
    from scripts.build_imu_inference_bundle import main

    artifacts, sample_submission = _source_artifacts(tmp_path)
    invalid = dict(EXPECTED_INFERENCE_CONFIG)
    invalid["deterministic_algorithms"] = 1
    artifacts["inference_config"].write_text(
        yaml.safe_dump(invalid, sort_keys=False), encoding="utf-8"
    )
    output = tmp_path / "bundle"
    argv: list[str] = []
    for role, path in artifacts.items():
        argv.extend(["--" + role.replace("_", "-"), str(path)])
    argv.extend(["--sample-submission", str(sample_submission), "--output-dir", str(output)])

    assert main(argv) == 2
    assert not output.exists()


def test_atomic_submission_requires_overwrite_and_preserves_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.inference import imu_stage2_pipeline as pipeline

    contract = {
        "submission_contract_version": "imu-submission-v1",
        "columns": ["sample_id", "class_id"],
        "sample_id_column": "sample_id",
        "prediction_column": "class_id",
        "encoding": "utf-8",
        "header": True,
        "row_order": "sample_submission",
        "sample_ids": ["SM_test_0001", "SM_test_0002"],
        "prediction_representation": "class_id",
    }
    rows = [("SM_test_0001", 2), ("SM_test_0002", 10)]
    output = tmp_path / "submission.csv"
    output.write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        pipeline.write_submission_atomic(output, rows, contract, overwrite=False)
    assert output.read_text(encoding="utf-8") == "old\n"

    original = pipeline.validate_submission_file
    monkeypatch.setattr(pipeline, "validate_submission_file", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("injected")))
    with pytest.raises(ValueError, match="injected"):
        pipeline.write_submission_atomic(output, rows, contract, overwrite=True)
    assert output.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(".submission.csv.tmp-*"))

    monkeypatch.setattr(pipeline, "validate_submission_file", original)
    pipeline.write_submission_atomic(output, rows, contract, overwrite=True)
    assert list(csv.reader(output.open(encoding="utf-8"))) == [
        ["sample_id", "class_id"],
        ["SM_test_0001", "2"],
        ["SM_test_0002", "10"],
    ]


def test_inference_cli_predicts_missing_imu_and_writes_bounded_audit(tmp_path: Path) -> None:
    from scripts.infer_imu_stage2 import main

    bundle = _build_bundle(tmp_path)
    raw = tmp_path / "test"
    imu = raw / "SM_test_0001" / "IMU"
    imu.mkdir(parents=True)
    _write_raw_csv(imu / "part1.csv", "2025-01-01 00:00:00.000000000", 1.0)
    (raw / ".claude").mkdir()
    (raw / "SM_test_0001" / "notes.txt").write_text("ignored", encoding="utf-8")
    (imu / "readme.md").write_text("ignored", encoding="utf-8")
    (raw / "SM_test_0002").mkdir()
    output = tmp_path / "submission.csv"
    audit = tmp_path / "audit"
    assert main(["--raw-test-root", str(raw), "--output-csv", str(output), "--bundle-root", str(bundle), "--audit-dir", str(audit)]) == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [row["sample_id"] for row in rows] == ["SM_test_0001", "SM_test_0002"]
    assert {int(row["class_id"]) for row in rows} <= {2, 10}
    runs = list(audit.iterdir())
    assert len(runs) == 1
    assert {path.name for path in runs[0].iterdir()} == {
        "inference_manifest.csv",
        "processing.log",
        "problematic_sample_qc.json",
        "inference_summary.json",
    }
    summary = json.loads((runs[0] / "inference_summary.json").read_text(encoding="utf-8"))
    assert summary["exit_code"] == 0
    assert summary["predicted_sample_count"] == 2
    assert summary["ignored_entry_count"] == 3
    assert summary["ignored_root_entries"] == [".claude"]
    assert summary["ignored_sample_entries"] == [
        "SM_test_0001/IMU/readme.md",
        "SM_test_0001/notes.txt",
    ]


def test_inference_cli_global_failure_never_replaces_existing_output(tmp_path: Path) -> None:
    from scripts.infer_imu_stage2 import main

    bundle = _build_bundle(tmp_path)
    raw = tmp_path / "test"
    (raw / "SM_test_0001").mkdir(parents=True)
    (raw / "SM_test_0002").mkdir()
    output = tmp_path / "submission.csv"
    output.write_text("old\n", encoding="utf-8")
    assert main(["--raw-test-root", str(raw), "--output-csv", str(output), "--bundle-root", str(bundle)]) == 2
    assert output.read_text(encoding="utf-8") == "old\n"


@pytest.mark.parametrize(
    "relation",
    [
        "output_equals_raw",
        "output_inside_raw",
        "audit_equals_raw",
        "audit_inside_raw",
        "output_equals_bundle",
        "output_inside_bundle",
        "audit_equals_bundle",
        "audit_inside_bundle",
        "output_inside_audit",
        "audit_inside_output",
        "output_equals_audit",
        "raw_inside_audit",
        "bundle_inside_audit",
    ],
)
def test_ordinary_path_overlap_is_rejected_before_any_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relation: str,
) -> None:
    from scripts import infer_imu_stage2

    raw = tmp_path / "raw"
    bundle = tmp_path / "bundle"
    raw.mkdir()
    bundle.mkdir()
    output = tmp_path / "submission.csv"
    audit = tmp_path / "audit"
    if relation == "output_equals_raw":
        output = raw
    elif relation == "output_inside_raw":
        output = raw / "submission.csv"
    elif relation == "audit_equals_raw":
        audit = raw
    elif relation == "audit_inside_raw":
        audit = raw / "audit"
    elif relation == "output_equals_bundle":
        output = bundle
    elif relation == "output_inside_bundle":
        output = bundle / "submission.csv"
    elif relation == "audit_equals_bundle":
        audit = bundle
    elif relation == "audit_inside_bundle":
        audit = bundle / "audit"
    elif relation == "output_inside_audit":
        output = audit / "submission.csv"
    elif relation == "audit_inside_output":
        audit = output / "audit"
    elif relation == "output_equals_audit":
        audit = output
    elif relation == "raw_inside_audit":
        audit = tmp_path
    elif relation == "bundle_inside_audit":
        audit = tmp_path

    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: object, **_kwargs: object) -> object:
            calls.append(name)
            raise AssertionError(f"{name} must not be called")
        return fail

    monkeypatch.setattr(infer_imu_stage2, "load_inference_bundle", forbidden("bundle"))
    monkeypatch.setattr(infer_imu_stage2, "discover_test_samples", forbidden("discovery"))
    monkeypatch.setattr(
        infer_imu_stage2,
        "plan_inference_sample",
        forbidden("preprocess"),
    )
    monkeypatch.setattr(infer_imu_stage2, "write_submission_atomic", forbidden("submission"))
    before_raw = sorted(path.relative_to(raw).as_posix() for path in raw.rglob("*"))
    before_bundle = sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*"))

    code = infer_imu_stage2.main(
        [
            "--raw-test-root", str(raw),
            "--output-csv", str(output),
            "--bundle-root", str(bundle),
            "--audit-dir", str(audit),
            "--overwrite-output",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert "overlap" in payload["error"].casefold()
    assert calls == []
    assert sorted(path.relative_to(raw).as_posix() for path in raw.rglob("*")) == before_raw
    assert sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*")) == before_bundle
    assert not list(tmp_path.rglob("*.tmp-*"))
    assert not list(tmp_path.rglob("*.staging-*"))
    assert not list(tmp_path.rglob("*.backup-*"))


def _transaction_payload() -> tuple[list[tuple[str, object]], dict[str, object]]:
    contract: dict[str, object] = {
        "submission_contract_version": "imu-submission-v1",
        "columns": ["sample_id", "class_id"],
        "sample_id_column": "sample_id",
        "prediction_column": "class_id",
        "encoding": "utf-8",
        "header": True,
        "row_order": "sample_submission",
        "sample_ids": ["SM_test_0001"],
        "prediction_representation": "class_id",
    }
    return [("SM_test_0001", 2)], contract


@pytest.mark.parametrize("existing", [False, True])
def test_submission_publish_failure_restores_initial_output_and_never_publishes_success_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    from scripts import infer_imu_stage2

    output = tmp_path / "submission.csv"
    old = b"sentinel-old-output\n"
    if existing:
        output.write_bytes(old)
    audit = tmp_path / "audit"
    rows, contract = _transaction_payload()
    original_replace = infer_imu_stage2.os.replace

    def fail_submission(source: object, destination: object) -> None:
        if Path(destination) == output:
            raise OSError("injected submission publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(infer_imu_stage2.os, "replace", fail_submission)
    with pytest.raises(OSError, match="submission publish"):
        infer_imu_stage2._publish_success_transaction(
            output,
            rows,
            contract,
            audit,
            [],
            [],
            {"status": "success", "exit_code": 0},
            overwrite=existing,
        )

    assert output.read_bytes() == old if existing else not output.exists()
    assert not list(audit.rglob("inference_summary.json")) if audit.exists() else True
    assert not list(tmp_path.rglob("*.tmp-*"))
    assert not list(tmp_path.rglob("*.staging-*"))
    assert not list(tmp_path.rglob("*.backup-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_success_audit_publish_failure_rolls_back_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    from scripts import infer_imu_stage2

    output = tmp_path / "submission.csv"
    old = b"sentinel-old-output\n"
    if existing:
        output.write_bytes(old)
    audit = tmp_path / "audit"
    rows, contract = _transaction_payload()
    original_replace = infer_imu_stage2.os.replace

    def fail_audit(source: object, destination: object) -> None:
        destination_path = Path(destination)
        if destination_path.parent == audit and not destination_path.name.startswith("."):
            raise OSError("injected audit publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(infer_imu_stage2.os, "replace", fail_audit)
    with pytest.raises(OSError, match="audit publish"):
        infer_imu_stage2._publish_success_transaction(
            output,
            rows,
            contract,
            audit,
            [],
            [],
            {"status": "success", "exit_code": 0},
            overwrite=existing,
        )

    assert output.read_bytes() == old if existing else not output.exists()
    assert not list(audit.rglob("inference_summary.json")) if audit.exists() else True
    assert not list(tmp_path.rglob("*.tmp-*"))
    assert not list(tmp_path.rglob("*.staging-*"))
    assert not list(tmp_path.rglob("*.backup-*"))


def test_post_commit_backup_cleanup_failure_keeps_successful_cli_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import infer_imu_stage2

    bundle = _build_bundle(tmp_path)
    capsys.readouterr()
    raw = tmp_path / "test"
    (raw / "SM_test_0001").mkdir(parents=True)
    (raw / "SM_test_0002").mkdir()
    output = tmp_path / "submission.csv"
    old = b"sentinel-old-output\n"
    output.write_bytes(old)
    audit = tmp_path / "audit"
    original_unlink = infer_imu_stage2.Path.unlink
    original_rmtree = infer_imu_stage2.shutil.rmtree

    def fail_backup_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if ".backup-" in path.name:
            raise OSError("injected post-commit backup cleanup failure")
        original_unlink(path, *args, **kwargs)

    def fail_published_audit_rollback(path: object, *args: object, **kwargs: object) -> None:
        candidate = Path(path)
        if candidate.parent == audit and not candidate.name.startswith("."):
            raise OSError("injected published audit rollback failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(infer_imu_stage2.Path, "unlink", fail_backup_cleanup)
    monkeypatch.setattr(infer_imu_stage2.shutil, "rmtree", fail_published_audit_rollback)

    code = infer_imu_stage2.main(
        [
            "--raw-test-root", str(raw),
            "--output-csv", str(output),
            "--bundle-root", str(bundle),
            "--audit-dir", str(audit),
            "--overwrite-output",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["status"] == "success"
    assert "cleanup" in captured.err.casefold()
    assert output.read_bytes() != old
    summaries = list(audit.rglob("inference_summary.json"))
    assert len(summaries) == 1
    audit_summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert audit_summary["status"] == "success"
    assert audit_summary["exit_code"] == 0
    assert len(list(tmp_path.rglob("*.backup-*"))) == 1


def test_unclassified_exception_is_global_code_two_not_sample_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import infer_imu_stage2

    def fail_bundle(_path: Path) -> object:
        raise AssertionError("code defect")

    monkeypatch.setattr(infer_imu_stage2, "load_inference_bundle", fail_bundle)
    output = tmp_path / "submission.csv"
    assert infer_imu_stage2.main(
        [
            "--raw-test-root", str(tmp_path / "raw"),
            "--output-csv", str(output),
            "--bundle-root", str(tmp_path / "bundle"),
        ]
    ) == 2
    assert not output.exists()


def test_inference_cli_exit_one_never_publishes_incomplete_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import infer_imu_stage2

    bundle = _build_bundle(tmp_path)
    raw = tmp_path / "test"
    (raw / "SM_test_0001").mkdir(parents=True)
    (raw / "SM_test_0002").mkdir()
    output = tmp_path / "submission.csv"
    unsupported = dict(EXPECTED_INFERENCE_CONFIG)
    unsupported["imu_unavailable_policy"] = "unsupported"
    monkeypatch.setattr(infer_imu_stage2, "_validate_config", lambda _bundle: unsupported)

    assert infer_imu_stage2.main(
        ["--raw-test-root", str(raw), "--output-csv", str(output), "--bundle-root", str(bundle)]
    ) == 1
    assert not output.exists()


def test_save_intermediates_requires_audit_and_never_writes_placeholders(tmp_path: Path) -> None:
    from scripts.infer_imu_stage2 import main

    bundle = _build_bundle(tmp_path)
    raw = tmp_path / "test"
    imu = raw / "SM_test_0001" / "IMU"
    imu.mkdir(parents=True)
    _write_raw_csv(imu / "part1.csv", "2025-01-01 00:00:00.000000000", 1.0)
    (raw / "SM_test_0002").mkdir()
    output = tmp_path / "submission.csv"
    assert main(["--raw-test-root", str(raw), "--output-csv", str(output), "--bundle-root", str(bundle), "--save-intermediates"]) == 2
    assert not output.exists()

    audit = tmp_path / "audit"
    assert main(["--raw-test-root", str(raw), "--output-csv", str(output), "--bundle-root", str(bundle), "--audit-dir", str(audit), "--save-intermediates"]) == 0
    run = next(audit.iterdir())
    stage1_root = run / "intermediates" / "stage1"
    stage2_root = run / "intermediates" / "stage2"
    assert len(list(stage2_root.rglob("imu_stage2.npz"))) == 1
    assert len(list(stage1_root.rglob("imu_merged.csv"))) == 1
    assert not (stage2_root / "SM_test_0002").exists()


def test_saved_online_intermediates_have_actual_provenance_and_complete_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.infer_imu_stage2 import main
    from scripts import preprocess_imu_stage2 as stage2_cli
    from src.data.imu_stage1_bridge import discover_stage1_artifacts
    from src.data.imu_stage2_contracts import NoUsableGridCellsError
    from src.data.imu_stage2_io import load_stage2_schema, validate_existing_action
    from src.inference import imu_stage2_pipeline as pipeline

    bundle = _build_bundle(tmp_path)
    training_schema_bytes = (bundle / "schema.json").read_bytes()
    raw = tmp_path / "test"
    for index in (1, 2):
        imu = raw / f"SM_test_{index:04d}" / "IMU"
        imu.mkdir(parents=True)
        _write_raw_csv(
            imu / "part1.csv",
            "2025-01-01 00:00:00.000000000",
            float(index),
        )
    original_stage2 = pipeline.process_stage2_action

    def stage2_with_one_degradation(stage1_result: object, **kwargs: object):
        if getattr(stage1_result, "sample_id") == "SM_test_0002":
            raise NoUsableGridCellsError("SM_test_0002", "injected no usable grid")
        return original_stage2(stage1_result, **kwargs)

    monkeypatch.setattr(pipeline, "process_stage2_action", stage2_with_one_degradation)
    output = tmp_path / "submission.csv"
    audit = tmp_path / "audit"
    assert main([
        "--raw-test-root", str(raw),
        "--output-csv", str(output),
        "--bundle-root", str(bundle),
        "--audit-dir", str(audit),
        "--save-intermediates",
    ]) == 0

    run = next(audit.iterdir())
    stage1_root = run / "intermediates" / "stage1"
    stage2_root = run / "intermediates" / "stage2"
    stage1_descriptors = discover_stage1_artifacts(stage1_root)
    assert [item.sample_id for item in stage1_descriptors] == [
        "SM_test_0001",
        "SM_test_0002",
    ]
    schema = load_stage2_schema(stage2_root / "schema.json")
    assert schema["provenance"]["source_stage1_manifest_sha256"] == sha256_file(
        stage1_root / "manifest.csv"
    )
    assert (stage2_root / "schema.json").read_bytes() != training_schema_bytes
    with (stage2_root / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == ["SM_test_0001"]
    descriptors_by_id = {item.sample_id: item for item in stage1_descriptors}
    for row in rows:
        descriptor = descriptors_by_id[row["sample_id"]]
        action = stage2_root / Path(row["relative_action_path"])
        fingerprints = stage2_cli._fingerprints(
            descriptor, str(schema["contract_sha256"])
        )
        validate_existing_action(action, fingerprints)
        assert (stage2_root / Path(row["stage2_npz_relpath"])).is_file()
    audit_rows = list(csv.DictReader((run / "inference_manifest.csv").open(encoding="utf-8")))
    second = next(row for row in audit_rows if row["sample_id"] == "SM_test_0002")
    assert second["source_status"] == "available"
    assert second["stage1_status"] == "success"
    assert second["stage2_status"] == "degraded"
    assert not (stage2_root / "SM_test_0002" / "imu_stage2.npz").exists()


def test_nontrivial_normalization_transforms_only_valid_cells(tmp_path: Path) -> None:
    from scripts import infer_imu_stage2

    values = np.asarray([[[[5.0] * 16] * 5]], dtype=np.float32)
    valid = np.zeros((1, 1, 5), dtype=bool)
    valid[0, 0, 0] = True
    bundle = Namespace(
        normalization_arrays={
            "mean": np.full((5, 16), 1.0, dtype=np.float64),
            "applied_scale": np.full((5, 16), 2.0, dtype=np.float64),
        }
    )
    normalized = infer_imu_stage2._normalize_batch(
        {"values": values, "valid_mask": valid}, bundle, torch.device("cpu")
    )

    result = normalized["values"]
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result[0, 0, 0], torch.full((16,), 2.0))
    assert torch.equal(result[0, 0, 1:], torch.zeros((4, 16)))
    assert torch.isfinite(result).all()


def test_streaming_batcher_flushes_at_budget_without_consuming_next_sample() -> None:
    from scripts import infer_imu_stage2
    from src.data.imu_stage2_contracts import InferenceSample, Stage2ActionResult, DataStatus

    events: list[str] = []

    def sample(sample_id: str, length: int) -> InferenceSample:
        valid = np.ones((length, 5), dtype=bool)
        result = Stage2ActionResult(
            sample_id=sample_id,
            values=np.ones((length, 5, 16), dtype=np.float32),
            sensor_mask=np.ones(5, dtype=bool),
            valid_mask=valid,
            timestamps_ms=np.arange(length, dtype=np.int64) * 100,
            qc={},
            status=DataStatus.SUCCESS,
        )
        return InferenceSample(sample_id, result, True, True)

    def prepared():
        for index in range(3):
            events.append(f"prepare-{index}")
            yield sample(f"SM_test_{index + 1:04d}", 2)

    config = {"maximum_batch_size": 16, "batch_feature_budget": 2 * 2 * 5 * 16}
    batches = infer_imu_stage2._streaming_batches(prepared(), config)
    first = next(batches)
    events.append("forward-0")

    assert [item.sample_id for item in first] == ["SM_test_0001", "SM_test_0002"]
    assert events == ["prepare-0", "prepare-1", "forward-0"]
    second = next(batches)
    assert [item.sample_id for item in second] == ["SM_test_0003"]


def test_cli_flushes_and_releases_pending_before_materializing_over_budget_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import infer_imu_stage2
    from src.inference import imu_stage2_pipeline as pipeline

    bundle = _build_bundle(tmp_path)
    raw = tmp_path / "test"
    first_imu = raw / "SM_test_0001" / "IMU"
    second_imu = raw / "SM_test_0002" / "IMU"
    first_imu.mkdir(parents=True)
    second_imu.mkdir(parents=True)
    _write_raw_csv_rows(
        first_imu / "part1.csv",
        ["2025-01-01 00:00:00.000000000"],
        1.0,
    )
    _write_raw_csv_rows(
        second_imu / "part1.csv",
        [
            "2025-01-01 00:00:00.000000000",
            "2025-01-01 00:00:00.100000000",
            "2025-01-01 00:00:00.200000000",
        ],
        2.0,
    )
    config = dict(EXPECTED_INFERENCE_CONFIG)
    config["batch_feature_budget"] = 160
    events: list[str] = []
    original_stage1 = pipeline.process_raw_imu_source
    original_stage2 = pipeline.process_stage2_action
    original_normalize = infer_imu_stage2._normalize_batch

    def observed_stage1(source: object):
        result = original_stage1(source)
        events.append(f"plan-{result.sample_id}")
        return result

    def observed_stage2(stage1_result: object, **kwargs: object):
        sample_id = str(getattr(stage1_result, "sample_id"))
        events.append(f"materialize-{sample_id}")
        result = original_stage2(stage1_result, **kwargs)
        weakref.finalize(result.values, events.append, f"release-{sample_id}")
        return result

    def observed_normalize(*args: object, **kwargs: object):
        events.append("forward")
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(infer_imu_stage2, "_validate_config", lambda _bundle: config)
    monkeypatch.setattr(pipeline, "process_raw_imu_source", observed_stage1)
    monkeypatch.setattr(pipeline, "process_stage2_action", observed_stage2)
    monkeypatch.setattr(infer_imu_stage2, "_normalize_batch", observed_normalize)

    code, _summary = infer_imu_stage2.run(
        Namespace(
            raw_test_root=raw,
            output_csv=tmp_path / "submission.csv",
            bundle_root=bundle,
            overwrite_output=False,
            audit_dir=None,
            save_intermediates=False,
            device="cpu",
        )
    )

    assert code == 0
    assert events.index("plan-SM_test_0002") < events.index("forward")
    assert events.index("forward") < events.index("release-SM_test_0001")
    assert events.index("release-SM_test_0001") < events.index(
        "materialize-SM_test_0002"
    )


def test_cli_releases_final_batch_tensors_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import infer_imu_stage2
    from src.inference import imu_stage2_pipeline as pipeline

    bundle = _build_bundle(tmp_path)
    raw = tmp_path / "test"
    finalizers: list[weakref.finalize] = []
    released: list[str] = []
    for index in (1, 2):
        imu = raw / f"SM_test_{index:04d}" / "IMU"
        imu.mkdir(parents=True)
        _write_raw_csv_rows(
            imu / "part1.csv",
            ["2025-01-01 00:00:00.000000000"],
            float(index),
        )
    original_stage2 = pipeline.process_stage2_action
    original_publish = infer_imu_stage2._publish_success_transaction

    def observed_stage2(stage1_result: object, **kwargs: object):
        result = original_stage2(stage1_result, **kwargs)
        finalizers.append(
            weakref.finalize(
                result.values,
                released.append,
                str(getattr(stage1_result, "sample_id")),
            )
        )
        return result

    def observed_publish(*args: object, **kwargs: object):
        assert sorted(released) == ["SM_test_0001", "SM_test_0002"]
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(pipeline, "process_stage2_action", observed_stage2)
    monkeypatch.setattr(
        infer_imu_stage2, "_publish_success_transaction", observed_publish
    )

    code, _summary = infer_imu_stage2.run(
        Namespace(
            raw_test_root=raw,
            output_csv=tmp_path / "submission.csv",
            bundle_root=bundle,
            overwrite_output=False,
            audit_dir=None,
            save_intermediates=False,
            device="cpu",
        )
    )

    assert code == 0
    assert all(not finalizer.alive for finalizer in finalizers)


@pytest.mark.parametrize(
    ("lengths", "budget", "expected_sizes"),
    [
        ([2, 2, 2], 2 * 2 * 5 * 16, [2, 1]),
        ([2, 3], 2 * 2 * 5 * 16, [1, 1]),
        ([5, 1], 2 * 2 * 5 * 16, [1, 1]),
        ([2, 2, 2], 3 * 2 * 5 * 16, [3]),
    ],
)
def test_streaming_batch_budget_boundaries(
    lengths: list[int],
    budget: int,
    expected_sizes: list[int],
) -> None:
    from scripts import infer_imu_stage2

    class Result:
        def __init__(self, length: int) -> None:
            self.values = np.empty((length, 5, 16), dtype=np.float32)

    class Sample:
        def __init__(self, length: int) -> None:
            self.imu_result = Result(length)

    groups = list(
        infer_imu_stage2._streaming_batches(
            (Sample(length) for length in lengths),
            {"maximum_batch_size": 16, "batch_feature_budget": budget},
        )
    )
    assert [len(group) for group in groups] == expected_sizes


def test_submission_bytes_are_partition_independent_and_repeatable(tmp_path: Path) -> None:
    from scripts import infer_imu_stage2
    from src.inference.imu_stage2_pipeline import write_submission_atomic

    rows, contract = _transaction_payload()
    rows = [("SM_test_0001", 2)]
    outputs: list[bytes] = []
    for index, budget in enumerate((80, 160, 80)):
        groups = list(
            infer_imu_stage2._streaming_batches(
                [Namespace(imu_result=None)],
                {"maximum_batch_size": 16, "batch_feature_budget": budget},
            )
        )
        assert [len(group) for group in groups] == [1]
        output = tmp_path / f"submission-{index}.csv"
        write_submission_atomic(output, rows, contract)
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1] == outputs[2]


def test_cli_submission_bytes_are_invariant_to_legal_batch_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import infer_imu_stage2

    bundle = _build_bundle(tmp_path)
    raw = tmp_path / "test"
    for index in (1, 2):
        imu = raw / f"SM_test_{index:04d}" / "IMU"
        imu.mkdir(parents=True)
        _write_raw_csv(
            imu / "part1.csv",
            "2025-01-01 00:00:00.000000000",
            float(index),
        )
    outputs: list[bytes] = []
    summaries: list[dict[str, object]] = []
    for index, budget in enumerate((80, 320)):
        config = dict(EXPECTED_INFERENCE_CONFIG)
        config["batch_feature_budget"] = budget
        monkeypatch.setattr(infer_imu_stage2, "_validate_config", lambda _bundle, value=config: value)
        output = tmp_path / f"budget-{index}.csv"
        code, summary = infer_imu_stage2.run(
            Namespace(
                raw_test_root=raw,
                output_csv=output,
                bundle_root=bundle,
                overwrite_output=False,
                audit_dir=None,
                save_intermediates=False,
                device="cpu",
            )
        )
        assert code == 0
        outputs.append(output.read_bytes())
        summaries.append(summary)
    assert summaries[0]["batch_sizes"] == [1, 1]
    assert summaries[1]["batch_sizes"] == [2]
    assert outputs[0] == outputs[1]


def test_wrapper_forwards_only_explicit_paths_and_packaged_bundle(tmp_path: Path) -> None:
    bash = os.environ.get("BASH_EXE", "bash")
    if subprocess.run([bash, "--version"], capture_output=True).returncode != 0:
        pytest.skip("Bash is unavailable")
    capture = tmp_path / "captured.txt"
    fake = tmp_path / "python"
    fake.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n", encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    def bash_path(path: Path) -> str:
        resolved = path.resolve()
        if resolved.drive:
            return f"/mnt/{resolved.drive[0].lower()}/{resolved.as_posix()[3:]}"
        return resolved.as_posix()

    outside = tmp_path / "outside"
    outside.mkdir()
    wrapper = Path("inference.sh").resolve()
    result = subprocess.run(
        [
            bash,
            "-c",
            f"export PATH={shlex.quote(bash_path(tmp_path))}:\"$PATH\"; "
            f"export CAPTURE={shlex.quote(bash_path(capture))}; "
            f'bash {shlex.quote(bash_path(wrapper))} "raw root" "output file.csv"',
        ],
        cwd=outside, capture_output=True, text=True, errors="replace",
    )
    assert result.returncode == 0, result.stderr
    forwarded = capture.read_text(encoding="utf-8").splitlines()
    assert forwarded[0].replace("\\", "/").endswith("/scripts/infer_imu_stage2.py")
    assert forwarded[1:] == [
        "--raw-test-root", "raw root", "--output-csv", "output file.csv", "--bundle-root", bash_path(wrapper.parent / "inference_bundle")
    ]
