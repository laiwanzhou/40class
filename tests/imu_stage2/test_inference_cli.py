from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
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
    assert len(list(run.rglob("imu_stage2.npz"))) == 1
    assert len(list(run.rglob("imu_merged.csv"))) == 1
    unavailable_root = run / "intermediates" / "SM_test_0002"
    assert not unavailable_root.exists()


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

    result = subprocess.run(
        [
            bash,
            "-c",
            f"export PATH={shlex.quote(bash_path(tmp_path))}:\"$PATH\"; "
            f"export CAPTURE={shlex.quote(bash_path(capture))}; "
            'bash inference.sh "raw root" "output file.csv"',
        ],
        cwd=Path.cwd(), capture_output=True, text=True, errors="replace",
    )
    assert result.returncode == 0, result.stderr
    forwarded = capture.read_text(encoding="utf-8").splitlines()
    assert forwarded[0].replace("\\", "/").endswith("/scripts/infer_imu_stage2.py")
    assert forwarded[1:] == [
        "--raw-test-root", "raw root", "--output-csv", "output file.csv", "--bundle-root", bash_path(Path.cwd() / "inference_bundle")
    ]
