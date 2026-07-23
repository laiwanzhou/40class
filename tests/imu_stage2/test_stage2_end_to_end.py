from __future__ import annotations

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

import scripts.preprocess_imu_stage1 as stage1
import scripts.preprocess_imu_stage2 as stage2_cli
from src.data.imu_stage2_contracts import sha256_file


def _load_test_helper(module_name: str):
    path = Path(__file__).with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load test helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INFERENCE_HELPERS = _load_test_helper("test_inference_cli")
_STAGE2_CLI_HELPERS = _load_test_helper("test_stage2_cli")
_build_bundle = _INFERENCE_HELPERS._build_bundle
make_stage1_root = _STAGE2_CLI_HELPERS.make_stage1_root


def _tree_snapshot(root: Path) -> list[tuple[str, int, str]]:
    return [
        (path.relative_to(root).as_posix(), path.stat().st_size, sha256_file(path))
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    ]


def _generate_stage2_root(
    tmp_path: Path,
    capsys,
    kinds: tuple[str, ...] = ("success",),
) -> tuple[Path, Path, dict[str, object]]:
    input_root = make_stage1_root(tmp_path, kinds)
    output_root = tmp_path / "new_IMU_stage2"
    code = stage2_cli.main(
        [
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--summary-format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert code == (1 if "failed" in kinds else 0)
    return input_root, output_root, json.loads(captured.out)


def test_read_only_validator_matches_canonical_summary_and_mutates_neither_root(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.validate_imu_stage2_output import main

    input_root, output_root, expected = _generate_stage2_root(tmp_path, capsys)
    expected_path = tmp_path / "expected-summary.json"
    expected_path.write_text(
        json.dumps(expected, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    audit_directory = tmp_path / "audit"
    audit_directory.mkdir()
    audit_output = audit_directory / "formal-validation-summary.json"
    input_before = _tree_snapshot(input_root)
    output_before = _tree_snapshot(output_root)

    code = main(
        [
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--expected-summary",
            str(expected_path),
            "--audit-output",
            str(audit_output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out) == expected
    assert json.loads(audit_output.read_text(encoding="utf-8")) == expected
    assert _tree_snapshot(input_root) == input_before
    assert _tree_snapshot(output_root) == output_before


def test_read_only_validator_rejects_manifest_count_tamper(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.validate_imu_stage2_output import main

    input_root, output_root, _expected = _generate_stage2_root(tmp_path, capsys)
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["valid_cell_count"] = "999"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assert main(["--input-root", str(input_root), "--output-root", str(output_root)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "manifest" in captured.err.casefold()


def test_read_only_validator_accepts_well_formed_failed_qc_with_exit_one(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.validate_imu_stage2_output import main

    input_root, output_root, expected = _generate_stage2_root(
        tmp_path, capsys, ("success", "failed")
    )

    assert main(["--input-root", str(input_root), "--output-root", str(output_root)]) == 1
    assert json.loads(capsys.readouterr().out) == expected


def test_read_only_validator_rejects_qc_tamper_residue_and_summary_mismatch(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.validate_imu_stage2_output import main

    for case in ("qc", "residue", "summary"):
        case_root = tmp_path / case
        case_root.mkdir()
        input_root, output_root, expected = _generate_stage2_root(case_root, capsys)
        argv = ["--input-root", str(input_root), "--output-root", str(output_root)]
        if case == "qc":
            qc_path = next(output_root.rglob("qc.json"))
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            qc["valid_cell_count"] = 999
            qc_path.write_text(
                json.dumps(qc, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
        elif case == "residue":
            (output_root / ".staging-orphan").mkdir()
        else:
            expected["action_count"] = 999
            expected_path = case_root / "wrong-summary.json"
            expected_path.write_text(
                json.dumps(expected, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            argv.extend(["--expected-summary", str(expected_path)])

        assert main(argv) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "validation error" in captured.err.casefold()


def test_validator_forbids_audit_output_inside_either_data_root(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.validate_imu_stage2_output import main

    input_root, output_root, _expected = _generate_stage2_root(tmp_path, capsys)
    for forbidden in (input_root / "audit.json", output_root / "audit.json"):
        assert main(
            [
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--audit-output",
                str(forbidden),
            ]
        ) == 2
        assert not forbidden.exists()
        assert capsys.readouterr().out == ""


def _write_sensor_csv(path: Path, sensor_code: str, base: float) -> None:
    rows = [
        [
            "2025-01-01 00:00:00.000000000",
            f"WT{sensor_code}(device)",
            *[base + index for index in range(16)],
        ],
        [
            "2025-01-01 00:00:00.100000000",
            f"WT{sensor_code}(device)",
            *[base + 1 + index for index in range(16)],
        ],
    ]
    pd.DataFrame(rows, columns=stage1.REQUIRED_SOURCE_COLUMNS).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _write_ll_rows(path: Path, timestamps: list[str], values: list[float]) -> None:
    rows = [
        [timestamp, "WTLL(device)", *values]
        for timestamp in timestamps
    ]
    pd.DataFrame(rows, columns=stage1.REQUIRED_SOURCE_COLUMNS).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


@pytest.mark.parametrize("scenario", ["missing", "no_valid", "no_usable", "safety"])
def test_raw_test_inference_preserves_predictions_for_typed_imu_unavailability(
    tmp_path: Path,
    scenario: str,
) -> None:
    from scripts import infer_imu_stage2

    bundle = _build_bundle(tmp_path)
    raw_root = tmp_path / "raw-test"
    sample = raw_root / "SM_test_0001"
    sample.mkdir(parents=True)
    (raw_root / "SM_test_0002").mkdir()
    if scenario != "missing":
        imu = sample / "IMU"
        imu.mkdir()
        if scenario == "no_valid":
            _write_ll_rows(imu / "part1.csv", ["not-a-time"], [0.0] * 16)
        elif scenario == "no_usable":
            _write_ll_rows(
                imu / "part1.csv",
                ["2025-01-01 00:00:00.000000000"],
                [0.0] * 16,
            )
        else:
            features = [0.0] * 16
            features[12] = 1.0
            _write_ll_rows(
                imu / "part1.csv",
                [
                    "2025-01-01 00:00:00.000000000",
                    "2025-01-01 00:16:40.000000000",
                ],
                features,
            )

    output = tmp_path / "submission.csv"
    code, summary = infer_imu_stage2.run(
        Namespace(
            raw_test_root=raw_root,
            output_csv=output,
            bundle_root=bundle,
            overwrite_output=False,
            audit_dir=None,
            save_intermediates=False,
            device="cpu",
        )
    )

    assert code == 0
    assert summary["predicted_sample_count"] == 2
    assert summary["unavailable_imu_count"] == 2
    with output.open("r", encoding="utf-8", newline="") as handle:
        assert [row["sample_id"] for row in csv.DictReader(handle)] == [
            "SM_test_0001",
            "SM_test_0002",
        ]


def test_raw_test_inference_is_byte_stable_for_normal_partial_and_ignored_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import infer_imu_stage2
    from src.inference.imu_stage2_pipeline import load_inference_bundle

    bundle = _build_bundle(tmp_path)
    base_config = infer_imu_stage2._validate_config(load_inference_bundle(bundle))
    raw_root = tmp_path / "raw-test"
    normal = raw_root / "SM_test_0001" / "IMU"
    partial = raw_root / "SM_test_0002" / "IMU"
    normal.mkdir(parents=True)
    partial.mkdir(parents=True)
    for index, sensor in enumerate(("LL", "RL", "LA", "RA", "C")):
        _write_sensor_csv(normal / f"part{index + 1}.csv", sensor, float(index))
    _write_sensor_csv(partial / "part1.csv", "LL", 10.0)
    (raw_root / ".claude").mkdir()

    outputs: list[bytes] = []
    summaries: list[dict[str, object]] = []
    for index, budget in enumerate((160, 640)):
        config = dict(base_config)
        config["batch_feature_budget"] = budget
        monkeypatch.setattr(
            infer_imu_stage2,
            "_validate_config",
            lambda _bundle, value=config: value,
        )
        output = tmp_path / f"submission-{index}.csv"
        code, summary = infer_imu_stage2.run(
            Namespace(
                raw_test_root=raw_root,
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

    assert outputs[0] == outputs[1]
    assert summaries[0]["sample_count"] == 2
    assert summaries[0]["unavailable_imu_count"] == 0
    assert summaries[0]["ignored_root_entries"] == [".claude"]
    assert summaries[0]["batch_sizes"] != summaries[1]["batch_sizes"]
