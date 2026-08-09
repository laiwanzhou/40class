from __future__ import annotations

from pathlib import Path

import pytest

from scripts.probe_x3d_s_environment import (
    INTERNAL_SIZE_LIMIT_BYTES,
    aggregate_unique_artifact_bytes,
    build_probe_payload,
    enforce_submission_size_gate,
    estimate_custom_head_parameters,
    inventory_weight_file,
    read_text_with_detected_encoding,
)


def test_x3d_s_environment_probe_declares_fixed_input_and_source() -> None:
    probe = build_probe_payload(run_forward=False)

    assert probe["input_shape"] == [1, 3, 13, 182, 182]
    assert probe["model_name"] == "x3d_s"
    assert probe["pretraining"] == "kinetics_400"
    assert probe["source_revision"]
    assert probe["internal_size_limit_bytes"] == 95_000_000
    assert probe["runtime"]["pytorchvideo"] == "0.1.5"


def test_aggregate_size_counts_the_same_artifact_once(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"a" * 11)
    second.write_bytes(b"b" * 17)

    assert aggregate_unique_artifact_bytes([first, first, second]) == 28


def test_submission_size_gate_rejects_limit_and_above() -> None:
    enforce_submission_size_gate(INTERNAL_SIZE_LIMIT_BYTES - 1)

    with pytest.raises(ValueError, match="95,000,000"):
        enforce_submission_size_gate(INTERNAL_SIZE_LIMIT_BYTES)


def test_custom_head_parameter_estimate_matches_fixed_architecture() -> None:
    assert estimate_custom_head_parameters(
        backbone_dim=2048,
        embedding_dim=256,
        num_classes=40,
    ) == 535_336


def test_weight_inventory_records_file_identity(tmp_path: Path) -> None:
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"official-weights")

    inventory = inventory_weight_file(
        weight,
        parameter_count=7,
        fp32_parameter_bytes=28,
    )

    assert inventory["path"] == str(weight.resolve())
    assert inventory["serialized_bytes"] == 16
    assert inventory["parameter_count"] == 7
    assert inventory["fp32_parameter_bytes"] == 28
    assert inventory["sha256"] == (
        "fea5088286ab8aaa4a11f54da5d535981b50c904726d6d748f05c93ade5334bf"
    )


def test_powershell_utf16_log_is_decoded_without_nul_characters(tmp_path: Path) -> None:
    log = tmp_path / "pip.log"
    log.write_text("Successfully installed pytorchvideo-0.1.5\n", encoding="utf-16")

    decoded = read_text_with_detected_encoding(log)

    assert decoded.splitlines() == ["Successfully installed pytorchvideo-0.1.5"]
    assert "\x00" not in decoded
