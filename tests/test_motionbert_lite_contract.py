from __future__ import annotations

from pathlib import Path

import pytest

from scripts.fetch_motionbert_lite_checkpoint import verify_motionbert_checkpoint
from src.experiments.motionbert_p6b_config import load_motionbert_p6b_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml"
EXPECTED_BYTES = 64_099_897
EXPECTED_SHA256 = "6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461"


def test_motionbert_contract_freezes_one_candidate_and_fixed_boundary() -> None:
    config = load_motionbert_p6b_config(CONFIG)

    assert config["stage"] == "P6-B"
    assert config["candidate"] == "motionbert_lite"
    assert config["population"]["train_samples"] == 2039
    assert config["population"]["validation_samples"] == 388
    assert config["input"] == {
        "frames": 96,
        "joints": 17,
        "channels": ["projected_x", "projected_y", "confidence"],
        "segment_policy": "longest_retained_then_smallest_index",
    }
    assert config["upstream"]["source_commit"] == (
        "705d3a95354db8bdb696b3492e47a3b5537174ff"
    )
    assert config["checkpoint"]["bytes"] == EXPECTED_BYTES
    assert config["checkpoint"]["sha256"] == EXPECTED_SHA256
    assert config["policy"]["grouped_cv_allowed"] is False
    assert config["policy"]["multiple_seeds_allowed"] is False


def test_fetch_rejects_wrong_bytes_or_sha(tmp_path: Path) -> None:
    target = tmp_path / "latest_epoch.bin"
    target.write_bytes(b"wrong")

    with pytest.raises(RuntimeError, match="checkpoint provenance"):
        verify_motionbert_checkpoint(
            target, expected_bytes=EXPECTED_BYTES, expected_sha256=EXPECTED_SHA256
        )
