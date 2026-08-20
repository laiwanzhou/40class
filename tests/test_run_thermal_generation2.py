from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_thermal_generation2 import main
from src.train_thermal_generation2 import TrainingAuthorizationError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/thermal_a_multistream_direct_train12_val2.yaml"


def test_validate_only_reports_gate_without_model_or_output_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_thermal_generation2",
            "--config",
            str(CONFIG),
            "--validate-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"training_authorized": true' in result.stdout
    assert '"status": "validated_not_started"' in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_cli_refuses_wrong_token_before_creating_output(tmp_path: Path) -> None:
    with pytest.raises(TrainingAuthorizationError):
        main(
            [
                "--config",
                str(CONFIG),
                "--authorize-training",
                "wrong-token",
                "--output-root",
                str(tmp_path),
            ]
        )

    assert list(tmp_path.iterdir()) == []


def test_trainer_does_not_import_frozen_ir_x3d_module() -> None:
    source = (ROOT / "src/train_thermal_generation2.py").read_text(encoding="utf-8")

    assert "train_x3d_s_visual_expert" not in source
