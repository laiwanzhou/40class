from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.training.imu_rf_trainer import summarize_rf_experiment, train_random_forest
from imu_rf.helpers import write_feature_root, write_rf_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_rf_cli_help_runs_outside_repository(tmp_path: Path) -> None:
    for script in (
        "build_imu_rf_features.py",
        "train_imu_random_forest.py",
        "summarize_imu_random_forest.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / "scripts" / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_summary_publishes_four_reports_and_sample_complementarity(tmp_path: Path) -> None:
    features = write_feature_root(tmp_path / "features")
    config = write_rf_config(tmp_path / "config.json")
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    for variant in ("plain", "balanced"):
        for seed in (20260724, 20260725, 20260726):
            train_random_forest(
                feature_root=features,
                config_path=config,
                variant=variant,
                random_state=seed,
                output_dir=experiment / f"rf_{variant}_seed{seed}",
                preflight_only=False,
            )
    with np.load(
        experiment / "rf_plain_seed20260724" / "validation_outputs.npz",
        allow_pickle=False,
    ) as archive:
        sample_ids = archive["sample_ids"].copy()
        labels = archive["labels"].copy()
        predictions = archive["predictions"].copy()
    baseline = tmp_path / "baseline.npz"
    np.savez(baseline, sample_ids=sample_ids, labels=labels, predictions=predictions)
    summary = summarize_rf_experiment(
        experiment_root=experiment,
        baseline_validation_outputs=baseline,
        config_path=config,
    )
    assert {
        "random_forest_comparison.csv",
        "random_forest_comparison.json",
        "per_class_multiseed_summary.csv",
        "per_user_multiseed_summary.csv",
    }.issubset({path.name for path in experiment.iterdir()})
    assert len(summary["runs"]) == 6
    assert all("rf_only_correct" in row for row in summary["runs"])
    assert set(summary["variants"]) == {"plain", "balanced"}
