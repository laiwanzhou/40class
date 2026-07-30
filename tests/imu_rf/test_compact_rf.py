from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from imu_rf.helpers import write_feature_root, write_rf_config
from src.training.imu_rf_compact import (
    compact_pareto_frontier,
    forest_structure_metrics,
    load_compact_config,
    measure_model_roundtrip,
    probabilities_equivalent,
    run_compact_screen,
    select_compact_finalists,
    train_compact_candidate,
)
from src.training.imu_rf_trainer import RF_RUN_FILES, train_random_forest, validate_rf_run


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def _candidate(candidate_id: str = "compact") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "n_estimators": 12,
        "max_depth": 6,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_samples": 0.8,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced_subsample",
    }


def _config(path: Path) -> Path:
    payload = {
        "config_version": "imu-rf-compact-screen-v1",
        "fold": 0,
        "num_classes": 3,
        "feature_schema_version": "imu-rf-summary-v1",
        "model_compression": {"method": "zlib", "level": 3},
        "compression_screen": [
            {"compression_id": "zlib_1", "method": "zlib", "level": 1},
            {"compression_id": "zlib_3", "method": "zlib", "level": 3},
        ],
        "structural_seed": 20260725,
        "random_states": [20260724, 20260725, 20260726],
        "structural_candidates": [_candidate()],
        "selection_thresholds": {
            "primary_max_mib": 16,
            "primary_macro_f1_drop": 0.015,
            "primary_accuracy_drop": 0.020,
            "primary_weighted_f1_drop": 0.020,
            "primary_zero_f1_increase": 2,
            "fallback_max_mib": 32,
            "fallback_macro_f1_drop": 0.010,
            "fallback_accuracy_drop": 0.015,
            "fallback_weighted_f1_drop": 0.020,
            "fallback_zero_f1_increase": 1,
            "maximum_prediction_share": 0.25,
        },
        "budgets_mib": [8, 16, 32, 50],
    }
    _write_json(path, payload)
    return path


def _tiny_bundle(path: Path) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(31)
    features = rng.normal(size=(30, 8))
    labels = np.repeat(np.arange(3), 10)
    features[:, 0] += labels * 3
    model = RandomForestClassifier(n_estimators=18, random_state=20260724).fit(
        features, labels
    )
    payload = {
        "model": model,
        "metadata": {
            "model_version": "imu-rf-model-v1",
            "feature_schema_sha256": "a" * 64,
            "imputer_sha256": "b" * 64,
            "class_order_sha256": "c" * 64,
        },
    }
    joblib.dump(payload, path, compress=0)
    return path, features


def test_lossless_compression_roundtrip_preserves_model_and_measures_exact_bytes(
    tmp_path: Path,
) -> None:
    source, features = _tiny_bundle(tmp_path / "source.joblib")
    destination = tmp_path / "compressed.joblib"
    result = measure_model_roundtrip(
        source_path=source,
        destination_path=destination,
        validation_features=features,
        compression_method="zlib",
        compression_level=3,
    )
    assert result["compressed_bytes"] == destination.stat().st_size
    assert result["uncompressed_bytes"] == source.stat().st_size
    assert result["prediction_equal"] is True
    assert result["probability_equal"] is True
    assert result["metadata_equal"] is True
    assert result["structure_equal"] is True
    assert result["load_seconds"] >= 0
    assert result["batch_inference_seconds"] >= 0
    assert result["single_inference_seconds"] >= 0


def test_probability_equivalence_accepts_only_machine_roundoff() -> None:
    baseline = np.asarray([[0.1, 0.2, 0.7]], dtype=np.float64)
    machine_roundoff = baseline.copy()
    machine_roundoff[0, 2] = np.nextafter(machine_roundoff[0, 2], np.inf)
    material_change = baseline.copy()
    material_change[0, 2] += 1e-12
    assert probabilities_equivalent(baseline, machine_roundoff) is True
    assert probabilities_equivalent(baseline, material_change) is False


def test_structure_metrics_match_estimator_nodes_and_depths(tmp_path: Path) -> None:
    source, _ = _tiny_bundle(tmp_path / "source.joblib")
    model = joblib.load(source)["model"]
    result = forest_structure_metrics(model)
    assert result["n_estimators"] == 18
    assert result["total_nodes"] == sum(tree.tree_.node_count for tree in model.estimators_)
    assert result["maximum_tree_depth"] == max(
        tree.tree_.max_depth for tree in model.estimators_
    )
    assert result["estimated_model_array_bytes"] > 0


def test_compact_config_is_strict_and_candidate_parameters_are_frozen(tmp_path: Path) -> None:
    path = _config(tmp_path / "compact.json")
    loaded = load_compact_config(path)
    assert loaded["structural_candidates"] == [_candidate()]
    changed = dict(loaded)
    changed["unknown"] = True
    _write_json(path, changed)
    with pytest.raises(ValueError, match="field set"):
        load_compact_config(path)
    changed.pop("unknown")
    changed["structural_candidates"] = [{**_candidate(), "n_estimators": 12.0}]
    _write_json(path, changed)
    with pytest.raises(ValueError, match="n_estimators"):
        load_compact_config(path)


def test_compact_candidate_is_reproducible_and_publishes_existing_ten_file_contract(
    tmp_path: Path,
) -> None:
    features = write_feature_root(tmp_path / "features")
    config = _config(tmp_path / "compact.json")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = train_compact_candidate(
        feature_root=features,
        config_path=config,
        candidate_id="compact",
        random_state=20260724,
        output_dir=first,
        preflight_only=False,
    )
    train_compact_candidate(
        feature_root=features,
        config_path=config,
        candidate_id="compact",
        random_state=20260724,
        output_dir=second,
        preflight_only=False,
    )
    assert {path.name for path in first.iterdir()} == RF_RUN_FILES
    first_arrays = validate_rf_run(first)
    second_arrays = validate_rf_run(second)
    np.testing.assert_array_equal(first_arrays["predictions"], second_arrays["predictions"])
    np.testing.assert_allclose(
        first_arrays["class_probabilities"], second_arrays["class_probabilities"], rtol=0, atol=0
    )
    assert first_summary["model_compression"] == {"method": "zlib", "level": 3}
    assert first_summary["compressed_model_bytes"] == (first / "model.joblib").stat().st_size
    assert first_summary["total_nodes"] > 0


def test_finalist_selection_uses_predeclared_gates_not_unqualified_best_score() -> None:
    baseline = {
        "macro_f1": 0.32,
        "validation_accuracy": 0.41,
        "weighted_f1": 0.38,
        "zero_f1_class_count": 9,
    }
    records = [
        {
            "candidate_id": "too_large_best",
            "compressed_mib": 40.0,
            "macro_f1": 0.33,
            "validation_accuracy": 0.42,
            "weighted_f1": 0.39,
            "zero_f1_class_count": 8,
            "maximum_prediction_share": 0.10,
        },
        {
            "candidate_id": "primary",
            "compressed_mib": 12.0,
            "macro_f1": 0.31,
            "validation_accuracy": 0.40,
            "weighted_f1": 0.37,
            "zero_f1_class_count": 10,
            "maximum_prediction_share": 0.12,
        },
        {
            "candidate_id": "fallback",
            "compressed_mib": 24.0,
            "macro_f1": 0.315,
            "validation_accuracy": 0.405,
            "weighted_f1": 0.375,
            "zero_f1_class_count": 9,
            "maximum_prediction_share": 0.12,
        },
    ]
    selected = select_compact_finalists(records, baseline, _config_payload()["selection_thresholds"])
    assert selected == {"primary": "primary", "fallback": "fallback"}


def _config_payload() -> dict[str, object]:
    path = Path("unused")
    payload = {
        "primary_max_mib": 16,
        "primary_macro_f1_drop": 0.015,
        "primary_accuracy_drop": 0.020,
        "primary_weighted_f1_drop": 0.020,
        "primary_zero_f1_increase": 2,
        "fallback_max_mib": 32,
        "fallback_macro_f1_drop": 0.010,
        "fallback_accuracy_drop": 0.015,
        "fallback_weighted_f1_drop": 0.020,
        "fallback_zero_f1_increase": 1,
        "maximum_prediction_share": 0.25,
    }
    return {"path": path, "selection_thresholds": payload}


def test_pareto_frontier_and_budget_winners_are_size_performance_nondominated() -> None:
    records = [
        {"candidate_id": "small", "compressed_mib": 7.0, "macro_f1": 0.28},
        {"candidate_id": "dominated", "compressed_mib": 9.0, "macro_f1": 0.27},
        {"candidate_id": "middle", "compressed_mib": 14.0, "macro_f1": 0.31},
        {"candidate_id": "large", "compressed_mib": 28.0, "macro_f1": 0.32},
    ]
    result = compact_pareto_frontier(records, budgets_mib=[8, 16, 32, 50])
    assert [row["candidate_id"] for row in result["frontier"]] == [
        "small",
        "middle",
        "large",
    ]
    assert result["budget_winners"] == {
        "8": "small",
        "16": "middle",
        "32": "large",
        "50": "large",
    }


def test_compact_cli_help_runs_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts" / "run_imu_rf_compact_screen.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_fresh_compact_screen_publishes_all_phases_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    features = write_feature_root(tmp_path / "features")
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    baseline_config = write_rf_config(tmp_path / "baseline.json")
    for seed in (20260724, 20260725, 20260726):
        train_random_forest(
            feature_root=features,
            config_path=baseline_config,
            variant="balanced",
            random_state=seed,
            output_dir=baseline_root / f"rf_balanced_seed{seed}",
            preflight_only=False,
        )
    config = _config(tmp_path / "compact.json")
    compact_payload = json.loads(config.read_text(encoding="utf-8"))
    compact_payload["selection_thresholds"].update(
        {
            "primary_macro_f1_drop": 1.0,
            "primary_accuracy_drop": 1.0,
            "primary_weighted_f1_drop": 1.0,
            "primary_zero_f1_increase": 3,
            "fallback_macro_f1_drop": 1.0,
            "fallback_accuracy_drop": 1.0,
            "fallback_weighted_f1_drop": 1.0,
            "fallback_zero_f1_increase": 3,
            "maximum_prediction_share": 1.0,
        }
    )
    _write_json(config, compact_payload)
    output = tmp_path / "compact"
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for root in (features, baseline_root)
        for path in root.rglob("*")
        if path.is_file()
    }
    result = run_compact_screen(
        feature_root=features,
        baseline_root=baseline_root,
        config_path=config,
        output_root=output,
        preflight_only=False,
    )
    assert result["status"] == "success"
    assert result["phase_a_runs"] == 6
    assert result["phase_b_runs"] == 1
    assert result["phase_c_runs"] == 3
    assert result["input_snapshot_identical"] is True
    assert {path.name for path in (output / "structural_screen_seed20260725" / "compact").iterdir()} == RF_RUN_FILES
    for seed in (20260724, 20260725, 20260726):
        run = output / "multiseed_confirmation" / "compact" / f"seed{seed}"
        assert {path.name for path in run.iterdir()} == RF_RUN_FILES
        validate_rf_run(run)
    for name in (
        "compact_rf_comparison.csv",
        "compact_rf_comparison.json",
        "compact_rf_pareto.csv",
        "compact_rf_pareto.json",
        "per_class_compact_summary.csv",
        "per_user_compact_summary.csv",
        "compact_rf_summary.json",
        "input_snapshot.json",
    ):
        assert (output / name).is_file()
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for root in (features, baseline_root)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    with pytest.raises(FileExistsError):
        run_compact_screen(
            feature_root=features,
            baseline_root=baseline_root,
            config_path=config,
            output_root=output,
            preflight_only=False,
        )
