from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from imu_rf.helpers import write_feature_root
from src.data.imu_stage2_contracts import sha256_file
from src.training.imu_rf_compact import train_compact_candidate
from src.training.imu_rf_finalization import (
    canonical_tree_state_sha256,
    compare_forest_structures,
    compare_reproduction_runs,
    select_formal_source_run,
    validate_frozen_estimator,
    validate_sklearn_compatibility,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def _formal_root(root: Path) -> Path:
    root.mkdir()
    _write_json(
        root / "compact_rf_summary.json",
        {
            "status": "success",
            "finalists": {"primary": "trees_150_leaf4", "fallback": "trees_150"},
            "budget_winners": {"8": "trees_150_leaf4"},
        },
    )
    _write_json(
        root / "compact_rf_comparison.json",
        {
            "comparison_version": "imu-rf-compact-comparison-v1",
            "finalists": {"primary": "trees_150_leaf4", "fallback": "trees_150"},
            "runs": [
                {
                    "candidate_id": "trees_150_leaf4",
                    "random_state": 20260725,
                    "model_sha256": "a" * 64,
                    "run_manifest_sha256": "b" * 64,
                }
            ],
        },
    )
    structural = root / "structural_screen_seed20260725" / "trees_150_leaf4"
    formal = root / "multiseed_confirmation" / "trees_150_leaf4" / "seed20260725"
    structural.mkdir(parents=True)
    formal.mkdir(parents=True)
    return formal


def test_source_selection_uses_formal_multiseed_record_not_structural_copy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "compact"
    expected = _formal_root(root)
    assert select_formal_source_run(root) == expected.resolve()


def test_source_selection_rejects_ambiguous_formal_records(tmp_path: Path) -> None:
    root = tmp_path / "compact"
    _formal_root(root)
    comparison_path = root / "compact_rf_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["runs"].append(dict(comparison["runs"][0]))
    _write_json(comparison_path, comparison)
    with pytest.raises(ValueError, match="unique"):
        select_formal_source_run(root)


def test_source_selection_rejects_non_primary_candidate(tmp_path: Path) -> None:
    root = tmp_path / "compact"
    _formal_root(root)
    summary_path = root / "compact_rf_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["finalists"]["primary"] = "trees_150"
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="primary"):
        select_formal_source_run(root)


def test_environment_compatibility_requires_same_sklearn_major_minor() -> None:
    validate_sklearn_compatibility("1.6.1", "1.6.7")
    with pytest.raises(ValueError, match="scikit-learn"):
        validate_sklearn_compatibility("1.5.2", "1.6.1")
    with pytest.raises(ValueError, match="version"):
        validate_sklearn_compatibility("not-a-version", "1.6.1")


def _forest(seed: int) -> RandomForestClassifier:
    rng = np.random.default_rng(123)
    features = rng.normal(size=(60, 6))
    labels = np.repeat(np.arange(3), 20)
    features[:, 0] += labels * 2
    return RandomForestClassifier(
        n_estimators=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    ).fit(features, labels)


def test_tree_state_sha_is_deterministic_and_structure_comparison_is_exact() -> None:
    first = _forest(17)
    second = _forest(17)
    digest = canonical_tree_state_sha256(first)
    assert digest == canonical_tree_state_sha256(second)
    comparison = compare_forest_structures(first, second)
    assert comparison["equal"] is True
    assert comparison["tree_count"] == 5
    assert comparison["source_tree_state_sha256"] == digest
    assert comparison["fresh_tree_state_sha256"] == digest


def test_tree_structure_comparison_rejects_float_change_beyond_tolerance() -> None:
    first = _forest(17)
    second = _forest(17)
    second.estimators_[0].tree_.threshold[0] += 1e-12
    comparison = compare_forest_structures(first, second)
    assert comparison["equal"] is False
    assert comparison["first_mismatch"]["field"] == "threshold"


def test_frozen_estimator_validator_checks_type_and_every_parameter() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(30, 4))
    labels = np.repeat(np.arange(3), 10)
    model = RandomForestClassifier(
        n_estimators=150,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=4,
        max_features="sqrt",
        max_leaf_nodes=None,
        bootstrap=True,
        max_samples=None,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=20260725,
    ).fit(features, labels)
    validate_frozen_estimator(model)
    model.n_jobs = 1
    with pytest.raises(ValueError, match="n_jobs"):
        validate_frozen_estimator(model)
    with pytest.raises(ValueError, match="RandomForestClassifier"):
        validate_frozen_estimator(object())


def _compact_config(path: Path) -> Path:
    _write_json(
        path,
        {
            "config_version": "imu-rf-compact-screen-v1",
            "fold": 0,
            "num_classes": 3,
            "feature_schema_version": "imu-rf-summary-v1",
            "model_compression": {"method": "lzma", "level": 3},
            "compression_screen": [
                {"compression_id": "lzma_3", "method": "lzma", "level": 3}
            ],
            "structural_seed": 20260725,
            "random_states": [20260725],
            "structural_candidates": [
                {
                    "candidate_id": "trees_150_leaf4",
                    "n_estimators": 5,
                    "max_depth": None,
                    "min_samples_leaf": 2,
                    "min_samples_split": 2,
                    "max_samples": None,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": "balanced_subsample",
                }
            ],
            "selection_thresholds": {
                "primary_max_mib": 16,
                "primary_macro_f1_drop": 0.015,
                "primary_accuracy_drop": 0.02,
                "primary_weighted_f1_drop": 0.02,
                "primary_zero_f1_increase": 2,
                "fallback_max_mib": 32,
                "fallback_macro_f1_drop": 0.01,
                "fallback_accuracy_drop": 0.015,
                "fallback_weighted_f1_drop": 0.02,
                "fallback_zero_f1_increase": 1,
                "maximum_prediction_share": 0.25,
            },
            "budgets_mib": [8],
        },
    )
    return path


def _tiny_reproducible_runs(tmp_path: Path) -> tuple[Path, Path]:
    features = write_feature_root(tmp_path / "features")
    config = _compact_config(tmp_path / "compact.json")
    source = tmp_path / "source"
    fresh = tmp_path / "fresh"
    for output in (source, fresh):
        train_compact_candidate(
            feature_root=features,
            config_path=config,
            candidate_id="trees_150_leaf4",
            random_state=20260725,
            output_dir=output,
            preflight_only=False,
        )
    return source, fresh


def test_reproduction_comparison_requires_exact_predictions_metrics_and_trees(
    tmp_path: Path,
) -> None:
    source, fresh = _tiny_reproducible_runs(tmp_path)
    comparison = compare_reproduction_runs(source, fresh)
    assert comparison["reproducibility_status"] == "exact_match"
    assert comparison["prediction_mismatch_count"] == 0
    assert comparison["probability_maximum_absolute_difference"] <= 1e-15
    assert comparison["probability_nonzero_difference_count"] >= 0
    assert comparison["tree_comparison"]["equal"] is True


def test_reproduction_comparison_rejects_probability_change_over_1e15(
    tmp_path: Path,
) -> None:
    source, fresh = _tiny_reproducible_runs(tmp_path)
    output_path = fresh / "validation_outputs.npz"
    with np.load(output_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["class_probabilities"][0, 0] += 2e-12
    arrays["class_probabilities"][0, 1] -= 2e-12
    np.savez(output_path, **arrays)
    manifest_path = fresh / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item for item in manifest["files"] if item["relative_path"] == "validation_outputs.npz"
    )
    record["size"] = output_path.stat().st_size
    record["sha256"] = sha256_file(output_path)
    _write_json(manifest_path, manifest)
    comparison = compare_reproduction_runs(source, fresh)
    assert comparison["reproducibility_status"] == "failed"
    assert comparison["probability_maximum_absolute_difference"] > 1e-15
    assert "probabilities" in comparison["failed_gates"]
