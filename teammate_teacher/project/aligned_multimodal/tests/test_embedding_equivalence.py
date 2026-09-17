from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p22_feature_fusion_model import (
    MODALITY_ORDER,
    build_p22_model,
    parameter_count,
)


CACHE_DIR = PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "cache"
PREFLIGHT_PATH = CACHE_DIR / "preflight.json"


def load_preflight() -> dict:
    assert PREFLIGHT_PATH.is_file(), (
        "Run build_p22_fold_conditioned_cache.py before this test; "
        "training is forbidden until preflight.json exists and passes."
    )
    return json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))


def test_matched_logit_control_is_not_larger_than_feature_model() -> None:
    logit_parameters = parameter_count(build_p22_model("P22-L"))
    feature_parameters = parameter_count(build_p22_model("P22-F"))
    assert logit_parameters <= feature_parameters


def test_real_sample_equivalence_passes_every_modality_and_fold() -> None:
    preflight = load_preflight()
    assert preflight["status"] == "passed"
    assert preflight["training_started"] is False
    results = preflight["equivalence"]
    assert len(results) == 12
    observed = {
        (int(result["fold"]), str(result["modality"])) for result in results
    }
    expected = {
        (fold, modality) for fold in range(3) for modality in MODALITY_ORDER
    }
    assert observed == expected
    for result in results:
        assert int(result["real_samples"]) >= 1
        assert float(result["fp32_max_abs_error"]) <= 1e-6
        assert float(result["fp16_max_abs_error"]) <= 1e-2
        assert float(result["argmax_consistency"]) >= 0.9999
        assert result["eval_mode"] is True
        assert result["inference_mode"] is True
        assert result["state_unchanged"] is True


def test_fold_conditioned_caches_have_identity_and_zero_missing_rows() -> None:
    preflight = load_preflight()
    commit = str(preflight["code_commit"])
    expected_presence = {
        "skeleton": 2914,
        "depth": 2914,
        "thermal": 2776,
        "imu": 2855,
    }
    for fold in range(3):
        path = CACHE_DIR / f"fold_{fold}_cache.npz"
        assert path.is_file()
        with np.load(path, allow_pickle=False) as cache:
            sample_ids = cache["sample_ids"].astype(str)
            labels = cache["labels"]
            subjects = cache["subjects"].astype(str)
            folds = cache["folds"]
            presence = cache["presence"]
            assert len(sample_ids) == 2914
            assert len(np.unique(sample_ids)) == 2914
            assert labels.shape == (2914,)
            assert subjects.shape == (2914,)
            assert folds.shape == (2914,)
            assert presence.shape == (2914, 4)
            assert int(cache["outer_fold"].item()) == fold
            assert str(cache["code_commit"].item()) == commit
            assert np.array_equal(
                cache["is_outer_train"], (folds != fold).astype(np.uint8)
            )
            for index, modality in enumerate(MODALITY_ORDER):
                assert int(presence[:, index].sum()) == expected_presence[modality]
                missing = presence[:, index] == 0
                assert np.all(cache[f"{modality}_embedding"][missing] == 0)
                assert np.all(cache["per_modality_logits"][missing, index] == 0)


def test_artifact_manifest_forbids_best_accuracy_checkpoints() -> None:
    manifest = json.loads(
        (PROJECT_DIR / "configs" / "p22_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for fold_info in manifest["folds"]:
        for checkpoint in fold_info["checkpoints"].values():
            assert "best_accuracy" not in checkpoint["path"]
            assert checkpoint["epoch"] in {15, 120}
