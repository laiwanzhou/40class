from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_ir_depth_videomaev2_p2a import _metrics
from src.experiments.hierarchical_midfusion_config import (
    load_midfusion_config,
    project_path,
)
from src.models.ir_depth_videomaev2_teacher import sha256_file
from src.train_hierarchical_multimodal_teacher import (
    CANDIDATE_MODALITIES,
    select_grouped_candidate,
)


METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "worst_user_accuracy",
    "nll",
    "top3_accuracy",
    "top5_accuracy",
)


def _validated_archive(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"prediction archive hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "sample_ids",
            "user_ids",
            "labels",
            "logits",
            "core_available",
            "group_attention",
            "segment_attention",
            "context_logits",
            "wrist_logits",
            "body_logits",
            "effective_group_mask",
            "availability",
            "skeleton_quality",
            "imu_quality",
            "skeleton_mask",
            "imu_role_mask",
        }
        if not required.issubset(archive.files):
            raise RuntimeError(f"prediction archive schema changed: {path}")
        sample_ids = archive["sample_ids"].astype(str)
        user_ids = archive["user_ids"].astype(str)
        labels = archive["labels"].astype(np.int64)
        logits = archive["logits"].astype(np.float32)
        if len(sample_ids) != expected_rows or len(np.unique(sample_ids)) != expected_rows:
            raise RuntimeError(f"prediction archive population changed: {path}")
        if logits.shape != (expected_rows, 40) or not np.isfinite(logits).all():
            raise RuntimeError(f"prediction logits changed: {path}")
        if archive["group_attention"].shape != (expected_rows, 40, 3):
            raise RuntimeError(f"group attention shape changed: {path}")
        if archive["segment_attention"].shape != (expected_rows, 40, 8):
            raise RuntimeError(f"segment attention shape changed: {path}")
        if expected_rows == 388 and set(labels.tolist()) != set(range(40)):
            raise RuntimeError("fixed validation class coverage changed")
        metrics = _metrics(labels, logits, user_ids)
    return metrics, (sample_ids, user_ids, labels)


def _assert_metrics_match(
    stored: dict[str, Any], recomputed: dict[str, Any], *, scope: str
) -> None:
    for key in METRIC_KEYS:
        if key not in stored or not np.isclose(
            float(stored[key]), float(recomputed[key]), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"{scope} metric mismatch: {key}")


def build_fixed_validation_report(run_report: Path) -> dict[str, Any]:
    report = json.loads(run_report.read_text(encoding="utf-8"))
    if report.get("evaluation_protocol") != "fixed_user6_user7":
        raise RuntimeError("fixed validation report protocol changed")
    if report.get("development_validation") is not True:
        raise RuntimeError("development validation marker changed")
    if report.get("independent_final_test") is not False:
        raise RuntimeError("independent final-test marker changed")
    if report.get("candidate_order") != list(CANDIDATE_MODALITIES):
        raise RuntimeError("candidate order changed")
    candidate_results = report.get("candidate_results", {})
    if list(candidate_results) != list(CANDIDATE_MODALITIES):
        raise RuntimeError("candidate result set changed")

    train_identity = None
    validation_identity = None
    recomputed_candidates: dict[str, dict[str, Any]] = {}
    for candidate, result in candidate_results.items():
        train_metrics, current_train_identity = _validated_archive(
            Path(result["train_predictions"]),
            expected_sha256=str(result["train_predictions_sha256"]),
            expected_rows=int(report["train_population_samples"]),
        )
        validation_metrics, current_validation_identity = _validated_archive(
            Path(result["validation_predictions"]),
            expected_sha256=str(result["validation_predictions_sha256"]),
            expected_rows=int(report["validation_population_samples"]),
        )
        if train_identity is None:
            train_identity = current_train_identity
            validation_identity = current_validation_identity
        elif any(
            not np.array_equal(left, right)
            for left, right in zip(
                train_identity, current_train_identity, strict=True
            )
        ) or any(
            not np.array_equal(left, right)
            for left, right in zip(
                validation_identity, current_validation_identity, strict=True
            )
        ):
            raise RuntimeError("candidate prediction ownership changed")
        _assert_metrics_match(
            result["train_metrics"], train_metrics, scope=f"{candidate} train"
        )
        _assert_metrics_match(
            result["validation_metrics"],
            validation_metrics,
            scope=f"{candidate} validation",
        )
        recomputed_candidates[candidate] = validation_metrics

    selected = select_grouped_candidate(recomputed_candidates)
    if selected != report.get("selected_candidate"):
        raise RuntimeError("selected candidate changed during recomputation")
    report["metrics_recomputed_from_archives"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify hierarchical multimodal fixed-validation evidence"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml",
    )
    parser.add_argument("--run-report", type=Path)
    args = parser.parse_args()
    config = load_midfusion_config(args.config.resolve())
    report_path = args.run_report or project_path(
        str(config["outputs"]["fixed_validation_report_json"])
    )
    report = build_fixed_validation_report(report_path.resolve())
    print(
        json.dumps(
            {
                "status": "verified",
                "selected_candidate": report["selected_candidate"],
            }
        )
    )


if __name__ == "__main__":
    main()
