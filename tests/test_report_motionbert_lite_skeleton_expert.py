from __future__ import annotations

from pathlib import Path

from scripts.report_motionbert_lite_skeleton_expert import (
    _markdown,
    build_motionbert_p6b_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml"


def test_report_recomputes_metrics_and_stops_after_failed_b1() -> None:
    report = build_motionbert_p6b_report(CONFIG)

    assert report["metrics_recomputed_from_archives"] is True
    assert report["development_validation"] is True
    assert report["independent_final_test"] is False
    assert report["b1_passed"] is False
    assert report["b2_executed"] is False
    assert report["status"] == "stopped_after_b1_rejection"
    assert report["validation_metrics"]["accuracy"] == 0.11855670103092783
    assert report["visual_comparison"]["unique_rescues"] == 5
    assert report["visual_comparison"]["oracle_accuracy"] == 0.7190721649484536
    assert "Zero-recall classes: `36/40`" in _markdown(report)
