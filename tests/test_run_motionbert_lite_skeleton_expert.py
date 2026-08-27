from __future__ import annotations

from pathlib import Path

from src.train_motionbert_lite_skeleton_expert import run_motionbert_smoke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml"


def test_smoke_loads_pretraining_and_updates_head_only(tmp_path: Path) -> None:
    report = run_motionbert_smoke(CONFIG, output_root=tmp_path / "smoke")

    assert report["status"] == "smoke_passed"
    assert report["pretrained_element_coverage"] >= 0.99
    assert report["finite_forward_backward"] is True
    assert report["changed_parameter_groups"] == ["head"]
    assert report["pretrained_random_embedding_max_abs_delta"] > 0
    assert set(report["gradient_user_ids"]).isdisjoint({"user6", "user7"})
    assert report["validation_forward_rows"] == 2
    assert report["peak_cuda_mib"] < 8151
    assert report["head_reload_exact"] is True
