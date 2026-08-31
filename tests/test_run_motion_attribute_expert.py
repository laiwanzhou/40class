from __future__ import annotations

from pathlib import Path

from src.train_motion_attribute_expert import run_motion_attribute_smoke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motion_attribute_expert.yaml"
CACHE = ROOT / "outputs/motion_attribute_expert/input_cache.npz"


def test_motion_attribute_smoke_updates_all_groups_without_validation_gradient(
    tmp_path: Path,
) -> None:
    report = run_motion_attribute_smoke(
        CONFIG, cache_path=CACHE, output_root=tmp_path / "smoke"
    )

    assert report["status"] == "smoke_passed"
    assert report["finite_forward_backward"] is True
    assert report["changed_parameter_groups"] == [
        "encoder",
        "family_head",
        "attribute_head",
        "action_head",
    ]
    assert set(report["gradient_user_ids"]).isdisjoint({"user6", "user7"})
    assert report["validation_forward_rows"] == 2
    assert report["unsupported_fallback_finite"] is True
    assert report["reload_max_abs_logit_delta"] == 0.0
    assert report["peak_cuda_mib"] < 8151
