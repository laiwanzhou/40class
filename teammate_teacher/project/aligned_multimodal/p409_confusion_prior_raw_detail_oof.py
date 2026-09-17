"""P399 sparse raw-detail head with continuous source confusion-prior features."""
from __future__ import annotations

import json
from pathlib import Path

import p399_candidate_conditioned_raw_detail_head_oof as p399


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p409_confusion_prior_raw_detail_oof_v1"


def main():
    print(
        "P409 adds continuous source-only base->target confusion support, user coverage, "
        "smoothed rate, and reverse support to the P399 sparse raw-detail head.",
        flush=True,
    )
    p399.OUT = OUT
    p399.USE_CONFUSION_PRIOR = True
    p399.DIRECTION_GUARD = False
    p399.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P409_confusion_prior_raw_detail_OOF"
    report["protocol"]["confusion_prior_features"] = {
        "base_to_target_error_count": True,
        "base_to_target_user_count": True,
        "smoothed_rate_within_base_errors": True,
        "reverse_direction_count": True,
        "target_error_support": True,
        "computed_from_model_training_users_only": True,
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: P399 sparse raw-detail head plus continuous source confusion priors.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
