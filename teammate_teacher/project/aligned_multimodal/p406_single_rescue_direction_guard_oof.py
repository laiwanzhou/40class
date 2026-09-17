"""P399 direction guard allowing one source rescue but still zero source harm."""
from __future__ import annotations

import json
from pathlib import Path

import p399_candidate_conditioned_raw_detail_head_oof as p399


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p406_single_rescue_direction_guard_oof_v1"


def main():
    print(
        "P406 allows a directed P399 route after one source-LOSO rescue only when source "
        "harm is zero and no source cohort/user regresses; outer held folds remain final.",
        flush=True,
    )
    p399.OUT = OUT
    p399.DIRECTION_GUARD = True
    p399.MIN_DIRECTION_RESCUE = 1
    p399.MIN_DIRECTION_POSITIVE_USERS = 1
    p399.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P406_single_rescue_direction_guard_OOF"
    report["protocol"]["direction_guard"] = {
        "minimum_rescue": 1,
        "harm": 0,
        "minimum_cohort_gain": 0,
        "minimum_user_gain": 0,
        "minimum_positive_users": 1,
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: one-rescue zero-harm directed guard around the P399 raw-detail head.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
