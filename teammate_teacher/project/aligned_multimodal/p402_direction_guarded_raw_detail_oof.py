"""P399 with source-only directed rescue/harm guard."""
from __future__ import annotations

import json
from pathlib import Path

import p399_candidate_conditioned_raw_detail_head_oof as p399


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p402_direction_guarded_raw_detail_oof_v1"


def main():
    print(
        "P402 adds a source-only directed pair guard to P399: each base->candidate direction "
        "requires at least two rescues, zero harm, and no source cohort/user regression.",
        flush=True,
    )
    p399.OUT = OUT
    p399.DIRECTION_GUARD = True
    p399.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P402_direction_guarded_raw_detail_OOF"
    report["protocol"]["direction_guard"] = {
        "minimum_rescue": 2,
        "harm": 0,
        "minimum_cohort_gain": 0,
        "minimum_user_gain": 0,
        "minimum_positive_users": 2,
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: P399 candidate scorer plus source-only directed pair harm guard.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
