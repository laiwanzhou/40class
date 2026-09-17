"""Risk audit of P384 with minimum source-user gain relaxed from -1 to -2."""
from __future__ import annotations

import json
from pathlib import Path

import p384_crossfit_class_teacher_competence_gate as p384


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p385_relaxed_user_class_competence_gate_v1"


def main():
    print(
        "P385 is a risk audit only: it relaxes P384's minimum source-user gain to -2 "
        "while keeping cohort constraints and all other thresholds unchanged.",
        flush=True,
    )
    p384.OUT = OUT
    p384.MINIMUM_USER_GAIN = -2
    p384.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P385_relaxed_user_class_competence_gate"
    report["protocol"]["minimum_source_user_gain"] = -2
    report["protocol"]["deployment_status"] = "risk_audit_not_automatically_promotable"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: risk-only P384 variant with minimum source-user gain -2.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
