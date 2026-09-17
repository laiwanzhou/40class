"""P386 with rank-weighted Top-3 support from nonvisual teachers."""
from __future__ import annotations

import json
from pathlib import Path

import p386_visual_scope_nonvisual_competence_gate as p386


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p387_visual_scope_nonvisual_top3_gate_v1"


def main():
    print(
        "P387 extends P386 from nonvisual Top-1 voting to rank-weighted nonvisual Top-3 "
        "support; visual models still only define scope and candidate classes.",
        flush=True,
    )
    p386.OUT = OUT
    p386.NONVISUAL_SUPPORT_K = 3
    p386.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P387_visual_scope_nonvisual_Top3_gate"
    report["protocol"]["nonvisual_support_k"] = 3
    report["protocol"]["rank_weights"] = [1.0, 2.0 / 3.0, 1.0 / 3.0]
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: rank-weighted nonvisual Top-3 competence behind visual disagreement scope.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
