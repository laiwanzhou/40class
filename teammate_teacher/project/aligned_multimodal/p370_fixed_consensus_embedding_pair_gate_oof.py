"""Exploratory fixed-consensus guard over P369 compact pair specialists."""
from __future__ import annotations

import json
from pathlib import Path

import p369_p87s_embedding_pair_specialists_oof as p369


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p370_fixed_consensus_embedding_pair_gate_oof_v1"


def main():
    print(
        "P370 checks the now-frozen +0.30 P307 posterior support guard over P87-S pair "
        "specialists; this is a post-diagnostic confirmation, not an untouched validation.",
        flush=True,
    )
    p369.OUT = OUT
    p369.SUPPORT_GAP = 0.30
    p369.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P370_fixed_consensus_embedding_pair_gate_OOF"
    report["protocol"]["p307_candidate_probability_advantage"] = 0.30
    report["protocol"]["evidence_status"] = "posthoc_discovery_requires_external_or_future_confirmation"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: fixed +0.30 P307 consensus guard over nested compact-embedding pair specialists.\n"
        "The threshold was frozen after inspecting P369 held rescue/harm and is not an untouched estimate.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
