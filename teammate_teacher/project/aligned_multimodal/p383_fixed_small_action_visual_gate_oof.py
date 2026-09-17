"""P382 visual disagreement gate with the pre-registered 21 small actions."""
from __future__ import annotations

import json
from pathlib import Path

import p382_visual_multimodal_disagreement_gate_oof as p382


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p383_fixed_small_action_visual_gate_oof_v1"
FIXED_SMALL_ACTIONS = (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39)


def fixed_hard_classes(_source):
    return list(FIXED_SMALL_ACTIONS)


def main():
    print(
        "P383 reruns the visual/multimodal disagreement gate with the 21 small actions "
        "that were frozen before the current P315 error audit.",
        flush=True,
    )
    p382.OUT = OUT
    p382.hard_classes = fixed_hard_classes
    p382.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P383_fixed_small_action_visual_gate_OOF"
    report["protocol"]["scope"] = "pre-registered 21 semantic small-action classes"
    report["protocol"]["fixed_small_actions"] = list(FIXED_SMALL_ACTIONS)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: P382 visual disagreement gate with the pre-registered 21 small actions.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
