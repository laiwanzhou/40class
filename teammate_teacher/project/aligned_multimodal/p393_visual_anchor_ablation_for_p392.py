"""Fixed visual-anchor ablation for the P392 nonvisual Top-K tournament."""
from __future__ import annotations

import json
from pathlib import Path

import p392_visual_topk_nonvisual_pair_tournament_oof as p392


HERE = Path(__file__).resolve().parent
ROOT_OUT = HERE / "runs/p393_visual_anchor_ablation_for_p392_v1"
ANCHORS = ("p90_visual_equal", "p123_old_ir_dense", "strong_visual_mean")


def main():
    print(
        "P393 runs the identical P392 tournament with each visual scope anchor fixed, "
        "testing whether anchor choice caused cohort instability.",
        flush=True,
    )
    results = {}
    original_selector = p392.select_visual_reference
    for anchor in ANCHORS:
        output = ROOT_OUT / anchor
        p392.OUT = output
        p392.select_visual_reference = lambda _parts, _source, name=anchor: (name, [])
        p392.main()
        results[anchor] = json.loads((output / "summary.json").read_text(encoding="utf-8"))["aggregate"]
    p392.select_visual_reference = original_selector
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "P393_visual_anchor_ablation_for_P392",
        "status": "complete",
        "protocol": {
            "only_changed_variable": "fixed pure-visual scope anchor",
            "anchors": list(ANCHORS),
            "held_labels_used_for_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
        },
        "results": results,
    }
    (ROOT_OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT_OUT / "notes.txt").write_text(
        "Run 1: fixed visual-anchor ablation for P392.\n" + json.dumps(results, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
