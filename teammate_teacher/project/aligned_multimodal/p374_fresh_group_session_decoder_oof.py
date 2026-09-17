"""Fresh decoder directly over the pure P307 30-teacher group output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p373_fresh_p315_session_decoder_oof as decoder


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p374_fresh_group_session_decoder_oof_v1"
ORIGINAL_LOAD = decoder.load_parts


def load_group_parts():
    data, parts = ORIGINAL_LOAD()
    group = np.load(decoder.P307)
    for cohort in decoder.COHORTS:
        parts[cohort]["base"] = group[f"{cohort}_group_prediction"].astype(int)
    return data, parts


def main():
    print(
        "P374 removes the old sequence decision entirely and learns a fresh decoder directly "
        "on the pure P307 30-teacher group posterior.",
        flush=True,
    )
    decoder.OUT = OUT
    decoder.load_parts = load_group_parts
    decoder.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    archive = np.load(OUT / "oof_predictions.npz")
    p310 = np.load(decoder.P310)
    prediction = archive["prediction"].astype(int)
    labels = archive["labels"].astype(int)
    p310_prediction = p310["prediction"].astype(int)
    report["stage"] = "P374_fresh_group_session_decoder_OOF"
    report["protocol"]["base"] = "pure P307 30-teacher group prediction"
    report["protocol"]["old_P270_or_P310_sequence_decision_used"] = False
    report["aggregate"]["p310_reference_correct"] = int(np.sum(p310_prediction == labels))
    report["aggregate"]["net_vs_p310"] = int(np.sum(prediction == labels) - np.sum(p310_prediction == labels))
    fold_lengths = [663, 834, 973]
    offsets = np.cumsum([0, *fold_lengths])
    report["aggregate"]["fold_nets_vs_p310"] = [
        int(
            np.sum(prediction[offsets[index] : offsets[index + 1]] == labels[offsets[index] : offsets[index + 1]])
            - np.sum(p310_prediction[offsets[index] : offsets[index + 1]] == labels[offsets[index] : offsets[index + 1]])
        )
        for index in range(3)
    ]
    strict = report["aggregate"]["fold_nets_vs_p310"]
    report["aggregate"]["strict_gate_pass_vs_p310"] = bool(
        all(net > 0 for net in strict)
        or (sum(net > 0 for net in strict) >= 2 and min(strict) >= -1)
    )
    report["aggregate"]["decision_vs_p310"] = (
        "eligible_for_test_refit"
        if report["aggregate"]["strict_gate_pass_vs_p310"]
        else "reject_before_test"
    )
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: old sequence removed; fresh decoder learned directly on P307 group probabilities.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
