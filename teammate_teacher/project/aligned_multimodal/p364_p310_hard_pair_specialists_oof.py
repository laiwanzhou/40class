"""Run the existing Top-K pair specialist protocol directly on P310/P315.

This changes neither the teacher bank nor the specialist architecture.  It
only replaces the older P245 base/posterior with the current P310 decisions
and their matched P307 group posterior, so the audit targets today's errors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p311_p245_topk_pair_reranker_oof as pair


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p364_p310_hard_pair_specialists_oof_v1"
P307 = HERE / "runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz"
P310 = HERE / "runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz"
ORIGINAL_BUILD = pair.build


def build_current():
    data, names = ORIGINAL_BUILD()
    group = np.load(P307)
    current = np.load(P310)
    offset = 0
    for cohort in pair.SPLITS:
        rows = len(data[cohort]["labels"])
        labels = data[cohort]["labels"].astype(int)
        if not np.array_equal(labels, current["labels"][offset : offset + rows]):
            raise RuntimeError(f"P310 alignment failure: {cohort}")
        posterior = group[f"{cohort}_group_probability"].astype(np.float32)
        data[cohort]["posterior"] = posterior
        data[cohort]["base"] = current["prediction"][offset : offset + rows].astype(int)
        data[cohort]["order"] = np.argsort(-posterior, axis=1, kind="stable")
        offset += rows
    return data, names


def main():
    print(
        "P364 tests small Top-3/Top-5 pair specialists against the current P310/P315 errors; "
        "no new teacher and no Test data are used.",
        flush=True,
    )
    pair.O = OUT
    pair.KS = (3, 5)
    pair.build = build_current
    pair.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P364_P310_Top3_Top5_hard_pair_specialists_OOF"
    report["protocol"]["base"] = "P310/P315 label-identical teacher"
    report["protocol"]["posterior"] = "P307 matched 30-teacher group posterior"
    report["aggregate"]["net_vs_p310"] = report["aggregate"].pop("net_vs_p245")
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: current-base Top-K pair specialists using the existing teacher bank.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
