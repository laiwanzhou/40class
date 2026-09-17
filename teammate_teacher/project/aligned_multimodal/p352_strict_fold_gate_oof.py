"""Combine independently cross-fitted gates and enforce the fold-level gate.

The P347 changes are applied first.  P351 may replace a row only where its
reranker actually changed the P245 base.  No held label participates in the
row-level combination.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "p352_p347_p351_strict_fold_gate_oof_v1"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
SIZES = (663, 834, 973)


def main() -> None:
    p310 = np.load(RUNS / "p310_union_repeat_precedence_teacher_v1/oof_predictions.npz")
    p347 = np.load(RUNS / "p347_loso_twohead_gate_v1/oof_predictions.npz")
    p351 = np.load(RUNS / "p351_p245_top3_pair_reranker_oof_v1/oof_predictions.npz")

    labels = p310["labels"].astype(int)
    base = p310["prediction"].astype(int)
    out = p347["prediction"].astype(int).copy()
    top3_route = p351["prediction"] != p351["base_prediction"]
    out[top3_route] = p351["prediction"][top3_route]

    folds = []
    start = 0
    for name, size in zip(SPLITS, SIZES):
        sl = slice(start, start + size)
        changed = out[sl] != base[sl]
        rescue = changed & (base[sl] != labels[sl]) & (out[sl] == labels[sl])
        harm = changed & (base[sl] == labels[sl]) & (out[sl] != labels[sl])
        folds.append(
            {
                "cohort": name,
                "rows": size,
                "base_correct": int(np.sum(base[sl] == labels[sl])),
                "correct": int(np.sum(out[sl] == labels[sl])),
                "net": int(np.sum(out[sl] == labels[sl]) - np.sum(base[sl] == labels[sl])),
                "changed": int(np.sum(changed)),
                "rescue": int(np.sum(rescue)),
                "harm": int(np.sum(harm)),
                "p347_routes": int(np.sum(p347["prediction"][sl] != base[sl])),
                "p351_routes": int(np.sum(top3_route[sl])),
            }
        )
        start += size

    nets = [row["net"] for row in folds]
    strict_gate = all(net > 0 for net in nets) or (
        sum(net > 0 for net in nets) >= 2 and min(nets) >= -1
    )
    report = {
        "stage": "P352_P347_P351_strict_fold_gate",
        "status": "pass" if strict_gate else "rejected",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "priority": "P347 nested-LOSO gate, then P351 Top-3-only pair route",
            "combination_uses_labels": False,
            "held_labels_used_for_component_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
            "acceptance_gate": "all three folds positive, or at least two positive and worst fold >= -1",
        },
        "folds": folds,
        "aggregate": {
            "rows": len(labels),
            "base_correct": int(np.sum(base == labels)),
            "correct": int(np.sum(out == labels)),
            "accuracy": float(np.mean(out == labels)),
            "net": int(np.sum(out == labels) - np.sum(base == labels)),
            "fold_nets": nets,
            "strict_gate_pass": strict_gate,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "oof_predictions.npz",
        labels=labels,
        base_prediction=base,
        prediction=out,
        p347_prediction=p347["prediction"],
        p351_prediction=p351["prediction"],
        p351_route=top3_route,
    )
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
