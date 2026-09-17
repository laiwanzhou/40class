"""Fold-gated combination of P347 and the supported P353 Top-3 reranker."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "p354_p347_p353_strict_fold_gate_oof_v1"
SPLITS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
SIZES = (663, 834, 973)


def main() -> None:
    p310 = np.load(RUNS / "p310_union_repeat_precedence_teacher_v1/oof_predictions.npz")
    p347 = np.load(RUNS / "p347_loso_twohead_gate_v1/oof_predictions.npz")
    p353 = np.load(RUNS / "p353_p245_top3_supported_pair_reranker_oof_v1/oof_predictions.npz")
    labels = p310["labels"].astype(int)
    base = p310["prediction"].astype(int)
    out = p347["prediction"].astype(int).copy()
    pair_route = p353["prediction"] != p353["base_prediction"]
    out[pair_route] = p353["prediction"][pair_route]

    folds = []
    start = 0
    for name, size in zip(SPLITS, SIZES):
        sl = slice(start, start + size)
        changed = out[sl] != base[sl]
        rescue = changed & (base[sl] != labels[sl]) & (out[sl] == labels[sl])
        harm = changed & (base[sl] == labels[sl]) & (out[sl] != labels[sl])
        folds.append({
            "cohort": name,
            "rows": size,
            "base_correct": int(np.sum(base[sl] == labels[sl])),
            "correct": int(np.sum(out[sl] == labels[sl])),
            "net": int(np.sum(out[sl] == labels[sl]) - np.sum(base[sl] == labels[sl])),
            "changed": int(np.sum(changed)),
            "rescue": int(np.sum(rescue)),
            "harm": int(np.sum(harm)),
            "p347_routes": int(np.sum(p347["prediction"][sl] != base[sl])),
            "p353_routes": int(np.sum(pair_route[sl])),
        })
        start += size
    nets = [row["net"] for row in folds]
    strict_gate = all(net > 0 for net in nets) or (sum(net > 0 for net in nets) >= 2 and min(nets) >= -1)
    report = {
        "stage": "P354_P347_P353_strict_supported_gate",
        "status": "pass" if strict_gate else "rejected",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "priority": "P347 nested-LOSO gate, then P353 supported Top-3 pair route",
            "pair_minimum_source_top3_errors": 7,
            "combination_uses_labels": False,
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
    np.savez_compressed(OUT / "oof_predictions.npz", labels=labels, base_prediction=base, prediction=out, p347_prediction=p347["prediction"], p353_prediction=p353["prediction"], p353_route=pair_route)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
