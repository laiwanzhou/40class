"""Cross-fit and Test audit for the P347 gate with P279 routing disabled.

P279 remains available as a context feature but cannot be selected as the
replacement candidate.  This is a model-level candidate-pool ablation, not a
row-level exception.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p117_transductive_multicandidate_router import load_candidate_splits
from p191_source_truth_calibrated_p150_distillation import choose
from p346_distributionally_robust_rescue_harm_gate import NAMES, P244, P255, P278, P279, P306, P307, P310, S, cat, part, score, train_pair
from p347_loso_twohead_gate import subset
from p357_full_nested_gate_test_audit import test_part


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "p358_no_p279_nested_gate_audit_v1"
KEEP = np.asarray([0, 1, 2, 4], dtype=int)
KEPT_NAMES = [NAMES[index] for index in KEEP]
SIZES = (663, 834, 973)


def restrict(value):
    result = dict(value)
    for key in ("x", "candidate", "disagree", "rescue", "harm", "gain"):
        if key in result:
            result[key] = result[key][:, KEEP]
    return result


def select_loso(source, cohorts):
    best = None
    for c_value in (0.003, 0.01, 0.03, 0.1):
        proposal = np.zeros(len(source["labels"]), dtype=int)
        robust_score = np.full(len(proposal), -np.inf)
        for user in sorted(set(source["users"].tolist())):
            held = source["users"] == user
            trained = ~held
            predicted, score_value = score([train_pair(subset(source, trained), c_value)], subset(source, held))
            proposal[held] = predicted
            robust_score[held] = score_value
        gain = (proposal == source["labels"]).astype(int) - (source["base"] == source["labels"]).astype(int)
        selected = choose(robust_score, gain, proposal != source["base"], source["users"], cohorts)
        key = (selected["minimum_user_gain"] >= 0, selected["minimum_cohort_gain"] >= 0, selected["net"], selected["rescue"], -selected["harm"], -selected["changed"], -c_value)
        candidate = (key, c_value, selected)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1], best[2]


def main() -> None:
    data = load_candidate_splits()
    z = {
        "p310": np.load(P310), "p244": np.load(P244), "p307": np.load(P307),
        "p255": np.load(P255), "p306": np.load(P306),
        "p328": np.load(RUNS / "p328_p87s_threefold_oof_audit_v1/oof_predictions.npz"),
        "p336": np.load(RUNS / "p336_siglip2_workspace_state_ridge_v1/oof_predictions.npz"),
        "p344": np.load(RUNS / "p344_siglip2_threeview_temporal_oof_v1/oof_predictions.npz"),
        "p279": np.load(P279), "p278": np.load(P278),
    }
    parts = {name: restrict(part(data, name, z)) for name in S}
    p353 = np.load(RUNS / "p353_p245_top3_supported_pair_reranker_oof_v1/oof_predictions.npz")
    p310 = np.load(P310)
    labels = p310["labels"].astype(int)
    base_all = p310["prediction"].astype(int)
    pair_route = p353["prediction"] != p353["base_prediction"]
    outputs = []
    outer = []
    offset = 0
    for held_name, size in zip(S, SIZES):
        source_names = [name for name in S if name != held_name]
        source = cat([parts[name] for name in source_names])
        cohorts = np.concatenate([np.full(len(parts[name]["labels"]), name, object) for name in source_names])
        c_value, selected = select_loso(source, cohorts)
        proposal, robust_score = score([train_pair(source, c_value)], parts[held_name])
        route = (proposal != parts[held_name]["base"]) & (robust_score >= selected["threshold"])
        output = parts[held_name]["base"].copy()
        output[route] = proposal[route]
        local_pair = pair_route[offset:offset + size]
        output[local_pair] = p353["prediction"][offset:offset + size][local_pair]
        labels_fold = parts[held_name]["labels"]
        base_fold = parts[held_name]["base"]
        changed = output != base_fold
        outer.append({
            "cohort": held_name, "C": c_value, "source_selection": selected,
            "held": {
                "rows": size,
                "net": int(np.sum(output == labels_fold) - np.sum(base_fold == labels_fold)),
                "changed": int(np.sum(changed)),
                "rescue": int(np.sum(changed & (base_fold != labels_fold) & (output == labels_fold))),
                "harm": int(np.sum(changed & (base_fold == labels_fold) & (output != labels_fold))),
                "gate_routes": int(np.sum(route)), "pair_routes": int(np.sum(local_pair)),
            },
        })
        outputs.append(output)
        offset += size
    oof = np.concatenate(outputs)
    fold_nets = [row["held"]["net"] for row in outer]
    strict_pass = all(value > 0 for value in fold_nets) or (sum(value > 0 for value in fold_nets) >= 2 and min(fold_nets) >= -1)

    source = cat([parts[name] for name in S])
    cohorts = np.concatenate([np.full(len(parts[name]["labels"]), name, object) for name in S])
    c_value, selected = select_loso(source, cohorts)
    test = restrict(test_part())
    proposal, robust_score = score([train_pair(source, c_value)], test)
    test_route = (proposal != test["base"]) & (robust_score >= selected["threshold"])
    test_prediction = test["base"].copy()
    test_prediction[test_route] = proposal[test_route]
    changed_rows = np.flatnonzero(test_route).tolist()
    frozen = [row for row in changed_rows if row in (77, 283)]
    report = {
        "stage": "P358_no_P279_nested_gate_audit",
        "status": "candidate" if strict_pass and changed_rows and not frozen else "rejected",
        "protocol": {"routable_candidates": KEPT_NAMES, "p279_context_only": True, "test_labels_read": False, "user_id_used_as_feature": False, "submission_generated": False},
        "outer": outer,
        "oof": {"base_correct": int(np.sum(base_all == labels)), "correct": int(np.sum(oof == labels)), "net": int(np.sum(oof == labels) - np.sum(base_all == labels)), "fold_nets": fold_nets, "strict_gate_pass": strict_pass},
        "full_selection": {"C": c_value, **selected},
        "test": {"changes": len(changed_rows), "changed_rows_zero_based": changed_rows, "changed_pairs": [f"{test['base'][row]}->{test_prediction[row]}" for row in changed_rows], "frozen_row_collisions": frozen, "test_labels_read": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "oof_predictions.npz", labels=labels, base_prediction=base_all, prediction=oof)
    np.savez_compressed(OUT / "test_predictions.npz", sample_ids=test["ids"], base_prediction=test["base"], prediction=test_prediction, proposal=proposal, robust_score=robust_score, route=test_route)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
