"""P399 candidate scorer with source-selected P307 group support guard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p399_candidate_conditioned_raw_detail_head_oof as p399


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p404_group_guarded_raw_detail_oof_v1"
GROUP_THRESHOLDS = (-0.50, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50)


def group_gap(part, proposal):
    rows = np.arange(len(proposal))
    return (
        part["group_probability"][rows, proposal]
        - part["group_probability"][rows, part["base"]]
    )


def choose(source, proposal, margin):
    gap = group_gap(source, proposal)
    best = None
    for margin_threshold in p399.THRESHOLDS:
        for group_threshold in GROUP_THRESHOLDS:
            route = (
                (proposal != source["base"])
                & (margin >= margin_threshold)
                & (gap >= group_threshold)
            )
            output = source["base"].copy()
            output[route] = proposal[route]
            gain = (output == source["labels"]).astype(int) - (source["base"] == source["labels"]).astype(int)
            per_cohort = {cohort: int(gain[source["cohort"] == cohort].sum()) for cohort in np.unique(source["cohort"])}
            per_user = {user: int(gain[source["users"] == user].sum()) for user in np.unique(source["users"])}
            rescue = int(np.sum(route & (source["base"] != source["labels"]) & (output == source["labels"])))
            harm = int(np.sum(route & (source["base"] == source["labels"]) & (output != source["labels"])))
            row = {
                "margin_threshold": float(margin_threshold),
                "group_gap_threshold": float(group_threshold),
                "changed": int(route.sum()),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "minimum_cohort_gain": min(per_cohort.values()),
                "minimum_user_gain": min(per_user.values()),
                "positive_cohorts": sum(value > 0 for value in per_cohort.values()),
                "positive_users": sum(value > 0 for value in per_user.values()),
                "per_cohort": per_cohort,
            }
            valid = row["net"] > 0 and row["minimum_cohort_gain"] >= 0 and row["minimum_user_gain"] >= -1
            key = (
                valid,
                row["minimum_cohort_gain"],
                row["net"],
                row["rescue"],
                -row["harm"],
                -row["changed"],
                margin_threshold,
                group_threshold,
            )
            if best is None or key > best[0]:
                best = (key, row)
    if not best[0][0]:
        best[1].update({"margin_threshold": float("inf"), "group_gap_threshold": float("inf")})
    return best[1]


def main():
    print(
        "P404 jointly selects the P399 detail margin and P307 candidate-vs-base support gap "
        "on source LOSO predictions before evaluating each held cohort.",
        flush=True,
    )
    parts = p399.load_data()
    physical_dim, motion_dim = p399.attach_raw(parts)
    report = {
        "stage": "P404_group_guarded_raw_detail_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "detail_head": "P399 shared L1 candidate scorer",
            "source_selected_thresholds": ["detail margin", "P307 candidate-minus-base probability"],
            "group_gap_grid": list(GROUP_THRESHOLDS),
            "held_labels_used_for_model_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    held_proposals = []
    held_margins = []
    held_routes = []
    for held in p399.COHORTS:
        source_names = [cohort for cohort in p399.COHORTS if cohort != held]
        source = p399.concatenate([parts[cohort] for cohort in source_names])
        best = None
        for k in p399.KS:
            for c_value in p399.CS:
                proposal = source["base"].copy()
                margin = np.full(len(proposal), -np.inf, dtype=float)
                for user in np.unique(source["users"]):
                    validation = source["users"] == user
                    model = p399.fit_model(p399.subset(source, ~validation), k, c_value)
                    proposal[validation], margin[validation] = p399.predict(
                        model, p399.subset(source, validation), k
                    )
                threshold = choose(source, proposal, margin)
                key = (
                    threshold["minimum_cohort_gain"] >= 0,
                    threshold["net"],
                    threshold["rescue"],
                    -threshold["harm"],
                    -threshold["changed"],
                    -k,
                    -c_value,
                )
                if best is None or key > best[0]:
                    best = (key, k, c_value, threshold)
        _, k, c_value, threshold = best
        model = p399.fit_model(source, k, c_value)
        proposal, margin = p399.predict(model, parts[held], k)
        gap = group_gap(parts[held], proposal)
        route = (
            (proposal != parts[held]["base"])
            & (margin >= float(threshold["margin_threshold"]))
            & (gap >= float(threshold["group_gap_threshold"]))
        )
        output = parts[held]["base"].copy()
        output[route] = proposal[route]
        outputs.append(output)
        held_proposals.append(proposal.astype(np.int16))
        held_margins.append(margin.astype(np.float32))
        held_routes.append(route)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
            "selected_k": k,
            "selected_C": c_value,
            "source_threshold": threshold,
            "held": {
                "rows": len(labels),
                "base_correct": int(np.sum(base == labels)),
                "correct": int(np.sum(output == labels)),
                "net": int(np.sum(output == labels) - np.sum(base == labels)),
                "changed": int(route.sum()),
                "rescue": int(np.sum(route & (base != labels) & (output == labels))),
                "harm": int(np.sum(route & (base == labels) & (output != labels))),
            },
        }
        print(json.dumps({"held": held, "k": k, "C": c_value, "threshold": threshold, "result": report["cohorts"][held]["held"]}), flush=True)
    labels = np.concatenate([parts[cohort]["labels"] for cohort in p399.COHORTS])
    base = np.concatenate([parts[cohort]["base"] for cohort in p399.COHORTS])
    prediction = np.concatenate(outputs)
    fold_nets = [report["cohorts"][cohort]["held"]["net"] for cohort in p399.COHORTS]
    strict_pass = bool(
        all(net > 0 for net in fold_nets)
        or (sum(net > 0 for net in fold_nets) >= 2 and min(fold_nets) >= -1)
    )
    report["aggregate"] = {
        "rows": len(labels),
        "base_correct": int(np.sum(base == labels)),
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "net_vs_p310": int(np.sum(prediction == labels) - np.sum(base == labels)),
        "fold_nets": fold_nets,
        "strict_gate_pass": strict_pass,
        "decision": "eligible_for_test_audit" if strict_pass else "reject_before_test",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "oof_predictions.npz",
        labels=labels,
        base_prediction=base,
        prediction=prediction,
        proposal=np.concatenate(held_proposals),
        margin=np.concatenate(held_margins),
        route=np.concatenate(held_routes),
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: P399 detail scorer with source-selected P307 support guard.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
