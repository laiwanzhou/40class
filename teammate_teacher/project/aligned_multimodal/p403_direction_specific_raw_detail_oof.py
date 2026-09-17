"""P399 shared detail head with independent source-safe thresholds per direction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p399_candidate_conditioned_raw_detail_head_oof as p399


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p403_direction_specific_raw_detail_oof_v1"


def select_rules(source, proposal, margin):
    rules = []
    disagreement = proposal != source["base"]
    for base_class, target_class in sorted(
        set(zip(source["base"][disagreement].tolist(), proposal[disagreement].tolist()))
    ):
        pair = disagreement & (source["base"] == base_class) & (proposal == target_class)
        values = np.unique(
            np.concatenate(
                (
                    [np.inf],
                    np.linspace(0.0, 0.8, 41),
                    np.quantile(margin[pair], np.linspace(0.0, 1.0, 11)),
                )
            )
        )
        best = None
        for threshold in values:
            route = pair & (margin >= threshold)
            rescue = route & (source["base"] != source["labels"]) & (proposal == source["labels"])
            harm = route & (source["base"] == source["labels"]) & (proposal != source["labels"])
            gain = rescue.astype(int) - harm.astype(int)
            per_cohort = {
                cohort: int(gain[source["cohort"] == cohort].sum())
                for cohort in np.unique(source["cohort"])
            }
            per_user = {
                user: int(gain[source["users"] == user].sum())
                for user in np.unique(source["users"])
            }
            row = {
                "base_class": int(base_class),
                "target_class": int(target_class),
                "threshold": float(threshold),
                "changed": int(route.sum()),
                "rescue": int(rescue.sum()),
                "harm": int(harm.sum()),
                "net": int(rescue.sum() - harm.sum()),
                "minimum_cohort_gain": min(per_cohort.values()),
                "minimum_user_gain": min(per_user.values()),
                "positive_users": sum(value > 0 for value in per_user.values()),
                "per_cohort": per_cohort,
            }
            eligible = (
                row["rescue"] >= 2
                and row["harm"] == 0
                and row["minimum_cohort_gain"] >= 0
                and row["minimum_user_gain"] >= 0
                and row["positive_users"] >= 2
            )
            key = (
                eligible,
                row["minimum_cohort_gain"],
                row["net"],
                row["rescue"],
                -row["changed"],
                float(threshold),
            )
            if best is None or key > best[0]:
                best = (key, row)
        if best[0][0]:
            rules.append(best[1])
    rules.sort(
        key=lambda row: (
            row["minimum_cohort_gain"], row["net"], row["rescue"], -row["changed"]
        ),
        reverse=True,
    )
    return rules


def apply_rules(base, proposal, margin, rules):
    route = np.zeros(len(base), dtype=bool)
    for rule in rules:
        route |= (
            (base == int(rule["base_class"]))
            & (proposal == int(rule["target_class"]))
            & (margin >= float(rule["threshold"]))
        )
    output = base.copy()
    output[route] = proposal[route]
    return output, route


def source_metrics(source, output, route):
    base = source["base"]
    labels = source["labels"]
    gain = (output == labels).astype(int) - (base == labels).astype(int)
    per_cohort = {
        cohort: int(gain[source["cohort"] == cohort].sum())
        for cohort in np.unique(source["cohort"])
    }
    return {
        "changed": int(route.sum()),
        "rescue": int(np.sum(route & (base != labels) & (output == labels))),
        "harm": int(np.sum(route & (base == labels) & (output != labels))),
        "net": int(np.sum(output == labels) - np.sum(base == labels)),
        "minimum_cohort_gain": min(per_cohort.values()),
        "per_cohort": per_cohort,
    }


def main():
    print(
        "P403 keeps the P399 shared raw-detail scorer but replaces its global threshold "
        "with zero-harm source-LOSO thresholds for each directed base->candidate pair.",
        flush=True,
    )
    parts = p399.load_data()
    physical_dim, motion_dim = p399.attach_raw(parts)
    report = {
        "stage": "P403_direction_specific_raw_detail_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "detail_head": "P399 shared L1 candidate scorer",
            "authorization": "independent source-LOSO threshold per directed base->candidate pair",
            "direction_requirements": "rescue>=2, harm=0, both cohorts nonnegative, no user regression, two positive users",
            "held_labels_used_for_model_or_rule": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
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
                rules = select_rules(source, proposal, margin)
                source_output, source_route = apply_rules(source["base"], proposal, margin, rules)
                metrics = source_metrics(source, source_output, source_route)
                key = (
                    metrics["minimum_cohort_gain"] >= 0,
                    metrics["net"],
                    metrics["rescue"],
                    -metrics["harm"],
                    -metrics["changed"],
                    -len(rules),
                    -k,
                    -c_value,
                )
                if best is None or key > best[0]:
                    best = (key, k, c_value, rules, metrics)
        _, k, c_value, rules, metrics = best
        model = p399.fit_model(source, k, c_value)
        proposal, margin = p399.predict(model, parts[held], k)
        output, route = apply_rules(parts[held]["base"], proposal, margin, rules)
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
            "selected_k": k,
            "selected_C": c_value,
            "source_rules": rules,
            "source_metrics": metrics,
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
        print(json.dumps({"held": held, "k": k, "C": c_value, "rules": rules, "source": metrics, "result": report["cohorts"][held]["held"]}), flush=True)
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
    np.savez_compressed(OUT / "oof_predictions.npz", labels=labels, base_prediction=base, prediction=prediction)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: P399 raw-detail scorer with per-direction zero-harm source thresholds.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
