"""LOSO source cross-prediction for sparse per-class nonvisual verifiers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p394_sparse_nonvisual_class_verifier_oof as sparse


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p395_loso_sparse_nonvisual_class_verifier_oof_v1"
THRESHOLD_TRANSFER = "absolute"


def subset(part, mask):
    result = {}
    for key, value in part.items():
        if key == "visual_references":
            result[key] = {
                name: probability[mask] for name, probability in value.items()
            }
        elif isinstance(value, np.ndarray) and len(value) == len(mask):
            result[key] = value[mask]
        else:
            result[key] = value
    return result


def main():
    print(
        "P395 replaces single-cohort class fitting with leave-one-subject-out prediction "
        "over both source cohorts, preserving held-user exclusion while increasing support.",
        flush=True,
    )
    parts = sparse.load_data()
    for cohort in sparse.COHORTS:
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
    names = sparse.feature_names()
    report = {
        "stage": "P395_LOSO_sparse_nonvisual_class_verifier_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_role": "strong visual mean disagreement and Top-K scope only",
            "candidate_set": "visual/P307 Top-K intersection",
            "features": "nonvisual candidate-vs-base relative log-odds, probability, rank, Top1/3/5",
            "model": "per-target L1 logistic regression",
            "source_cross_prediction": "leave one subject out across both source cohorts",
            "threshold_transfer": THRESHOLD_TRANSFER,
            "held_labels_used_for_training_rule_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    for held in sparse.COHORTS:
        source_names = [cohort for cohort in sparse.COHORTS if cohort != held]
        source = sparse.concatenate([parts[cohort] for cohort in source_names])
        best = None
        all_audits = []
        for k in sparse.KS:
            for c_value in sparse.CS:
                rules = []
                for target in range(40):
                    target_scope = sparse.scope(source, target, k)
                    positive = int(np.sum(target_scope & (source["labels"] == target)))
                    negative = int(np.sum(target_scope & (source["labels"] != target)))
                    if positive < 4 or negative < 8:
                        continue
                    score = np.zeros(len(source["labels"]), dtype=float)
                    valid = True
                    for user in np.unique(source["users"]):
                        validation = source["users"] == user
                        training = ~validation
                        result = sparse.fit_predict(
                            subset(source, training), subset(source, validation), target, k, c_value
                        )
                        if result is None:
                            valid = False
                            break
                        score[validation] = result[0]
                    if not valid:
                        continue
                    threshold, eligible = sparse.choose_threshold(source, score, target, k)
                    audit = {
                        "target_class": target,
                        "k": k,
                        "C": c_value,
                        "positive_scope_rows": positive,
                        "negative_scope_rows": negative,
                        "eligible": eligible,
                        **threshold,
                    }
                    all_audits.append(audit)
                    if eligible:
                        rules.append(audit)
                rules.sort(
                    key=lambda row: (
                        row["minimum_cohort_gain"], row["net"], row["rescue"], -row["changed"]
                    ),
                    reverse=True,
                )
                rules = rules[: sparse.MAX_RULES]
                key = (
                    sum(rule["net"] for rule in rules),
                    sum(rule["rescue"] for rule in rules),
                    -sum(rule["harm"] for rule in rules),
                    -len(rules),
                    -k,
                    -c_value,
                )
                if best is None or key > best[0]:
                    best = (key, k, c_value, rules)
        _, k, c_value, rules = best
        output = parts[held]["base"].copy()
        best_excess = np.full(len(output), -np.inf, dtype=float)
        coefficient_records = []
        for rule in rules:
            target = int(rule["target_class"])
            result = sparse.fit_predict(
                source, parts[held], target, int(rule["k"]), float(rule["C"])
            )
            if result is None:
                continue
            score, coefficient = result
            held_scope = sparse.scope(parts[held], target, int(rule["k"]))
            threshold = float(rule["threshold"])
            if THRESHOLD_TRANSFER == "quantile" and held_scope.any():
                source_scope_rows = int(rule["positive_scope_rows"]) + int(rule["negative_scope_rows"])
                route_fraction = float(rule["changed"]) / max(source_scope_rows, 1)
                selected_count = min(
                    int(held_scope.sum()),
                    max(1, int(np.ceil(route_fraction * int(held_scope.sum())))),
                )
                threshold = float(
                    np.partition(score[held_scope], len(score[held_scope]) - selected_count)[
                        len(score[held_scope]) - selected_count
                    ]
                )
            elif THRESHOLD_TRANSFER != "absolute":
                raise ValueError(THRESHOLD_TRANSFER)
            route = (
                held_scope
                & (score >= threshold)
                & ((score - threshold) > best_excess)
            )
            output[route] = target
            best_excess[route] = score[route] - threshold
            nonzero = np.flatnonzero(np.abs(coefficient) > 1e-10)
            coefficient_records.append(
                {
                    "target_class": target,
                    "nonzero_feature_count": int(len(nonzero)),
                    "selected_features": [names[index] for index in nonzero],
                    "coefficients": [float(coefficient[index]) for index in nonzero],
                    "held_threshold": threshold,
                }
            )
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        changed = output != base
        report["cohorts"][held] = {
            "source": source_names,
            "selected_k": k,
            "selected_C": c_value,
            "rules": rules,
            "sparse_coefficients": coefficient_records,
            "top_rejected_source_audits": sorted(
                all_audits,
                key=lambda row: (
                    row["minimum_cohort_gain"], row["net"], row["rescue"],
                    -row["harm"], -row["changed"],
                ),
                reverse=True,
            )[:20],
            "held": {
                "rows": len(labels),
                "base_correct": int(np.sum(base == labels)),
                "correct": int(np.sum(output == labels)),
                "net": int(np.sum(output == labels) - np.sum(base == labels)),
                "changed": int(changed.sum()),
                "rescue": int(np.sum(changed & (base != labels) & (output == labels))),
                "harm": int(np.sum(changed & (base == labels) & (output != labels))),
            },
        }
        print(json.dumps({"held": held, "k": k, "C": c_value, "rules": rules, "sparse": coefficient_records, "result": report["cohorts"][held]["held"]}, ensure_ascii=False), flush=True)

    labels = np.concatenate([parts[cohort]["labels"] for cohort in sparse.COHORTS])
    base = np.concatenate([parts[cohort]["base"] for cohort in sparse.COHORTS])
    prediction = np.concatenate(outputs)
    fold_nets = [report["cohorts"][cohort]["held"]["net"] for cohort in sparse.COHORTS]
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
        "Run 1: pooled-source LOSO sparse nonvisual per-class verifier.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
