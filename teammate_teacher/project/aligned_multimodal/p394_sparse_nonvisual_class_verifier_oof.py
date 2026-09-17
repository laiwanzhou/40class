"""Sparse per-class nonvisual verifiers inside visual/P307 Top-K disagreement scope."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p386_visual_scope_nonvisual_competence_gate import NONVISUAL_TEACHERS, load_data


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p394_sparse_nonvisual_class_verifier_oof_v1"
VISUAL_REFERENCE = "strong_visual_mean"
KS = (3, 5)
CS = (0.03, 0.10, 0.30)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
MAX_RULES = 4
MINIMUM_RESCUE = 2
MINIMUM_COHORT_GAIN = 1


def candidate_mask(part, k):
    visual = part["visual_references"][VISUAL_REFERENCE]
    visual_top = np.argsort(-visual, axis=1, kind="stable")[:, :k]
    group_top = np.argsort(-part["group_probability"], axis=1, kind="stable")[:, :k]
    visual_mask = np.zeros((len(visual), 40), dtype=bool)
    group_mask = np.zeros((len(visual), 40), dtype=bool)
    rows = np.arange(len(visual))[:, None]
    visual_mask[rows, visual_top] = True
    group_mask[rows, group_top] = True
    return visual_mask & group_mask, visual.argmax(axis=1) != part["base"]


def feature_names():
    names = ["p307_group_candidate_minus_base"]
    for teacher in NONVISUAL_TEACHERS:
        names.extend(
            (
                f"{teacher}:logodds_candidate_vs_base",
                f"{teacher}:candidate_probability",
                f"{teacher}:candidate_rank",
                f"{teacher}:candidate_top1",
                f"{teacher}:candidate_top3",
                f"{teacher}:candidate_top5",
            )
        )
    return names


def features(part, target):
    probability = np.clip(part["nonvisual_probability"].astype(np.float64), 1e-7, 1.0)
    rows = np.arange(len(part["base"]))
    rank = np.argsort(np.argsort(-probability, axis=2, kind="stable"), axis=2) + 1
    values = [
        (
            part["group_probability"][:, target]
            - part["group_probability"][rows, part["base"]]
        )[:, None]
    ]
    for teacher in range(len(NONVISUAL_TEACHERS)):
        candidate_probability = probability[:, teacher, target]
        base_probability = probability[rows, teacher, part["base"]]
        candidate_rank = rank[:, teacher, target]
        values.append(
            np.column_stack(
                (
                    np.log(candidate_probability) - np.log(base_probability),
                    candidate_probability,
                    candidate_rank / 40.0,
                    candidate_rank <= 1,
                    candidate_rank <= 3,
                    candidate_rank <= 5,
                )
            )
        )
    return np.concatenate(values, axis=1).astype(np.float32)


def scope(part, target, k):
    candidates, disagreement = candidate_mask(part, k)
    return disagreement & (part["base"] != target) & candidates[:, target]


def fit_predict(train, held, target, k, c_value):
    selected = scope(train, target, k)
    labels = (train["labels"] == target).astype(int)
    positives = int(labels[selected].sum())
    negatives = int(selected.sum() - positives)
    if positives < 3 or negatives < 8:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            l1_ratio=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=20260903,
        ),
    )
    model.fit(features(train, target)[selected], labels[selected])
    probability = model.predict_proba(features(held, target))[:, 1]
    coefficient = model.named_steps["logisticregression"].coef_[0]
    return probability, coefficient


def concatenate(items):
    result = {}
    for key in (
        "ids", "users", "labels", "base", "group_probability", "nonvisual_probability", "cohort"
    ):
        result[key] = np.concatenate([item[key] for item in items], axis=0)
    result["visual_references"] = {
        VISUAL_REFERENCE: np.concatenate(
            [item["visual_references"][VISUAL_REFERENCE] for item in items], axis=0
        )
    }
    return result


def choose_threshold(source, score, target, k):
    valid_scope = scope(source, target, k)
    best = None
    for threshold in THRESHOLDS:
        route = valid_scope & (score >= threshold)
        output = source["base"].copy()
        output[route] = target
        gain = (output == source["labels"]).astype(int) - (source["base"] == source["labels"]).astype(int)
        per_cohort = {cohort: int(gain[source["cohort"] == cohort].sum()) for cohort in np.unique(source["cohort"])}
        per_user = {user: int(gain[source["users"] == user].sum()) for user in np.unique(source["users"])}
        rescue = int(np.sum(route & (source["base"] != source["labels"]) & (source["labels"] == target)))
        harm = int(np.sum(route & (source["base"] == source["labels"])))
        row = {
            "threshold": threshold,
            "changed": int(route.sum()),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "minimum_cohort_gain": min(per_cohort.values()),
            "minimum_user_gain": min(per_user.values()),
            "positive_users": sum(value > 0 for value in per_user.values()),
            "per_cohort": per_cohort,
        }
        eligible = (
            rescue >= MINIMUM_RESCUE
            and harm == 0
            and row["minimum_cohort_gain"] >= MINIMUM_COHORT_GAIN
            and row["minimum_user_gain"] >= 0
            and row["positive_users"] >= 2
        )
        key = (
            eligible,
            row["minimum_cohort_gain"],
            row["net"],
            row["rescue"],
            -row["changed"],
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, row)
    return best[1], bool(best[0][0])


def main():
    print(
        "P394 trains one L1-sparse verifier per target class using only nonvisual relative "
        "posterior/rank features inside the visual/P307 Top-K disagreement scope.",
        flush=True,
    )
    parts = load_data()
    for cohort in COHORTS:
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
    names = feature_names()
    report = {
        "stage": "P394_sparse_nonvisual_class_verifier_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_role": "strong visual mean disagreement and Top-K scope only",
            "candidate_set": "visual/P307 Top-K intersection",
            "features": "nonvisual candidate-vs-base relative log-odds, probability, rank, Top1/3/5",
            "model": "per-target L1 logistic regression",
            "nonvisual_teachers": list(NONVISUAL_TEACHERS),
            "candidate_k": list(KS),
            "regularization_C": list(CS),
            "held_labels_used_for_training_rule_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source = concatenate([parts[cohort] for cohort in source_names])
        offsets = {
            source_names[0]: (0, len(parts[source_names[0]]["labels"])),
            source_names[1]: (len(parts[source_names[0]]["labels"]), len(source["labels"])),
        }
        best = None
        all_audits = []
        for k in KS:
            for c_value in CS:
                rules = []
                for target in range(40):
                    score = np.zeros(len(source["labels"]), dtype=float)
                    valid = True
                    for train_name, validation_name in (
                        (source_names[0], source_names[1]),
                        (source_names[1], source_names[0]),
                    ):
                        result = fit_predict(parts[train_name], parts[validation_name], target, k, c_value)
                        if result is None:
                            valid = False
                            break
                        lo, hi = offsets[validation_name]
                        score[lo:hi] = result[0]
                    if not valid:
                        continue
                    threshold, eligible = choose_threshold(source, score, target, k)
                    all_audits.append(
                        {"target_class": target, "k": k, "C": c_value, "eligible": eligible, **threshold}
                    )
                    if eligible:
                        rules.append(
                            {
                                "target_class": target,
                                "k": k,
                                "C": c_value,
                                **threshold,
                            }
                        )
                rules.sort(
                    key=lambda row: (
                        row["minimum_cohort_gain"], row["net"], row["rescue"], -row["changed"]
                    ),
                    reverse=True,
                )
                rules = rules[:MAX_RULES]
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
            result = fit_predict(source, parts[held], target, int(rule["k"]), float(rule["C"]))
            if result is None:
                continue
            score, coefficient = result
            route = (
                scope(parts[held], target, int(rule["k"]))
                & (score >= float(rule["threshold"]))
                & ((score - float(rule["threshold"])) > best_excess)
            )
            output[route] = target
            best_excess[route] = score[route] - float(rule["threshold"])
            nonzero = np.flatnonzero(np.abs(coefficient) > 1e-10)
            coefficient_records.append(
                {
                    "target_class": target,
                    "nonzero_feature_count": int(len(nonzero)),
                    "selected_features": [names[index] for index in nonzero],
                    "coefficients": [float(coefficient[index]) for index in nonzero],
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
    labels = np.concatenate([parts[cohort]["labels"] for cohort in COHORTS])
    base = np.concatenate([parts[cohort]["base"] for cohort in COHORTS])
    prediction = np.concatenate(outputs)
    fold_nets = [report["cohorts"][cohort]["held"]["net"] for cohort in COHORTS]
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
        "Run 1: per-target L1 sparse nonvisual verifier behind visual/P307 candidate scope.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
