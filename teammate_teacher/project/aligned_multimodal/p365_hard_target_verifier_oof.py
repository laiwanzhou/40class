"""Source-safe one-vs-rest verifiers for current hard target classes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p361_bidirectional_existing_teacher_gate_oof import COHORTS, load_parts


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p365_hard_target_verifier_oof_v1"
CS = (0.003, 0.01, 0.03, 0.1)
KS = (3, 5)


def features(part, target):
    bank = np.asarray(part["bank"], np.float32)
    group = np.asarray(part["group_probability"], np.float32)
    base = part["base"].astype(int)
    rows = np.arange(len(base))
    pt = np.clip(bank[:, :, target], 1e-7, 1.0)
    pb = np.clip(bank[rows[:, None], np.arange(bank.shape[1])[None, :], base[:, None]], 1e-7, 1.0)
    vote = bank.argmax(axis=2)
    order = np.argsort(np.argsort(-group, axis=1, kind="stable"), axis=1) + 1
    scalar = np.stack(
        (
            pt.mean(1), pt.max(1), pt.std(1),
            (vote == target).mean(1),
            group[:, target], group[rows, base],
            np.log(np.clip(group[:, target], 1e-7, 1.0)) - np.log(np.clip(group[rows, base], 1e-7, 1.0)),
            order[:, target] / 40.0,
        ),
        axis=1,
    )
    return np.concatenate(
        (np.sqrt(pt), np.log(pt) - np.log(pb), scalar, np.eye(40, dtype=np.float32)[base]),
        axis=1,
    ).astype(np.float32)


def eligible(part, target, k):
    group_topk = np.argsort(-part["group_probability"], axis=1, kind="stable")[:, :k]
    expert_vote = part["bank"].argmax(axis=2)
    return (
        (part["base"] != target)
        & (np.any(group_topk == target, axis=1) | np.any(expert_vote == target, axis=1))
    )


def fit_predict(train, held, target, c_value, k):
    mask = eligible(train, target, k)
    y = (train["labels"] == target).astype(int)
    if int(y[mask].sum()) < 4 or int((1 - y[mask]).sum()) < 8:
        return None
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=20260903,
        ),
    )
    classifier.fit(features(train, target)[mask], y[mask])
    return classifier.predict_proba(features(held, target))[:, 1]


def concatenate(items):
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in ("ids", "users", "labels", "base", "bank", "group_probability", "cohort")
    }


def source_targets(source, k):
    targets = []
    for target in range(40):
        errors = (source["labels"] == target) & (source["base"] != target)
        oracle = errors & eligible(source, target, k)
        per_cohort_errors = [int(errors[source["cohort"] == cohort].sum()) for cohort in np.unique(source["cohort"])]
        if int(errors.sum()) >= 7 and int(oracle.sum()) >= 5 and min(per_cohort_errors) >= 1:
            targets.append(target)
    return targets


def choose_threshold(source, target, k, score):
    candidate = eligible(source, target, k)
    values = np.unique(
        np.concatenate(
            ([0.0, 1.0], np.linspace(0.50, 0.995, 100), np.quantile(score[candidate], np.linspace(0.1, 0.95, 18)))
        )
    )
    best = None
    for threshold in values:
        route = candidate & (score >= threshold)
        proposal = np.full(len(score), target, dtype=int)
        gain = route.astype(int) * (
            (proposal == source["labels"]).astype(int)
            - (source["base"] == source["labels"]).astype(int)
        )
        per_cohort = {
            cohort: int(gain[source["cohort"] == cohort].sum())
            for cohort in np.unique(source["cohort"])
        }
        per_user = {
            user: int(gain[source["users"] == user].sum())
            for user in np.unique(source["users"])
        }
        rescue = int(np.sum(route & (source["base"] != source["labels"]) & (source["labels"] == target)))
        harm = int(np.sum(route & (source["base"] == source["labels"])))
        result = {
            "threshold": float(threshold),
            "changed": int(route.sum()),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "minimum_cohort_gain": min(per_cohort.values()),
            "positive_cohorts": sum(value > 0 for value in per_cohort.values()),
            "minimum_user_gain": min(per_user.values()),
            "positive_users": sum(value > 0 for value in per_user.values()),
            "per_cohort": per_cohort,
            "per_user": per_user,
        }
        if result["changed"] == 0:
            continue
        key = (
            result["minimum_cohort_gain"] >= 1,
            harm == 0,
            result["positive_users"] >= 2,
            result["net"],
            rescue,
            -result["changed"],
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, result)
    if best is None:
        return {
            "threshold": float("inf"),
            "changed": 0,
            "rescue": 0,
            "harm": 0,
            "net": 0,
            "minimum_cohort_gain": 0,
            "positive_cohorts": 0,
            "minimum_user_gain": 0,
            "positive_users": 0,
            "per_cohort": {},
            "per_user": {},
        }
    return best[1]


def main():
    print(
        "P365 tests one-vs-rest hard-target verifiers over existing teacher probabilities; "
        "target discovery, fitting, and thresholds are source-only.",
        flush=True,
    )
    parts, names = load_parts()
    for cohort in COHORTS:
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
    report = {
        "stage": "P365_hard_target_verifier_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "expert_count": len(names),
            "specialist": "one-vs-rest target verifier",
            "candidate_support": "P307 Top-K or any existing teacher Top-1",
            "target_discovery_source_only": True,
            "inner_cross_cohort_thresholds": True,
            "held_labels_used_for_selection": False,
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
        best = None
        all_audits = []
        for k in KS:
            targets = source_targets(source, k)
            for c_value in CS:
                rules = []
                for target in targets:
                    scores = []
                    valid = True
                    for train_name, validation_name in ((source_names[0], source_names[1]), (source_names[1], source_names[0])):
                        prediction = fit_predict(parts[train_name], parts[validation_name], target, c_value, k)
                        if prediction is None:
                            valid = False
                            break
                        scores.append((validation_name, prediction))
                    if not valid:
                        continue
                    score = np.empty(len(source["labels"]), dtype=float)
                    offset = {source_names[0]: (0, len(parts[source_names[0]]["labels"])), source_names[1]: (len(parts[source_names[0]]["labels"]), len(source["labels"]))}
                    for validation_name, prediction in scores:
                        lo, hi = offset[validation_name]
                        score[lo:hi] = prediction
                    threshold = choose_threshold(source, target, k, score)
                    audit = {"target_class": target, "k": k, "C": c_value, **threshold}
                    all_audits.append(audit)
                    if (
                        threshold["rescue"] >= 2
                        and threshold["harm"] == 0
                        and threshold["minimum_cohort_gain"] >= 1
                        and threshold["positive_users"] >= 2
                        and threshold["minimum_user_gain"] >= 0
                    ):
                        rules.append(audit)
                key = (
                    sum(rule["net"] for rule in rules),
                    sum(rule["rescue"] for rule in rules),
                    -sum(rule["harm"] for rule in rules),
                    -len(rules),
                    -k,
                    -c_value,
                )
                if best is None or key > best[0]:
                    best = (key, rules)
        rules = best[1] if best is not None else []
        output = parts[held]["base"].copy()
        best_excess = np.full(len(output), -np.inf, dtype=float)
        for rule in rules:
            score = fit_predict(source, parts[held], int(rule["target_class"]), float(rule["C"]), int(rule["k"]))
            if score is None:
                continue
            route = (
                eligible(parts[held], int(rule["target_class"]), int(rule["k"]))
                & (score >= float(rule["threshold"]))
                & ((score - float(rule["threshold"])) > best_excess)
            )
            output[route] = int(rule["target_class"])
            best_excess[route] = score[route] - float(rule["threshold"])
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        changed = output != base
        report["cohorts"][held] = {
            "source": source_names,
            "rules": rules,
            "source_audit_count": len(all_audits),
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
        "Run 1: source-safe one-vs-rest hard-target verifier over existing teachers.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": report["aggregate"], "folds": {cohort: report["cohorts"][cohort]["held"] for cohort in COHORTS}, "rules": {cohort: report["cohorts"][cohort]["rules"] for cohort in COHORTS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
