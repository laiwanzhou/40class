"""Subject-safe Radar pair specialists over the current P310 hard pool."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p311_p245_topk_pair_reranker_oof import choose, eligible
from p361_bidirectional_existing_teacher_gate_oof import COHORTS, load_parts


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p377_radar_hard_pair_specialists_oof_v1"
RADAR = HERE / "runs/p89_radar_temporal_expert_v1"
P307 = HERE / "runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz"
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
KS = (3, 5)


def radar_key(sample_id):
    action, user, trial = str(sample_id).split("/")
    return int(action.split("_", 1)[0]), user, trial


def current_key(sample_id):
    _, class_code, user, trial = str(sample_id).split("__")
    return int(class_code[1:]), user, trial


def load_data():
    parts, names = load_parts()
    group = np.load(P307)
    features = np.load(RADAR / "train_features.npz")
    labels = np.load(RADAR / "oof_logits.npz")["labels"].astype(int)
    radar_ids = features["sample_ids"].astype(str)
    if len(radar_ids) != len(labels):
        raise RuntimeError("Radar feature/label row mismatch")
    cache = {
        "ids": radar_ids,
        "labels": labels,
        "users": np.asarray([value.split("/")[1] for value in radar_ids]),
        "features": features["features"].astype(np.float32),
    }
    position = {radar_key(sample_id): index for index, sample_id in enumerate(radar_ids)}
    for cohort in COHORTS:
        probability = group[f"{cohort}_group_probability"].astype(np.float32)
        parts[cohort]["posterior"] = probability
        parts[cohort]["order"] = np.argsort(-probability, axis=1, kind="stable")
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
        index = np.asarray([position.get(current_key(sample_id), -1) for sample_id in parts[cohort]["ids"]])
        parts[cohort]["radar_index"] = index
        parts[cohort]["radar_available"] = index >= 0
    return parts, cache, names


def concatenate(items):
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in (
            "ids", "users", "labels", "base", "bank", "posterior", "order",
            "cohort", "radar_index", "radar_available",
        )
    }


def source_pairs(part, k):
    pairs = {}
    labels = part["labels"]
    base = part["base"]
    hit = np.any(part["order"][:, :k] == labels[:, None], axis=1)
    for row in np.flatnonzero(part["radar_available"] & (base != labels) & hit):
        pair = tuple(sorted((int(base[row]), int(labels[row]))))
        pairs[pair] = pairs.get(pair, 0) + 1
    return {pair for pair, count in pairs.items() if count >= 2}


def fit_predict(cache, held, left, right, alpha, excluded_users):
    train = np.isin(cache["labels"], (left, right)) & ~np.isin(cache["users"], list(excluded_users))
    if int(np.sum(cache["labels"][train] == left)) < 3 or int(np.sum(cache["labels"][train] == right)) < 3:
        return None
    model = make_pipeline(
        StandardScaler(),
        RidgeClassifier(
            alpha=float(alpha), class_weight="balanced", solver="lsqr", tol=1e-5, max_iter=5000
        ),
    )
    model.fit(cache["features"][train], cache["labels"][train])
    output = np.zeros(len(held["labels"]), dtype=float)
    available = held["radar_available"]
    output[available] = model.decision_function(cache["features"][held["radar_index"][available]])
    return output


def main():
    print(
        "P377 tests low-capacity Radar pair specialists only where Radar is non-empty and "
        "the alternate class is already in P307 Top-3/Top-5.",
        flush=True,
    )
    parts, cache, teacher_names = load_data()
    report = {
        "stage": "P377_Radar_hard_pair_specialists_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "radar_rows": len(cache["ids"]),
            "feature_dimensions": cache["features"].shape[1],
            "model": "standardized class-balanced Ridge pair head",
            "outer_and_inner_held_users_excluded": True,
            "availability_used_only_as_abstention": True,
            "absolute_timestamp_or_device_id_used": False,
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
        outer_users = set(parts[held]["users"].tolist())
        offsets = {
            source_names[0]: (0, len(parts[source_names[0]]["labels"])),
            source_names[1]: (len(parts[source_names[0]]["labels"]), len(source["labels"])),
        }
        best = None
        audits = 0
        for k in KS:
            pairs = source_pairs(source, k)
            for alpha in ALPHAS:
                rules = []
                for left, right in sorted(pairs):
                    proposal = source["base"].copy()
                    score = np.full(len(proposal), -np.inf, dtype=float)
                    valid = True
                    for validation_name in source_names:
                        excluded = outer_users | set(parts[validation_name]["users"].tolist())
                        decision = fit_predict(cache, parts[validation_name], left, right, alpha, excluded)
                        if decision is None:
                            valid = False
                            break
                        lo, hi = offsets[validation_name]
                        available = parts[validation_name]["radar_available"]
                        proposal[lo:hi][available] = np.where(decision[available] >= 0.0, right, left)
                        score[lo:hi][available] = np.abs(decision[available])
                    if not valid:
                        continue
                    row = choose(score, proposal, source, left, right, k)
                    row.update({"alpha": alpha})
                    audits += 1
                    if (
                        row["rescue"] >= 2
                        and row["harm"] == 0
                        and row["minimum_user_gain"] >= 0
                        and row["minimum_cohort_gain"] >= 0
                        and row["positive_users"] >= 2
                    ):
                        rules.append(row)
                key = (
                    sum(rule["net"] for rule in rules),
                    sum(rule["rescue"] for rule in rules),
                    -sum(rule["harm"] for rule in rules),
                    -len(rules),
                    -k,
                    -alpha,
                )
                if best is None or key > best[0]:
                    best = (key, k, alpha, rules)
        _, k, alpha, rules = best
        output = parts[held]["base"].copy()
        best_score = np.full(len(output), -np.inf, dtype=float)
        for rule in rules:
            left, right = rule["pair"]
            decision = fit_predict(cache, parts[held], left, right, float(rule["alpha"]), outer_users)
            if decision is None:
                continue
            proposal = np.where(decision >= 0.0, right, left)
            score = np.abs(decision)
            route = (
                eligible(parts[held], left, right, int(rule["k"]))
                & parts[held]["radar_available"]
                & (proposal != parts[held]["base"])
                & (score >= float(rule["threshold"]))
                & (score > best_score)
            )
            output[route] = proposal[route]
            best_score[route] = score[route]
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        changed = output != base
        report["cohorts"][held] = {
            "source": source_names,
            "selected_k": k,
            "selected_alpha": alpha,
            "rules": rules,
            "source_audit_count": audits,
            "held_radar_available": int(parts[held]["radar_available"].sum()),
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
        print(json.dumps({"held": held, "rules": rules, "result": report["cohorts"][held]["held"]}), flush=True)

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
        "Run 1: subject-safe Radar pair specialists with availability abstention.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
