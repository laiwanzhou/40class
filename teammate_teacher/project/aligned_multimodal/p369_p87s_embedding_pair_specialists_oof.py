"""Nested hard-pair specialists in the compact P87-S embedding spaces."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

from p311_p245_topk_pair_reranker_oof import choose, eligible, source_pairs
from p361_bidirectional_existing_teacher_gate_oof import COHORTS, load_parts
from p90_teacher_common import load_protocol


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p369_p87s_embedding_pair_specialists_oof_v1"
EMBEDDINGS = HERE / "runs/p333_p87s_outer_embedding_spaces_v1/embedding_spaces.npz"
P307 = HERE / "runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz"
ALPHAS = (10.0, 100.0, 1000.0)
KS = (3, 5)
FEATURE_VARIANTS = ("embedding", "embedding_logits")
SUPPORT_GAP = -np.inf


def l2(values):
    values = np.asarray(values, np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-6, None)


def concatenate(items):
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in ("ids", "users", "labels", "base", "bank", "posterior", "order", "cohort")
    }


def load_data():
    parts, names = load_parts()
    protocol = load_protocol()
    archive = np.load(EMBEDDINGS)
    group = np.load(P307)
    spaces = {}
    for cohort in COHORTS:
        ids = archive[f"{cohort}_sample_ids"].astype(str)
        position = {sample_id: index for index, sample_id in enumerate(ids)}
        if set(position) != set(protocol.sample_ids.astype(str).tolist()):
            raise RuntimeError(f"P87-S embedding sample set mismatch: {cohort}")
        order = np.asarray([position[sample_id] for sample_id in protocol.sample_ids.astype(str)])
        ids = protocol.sample_ids.astype(str)
        embedding = l2(archive[f"{cohort}_embedding"][order])
        logits = archive[f"{cohort}_logits"][order].astype(np.float32)
        motion = archive[f"{cohort}_motion_logits"][order].astype(np.float32)
        spaces[cohort] = {
            "ids": ids,
            "users": protocol.users.astype(str),
            "labels": protocol.labels.astype(int),
            "embedding": embedding,
            "embedding_logits": np.concatenate((embedding, l2(logits), l2(motion)), axis=1).astype(np.float32),
        }
        probability = group[f"{cohort}_group_probability"].astype(np.float32)
        parts[cohort]["posterior"] = probability
        parts[cohort]["order"] = np.argsort(-probability, axis=1, kind="stable")
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
        position = {sample_id: index for index, sample_id in enumerate(ids)}
        parts[cohort]["space_index"] = np.asarray([position[sample_id] for sample_id in parts[cohort]["ids"]])
    return parts, spaces, names


def fit_predict(space, held_part, left, right, alpha, feature, excluded_users):
    train = np.isin(space["labels"], (left, right)) & ~np.isin(space["users"], list(excluded_users))
    if int(np.sum(space["labels"][train] == left)) < 3 or int(np.sum(space["labels"][train] == right)) < 3:
        return None
    model = RidgeClassifier(
        alpha=float(alpha), class_weight="balanced", solver="lsqr", tol=1e-5, max_iter=5000
    )
    model.fit(space[feature][train], space["labels"][train])
    return np.asarray(model.decision_function(space[feature][held_part["space_index"]]), float)


def main():
    print(
        "P369 tests tiny hard-pair readouts in each source-trained compact P87-S embedding "
        "space with nested held-user exclusion.",
        flush=True,
    )
    parts, spaces, teacher_names = load_data()
    report = {
        "stage": "P369_P87S_embedding_pair_specialists_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "representation": "P333 source-trained outer P87-S embedding spaces",
            "feature_variants": list(FEATURE_VARIANTS),
            "model": "class-balanced Ridge pair head",
            "outer_held_users_excluded_from_representation_and_head": True,
            "inner_validation_users_excluded_from_head": True,
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
            for feature in FEATURE_VARIANTS:
                for alpha in ALPHAS:
                    rules = []
                    for left, right in sorted(pairs):
                        proposal = source["base"].copy()
                        score = np.full(len(proposal), -np.inf, dtype=float)
                        valid = True
                        for validation_name in source_names:
                            validation_users = set(parts[validation_name]["users"].tolist())
                            decision = fit_predict(
                                spaces[held],
                                parts[validation_name],
                                left,
                                right,
                                alpha,
                                feature,
                                outer_users | validation_users,
                            )
                            if decision is None:
                                valid = False
                                break
                            lo, hi = offsets[validation_name]
                            proposal[lo:hi] = np.where(decision >= 0.0, right, left)
                            score[lo:hi] = np.abs(decision)
                        if not valid:
                            continue
                        row = choose(score, proposal, source, left, right, k)
                        row.update({"alpha": alpha, "feature": feature})
                        audits += 1
                        if (
                            row["rescue"] >= 2
                            and row["harm"] == 0
                            and row["minimum_user_gain"] >= 0
                            and row["minimum_cohort_gain"] >= 1
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
                        best = (key, k, feature, alpha, rules)

        _, k, feature, alpha, rules = best
        output = parts[held]["base"].copy()
        best_score = np.full(len(output), -np.inf, dtype=float)
        for rule in rules:
            left, right = rule["pair"]
            decision = fit_predict(
                spaces[held],
                parts[held],
                left,
                right,
                float(rule["alpha"]),
                str(rule["feature"]),
                outer_users,
            )
            if decision is None:
                continue
            proposal = np.where(decision >= 0.0, right, left)
            score = np.abs(decision)
            route = (
                eligible(parts[held], left, right, int(rule["k"]))
                & (proposal != parts[held]["base"])
                & (score >= float(rule["threshold"]))
                & (
                    parts[held]["posterior"][np.arange(len(proposal)), proposal]
                    - parts[held]["posterior"][np.arange(len(proposal)), parts[held]["base"]]
                    >= SUPPORT_GAP
                )
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
            "selected_feature": feature,
            "selected_alpha": alpha,
            "rules": rules,
            "source_audit_count": audits,
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
    np.savez_compressed(
        OUT / "oof_predictions.npz",
        labels=labels,
        base_prediction=base,
        prediction=prediction,
        **{f"{cohort}_held_prediction": outputs[index] for index, cohort in enumerate(COHORTS)},
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: nested compact-embedding hard-pair specialists.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": report["aggregate"], "cohorts": {cohort: report["cohorts"][cohort]["held"] for cohort in COHORTS}, "selected": {cohort: {"feature": report["cohorts"][cohort]["selected_feature"], "k": report["cohorts"][cohort]["selected_k"], "alpha": report["cohorts"][cohort]["selected_alpha"], "rules": report["cohorts"][cohort]["rules"]} for cohort in COHORTS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
