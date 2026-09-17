"""Subject-safe hard-pair specialists using the cached 3,036-row Thermal union."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

from p311_p245_topk_pair_reranker_oof import choose, eligible, source_pairs
from p361_bidirectional_existing_teacher_gate_oof import COHORTS, load_parts


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p368_union_thermal_hard_pair_specialists_oof_v1"
CACHE = HERE / "runs/p294_thermal_scene_union_v1/train_features.npz"
P307 = HERE / "runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz"
ALPHAS = (10.0, 100.0, 1000.0)
KS = (3, 5)
FEATURES = ("latent", "action", "joint")
TRAIN_VARIANTS = ("main_only", "union_augmented")


def l2(values):
    values = np.asarray(values, np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-6, None)


def load_data():
    parts, teacher_names = load_parts()
    group = np.load(P307)
    for cohort in COHORTS:
        probability = group[f"{cohort}_group_probability"].astype(np.float32)
        parts[cohort]["posterior"] = probability
        parts[cohort]["order"] = np.argsort(-probability, axis=1, kind="stable")
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)

    archive = np.load(CACHE)
    ids = archive["sample_ids"].astype(str)
    latent = l2(archive["features"].astype(np.float32).reshape(len(ids), -1))
    action = l2(archive["action_logits"].astype(np.float32).reshape(len(ids), -1))
    cache = {
        "ids": ids,
        "users": archive["users"].astype(str),
        "labels": archive["labels"].astype(int),
        "available": archive["available"].astype(bool),
        "is_extra": np.char.startswith(ids, "extra__"),
        "latent": latent,
        "action": action,
        "joint": np.concatenate((latent, action), axis=1).astype(np.float32),
    }
    position = {sample_id: index for index, sample_id in enumerate(ids)}
    for cohort in COHORTS:
        index = np.asarray([position[sample_id] for sample_id in parts[cohort]["ids"]])
        if not np.array_equal(cache["labels"][index], parts[cohort]["labels"]):
            raise RuntimeError(f"Thermal cache alignment failed: {cohort}")
        parts[cohort]["thermal_index"] = index
        parts[cohort]["thermal_available"] = cache["available"][index]
    return parts, cache, teacher_names


def concatenate(items):
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in (
            "ids", "users", "labels", "base", "bank", "posterior", "order",
            "cohort", "thermal_index", "thermal_available",
        )
    }


def fit_predict(cache, held, left, right, alpha, feature, train_variant, excluded_users):
    train = (
        cache["available"]
        & np.isin(cache["labels"], (left, right))
        & ~np.isin(cache["users"], list(excluded_users))
    )
    if train_variant == "main_only":
        train &= ~cache["is_extra"]
    elif train_variant != "union_augmented":
        raise ValueError(train_variant)
    if int(np.sum(cache["labels"][train] == left)) < 3 or int(np.sum(cache["labels"][train] == right)) < 3:
        return None
    model = RidgeClassifier(
        alpha=float(alpha),
        class_weight="balanced",
        solver="lsqr",
        tol=1e-5,
        max_iter=5000,
    )
    model.fit(cache[feature][train], cache["labels"][train])
    return np.asarray(model.decision_function(cache[feature][held["thermal_index"]]), float)


def main():
    print(
        "P368 tests corrected subject-safe Thermal hard-pair specialists, comparing the "
        "2,914-row main set with the cached 3,036-row union.",
        flush=True,
    )
    parts, cache, teacher_names = load_data()
    report = {
        "stage": "P368_union_Thermal_hard_pair_specialists_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "posterior": "P307 matched group posterior",
            "cache_rows": len(cache["ids"]),
            "extra_rows": int(cache["is_extra"].sum()),
            "extra_thermal_available": int(np.sum(cache["is_extra"] & cache["available"])),
            "outer_held_users_excluded_from_specialist_fit": True,
            "inner_validation_users_excluded_from_specialist_fit": True,
            "feature_variants": list(FEATURES),
            "train_variants": list(TRAIN_VARIANTS),
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
            for feature in FEATURES:
                for train_variant in TRAIN_VARIANTS:
                    for alpha in ALPHAS:
                        rules = []
                        for left, right in sorted(pairs):
                            proposal = source["base"].copy()
                            score = np.full(len(proposal), -np.inf, dtype=float)
                            valid = True
                            for validation_name in source_names:
                                validation_users = set(parts[validation_name]["users"].tolist())
                                decision = fit_predict(
                                    cache,
                                    parts[validation_name],
                                    left,
                                    right,
                                    alpha,
                                    feature,
                                    train_variant,
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
                            row.update({"alpha": alpha, "feature": feature, "train_variant": train_variant})
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
                            train_variant == "union_augmented",
                            -k,
                            -alpha,
                        )
                        if best is None or key > best[0]:
                            best = (key, k, feature, train_variant, alpha, rules)

        _, k, feature, train_variant, alpha, rules = best
        output = parts[held]["base"].copy()
        best_score = np.full(len(output), -np.inf, dtype=float)
        for rule in rules:
            left, right = rule["pair"]
            decision = fit_predict(
                cache,
                parts[held],
                left,
                right,
                float(rule["alpha"]),
                str(rule["feature"]),
                str(rule["train_variant"]),
                outer_users,
            )
            if decision is None:
                continue
            proposal = np.where(decision >= 0.0, right, left)
            score = np.abs(decision)
            route = (
                eligible(parts[held], left, right, int(rule["k"]))
                & parts[held]["thermal_available"]
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
            "selected_feature": feature,
            "selected_train_variant": train_variant,
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
        "Run 1: corrected nested subject exclusion for main-only versus union Thermal pair specialists.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": report["aggregate"], "cohorts": {cohort: report["cohorts"][cohort]["held"] for cohort in COHORTS}, "selected": {cohort: {"feature": report["cohorts"][cohort]["selected_feature"], "train_variant": report["cohorts"][cohort]["selected_train_variant"], "k": report["cohorts"][cohort]["selected_k"], "alpha": report["cohorts"][cohort]["selected_alpha"], "rules": report["cohorts"][cohort]["rules"]} for cohort in COHORTS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
