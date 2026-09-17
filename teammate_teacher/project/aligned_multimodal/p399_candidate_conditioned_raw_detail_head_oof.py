"""Candidate-conditioned sparse detail head using raw cached physical/motion prototypes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p366_cached_feature_hard_target_verifier_oof import motion_features, physical_features
from p386_visual_scope_nonvisual_competence_gate import NONVISUAL_TEACHERS, load_data
from p90_teacher_common import load_protocol


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p399_candidate_conditioned_raw_detail_head_oof_v1"
VISUAL_REFERENCE = "strong_visual_mean"
KS = (3, 5)
CS = (0.01, 0.03, 0.10)
THRESHOLDS = tuple(np.linspace(0.0, 0.8, 41))
DIRECTION_GUARD = False
MIN_DIRECTION_RESCUE = 2
MIN_DIRECTION_POSITIVE_USERS = 2
USE_CONFUSION_PRIOR = False


def l2(values):
    value = np.asarray(values, np.float32)
    return value / np.clip(np.linalg.norm(value, axis=1, keepdims=True), 1e-6, None)


def attach_raw(parts):
    protocol = load_protocol()
    physical = l2(physical_features(protocol))
    motion = l2(motion_features(protocol))
    lookup = {sample_id: index for index, sample_id in enumerate(protocol.sample_ids.astype(str))}
    for cohort in COHORTS:
        index = np.asarray([lookup[sample_id] for sample_id in parts[cohort]["ids"]])
        parts[cohort]["physical"] = physical[index]
        parts[cohort]["motion"] = motion[index]
        parts[cohort]["cohort"] = np.full(len(index), cohort, dtype=object)
    return physical.shape[1], motion.shape[1]


def subset(part, mask):
    result = {}
    for key, value in part.items():
        if key == "visual_references":
            result[key] = {name: probability[mask] for name, probability in value.items()}
        elif isinstance(value, np.ndarray) and len(value) == len(mask):
            result[key] = value[mask]
        else:
            result[key] = value
    return result


def concatenate(items):
    result = {}
    for key in items[0]:
        if key == "visual_references":
            result[key] = {
                name: np.concatenate([item[key][name] for item in items], axis=0)
                for name in items[0][key]
            }
        elif isinstance(items[0][key], np.ndarray) and len(items[0][key]) == len(items[0]["labels"]):
            result[key] = np.concatenate([item[key] for item in items], axis=0)
        else:
            result[key] = items[0][key]
    return result


def prototypes(part):
    output = {}
    for name in ("physical", "motion"):
        value = part[name]
        centroids = []
        for class_id in range(40):
            selected = part["labels"] == class_id
            if selected.any():
                centroid = value[selected].mean(axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-6)
            else:
                centroid = np.zeros(value.shape[1], dtype=np.float32)
            centroids.append(centroid)
        output[name] = np.stack(centroids).astype(np.float32)
    return output


def candidates(part, k):
    visual = part["visual_references"][VISUAL_REFERENCE]
    visual_top = np.argsort(-visual, axis=1, kind="stable")[:, :k]
    group_top = np.argsort(-part["group_probability"], axis=1, kind="stable")[:, :k]
    result = []
    for row in range(len(part["base"])):
        if visual_top[row, 0] == part["base"][row]:
            result.append(np.asarray([part["base"][row]], dtype=int))
            continue
        shared = sorted(set(visual_top[row].tolist()) & set(group_top[row].tolist()))
        ordered = [int(part["base"][row]), *[value for value in shared if value != part["base"][row]]]
        result.append(np.asarray(ordered, dtype=int))
    return result


def confusion_prior(part):
    count = np.zeros((40, 40), dtype=np.float32)
    user_count = np.zeros((40, 40), dtype=np.float32)
    for base_class in range(40):
        for target_class in range(40):
            selected = (part["base"] == base_class) & (part["labels"] == target_class)
            count[base_class, target_class] = int(selected.sum())
            user_count[base_class, target_class] = len(np.unique(part["users"][selected]))
    return {"count": count, "user_count": user_count}


def pair_features(part, prototype, k, prior=None):
    sets = candidates(part, k)
    rows = []
    candidate_ids = []
    sample_rows = []
    visual = np.clip(part["visual_references"][VISUAL_REFERENCE], 1e-7, 1.0)
    group = np.clip(part["group_probability"], 1e-7, 1.0)
    nonvisual = np.clip(part["nonvisual_probability"], 1e-7, 1.0)
    visual_rank = np.argsort(np.argsort(-visual, axis=1, kind="stable"), axis=1) + 1
    group_rank = np.argsort(np.argsort(-group, axis=1, kind="stable"), axis=1) + 1
    nonvisual_rank = np.argsort(np.argsort(-nonvisual, axis=2, kind="stable"), axis=2) + 1
    for sample, values in enumerate(sets):
        base = int(part["base"][sample])
        for candidate in values:
            candidate = int(candidate)
            feature = [
                np.log(visual[sample, candidate]) - np.log(visual[sample, base]),
                visual_rank[sample, candidate] / 40.0,
                np.log(group[sample, candidate]) - np.log(group[sample, base]),
                group_rank[sample, candidate] / 40.0,
                float(candidate == base),
            ]
            for teacher in range(len(NONVISUAL_TEACHERS)):
                feature.extend(
                    (
                        np.log(nonvisual[sample, teacher, candidate])
                        - np.log(nonvisual[sample, teacher, base]),
                        nonvisual_rank[sample, teacher, candidate] / 40.0,
                        float(nonvisual_rank[sample, teacher, candidate] <= 1),
                        float(nonvisual_rank[sample, teacher, candidate] <= 3),
                        float(nonvisual_rank[sample, teacher, candidate] <= 5),
                    )
                )
            for name in ("physical", "motion"):
                value = part[name][sample]
                candidate_similarity = float(value @ prototype[name][candidate])
                base_similarity = float(value @ prototype[name][base])
                feature.extend((candidate_similarity, base_similarity, candidate_similarity - base_similarity))
            if USE_CONFUSION_PRIOR:
                if prior is None:
                    raise RuntimeError("confusion prior required")
                count = prior["count"]
                user_count = prior["user_count"]
                base_errors = float(count[base].sum() - count[base, base])
                candidate_errors = float(count[:, candidate].sum() - count[candidate, candidate])
                support = float(count[base, candidate])
                feature.extend(
                    (
                        np.log1p(support),
                        user_count[base, candidate] / 18.0,
                        (support + 0.5) / (base_errors + 20.0),
                        np.log1p(float(count[candidate, base])),
                        np.log1p(candidate_errors),
                    )
                )
            feature.extend(np.eye(40, dtype=np.float32)[candidate].tolist())
            feature.extend(np.eye(40, dtype=np.float32)[base].tolist())
            rows.append(feature)
            candidate_ids.append(candidate)
            sample_rows.append(sample)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(sample_rows, dtype=np.int64),
        np.asarray(candidate_ids, dtype=np.int64),
        sets,
    )


def fit_model(train, k, c_value, sample_weight=None):
    prototype = prototypes(train)
    prior = confusion_prior(train) if USE_CONFUSION_PRIOR else None
    matrix, sample_rows, candidate_ids, _ = pair_features(train, prototype, k, prior)
    target = (candidate_ids == train["labels"][sample_rows]).astype(int)
    if int(target.sum()) < 20 or int((1 - target).sum()) < 20:
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
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["logisticregression__sample_weight"] = np.asarray(sample_weight)[sample_rows]
    model.fit(matrix, target, **fit_kwargs)
    return model, prototype, prior


def model_feature_count(model_and_prototype):
    model = model_and_prototype[0]
    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        coefficient = model.named_steps["logisticregression"].coef_[0]
        return int(np.sum(np.abs(coefficient) > 1e-10))
    if hasattr(model, "feature_importances_"):
        return int(np.sum(np.asarray(model.feature_importances_) > 0))
    return -1


def predict(model_and_prototype, part, k):
    model, prototype = model_and_prototype[:2]
    prior = model_and_prototype[2] if len(model_and_prototype) > 2 else None
    matrix, sample_rows, candidate_ids, sets = pair_features(part, prototype, k, prior)
    probability = model.predict_proba(matrix)[:, 1]
    proposal = part["base"].copy()
    margin = np.full(len(proposal), -np.inf, dtype=float)
    for row, values in enumerate(sets):
        selected = sample_rows == row
        scores = probability[selected]
        ids = candidate_ids[selected]
        base_score = float(scores[ids == part["base"][row]][0])
        best = int(np.argmax(scores))
        proposal[row] = int(ids[best])
        margin[row] = float(scores[best] - base_score)
    return proposal, margin


def choose_threshold(source, proposal, margin):
    best = None
    for threshold in THRESHOLDS:
        route = (proposal != source["base"]) & (margin >= threshold)
        output = source["base"].copy()
        output[route] = proposal[route]
        gain = (output == source["labels"]).astype(int) - (source["base"] == source["labels"]).astype(int)
        per_cohort = {cohort: int(gain[source["cohort"] == cohort].sum()) for cohort in np.unique(source["cohort"])}
        per_user = {user: int(gain[source["users"] == user].sum()) for user in np.unique(source["users"])}
        rescue = int(np.sum(route & (source["base"] != source["labels"]) & (output == source["labels"])))
        harm = int(np.sum(route & (source["base"] == source["labels"]) & (output != source["labels"])))
        row = {
            "threshold": float(threshold),
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
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, row)
    if not best[0][0]:
        best[1]["threshold"] = float("inf")
    return best[1]


def select_direction_pairs(source, proposal, margin, threshold):
    route = (proposal != source["base"]) & (margin >= float(threshold))
    result = []
    for base_class, target_class in sorted(
        set(zip(source["base"][route].tolist(), proposal[route].tolist()))
    ):
        selected = route & (source["base"] == base_class) & (proposal == target_class)
        rescue = selected & (source["base"] != source["labels"]) & (proposal == source["labels"])
        harm = selected & (source["base"] == source["labels"]) & (proposal != source["labels"])
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
            "changed": int(selected.sum()),
            "rescue": int(rescue.sum()),
            "harm": int(harm.sum()),
            "net": int(rescue.sum() - harm.sum()),
            "minimum_cohort_gain": min(per_cohort.values()),
            "minimum_user_gain": min(per_user.values()),
            "positive_users": sum(value > 0 for value in per_user.values()),
            "per_cohort": per_cohort,
        }
        row["eligible"] = bool(
            row["rescue"] >= MIN_DIRECTION_RESCUE
            and row["harm"] == 0
            and row["minimum_cohort_gain"] >= 0
            and row["minimum_user_gain"] >= 0
            and row["positive_users"] >= MIN_DIRECTION_POSITIVE_USERS
        )
        result.append(row)
    return result


def main():
    print(
        "P399 trains a sparse shared candidate scorer from nonvisual relative posteriors "
        "and source-only physical/motion class prototypes inside visual disagreements.",
        flush=True,
    )
    parts = load_data()
    physical_dim, motion_dim = attach_raw(parts)
    report = {
        "stage": "P399_candidate_conditioned_raw_detail_head_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "visual_role": "agreement lock and Top-K scope only",
            "candidate_set": "P310 plus visual/P307 Top-K intersection",
            "nonvisual_teachers": list(NONVISUAL_TEACHERS),
            "raw_features": {"physical": physical_dim, "motion": motion_dim},
            "raw_feature_use": "source-label class prototypes and candidate/base cosine similarities",
            "confusion_prior_features": USE_CONFUSION_PRIOR,
            "model": "shared L1 candidate correctness scorer",
            "source_cross_prediction": "leave one subject out",
            "direction_guard": DIRECTION_GUARD,
            "held_labels_used_for_prototypes_model_or_threshold": False,
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
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source = concatenate([parts[cohort] for cohort in source_names])
        best = None
        for k in KS:
            for c_value in CS:
                proposal = source["base"].copy()
                margin = np.full(len(proposal), -np.inf, dtype=float)
                valid = True
                for user in np.unique(source["users"]):
                    validation = source["users"] == user
                    training = ~validation
                    model = fit_model(subset(source, training), k, c_value)
                    if model is None:
                        valid = False
                        break
                    proposal[validation], margin[validation] = predict(
                        model, subset(source, validation), k
                    )
                if not valid:
                    continue
                threshold = choose_threshold(source, proposal, margin)
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
                    best = (key, k, c_value, threshold, proposal.copy(), margin.copy())
        _, k, c_value, threshold, source_proposal, source_margin = best
        direction_audit = select_direction_pairs(
            source, source_proposal, source_margin, threshold["threshold"]
        )
        allowed_pairs = {
            (row["base_class"], row["target_class"])
            for row in direction_audit
            if row["eligible"]
        }
        final_model = fit_model(source, k, c_value)
        proposal, margin = predict(final_model, parts[held], k)
        route = (proposal != parts[held]["base"]) & (margin >= float(threshold["threshold"]))
        if DIRECTION_GUARD:
            route &= np.asarray(
                [
                    (int(base_class), int(target_class)) in allowed_pairs
                    for base_class, target_class in zip(parts[held]["base"], proposal, strict=True)
                ],
                dtype=bool,
            )
        output = parts[held]["base"].copy()
        output[route] = proposal[route]
        outputs.append(output)
        held_proposals.append(proposal.astype(np.int16))
        held_margins.append(margin.astype(np.float32))
        held_routes.append(route.astype(bool))
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
            "selected_k": k,
            "selected_C": c_value,
            "source_threshold": threshold,
            "direction_audit": direction_audit,
            "allowed_direction_pairs": [list(pair) for pair in sorted(allowed_pairs)],
            "nonzero_coefficients": model_feature_count(final_model),
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
        print(json.dumps({"held": held, "k": k, "C": c_value, "threshold": threshold, "nonzero": report["cohorts"][held]["nonzero_coefficients"], "result": report["cohorts"][held]["held"]}), flush=True)
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
        proposal=np.concatenate(held_proposals),
        margin=np.concatenate(held_margins),
        route=np.concatenate(held_routes),
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: candidate-conditioned sparse raw-detail head with LOSO source scores.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
