"""Target-domain importance-weighted P404 raw-detail head under strict outer OOF."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import p399_candidate_conditioned_raw_detail_head_oof as p399
import p404_group_guarded_raw_detail_oof as p404


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p410_target_aware_weighted_raw_detail_oof_v1"
DOMAIN_C = 0.10
WEIGHT_CLIP = (0.25, 4.0)


def distribution_features(part):
    visual = np.clip(part["visual_references"]["strong_visual_mean"], 1e-7, 1.0)
    group = np.clip(part["group_probability"], 1e-7, 1.0)
    nonvisual = np.clip(part["nonvisual_probability"], 1e-7, 1.0)
    ordered = np.sort(nonvisual, axis=2)[:, :, ::-1]
    entropy = -(nonvisual * np.log(nonvisual)).sum(axis=2) / np.log(40.0)
    teacher_scalar = np.concatenate(
        (
            ordered[:, :, :1],
            ordered[:, :, :1] - ordered[:, :, 1:2],
            entropy[:, :, None],
        ),
        axis=2,
    ).reshape(len(group), -1)
    raw_scalar = []
    for name in ("physical", "motion"):
        value = part[name]
        raw_scalar.extend(
            (
                value.mean(axis=1),
                value.std(axis=1),
                np.mean(np.abs(value), axis=1),
                np.quantile(value, 0.10, axis=1),
                np.quantile(value, 0.50, axis=1),
                np.quantile(value, 0.90, axis=1),
            )
        )
    return np.concatenate(
        (np.sqrt(visual), np.sqrt(group), teacher_scalar, np.column_stack(raw_scalar)),
        axis=1,
    ).astype(np.float32)


def domain_weights(source, target):
    source_x = distribution_features(source)
    target_x = distribution_features(target)
    x = np.concatenate((source_x, target_x), axis=0)
    y = np.concatenate((np.zeros(len(source_x), dtype=int), np.ones(len(target_x), dtype=int)))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=DOMAIN_C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1500,
            random_state=20260903,
        ),
    )
    model.fit(x, y)
    probability = np.clip(model.predict_proba(source_x)[:, 1], 1e-4, 1.0 - 1e-4)
    ratio = probability / (1.0 - probability)
    ratio *= len(source_x) / max(len(target_x), 1)
    ratio /= max(float(ratio.mean()), 1e-8)
    ratio = np.clip(ratio, WEIGHT_CLIP[0], WEIGHT_CLIP[1])
    ratio /= max(float(ratio.mean()), 1e-8)
    domain_accuracy = float(np.mean(model.predict(x) == y))
    return ratio.astype(np.float32), {
        "domain_accuracy": domain_accuracy,
        "source_rows": len(source_x),
        "target_rows": len(target_x),
        "weight_min": float(ratio.min()),
        "weight_mean": float(ratio.mean()),
        "weight_max": float(ratio.max()),
        "weight_p90": float(np.quantile(ratio, 0.90)),
    }


def main():
    print(
        "P410 treats every held user/cohort as an unlabeled target domain, estimates "
        "source density-ratio weights, and trains the P404 detail head with those weights.",
        flush=True,
    )
    parts = p399.load_data()
    physical_dim, motion_dim = p399.attach_raw(parts)
    report = {
        "stage": "P410_target_aware_weighted_raw_detail_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "detail_head": "P399 shared L1 candidate scorer",
            "gate": "P404 joint detail-margin/P307 support",
            "target_adaptation": "label-free source-vs-target density ratio",
            "domain_features": "visual/P307 probabilities, nonvisual confidence/margin/entropy, raw feature statistics",
            "domain_C": DOMAIN_C,
            "weight_clip": list(WEIGHT_CLIP),
            "held_labels_used_for_domain_model_detail_model_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    proposals = []
    margins = []
    routes = []
    for held in p399.COHORTS:
        source_names = [cohort for cohort in p399.COHORTS if cohort != held]
        source = p399.concatenate([parts[cohort] for cohort in source_names])
        best = None
        domain_audits = []
        for k in p399.KS:
            for c_value in p399.CS:
                proposal = source["base"].copy()
                margin = np.full(len(proposal), -np.inf, dtype=float)
                for user in np.unique(source["users"]):
                    validation = source["users"] == user
                    train_part = p399.subset(source, ~validation)
                    target_part = p399.subset(source, validation)
                    weights, audit = domain_weights(train_part, target_part)
                    domain_audits.append({"user": str(user), "k": k, "C": c_value, **audit})
                    model = p399.fit_model(train_part, k, c_value, sample_weight=weights)
                    proposal[validation], margin[validation] = p399.predict(model, target_part, k)
                threshold = p404.choose(source, proposal, margin)
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
        weights, held_domain_audit = domain_weights(source, parts[held])
        model = p399.fit_model(source, k, c_value, sample_weight=weights)
        proposal, margin = p399.predict(model, parts[held], k)
        gap = p404.group_gap(parts[held], proposal)
        route = (
            (proposal != parts[held]["base"])
            & (margin >= float(threshold["margin_threshold"]))
            & (gap >= float(threshold["group_gap_threshold"]))
        )
        output = parts[held]["base"].copy()
        output[route] = proposal[route]
        outputs.append(output)
        proposals.append(proposal.astype(np.int16))
        margins.append(margin.astype(np.float32))
        routes.append(route)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
            "selected_k": k,
            "selected_C": c_value,
            "source_threshold": threshold,
            "held_domain_audit": held_domain_audit,
            "inner_domain_accuracy_mean": float(np.mean([row["domain_accuracy"] for row in domain_audits if row["k"] == k and row["C"] == c_value])),
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
        print(json.dumps({"held": held, "k": k, "C": c_value, "threshold": threshold, "domain": held_domain_audit, "result": report["cohorts"][held]["held"]}), flush=True)
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
        proposal=np.concatenate(proposals),
        margin=np.concatenate(margins),
        route=np.concatenate(routes),
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: target-domain importance-weighted P404 raw-detail head.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
