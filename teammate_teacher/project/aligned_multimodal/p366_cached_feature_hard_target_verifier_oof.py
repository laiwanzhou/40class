"""Hard-target verifiers from cached physical and motion representations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

from p238_physical_token_transformer_oof import PATHS as PHYSICAL_PATHS
from p365_hard_target_verifier_oof import COHORTS, choose_threshold, eligible, source_targets
from p361_bidirectional_existing_teacher_gate_oof import load_parts
from p90_teacher_common import load_protocol


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p366_cached_feature_hard_target_verifier_oof_v1"
MOTION = HERE / "runs/p86_motion_window_cache_t16_v1"
ALPHAS = (100.0, 1000.0, 3000.0)
KS = (3, 5)
VARIANTS = ("physical", "motion", "joint")


def l2(values):
    values = np.asarray(values, np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-6, None)


def physical_features(protocol):
    blocks = []
    for path in PHYSICAL_PATHS:
        archive = np.load(path)
        if not np.array_equal(archive["sample_ids"].astype(str), protocol.sample_ids.astype(str)):
            raise RuntimeError(f"physical feature order mismatch: {path}")
        token = archive["features"].astype(np.float32).reshape(len(protocol.labels), -1, 768)
        blocks.extend((l2(token.mean(axis=1)), l2(token.std(axis=1))))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def motion_features(protocol):
    with (MOTION / "rows.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    id_key = "sample_id" if "sample_id" in rows[0] else "source_id"
    ids = np.asarray([row[id_key] for row in rows]).astype(str)
    if not np.array_equal(ids, protocol.sample_ids.astype(str)):
        position = {sample_id: index for index, sample_id in enumerate(ids)}
        order = np.asarray([position[sample_id] for sample_id in protocol.sample_ids.astype(str)])
    else:
        order = np.arange(len(ids))
    skeleton = np.asarray(np.load(MOTION / "skeleton_features.npy", mmap_mode="r")[order], np.float32)
    relation = np.asarray(np.load(MOTION / "skeleton_relations.npy", mmap_mode="r")[order], np.float32)
    imu = np.asarray(np.load(MOTION / "imu_bin_statistics.npy", mmap_mode="r")[order], np.float32)
    imu_global = np.asarray(np.load(MOTION / "imu_global_statistics.npy", mmap_mode="r")[order], np.float32)
    blocks = []
    for value in (skeleton, relation, imu):
        temporal_axes = tuple(range(1, value.ndim - 2))
        blocks.extend(
            (
                l2(value.mean(axis=temporal_axes).reshape(len(value), -1)),
                l2(value.std(axis=temporal_axes).reshape(len(value), -1)),
            )
        )
    blocks.append(l2(imu_global.reshape(len(imu_global), -1)))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def attach_features(parts):
    protocol = load_protocol()
    physical = physical_features(protocol)
    motion = motion_features(protocol)
    lookup = {sample_id: index for index, sample_id in enumerate(protocol.sample_ids.astype(str))}
    for cohort in COHORTS:
        index = np.asarray([lookup[sample_id] for sample_id in parts[cohort]["ids"]])
        parts[cohort]["physical"] = physical[index]
        parts[cohort]["motion"] = motion[index]
        parts[cohort]["joint"] = np.concatenate((physical[index], motion[index]), axis=1)
        parts[cohort]["cohort"] = np.full(len(index), cohort, dtype=object)
    return {"physical": physical.shape[1], "motion": motion.shape[1], "joint": physical.shape[1] + motion.shape[1]}


def concatenate(items, variant):
    result = {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in ("ids", "users", "labels", "base", "bank", "group_probability", "cohort")
    }
    result[variant] = np.concatenate([item[variant] for item in items], axis=0)
    return result


def fit_predict(train, held, target, alpha, k, variant):
    mask = eligible(train, target, k)
    y = (train["labels"] == target).astype(int)
    if int(y[mask].sum()) < 4 or int((1 - y[mask]).sum()) < 8:
        return None
    model = RidgeClassifier(
        alpha=float(alpha),
        class_weight="balanced",
        solver="lsqr",
        tol=1e-5,
        max_iter=5000,
    )
    model.fit(train[variant][mask], y[mask])
    return np.asarray(model.decision_function(held[variant]), float)


def main():
    print(
        "P366 tests lightweight hard-target verifiers on cached physical, motion, and joint "
        "representations; the large encoders are not executed.",
        flush=True,
    )
    parts, teacher_names = load_parts()
    dimensions = attach_features(parts)
    report = {
        "stage": "P366_cached_feature_hard_target_verifier_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "cached_feature_variants": dimensions,
            "model": "class-balanced Ridge one-vs-rest target verifier",
            "alphas": list(ALPHAS),
            "candidate_k": list(KS),
            "source_target_discovery": True,
            "inner_cross_cohort_thresholds": True,
            "held_labels_used_for_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    held_outputs = []
    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        best = None
        audit_count = 0
        all_audits = []
        for variant in VARIANTS:
            source = concatenate([parts[cohort] for cohort in source_names], variant)
            offsets = {
                source_names[0]: (0, len(parts[source_names[0]]["labels"])),
                source_names[1]: (len(parts[source_names[0]]["labels"]), len(source["labels"])),
            }
            for k in KS:
                targets = source_targets(source, k)
                for alpha in ALPHAS:
                    rules = []
                    for target in targets:
                        score = np.empty(len(source["labels"]), dtype=float)
                        valid = True
                        for train_name, validation_name in (
                            (source_names[0], source_names[1]),
                            (source_names[1], source_names[0]),
                        ):
                            decision = fit_predict(
                                parts[train_name], parts[validation_name], target, alpha, k, variant
                            )
                            if decision is None:
                                valid = False
                                break
                            lo, hi = offsets[validation_name]
                            score[lo:hi] = decision
                        if not valid:
                            continue
                        selected = choose_threshold(source, target, k, score)
                        audit_count += 1
                        rule = {
                            "target_class": target,
                            "variant": variant,
                            "k": k,
                            "alpha": alpha,
                            **selected,
                        }
                        all_audits.append(rule)
                        if (
                            selected["rescue"] >= 2
                            and selected["harm"] == 0
                            and selected["minimum_cohort_gain"] >= 1
                            and selected["positive_users"] >= 2
                            and selected["minimum_user_gain"] >= 0
                        ):
                            rules.append(rule)
                    key = (
                        sum(rule["net"] for rule in rules),
                        sum(rule["rescue"] for rule in rules),
                        -sum(rule["harm"] for rule in rules),
                        -len(rules),
                        -k,
                        -alpha,
                    )
                    if best is None or key > best[0]:
                        best = (key, variant, k, alpha, rules, source)

        _, variant, k, alpha, rules, source = best
        output = parts[held]["base"].copy()
        best_excess = np.full(len(output), -np.inf, dtype=float)
        for rule in rules:
            score = fit_predict(
                source,
                parts[held],
                int(rule["target_class"]),
                float(rule["alpha"]),
                int(rule["k"]),
                str(rule["variant"]),
            )
            if score is None:
                continue
            route = (
                eligible(parts[held], int(rule["target_class"]), int(rule["k"]))
                & (score >= float(rule["threshold"]))
                & ((score - float(rule["threshold"])) > best_excess)
            )
            output[route] = int(rule["target_class"])
            best_excess[route] = score[route] - float(rule["threshold"])
        held_outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        changed = output != base
        report["cohorts"][held] = {
            "source": source_names,
            "selected_variant": variant,
            "selected_k": k,
            "selected_alpha": alpha,
            "rules": rules,
            "source_audit_count": audit_count,
            "top_rejected_source_audits": sorted(
                all_audits,
                key=lambda rule: (
                    rule["minimum_cohort_gain"],
                    rule["net"],
                    rule["rescue"],
                    -rule["harm"],
                    -rule["changed"],
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

    labels = np.concatenate([parts[cohort]["labels"] for cohort in COHORTS])
    base = np.concatenate([parts[cohort]["base"] for cohort in COHORTS])
    prediction = np.concatenate(held_outputs)
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
        **{
            f"{cohort}_held_prediction": held_outputs[index]
            for index, cohort in enumerate(COHORTS)
        },
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: cached physical/motion/joint target verifiers with inner cross-cohort gating.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": report["aggregate"], "cohorts": {cohort: report["cohorts"][cohort]["held"] for cohort in COHORTS}, "selected": {cohort: {"variant": report["cohorts"][cohort]["selected_variant"], "k": report["cohorts"][cohort]["selected_k"], "alpha": report["cohorts"][cohort]["selected_alpha"], "rules": report["cohorts"][cohort]["rules"]} for cohort in COHORTS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
