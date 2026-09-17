"""Non-negative per-class family residual stacker over the frozen P307 posterior."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p363_target_class_family_boost_oof import family_parts


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p372_nonnegative_class_family_stacker_oof_v1"
LAMBDAS = (0.01, 0.1, 1.0)
EPOCHS = 250
SEED = 20260903


def geometric_family_indices(names):
    return [index for index, name in enumerate(names) if name.endswith("__geometric")]


def train_model(part, family_indices, regularization):
    torch.manual_seed(SEED)
    group = torch.from_numpy(np.log(np.clip(part["group_probability"], 1e-7, 1.0))).float()
    family = torch.from_numpy(
        np.log(np.clip(part["families"][:, family_indices, :], 1e-7, 1.0))
    ).float()
    labels = torch.from_numpy(part["labels"].astype(np.int64))
    raw = torch.nn.Parameter(torch.full((len(family_indices), 40), -4.0))
    bias = torch.nn.Parameter(torch.zeros(40))
    optimizer = torch.optim.Adam((raw, bias), lr=0.05)
    for _ in range(EPOCHS):
        weight = torch.sigmoid(raw)
        logits = group + ((family - group[:, None, :]) * weight[None, :, :]).sum(dim=1) + bias
        loss = F.cross_entropy(logits, labels) + float(regularization) * weight.square().mean() + 0.01 * bias.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return torch.sigmoid(raw).detach().numpy(), bias.detach().numpy()


def predict(part, family_indices, models):
    group = np.log(np.clip(part["group_probability"], 1e-7, 1.0)).astype(np.float64)
    family = np.log(np.clip(part["families"][:, family_indices, :], 1e-7, 1.0)).astype(np.float64)
    logits = []
    for weight, bias in models:
        value = group + ((family - group[:, None, :]) * weight[None, :, :]).sum(axis=1) + bias
        logits.append(value)
    value = np.mean(logits, axis=0)
    value -= value.max(axis=1, keepdims=True)
    probability = np.exp(value)
    return probability / probability.sum(axis=1, keepdims=True)


def concatenate(items):
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in ("ids", "users", "labels", "base", "group_probability", "families", "cohort")
    }


def choose_threshold(source, probability):
    rows = np.arange(len(source["base"]))
    proposal = probability.argmax(axis=1)
    score = probability[rows, proposal] - probability[rows, source["base"]]
    changed = proposal != source["base"]
    values = np.unique(
        np.concatenate(([-np.inf, np.inf], np.linspace(0.0, 0.80, 81), np.quantile(score[changed], np.linspace(0.1, 0.95, 18)) if changed.any() else [np.inf]))
    )
    best = None
    for threshold in values:
        route = changed & (score >= threshold)
        gain = route.astype(int) * (
            (proposal == source["labels"]).astype(int)
            - (source["base"] == source["labels"]).astype(int)
        )
        per_cohort = {cohort: int(gain[source["cohort"] == cohort].sum()) for cohort in np.unique(source["cohort"])}
        per_user = {user: int(gain[source["users"] == user].sum()) for user in np.unique(source["users"])}
        rescue = int(np.sum(route & (source["base"] != source["labels"]) & (proposal == source["labels"])))
        harm = int(np.sum(route & (source["base"] == source["labels"]) & (proposal != source["labels"])))
        result = {
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
        valid = result["net"] > 0 and result["minimum_cohort_gain"] >= 0 and result["minimum_user_gain"] >= -1
        key = (
            valid,
            result["minimum_cohort_gain"],
            result["net"],
            result["rescue"],
            -result["harm"],
            -result["changed"],
            float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, result)
    if not best[0][0]:
        best[1]["threshold"] = float("inf")
        best[1]["changed"] = best[1]["rescue"] = best[1]["harm"] = best[1]["net"] = 0
    return best[1]


def main():
    print(
        "P372 tests non-negative per-class teacher-family residual weights with source-cohort "
        "model averaging and a source-only P310 replacement threshold.",
        flush=True,
    )
    parts, family_names, family_members = family_parts()
    family_indices = geometric_family_indices(family_names)
    selected_family_names = [family_names[index] for index in family_indices]
    for cohort in COHORTS:
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
    report = {
        "stage": "P372_nonnegative_class_family_stacker_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "anchor_probability": "P307 group posterior",
            "families": selected_family_names,
            "weights": "non-negative sigmoid residual, one coefficient per family/class",
            "held_prediction": "average of the two separately source-cohort-trained models",
            "regularization": list(LAMBDAS),
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
        for regularization in LAMBDAS:
            inner_probabilities = []
            for train_name, validation_name in ((source_names[0], source_names[1]), (source_names[1], source_names[0])):
                model = train_model(parts[train_name], family_indices, regularization)
                inner_probabilities.append((validation_name, predict(parts[validation_name], family_indices, [model])))
            probability = np.empty((len(source["labels"]), 40), dtype=float)
            offsets = {
                source_names[0]: (0, len(parts[source_names[0]]["labels"])),
                source_names[1]: (len(parts[source_names[0]]["labels"]), len(source["labels"])),
            }
            for validation_name, value in inner_probabilities:
                lo, hi = offsets[validation_name]
                probability[lo:hi] = value
            threshold = choose_threshold(source, probability)
            key = (
                threshold["minimum_cohort_gain"] >= 0,
                threshold["net"],
                threshold["rescue"],
                -threshold["harm"],
                -threshold["changed"],
                -regularization,
            )
            if best is None or key > best[0]:
                best = (key, regularization, threshold)
        _, regularization, threshold = best
        models = [train_model(parts[source_name], family_indices, regularization) for source_name in source_names]
        probability = predict(parts[held], family_indices, models)
        rows = np.arange(len(parts[held]["base"]))
        proposal = probability.argmax(axis=1)
        score = probability[rows, proposal] - probability[rows, parts[held]["base"]]
        route = (proposal != parts[held]["base"]) & (score >= float(threshold["threshold"]))
        output = parts[held]["base"].copy()
        output[route] = proposal[route]
        outputs.append(output)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        changed = output != base
        report["cohorts"][held] = {
            "source": source_names,
            "regularization": regularization,
            "source_threshold": threshold,
            "mean_family_weight": {
                selected_family_names[index]: float(np.mean([model[0][index].mean() for model in models]))
                for index in range(len(selected_family_names))
            },
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
        "Run 1: non-negative class-conditional family residual stacker.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": report["aggregate"], "cohorts": {cohort: report["cohorts"][cohort]["held"] for cohort in COHORTS}, "weights": {cohort: report["cohorts"][cohort]["mean_family_weight"] for cohort in COHORTS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
