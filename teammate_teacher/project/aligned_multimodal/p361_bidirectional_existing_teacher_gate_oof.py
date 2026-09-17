"""Outer-cross-fit audit of bidirectional routing over the existing 30-teacher bank.

The positive branch softly increases one existing teacher's weight relative to
the frozen P307 group posterior.  The reverse branch lets one existing teacher
act as a directed specialist for a source-prediction -> target-class confusion.
No Test asset is loaded by this script.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from p173_vjepa_augmented_group_teacher import build_train_bank
from p255_repeat_augmented_physical_group import al
from p307_union_repeat_group_sequence_audit import SOURCES


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "p361_bidirectional_existing_teacher_gate_oof_v1"
P307 = RUNS / "p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz"
P310 = RUNS / "p310_union_repeat_precedence_teacher_v1/oof_predictions.npz"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
ALPHAS = (0.15, 0.25, 0.40, 0.60)


@dataclass(frozen=True)
class Candidate:
    branch: str
    teacher_index: int
    teacher_name: str
    alpha: float


def normalized(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability.astype(np.float64), 1e-7, None)
    return value / value.sum(axis=1, keepdims=True)


def load_parts() -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    train, names = build_train_bank()
    for path, key, name in SOURCES:
        archive = np.load(path)
        for cohort in COHORTS:
            aligned = al(archive[key], archive["sample_ids"], train[cohort]["ids"])
            train[cohort]["bank"] = np.concatenate(
                (train[cohort]["bank"], aligned[:, None, :]), axis=1
            )
        names.append(name)

    p307 = np.load(P307)
    p310 = np.load(P310)
    offset = 0
    parts: dict[str, dict[str, np.ndarray]] = {}
    for cohort in COHORTS:
        rows = len(train[cohort]["labels"])
        labels = train[cohort]["labels"].astype(int)
        if not np.array_equal(labels, p310["labels"][offset : offset + rows]):
            raise RuntimeError(f"P310 label alignment failed for {cohort}")
        parts[cohort] = {
            "ids": train[cohort]["ids"].astype(str),
            "users": train[cohort]["users"].astype(str),
            "labels": labels,
            "base": p310["prediction"][offset : offset + rows].astype(int),
            "group_probability": normalized(
                p307[f"{cohort}_group_probability"].astype(float)
            ),
            "bank": np.stack(
                [normalized(train[cohort]["bank"][:, i, :]) for i in range(len(names))],
                axis=1,
            ),
        }
        offset += rows
    if offset != len(p310["labels"]):
        raise RuntimeError("unexpected P310 row count")
    return parts, names


def candidate_outputs(
    part: dict[str, np.ndarray], candidates: list[Candidate]
) -> tuple[np.ndarray, np.ndarray]:
    group = part["group_probability"]
    base = part["base"]
    rows = np.arange(len(base))
    predictions = np.empty((len(base), len(candidates)), dtype=np.int16)
    scores = np.empty((len(base), len(candidates)), dtype=np.float32)
    for column, candidate in enumerate(candidates):
        teacher = part["bank"][:, candidate.teacher_index, :]
        if candidate.branch == "positive_weight":
            # Geometric interpolation preserves calibrated rank geometry while
            # granting only a partial increase to the selected teacher.
            logp = (1.0 - candidate.alpha) * np.log(group) + candidate.alpha * np.log(teacher)
            proposal = logp.argmax(axis=1)
            score = logp[rows, proposal] - logp[rows, base]
        elif candidate.branch == "reverse_specialist":
            proposal = teacher.argmax(axis=1)
            logp = np.log(teacher)
            score = logp[rows, proposal] - logp[rows, base]
        else:
            raise ValueError(candidate.branch)
        predictions[:, column] = proposal
        scores[:, column] = score
    return predictions, scores


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in ("ids", "users", "labels", "base")
    }


def threshold_grid(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0, dtype=float)
    quantiles = np.quantile(values, np.linspace(0.0, 0.95, 20))
    return np.unique(np.concatenate(([-np.inf], quantiles)))


def rule_metrics(
    source: dict[str, np.ndarray],
    proposal: np.ndarray,
    score: np.ndarray,
    base_class: int,
    target_class: int,
    threshold: float,
    cohort_ids: np.ndarray,
) -> dict[str, object]:
    route = (
        (source["base"] == base_class)
        & (proposal == target_class)
        & (proposal != source["base"])
        & (score >= threshold)
    )
    labels = source["labels"]
    gain = route.astype(int) * (
        (proposal == labels).astype(int) - (source["base"] == labels).astype(int)
    )
    per_cohort = {
        cohort: int(gain[cohort_ids == cohort].sum()) for cohort in np.unique(cohort_ids)
    }
    per_user = {
        user: int(gain[source["users"] == user].sum()) for user in np.unique(source["users"])
    }
    changed = int(route.sum())
    rescue = int(np.sum(route & (source["base"] != labels) & (proposal == labels)))
    harm = int(np.sum(route & (source["base"] == labels) & (proposal != labels)))
    return {
        "threshold": float(threshold),
        "changed": changed,
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


def select_rules(
    source: dict[str, np.ndarray],
    proposals: np.ndarray,
    scores: np.ndarray,
    candidates: list[Candidate],
    cohort_ids: np.ndarray,
) -> list[dict[str, object]]:
    labels = source["labels"]
    base = source["base"]
    best_by_pair: dict[tuple[int, int], tuple[tuple[float, ...], dict[str, object]]] = {}
    for column, candidate in enumerate(candidates):
        proposal = proposals[:, column]
        score = scores[:, column]
        for base_class, target_class in np.unique(
            np.stack((base, proposal), axis=1), axis=0
        ):
            base_class = int(base_class)
            target_class = int(target_class)
            if base_class == target_class:
                continue
            confusion_support = int(np.sum((base == base_class) & (labels == target_class)))
            if confusion_support < 4:
                continue
            pair = (base == base_class) & (proposal == target_class)
            for threshold in threshold_grid(score[pair]):
                metrics = rule_metrics(
                    source,
                    proposal,
                    score,
                    base_class,
                    target_class,
                    float(threshold),
                    cohort_ids,
                )
                # This is deliberately stricter than maximizing aggregate OOF.
                # Both source cohorts must benefit and evidence must span users.
                if (
                    metrics["changed"] < 3
                    or metrics["net"] < 2
                    or metrics["minimum_cohort_gain"] < 1
                    or metrics["positive_users"] < 2
                    or metrics["minimum_user_gain"] < -1
                    or metrics["harm"] > max(1, metrics["rescue"] // 3)
                ):
                    continue
                precision = metrics["rescue"] / metrics["changed"]
                key = (
                    float(metrics["minimum_cohort_gain"]),
                    float(metrics["net"]),
                    float(precision),
                    float(metrics["rescue"]),
                    float(-metrics["harm"]),
                    float(-metrics["changed"]),
                    float(-candidate.alpha),
                )
                record = {
                    "base_class": base_class,
                    "target_class": target_class,
                    "confusion_support": confusion_support,
                    "candidate_column": column,
                    "branch": candidate.branch,
                    "teacher_index": candidate.teacher_index,
                    "teacher_name": candidate.teacher_name,
                    "alpha": candidate.alpha,
                    **metrics,
                }
                old = best_by_pair.get((base_class, target_class))
                if old is None or key > old[0]:
                    best_by_pair[(base_class, target_class)] = (key, record)
    rules = [value[1] for value in best_by_pair.values()]
    rules.sort(
        key=lambda rule: (
            rule["minimum_cohort_gain"],
            rule["net"],
            rule["rescue"] / rule["changed"],
            -rule["harm"],
        ),
        reverse=True,
    )
    return rules[:12]


def apply_rules(
    part: dict[str, np.ndarray],
    proposals: np.ndarray,
    scores: np.ndarray,
    rules: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = part["base"].copy()
    selected_rule = np.full(len(output), -1, dtype=int)
    selected_strength = np.full(len(output), -np.inf, dtype=float)
    for index, rule in enumerate(rules):
        column = int(rule["candidate_column"])
        proposal = proposals[:, column]
        score = scores[:, column]
        threshold = float(rule["threshold"])
        if np.isneginf(threshold):
            strength = score
        else:
            scale = max(1e-4, abs(threshold))
            strength = (score - threshold) / scale
        route = (
            (part["base"] == int(rule["base_class"]))
            & (proposal == int(rule["target_class"]))
            & (score >= threshold)
            & (strength > selected_strength)
        )
        output[route] = proposal[route]
        selected_rule[route] = index
        selected_strength[route] = strength[route]
    return output, selected_rule, selected_strength


def main() -> None:
    print(
        "P361 tests whether partial teacher reweighting and reverse hard-class specialists "
        "can improve the frozen P310 teacher under outer cross-fit.",
        flush=True,
    )
    parts, names = load_parts()
    candidates = [
        Candidate("positive_weight", index, name, alpha)
        for index, name in enumerate(names)
        for alpha in ALPHAS
    ] + [
        Candidate("reverse_specialist", index, name, 1.0)
        for index, name in enumerate(names)
    ]
    outputs = {
        cohort: candidate_outputs(parts[cohort], candidates) for cohort in COHORTS
    }

    report: dict[str, object] = {
        "stage": "P361_bidirectional_existing_teacher_gate_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "existing_teacher_count": len(names),
            "positive_branch": "partial geometric weight increase over P307 group posterior",
            "reverse_branch": "directed existing-teacher specialist for hard confusion targets",
            "outer_crossfit": True,
            "source_rule_constraints": {
                "minimum_confusion_support": 4,
                "minimum_changed": 3,
                "minimum_net": 2,
                "both_source_cohorts_minimum_gain": 1,
                "minimum_positive_users": 2,
                "maximum_harm": "max(1, rescue//3)",
            },
            "held_labels_used_for_rule_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "teachers": names,
        "cohorts": {},
    }
    held_outputs: list[np.ndarray] = []
    rule_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for held in COHORTS:
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source = concatenate([parts[cohort] for cohort in source_names])
        source_proposals = np.concatenate([outputs[cohort][0] for cohort in source_names], axis=0)
        source_scores = np.concatenate([outputs[cohort][1] for cohort in source_names], axis=0)
        cohort_ids = np.concatenate(
            [np.full(len(parts[cohort]["labels"]), cohort, dtype=object) for cohort in source_names]
        )
        rules = select_rules(
            source, source_proposals, source_scores, candidates, cohort_ids
        )
        prediction, selected_rule, strength = apply_rules(
            parts[held], outputs[held][0], outputs[held][1], rules
        )
        held_outputs.append(prediction)
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        changed = prediction != base
        rescue = changed & (base != labels) & (prediction == labels)
        harm = changed & (base == labels) & (prediction != labels)
        fold_report = {
            "source": source_names,
            "selected_rule_count": len(rules),
            "selected_positive_weight_rules": sum(rule["branch"] == "positive_weight" for rule in rules),
            "selected_reverse_specialist_rules": sum(rule["branch"] == "reverse_specialist" for rule in rules),
            "held": {
                "rows": len(labels),
                "base_correct": int(np.sum(base == labels)),
                "correct": int(np.sum(prediction == labels)),
                "net": int(np.sum(prediction == labels) - np.sum(base == labels)),
                "changed": int(changed.sum()),
                "rescue": int(rescue.sum()),
                "harm": int(harm.sum()),
            },
            "rules": rules,
        }
        report["cohorts"][held] = fold_report
        for index, rule in enumerate(rules):
            rule_rows.append({"held_cohort": held, "rule_index": index, **rule})
        for row in np.flatnonzero(changed):
            rule = rules[selected_rule[row]]
            prediction_rows.append(
                {
                    "held_cohort": held,
                    "sample_id": parts[held]["ids"][row],
                    "base_prediction": int(base[row]),
                    "prediction": int(prediction[row]),
                    "label": int(labels[row]),
                    "gain": int(prediction[row] == labels[row]) - int(base[row] == labels[row]),
                    "branch": rule["branch"],
                    "teacher_name": rule["teacher_name"],
                    "alpha": rule["alpha"],
                    "strength": float(strength[row]),
                }
            )

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
    with (OUT / "selected_rules.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "held_cohort", "rule_index", "base_class", "target_class", "branch",
            "teacher_index", "teacher_name", "alpha", "threshold", "confusion_support",
            "changed", "rescue", "harm", "net", "minimum_cohort_gain",
            "positive_cohorts", "minimum_user_gain", "positive_users",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rule_rows)
    with (OUT / "changed_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "held_cohort", "sample_id", "base_prediction", "prediction", "label",
            "gain", "branch", "teacher_name", "alpha", "strength",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "notes.txt").write_text(
        "Run 1: strict outer-cross-fit bidirectional routing over the existing 30-teacher bank.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
