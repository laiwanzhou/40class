"""Subject-safe threshold audit for selective A18 replacement of P89 safe.

This program never fits a model.  It consumes the frozen P89 safe OOF path and
the frozen A18 best-checkpoint + Session OOF probabilities.  Every rule defaults
to P89 and may replace it only with A18 on rows where their predictions differ.

Threshold selection is outer cross-fitted over the existing disjoint H1/H2/H3
subject cohorts.  For one held cohort, thresholds are selected exclusively on
the other two cohorts' OOF labels and then applied unchanged to the held cohort.
No user identifier is a rule feature or a threshold-search variable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_A18 = PROJECT / "runs/a18_subject_safe_revalidation/oof_predictions.npz"
DEFAULT_P89_REFERENCE = PROJECT / "runs/p90_crossuser_visual_router_v1/full_predictions.npz"
DEFAULT_OUTPUT = PROJECT / "runs/a18_p89_selective_replacement_v1"
SPLIT_NAMES = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
EPSILON = 1e-12


Direction = Literal["lower", "upper"]


@dataclass(frozen=True)
class Condition:
    feature: str
    direction: Direction


@dataclass(frozen=True)
class Rule:
    name: str
    conditions: tuple[Condition, ...]
    note: str


@dataclass(frozen=True)
class AuditData:
    split: str
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    p89_prediction: np.ndarray
    a18_prediction: np.ndarray
    features: dict[str, np.ndarray]


RULES = (
    Rule(
        "a18_confidence_only",
        (Condition("a18_confidence", "lower"),),
        "replace when A18 confidence is at least the selected threshold",
    ),
    Rule(
        "p89_max_confidence_only",
        (Condition("p89_max_confidence", "upper"),),
        "replace when P89 adjusted-emission max confidence is low",
    ),
    Rule(
        "p89_safe_support_only",
        (Condition("p89_safe_support", "upper"),),
        "replace when P89 probability support for its deployed safe label is low",
    ),
    Rule(
        "confidence_gap_only",
        (Condition("confidence_gap", "lower"),),
        "A18 confidence minus P89 max confidence",
    ),
    Rule(
        "safe_support_gap_only",
        (Condition("safe_support_gap", "lower"),),
        "A18 confidence minus P89 support for the deployed safe label",
    ),
    Rule(
        "margin_gap_only",
        (Condition("margin_gap", "lower"),),
        "A18 top1-top2 margin minus P89 adjusted-emission top1-top2 margin",
    ),
    Rule(
        "a18_confidence_and_p89_max_confidence",
        (
            Condition("a18_confidence", "lower"),
            Condition("p89_max_confidence", "upper"),
        ),
        "absolute A18 confidence with a weak-P89 max-confidence gate",
    ),
    Rule(
        "a18_confidence_and_p89_safe_support",
        (
            Condition("a18_confidence", "lower"),
            Condition("p89_safe_support", "upper"),
        ),
        "absolute A18 confidence with a weak deployed-P89-label support gate",
    ),
    Rule(
        "a18_confidence_and_confidence_gap",
        (
            Condition("a18_confidence", "lower"),
            Condition("confidence_gap", "lower"),
        ),
        "absolute A18 confidence plus advantage over P89 max confidence",
    ),
    Rule(
        "a18_confidence_and_safe_support_gap",
        (
            Condition("a18_confidence", "lower"),
            Condition("safe_support_gap", "lower"),
        ),
        "absolute A18 confidence plus advantage over P89 deployed-label support",
    ),
    Rule(
        "a18_confidence_and_margin_gap",
        (
            Condition("a18_confidence", "lower"),
            Condition("margin_gap", "lower"),
        ),
        "absolute A18 confidence plus top1-top2 margin advantage",
    ),
    Rule(
        "a18_confidence_p89max_confidence_gap",
        (
            Condition("a18_confidence", "lower"),
            Condition("p89_max_confidence", "upper"),
            Condition("confidence_gap", "lower"),
        ),
        "three-variable standard-confidence rule",
    ),
    Rule(
        "a18_confidence_p89support_support_gap",
        (
            Condition("a18_confidence", "lower"),
            Condition("p89_safe_support", "upper"),
            Condition("safe_support_gap", "lower"),
        ),
        "three-variable deployed-label-support rule",
    ),
    Rule(
        "a18_confidence_p89max_margin_gap",
        (
            Condition("a18_confidence", "lower"),
            Condition("p89_max_confidence", "upper"),
            Condition("margin_gap", "lower"),
        ),
        "A18 confidence, weak P89 confidence, and margin advantage",
    ),
    Rule(
        "all_standard_variables",
        (
            Condition("a18_confidence", "lower"),
            Condition("p89_max_confidence", "upper"),
            Condition("confidence_gap", "lower"),
            Condition("margin_gap", "lower"),
        ),
        "all class-agnostic standard-confidence variables",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a18", type=Path, default=DEFAULT_A18)
    parser.add_argument("--p89-reference", type=Path, default=DEFAULT_P89_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    probability = np.clip(probability, EPSILON, None)
    return probability / probability.sum(axis=1, keepdims=True)


def probability_margin(probability: np.ndarray) -> np.ndarray:
    top = np.partition(probability, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def load_reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def load_data(a18_path: Path, p89_reference_path: Path) -> dict[str, AuditData]:
    # Delayed import keeps the rule-search helpers testable without loading any
    # research artifact.  load_splits reconstructs the frozen P89 safe path; it
    # does not fit an estimator.
    from p90_crossuser_visual_router import load_splits

    p89_splits = load_splits()
    reference = load_reference(p89_reference_path)
    with np.load(a18_path, allow_pickle=False) as source:
        a18_ids = source["sample_ids"].astype(str)
        a18_labels = source["labels"].astype(np.int64)
        a18_probability = normalise_probability(source["best_session_probability"])

    a18_lookup = {sample_id: row for row, sample_id in enumerate(a18_ids)}
    if len(a18_lookup) != len(a18_ids):
        raise RuntimeError("A18 OOF sample IDs are not unique")

    output: dict[str, AuditData] = {}
    for name in SPLIT_NAMES:
        split = p89_splits[name]
        sample_ids = split.sample_ids.astype(str)
        missing = [sample_id for sample_id in sample_ids if sample_id not in a18_lookup]
        if missing:
            raise RuntimeError(f"A18 OOF misses {len(missing)} rows in {name}")
        rows = np.asarray([a18_lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
        labels = np.asarray(split.labels, dtype=np.int64)
        if not np.array_equal(labels, a18_labels[rows]):
            raise RuntimeError(f"A18/P89 labels disagree in {name}")

        reference_ids = reference[f"{name}_sample_ids"].astype(str)
        reference_labels = reference[f"{name}_labels"].astype(np.int64)
        reference_safe = reference[f"{name}_safe_prediction"].astype(np.int64)
        if not np.array_equal(sample_ids, reference_ids):
            raise RuntimeError(f"P89 sample order differs from frozen reference in {name}")
        if not np.array_equal(labels, reference_labels):
            raise RuntimeError(f"P89 labels differ from frozen reference in {name}")
        if not np.array_equal(split.safe_prediction, reference_safe):
            raise RuntimeError(f"P89 safe prediction differs from frozen reference in {name}")

        a_probability = a18_probability[rows]
        p_probability = normalise_probability(split.safe_probability)
        a_prediction = a_probability.argmax(axis=1).astype(np.int64)
        p_prediction = np.asarray(split.safe_prediction, dtype=np.int64)
        row_index = np.arange(len(labels), dtype=np.int64)

        a_confidence = a_probability.max(axis=1)
        p_max_confidence = p_probability.max(axis=1)
        p_safe_support = p_probability[row_index, p_prediction]
        a_margin = probability_margin(a_probability)
        p_margin = probability_margin(p_probability)
        features = {
            "a18_confidence": a_confidence,
            "p89_max_confidence": p_max_confidence,
            "p89_safe_support": p_safe_support,
            "confidence_gap": a_confidence - p_max_confidence,
            "safe_support_gap": a_confidence - p_safe_support,
            "margin_gap": a_margin - p_margin,
        }
        for feature_name, values in features.items():
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"non-finite {feature_name} in {name}")

        output[name] = AuditData(
            split=name,
            sample_ids=sample_ids,
            labels=labels,
            users=np.asarray(split.users).astype(str),
            p89_prediction=p_prediction,
            a18_prediction=a_prediction,
            features=features,
        )
    return output


def concatenate(items: list[AuditData], name: str) -> AuditData:
    feature_names = tuple(items[0].features)
    if any(tuple(item.features) != feature_names for item in items):
        raise RuntimeError("feature sets differ across audit splits")
    return AuditData(
        split=name,
        sample_ids=np.concatenate([item.sample_ids for item in items]),
        labels=np.concatenate([item.labels for item in items]),
        users=np.concatenate([item.users for item in items]),
        p89_prediction=np.concatenate([item.p89_prediction for item in items]),
        a18_prediction=np.concatenate([item.a18_prediction for item in items]),
        features={
            feature: np.concatenate([item.features[feature] for item in items])
            for feature in feature_names
        },
    )


def threshold_values(
    values: np.ndarray, direction: Direction, dimensions: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.asarray([np.inf if direction == "lower" else -np.inf])

    # One- and two-variable rules use every observed development boundary.  The
    # higher-dimensional rules use a deterministic quantile lattice to avoid an
    # exponentially large, unstable threshold search.
    if dimensions <= 2:
        finite = np.unique(values)
    else:
        points = 25 if dimensions == 3 else 13
        finite = np.unique(np.quantile(values, np.linspace(0.0, 1.0, points)))

    if direction == "lower":
        # Strict-to-loose order gives deterministic conservative tie handling.
        return np.concatenate(([np.inf], finite[::-1]))
    return np.concatenate(([-np.inf], finite))


def replacement_mask(
    data: AuditData, rule: Rule, thresholds: dict[str, float]
) -> np.ndarray:
    mask = data.a18_prediction != data.p89_prediction
    for condition in rule.conditions:
        threshold = thresholds[condition.feature]
        if condition.direction == "lower":
            mask &= data.features[condition.feature] >= threshold
        else:
            mask &= data.features[condition.feature] <= threshold
    return mask


def decision_counts(data: AuditData, mask: np.ndarray) -> dict[str, int | float]:
    p89_correct = data.p89_prediction == data.labels
    a18_correct = data.a18_prediction == data.labels
    rescue = int(np.sum(mask & ~p89_correct & a18_correct))
    harm = int(np.sum(mask & p89_correct & ~a18_correct))
    replacement = int(np.sum(mask))
    neutral = replacement - rescue - harm
    correct = int(np.sum(np.where(mask, data.a18_prediction, data.p89_prediction) == data.labels))
    baseline_correct = int(np.sum(p89_correct))
    return {
        "rows": int(len(data.labels)),
        "baseline_correct": baseline_correct,
        "correct": correct,
        "accuracy": correct / len(data.labels),
        "delta_correct": correct - baseline_correct,
        "replacement_count": replacement,
        "coverage": replacement / len(data.labels),
        "rescue_count": rescue,
        "harm_count": harm,
        "neutral_count": neutral,
        "rescue_precision_among_decisive": (
            rescue / (rescue + harm) if rescue + harm else 0.0
        ),
    }


def select_thresholds(data: AuditData, rule: Rule) -> tuple[dict[str, float], dict[str, Any]]:
    disagreement = data.a18_prediction != data.p89_prediction
    grids = [
        threshold_values(
            data.features[condition.feature][disagreement],
            condition.direction,
            len(rule.conditions),
        )
        for condition in rule.conditions
    ]
    masks = []
    for condition, grid in zip(rule.conditions, grids):
        values = data.features[condition.feature][disagreement]
        if condition.direction == "lower":
            masks.append([values >= threshold for threshold in grid])
        else:
            masks.append([values <= threshold for threshold in grid])

    p89_correct = data.p89_prediction[disagreement] == data.labels[disagreement]
    a18_correct = data.a18_prediction[disagreement] == data.labels[disagreement]
    rescue_rows = ~p89_correct & a18_correct
    harm_rows = p89_correct & ~a18_correct

    best_score: tuple[int, int, int] | None = None
    best_indices: tuple[int, ...] | None = None
    best_counts: tuple[int, int, int] | None = None
    for indices in itertools.product(*(range(len(grid)) for grid in grids)):
        eligible = masks[0][indices[0]].copy()
        for dimension in range(1, len(indices)):
            eligible &= masks[dimension][indices[dimension]]
        rescue = int(np.sum(eligible & rescue_rows))
        harm = int(np.sum(eligible & harm_rows))
        replacement = int(np.sum(eligible))
        score = (rescue - harm, -harm, -replacement)
        if best_score is None or score > best_score:
            best_score = score
            best_indices = indices
            best_counts = (rescue, harm, replacement)

    if best_indices is None or best_counts is None or best_score is None:
        raise RuntimeError(f"threshold search failed for {rule.name}")
    selected = {
        condition.feature: float(grids[dimension][best_indices[dimension]])
        for dimension, condition in enumerate(rule.conditions)
    }
    selected_mask = replacement_mask(data, rule, selected)
    selected_metrics = decision_counts(data, selected_mask)
    if selected_metrics["rescue_count"] != best_counts[0]:
        raise RuntimeError("threshold search rescue count mismatch")
    if selected_metrics["harm_count"] != best_counts[1]:
        raise RuntimeError("threshold search harm count mismatch")
    if selected_metrics["replacement_count"] != best_counts[2]:
        raise RuntimeError("threshold search replacement count mismatch")
    return selected, selected_metrics


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isposinf(number):
            return "Infinity"
        if np.isneginf(number):
            return "-Infinity"
        return number
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = load_data(args.a18.resolve(), args.p89_reference.resolve())
    all_data = concatenate([splits[name] for name in SPLIT_NAMES], "all")

    baseline_correct = int(np.sum(all_data.p89_prediction == all_data.labels))
    baseline_accuracy = baseline_correct / len(all_data.labels)
    if baseline_correct != 2117 or len(all_data.labels) != 2470:
        raise RuntimeError(
            f"frozen P89 baseline mismatch: {baseline_correct}/{len(all_data.labels)}"
        )

    rule_results: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    all_per_subject_rows: list[dict[str, Any]] = []
    prediction_store: dict[str, np.ndarray] = {
        "sample_ids": all_data.sample_ids,
        "users": all_data.users,
        "labels": all_data.labels,
        "p89_safe_prediction": all_data.p89_prediction,
        "a18_best_session_prediction": all_data.a18_prediction,
    }
    thresholds_by_rule: dict[str, Any] = {}

    for rule in RULES:
        held_predictions: list[np.ndarray] = []
        held_masks: list[np.ndarray] = []
        held_ids: list[np.ndarray] = []
        threshold_rows: dict[str, Any] = {}

        for held_name in SPLIT_NAMES:
            development_names = [name for name in SPLIT_NAMES if name != held_name]
            development = concatenate(
                [splits[name] for name in development_names],
                f"development_for_{held_name}",
            )
            held = splits[held_name]
            thresholds, development_metrics = select_thresholds(development, rule)
            mask = replacement_mask(held, rule, thresholds)
            held_metrics = decision_counts(held, mask)
            prediction = np.where(mask, held.a18_prediction, held.p89_prediction)

            threshold_rows[held_name] = {
                "selected_on": development_names,
                "thresholds": thresholds,
                "development_metrics": development_metrics,
                "held_metrics": held_metrics,
            }
            outer_rows.append(
                finite_json(
                    {
                        "rule": rule.name,
                        "held_split": held_name,
                        "development_splits": "+".join(development_names),
                        **{
                            f"threshold_{key}": value for key, value in thresholds.items()
                        },
                        **{f"development_{key}": value for key, value in development_metrics.items()},
                        **{f"held_{key}": value for key, value in held_metrics.items()},
                    }
                )
            )
            held_ids.append(held.sample_ids)
            held_predictions.append(prediction.astype(np.int64))
            held_masks.append(mask)

        crossfit_lookup: dict[str, tuple[int, bool]] = {}
        for ids, predictions, masks_for_split in zip(held_ids, held_predictions, held_masks):
            for sample_id, prediction, mask in zip(ids, predictions, masks_for_split):
                crossfit_lookup[str(sample_id)] = (int(prediction), bool(mask))
        crossfit_prediction = np.asarray(
            [crossfit_lookup[sample_id][0] for sample_id in all_data.sample_ids],
            dtype=np.int64,
        )
        crossfit_mask = np.asarray(
            [crossfit_lookup[sample_id][1] for sample_id in all_data.sample_ids],
            dtype=bool,
        )
        crossfit_metrics = decision_counts(all_data, crossfit_mask)
        if not np.array_equal(
            crossfit_prediction,
            np.where(crossfit_mask, all_data.a18_prediction, all_data.p89_prediction),
        ):
            raise RuntimeError(f"cross-fit assembly mismatch for {rule.name}")

        subject_deltas: list[int] = []
        for user in sorted(np.unique(all_data.users), key=lambda value: int(value[4:])):
            subject_rows = all_data.users == user
            subject_data = AuditData(
                split=str(user),
                sample_ids=all_data.sample_ids[subject_rows],
                labels=all_data.labels[subject_rows],
                users=all_data.users[subject_rows],
                p89_prediction=all_data.p89_prediction[subject_rows],
                a18_prediction=all_data.a18_prediction[subject_rows],
                features={
                    key: value[subject_rows] for key, value in all_data.features.items()
                },
            )
            subject_metrics = decision_counts(subject_data, crossfit_mask[subject_rows])
            subject_deltas.append(int(subject_metrics["delta_correct"]))
            all_per_subject_rows.append(
                {"rule": rule.name, "subject": str(user), **subject_metrics}
            )

        rule_result = {
            "rule": rule.name,
            "note": rule.note,
            "conditions": [condition.__dict__ for condition in rule.conditions],
            **crossfit_metrics,
            "delta_pp": 100.0 * (crossfit_metrics["accuracy"] - baseline_accuracy),
            "passes_p89_gate": crossfit_metrics["accuracy"] > baseline_accuracy,
            "subjects_improved": int(np.sum(np.asarray(subject_deltas) > 0)),
            "subjects_unchanged": int(np.sum(np.asarray(subject_deltas) == 0)),
            "subjects_harmed": int(np.sum(np.asarray(subject_deltas) < 0)),
            "worst_subject_delta_correct": int(min(subject_deltas)),
            "best_subject_delta_correct": int(max(subject_deltas)),
        }
        rule_results.append(rule_result)
        thresholds_by_rule[rule.name] = threshold_rows
        prediction_store[f"{rule.name}_prediction"] = crossfit_prediction
        prediction_store[f"{rule.name}_replacement_mask"] = crossfit_mask

    champion = max(
        rule_results,
        key=lambda row: (
            int(row["correct"]),
            -int(row["harm_count"]),
            -int(row["replacement_count"]),
        ),
    )
    champion_name = str(champion["rule"])

    per_subject_rows = [
        {key: value for key, value in row.items() if key != "rule"}
        for row in all_per_subject_rows
        if row["rule"] == champion_name
    ]
    stable_candidates = [
        row
        for row in rule_results
        if int(row["worst_subject_delta_correct"]) >= 0
        and int(row["correct"]) > baseline_correct
    ]
    stability_champion = (
        max(
            stable_candidates,
            key=lambda row: (
                int(row["correct"]),
                -int(row["harm_count"]),
                -int(row["replacement_count"]),
            ),
        )
        if stable_candidates
        else None
    )

    metrics = {
        "protocol": {
            "training_performed": False,
            "default_prediction": "P89 safe",
            "replacement_prediction": "A18 best checkpoint + source-safe Session",
            "class_agnostic": True,
            "user_id_used_as_rule_feature": False,
            "test_labels_read": False,
            "threshold_selection": (
                "Three outer subject-cohort folds. For each held H1/H2/H3 cohort, "
                "thresholds maximize net correct on the other two cohorts' frozen OOF "
                "predictions; ties prefer fewer harms then fewer replacements."
            ),
            "threshold_lattice": {
                "one_or_two_variables": "all unique development OOF boundaries",
                "three_variables": "25-point development OOF quantile grid",
                "four_variables": "13-point development OOF quantile grid",
            },
            "coverage_definition": "replacement_count / all 2470 OOF rows",
            "p89_confidence_definitions": {
                "p89_max_confidence": "max of adjusted single-row P89 probability",
                "p89_safe_support": "probability assigned to the deployed P89 safe label",
            },
        },
        "inputs": {
            "a18_oof": str(args.a18.resolve()),
            "a18_oof_sha256": sha256(args.a18.resolve()),
            "p89_reference": str(args.p89_reference.resolve()),
            "p89_reference_sha256": sha256(args.p89_reference.resolve()),
        },
        "baseline": {
            "model": "P89 safe",
            "rows": int(len(all_data.labels)),
            "correct": baseline_correct,
            "accuracy": baseline_accuracy,
        },
        "a18_control": {
            "correct": int(np.sum(all_data.a18_prediction == all_data.labels)),
            "accuracy": float(np.mean(all_data.a18_prediction == all_data.labels)),
            "disagreements_with_p89": int(
                np.sum(all_data.a18_prediction != all_data.p89_prediction)
            ),
        },
        "rule_results": rule_results,
        "champion": champion,
        "stability_champion": stability_champion,
        "submission_gate": {
            "threshold": baseline_accuracy,
            "passes": bool(champion["accuracy"] > baseline_accuracy),
            "next_stage_eligible": bool(champion["accuracy"] > baseline_accuracy),
            "test_or_submission_generated": False,
        },
    }

    (output / "metrics.json").write_text(
        json.dumps(finite_json(metrics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "selected_thresholds.json").write_text(
        json.dumps(finite_json(thresholds_by_rule), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "rule_results.csv", [finite_json(row) for row in rule_results])
    write_csv(output / "outer_fold_results.csv", outer_rows)
    write_csv(output / "per_subject_results.csv", all_per_subject_rows)
    write_csv(output / "champion_per_subject.csv", per_subject_rows)
    np.savez_compressed(output / "crossfit_predictions.npz", **prediction_store)

    print(json.dumps(finite_json(metrics["baseline"]), ensure_ascii=False))
    for row in sorted(rule_results, key=lambda item: int(item["correct"]), reverse=True):
        print(
            f"{row['rule']}: {row['correct']}/{row['rows']}="
            f"{100.0 * row['accuracy']:.4f}% delta={row['delta_correct']:+d} "
            f"replace={row['replacement_count']} rescue/harm="
            f"{row['rescue_count']}/{row['harm_count']}"
        )
    print(
        f"champion={champion_name} gate="
        f"{'PASS' if metrics['submission_gate']['passes'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
