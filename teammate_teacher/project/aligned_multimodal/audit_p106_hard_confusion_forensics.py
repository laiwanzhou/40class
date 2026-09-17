"""Run the frozen P106 fine-grained hard-confusion evidence audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_p102_session_closure import load_npz
from p100a_global_teacher_data import FOLD_USERS, H3_USERS, load_p100a_data
from p106_forensic_features import FINE_BLOCKS, FineFeatureBlock, load_forensic_blocks
from train_p104_modality_specialists_oof import (
    DEFAULT_SESSION,
    PCA_COMPONENTS,
    Projection,
    cyclic_shuffle_source,
    family_mask,
    fit_specialist,
    intervention_metrics,
    metric_bundle,
    trigger_metrics,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "runs/p106_hard_confusion_forensics_v1"
FORENSIC_FAMILIES = (
    {"family_key": "8__10", "classes": [8, 10], "priority": 1},
    {"family_key": "24__26", "classes": [24, 26], "priority": 2},
    {"family_key": "24__27", "classes": [24, 27], "priority": 2},
    {"family_key": "32__34", "classes": [32, 34], "priority": 3},
    {"family_key": "6__37", "classes": [6, 37], "priority": 4},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty P106 CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selection_key(result: dict[str, Any]) -> tuple[float, float, float, int]:
    metrics = result["source_inner_specialist"]
    return (
        float(metrics["balanced_accuracy"]),
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        -FINE_BLOCKS.index(result["block"]),
    )


def per_subject_intervention(
    labels: np.ndarray,
    a_prediction: np.ndarray,
    specialist: np.ndarray,
    users: np.ndarray,
) -> dict[str, dict[str, Any]]:
    output = {}
    for user in sorted(set(users.tolist())):
        selected = users == user
        output[user] = {
            "rows": int(selected.sum()),
            **intervention_metrics(
                labels[selected], a_prediction[selected], specialist[selected]
            ),
        }
    return output


def evaluate_block_fold(
    block: FineFeatureBlock,
    outer_fold: int,
    families: tuple[dict[str, Any], ...],
    components: int,
    inner_limit: int | None,
    sample_ids: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    a_probability: np.ndarray,
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source = fold_ids != outer_fold
    held = ~source
    source_rows = np.flatnonzero(source).astype(np.int64)
    held_rows = np.flatnonzero(held).astype(np.int64)
    inner_users = sorted(set(users[source].tolist()))
    if inner_limit is not None:
        inner_users = inner_users[:inner_limit]
    inner_prediction = {
        config["family_key"]: np.full(len(labels), -1, dtype=np.int64)
        for config in families
    }
    inner_covered = np.zeros(len(labels), dtype=bool)
    for inner_user in inner_users:
        inner_train = np.flatnonzero(source & (users != inner_user)).astype(np.int64)
        inner_held = source & (users == inner_user)
        projection = Projection.fit(block.aligned, inner_train, components)
        projected = np.full((len(labels), projection.components), np.nan, dtype=np.float32)
        projected[source_rows] = projection.transform(block.aligned[source_rows])
        for config in families:
            classes = config["classes"]
            train_rows = np.flatnonzero(
                source & (users != inner_user) & family_mask(labels, classes)
            ).astype(np.int64)
            validation_rows = np.flatnonzero(
                inner_held & family_mask(labels, classes)
            ).astype(np.int64)
            if not len(validation_rows):
                continue
            classifier = fit_specialist(projected, labels, train_rows, classes)
            inner_prediction[config["family_key"]][validation_rows] = classifier.predict(
                projected[validation_rows]
            )
        inner_covered |= inner_held

    projection = Projection.fit(block.aligned, source_rows, components)
    projected_source = projection.transform(block.aligned[source_rows])
    source_position = {int(row): index for index, row in enumerate(source_rows.tolist())}
    shuffle_map = cyclic_shuffle_source(sample_ids, users, held)
    variants = {
        "aligned": projection.transform(block.aligned[held_rows]),
        "shuffle": projection.transform(block.aligned[shuffle_map[held_rows]]),
        "zero": projection.zero(len(held_rows)),
    }
    held_position = {int(row): index for index, row in enumerate(held_rows.tolist())}
    a_held_probability = a_probability[held_rows]
    a_held_prediction = a_held_probability.argmax(axis=1)
    output: list[dict[str, Any]] = []
    for config in families:
        key = config["family_key"]
        classes = config["classes"]
        source_eval_rows = np.flatnonzero(
            inner_covered & family_mask(labels, classes)
        ).astype(np.int64)
        if np.any(inner_prediction[key][source_eval_rows] < 0):
            raise RuntimeError(f"P106 inner coverage failed: {block.name}/{outer_fold}/{key}")
        source_metrics = metric_bundle(
            labels[source_eval_rows],
            inner_prediction[key][source_eval_rows],
            users[source_eval_rows],
            classes,
        )
        family_source_rows = np.flatnonzero(source & family_mask(labels, classes)).astype(
            np.int64
        )
        family_source_positions = np.asarray(
            [source_position[int(row)] for row in family_source_rows], dtype=np.int64
        )
        classifier = fit_specialist(
            projected_source,
            labels[source_rows],
            family_source_positions,
            classes,
        )
        all_predictions = {
            variant: classifier.predict(values).astype(np.int64)
            for variant, values in variants.items()
        }
        family_held_rows = np.flatnonzero(held & family_mask(labels, classes)).astype(
            np.int64
        )
        family_positions = np.asarray(
            [held_position[int(row)] for row in family_held_rows], dtype=np.int64
        )
        family_labels = labels[family_held_rows]
        family_users = users[family_held_rows]
        family_a = a_held_prediction[family_positions]
        variant_results = {}
        for variant, predictions in all_predictions.items():
            family_prediction = predictions[family_positions]
            confusion = {
                f"{true_class}->{predicted_class}": int(
                    np.sum(
                        (family_labels == true_class)
                        & (family_prediction == predicted_class)
                    )
                )
                for true_class in classes
                for predicted_class in classes
            }
            variant_results[variant] = {
                "metrics": metric_bundle(
                    family_labels, family_prediction, family_users, classes
                ),
                "intervention": intervention_metrics(
                    family_labels, family_a, family_prediction
                ),
                "per_subject_intervention": per_subject_intervention(
                    family_labels, family_a, family_prediction, family_users
                ),
                "confusion": confusion,
            }
            for row, predicted, a_predicted in zip(
                family_held_rows.tolist(),
                family_prediction.tolist(),
                family_a.tolist(),
            ):
                prediction_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "family_key": key,
                        "classes": "|".join(map(str, classes)),
                        "block": block.name,
                        "variant": variant,
                        "row_index": row,
                        "sample_id": sample_ids[row],
                        "subject": users[row],
                        "label": int(labels[row]),
                        "a_prediction": int(a_predicted),
                        "specialist_prediction": int(predicted),
                    }
                )
        output.append(
            {
                "outer_fold": outer_fold,
                "held_users": list(FOLD_USERS[outer_fold]),
                "family_key": key,
                "classes": classes,
                "priority": config["priority"],
                "block": block.name,
                "source_sample_count": len(family_source_rows),
                "source_inner_oof_rows": len(source_eval_rows),
                "held_sample_count": len(family_held_rows),
                "held_class_support": {
                    str(class_id): int(np.sum(family_labels == class_id))
                    for class_id in classes
                },
                "held_available_fraction": float(np.mean(block.available[family_held_rows])),
                "a_family": metric_bundle(
                    family_labels, family_a, family_users, classes
                ),
                "source_inner_specialist": source_metrics,
                "variants": variant_results,
                "source_vs_held_balanced_accuracy_gap": float(
                    source_metrics["balanced_accuracy"]
                    - variant_results["aligned"]["metrics"]["balanced_accuracy"]
                ),
                "deployable_trigger": trigger_metrics(
                    labels[held_rows],
                    a_held_probability,
                    all_predictions["aligned"],
                    classes,
                ),
                "projection": {
                    "components": projection.components,
                    "explained_variance": float(
                        np.sum(projection.pca.explained_variance_ratio_)
                    ),
                    "fit_rows_all_source_classes": len(source_rows),
                },
            }
        )
    return output


def aggregate_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    first = results[0]
    variants = {}
    for variant in ("aligned", "shuffle", "zero"):
        correct = sum(value["variants"][variant]["metrics"]["correct"] for value in results)
        rescue = sum(
            value["variants"][variant]["intervention"]["rescue"] for value in results
        )
        harm = sum(
            value["variants"][variant]["intervention"]["harm"] for value in results
        )
        variants[variant] = {
            "correct": int(correct),
            "rescue": int(rescue),
            "harm": int(harm),
            "net": int(rescue - harm),
        }
    subject_net: dict[str, int] = defaultdict(int)
    subject_rows: dict[str, int] = defaultdict(int)
    subject_correct: dict[str, int] = defaultdict(int)
    for value in results:
        for user, metrics in value["variants"]["aligned"]["per_subject_intervention"].items():
            subject_net[user] += int(metrics["net"])
        for user, metrics in value["variants"]["aligned"]["metrics"]["per_subject"].items():
            subject_rows[user] += int(metrics["rows"])
            subject_correct[user] += int(round(metrics["accuracy"] * metrics["rows"]))
    subject_accuracy = {
        user: float(subject_correct[user] / subject_rows[user])
        for user in sorted(subject_rows)
    }
    per_class_support: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    confusion: dict[str, int] = defaultdict(int)
    for value in results:
        recalls = value["variants"]["aligned"]["metrics"]["per_class_recall"]
        for class_id, support in value["held_class_support"].items():
            per_class_support[class_id] += int(support)
            if support:
                per_class_correct[class_id] += int(
                    round(float(recalls[class_id]) * support)
                )
        for edge, count in value["variants"]["aligned"]["confusion"].items():
            confusion[edge] += int(count)
    per_class_recall = {
        class_id: float(per_class_correct[class_id] / max(support, 1))
        for class_id, support in sorted(per_class_support.items(), key=lambda item: int(item[0]))
    }
    class_ids = list(map(int, first["classes"]))
    f1_values = []
    for class_id in class_ids:
        true_positive = confusion[f"{class_id}->{class_id}"]
        false_negative = sum(
            confusion[f"{class_id}->{other}"]
            for other in class_ids
            if other != class_id
        )
        false_positive = sum(
            confusion[f"{other}->{class_id}"]
            for other in class_ids
            if other != class_id
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(float(2 * true_positive / denominator) if denominator else 0.0)
    rows = sum(value["held_sample_count"] for value in results)
    aligned = variants["aligned"]
    return {
        "family_key": first["family_key"],
        "classes": first["classes"],
        "block": first["block"],
        "selected_folds": [int(value["outer_fold"]) for value in results],
        "selected_fold_count": len(results),
        "held_rows": rows,
        "a_family_correct": sum(value["a_family"]["correct"] for value in results),
        "aligned_accuracy": float(aligned["correct"] / max(rows, 1)),
        "held_balanced_accuracy": float(np.mean(list(per_class_recall.values()))),
        "held_macro_f1": float(np.mean(f1_values)),
        "mean_held_macro_f1": float(
            np.mean([value["variants"]["aligned"]["metrics"]["macro_f1"] for value in results])
        ),
        "per_class_recall": per_class_recall,
        "variants": variants,
        "aligned_minus_shuffle_correct": int(
            aligned["correct"] - variants["shuffle"]["correct"]
        ),
        "aligned_minus_zero_correct": int(
            aligned["correct"] - variants["zero"]["correct"]
        ),
        "positive_folds": int(
            sum(value["variants"]["aligned"]["intervention"]["net"] > 0 for value in results)
        ),
        "negative_folds": int(
            sum(value["variants"]["aligned"]["intervention"]["net"] < 0 for value in results)
        ),
        "subject_net": dict(sorted(subject_net.items())),
        "subject_accuracy": subject_accuracy,
        "worst_subject": min(
            (
                {
                    "subject": user,
                    "rows": subject_rows[user],
                    "accuracy": accuracy,
                    "net": subject_net[user],
                }
                for user, accuracy in subject_accuracy.items()
            ),
            key=lambda value: (value["accuracy"], value["subject"]),
        ),
        "positive_subjects": int(sum(value > 0 for value in subject_net.values())),
        "negative_subjects": int(sum(value < 0 for value in subject_net.values())),
        "deployable_trigger_net": int(
            sum(value["deployable_trigger"]["net"] for value in results)
        ),
        "mean_source_balanced_accuracy": float(
            np.mean([value["source_inner_specialist"]["balanced_accuracy"] for value in results])
        ),
        "mean_held_balanced_accuracy": float(
            np.mean([value["variants"]["aligned"]["metrics"]["balanced_accuracy"] for value in results])
        ),
    }


def verdict(aggregate: dict[str, Any]) -> str:
    if aggregate["selected_fold_count"] < 2:
        return "INSUFFICIENT_SELECTION_STABILITY"
    aligned = aggregate["variants"]["aligned"]
    if (
        aligned["net"] > 0
        and aggregate["aligned_minus_shuffle_correct"] > 0
        and aggregate["aligned_minus_zero_correct"] > 0
        and aggregate["positive_folds"] >= 2
        and aggregate["positive_subjects"] >= aggregate["negative_subjects"]
        and aggregate["deployable_trigger_net"] >= 0
    ):
        return "FORENSIC_SPECIALIST_CANDIDATE"
    if (
        aggregate["aligned_minus_shuffle_correct"] > 0
        and aggregate["aligned_minus_zero_correct"] > 0
    ):
        return "MECHANISM_ONLY"
    return "NO_CURRENT_FINE_FEATURE_EVIDENCE"


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = load_npz(args.session.resolve())
    data = load_p100a_data()
    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    if str(np.asarray(session["selected_system"]).item()) != "VS_session":
        raise RuntimeError("P106 forensics requires frozen A = VS + Session")
    if not np.array_equal(sample_ids, data.sample_ids.astype(str)):
        raise RuntimeError("P100/P106 sample order differs")
    if not np.array_equal(users, data.users) or not np.array_equal(labels, data.labels):
        raise RuntimeError("P100/P106 metadata differs")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P106 forensics")
    blocks = load_forensic_blocks(data)
    selected_blocks = FINE_BLOCKS[:2] if args.smoke else FINE_BLOCKS
    families = FORENSIC_FAMILIES[:1] if args.smoke else FORENSIC_FAMILIES
    folds = [0] if args.smoke else [0, 1, 2, 3]
    components = 8 if args.smoke else PCA_COMPONENTS
    inner_limit = 2 if args.smoke else None
    prediction_rows: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    for outer_fold in folds:
        for name in selected_blocks:
            fold_results.extend(
                evaluate_block_fold(
                    blocks[name],
                    outer_fold,
                    families,
                    components,
                    inner_limit,
                    sample_ids,
                    users,
                    labels,
                    fold_ids,
                    a_probability,
                    prediction_rows,
                )
            )
            print(
                f"P106 forensic fold={outer_fold} block={name} families={len(families)}",
                flush=True,
            )

    by_fold_family: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_family_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in fold_results:
        by_fold_family[(result["outer_fold"], result["family_key"])].append(result)
        by_family_block[(result["family_key"], result["block"])].append(result)
    selected_results = []
    for values in by_fold_family.values():
        chosen = max(values, key=selection_key)
        chosen["source_selected"] = True
        selected_results.append(chosen)
        for value in values:
            value.setdefault("source_selected", False)
    all_aggregates = [
        aggregate_group(values) for _, values in sorted(by_family_block.items())
    ]
    selected_by_family_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in selected_results:
        selected_by_family_block[(value["family_key"], value["block"])].append(value)
    selected_support = [
        aggregate_group(values)
        for _, values in sorted(selected_by_family_block.items())
    ]
    for aggregate in selected_support:
        aggregate["verdict"] = verdict(aggregate)
    selected_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in selected_results:
        selected_by_family[value["family_key"]].append(value)
    selected_family_aggregates = []
    for family, values in sorted(selected_by_family.items()):
        aggregate = aggregate_group(values)
        aggregate["block"] = "source_selected_per_fold"
        aggregate["block_counts"] = dict(
            sorted(Counter(value["block"] for value in values).items())
        )
        selected_family_aggregates.append(aggregate)

    confusion_map = []
    for config in families:
        family = config["family_key"]
        supports = [value for value in selected_support if value["family_key"] == family]
        best = max(
            supports,
            key=lambda value: (
                value["selected_fold_count"],
                value["mean_source_balanced_accuracy"],
                -FINE_BLOCKS.index(value["block"]),
            ),
        )
        best_verdict = best["verdict"]
        strength = (
            "STRONG"
            if best_verdict == "FORENSIC_SPECIALIST_CANDIDATE" and best["selected_fold_count"] >= 3
            else "MODERATE"
            if best_verdict == "FORENSIC_SPECIALIST_CANDIDATE"
            else "WEAK_MECHANISM"
            if best_verdict == "MECHANISM_ONLY"
            else "NONE"
        )
        confusion_map.append(
            {
                "family_key": family,
                "best_evidence": best["block"],
                "selection_count": best["selected_fold_count"],
                "specialist_candidate": best_verdict == "FORENSIC_SPECIALIST_CANDIDATE",
                "evidence_strength": strength,
                "verdict": best_verdict,
                "aligned_net": best["variants"]["aligned"]["net"],
                "aligned_minus_shuffle_correct": best["aligned_minus_shuffle_correct"],
                "aligned_minus_zero_correct": best["aligned_minus_zero_correct"],
                "trigger_net": best["deployable_trigger_net"],
            }
        )

    write_csv(output / "predictions.csv", prediction_rows)
    write_csv(
        output / "fold_selection.csv",
        [
            {
                "outer_fold": value["outer_fold"],
                "family_key": value["family_key"],
                "block": value["block"],
                "source_balanced_accuracy": value["source_inner_specialist"]["balanced_accuracy"],
                "source_macro_f1": value["source_inner_specialist"]["macro_f1"],
                "source_accuracy": value["source_inner_specialist"]["accuracy"],
                "held_balanced_accuracy": value["variants"]["aligned"]["metrics"]["balanced_accuracy"],
                "held_rescue": value["variants"]["aligned"]["intervention"]["rescue"],
                "held_harm": value["variants"]["aligned"]["intervention"]["harm"],
                "held_net": value["variants"]["aligned"]["intervention"]["net"],
                "trigger_net": value["deployable_trigger"]["net"],
            }
            for value in sorted(
                selected_results, key=lambda item: (item["outer_fold"], item["priority"])
            )
        ],
    )
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "protocol": "P106 source-inner-LOSO fine-feature panel; no unified B/router/Student",
        "data": {
            "rows": len(labels),
            "subjects": sorted(set(users.tolist())),
            "folds": folds,
            "families": [value["family_key"] for value in families],
            "blocks": list(selected_blocks),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "recipe": {
            "projection": "inner/outer source-train-only StandardScaler + randomized PCA",
            "pca_components": components,
            "classifier": "balanced LogisticRegression C=1.0",
            "selection": "source-inner balanced accuracy -> macro F1 -> accuracy -> fixed block order",
            "counterfactuals": "aligned / within-held-subject cyclic shuffle / source-mean zero",
            "training_population": "all source samples of the family classes",
        },
        "block_audit": {
            name: {
                **blocks[name].audit,
                "descriptor_dim": int(blocks[name].aligned.shape[1]),
                "available_rows": int(blocks[name].available.sum()),
            }
            for name in selected_blocks
        },
        "fold_results": fold_results,
        "fold_selection": selected_results,
        "all_block_aggregates": all_aggregates,
        "source_selected_block_support": selected_support,
        "source_selected_family_aggregates": selected_family_aggregates,
        "confusion_map": confusion_map,
        "optical_flow": {
            "audited": False,
            "reason": "no complete verified descriptor; no backbone trained in P106",
        },
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
        "unified_b_teacher_trained": False,
        "candidate_reranker_trained": False,
        "student_started": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "fold_results": len(fold_results),
                "fold_selection": [
                    {
                        "fold": value["outer_fold"],
                        "family": value["family_key"],
                        "block": value["block"],
                        "source_ba": value["source_inner_specialist"]["balanced_accuracy"],
                        "held_net": value["variants"]["aligned"]["intervention"]["net"],
                    }
                    for value in selected_results
                ],
                "confusion_map": confusion_map,
                "h3_rows_selected": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
