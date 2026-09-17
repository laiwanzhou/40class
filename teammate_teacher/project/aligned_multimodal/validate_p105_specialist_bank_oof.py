"""Validate the locked P105 confusion specialists and their simple bank router."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_p102_hard_set import source_crossfit_session_probability
from audit_p102_session_closure import load_npz
from audit_p87_sequence_decoder import DecoderConfig, align_metadata
from p100a_global_teacher_data import FOLD_USERS, H3_USERS, load_p100a_data
from p104_modality_data import ModalityFeatures, load_modality_features
from train_p104_modality_specialists_oof import (
    PCA_COMPONENTS,
    Projection,
    cyclic_shuffle_source,
    family_mask,
    fit_specialist,
    intervention_metrics,
    metric_bundle,
)
from train_p104_pair_specialists_oof import held_variants


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p105_specialist_bank_oof_v1"
SPECIALISTS = (
    {"family_key": "3__5", "classes": [3, 5], "modalities": ["LocalV", "Skeleton"]},
    {"family_key": "7__37", "classes": [7, 37], "modalities": ["LocalV"]},
    {"family_key": "7__8", "classes": [7, 8], "modalities": ["GlobalV"]},
    {"family_key": "8__9", "classes": [8, 9], "modalities": ["LocalV"]},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--session-summary", type=Path, default=DEFAULT_SESSION_SUMMARY)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--families", default=",".join(value["family_key"] for value in SPECIALISTS))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_folds(value: str) -> list[int]:
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds or not set(folds) <= set(range(4)):
        raise ValueError(f"invalid P105 folds: {value}")
    return folds


def select_specialists(value: str) -> list[dict[str, Any]]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    lookup = {item["family_key"]: item for item in SPECIALISTS}
    unknown = sorted(set(requested) - set(lookup))
    if not requested or unknown:
        raise ValueError(f"invalid P105 families: unknown={unknown}")
    return [lookup[key] for key in requested]


def decoder_config(summary: dict[str, Any], fold: int) -> DecoderConfig:
    selected = summary["folds"][fold]["selected"]
    return DecoderConfig(
        gap_seconds=float(selected["gap_seconds"]),
        transition_weight=float(selected["transition_weight"]),
        trigram_backoff=float(selected["trigram_backoff"]),
        beam_width=int(selected["beam_width"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty P105 CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concatenate(values: dict[str, np.ndarray], modalities: list[str]) -> np.ndarray:
    return np.concatenate([values[name] for name in modalities], axis=1)


def canonical_variants(
    config: dict[str, Any],
    representations: dict[str, ModalityFeatures],
    projections: dict[str, Projection],
    held_rows: np.ndarray,
    shuffle_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    modalities = list(config["modalities"])
    if len(modalities) == 2:
        variants = held_variants(
            representations,
            projections,
            modalities,
            held_rows,
            shuffle_rows,
        )
        variants["shuffle"] = variants["shuffle_both"]
        variants["zero"] = variants["zero_both"]
        return variants
    name = modalities[0]
    projection = projections[name]
    representation = representations[name]
    variants = {
        "aligned": projection.transform(representation.aligned[held_rows]),
        "shuffle": projection.transform(representation.aligned[shuffle_rows]),
        "zero": projection.zero(len(held_rows)),
    }
    for intervention, raw in representation.interventions.items():
        variants[f"{name.lower()}_{intervention}"] = projection.transform(raw[held_rows])
    return variants


def route_assignments(
    probability: np.ndarray,
    configs: list[dict[str, Any]],
    enabled: set[str],
) -> np.ndarray:
    """Apply the one frozen, label-free top-3 confusion-graph router."""

    values = np.asarray(probability, dtype=np.float64)
    order = np.argsort(values, axis=1)[:, ::-1]
    output = np.full(len(values), "", dtype=object)
    for row in range(len(values)):
        top1 = int(order[row, 0])
        top3 = set(map(int, order[row, :3].tolist()))
        candidates: list[tuple[float, int, str]] = []
        for index, config in enumerate(configs):
            key = str(config["family_key"])
            if key not in enabled or top1 not in config["classes"]:
                continue
            other = next(value for value in config["classes"] if value != top1)
            if other in top3:
                candidates.append((float(values[row, other]), -index, key))
        if candidates:
            output[row] = max(candidates)[2]
    return output


def exact_edge_oracle_assignments(
    labels: np.ndarray,
    a_prediction: np.ndarray,
    configs: list[dict[str, Any]],
) -> np.ndarray:
    output = np.full(len(labels), "", dtype=object)
    for config in configs:
        classes = set(map(int, config["classes"]))
        selected = np.asarray(
            [
                int(label) != int(prediction)
                and {int(label), int(prediction)} == classes
                for label, prediction in zip(labels, a_prediction)
            ],
            dtype=bool,
        )
        if np.any(output[selected] != ""):
            raise RuntimeError("P105 exact-edge oracle is not unique")
        output[selected] = config["family_key"]
    return output


def system_metrics(
    labels: np.ndarray,
    a_prediction: np.ndarray,
    system_prediction: np.ndarray,
    users: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    a_correct = a_prediction == labels
    system_correct = system_prediction == labels
    per_subject = []
    for subject in sorted(set(users.tolist())):
        selected = users == subject
        per_subject.append(
            {
                "subject": subject,
                "rows": int(selected.sum()),
                "a_correct": int(a_correct[selected].sum()),
                "system_correct": int(system_correct[selected].sum()),
                "net": int(system_correct[selected].sum() - a_correct[selected].sum()),
            }
        )
    return {
        "rows": len(labels),
        "a_correct": int(a_correct.sum()),
        "system_correct": int(system_correct.sum()),
        "a_accuracy": float(a_correct.mean()),
        "system_accuracy": float(system_correct.mean()),
        "rescue": int(np.sum((~a_correct) & system_correct)),
        "harm": int(np.sum(a_correct & (~system_correct))),
        "net": int(system_correct.sum() - a_correct.sum()),
        "per_fold": {
            str(fold): {
                "rows": int(np.sum(fold_ids == fold)),
                "a_correct": int(a_correct[fold_ids == fold].sum()),
                "system_correct": int(system_correct[fold_ids == fold].sum()),
                "net": int(
                    system_correct[fold_ids == fold].sum()
                    - a_correct[fold_ids == fold].sum()
                ),
            }
            for fold in sorted(set(fold_ids.tolist()))
        },
        "per_subject": per_subject,
        "subject_mean_accuracy": float(
            np.mean(
                [
                    value["system_correct"] / value["rows"]
                    for value in per_subject
                ]
            )
        ),
        "worst_subject": min(
            per_subject, key=lambda value: (value["net"], value["subject"])
        ),
    }


def apply_routes(
    a_prediction: np.ndarray,
    routes: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.asarray(a_prediction, dtype=np.int64).copy()
    for family, values in predictions.items():
        selected = routes == family
        output[selected] = values[selected]
    return output


def route_details(
    routes: np.ndarray,
    labels: np.ndarray,
    a_prediction: np.ndarray,
    system_prediction: np.ndarray,
    configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def contribution(selected: np.ndarray) -> dict[str, int]:
        a_correct = a_prediction[selected] == labels[selected]
        system_correct = system_prediction[selected] == labels[selected]
        return {
            "rows": int(selected.sum()),
            "rescue": int(np.sum((~a_correct) & system_correct)),
            "harm": int(np.sum(a_correct & (~system_correct))),
            "net": int(system_correct.sum() - a_correct.sum()),
        }

    output = []
    for config in configs:
        key = config["family_key"]
        selected = routes == key
        a_correct = a_prediction[selected] == labels[selected]
        system_correct = system_prediction[selected] == labels[selected]
        true_family = family_mask(labels[selected], config["classes"])
        in_family = selected & family_mask(labels, config["classes"])
        outside_family = selected & (~family_mask(labels, config["classes"]))
        output.append(
            {
                "family_key": key,
                "classes": config["classes"],
                "routed_rows": int(selected.sum()),
                "true_family_rows": int(true_family.sum()),
                "true_family_precision": float(true_family.mean()) if selected.any() else 0.0,
                "rescue": int(np.sum((~a_correct) & system_correct)),
                "harm": int(np.sum(a_correct & (~system_correct))),
                "net": int(system_correct.sum() - a_correct.sum()),
                "in_family": contribution(in_family),
                "outside_family": contribution(outside_family),
            }
        )
    return output


def subject_family_summary(
    labels: np.ndarray,
    a_prediction: np.ndarray,
    specialist_prediction: np.ndarray,
    users: np.ndarray,
) -> dict[str, Any]:
    records = []
    for subject in sorted(set(users.tolist())):
        selected = users == subject
        a_correct = int(np.sum(a_prediction[selected] == labels[selected]))
        specialist = int(np.sum(specialist_prediction[selected] == labels[selected]))
        records.append(
            {
                "subject": subject,
                "rows": int(selected.sum()),
                "a_correct": a_correct,
                "specialist_correct": specialist,
                "net": specialist - a_correct,
            }
        )
    return {
        "records": records,
        "subject_mean_accuracy": float(
            np.mean([value["specialist_correct"] / value["rows"] for value in records])
        ),
        "positive_subjects": sum(value["net"] > 0 for value in records),
        "neutral_subjects": sum(value["net"] == 0 for value in records),
        "negative_subjects": sum(value["net"] < 0 for value in records),
        "worst_subject": min(records, key=lambda value: (value["net"], value["subject"])),
    }


def stable_specialist(aggregate: dict[str, Any]) -> bool:
    fold_nets = list(aggregate["per_fold_net"].values())
    subjects = aggregate["per_subject"]
    return bool(
        aggregate["net"] > 0
        and aggregate["aligned_minus_shuffle_correct"] > 0
        and aggregate["aligned_minus_zero_correct"] > 0
        and sum(value >= 0 for value in fold_nets) >= 3
        and sum(value > 0 for value in fold_nets) >= 2
        and subjects["positive_subjects"] >= subjects["negative_subjects"]
    )


def evaluate_specialists(
    configs: list[dict[str, Any]],
    folds: list[int],
    smoke: bool,
    representations: dict[str, ModalityFeatures],
    sample_ids: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    a_probability: np.ndarray,
    source_probability: dict[int, np.ndarray],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]],
]:
    fold_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    bank_predictions: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    components = 8 if smoke else PCA_COMPONENTS
    needed = sorted({name for config in configs for name in config["modalities"]})
    for outer_fold in folds:
        source = fold_ids != outer_fold
        held = ~source
        source_rows = np.flatnonzero(source).astype(np.int64)
        held_rows = np.flatnonzero(held).astype(np.int64)
        inner_users = sorted(set(users[source].tolist()))
        if smoke:
            inner_users = inner_users[:2]
        inner_prediction = {
            config["family_key"]: np.full(len(labels), -1, dtype=np.int64)
            for config in configs
        }
        inner_covered = np.zeros(len(labels), dtype=bool)
        for inner_user in inner_users:
            train_all = np.flatnonzero(source & (users != inner_user)).astype(np.int64)
            active_rows = source_rows
            projected: dict[str, np.ndarray] = {}
            for name in needed:
                projection = Projection.fit(representations[name].aligned, train_all, components)
                values = np.full((len(labels), projection.components), np.nan, dtype=np.float32)
                values[active_rows] = projection.transform(
                    representations[name].aligned[active_rows]
                )
                projected[name] = values
            for config in configs:
                train_rows = np.flatnonzero(
                    source
                    & (users != inner_user)
                    & family_mask(labels, config["classes"])
                ).astype(np.int64)
                validation_rows = np.flatnonzero(
                    source
                    & (users == inner_user)
                    & family_mask(labels, config["classes"])
                ).astype(np.int64)
                if not len(validation_rows):
                    continue
                pair_values = concatenate(projected, config["modalities"])
                classifier = fit_specialist(
                    pair_values, labels, train_rows, config["classes"]
                )
                inner_prediction[config["family_key"]][validation_rows] = classifier.predict(
                    pair_values[validation_rows]
                )
            inner_covered |= source & (users == inner_user)

        projections = {
            name: Projection.fit(representations[name].aligned, source_rows, components)
            for name in needed
        }
        source_projected = {
            name: projections[name].transform(representations[name].aligned[source_rows])
            for name in needed
        }
        source_position = {int(row): index for index, row in enumerate(source_rows.tolist())}
        held_position = {int(row): index for index, row in enumerate(held_rows.tolist())}
        shuffle_map = cyclic_shuffle_source(sample_ids, users, held)
        shuffle_rows = shuffle_map[held_rows]
        held_a_prediction = a_probability[held_rows].argmax(axis=1)
        source_a_prediction = source_probability[outer_fold].argmax(axis=1)
        for config in configs:
            family = config["family_key"]
            classes = config["classes"]
            source_eval_rows = np.flatnonzero(
                inner_covered & family_mask(labels, classes)
            ).astype(np.int64)
            if np.any(inner_prediction[family][source_eval_rows] < 0):
                raise RuntimeError(f"P105 inner coverage failed: {outer_fold}/{family}")
            source_metrics = metric_bundle(
                labels[source_eval_rows],
                inner_prediction[family][source_eval_rows],
                users[source_eval_rows],
                classes,
            )
            source_a_metrics = metric_bundle(
                labels[source_eval_rows],
                source_a_prediction[source_eval_rows],
                users[source_eval_rows],
                classes,
            )
            source_intervention = intervention_metrics(
                labels[source_eval_rows],
                source_a_prediction[source_eval_rows],
                inner_prediction[family][source_eval_rows],
            )
            family_source_rows = np.flatnonzero(
                source & family_mask(labels, classes)
            ).astype(np.int64)
            family_source_positions = np.asarray(
                [source_position[int(row)] for row in family_source_rows], dtype=np.int64
            )
            source_values = concatenate(source_projected, config["modalities"])
            classifier = fit_specialist(
                source_values,
                labels[source_rows],
                family_source_positions,
                classes,
            )
            variants = canonical_variants(
                config,
                representations,
                projections,
                held_rows,
                shuffle_rows,
            )
            all_predictions = {
                variant: classifier.predict(values) for variant, values in variants.items()
            }
            for variant in ("aligned", "shuffle", "zero"):
                bank_predictions[(outer_fold, family, variant)] = (
                    held_rows,
                    all_predictions[variant],
                )
            family_held_rows = np.flatnonzero(
                held & family_mask(labels, classes)
            ).astype(np.int64)
            local_positions = np.asarray(
                [held_position[int(row)] for row in family_held_rows], dtype=np.int64
            )
            a_family_prediction = held_a_prediction[local_positions]
            variant_results = {}
            for variant, predictions in all_predictions.items():
                selected_prediction = predictions[local_positions]
                variant_results[variant] = {
                    "metrics": metric_bundle(
                        labels[family_held_rows],
                        selected_prediction,
                        users[family_held_rows],
                        classes,
                    ),
                    "family_oracle": intervention_metrics(
                        labels[family_held_rows],
                        a_family_prediction,
                        selected_prediction,
                    ),
                }
            for variant in ("aligned", "shuffle", "zero"):
                predictions = all_predictions[variant]
                for position, row in enumerate(held_rows.tolist()):
                    csv_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "family_key": family,
                            "classes": "|".join(map(str, classes)),
                            "modalities": "+".join(config["modalities"]),
                            "split": "outer_held_all",
                            "variant": variant,
                            "row_index": row,
                            "sample_id": sample_ids[row],
                            "subject": users[row],
                            "label": int(labels[row]),
                            "a_prediction": int(held_a_prediction[position]),
                            "specialist_prediction": int(predictions[position]),
                        }
                    )
            fold_results.append(
                {
                    "outer_fold": outer_fold,
                    "held_users": list(FOLD_USERS[outer_fold]),
                    "family_key": family,
                    "classes": classes,
                    "modalities": config["modalities"],
                    "source_family_rows": len(family_source_rows),
                    "source_inner_oof_rows": len(source_eval_rows),
                    "source_a": source_a_metrics,
                    "source_inner_specialist": source_metrics,
                    "source_inner_family_oracle": source_intervention,
                    "source_authorized": source_intervention["net"] > 0,
                    "held_family_rows": len(family_held_rows),
                    "a_family": metric_bundle(
                        labels[family_held_rows],
                        a_family_prediction,
                        users[family_held_rows],
                        classes,
                    ),
                    "variants": variant_results,
                    "source_vs_held_balanced_accuracy_gap": float(
                        source_metrics["balanced_accuracy"]
                        - variant_results["aligned"]["metrics"]["balanced_accuracy"]
                    ),
                }
            )
        print(
            f"P105 fold={outer_fold} specialists={len(configs)} modalities={','.join(needed)}",
            flush=True,
        )
    return fold_results, csv_rows, bank_predictions


def aggregate_specialists(
    configs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    csv_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped_results[result["family_key"]].append(result)
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in csv_rows:
        config = next(value for value in configs if value["family_key"] == row["family_key"])
        if int(row["label"]) in config["classes"]:
            grouped_rows[(row["family_key"], row["variant"])].append(row)
    output = []
    for config in configs:
        family = config["family_key"]
        values = sorted(grouped_results[family], key=lambda value: value["outer_fold"])
        rows = grouped_rows[(family, "aligned")]
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        users = np.asarray([row["subject"] for row in rows])
        a_prediction = np.asarray([row["a_prediction"] for row in rows], dtype=np.int64)
        predictions = {
            variant: np.asarray(
                [row["specialist_prediction"] for row in grouped_rows[(family, variant)]],
                dtype=np.int64,
            )
            for variant in ("aligned", "shuffle", "zero")
        }
        aligned_metrics = metric_bundle(
            labels, predictions["aligned"], users, config["classes"]
        )
        family_oracle = intervention_metrics(labels, a_prediction, predictions["aligned"])
        aggregate = {
            "family_key": family,
            "classes": config["classes"],
            "modalities": config["modalities"],
            "held_family_rows": len(rows),
            "a_family": metric_bundle(labels, a_prediction, users, config["classes"]),
            "specialist": aligned_metrics,
            "rescue": family_oracle["rescue"],
            "harm": family_oracle["harm"],
            "net": family_oracle["net"],
            "a_error_family_rows": int(np.sum(a_prediction != labels)),
            "a_error_rescued": family_oracle["rescue"],
            "a_correct_harmed": family_oracle["harm"],
            "aligned_correct": aligned_metrics["correct"],
            "shuffle_correct": int(np.sum(predictions["shuffle"] == labels)),
            "zero_correct": int(np.sum(predictions["zero"] == labels)),
            "aligned_minus_shuffle_correct": int(
                np.sum(predictions["aligned"] == labels)
                - np.sum(predictions["shuffle"] == labels)
            ),
            "aligned_minus_zero_correct": int(
                np.sum(predictions["aligned"] == labels)
                - np.sum(predictions["zero"] == labels)
            ),
            "per_fold_net": {
                str(value["outer_fold"]): value["variants"]["aligned"]["family_oracle"]["net"]
                for value in values
            },
            "per_fold": values,
            "per_subject": subject_family_summary(
                labels, a_prediction, predictions["aligned"], users
            ),
            "source_authorized_folds": [
                value["outer_fold"] for value in values if value["source_authorized"]
            ],
            "mean_source_vs_held_balanced_accuracy_gap": float(
                np.mean([value["source_vs_held_balanced_accuracy_gap"] for value in values])
            ),
        }
        aggregate["stable"] = stable_specialist(aggregate)
        output.append(aggregate)
    return output


def evaluate_bank(
    configs: list[dict[str, Any]],
    folds: list[int],
    labels: np.ndarray,
    users: np.ndarray,
    fold_ids: np.ndarray,
    a_probability: np.ndarray,
    results: list[dict[str, Any]],
    stored: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated = np.isin(fold_ids, np.asarray(folds, dtype=np.int64))
    eval_rows = np.flatnonzero(evaluated).astype(np.int64)
    local_position = {int(row): index for index, row in enumerate(eval_rows.tolist())}
    eval_probability = a_probability[eval_rows]
    eval_labels = labels[eval_rows]
    eval_users = users[eval_rows]
    eval_folds = fold_ids[eval_rows]
    a_prediction = eval_probability.argmax(axis=1)
    prediction_by_variant: dict[str, dict[str, np.ndarray]] = {
        variant: {
            config["family_key"]: np.full(len(eval_rows), -1, dtype=np.int64)
            for config in configs
        }
        for variant in ("aligned", "shuffle", "zero")
    }
    for outer_fold in folds:
        for config in configs:
            family = config["family_key"]
            for variant in prediction_by_variant:
                rows, predictions = stored[(outer_fold, family, variant)]
                positions = np.asarray([local_position[int(row)] for row in rows], dtype=np.int64)
                prediction_by_variant[variant][family][positions] = predictions
    for variants in prediction_by_variant.values():
        if any(np.any(values < 0) for values in variants.values()):
            raise RuntimeError("P105 bank prediction coverage failed")
    source_enabled = {
        fold: {
            result["family_key"]
            for result in results
            if result["outer_fold"] == fold and result["source_authorized"]
        }
        for fold in folds
    }
    locked_routes = route_assignments(
        eval_probability, configs, {value["family_key"] for value in configs}
    )
    source_routes = np.full(len(eval_rows), "", dtype=object)
    for fold in folds:
        selected = eval_folds == fold
        source_routes[selected] = route_assignments(
            eval_probability[selected], configs, source_enabled[fold]
        )
    oracle_routes = exact_edge_oracle_assignments(eval_labels, a_prediction, configs)
    systems: dict[str, Any] = {}
    route_sets = {
        "locked_top3": locked_routes,
        "source_authorized_top3": source_routes,
        "exact_edge_oracle": oracle_routes,
    }
    routing_rows = []
    for name, routes in route_sets.items():
        aligned_system = apply_routes(
            a_prediction, routes, prediction_by_variant["aligned"]
        )
        record: dict[str, Any] = {
            "aligned": system_metrics(
                eval_labels, a_prediction, aligned_system, eval_users, eval_folds
            ),
            "routes": route_details(
                routes, eval_labels, a_prediction, aligned_system, configs
            ),
        }
        if name != "exact_edge_oracle":
            for variant in ("shuffle", "zero"):
                system = apply_routes(
                    a_prediction, routes, prediction_by_variant[variant]
                )
                record[variant] = system_metrics(
                    eval_labels, a_prediction, system, eval_users, eval_folds
                )
        else:
            record["eligible_a_errors"] = int(np.sum(routes != ""))
            record["resolved"] = record["aligned"]["rescue"]
            record["unresolved"] = int(
                record["eligible_a_errors"] - record["resolved"]
            )
        systems[name] = record
    locked_system = apply_routes(
        a_prediction, locked_routes, prediction_by_variant["aligned"]
    )
    source_system = apply_routes(
        a_prediction, source_routes, prediction_by_variant["aligned"]
    )
    oracle_system = apply_routes(
        a_prediction, oracle_routes, prediction_by_variant["aligned"]
    )
    for position, row in enumerate(eval_rows.tolist()):
        routing_rows.append(
            {
                "row_index": row,
                "sample_id": "",
                "subject": users[row],
                "outer_fold": int(fold_ids[row]),
                "label": int(labels[row]),
                "a_prediction": int(a_prediction[position]),
                "locked_route": str(locked_routes[position]),
                "locked_prediction": int(locked_system[position]),
                "source_authorized_route": str(source_routes[position]),
                "source_authorized_prediction": int(source_system[position]),
                "exact_edge_oracle_route": str(oracle_routes[position]),
                "exact_edge_oracle_prediction": int(oracle_system[position]),
            }
        )
    return {
        "source_enabled_by_fold": {
            str(fold): sorted(values) for fold, values in source_enabled.items()
        },
        "systems": systems,
    }, routing_rows


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)
    configs = select_specialists(args.families)
    if args.smoke:
        folds = folds[:1]
        configs = configs[:1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = load_npz(args.session.resolve())
    session_summary = json.loads(args.session_summary.resolve().read_text(encoding="utf-8"))
    data = load_p100a_data()
    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    if not np.array_equal(sample_ids, data.sample_ids.astype(str)):
        raise RuntimeError("P100/P105 row order differs")
    if not np.array_equal(users, data.users.astype(str)) or not np.array_equal(labels, data.labels):
        raise RuntimeError("P100/P105 metadata differs")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P105")
    if str(np.asarray(session["selected_system"]).item()) != "VS_session":
        raise RuntimeError("P105 baseline is not VS_session")
    needed = sorted({name for config in configs for name in config["modalities"]})
    representations = {
        name: load_modality_features(name, data=data) for name in needed
    }
    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    source_probability = {}
    for outer_fold in folds:
        probability, _ = source_crossfit_session_probability(
            outer_fold,
            args.nested_root.resolve(),
            sample_ids,
            users,
            fold_ids,
            labels,
            metadata,
            decoder_config(session_summary, outer_fold),
        )
        source_probability[outer_fold] = probability
    fold_results, specialist_rows, stored = evaluate_specialists(
        configs,
        folds,
        args.smoke,
        representations,
        sample_ids,
        users,
        labels,
        fold_ids,
        a_probability,
        source_probability,
    )
    aggregates = aggregate_specialists(configs, fold_results, specialist_rows)
    bank, routing_rows = evaluate_bank(
        configs,
        folds,
        labels,
        users,
        fold_ids,
        a_probability,
        fold_results,
        stored,
    )
    sample_lookup = {
        int(index): sample_id for index, sample_id in enumerate(sample_ids.tolist())
    }
    for row in routing_rows:
        row["sample_id"] = sample_lookup[row["row_index"]]
    primary_details = {
        value["family_key"]: value
        for value in bank["systems"]["source_authorized_top3"]["routes"]
    }
    for aggregate in aggregates:
        route_net = primary_details[aggregate["family_key"]]["net"]
        if aggregate["stable"] and route_net > 0:
            verdict = "VALIDATED"
        elif (
            aggregate["net"] > 0
            and aggregate["aligned_minus_shuffle_correct"] > 0
            and aggregate["aligned_minus_zero_correct"] > 0
        ):
            verdict = "CAPABILITY_ONLY"
        else:
            verdict = "REJECTED"
        aggregate["primary_router_net"] = route_net
        aggregate["verdict"] = verdict
    write_csv(output / "specialist_predictions.csv", specialist_rows)
    write_csv(output / "routing_predictions.csv", routing_rows)
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "protocol": "P105 locked-list source-safe specialist validation; no unified B/learned router/Student",
        "post_selection_limitation": "P104 shortlist and P105 validation share H1 OOF subjects; model fitting and outer authorization are source-safe, but this is not independent-data replication.",
        "data": {
            "rows": int(np.sum(np.isin(fold_ids, folds))),
            "folds_run": folds,
            "specialists_run": [value["family_key"] for value in configs],
            "modalities_loaded": needed,
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "recipe": {
            "projection": "per-modality outer/inner-source-only StandardScaler + randomized PCA",
            "pca_components": 8 if args.smoke else PCA_COMPONENTS,
            "classifier": "balanced LogisticRegression C=1.0 lbfgs max_iter=2000",
            "training_population": "all source samples whose true label belongs to the fixed family",
            "source_authorization": "source-inner family-oracle net > 0",
            "router": "A top1 in family and alternate member in A top3; overlap by alternate probability",
        },
        "modality_audit": {name: value.audit for name, value in representations.items()},
        "fold_results": fold_results,
        "specialist_capability_table": aggregates,
        "bank": bank,
        "h3_rows_selected": 0,
        "h3_users_loaded": [],
        "student_started": False,
        "unified_b_teacher_trained": False,
        "learned_router_trained": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "specialists": [
                    {
                        "family_key": value["family_key"],
                        "modalities": value["modalities"],
                        "family_accuracy": value["specialist"]["accuracy"],
                        "balanced_accuracy": value["specialist"]["balanced_accuracy"],
                        "macro_f1": value["specialist"]["macro_f1"],
                        "rescue": value["rescue"],
                        "harm": value["harm"],
                        "net": value["net"],
                        "stable": value["stable"],
                        "source_authorized_folds": value["source_authorized_folds"],
                        "primary_router_net": value["primary_router_net"],
                        "verdict": value["verdict"],
                    }
                    for value in aggregates
                ],
                "bank": {
                    name: {
                        "a_correct": value["aligned"]["a_correct"],
                        "system_correct": value["aligned"]["system_correct"],
                        "rescue": value["aligned"]["rescue"],
                        "harm": value["aligned"]["harm"],
                        "net": value["aligned"]["net"],
                        "shuffle_net": value.get("shuffle", {}).get("net"),
                        "zero_net": value.get("zero", {}).get("net"),
                    }
                    for name, value in bank["systems"].items()
                },
                "h3_rows_selected": 0,
                "h3_users_loaded": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
