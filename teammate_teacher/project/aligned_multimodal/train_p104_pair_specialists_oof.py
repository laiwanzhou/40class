"""Run the source-selected, restricted P104 two-modality family probes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    trigger_metrics,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_PLAN = HERE / "runs/p104_single_modality_audit_v1/pair_plan.json"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p104_pair_specialists_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--session-summary", type=Path, default=DEFAULT_SESSION_SUMMARY)
    parser.add_argument("--pair-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_folds(value: str) -> list[int]:
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds or not set(folds) <= set(range(4)):
        raise ValueError(f"invalid P104 pair folds: {value}")
    return folds


def decoder_config(summary: dict[str, Any], fold: int) -> DecoderConfig:
    selected = summary["folds"][fold]["selected"]
    return DecoderConfig(
        gap_seconds=float(selected["gap_seconds"]),
        transition_weight=float(selected["transition_weight"]),
        trigram_backoff=float(selected["trigram_backoff"]),
        beam_width=int(selected["beam_width"]),
    )


def selected_plans(
    archive: dict[str, Any], folds: list[int], smoke: bool
) -> list[dict[str, Any]]:
    plans = [
        value for value in archive["plans"]
        if int(value["outer_fold"]) in folds and value["selected_pair"] is not None
    ]
    plans.sort(key=lambda value: (int(value["outer_fold"]), value["family_key"]))
    return plans[:1] if smoke else plans


def pair_key(modalities: list[str]) -> str:
    return "+".join(modalities)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty P104 pair CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concatenate(
    projected: dict[str, np.ndarray], modalities: list[str], rows: np.ndarray
) -> np.ndarray:
    return np.concatenate([projected[name][rows] for name in modalities], axis=1)


def held_variants(
    representations: dict[str, ModalityFeatures],
    projections: dict[str, Projection],
    modalities: list[str],
    held_rows: np.ndarray,
    shuffle_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    aligned = {
        name: projections[name].transform(representations[name].aligned[held_rows])
        for name in modalities
    }
    shuffled = {
        name: projections[name].transform(representations[name].aligned[shuffle_rows])
        for name in modalities
    }
    zero = {name: projections[name].zero(len(held_rows)) for name in modalities}
    first, second = modalities
    variants = {
        "aligned": np.concatenate([aligned[first], aligned[second]], axis=1),
        "shuffle_both": np.concatenate([shuffled[first], shuffled[second]], axis=1),
        "shuffle_first": np.concatenate([shuffled[first], aligned[second]], axis=1),
        "shuffle_second": np.concatenate([aligned[first], shuffled[second]], axis=1),
        "zero_both": np.concatenate([zero[first], zero[second]], axis=1),
        "zero_first": np.concatenate([zero[first], aligned[second]], axis=1),
        "zero_second": np.concatenate([aligned[first], zero[second]], axis=1),
    }
    for name in modalities:
        other = second if name == first else first
        for intervention, raw in representations[name].interventions.items():
            changed = projections[name].transform(raw[held_rows])
            pieces = (
                [changed, aligned[other]] if name == first else [aligned[other], changed]
            )
            variants[f"{name.lower()}_{intervention}"] = np.concatenate(pieces, axis=1)
    return variants


def aggregate_pairs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["family_key"], result["pair_key"])].append(result)
    output: list[dict[str, Any]] = []
    for (family, key), values in sorted(grouped.items()):
        held = sum(value["held_sample_count"] for value in values)
        a_correct = sum(value["a_family"]["correct"] for value in values)
        variant_names = set.intersection(
            *(set(value["variants"]) for value in values)
        )
        variant_correct = {
            name: sum(value["variants"][name]["metrics"]["correct"] for value in values)
            for name in sorted(variant_names)
        }
        aligned = variant_correct["aligned"]
        rescue = sum(value["variants"]["aligned"]["intervention"]["rescue"] for value in values)
        harm = sum(value["variants"]["aligned"]["intervention"]["harm"] for value in values)
        output.append(
            {
                "family_key": family,
                "classes": values[0]["classes"],
                "pair_key": key,
                "modalities": values[0]["modalities"],
                "selected_folds": sorted(value["outer_fold"] for value in values),
                "selected_fold_count": len(values),
                "formal_verdict_eligible": len(values) >= 2,
                "held_rows": held,
                "a_correct": a_correct,
                "aligned_correct": aligned,
                "aligned_minus_a_correct": aligned - a_correct,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "variant_correct": variant_correct,
                "aligned_minus_shuffle_both_correct": aligned - variant_correct["shuffle_both"],
                "aligned_minus_zero_both_correct": aligned - variant_correct["zero_both"],
                "deployable_trigger_net": sum(value["deployable_trigger"]["net"] for value in values),
                "per_fold_net": {
                    str(value["outer_fold"]): value["variants"]["aligned"]["intervention"]["net"]
                    for value in values
                },
            }
        )
    return output


def evaluate(
    plans: list[dict[str, Any]],
    representations: dict[str, ModalityFeatures],
    smoke: bool,
    sample_ids: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    a_probability: np.ndarray,
    source_probability: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    components = 8 if smoke else PCA_COMPONENTS
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for plan in plans:
        by_fold[int(plan["outer_fold"])].append(plan)
    for outer_fold, fold_plans in sorted(by_fold.items()):
        source = fold_ids != outer_fold
        held = ~source
        held_rows = np.flatnonzero(held).astype(np.int64)
        source_users = sorted(set(users[source].tolist()))
        inner_users = source_users[:2] if smoke else source_users
        needed = sorted(
            {
                name
                for plan in fold_plans
                for name in plan["selected_pair"]["modalities"]
            }
        )
        inner_predictions = {
            plan["family_key"]: np.full(len(labels), -1, dtype=np.int64)
            for plan in fold_plans
        }
        inner_covered = np.zeros(len(labels), dtype=bool)
        for inner_user in inner_users:
            train_all = np.flatnonzero(source & (users != inner_user)).astype(np.int64)
            active_rows = np.flatnonzero(source).astype(np.int64)
            projected: dict[str, np.ndarray] = {}
            for name in needed:
                projection = Projection.fit(representations[name].aligned, train_all, components)
                values = np.full((len(labels), projection.components), np.nan, dtype=np.float32)
                values[active_rows] = projection.transform(
                    representations[name].aligned[active_rows]
                )
                projected[name] = values
            for plan in fold_plans:
                classes = list(map(int, plan["classes"]))
                modalities = list(plan["selected_pair"]["modalities"])
                train_rows = np.flatnonzero(
                    source & (users != inner_user) & family_mask(labels, classes)
                ).astype(np.int64)
                validation_rows = np.flatnonzero(
                    source & (users == inner_user) & family_mask(labels, classes)
                ).astype(np.int64)
                if not len(validation_rows):
                    continue
                pair_values = np.concatenate(
                    [projected[name] for name in modalities], axis=1
                )
                classifier = fit_specialist(pair_values, labels, train_rows, classes)
                inner_predictions[plan["family_key"]][validation_rows] = classifier.predict(
                    pair_values[validation_rows]
                )
            inner_covered |= source & (users == inner_user)

        outer_source_rows = np.flatnonzero(source).astype(np.int64)
        projections = {
            name: Projection.fit(
                representations[name].aligned, outer_source_rows, components
            )
            for name in needed
        }
        source_projected = {
            name: projections[name].transform(
                representations[name].aligned[outer_source_rows]
            )
            for name in needed
        }
        source_position = {
            int(row): index for index, row in enumerate(outer_source_rows.tolist())
        }
        shuffle_map = cyclic_shuffle_source(sample_ids, users, held)
        held_shuffle_rows = shuffle_map[held_rows]
        held_position = {int(row): index for index, row in enumerate(held_rows.tolist())}
        held_a_probability = a_probability[held_rows]
        held_a_prediction = held_a_probability.argmax(axis=1)
        source_a_prediction = source_probability[outer_fold].argmax(axis=1)
        for plan in fold_plans:
            family = plan["family_key"]
            classes = list(map(int, plan["classes"]))
            modalities = list(plan["selected_pair"]["modalities"])
            source_eval_rows = np.flatnonzero(
                inner_covered & family_mask(labels, classes)
            ).astype(np.int64)
            if np.any(inner_predictions[family][source_eval_rows] < 0):
                raise RuntimeError(f"P104 pair inner coverage failed: {outer_fold}/{family}")
            source_metrics = metric_bundle(
                labels[source_eval_rows],
                inner_predictions[family][source_eval_rows],
                users[source_eval_rows],
                classes,
            )
            source_a_metrics = metric_bundle(
                labels[source_eval_rows],
                source_a_prediction[source_eval_rows],
                users[source_eval_rows],
                classes,
            )
            for row in source_eval_rows.tolist():
                prediction_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "family_key": family,
                        "classes": "|".join(map(str, classes)),
                        "pair_key": pair_key(modalities),
                        "split": "source_inner_oof",
                        "variant": "aligned",
                        "row_index": row,
                        "sample_id": sample_ids[row],
                        "subject": users[row],
                        "label": int(labels[row]),
                        "a_prediction": int(source_a_prediction[row]),
                        "specialist_prediction": int(inner_predictions[family][row]),
                    }
                )
            family_source_rows = np.flatnonzero(
                source & family_mask(labels, classes)
            ).astype(np.int64)
            family_source_positions = np.asarray(
                [source_position[int(row)] for row in family_source_rows], dtype=np.int64
            )
            source_pair = np.concatenate(
                [source_projected[name] for name in modalities], axis=1
            )
            classifier = fit_specialist(
                source_pair, labels[outer_source_rows], family_source_positions, classes
            )
            variants = held_variants(
                representations,
                projections,
                modalities,
                held_rows,
                held_shuffle_rows,
            )
            all_predictions = {
                name: classifier.predict(values) for name, values in variants.items()
            }
            family_held_rows = np.flatnonzero(
                held & family_mask(labels, classes)
            ).astype(np.int64)
            local_positions = np.asarray(
                [held_position[int(row)] for row in family_held_rows], dtype=np.int64
            )
            a_family_prediction = held_a_prediction[local_positions]
            variant_results: dict[str, Any] = {}
            for variant, predictions in all_predictions.items():
                selected = predictions[local_positions]
                variant_results[variant] = {
                    "metrics": metric_bundle(
                        labels[family_held_rows], selected, users[family_held_rows], classes
                    ),
                    "intervention": intervention_metrics(
                        labels[family_held_rows], a_family_prediction, selected
                    ),
                }
                for row, prediction, a_prediction in zip(
                    family_held_rows.tolist(), selected.tolist(), a_family_prediction.tolist()
                ):
                    prediction_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "family_key": family,
                            "classes": "|".join(map(str, classes)),
                            "pair_key": pair_key(modalities),
                            "split": "outer_held_family",
                            "variant": variant,
                            "row_index": row,
                            "sample_id": sample_ids[row],
                            "subject": users[row],
                            "label": int(labels[row]),
                            "a_prediction": int(a_prediction),
                            "specialist_prediction": int(prediction),
                        }
                    )
            result = {
                "outer_fold": outer_fold,
                "held_users": list(FOLD_USERS[outer_fold]),
                "family_key": family,
                "classes": classes,
                "pair_name": plan["selected_pair"]["name"],
                "pair_key": pair_key(modalities),
                "modalities": modalities,
                "source_selection": plan["selected_pair"],
                "source_sample_count": len(family_source_rows),
                "source_inner_oof_rows": len(source_eval_rows),
                "held_sample_count": len(family_held_rows),
                "source_a": source_a_metrics,
                "source_inner_specialist": source_metrics,
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
                "deployable_trigger": trigger_metrics(
                    labels[held_rows],
                    held_a_probability,
                    all_predictions["aligned"],
                    classes,
                ),
                "projection": {
                    name: {
                        "components": projections[name].components,
                        "explained_variance": float(
                            np.sum(projections[name].pca.explained_variance_ratio_)
                        ),
                    }
                    for name in modalities
                },
            }
            results.append(result)
        print(
            f"P104 pairs fold={outer_fold} plans={len(fold_plans)} "
            f"modalities={','.join(needed)}",
            flush=True,
        )
    return results, prediction_rows


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_archive = json.loads(args.pair_plan.resolve().read_text(encoding="utf-8"))
    if plan_archive["h3_rows_selected"] != 0 or plan_archive["h3_users_loaded"]:
        raise RuntimeError("H3 reached P104 pair plan")
    plans = selected_plans(plan_archive, folds, args.smoke)
    if not plans:
        raise RuntimeError("P104 pair plan selected no work")
    session = load_npz(args.session.resolve())
    session_summary = json.loads(args.session_summary.resolve().read_text(encoding="utf-8"))
    data = load_p100a_data()
    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    if not np.array_equal(sample_ids, data.sample_ids.astype(str)):
        raise RuntimeError("P100/P104 pair row order differs")
    if not np.array_equal(users, data.users.astype(str)) or not np.array_equal(labels, data.labels):
        raise RuntimeError("P100/P104 pair metadata differs")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P104 pair specialists")
    if str(np.asarray(session["selected_system"]).item()) != "VS_session":
        raise RuntimeError("P104 pair baseline is not VS_session")
    needed = sorted(
        {name for plan in plans for name in plan["selected_pair"]["modalities"]}
    )
    representations = {
        name: load_modality_features(name, data=data) for name in needed
    }
    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    source_probabilities: dict[int, np.ndarray] = {}
    for outer_fold in sorted({int(plan["outer_fold"]) for plan in plans}):
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
        source_probabilities[outer_fold] = probability
    results, prediction_rows = evaluate(
        plans,
        representations,
        args.smoke,
        sample_ids,
        users,
        labels,
        fold_ids,
        a_probability,
        source_probabilities,
    )
    write_csv(output / "pair_predictions.csv", prediction_rows)
    aggregates = aggregate_pairs(results)
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "protocol": "P104 source-selected restricted pair specialists; no final B/router/Student",
        "data": {
            "rows": len(labels),
            "subjects": sorted(set(users.tolist())),
            "folds_run": sorted({int(plan["outer_fold"]) for plan in plans}),
            "pair_evaluations": len(plans),
            "modalities_loaded": needed,
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "recipe": {
            "projection": "separate source-train-only 64D PCA per modality, then concatenate",
            "pca_components_each": 8 if args.smoke else PCA_COMPONENTS,
            "classifier": "balanced LogisticRegression C=1.0 lbfgs max_iter=2000",
            "counterfactuals": "aligned, shuffle both/each, zero both/each, representation-specific reverse/swap",
        },
        "modality_audit": {name: value.audit for name, value in representations.items()},
        "fold_results": results,
        "aggregates": aggregates,
        "formal_exact_pair_aggregates": [
            value for value in aggregates if value["formal_verdict_eligible"]
        ],
        "student_started": False,
        "final_b_teacher_trained": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "fold_results": len(results),
                "aggregates": aggregates,
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
