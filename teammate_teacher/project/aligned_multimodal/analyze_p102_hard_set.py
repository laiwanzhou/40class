"""Rebuild the P102 hard set from the selected VS+Session OOF system.

Candidate graphs and candidate-recipe selection are rebuilt inside every outer
fold.  Only source-cross-fitted nested coarse-VS emissions and source labels are
used.  The outer-held true label is never passed to candidate construction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audit_p102_session_closure import (
    classification_metrics,
    comparison,
    load_npz,
    log_softmax,
    topk_correct,
    true_rank,
)
from audit_p87_sequence_decoder import DecoderConfig, align_metadata
from build_p87s_structured_targets import backed_off_structured_probability, build_targets
from p100a_global_teacher_data import FOLD_USERS, H3_USERS


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p102_hard_set_v1"


@dataclass(frozen=True)
class CandidateRecipe:
    base_k: int
    max_size: int
    neighbors_per_anchor: int
    min_neighbor_subjects: int = 2
    min_neighbor_count: int = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-oof", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--session-summary", type=Path, default=DEFAULT_SESSION_SUMMARY)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def entropy(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    return -np.sum(values * np.log(np.maximum(values, 1e-12)), axis=1)


def top_margin(probability: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(probability, dtype=np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def source_crossfit_session_probability(
    outer_fold: int,
    nested_root: Path,
    sample_ids: np.ndarray,
    users: np.ndarray,
    folds: np.ndarray,
    labels: np.ndarray,
    metadata: Any,
    config: DecoderConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    nested_path = nested_root / f"outer{outer_fold}" / "nested_predictions.npz"
    nested = load_npz(nested_path)
    if not np.array_equal(nested["sample_ids"].astype(str), sample_ids):
        raise RuntimeError("nested/session sample order differs")
    covered = np.asarray(nested["covered"], dtype=bool)
    source = folds != outer_fold
    if not np.array_equal(covered, source):
        raise RuntimeError("nested coverage is not the outer-source mask")
    raw_log_probability = np.full((len(labels), 40), -np.log(40.0), dtype=np.float64)
    raw_log_probability[source] = log_softmax(np.asarray(nested["logits"], dtype=np.float64)[source])
    raw_probability = np.exp(raw_log_probability)
    result = np.full((len(labels), 40), np.nan, dtype=np.float64)
    audit: dict[str, Any] = {"users": {}, "source_rows": int(source.sum())}
    for user in sorted(set(users[source].tolist())):
        held = np.flatnonzero(source & (users == user)).astype(np.int64)
        train = np.flatnonzero(source & (users != user)).astype(np.int64)
        masked = labels.copy()
        masked[~np.isin(np.arange(len(labels)), train)] = -10_000
        target = build_targets(
            raw_log_probability,
            masked,
            train,
            held,
            metadata,
            config,
            posterior_temperature=1.0,
        )
        structured = np.asarray(target["structured_probability"], dtype=np.float64)
        probability, weight = backed_off_structured_probability(
            raw_probability, structured, beam_width=config.beam_width
        )
        result[held] = probability[held]
        audit["users"][user] = {
            "held_rows": int(len(held)),
            "transition_fit_users": sorted(set(users[train].tolist())),
            "held_fit_overlap": sorted(set(users[held].tolist()) & set(users[train].tolist())),
            "held_sessions": int(target["holdout_session_count"]),
            "mean_structured_weight": float(weight[held].mean()),
        }
    if not np.isfinite(result[source]).all() or np.isfinite(result[~source]).any():
        raise RuntimeError("source cross-fit probability coverage failed")
    return result, audit


def confusion_neighbors(
    labels: np.ndarray,
    predictions: np.ndarray,
    users: np.ndarray,
    indices: np.ndarray,
    recipe: CandidateRecipe,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    counts: Counter[tuple[int, int]] = Counter()
    subjects: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in map(int, indices):
        true = int(labels[row])
        predicted = int(predictions[row])
        if true == predicted:
            continue
        edge = (predicted, true)
        counts[edge] += 1
        subjects[edge].add(str(users[row]))
    neighbors: dict[int, list[int]] = {class_id: [] for class_id in range(40)}
    records: list[dict[str, Any]] = []
    for predicted in range(40):
        values: list[tuple[int, int, int]] = []
        for (source_prediction, true), count in counts.items():
            if source_prediction != predicted:
                continue
            subject_count = len(subjects[(source_prediction, true)])
            if subject_count < recipe.min_neighbor_subjects or count < recipe.min_neighbor_count:
                continue
            values.append((subject_count, count, true))
        values.sort(key=lambda value: (-value[0], -value[1], value[2]))
        neighbors[predicted] = [value[2] for value in values]
        for subject_count, count, true in values:
            records.append(
                {
                    "predicted": predicted,
                    "neighbor_true": true,
                    "rows": count,
                    "subjects": subject_count,
                }
            )
    records.sort(key=lambda value: (-value["subjects"], -value["rows"], value["predicted"], value["neighbor_true"]))
    return neighbors, records


def build_candidate_ids(
    probability: np.ndarray,
    neighbors: dict[int, list[int]],
    recipe: CandidateRecipe,
) -> np.ndarray:
    order = np.argsort(np.asarray(probability), axis=1)[:, ::-1]
    output = np.full((len(order), recipe.max_size), -1, dtype=np.int64)
    for row in range(len(order)):
        selected: list[int] = []
        for class_id in order[row, : recipe.base_k]:
            value = int(class_id)
            if value not in selected:
                selected.append(value)
        for anchor in order[row, : recipe.base_k]:
            if len(selected) >= recipe.max_size:
                break
            added = 0
            for value in neighbors[int(anchor)]:
                if len(selected) >= recipe.max_size:
                    break
                if value in selected:
                    continue
                selected.append(int(value))
                added += 1
                if len(selected) >= recipe.max_size or added >= recipe.neighbors_per_anchor:
                    break
            if len(selected) >= recipe.max_size:
                break
        output[row, : len(selected)] = selected
    return output


def candidate_recall(candidate_ids: np.ndarray, labels: np.ndarray, selected: np.ndarray) -> tuple[int, int, float]:
    hit = np.any(candidate_ids == labels[:, None], axis=1)
    total = int(selected.sum())
    correct = int(np.sum(hit & selected))
    return correct, total, float(correct / max(total, 1))


def candidate_grid() -> list[CandidateRecipe]:
    recipes: list[CandidateRecipe] = []
    for base_k in (2, 3, 5):
        for max_size in (5, 6, 8):
            if max_size < base_k:
                continue
            for neighbors_per_anchor in (1, 2):
                recipes.append(CandidateRecipe(base_k, max_size, neighbors_per_anchor))
    return recipes


def select_candidate_recipe(
    labels: np.ndarray,
    users: np.ndarray,
    source: np.ndarray,
    probability: np.ndarray,
) -> tuple[CandidateRecipe, list[dict[str, Any]]]:
    prediction = probability.argmax(axis=1)
    source_errors = source & (prediction != labels)
    records: list[dict[str, Any]] = []
    for recipe in candidate_grid():
        candidates = np.full((len(labels), recipe.max_size), -1, dtype=np.int64)
        for user in sorted(set(users[source].tolist())):
            validation = source & (users == user)
            graph_fit = np.flatnonzero(source & (users != user)).astype(np.int64)
            neighbors, _ = confusion_neighbors(labels, prediction, users, graph_fit, recipe)
            candidates[validation] = build_candidate_ids(probability[validation], neighbors, recipe)
        hit = np.any(candidates == labels[:, None], axis=1)
        sizes = np.sum(candidates >= 0, axis=1)
        records.append(
            {
                **asdict(recipe),
                "source_error_rows": int(source_errors.sum()),
                "source_error_candidate_hits": int(np.sum(source_errors & hit)),
                "source_error_candidate_recall": float(np.mean(hit[source_errors])),
                "source_all_candidate_recall": float(np.mean(hit[source])),
                "source_mean_candidate_size": float(np.mean(sizes[source])),
            }
        )
    best = max(
        records,
        key=lambda value: (
            value["source_error_candidate_recall"],
            -value["source_mean_candidate_size"],
            -value["max_size"],
            -value["base_k"],
            -value["neighbors_per_anchor"],
        ),
    )
    recipe = CandidateRecipe(
        base_k=int(best["base_k"]),
        max_size=int(best["max_size"]),
        neighbors_per_anchor=int(best["neighbors_per_anchor"]),
        min_neighbor_subjects=int(best["min_neighbor_subjects"]),
        min_neighbor_count=int(best["min_neighbor_count"]),
    )
    return recipe, records


def top_confusions(labels: np.ndarray, prediction: np.ndarray, selected: np.ndarray, limit: int = 30) -> list[dict[str, int]]:
    counts = Counter(
        (int(labels[row]), int(prediction[row]))
        for row in np.flatnonzero(selected)
        if int(labels[row]) != int(prediction[row])
    )
    return [
        {"true": true, "predicted": predicted, "rows": count}
        for (true, predicted), count in counts.most_common(limit)
    ]


def grouped_error_summary(labels: np.ndarray, users: np.ndarray, error: np.ndarray) -> dict[str, Any]:
    by_subject = {
        user: {"rows": int(np.sum(users == user)), "errors": int(np.sum(error & (users == user))), "error_rate": float(np.mean(error[users == user]))}
        for user in sorted(set(users.tolist()))
    }
    by_class = {
        str(class_id): {"support": int(np.sum(labels == class_id)), "errors": int(np.sum(error & (labels == class_id))), "error_rate": float(np.mean(error[labels == class_id]))}
        for class_id in range(40)
    }
    concentration = sorted(
        ({"class_id": int(class_id), **values} for class_id, values in ((key, by_class[str(key)]) for key in range(40))),
        key=lambda value: (-value["errors"], -value["error_rate"], value["class_id"]),
    )
    return {"per_subject": by_subject, "per_class": by_class, "class_error_concentration": concentration}


def main() -> None:
    args = parse_args()
    session_path = args.session_oof.resolve()
    session_summary_path = args.session_summary.resolve()
    nested_root = args.nested_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = load_npz(session_path)
    summary = json.loads(session_summary_path.read_text(encoding="utf-8"))

    sample_ids = archive["sample_ids"].astype(str)
    users = archive["users"].astype(str)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    folds = np.asarray(archive["fold_ids"], dtype=np.int64)
    probability = np.asarray(archive["selected_probability"], dtype=np.float64)
    raw_probability = np.asarray(archive["vs_raw_probability"], dtype=np.float64)
    if str(np.asarray(archive["selected_system"]).item()) != "VS_session":
        raise RuntimeError("hard-set input is not the frozen VS_session baseline")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P102 hard-set analysis")
    metadata = align_metadata(args.metadata.resolve(), sample_ids)

    candidate_ids = np.full((len(labels), 8), -1, dtype=np.int64)
    direct_k = np.zeros(len(labels), dtype=np.int64)
    fold_reports: list[dict[str, Any]] = []
    for outer_fold in range(4):
        fold_summary = summary["folds"][outer_fold]
        selected = fold_summary["selected"]
        config = DecoderConfig(
            gap_seconds=float(selected["gap_seconds"]),
            transition_weight=float(selected["transition_weight"]),
            trigram_backoff=float(selected["trigram_backoff"]),
            beam_width=int(selected["beam_width"]),
        )
        source = folds != outer_fold
        held = folds == outer_fold
        source_probability, source_audit = source_crossfit_session_probability(
            outer_fold, nested_root, sample_ids, users, folds, labels, metadata, config
        )
        recipe, grid = select_candidate_recipe(labels, users, source, source_probability)
        source_prediction = source_probability.argmax(axis=1)
        neighbors, edge_records = confusion_neighbors(
            labels,
            source_prediction,
            users,
            np.flatnonzero(source).astype(np.int64),
            recipe,
        )
        held_candidates = build_candidate_ids(probability[held], neighbors, recipe)
        candidate_ids[held, : recipe.max_size] = held_candidates
        direct_k[held] = recipe.base_k
        held_error = held & (probability.argmax(axis=1) != labels)
        hit = np.any(candidate_ids == labels[:, None], axis=1)
        direct_hit = true_rank(probability, labels) <= direct_k
        fold_reports.append(
            {
                "fold": outer_fold,
                "held_users": list(FOLD_USERS[outer_fold]),
                "source_users": sorted(set(users[source].tolist())),
                "source_held_overlap": sorted(set(users[source].tolist()) & set(FOLD_USERS[outer_fold])),
                "source_session_crossfit": source_audit,
                "selected_recipe": asdict(recipe),
                "candidate_grid": grid,
                "eligible_confusion_edges": edge_records,
                "held": {
                    "rows": int(held.sum()),
                    "a_errors": int(held_error.sum()),
                    "direct_candidate_hits_on_errors": int(np.sum(held_error & direct_hit)),
                    "expanded_candidate_hits_on_errors": int(np.sum(held_error & hit)),
                    "expanded_candidate_recall_on_errors": float(np.mean(hit[held_error])),
                    "mean_candidate_size": float(np.mean(np.sum(candidate_ids[held] >= 0, axis=1))),
                },
            }
        )

    if np.any(direct_k == 0) or np.any(candidate_ids[:, 0] < 0):
        raise RuntimeError("candidate coverage is incomplete")
    prediction = probability.argmax(axis=1)
    error = prediction != labels
    ranks = true_rank(probability, labels)
    candidate_hit = np.any(candidate_ids == labels[:, None], axis=1)
    direct_hit = ranks <= direct_k
    category = np.full(len(labels), "correct", dtype="<U8")
    category[error & direct_hit] = "A"
    category[error & (~direct_hit) & candidate_hit] = "B"
    category[error & (~candidate_hit)] = "C"
    margin = top_margin(probability)
    uncertainty = entropy(probability)
    raw_prediction = raw_probability.argmax(axis=1)
    session_rescue = (raw_prediction != labels) & (~error)
    session_harm = (raw_prediction == labels) & error

    row_records: list[dict[str, Any]] = []
    for row in np.flatnonzero(error):
        ids = candidate_ids[row][candidate_ids[row] >= 0]
        row_records.append(
            {
                "sample_id": str(sample_ids[row]),
                "user": str(users[row]),
                "fold": int(folds[row]),
                "true": int(labels[row]),
                "prediction": int(prediction[row]),
                "true_rank": int(ranks[row]),
                "margin": float(margin[row]),
                "entropy": float(uncertainty[row]),
                "direct_k": int(direct_k[row]),
                "candidate_ids": list(map(int, ids.tolist())),
                "category": str(category[row]),
                "session_harm": bool(session_harm[row]),
            }
        )

    direct_coverage = {
        str(k): {
            "all_hits": int(np.sum(ranks <= k)),
            "all_recall": float(np.mean(ranks <= k)),
            "error_hits": int(np.sum(error & (ranks <= k))),
            "error_recall": float(np.mean(ranks[error] <= k)),
        }
        for k in (2, 3, 5, 10)
    }
    categories = {
        name: {
            "rows": int(np.sum(category == name)),
            "mean_true_rank": float(np.mean(ranks[category == name])) if np.any(category == name) else None,
            "mean_margin": float(np.mean(margin[category == name])) if np.any(category == name) else None,
            "mean_entropy": float(np.mean(uncertainty[category == name])) if np.any(category == name) else None,
        }
        for name in ("A", "B", "C")
    }
    selected_candidate_rows = error & candidate_hit
    result = {
        "status": "complete",
        "protocol": "P102 hard set from frozen VS+Session; outer-source cross-fitted confusion candidates; no oracle family or true-label insertion",
        "data": {"rows": int(len(labels)), "users": sorted(set(users.tolist())), "h3_rows_selected": 0, "h3_users_loaded": []},
        "a_system": {
            "metrics": classification_metrics(probability, labels, users),
            "errors": int(error.sum()),
            "accuracy_rows": int((~error).sum()),
            "margin": {"mean_all": float(margin.mean()), "mean_error": float(margin[error].mean())},
            "entropy": {"mean_all": float(uncertainty.mean()), "mean_error": float(uncertainty[error].mean())},
            "direct_topk_coverage": direct_coverage,
            "top_confusions": top_confusions(labels, prediction, error),
            **grouped_error_summary(labels, users, error),
        },
        "session_effect": {
            **comparison(labels, users, raw_probability, probability),
            "rescue_rows_in_final_error": int(np.sum(session_rescue & error)),
            "harm_rows_in_final_error": int(np.sum(session_harm & error)),
            "final_errors_that_are_session_harm": int(np.sum(session_harm)),
        },
        "candidate_system": {
            "candidate_rows": int(selected_candidate_rows.sum()),
            "candidate_recall_on_a_errors": float(np.mean(candidate_hit[error])),
            "candidate_hits_on_a_errors": int(np.sum(candidate_hit & error)),
            "mean_candidate_size": float(np.mean(np.sum(candidate_ids >= 0, axis=1))),
            "max_candidate_size": int(np.max(np.sum(candidate_ids >= 0, axis=1))),
            "oracle_rerank_upper_bound_correct": int((~error).sum() + np.sum(candidate_hit & error)),
            "oracle_rerank_upper_bound_top1": float(((~error).sum() + np.sum(candidate_hit & error)) / len(labels)),
            "categories": categories,
            "selection_note": "Each outer recipe maximizes source-error candidate recall, then minimizes mean size/max size/base K; held labels are evaluation-only.",
        },
        "folds": fold_reports,
        "hard_rows": row_records,
        "b_authorized": bool(np.sum(selected_candidate_rows) >= 20),
        "student_started": False,
    }
    np.savez_compressed(
        output / "hard_set.npz",
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=folds,
        a_probability=probability.astype(np.float32),
        a_prediction=prediction,
        true_rank=ranks,
        margin=margin.astype(np.float32),
        entropy=uncertainty.astype(np.float32),
        direct_k=direct_k,
        candidate_ids=candidate_ids,
        candidate_hit=candidate_hit,
        category=category,
        session_rescue=session_rescue,
        session_harm=session_harm,
    )
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
