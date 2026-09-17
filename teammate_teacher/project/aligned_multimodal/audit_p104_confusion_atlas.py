"""Build the P104 A/A+Session confusion atlas and source-safe pair families."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_p102_hard_set import source_crossfit_session_probability
from audit_p102_session_closure import load_npz, true_rank
from audit_p87_sequence_decoder import DecoderConfig, align_metadata
from p100a_global_teacher_data import FOLD_USERS, H3_USERS


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_RAW = HERE / "runs/p100a_a0_global_teacher_oof_v1/VS_complete_oof.npz"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_CLASSES = PROJECT / "class_mapping.csv"
DEFAULT_OUTPUT = HERE / "runs/p104_confusion_atlas_v1"
MIN_PAIR_ERRORS = 5
MIN_PAIR_SUBJECTS = 3
MAX_FAMILIES_PER_FOLD = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--session-summary", type=Path, default=DEFAULT_SESSION_SUMMARY)
    parser.add_argument("--raw-oof", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def entropy(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    return -np.sum(values * np.log(np.maximum(values, 1e-12)), axis=1)


def margin(probability: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(probability, dtype=np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def top_classes(probability: np.ndarray, count: int = 5) -> np.ndarray:
    return np.argsort(np.asarray(probability), axis=1)[:, ::-1][:, :count]


def read_class_names(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["action_id"]): str(row["action_name"])
            for row in csv.DictReader(handle)
        }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def pair_key(classes: Iterable[int]) -> str:
    return "__".join(str(value) for value in sorted(map(int, classes)))


def select_pair_families(
    labels: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    users: np.ndarray,
    source: np.ndarray,
    *,
    min_errors: int = MIN_PAIR_ERRORS,
    min_subjects: int = MIN_PAIR_SUBJECTS,
    max_families: int = MAX_FAMILIES_PER_FOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select pair families using source errors only.

    The function deliberately accepts a source mask and never indexes rows outside it.
    """

    selected_rows = np.flatnonzero(np.asarray(source, dtype=bool))
    counts: Counter[tuple[int, int]] = Counter()
    directed: Counter[tuple[int, int]] = Counter()
    subjects: dict[tuple[int, int], set[str]] = defaultdict(set)
    top3_hits: Counter[tuple[int, int]] = Counter()
    ranks = true_rank(probability[selected_rows], labels[selected_rows])
    for offset, row in enumerate(selected_rows.tolist()):
        true = int(labels[row])
        predicted = int(prediction[row])
        if true == predicted:
            continue
        key = tuple(sorted((true, predicted)))
        counts[key] += 1
        directed[(true, predicted)] += 1
        subjects[key].add(str(users[row]))
        if int(ranks[offset]) <= 3:
            top3_hits[key] += 1

    eligible: list[dict[str, Any]] = []
    for classes, rows in counts.items():
        subject_count = len(subjects[classes])
        if rows < min_errors or subject_count < min_subjects:
            continue
        a, b = classes
        eligible.append(
            {
                "family_key": pair_key(classes),
                "classes": [a, b],
                "error_rows": int(rows),
                "subjects": subject_count,
                "top3_hit_rows": int(top3_hits[classes]),
                "directed": {
                    f"{a}->{b}": int(directed[(a, b)]),
                    f"{b}->{a}": int(directed[(b, a)]),
                },
                "source_sample_count": int(
                    np.sum(source & np.isin(labels, np.asarray(classes, dtype=np.int64)))
                ),
            }
        )
    eligible.sort(
        key=lambda value: (
            -value["subjects"],
            -value["error_rows"],
            -value["top3_hit_rows"],
            value["classes"],
        )
    )
    selected = eligible[:max_families]
    clusters = connected_components(eligible)
    return selected, eligible, clusters


def connected_components(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    edge_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for edge in edges:
        a, b = map(int, edge["classes"])
        adjacency[a].add(b)
        adjacency[b].add(a)
        edge_lookup[(min(a, b), max(a, b))] = edge
    visited: set[int] = set()
    output: list[dict[str, Any]] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            value = stack.pop()
            if value in visited:
                continue
            visited.add(value)
            component.append(value)
            stack.extend(sorted(adjacency[value] - visited, reverse=True))
        component.sort()
        component_edges = [
            edge
            for key, edge in edge_lookup.items()
            if key[0] in component and key[1] in component
        ]
        output.append(
            {
                "classes": component,
                "size": len(component),
                "benchmark_candidate": 3 <= len(component) <= 4,
                "oversized": len(component) > 4,
                "edges": [edge["family_key"] for edge in component_edges],
                "error_rows_sum_nonunique": int(
                    sum(edge["error_rows"] for edge in component_edges)
                ),
            }
        )
    output.sort(key=lambda value: (-value["error_rows_sum_nonunique"], value["classes"]))
    return output


def confusion_records(
    labels: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
    names: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directed = Counter()
    directed_users: dict[tuple[int, int], set[str]] = defaultdict(set)
    symmetric = Counter()
    symmetric_users: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in range(len(labels)):
        true = int(labels[row])
        predicted = int(prediction[row])
        if true == predicted:
            continue
        directed[(true, predicted)] += 1
        directed_users[(true, predicted)].add(str(users[row]))
        pair = tuple(sorted((true, predicted)))
        symmetric[pair] += 1
        symmetric_users[pair].add(str(users[row]))
    directed_rows = [
        {
            "true_class": true,
            "true_name": names[true],
            "predicted_class": predicted,
            "predicted_name": names[predicted],
            "error_rows": rows,
            "subjects": len(directed_users[(true, predicted)]),
        }
        for (true, predicted), rows in directed.most_common()
    ]
    symmetric_rows = [
        {
            "family_key": pair_key(classes),
            "class_a": classes[0],
            "class_a_name": names[classes[0]],
            "class_b": classes[1],
            "class_b_name": names[classes[1]],
            "error_rows": rows,
            "subjects": len(symmetric_users[classes]),
            "a_to_b": int(directed[(classes[0], classes[1])]),
            "b_to_a": int(directed[(classes[1], classes[0])]),
        }
        for classes, rows in symmetric.most_common()
    ]
    return directed_rows, symmetric_rows


def system_summary(
    probability: np.ndarray, labels: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    prediction = probability.argmax(axis=1)
    ranks = true_rank(probability, labels)
    return {
        "top1_correct": int(np.sum(prediction == labels)),
        "top1": float(np.mean(prediction == labels)),
        "top2": float(np.mean(ranks <= 2)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "errors": int(np.sum(prediction != labels)),
        "per_subject": {
            user: {
                "rows": int(np.sum(users == user)),
                "correct": int(np.sum((users == user) & (prediction == labels))),
                "accuracy": float(np.mean(prediction[users == user] == labels[users == user])),
            }
            for user in sorted(set(users.tolist()))
        },
    }


def decoder_config(fold_summary: dict[str, Any]) -> DecoderConfig:
    selected = fold_summary["selected"]
    return DecoderConfig(
        gap_seconds=float(selected["gap_seconds"]),
        transition_weight=float(selected["transition_weight"]),
        trigram_backoff=float(selected["trigram_backoff"]),
        beam_width=int(selected["beam_width"]),
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = load_npz(args.session.resolve())
    raw_archive = load_npz(args.raw_oof.resolve())
    session_summary = json.loads(args.session_summary.resolve().read_text(encoding="utf-8"))
    names = read_class_names(args.class_mapping.resolve())

    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    folds = np.asarray(session["fold_ids"], dtype=np.int64)
    raw_probability = np.asarray(session["vs_raw_probability"], dtype=np.float64)
    session_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    if str(np.asarray(session["selected_system"]).item()) != "VS_session":
        raise RuntimeError("P104 input is not frozen VS_session")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached P104 atlas")
    if not np.array_equal(raw_archive["sample_ids"].astype(str), sample_ids):
        raise RuntimeError("P100/P102 sample order differs")
    raw_reference = np.asarray(raw_archive["direct_probability"], dtype=np.float64)
    if float(np.max(np.abs(raw_reference - raw_probability))) > 1e-6:
        raise RuntimeError("P100 raw probability changed inside P102")

    raw_prediction = raw_probability.argmax(axis=1)
    session_prediction = session_probability.argmax(axis=1)
    raw_top = top_classes(raw_probability)
    session_top = top_classes(session_probability)
    raw_ranks = true_rank(raw_probability, labels)
    session_ranks = true_rank(session_probability, labels)
    raw_margin = margin(raw_probability)
    session_margin = margin(session_probability)
    raw_entropy = entropy(raw_probability)
    session_entropy = entropy(session_probability)
    row_records: list[dict[str, Any]] = []
    for row in range(len(labels)):
        row_records.append(
            {
                "sample_id": sample_ids[row],
                "subject": users[row],
                "fold": int(folds[row]),
                "true_class": int(labels[row]),
                "true_name": names[int(labels[row])],
                "a_raw_top1": int(raw_prediction[row]),
                "a_session_top1": int(session_prediction[row]),
                "a_session_top2": "|".join(map(str, session_top[row, :2].tolist())),
                "a_session_top3": "|".join(map(str, session_top[row, :3].tolist())),
                "a_session_top5": "|".join(map(str, session_top[row, :5].tolist())),
                "a_raw_true_rank": int(raw_ranks[row]),
                "a_session_true_rank": int(session_ranks[row]),
                "a_raw_margin": float(raw_margin[row]),
                "a_session_margin": float(session_margin[row]),
                "a_raw_entropy": float(raw_entropy[row]),
                "a_session_entropy": float(session_entropy[row]),
                "a_raw_correct": int(raw_prediction[row] == labels[row]),
                "a_session_correct": int(session_prediction[row] == labels[row]),
                "session_rescue": int(
                    raw_prediction[row] != labels[row]
                    and session_prediction[row] == labels[row]
                ),
                "session_harm": int(
                    raw_prediction[row] == labels[row]
                    and session_prediction[row] != labels[row]
                ),
            }
        )
    write_csv(output / "atlas_rows.csv", row_records)
    raw_directed, raw_symmetric = confusion_records(
        labels, raw_prediction, users, names
    )
    session_directed, session_symmetric = confusion_records(
        labels, session_prediction, users, names
    )
    write_csv(output / "a_raw_directed_confusions.csv", raw_directed)
    write_csv(output / "a_raw_symmetric_confusions.csv", raw_symmetric)
    write_csv(output / "a_session_directed_confusions.csv", session_directed)
    write_csv(output / "a_session_symmetric_confusions.csv", session_symmetric)

    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    fold_reports: list[dict[str, Any]] = []
    family_occurrence: dict[str, list[int]] = defaultdict(list)
    for outer_fold in range(4):
        source = folds != outer_fold
        held = ~source
        source_probability, source_session_audit = source_crossfit_session_probability(
            outer_fold,
            args.nested_root.resolve(),
            sample_ids,
            users,
            folds,
            labels,
            metadata,
            decoder_config(session_summary["folds"][outer_fold]),
        )
        source_prediction = source_probability.argmax(axis=1)
        selected, eligible, clusters = select_pair_families(
            labels, source_prediction, source_probability, users, source
        )
        for family in selected:
            family["held_sample_count"] = int(
                np.sum(held & np.isin(labels, family["classes"]))
            )
            family_occurrence[family["family_key"]].append(outer_fold)
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "held_users": list(FOLD_USERS[outer_fold]),
                "source_users": sorted(set(users[source].tolist())),
                "source_rows": int(source.sum()),
                "held_rows": int(held.sum()),
                "source_session_audit": source_session_audit,
                "selection_rule": {
                    "min_pair_errors": MIN_PAIR_ERRORS,
                    "min_pair_subjects": MIN_PAIR_SUBJECTS,
                    "max_families": MAX_FAMILIES_PER_FOLD,
                },
                "selected_families": selected,
                "eligible_families": eligible,
                "cluster_suggestions": clusters,
            }
        )
    stable_families = [
        {
            "family_key": key,
            "classes": [int(value) for value in key.split("__")],
            "selected_folds": value,
            "selected_fold_count": len(value),
            "formal_verdict_eligible": len(value) >= 2,
        }
        for key, value in sorted(
            family_occurrence.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]
    (output / "fold_families.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": "P104 source-safe outer-source confusion pair construction",
                "folds": fold_reports,
                "stable_families": stable_families,
                "h3_rows_selected": 0,
                "h3_users_loaded": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    raw_correct = raw_prediction == labels
    session_correct = session_prediction == labels
    summary = {
        "status": "complete",
        "protocol": "P104 confusion atlas from frozen source-safe subject-disjoint A/A+Session OOF",
        "data": {
            "rows": len(labels),
            "subjects": sorted(set(users.tolist())),
            "fold_counts": [int(np.sum(folds == fold)) for fold in range(4)],
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "a_raw": system_summary(raw_probability, labels, users),
        "a_session": system_summary(session_probability, labels, users),
        "session_change": {
            "rescue": int(np.sum((~raw_correct) & session_correct)),
            "harm": int(np.sum(raw_correct & (~session_correct))),
            "net": int(np.sum(session_correct) - np.sum(raw_correct)),
        },
        "a_session_error_true_rank": {
            f"top{k}": int(np.sum((~session_correct) & (session_ranks <= k)))
            for k in (2, 3, 5)
        },
        "top_directed_confusions": session_directed[:30],
        "top_symmetric_confusions": session_symmetric[:30],
        "stable_families": stable_families,
        "artifacts": {
            "atlas_rows": "atlas_rows.csv",
            "raw_directed": "a_raw_directed_confusions.csv",
            "raw_symmetric": "a_raw_symmetric_confusions.csv",
            "session_directed": "a_session_directed_confusions.csv",
            "session_symmetric": "a_session_symmetric_confusions.csv",
            "fold_families": "fold_families.json",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
