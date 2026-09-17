"""P113 source-safe unanimous multimodal arbitration for P89 Top-2 pairs.

The four base specialists (GlobalV, LocalV, Skeleton, IMU) are trained for one
binary boundary at a time.  A P89 decision is changed only when all four
specialists agree on the other member of the exact P89 Top-2 pair.  Pair
eligibility and the call/abstain decision are estimated exclusively from the
other source users of each outer fold.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from audit_p112_multimodal_pair_tree import BinaryHead, OUTER_USERS, _align
from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
DEFAULT_DESCRIPTORS = HERE / "runs/p112_multimodal_pair_tree_v1/modality_descriptors.npz"
DEFAULT_OUTPUT = HERE / "runs/p113_multimodal_unanimous_arbitration_v1"
MODALITIES = ("GlobalV", "LocalV", "Skeleton", "IMU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def unanimous_prediction(
    predictions: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions and a mask where every specialist agrees against A."""

    if predictions.ndim != 2 or predictions.shape[0] != len(MODALITIES):
        raise ValueError(f"unexpected specialist prediction shape: {predictions.shape}")
    agreement = np.all(predictions == predictions[:1], axis=0)
    change = agreement & (predictions[0] != base)
    output = base.copy()
    output[change] = predictions[0, change]
    return output, change


def transition(labels: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    base_correct = base == labels
    candidate_correct = candidate == labels
    rescue = int(np.sum((~base_correct) & candidate_correct))
    harm = int(np.sum(base_correct & (~candidate_correct)))
    return {
        "routes": int(len(labels)),
        "changes": int(np.sum(base != candidate)),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
    }


def run(descriptor_path: Path, output: Path) -> dict[str, Any]:
    with np.load(descriptor_path, allow_pickle=False) as archive:
        full_ids = archive["sample_ids"].astype(str)
        full_users = archive["users"].astype(str)
        full_labels = archive["labels"].astype(np.int64)
        full_values = {
            name: archive[name].astype(np.float32) for name in MODALITIES
        }

    splits = load_splits()
    fold_order = tuple(OUTER_USERS)
    ids = np.concatenate([splits[name].sample_ids.astype(str) for name in fold_order])
    users = np.concatenate([splits[name].users.astype(str) for name in fold_order])
    labels = np.concatenate([splits[name].labels.astype(np.int64) for name in fold_order])
    safe = np.concatenate([splits[name].safe_prediction.astype(np.int64) for name in fold_order])
    probability = np.concatenate(
        [splits[name].safe_probability.astype(np.float64) for name in fold_order]
    )
    fold_names = np.concatenate(
        [np.repeat(name, len(splits[name].sample_ids)) for name in fold_order]
    ).astype(str)
    order = _align(full_ids, ids)
    canonical = {name: values[order] for name, values in full_values.items()}
    top2 = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    top2_pair = np.sort(top2, axis=1)

    system = safe.copy()
    pair_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    fold_summary: dict[str, Any] = {}

    for outer in fold_order:
        held = fold_names == outer
        source = ~held
        held_users = set(OUTER_USERS[outer])
        opportunities: Counter[tuple[int, int]] = Counter()
        for row in np.flatnonzero(source & (safe != labels)):
            pair = tuple(map(int, top2_pair[row]))
            if int(labels[row]) in pair and int(safe[row]) in pair:
                opportunities[pair] += 1
        candidates = sorted(
            (pair for pair, count in opportunities.items() if count >= 2),
            key=lambda pair: (-opportunities[pair], pair),
        )
        activated: dict[tuple[int, int], dict[str, BinaryHead]] = {}

        for pair in candidates:
            route_predictions: dict[str, dict[int, int]] = {
                name: {} for name in MODALITIES
            }
            for inner_user in sorted(set(users[source].tolist())):
                eval_rows = np.flatnonzero(
                    source
                    & (users == inner_user)
                    & np.all(top2_pair == np.asarray(pair)[None], axis=1)
                    & np.isin(safe, pair)
                )
                if not len(eval_rows):
                    continue
                train = (
                    ~np.isin(full_users, list(held_users) + [inner_user])
                    & np.isin(full_labels, pair)
                )
                if len(set(full_labels[train].tolist())) != 2:
                    continue
                for name in MODALITIES:
                    head = BinaryHead.fit(full_values[name][train], full_labels[train])
                    predicted = head.predict(canonical[name][eval_rows])
                    route_predictions[name].update(
                        zip(eval_rows.tolist(), predicted.tolist())
                    )

            common = sorted(
                set.intersection(*(set(values) for values in route_predictions.values()))
            )
            evaluated = np.asarray(common, dtype=np.int64)
            stacked = np.stack(
                [
                    np.asarray([route_predictions[name][int(row)] for row in evaluated])
                    for name in MODALITIES
                ],
                axis=0,
            )
            source_candidate, source_change = unanimous_prediction(stacked, safe[evaluated])
            source_metrics = transition(labels[evaluated], safe[evaluated], source_candidate)
            per_user_net = []
            for subject in sorted(set(users[evaluated].tolist())):
                selected = users[evaluated] == subject
                per_user_net.append(
                    transition(
                        labels[evaluated][selected],
                        safe[evaluated][selected],
                        source_candidate[selected],
                    )["net"]
                )
            deploy = (
                source_metrics["rescue"] >= 2
                and source_metrics["net"] >= 2
                and source_metrics["harm"] == 0
                and (min(per_user_net) if per_user_net else -999) >= 0
            )

            train = (~np.isin(full_users, list(held_users))) & np.isin(full_labels, pair)
            final_heads = {
                name: BinaryHead.fit(full_values[name][train], full_labels[train])
                for name in MODALITIES
            }
            held_route = np.flatnonzero(
                held
                & np.all(top2_pair == np.asarray(pair)[None], axis=1)
                & np.isin(safe, pair)
            )
            held_stacked = np.stack(
                [final_heads[name].predict(canonical[name][held_route]) for name in MODALITIES],
                axis=0,
            )
            held_candidate, held_change = unanimous_prediction(
                held_stacked, safe[held_route]
            )
            held_metrics = transition(labels[held_route], safe[held_route], held_candidate)
            pair_rows.append(
                {
                    "outer_fold": outer,
                    "pair": f"{pair[0]}<->{pair[1]}",
                    "source_error_opportunities": opportunities[pair],
                    "source_routes": source_metrics["routes"],
                    "source_changes": source_metrics["changes"],
                    "source_rescue": source_metrics["rescue"],
                    "source_harm": source_metrics["harm"],
                    "source_net": source_metrics["net"],
                    "source_worst_user_net": min(per_user_net) if per_user_net else -999,
                    "deploy": int(deploy),
                    "held_routes": held_metrics["routes"],
                    "held_changes_diagnostic": held_metrics["changes"],
                    "held_rescue_diagnostic": held_metrics["rescue"],
                    "held_harm_diagnostic": held_metrics["harm"],
                    "held_net_diagnostic": held_metrics["net"],
                }
            )
            if deploy:
                activated[pair] = final_heads

        before = system.copy()
        for pair, heads in activated.items():
            rows = np.flatnonzero(
                held
                & np.all(top2_pair == np.asarray(pair)[None], axis=1)
                & np.isin(safe, pair)
            )
            stacked = np.stack(
                [heads[name].predict(canonical[name][rows]) for name in MODALITIES],
                axis=0,
            )
            candidate, _ = unanimous_prediction(stacked, safe[rows])
            system[rows] = candidate
        metrics = transition(labels[held], safe[held], system[held])
        fold_summary[outer] = {
            "rows": int(held.sum()),
            "p89_correct": int(np.sum(held & (safe == labels))),
            "system_correct": int(np.sum(held & (system == labels))),
            "accuracy": float(np.mean(system[held] == labels[held])),
            **metrics,
            "activated_pairs": [f"{pair[0]}<->{pair[1]}" for pair in activated],
        }

    base_correct = safe == labels
    final_correct = system == labels
    for row in range(len(ids)):
        sample_rows.append(
            {
                "sample_id": ids[row],
                "subject": users[row],
                "outer_fold": fold_names[row],
                "true_label": int(labels[row]),
                "p89_prediction": int(safe[row]),
                "p89_top2": f"{int(top2[row, 0])}|{int(top2[row, 1])}",
                "system_prediction": int(system[row]),
                "changed": int(system[row] != safe[row]),
                "rescued": int((not base_correct[row]) and final_correct[row]),
                "harmed": int(base_correct[row] and (not final_correct[row])),
            }
        )

    total = transition(labels, safe, system)
    summary = {
        "stage": "P113_multimodal_unanimous_arbitration",
        "status": "complete",
        "protocol": {
            "route": "exact P89 adjusted Top-2 pair",
            "change": "GlobalV, LocalV, Skeleton and IMU unanimous against P89",
            "deploy_gate": "source rescue>=2, net>=2, harm=0, worst user net>=0",
            "held_labels_used_for_selection": False,
        },
        "p89": {"correct": int(base_correct.sum()), "rows": len(labels), "accuracy": float(base_correct.mean())},
        "system": {
            "correct": int(final_correct.sum()),
            "rows": len(labels),
            "accuracy": float(final_correct.mean()),
            **total,
        },
        "folds": fold_summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "pair_audit.csv", pair_rows)
    write_csv(output / "sample_predictions.csv", sample_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    run(args.descriptors.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
