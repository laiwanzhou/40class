"""Source-selected classwise thresholds over frozen P118 route scores.

This is a calibration-only audit.  It does not refit the router.  For every
outer-held cohort, threshold maps and the policy complexity are selected solely
from the other cohorts' nested leave-one-user-out row scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_p87_sequence_decoder import classification_metrics
from p117_transductive_multicandidate_router import load_candidate_splits


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/p118_candidate_conditioned_structured_maxnet_v2"
OUTPUT = HERE / "runs/p118_classwise_threshold_router_v1"


def global_candidate_bank():
    splits = load_candidate_splits(full_visual_bank=True, structured_bank=True)
    names = list(next(iter(splits.values())).candidates)
    lookups = {}
    for split_name, value in splits.items():
        for row, sample_id in enumerate(value.split.sample_ids.astype(str)):
            lookups[sample_id] = (split_name, row)
    return splits, names, lookups


def chosen_labels(
    sample_ids: np.ndarray,
    candidate_index: np.ndarray,
    splits,
    candidate_names: list[str],
    lookups,
) -> np.ndarray:
    output = np.full(len(sample_ids), -1, dtype=np.int64)
    for row, (sample_id, index) in enumerate(
        zip(sample_ids.astype(str), candidate_index.astype(np.int64))
    ):
        if index < 0:
            continue
        split_name, position = lookups[sample_id]
        output[row] = int(
            splits[split_name].candidates[candidate_names[index]][position].argmax()
        )
    return output


def group_keys(kind: str, safe: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if kind == "safe_class":
        return safe.astype(str)
    if kind == "candidate_class":
        return candidate.astype(str)
    if kind == "directed_pair":
        return np.asarray(
            [f"{left}>{right}" for left, right in zip(safe, candidate)], dtype=str
        )
    raise ValueError(kind)


def threshold_result(
    labels: np.ndarray,
    users: np.ndarray,
    safe: np.ndarray,
    candidate: np.ndarray,
    score: np.ndarray,
    threshold: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    route = (candidate >= 0) & (candidate != safe) & (score >= threshold)
    output = safe.copy()
    output[route] = candidate[route]
    rescue = int(np.sum((output == labels) & (safe != labels)))
    harm = int(np.sum((output != labels) & (safe == labels)))
    per_user = {
        user: int(
            np.sum(output[users.astype(str) == user] == labels[users.astype(str) == user])
            - np.sum(safe[users.astype(str) == user] == labels[users.astype(str) == user])
        )
        for user in sorted(set(users.astype(str).tolist()))
    }
    return (
        {
            "correct": int(np.sum(output == labels)),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "changed": int(route.sum()),
            "minimum_user_gain": int(min(per_user.values())),
            "positive_users": int(sum(value > 0 for value in per_user.values())),
            "per_user_gain": per_user,
        },
        output,
    )


def fit_map(
    kind: str,
    labels: np.ndarray,
    users: np.ndarray,
    safe: np.ndarray,
    candidate: np.ndarray,
    score: np.ndarray,
    global_threshold: float,
    min_group_rows: int,
    min_selected_rows: int,
    min_improvement: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    keys = group_keys(kind, safe, candidate)
    gain = (candidate == labels).astype(np.int8) - (safe == labels).astype(np.int8)
    valid = (candidate >= 0) & (candidate != safe) & np.isfinite(score)
    thresholds = np.unique(
        np.concatenate((np.arange(0.30, 0.901, 0.025), [global_threshold]))
    )
    mapping: dict[str, float] = {}
    audits = {}
    for key in sorted(set(keys[valid].tolist())):
        group = valid & (keys == key)
        if int(group.sum()) < min_group_rows:
            continue
        base_route = group & (score >= global_threshold)
        base_net = int(gain[base_route].sum())
        candidates = []
        for threshold in thresholds:
            route = group & (score >= threshold)
            if int(route.sum()) < min_selected_rows:
                continue
            route_users = len(set(users[route].astype(str).tolist()))
            rescue = int(np.sum(route & (gain > 0)))
            harm = int(np.sum(route & (gain < 0)))
            candidates.append(
                {
                    "threshold": float(threshold),
                    "net": rescue - harm,
                    "rescue": rescue,
                    "harm": harm,
                    "changed": int(route.sum()),
                    "route_users": route_users,
                }
            )
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                row["net"],
                -row["harm"],
                row["route_users"],
                -row["changed"],
            ),
            reverse=True,
        )
        best = candidates[0]
        if (
            best["net"] >= base_net + min_improvement
            and best["net"] > 0
            and best["route_users"] >= 2
        ):
            mapping[key] = best["threshold"]
            audits[key] = {"base_net": base_net, **best}
    applied = np.full(len(labels), global_threshold, dtype=np.float64)
    for key, threshold in mapping.items():
        applied[keys == key] = threshold
    result, _ = threshold_result(labels, users, safe, candidate, score, applied)
    return mapping, {
        "kind": kind,
        "min_group_rows": min_group_rows,
        "min_selected_rows": min_selected_rows,
        "min_improvement": min_improvement,
        "override_count": len(mapping),
        "source_result": result,
        "override_audit": audits,
        "selection_score": result["net"] - 0.5 * len(mapping),
    }


def main() -> None:
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    saved = np.load(SOURCE / "predictions.npz")
    splits, candidate_names, lookups = global_candidate_bank()
    outer_reports = {}
    payload = {}
    total_safe = total_selected = 0
    for held_name, held in splits.items():
        source_ids = saved[f"{held_name}_source_sample_ids"]
        source_labels = saved[f"{held_name}_source_labels"]
        source_users = saved[f"{held_name}_source_users"]
        source_safe = saved[f"{held_name}_source_safe_prediction"]
        source_score = saved[f"{held_name}_source_route_score"]
        source_index = saved[f"{held_name}_source_candidate_index"]
        source_candidate = chosen_labels(
            source_ids, source_index, splits, candidate_names, lookups
        )
        global_threshold = float(
            summary["cohorts"][held_name]["selected_threshold"]["threshold"]
        )
        policy_candidates = [
            ({}, {
                "kind": "global",
                "min_group_rows": None,
                "min_selected_rows": None,
                "min_improvement": None,
                "override_count": 0,
                "source_result": threshold_result(
                    source_labels,
                    source_users,
                    source_safe,
                    source_candidate,
                    source_score,
                    np.full(len(source_labels), global_threshold),
                )[0],
            })
        ]
        policy_candidates[0][1]["selection_score"] = policy_candidates[0][1][
            "source_result"
        ]["net"]
        for kind in ("safe_class", "candidate_class", "directed_pair"):
            for min_group_rows in (8, 12, 20):
                for min_selected_rows in (3, 5):
                    for min_improvement in (1, 2):
                        policy_candidates.append(
                            fit_map(
                                kind,
                                source_labels,
                                source_users,
                                source_safe,
                                source_candidate,
                                source_score,
                                global_threshold,
                                min_group_rows,
                                min_selected_rows,
                                min_improvement,
                            )
                        )
        policy_candidates.sort(
            key=lambda item: (
                item[1]["selection_score"],
                item[1]["source_result"]["net"],
                -item[1]["source_result"]["harm"],
                -item[1]["override_count"],
            ),
            reverse=True,
        )
        mapping, selected_policy = policy_candidates[0]

        held_score = saved[f"{held_name}_route_score"]
        held_index = saved[f"{held_name}_candidate_index"]
        held_candidate = chosen_labels(
            held.split.sample_ids, held_index, splits, candidate_names, lookups
        )
        held_keys = (
            group_keys(selected_policy["kind"], held.split.safe_prediction, held_candidate)
            if selected_policy["kind"] != "global"
            else np.full(len(held_score), "global")
        )
        held_threshold = np.full(len(held_score), global_threshold, dtype=np.float64)
        for key, threshold in mapping.items():
            held_threshold[held_keys == key] = threshold
        held_result, output = threshold_result(
            held.split.labels,
            held.split.users,
            held.split.safe_prediction,
            held_candidate,
            held_score,
            held_threshold,
        )
        outer_reports[held_name] = {
            "global_threshold": global_threshold,
            "selected_policy": selected_policy,
            "threshold_map": mapping,
            "held_result": held_result,
            "held_metrics": classification_metrics(held.split.labels, output),
            "top_source_policies": [item[1] for item in policy_candidates[:10]],
        }
        total_safe += int(np.sum(held.split.safe_prediction == held.split.labels))
        total_selected += held_result["correct"]
        payload[f"{held_name}_sample_ids"] = held.split.sample_ids
        payload[f"{held_name}_labels"] = held.split.labels
        payload[f"{held_name}_safe_prediction"] = held.split.safe_prediction
        payload[f"{held_name}_prediction"] = output
        payload[f"{held_name}_threshold"] = held_threshold
    report = {
        "stage": "P118_source_selected_classwise_threshold_router_v1",
        "status": "complete",
        "protocol": {
            "router_refit": False,
            "policy_selection": "source nested LOUO only with 0.5-correct complexity penalty per override",
            "outer_held_labels_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "cohorts": outer_reports,
        "aggregate": {
            "rows": 2470,
            "safe_correct": total_safe,
            "correct": total_selected,
            "accuracy": total_selected / 2470,
            "net": total_selected - total_safe,
            "target_0.91_correct": 2248,
            "gap_to_0.91_correct": 2248 - total_selected,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
