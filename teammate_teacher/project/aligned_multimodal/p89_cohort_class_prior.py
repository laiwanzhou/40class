from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics, decode_sessions
from p88_train_depth_residual import rescue_harm
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_cohort_class_prior_v1"


def date_number(values: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float64)
    for index, value in enumerate(values.astype(str)):
        if value and value.lower() not in {"nan", "unknown", "none"}:
            try:
                output[index] = float(
                    np.datetime64(value, "D").astype("datetime64[D]").astype(np.int64)
                )
            except ValueError:
                pass
    return output


def user_centers(metadata) -> dict[str, float]:
    dates = date_number(metadata.dates)
    result = {}
    for user in sorted(set(metadata.users.astype(str).tolist())):
        values = dates[metadata.users.astype(str) == user]
        values = values[np.isfinite(values)]
        if len(values):
            result[user] = float(np.median(values))
    return result


def class_biases(
    all_labels: np.ndarray,
    all_metadata,
    fit_mask: np.ndarray,
    target_metadata,
    target_groups: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    fit_users = all_metadata.users.astype(str)[fit_mask]
    allowed_users = sorted(set(fit_users.tolist()))
    all_centers = user_centers(all_metadata)
    target_dates = date_number(target_metadata.dates)
    global_counts = np.bincount(all_labels[fit_mask], minlength=40).astype(np.float64) + 1.0
    global_prior = global_counts / global_counts.sum()
    biases = np.zeros((len(target_groups), 40), dtype=np.float64)
    audit = {}
    for group in sorted(set(target_groups.astype(str).tolist())):
        selected = target_groups.astype(str) == group
        dates = target_dates[selected]
        dates = dates[np.isfinite(dates)]
        if not len(dates):
            audit[group] = []
            continue
        center = float(np.median(dates))
        nearest = sorted(
            allowed_users,
            key=lambda user: (abs(all_centers[user] - center), user),
        )[:neighbors]
        counts = np.full(40, 1e-3, dtype=np.float64)
        for user in nearest:
            rows = fit_mask & (all_metadata.users.astype(str) == user)
            user_counts = np.bincount(all_labels[rows], minlength=40).astype(np.float64)
            counts += user_counts / max(float(user_counts.sum()), 1.0)
        cohort_prior = counts / counts.sum()
        biases[selected] = np.log(np.maximum(cohort_prior, 1e-12)) - np.log(
            np.maximum(global_prior, 1e-12)
        )
        audit[group] = nearest
    return biases, audit


def evaluate(
    protocol_value,
    all_labels: np.ndarray,
    all_metadata,
    fit_mask: np.ndarray,
    target_groups: np.ndarray,
    neighbors: int,
    weight: float,
    method: str,
    grouping_config: GlobalRepeatConfig,
) -> tuple[dict, np.ndarray]:
    biases, neighbor_audit = class_biases(
        all_labels,
        all_metadata,
        fit_mask,
        protocol_value[4],
        target_groups,
        neighbors,
    )
    logp = np.log(np.maximum(protocol_value[2], 1e-12)) + weight * biases
    adjusted = np.exp(logp - np.logaddexp.reduce(logp, axis=1, keepdims=True))
    decoded = decode_sessions(
        np.log(np.maximum(adjusted, 1e-12)),
        protocol_value[6],
        protocol_value[7],
        protocol_value[8],
    )
    if method == "decoded":
        prediction = decoded
        grouping = None
    else:
        adjusted_protocol = list(protocol_value)
        adjusted_protocol[2] = adjusted
        prediction, grouping = joint_decode(
            adjusted,
            protocol_value[3],
            tuple(adjusted_protocol),
            grouping_config,
            evidence_weight=0.25,
            transition_scale=1.0,
        )
    users = protocol_value[4].users.astype(str)
    gains = []
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        base_correct = int(np.sum(protocol_value[3][rows] == protocol_value[1][rows]))
        candidate_correct = int(np.sum(prediction[rows] == protocol_value[1][rows]))
        gains.append(candidate_correct - base_correct)
        per_user[user] = candidate_correct - base_correct
    return (
        {
            "configuration": {
                "neighbors": neighbors,
                "weight": weight,
                "method": method,
            },
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_p87": rescue_harm(
                protocol_value[1], protocol_value[3], prediction
            ),
            "minimum_user_gain": int(min(gains)),
            "positive_users": int(np.sum(np.asarray(gains) > 0)),
            "per_user_gain": per_user,
            "nearest_fit_users": neighbor_audit,
            "grouping": grouping,
        },
        prediction,
    )


def main() -> None:
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = full40.align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    grouping_source = json.loads(
        (
            PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    grouping_config = GlobalRepeatConfig(
        **grouping_source["H1_selected"]["configuration"]
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    fit_h1 = ~np.isin(all_metadata.users.astype(str), full40.H1_USERS)
    fit_h2 = ~np.isin(all_metadata.users.astype(str), full40.H2_USERS)
    candidates = []
    predictions = []
    for neighbors in (1, 2, 3, 4, 5, 8):
        for weight in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
            for method in ("decoded", "joint"):
                item, prediction = evaluate(
                    h1,
                    all_labels,
                    all_metadata,
                    fit_h1,
                    h1[4].users.astype(str),
                    neighbors,
                    weight,
                    method,
                    grouping_config,
                )
                candidates.append(item)
                predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["positive_users"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_p87"]["net"],
            -candidates[index]["rescue_harm_vs_p87"]["harm"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    configuration = selected["configuration"]
    confirmation, h2_prediction = evaluate(
        h2,
        all_labels,
        all_metadata,
        fit_h2,
        h2[4].users.astype(str),
        int(configuration["neighbors"]),
        float(configuration["weight"]),
        str(configuration["method"]),
        grouping_config,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_recording_cohort_class_prior_v1",
        "protocol": (
            "Use only recording dates to find nearest labeled subject cohorts. "
            "Estimate a cohort-vs-global class prior after excluding the entire "
            "holdout set, select neighbor count/prior weight/method on H1, and "
            "transfer unchanged to H2."
        ),
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "grid_size": len(candidates),
        "all_H1_candidates": [candidates[index] for index in order],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "all_H1_candidates"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
