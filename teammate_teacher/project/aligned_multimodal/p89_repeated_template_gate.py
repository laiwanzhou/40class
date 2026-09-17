from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_unique_beam,
)
from p88_session_template_decoder import path_score
from p88_train_depth_residual import rescue_harm
from p89_deterministic_triple_repeat import triple_groups
from p89_supported_template_gate import (
    fit_supported_templates,
    h3_protocol,
    load_grouping,
    load_imu,
    safe_probability_and_prediction,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_repeated_template_gate_v1"


def repeated_records(protocol_value, probability, templates):
    logp = np.log(np.maximum(probability, 1e-12))
    records = []
    for group_index, group in enumerate(triple_groups(protocol_value, "exact_three")):
        source = templates.get(len(group[0]))
        if source is None:
            continue
        paths, frequencies, supports = source
        emission = np.stack([logp[session] for session in group]).mean(axis=0)
        scores = np.asarray(
            [
                path_score(
                    emission,
                    path,
                    protocol_value[7],
                    protocol_value[8].transition_weight,
                )
                for path in paths
            ]
        )
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_path = paths[best_index]
        unconstrained = decode_unique_beam(
            emission,
            protocol_value[7],
            protocol_value[8].transition_weight,
            protocol_value[8].beam_width,
        )
        unconstrained_score = path_score(
            emission,
            unconstrained,
            protocol_value[7],
            protocol_value[8].transition_weight,
        )
        second_score = float(scores[int(order[1])]) if len(order) > 1 else -np.inf
        records.append(
            {
                "group_index": group_index,
                "group": group,
                "path": best_path,
                "frequency": int(frequencies[best_index]),
                "subject_support": int(supports[best_index]),
                "loss_per_row": float(
                    max(unconstrained_score - float(scores[best_index]), 0.0)
                    / len(best_path)
                ),
                "margin_per_row": float(
                    (float(scores[best_index]) - second_score) / len(best_path)
                ),
            }
        )
    return records


def apply(safe, records, config):
    prediction = safe.copy()
    accepted_groups = 0
    changed_rows = 0
    for record in records:
        if record["subject_support"] < config["minimum_subject_support"]:
            continue
        if record["frequency"] < config["minimum_frequency"]:
            continue
        if record["loss_per_row"] > config["maximum_loss_per_row"] + 1e-12:
            continue
        if record["margin_per_row"] < config["minimum_margin_per_row"]:
            continue
        changed = 0
        for session in record["group"]:
            changed += int(np.sum(prediction[session] != record["path"]))
            prediction[session] = record["path"]
        if changed:
            accepted_groups += 1
            changed_rows += changed
    return prediction, {
        "accepted_groups": accepted_groups,
        "changed_rows": changed_rows,
    }


def per_user(protocol_value, prediction, safe):
    users = protocol_value[4].users.astype(str)
    labels = protocol_value[1]
    return {
        user: {
            "gain": int(
                np.sum(prediction[users == user] == labels[users == user])
                - np.sum(safe[users == user] == labels[users == user])
            ),
            "changes": int(np.sum(prediction[users == user] != safe[users == user])),
        }
        for user in sorted(set(users.tolist()))
    }


def evaluate(protocol_value, safe, records, config):
    prediction, gate = apply(safe, records, config)
    users = per_user(protocol_value, prediction, safe)
    return (
        {
            "configuration": config,
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_safe": rescue_harm(
                protocol_value[1], safe, prediction
            ),
            "minimum_user_gain": min(value["gain"] for value in users.values()),
            "positive_users": sum(value["gain"] > 0 for value in users.values()),
            "per_user": users,
            "gate": gate,
        },
        prediction,
    )


def truth_audit(protocol_value, safe, records):
    labels = protocol_value[1]
    exact_repeat = true_template_present = best_template_exact = 0
    covered_rows = safe_correct = candidate_correct = 0
    for record in records:
        paths = [tuple(labels[session].tolist()) for session in record["group"]]
        exact_repeat += int(len(set(paths)) == 1)
        if len(set(paths)) == 1:
            # Presence is inferred here only for validation audit; it is never a gate.
            true_template_present += int(
                tuple(record["path"].tolist()) == paths[0]
                or record["subject_support"] >= 0
            )
            best_template_exact += int(tuple(record["path"].tolist()) == paths[0])
        rows = np.concatenate(record["group"])
        candidate = np.tile(record["path"], len(record["group"]))
        covered_rows += len(rows)
        safe_correct += int(np.sum(safe[rows] == labels[rows]))
        candidate_correct += int(np.sum(candidate == labels[rows]))
    return {
        "groups": len(records),
        "exact_repeat_groups": exact_repeat,
        "best_template_exact_groups": best_template_exact,
        "covered_rows": covered_rows,
        "safe_correct_in_coverage": safe_correct,
        "ungated_best_template_correct": candidate_correct,
    }


def prepare(protocol_value, fit_mask, all_labels, all_metadata, imu, grouping):
    probability, safe = safe_probability_and_prediction(
        protocol_value, imu[0], imu[1], grouping
    )
    fit_sessions = build_sessions(
        np.flatnonzero(fit_mask),
        all_metadata,
        protocol_value[8].gap_seconds,
        "known_user",
    )
    templates = fit_supported_templates(
        all_labels, fit_sessions, all_metadata, maximum_length=20
    )
    records = repeated_records(protocol_value, probability, templates)
    return probability, safe, records, truth_audit(protocol_value, safe, records)


def main() -> None:
    grouping = load_grouping()
    imu = load_imu()
    with np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    ) as source:
        all_ids = source["oof_sample_ids"].astype(str)
        all_labels = source["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_values = prepare(
        h1,
        ~np.isin(all_metadata.users, full40.H1_USERS),
        all_labels,
        all_metadata,
        imu,
        grouping,
    )
    h2_values = prepare(
        h2,
        ~np.isin(all_metadata.users, full40.H2_USERS),
        all_labels,
        all_metadata,
        imu,
        grouping,
    )
    h3_values = h3_protocol(imu[0], imu[1], grouping)
    h3, h3_probability, h3_safe, _, h3_labels, h3_metadata, h3_fit = h3_values
    h3_fit_sessions = build_sessions(
        np.flatnonzero(h3_fit), h3_metadata, h3[8].gap_seconds, "known_user"
    )
    h3_templates = fit_supported_templates(
        h3_labels, h3_fit_sessions, h3_metadata, maximum_length=20
    )
    h3_records = repeated_records(h3, h3_probability, h3_templates)
    h3_truth = truth_audit(h3, h3_safe, h3_records)

    configurations = [
        {
            "minimum_subject_support": support,
            "minimum_frequency": frequency,
            "maximum_loss_per_row": loss,
            "minimum_margin_per_row": margin,
        }
        for support in (1, 2, 3)
        for frequency in (1, 2, 3, 5, 8)
        for loss in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0)
        for margin in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0)
    ]
    candidates = []
    h1_predictions = []
    for config in configurations:
        result, prediction = evaluate(h1, h1_values[1], h1_values[2], config)
        candidates.append(result)
        h1_predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["minimum_user_gain"],
            candidates[index]["positive_users"],
            -candidates[index]["rescue_harm_vs_safe"]["harm"],
            -candidates[index]["gate"]["changed_rows"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    config = selected["configuration"]
    h2_result, h2_prediction = evaluate(h2, h2_values[1], h2_values[2], config)
    h3_result, h3_prediction = evaluate(h3, h3_safe, h3_records, config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h1_safe=h1_values[1],
        h1_prediction=h1_predictions[selected_index],
        h2_sample_ids=h2[0],
        h2_safe=h2_values[1],
        h2_prediction=h2_prediction,
        h3_sample_ids=h3[0],
        h3_safe=h3_safe,
        h3_prediction=h3_prediction,
    )
    report = {
        "stage": "P89_three_take_aggregated_training_script_gate_v1",
        "protocol": (
            "Aggregate only exact runs of three equal-length sessions, score unique "
            "action scripts learned without held subjects, select model-only loss/"
            "margin/support thresholds on H1, and transfer once to H2 and H3."
        ),
        "H1": {
            "safe": classification_metrics(h1[1], h1_values[1]),
            "truth_audit": h1_values[3],
            "selected": selected,
        },
        "H2": {
            "safe": classification_metrics(h2[1], h2_values[1]),
            "truth_audit": h2_values[3],
            "confirmation": h2_result,
        },
        "H3": {
            "safe": classification_metrics(h3[1], h3_safe),
            "truth_audit": h3_truth,
            "confirmation": h3_result,
        },
        "grid_size": len(candidates),
        "top_H1_candidates": [candidates[index] for index in order[:50]],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "top_H1_candidates"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
