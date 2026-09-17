from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    fit_transition_model,
)
from p88_session_template_decoder import path_score
from p88_train_depth_residual import rescue_harm
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_probability_blend import evaluate as evaluate_imu
from p89_imu_rescue_gate import aligned_imu


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_supported_template_gate_v1"
MAXIMUM_LENGTH = 20


def load_grouping() -> GlobalRepeatConfig:
    source = json.loads(
        (PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    return GlobalRepeatConfig(**source["H1_selected"]["configuration"])


def load_imu() -> tuple[np.ndarray, np.ndarray]:
    with np.load(
        PROJECT_DIR
        / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    ) as source:
        return (
            source["sample_ids"].astype(str),
            np.asarray(source["imu_logits"], dtype=np.float64),
        )


def safe_probability_and_prediction(
    protocol_value, imu_ids: np.ndarray, imu_logits: np.ndarray, grouping
) -> tuple[np.ndarray, np.ndarray]:
    probability, present = aligned_imu(
        imu_ids, imu_logits, protocol_value[0], 3.0
    )
    adjusted = protocol_value[2].copy()
    adjusted[present] = 0.95 * adjusted[present] + 0.05 * probability[present]
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    prediction = evaluate_imu(
        protocol_value, probability, present, 0.05, "joint", grouping
    )[1]
    return adjusted, prediction


def h3_protocol(imu_ids: np.ndarray, imu_logits: np.ndarray, grouping):
    with np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    ) as source:
        all_ids = source["oof_sample_ids"].astype(str)
        all_labels = source["oof_labels"].astype(np.int64)
        folds = source["oof_folds"].astype(np.int64)
        all_probability = source["oof_teacher_probability"].astype(np.float64)
    all_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    decoder_summary = json.loads(
        (PROJECT_DIR / "runs/p87_sequence_decoder_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    fold_summary = next(
        item for item in decoder_summary["folds"] if int(item["fold"]) == 0
    )
    decoder = DecoderConfig(**fold_summary["selected"])
    fit = np.flatnonzero(folds != 0)
    transition = fit_transition_model(
        all_labels,
        build_sessions(fit, all_metadata, decoder.gap_seconds, "known_user"),
        40,
        decoder.trigram_backoff,
    )
    validation = np.flatnonzero(folds == 0)
    sample_ids = all_ids[validation]
    labels = all_labels[validation]
    probability = all_probability[validation].copy()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        sample_ids,
    )
    with np.load(
        PROJECT_DIR / "runs/p87_sequence_decoder_v1/oof_predictions.npz"
    ) as source:
        p87 = source["sequence_predictions"].astype(np.int64)[validation]
    indices = np.arange(len(sample_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    protocol_value = (
        sample_ids,
        labels,
        probability,
        p87,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        None,
    )
    adjusted, safe = safe_probability_and_prediction(
        protocol_value, imu_ids, imu_logits, grouping
    )
    return protocol_value, adjusted, safe, all_ids, all_labels, all_metadata, folds != 0


def fit_supported_templates(
    labels: np.ndarray,
    sessions: list[np.ndarray],
    metadata,
    maximum_length: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    counts: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    subjects: dict[int, dict[tuple[int, ...], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for session in sessions:
        path = tuple(map(int, labels[session]))
        if not path or len(path) > maximum_length or len(set(path)) != len(path):
            continue
        counts[len(path)][path] += 1
        subjects[len(path)][path].add(str(metadata.users[session[0]]))
    result = {}
    for length, values in counts.items():
        paths = np.asarray(list(values), dtype=np.int64)
        frequencies = np.asarray(
            [values[tuple(map(int, row))] for row in paths], dtype=np.int64
        )
        subject_support = np.asarray(
            [len(subjects[length][tuple(map(int, row))]) for row in paths],
            dtype=np.int64,
        )
        result[length] = paths, frequencies, subject_support
    return result


def session_candidates(
    probability: np.ndarray,
    sessions: list[np.ndarray],
    templates,
    transition,
    decoder,
) -> list[dict[str, object]]:
    logp = np.log(np.maximum(probability, 1e-12))
    records: list[dict[str, object]] = []
    for session_index, session in enumerate(sessions):
        source = templates.get(len(session))
        if source is None:
            continue
        paths, frequencies, supports = source
        emission = logp[session]
        scores = np.asarray(
            [
                path_score(emission, path, transition, decoder.transition_weight)
                for path in paths
            ],
            dtype=np.float64,
        )
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_score = float(scores[best_index])
        second_score = float(scores[int(order[1])]) if len(order) > 1 else -np.inf
        # The row-wise top-1 sum is an optimistic, model-only reference that is
        # independent of ground truth and comparable across decoder variants.
        optimistic = float(np.max(emission, axis=1).sum())
        records.append(
            {
                "session_index": session_index,
                "session": session,
                "path": paths[best_index],
                "frequency": int(frequencies[best_index]),
                "subject_support": int(supports[best_index]),
                "loss_per_row": float((optimistic - best_score) / len(session)),
                "margin_per_row": float((best_score - second_score) / len(session)),
            }
        )
    return records


def apply_gate(
    safe: np.ndarray,
    records: list[dict[str, object]],
    configuration: dict[str, float | int],
) -> tuple[np.ndarray, dict[str, int]]:
    prediction = safe.copy()
    accepted_sessions = 0
    changed_rows = 0
    for record in records:
        if int(record["subject_support"]) < int(configuration["minimum_subject_support"]):
            continue
        if int(record["frequency"]) < int(configuration["minimum_frequency"]):
            continue
        if float(record["loss_per_row"]) > float(configuration["maximum_loss_per_row"]):
            continue
        if float(record["margin_per_row"]) < float(configuration["minimum_margin_per_row"]):
            continue
        session = np.asarray(record["session"], dtype=np.int64)
        path = np.asarray(record["path"], dtype=np.int64)
        changes = int(np.sum(prediction[session] != path))
        if not changes:
            continue
        prediction[session] = path
        accepted_sessions += 1
        changed_rows += changes
    return prediction, {
        "accepted_sessions": accepted_sessions,
        "changed_rows": changed_rows,
    }


def per_user(labels, prediction, safe, metadata) -> dict[str, dict[str, int]]:
    result = {}
    for user in sorted(set(metadata.users.astype(str).tolist())):
        rows = metadata.users.astype(str) == user
        safe_correct = int(np.sum(safe[rows] == labels[rows]))
        candidate_correct = int(np.sum(prediction[rows] == labels[rows]))
        result[user] = {
            "safe_correct": safe_correct,
            "candidate_correct": candidate_correct,
            "gain": candidate_correct - safe_correct,
            "changes": int(np.sum(prediction[rows] != safe[rows])),
        }
    return result


def evaluate_configuration(
    labels, safe, records, configuration, metadata
) -> tuple[dict[str, object], np.ndarray]:
    prediction, gate = apply_gate(safe, records, configuration)
    user = per_user(labels, prediction, safe, metadata)
    return (
        {
            "configuration": configuration,
            "metrics": classification_metrics(labels, prediction),
            "rescue_harm_vs_safe": rescue_harm(labels, safe, prediction),
            "minimum_user_gain": min(value["gain"] for value in user.values()),
            "positive_users": sum(value["gain"] > 0 for value in user.values()),
            "per_user": user,
            "gate": gate,
        },
        prediction,
    )


def oracle_audit(labels, safe, records) -> dict[str, int]:
    covered = np.zeros(len(labels), dtype=bool)
    exact = np.zeros(len(labels), dtype=bool)
    for record in records:
        session = np.asarray(record["session"], dtype=np.int64)
        covered[session] = True
        exact[session] = np.asarray(record["path"], dtype=np.int64) == labels[session]
    return {
        "covered_rows": int(np.sum(covered)),
        "safe_errors_in_covered_rows": int(np.sum(covered & (safe != labels))),
        "best_scored_template_correct_rows": int(np.sum(covered & exact)),
        "safe_correct_rows_in_coverage": int(np.sum(covered & (safe == labels))),
    }


def prepare_split(name, protocol_value, fit_mask, all_ids, all_labels, all_metadata, imu, grouping):
    adjusted, safe = safe_probability_and_prediction(
        protocol_value, imu[0], imu[1], grouping
    )
    fit_sessions = build_sessions(
        np.flatnonzero(fit_mask),
        all_metadata,
        protocol_value[8].gap_seconds,
        "known_user",
    )
    templates = fit_supported_templates(
        all_labels, fit_sessions, all_metadata, MAXIMUM_LENGTH
    )
    records = session_candidates(
        adjusted,
        protocol_value[6],
        templates,
        protocol_value[7],
        protocol_value[8],
    )
    return {
        "name": name,
        "protocol": protocol_value,
        "adjusted": adjusted,
        "safe": safe,
        "records": records,
        "template_counts": {str(k): int(len(v[0])) for k, v in templates.items()},
        "oracle": oracle_audit(protocol_value[1], safe, records),
    }


def main() -> None:
    grouping = load_grouping()
    imu = load_imu()
    with np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    ) as teacher:
        all_ids = teacher["oof_sample_ids"].astype(str)
        all_labels = teacher["oof_labels"].astype(np.int64)
    all_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv",
        all_ids,
    )
    h1_protocol = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2_protocol = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1 = prepare_split(
        "H1_selection",
        h1_protocol,
        ~np.isin(all_metadata.users, full40.H1_USERS),
        all_ids,
        all_labels,
        all_metadata,
        imu,
        grouping,
    )
    h2 = prepare_split(
        "H2_confirmation",
        h2_protocol,
        ~np.isin(all_metadata.users, full40.H2_USERS),
        all_ids,
        all_labels,
        all_metadata,
        imu,
        grouping,
    )
    h3_values = h3_protocol(imu[0], imu[1], grouping)
    h3_protocol_value, h3_adjusted, h3_safe, h3_ids, h3_labels, h3_metadata, h3_fit = h3_values
    h3_fit_sessions = build_sessions(
        np.flatnonzero(h3_fit), h3_metadata, h3_protocol_value[8].gap_seconds, "known_user"
    )
    h3_templates = fit_supported_templates(
        h3_labels, h3_fit_sessions, h3_metadata, MAXIMUM_LENGTH
    )
    h3_records = session_candidates(
        h3_adjusted,
        h3_protocol_value[6],
        h3_templates,
        h3_protocol_value[7],
        h3_protocol_value[8],
    )
    h3 = {
        "name": "H3_independent_fold0",
        "protocol": h3_protocol_value,
        "adjusted": h3_adjusted,
        "safe": h3_safe,
        "records": h3_records,
        "template_counts": {
            str(k): int(len(v[0])) for k, v in h3_templates.items()
        },
        "oracle": oracle_audit(h3_protocol_value[1], h3_safe, h3_records),
    }

    configurations = [
        {
            "minimum_subject_support": support,
            "minimum_frequency": frequency,
            "maximum_loss_per_row": loss,
            "minimum_margin_per_row": margin,
        }
        for support in (1, 2, 3, 4, 5)
        for frequency in (1, 2, 3, 5, 8)
        for loss in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
        for margin in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.75, 1.5)
    ]
    candidates = []
    predictions = []
    for configuration in configurations:
        result, prediction = evaluate_configuration(
            h1_protocol[1],
            h1["safe"],
            h1["records"],
            configuration,
            h1_protocol[4],
        )
        candidates.append(result)
        predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["minimum_user_gain"],
            candidates[index]["positive_users"],
            candidates[index]["metrics"]["balanced_accuracy"],
            -candidates[index]["rescue_harm_vs_safe"]["harm"],
            -candidates[index]["gate"]["changed_rows"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    configuration = selected["configuration"]
    confirmations = {}
    saved = {
        "h1_sample_ids": h1_protocol[0],
        "h1_safe": h1["safe"],
        "h1_prediction": predictions[selected_index],
    }
    for split in (h2, h3):
        protocol_value = split["protocol"]
        result, prediction = evaluate_configuration(
            protocol_value[1],
            split["safe"],
            split["records"],
            configuration,
            protocol_value[4],
        )
        confirmations[split["name"]] = result
        prefix = "h2" if split is h2 else "h3"
        saved[f"{prefix}_sample_ids"] = protocol_value[0]
        saved[f"{prefix}_safe"] = split["safe"]
        saved[f"{prefix}_prediction"] = prediction

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "validation_predictions.npz", **saved)
    report = {
        "stage": "P89_cross_subject_supported_high_margin_template_gate_v1",
        "protocol": (
            "Fit unique action-script paths without each holdout's subjects. "
            "Select only on H1 using model-score loss/margin and distinct-subject "
            "support; transfer the frozen gate once to H2 and independent H3."
        ),
        "maximum_template_length": MAXIMUM_LENGTH,
        "grouping_configuration": asdict(grouping),
        "H1": {
            "safe_metrics": classification_metrics(h1_protocol[1], h1["safe"]),
            "template_counts": h1["template_counts"],
            "oracle": h1["oracle"],
            "selected": selected,
        },
        "H2": {
            "safe_metrics": classification_metrics(h2_protocol[1], h2["safe"]),
            "template_counts": h2["template_counts"],
            "oracle": h2["oracle"],
            "confirmation": confirmations["H2_confirmation"],
        },
        "H3": {
            "safe_metrics": classification_metrics(h3_protocol_value[1], h3["safe"]),
            "template_counts": h3["template_counts"],
            "oracle": h3["oracle"],
            "confirmation": confirmations["H3_independent_fold0"],
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
