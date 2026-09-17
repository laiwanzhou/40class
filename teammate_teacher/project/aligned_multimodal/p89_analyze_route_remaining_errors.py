from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import build_sessions


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_route_remaining_error_audit_v1"


def class_names() -> dict[int, str]:
    result = {}
    with (PROJECT_DIR / "data/manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            result[int(row["class_id"])] = row["class_name"]
    return result


def align_rows(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    return np.asarray(
        [lookup[value] for value in target_ids.astype(str)], dtype=np.int64
    )


def analyze(protocol_value, prediction: np.ndarray, expert: np.ndarray, names: dict[int, str]) -> dict:
    labels = protocol_value[1]
    errors = prediction != labels
    top = np.argmax(expert, axis=2)
    sessions = build_sessions(
        protocol_value[5],
        protocol_value[4],
        protocol_value[8].gap_seconds,
        "anonymous_date",
    )
    session_position = np.full(len(labels), -1, dtype=np.int64)
    session_length = np.zeros(len(labels), dtype=np.int64)
    for session in sessions:
        session_position[session] = np.arange(len(session))
        session_length[session] = len(session)
    confusion = Counter(
        (int(prediction[row]), int(labels[row])) for row in np.flatnonzero(errors)
    )
    classes = []
    for class_id in range(40):
        selected = labels == class_id
        classes.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "rows": int(np.sum(selected)),
                "correct": int(np.sum(prediction[selected] == labels[selected])),
                "errors": int(np.sum(selected & errors)),
            }
        )
    oracle = errors & np.any(top == labels[:, None], axis=1)
    true_votes = np.sum(top == labels[:, None], axis=1)
    current_votes = np.sum(top == prediction[:, None], axis=1)
    records = []
    for row in np.flatnonzero(errors):
        records.append(
            {
                "row": int(row),
                "sample_id": str(protocol_value[0][row]),
                "user": str(protocol_value[4].users[row]),
                "date": str(protocol_value[4].dates[row]),
                "session_position": int(session_position[row]),
                "session_length": int(session_length[row]),
                "truth": int(labels[row]),
                "truth_name": names.get(int(labels[row]), str(labels[row])),
                "prediction": int(prediction[row]),
                "prediction_name": names.get(int(prediction[row]), str(prediction[row])),
                "experts_correct": int(true_votes[row]),
                "experts_support_prediction": int(current_votes[row]),
            }
        )
    pair_table = [
        {
            "prediction": prediction_id,
            "prediction_name": names.get(prediction_id, str(prediction_id)),
            "truth": truth_id,
            "truth_name": names.get(truth_id, str(truth_id)),
            "count": count,
        }
        for (prediction_id, truth_id), count in confusion.most_common()
    ]
    by_position = defaultdict(lambda: {"rows": 0, "errors": 0})
    for row in range(len(labels)):
        key = f"{session_position[row]}/{session_length[row]}"
        by_position[key]["rows"] += 1
        by_position[key]["errors"] += int(errors[row])
    return {
        "correct": int(np.sum(~errors)),
        "errors": int(np.sum(errors)),
        "expert_oracle_recoverable_errors": int(np.sum(oracle)),
        "errors_with_zero_correct_experts": int(np.sum(errors & (true_votes == 0))),
        "top_confusions": pair_table,
        "per_class": classes,
        "by_session_position": dict(by_position),
        "records": records,
    }


def main() -> None:
    names = class_names()
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    expert_probability, expert_names = full40.train_probabilities(all_ids)
    route = np.load(
        PROJECT_DIR / "runs/p89_route_vote_gate_v1/validation_predictions.npz"
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    report = {
        "stage": "P89_route_gate_remaining_error_audit_v1",
        "expert_names": expert_names,
        "H1": analyze(
            h1,
            route["h1_prediction"],
            expert_probability[align_rows(all_ids, h1[0])],
            names,
        ),
        "H2": analyze(
            h2,
            route["h2_prediction"],
            expert_probability[align_rows(all_ids, h2[0])],
            names,
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {}
    for split in ("H1", "H2"):
        concise[split] = {
            "correct": report[split]["correct"],
            "errors": report[split]["errors"],
            "expert_oracle_recoverable_errors": report[split]["expert_oracle_recoverable_errors"],
            "errors_with_zero_correct_experts": report[split]["errors_with_zero_correct_experts"],
            "top_confusions": report[split]["top_confusions"][:15],
            "worst_classes": sorted(
                report[split]["per_class"], key=lambda item: item["errors"], reverse=True
            )[:12],
        }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
