from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_joint_evidence_audit_v1"


def align_rows(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    return np.asarray(
        [lookup[value] for value in target_ids.astype(str)], dtype=np.int64
    )


def analyze(protocol_value, prediction: np.ndarray, expert: np.ndarray) -> dict:
    labels = protocol_value[1]
    base = protocol_value[3]
    base_probability = protocol_value[2]
    changed = prediction != base
    top = np.argmax(expert, axis=2)
    rows = np.arange(len(labels))
    candidate_votes = np.sum(top == prediction[:, None], axis=1)
    base_votes = np.sum(top == base[:, None], axis=1)
    preference_count = np.sum(
        expert[rows[:, None], np.arange(expert.shape[1])[None, :], prediction[:, None]]
        > expert[rows[:, None], np.arange(expert.shape[1])[None, :], base[:, None]],
        axis=1,
    )
    expert_delta = np.mean(
        expert[rows[:, None], np.arange(expert.shape[1])[None, :], prediction[:, None]]
        - expert[rows[:, None], np.arange(expert.shape[1])[None, :], base[:, None]],
        axis=1,
    )
    base_delta = base_probability[rows, prediction] - base_probability[rows, base]
    status = np.full(len(labels), "unchanged", dtype=object)
    status[changed & (prediction == labels)] = "rescue"
    status[changed & (base == labels)] = "harm"
    status[changed & (prediction != labels) & (base != labels)] = "wrong_to_wrong"
    records = []
    for row in np.flatnonzero(changed):
        records.append(
            {
                "row": int(row),
                "sample_id": str(protocol_value[0][row]),
                "user": str(protocol_value[4].users[row]),
                "label": int(labels[row]),
                "p87": int(base[row]),
                "candidate": int(prediction[row]),
                "status": str(status[row]),
                "candidate_votes": int(candidate_votes[row]),
                "base_votes": int(base_votes[row]),
                "vote_margin": int(candidate_votes[row] - base_votes[row]),
                "preference_count": int(preference_count[row]),
                "expert_delta": float(expert_delta[row]),
                "p87_probability_delta": float(base_delta[row]),
            }
        )
    vote_table = defaultdict(lambda: defaultdict(int))
    preference_table = defaultdict(lambda: defaultdict(int))
    for record in records:
        vote_table[str(record["candidate_votes"])][record["status"]] += 1
        preference_table[str(record["preference_count"])][record["status"]] += 1
    return {
        "changed": len(records),
        "vote_table": {key: dict(value) for key, value in sorted(vote_table.items(), key=lambda item: int(item[0]))},
        "preference_table": {key: dict(value) for key, value in sorted(preference_table.items(), key=lambda item: int(item[0]))},
        "records": records,
    }


def main() -> None:
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    all_ids = teacher["oof_sample_ids"].astype(str)
    expert_probability, names = full40.train_probabilities(all_ids)
    joint = np.load(
        PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/validation_predictions.npz"
    )
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_prediction = joint["h1_prediction"]
    h2_prediction = joint["h2_prediction"]
    h1_expert = expert_probability[align_rows(all_ids, h1[0])]
    h2_expert = expert_probability[align_rows(all_ids, h2[0])]
    report = {
        "stage": "P89_shared_path_correction_evidence_audit_v1",
        "expert_names": names,
        "H1": analyze(h1, h1_prediction, h1_expert),
        "H2": analyze(h2, h2_prediction, h2_expert),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {
        split: {
            "changed": report[split]["changed"],
            "vote_table": report[split]["vote_table"],
            "preference_table": report[split]["preference_table"],
        }
        for split in ("H1", "H2")
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
