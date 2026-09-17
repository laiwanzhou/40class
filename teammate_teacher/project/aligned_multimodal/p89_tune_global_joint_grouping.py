from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1"
EVIDENCE_WEIGHT = 0.25
TRANSITION_SCALE = 1.0


def evaluate(protocol_value, configuration: GlobalRepeatConfig) -> tuple[dict, np.ndarray]:
    prediction, grouping = joint_decode(
        protocol_value[2],
        protocol_value[3],
        protocol_value,
        configuration,
        EVIDENCE_WEIGHT,
        TRANSITION_SCALE,
    )
    gains = []
    per_user = {}
    for user in sorted(set(protocol_value[4].users.astype(str).tolist())):
        selected = protocol_value[4].users.astype(str) == user
        base_correct = int(np.sum(protocol_value[3][selected] == protocol_value[1][selected]))
        candidate_correct = int(np.sum(prediction[selected] == protocol_value[1][selected]))
        gains.append(candidate_correct - base_correct)
        per_user[user] = {
            "p87_correct": base_correct,
            "candidate_correct": candidate_correct,
            "gain": candidate_correct - base_correct,
            "changes": int(np.sum(prediction[selected] != protocol_value[3][selected])),
        }
    return (
        {
            "configuration": asdict(configuration),
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_p87": rescue_harm(
                protocol_value[1], protocol_value[3], prediction
            ),
            "minimum_user_gain": int(min(gains)),
            "positive_users": int(np.sum(np.asarray(gains) > 0)),
            "per_user": per_user,
            "grouping": grouping,
        },
        prediction,
    )


def main() -> None:
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    configurations = [
        GlobalRepeatConfig(
            maximum_session_rank_distance=rank,
            maximum_start_gap_seconds=gap,
            minimum_probability_similarity=similarity,
            minimum_path_overlap=overlap,
            minimum_length_ratio=length_ratio,
            consensus_weight=0.5,
            alignment_gap_penalty=0.2,
            maximum_group_size=3,
        )
        for rank in (2, 3, 5, 8)
        for gap in (120.0, 180.0, 300.0, 600.0)
        for similarity in (0.78, 0.84, 0.90)
        for overlap in (0.20, 0.40, 0.60)
        for length_ratio in (0.65, 0.80, 0.90)
    ]
    candidates = []
    predictions = []
    for index, configuration in enumerate(configurations, start=1):
        item, prediction = evaluate(h1, configuration)
        candidates.append(item)
        predictions.append(prediction)
        if index % 50 == 0:
            print(f"evaluated {index}/{len(configurations)} grouping configs", flush=True)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["positive_users"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_p87"]["net"],
            -candidates[index]["rescue_harm_vs_p87"]["harm"],
            -candidates[index]["grouping"]["grouped_rows"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    selected_config = GlobalRepeatConfig(**selected["configuration"])
    confirmation, h2_prediction = evaluate(h2, selected_config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_H1_tuned_global_joint_grouping_v1",
        "protocol": (
            "Keep joint decoding weights frozen, select repeated-script grouping "
            "thresholds on H1 with a no-user-regression constraint, and transfer "
            "the selected grouping once to untouched H2."
        ),
        "joint_decoder": {
            "evidence_weight": EVIDENCE_WEIGHT,
            "transition_scale": TRANSITION_SCALE,
        },
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
