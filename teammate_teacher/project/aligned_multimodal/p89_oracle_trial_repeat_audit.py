from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics, decode_unique_beam
from p88_aligned_repeat_holdout import align_probabilities
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_oracle_trial_repeat_audit_v1"
TRIAL_PATTERN = re.compile(r"__(\d+)-(\d+)-(\d+)$")


def oracle_trial_groups(sample_ids: np.ndarray, metadata) -> list[list[np.ndarray]]:
    grouped: dict[tuple[str, int, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, sample_id in enumerate(sample_ids.astype(str)):
        match = TRIAL_PATTERN.search(sample_id)
        if match is None:
            raise ValueError(f"sample id has no trial suffix: {sample_id}")
        script, block, repeat = map(int, match.groups())
        grouped[(str(metadata.users[index]), script, block)][repeat].append(index)

    result: list[list[np.ndarray]] = []
    for takes in grouped.values():
        if len(takes) < 2:
            continue
        ordered_takes = []
        for repeat in sorted(takes):
            indices = np.asarray(takes[repeat], dtype=np.int64)
            order = np.argsort(metadata.starts[indices], kind="stable")
            ordered_takes.append(indices[order])
        result.append(ordered_takes)
    return result


def decode_oracle(protocol_value, evidence_weight: float, transition_scale: float):
    sample_ids, _, probability, p87, metadata = protocol_value[:5]
    transition, decoder = protocol_value[7:9]
    prediction = np.asarray(p87, dtype=np.int64).copy()
    log_probability = np.log(np.maximum(probability, 1e-12))
    groups = oracle_trial_groups(sample_ids, metadata)
    aligned_pairs = assigned_rows = 0
    for group in groups:
        reference = max(group, key=len)
        columns: list[list[int]] = [[int(index)] for index in reference]
        for take in group:
            if take is reference:
                continue
            pairs, _ = align_probabilities(
                probability[reference], probability[take], gap_penalty=0.2
            )
            for reference_position, other_position in pairs:
                columns[reference_position].append(int(take[other_position]))
            aligned_pairs += len(pairs)
        aggregate = np.empty((len(columns), probability.shape[1]), dtype=np.float64)
        for position, rows in enumerate(columns):
            reference_logp = log_probability[rows[0]]
            if len(rows) == 1:
                aggregate[position] = reference_logp
            else:
                other_logp = log_probability[np.asarray(rows[1:])].mean(axis=0)
                aggregate[position] = (
                    reference_logp + evidence_weight * other_logp
                ) / (1.0 + evidence_weight)
        shared_path = decode_unique_beam(
            aggregate,
            transition,
            transition_weight=decoder.transition_weight * transition_scale,
            beam_width=decoder.beam_width,
        )
        for position, rows in enumerate(columns):
            prediction[np.asarray(rows, dtype=np.int64)] = int(shared_path[position])
            assigned_rows += len(rows)
    return prediction, {
        "groups": len(groups),
        "takes": int(sum(map(len, groups))),
        "rows": int(sum(len(take) for group in groups for take in group)),
        "aligned_pairs": aligned_pairs,
        "assigned_rows_with_duplicates": assigned_rows,
    }


def evaluate(protocol_value, evidence_weight: float, transition_scale: float):
    prediction, grouping = decode_oracle(
        protocol_value, evidence_weight=evidence_weight, transition_scale=transition_scale
    )
    return {
        "configuration": {
            "evidence_weight": evidence_weight,
            "transition_scale": transition_scale,
        },
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_p87": rescue_harm(
            protocol_value[1], protocol_value[3], prediction
        ),
        "grouping": grouping,
    }, prediction


def main() -> None:
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    candidates = []
    predictions = []
    for evidence_weight in (0.05, 0.10, 0.25, 0.50, 1.0, 2.0):
        for transition_scale in (0.0, 0.5, 1.0, 1.5, 2.0):
            item, prediction = evaluate(h1, evidence_weight, transition_scale)
            candidates.append(item)
            predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["metrics"]["correct"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_p87"]["net"],
            -candidates[index]["rescue_harm_vs_p87"]["harm"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    confirmation, h2_prediction = evaluate(h2, **selected["configuration"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_oracle_trial_id_repeat_grouping_audit_v1",
        "warning": (
            "Oracle ceiling only: train sample-id trial suffixes are unavailable on Test. "
            "No submission may be generated from this grouping."
        ),
        "H1_selected": selected,
        "H2_confirmation": confirmation,
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
