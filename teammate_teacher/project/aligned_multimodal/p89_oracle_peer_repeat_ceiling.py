from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics, decode_unique_beam
from p88_aligned_repeat_holdout import align_probabilities
from p88_train_depth_residual import rescue_harm
from p89_deterministic_triple_repeat import prepared_probability
from p89_oracle_trial_repeat_audit import oracle_trial_groups


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_oracle_peer_repeat_ceiling_v1"
SAFE_VALIDATION = PROJECT_DIR / "runs/p89_imu_probability_blend_v1/validation_predictions.npz"


def aligned_columns(
    group: list[np.ndarray], probability: np.ndarray
) -> list[list[int]]:
    reference = max(group, key=len)
    columns: list[list[int]] = [[int(row)] for row in reference]
    for take in group:
        if take is reference:
            continue
        pairs, _ = align_probabilities(
            probability[reference], probability[take], gap_penalty=0.2
        )
        for reference_position, other_position in pairs:
            columns[reference_position].append(int(take[other_position]))
    return columns


def decode(
    protocol_value,
    probability: np.ndarray,
    baseline: np.ndarray,
    method: str,
    minimum_column_size: int,
    minimum_peer_support: int,
) -> tuple[np.ndarray, dict[str, int]]:
    prediction = np.asarray(baseline, dtype=np.int64).copy()
    groups = oracle_trial_groups(protocol_value[0], protocol_value[4])
    eligible_columns = supported_columns = changed_rows = 0
    for group in groups:
        columns = aligned_columns(group, probability)
        aggregate = np.stack(
            [probability[np.asarray(rows, dtype=np.int64)].mean(axis=0) for rows in columns]
        )
        if method == "shared_path":
            candidate_path = decode_unique_beam(
                np.log(np.maximum(aggregate, 1e-12)),
                protocol_value[7],
                protocol_value[8].transition_weight,
                protocol_value[8].beam_width,
            )
        elif method == "column_argmax":
            candidate_path = np.argmax(aggregate, axis=1)
        else:
            raise ValueError(method)
        for rows_list, candidate in zip(columns, candidate_path, strict=True):
            rows = np.asarray(rows_list, dtype=np.int64)
            if len(rows) < minimum_column_size:
                continue
            eligible_columns += 1
            if int(np.sum(baseline[rows] == candidate)) < minimum_peer_support:
                continue
            supported_columns += 1
            changed_rows += int(np.sum(prediction[rows] != candidate))
            prediction[rows] = int(candidate)
    return prediction, {
        "oracle_groups": len(groups),
        "eligible_columns": eligible_columns,
        "peer_supported_columns": supported_columns,
        "changed_rows_with_duplicates": changed_rows,
    }


def evaluate(
    protocol_value,
    probability,
    baseline,
    method,
    minimum_column_size,
    minimum_peer_support,
):
    prediction, grouping = decode(
        protocol_value,
        probability,
        baseline,
        method,
        minimum_column_size,
        minimum_peer_support,
    )
    users = protocol_value[4].users.astype(str)
    per_user = {
        user: int(
            np.sum(prediction[users == user] == protocol_value[1][users == user])
            - np.sum(baseline[users == user] == protocol_value[1][users == user])
        )
        for user in sorted(set(users.tolist()))
    }
    return {
        "configuration": {
            "method": method,
            "minimum_column_size": minimum_column_size,
            "minimum_peer_support": minimum_peer_support,
        },
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_safe": rescue_harm(protocol_value[1], baseline, prediction),
        "per_user_gain": per_user,
        "minimum_user_gain": int(min(per_user.values())),
        "grouping": grouping,
    }, prediction


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(SAFE_VALIDATION) as source:
        if not np.array_equal(source["h1_sample_ids"].astype(str), h1[0]):
            raise RuntimeError("H1 alignment changed")
        if not np.array_equal(source["h2_sample_ids"].astype(str), h2[0]):
            raise RuntimeError("H2 alignment changed")
        h1_safe = source["h1_prediction"].astype(np.int64)
        h2_safe = source["h2_prediction"].astype(np.int64)
    h1_probability = prepared_probability(h1)
    h2_probability = prepared_probability(h2)
    results = []
    predictions = []
    for method in ("column_argmax", "shared_path"):
        for minimum_column_size in (2, 3):
            for minimum_peer_support in (1, 2):
                h1_report, h1_prediction = evaluate(
                    h1,
                    h1_probability,
                    h1_safe,
                    method,
                    minimum_column_size,
                    minimum_peer_support,
                )
                h2_report, h2_prediction = evaluate(
                    h2,
                    h2_probability,
                    h2_safe,
                    method,
                    minimum_column_size,
                    minimum_peer_support,
                )
                results.append({"H1": h1_report, "H2_confirmation": h2_report})
                predictions.append((h1_prediction, h2_prediction))
    order = sorted(
        range(len(results)),
        key=lambda index: (
            results[index]["H1"]["minimum_user_gain"] >= 0,
            results[index]["H1"]["rescue_harm_vs_safe"]["net"],
            -results[index]["H1"]["rescue_harm_vs_safe"]["harm"],
        ),
        reverse=True,
    )
    selected = order[0]
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=predictions[selected][0],
        h2_prediction=predictions[selected][1],
    )
    report = {
        "stage": "P89_oracle_peer_supported_repeat_ceiling_v1",
        "warning": (
            "Train-only oracle grouping by trial-id suffix. This establishes whether a "
            "safe anonymous grouping model is worth building; it must never generate Test predictions."
        ),
        "selected_on_H1": results[selected],
        "all_variants": [results[index] for index in order],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
