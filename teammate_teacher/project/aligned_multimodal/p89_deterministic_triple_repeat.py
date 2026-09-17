from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_unique_beam,
)
from p88_train_depth_residual import rescue_harm
from p89_deploy_supervised_router import load_decoder
from p89_global_repeat_decoder import date_session_lists
from p89_imu_rescue_gate import aligned_imu, softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_deterministic_triple_repeat_v1"
H1_SAFE = PROJECT_DIR / "runs/p89_imu_probability_blend_v1/validation_predictions.npz"
TEST_SAFE = PROJECT_DIR / "runs/p89_imu_probability_blend_test_v1/submission_p89_imu_probability_blend.csv"


@dataclass(frozen=True)
class TripleConfig:
    run_policy: str
    method: str
    minimum_similarity: float
    minimum_overlap: float
    consensus_weight: float
    transition_scale: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared_probability(protocol_value) -> np.ndarray:
    with np.load(
        PROJECT_DIR / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
    ) as source:
        source_ids = source["sample_ids"].astype(str)
        source_logits = np.asarray(source["imu_logits"], dtype=np.float64)
    imu_probability, present = aligned_imu(
        source_ids, source_logits, protocol_value[0], 3.0
    )
    probability = np.asarray(protocol_value[2], dtype=np.float64).copy()
    probability[present] = 0.95 * probability[present] + 0.05 * imu_probability[present]
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def triple_groups(protocol_value, run_policy: str) -> list[list[np.ndarray]]:
    if run_policy not in {"exact_three", "chunk_three"}:
        raise ValueError(run_policy)
    result = []
    for sessions in date_session_lists(
        protocol_value[5], protocol_value[4], protocol_value[8].gap_seconds
    ):
        index = 0
        while index < len(sessions):
            stop = index + 1
            while stop < len(sessions) and len(sessions[stop]) == len(sessions[index]):
                stop += 1
            run = sessions[index:stop]
            if run_policy == "exact_three" and len(run) == 3:
                result.append(run)
            elif run_policy == "chunk_three":
                for start in range(0, len(run) - 2, 3):
                    result.append(run[start : start + 3])
            index = stop
    return result


def group_evidence(
    group: list[np.ndarray], probability: np.ndarray, baseline: np.ndarray
) -> tuple[float, float]:
    stacked = np.stack([probability[session] for session in group])
    similarities = [
        float(np.sqrt(stacked[left] * stacked[right]).sum(axis=1).mean())
        for left, right in itertools.combinations(range(3), 2)
    ]
    sets = [set(map(int, baseline[session])) for session in group]
    overlaps = [
        len(sets[left] & sets[right]) / max(len(sets[left] | sets[right]), 1)
        for left, right in itertools.combinations(range(3), 2)
    ]
    return float(np.mean(similarities)), float(np.mean(overlaps))


def decode(
    protocol_value,
    probability: np.ndarray,
    baseline: np.ndarray,
    configuration: TripleConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    prediction = np.asarray(baseline, dtype=np.int64).copy()
    groups = triple_groups(protocol_value, configuration.run_policy)
    accepted_groups = accepted_sessions = accepted_rows = 0
    for group in groups:
        similarity, overlap = group_evidence(group, probability, baseline)
        if (
            similarity < configuration.minimum_similarity
            or overlap < configuration.minimum_overlap
        ):
            continue
        stacked = np.stack([probability[session] for session in group])
        if configuration.method == "shared_probability":
            aggregate = stacked.mean(axis=0)
            shared = decode_unique_beam(
                np.log(np.maximum(aggregate, 1e-12)),
                protocol_value[7],
                protocol_value[8].transition_weight * configuration.transition_scale,
                protocol_value[8].beam_width,
            )
            for session in group:
                prediction[session] = shared
        elif configuration.method == "shared_log_probability":
            aggregate = np.log(np.maximum(stacked, 1e-12)).mean(axis=0)
            shared = decode_unique_beam(
                aggregate,
                protocol_value[7],
                protocol_value[8].transition_weight * configuration.transition_scale,
                protocol_value[8].beam_width,
            )
            for session in group:
                prediction[session] = shared
        elif configuration.method == "individual_probability":
            consensus = stacked.mean(axis=0)
            for session, original in zip(group, stacked, strict=True):
                mixed = (
                    (1.0 - configuration.consensus_weight) * original
                    + configuration.consensus_weight * consensus
                )
                mixed /= mixed.sum(axis=1, keepdims=True)
                prediction[session] = decode_unique_beam(
                    np.log(np.maximum(mixed, 1e-12)),
                    protocol_value[7],
                    protocol_value[8].transition_weight * configuration.transition_scale,
                    protocol_value[8].beam_width,
                )
        else:
            raise ValueError(configuration.method)
        accepted_groups += 1
        accepted_sessions += 3
        accepted_rows += sum(map(len, group))
    return prediction, {
        "candidate_groups": len(groups),
        "accepted_groups": accepted_groups,
        "accepted_sessions": accepted_sessions,
        "accepted_rows": accepted_rows,
    }


def evaluate(protocol_value, probability, baseline, configuration) -> tuple[dict, np.ndarray]:
    prediction, grouping = decode(protocol_value, probability, baseline, configuration)
    users = protocol_value[4].users.astype(str)
    per_user = {}
    for user in sorted(set(users.tolist())):
        rows = users == user
        per_user[user] = int(
            np.sum(prediction[rows] == protocol_value[1][rows])
            - np.sum(baseline[rows] == protocol_value[1][rows])
        )
    return (
        {
            "configuration": asdict(configuration),
            "metrics": classification_metrics(protocol_value[1], prediction),
            "rescue_harm_vs_safe": rescue_harm(
                protocol_value[1], baseline, prediction
            ),
            "per_user_gain": per_user,
            "minimum_user_gain": min(per_user.values()),
            "grouping": grouping,
        },
        prediction,
    )


def configurations() -> list[TripleConfig]:
    result = []
    for run_policy in ("exact_three", "chunk_three"):
        for method in (
            "shared_probability",
            "shared_log_probability",
            "individual_probability",
        ):
            for similarity in (0.0, 0.50, 0.70, 0.80, 0.85, 0.90):
                for overlap in (0.0, 0.25, 0.50, 0.75):
                    weights = (
                        (0.25, 0.50, 0.75, 1.0)
                        if method == "individual_probability"
                        else (1.0,)
                    )
                    for weight in weights:
                        for transition_scale in (0.5, 1.0):
                            result.append(
                                TripleConfig(
                                    run_policy,
                                    method,
                                    similarity,
                                    overlap,
                                    weight,
                                    transition_scale,
                                )
                            )
    return result


def true_repeat_audit(protocol_value, configuration: TripleConfig) -> dict[str, int | float]:
    groups = triple_groups(protocol_value, configuration.run_policy)
    exact = 0
    same_user = 0
    for group in groups:
        sequences = [tuple(protocol_value[1][session].tolist()) for session in group]
        users = [str(protocol_value[4].users[session[0]]) for session in group]
        exact += int(len(set(sequences)) == 1)
        same_user += int(len(set(users)) == 1)
    return {
        "groups": len(groups),
        "exact_label_sequence_groups": exact,
        "exact_fraction": exact / max(len(groups), 1),
        "same_user_groups": same_user,
    }


def test_protocol(sample_ids: np.ndarray, probability: np.ndarray, baseline: np.ndarray):
    transition, decoder = load_decoder()
    metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv",
        sample_ids,
    )
    indices = np.arange(len(sample_ids), dtype=np.int64)
    sessions = build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date")
    return (
        sample_ids,
        None,
        probability,
        baseline,
        metadata,
        indices,
        sessions,
        transition,
        decoder,
        None,
    )


def deploy(configuration: TripleConfig) -> dict[str, object]:
    with np.load(
        PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
    ) as source:
        sample_ids = source["sample_ids"].astype(str)
        probability = np.asarray(source["base_probability"], dtype=np.float64)
    with np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz") as source:
        lookup = {
            sample_id: index
            for index, sample_id in enumerate(source["sample_ids"].astype(str))
        }
        rows = np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
        imu_probability = softmax(
            np.asarray(source["imu_logits"], dtype=np.float64)[rows], 3.0
        )
    probability = 0.95 * probability + 0.05 * imu_probability
    probability /= probability.sum(axis=1, keepdims=True)
    source_rows = submission_io.read_rows(TEST_SAFE)
    safe = submission_io.read_prediction(TEST_SAFE)
    protocol_value = test_protocol(sample_ids, probability, safe)
    direct, grouping = decode(protocol_value, probability, safe, configuration)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    direct_path = OUTPUT / "submission_p89_deterministic_triple_repeat.csv"
    submission_io.write_submission(direct_path, source_rows, direct)

    # A second file composes the separately validated missing-IR fallback without
    # changing the frozen triple rule.
    combined = direct.copy()
    manifest = read_csv(PROJECT_DIR / "data/p46_test_union_manifest.csv")
    missing_ir = {
        row["official_sample_id"]
        for row in manifest
        if row["p46_ir_readable"] == "0"
    }
    with np.load(
        PROJECT_DIR / "runs/p11_final_package/test_candidate/test_logits.npz"
    ) as p12:
        p12_lookup = {
            sample_id: index
            for index, sample_id in enumerate(p12["sample_ids"].astype(str))
        }
        p12_prediction = np.asarray(p12["routed_predictions"], dtype=np.int64)
    for index, sample_id in enumerate(sample_ids):
        if sample_id in missing_ir:
            combined[index] = p12_prediction[p12_lookup[sample_id]]
    combined_path = OUTPUT / "submission_p89_triple_repeat_missing_ir.csv"
    submission_io.write_submission(combined_path, source_rows, combined)
    return {
        "grouping": grouping,
        "direct_changes_vs_safe": int(np.sum(direct != safe)),
        "combined_changes_vs_safe": int(np.sum(combined != safe)),
        "combined_changes_vs_direct": int(np.sum(combined != direct)),
        "direct_submission": {
            "path": str(direct_path.resolve()),
            "sha256": submission_io.digest(direct_path),
        },
        "combined_submission": {
            "path": str(combined_path.resolve()),
            "sha256": submission_io.digest(combined_path),
        },
    }


def main() -> None:
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(H1_SAFE) as source:
        if not np.array_equal(source["h1_sample_ids"].astype(str), h1[0]):
            raise RuntimeError("H1 safe IDs changed")
        if not np.array_equal(source["h2_sample_ids"].astype(str), h2[0]):
            raise RuntimeError("H2 safe IDs changed")
        h1_safe = np.asarray(source["h1_prediction"], dtype=np.int64)
        h2_safe = np.asarray(source["h2_prediction"], dtype=np.int64)
    h1_probability = prepared_probability(h1)
    h2_probability = prepared_probability(h2)
    candidates = []
    h1_predictions = []
    for configuration in configurations():
        item, prediction = evaluate(h1, h1_probability, h1_safe, configuration)
        candidates.append(item)
        h1_predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["minimum_user_gain"],
            -candidates[index]["rescue_harm_vs_safe"]["harm"],
            -candidates[index]["rescue_harm_vs_safe"]["changed"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    configuration = TripleConfig(**selected["configuration"])
    confirmation, h2_prediction = evaluate(
        h2, h2_probability, h2_safe, configuration
    )
    deployment = deploy(configuration)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_prediction=h1_predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_deterministic_consecutive_triple_repeat_v1",
        "protocol": (
            "Group only consecutive equal-length sessions in triples using recording "
            "order. Select all thresholds on H1 with a no-user-regression preference; "
            "freeze and transfer unchanged to H2 and Test. Labels are used only for "
            "validation metrics, never for grouping or Test inference."
        ),
        "H1_selected": selected,
        "H2_confirmation": confirmation,
        "H1_true_repeat_audit": true_repeat_audit(h1, configuration),
        "H2_true_repeat_audit": true_repeat_audit(h2, configuration),
        "grid_size": len(candidates),
        "deployment": deployment,
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
