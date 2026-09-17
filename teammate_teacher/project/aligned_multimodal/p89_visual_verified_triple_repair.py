from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import (
    DecoderConfig,
    align_metadata,
    build_sessions,
    classification_metrics,
    decode_unique_beam,
    fit_transition_model,
)
from p88_train_depth_residual import rescue_harm
from p89_deterministic_triple_repeat import prepared_probability, test_protocol, triple_groups
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import aligned_imu
from p89_local_visual_neighbor_consistency import (
    TRAIN_FEATURES,
    TEST_FEATURES,
    align_feature_file,
)
from p89_peer_supported_triple_repair import SAFE_VALIDATION, test_probability


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_visual_verified_triple_repair_v1"


def build_h3():
    with np.load(PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz") as source:
        all_ids = source["oof_sample_ids"].astype(str)
        all_labels = source["oof_labels"].astype(np.int64)
        folds = source["oof_folds"].astype(np.int64)
        all_probability = source["oof_teacher_probability"].astype(np.float64)
    all_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv", all_ids
    )
    summary = json.loads(
        (PROJECT_DIR / "runs/p87_sequence_decoder_v1/summary.json").read_text(encoding="utf-8")
    )
    fold_summary = next(item for item in summary["folds"] if int(item["fold"]) == 0)
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
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv", sample_ids
    )
    with np.load(PROJECT_DIR / "runs/p87_sequence_decoder_v1/oof_predictions.npz") as source:
        p87 = source["sequence_predictions"].astype(np.int64)[validation]
    with np.load(PROJECT_DIR / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz") as source:
        imu_probability, present = aligned_imu(
            source["sample_ids"].astype(str),
            np.asarray(source["imu_logits"], dtype=np.float64),
            sample_ids,
            3.0,
        )
    probability[present] = 0.95 * probability[present] + 0.05 * imu_probability[present]
    probability /= probability.sum(axis=1, keepdims=True)
    indices = np.arange(len(sample_ids), dtype=np.int64)
    protocol = (
        sample_ids,
        labels,
        probability,
        p87,
        metadata,
        indices,
        build_sessions(indices, metadata, decoder.gap_seconds, "anonymous_date"),
        transition,
        decoder,
        None,
    )
    grouping_summary = json.loads(
        (PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json").read_text(encoding="utf-8")
    )
    grouping = GlobalRepeatConfig(**grouping_summary["H1_selected"]["configuration"])
    safe = joint_decode(
        probability,
        p87,
        protocol,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )[0]
    return protocol, probability, safe


def decode(
    protocol_value,
    probability: np.ndarray,
    baseline: np.ndarray,
    features: np.ndarray,
    present: np.ndarray,
    gate_source: str,
    gate_kind: str,
    threshold: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    prediction = baseline.copy()
    groups = triple_groups(protocol_value, "exact_three")
    visual_positions = supported_positions = changed_rows = 0
    accepted_similarities = []
    for group in groups:
        aggregate = np.stack([probability[session] for session in group]).mean(axis=0)
        shared = decode_unique_beam(
            np.log(np.maximum(aggregate, 1e-12)),
            protocol_value[7],
            protocol_value[8].transition_weight,
            protocol_value[8].beam_width,
        )
        for position, candidate in enumerate(shared):
            rows = np.asarray([session[position] for session in group], dtype=np.int64)
            if not np.all(present[rows]):
                continue
            similarities = np.asarray(
                [float(features[left] @ features[right]) for left, right in combinations(rows, 2)],
                dtype=np.float64,
            )
            probability_similarities = np.asarray(
                [
                    float(np.sqrt(probability[left] * probability[right]).sum())
                    for left, right in combinations(rows, 2)
                ],
                dtype=np.float64,
            )
            reduce = np.min if gate_kind == "minimum" else np.mean
            visual_score = float(reduce(similarities))
            probability_score = float(reduce(probability_similarities))
            if gate_source == "visual":
                score = visual_score
            elif gate_source == "probability":
                score = probability_score
            elif gate_source == "combined":
                score = min(visual_score, probability_score)
            else:
                raise ValueError(gate_source)
            if score < threshold:
                continue
            visual_positions += 1
            accepted_similarities.append(score)
            if not np.any(baseline[rows] == candidate):
                continue
            supported_positions += 1
            changed_rows += int(np.sum(prediction[rows] != candidate))
            prediction[rows] = int(candidate)
    return prediction, {
        "candidate_groups": len(groups),
        "visual_verified_positions": visual_positions,
        "peer_supported_positions": supported_positions,
        "changed_rows_with_duplicates": changed_rows,
        "mean_accepted_similarity": float(np.mean(accepted_similarities)) if accepted_similarities else 0.0,
    }


def evaluate(protocol_value, probability, baseline, features, present, config):
    prediction, grouping = decode(
        protocol_value,
        probability,
        baseline,
        features,
        present,
        config["gate_source"],
        config["gate_kind"],
        config["threshold"],
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
        "configuration": config,
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
        h1_safe = source["h1_prediction"].astype(np.int64)
        h2_safe = source["h2_prediction"].astype(np.int64)
    h1_probability = prepared_probability(h1)
    h2_probability = prepared_probability(h2)
    h3, h3_probability, h3_safe = build_h3()
    h1_features, h1_present = align_feature_file(TRAIN_FEATURES, h1[0])
    h2_features, h2_present = align_feature_file(TRAIN_FEATURES, h2[0])
    h3_features, h3_present = align_feature_file(TRAIN_FEATURES, h3[0])
    protocols = (
        ("H1", h1, h1_probability, h1_safe, h1_features, h1_present),
        ("H2", h2, h2_probability, h2_safe, h2_features, h2_present),
        ("H3", h3, h3_probability, h3_safe, h3_features, h3_present),
    )
    candidates = []
    predictions = []
    for gate_source in ("visual", "probability", "combined"):
        for gate_kind in ("minimum", "mean"):
            for threshold in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
                config = {"gate_source": gate_source, "gate_kind": gate_kind, "threshold": threshold}
                reports = {}
                values = {}
                for name, protocol_value, probability, safe, features, present in protocols:
                    reports[name], values[name] = evaluate(
                        protocol_value, probability, safe, features, present, config
                    )
                candidates.append({"configuration": config, "splits": reports})
                predictions.append(values)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            min(candidates[index]["splits"][name]["minimum_user_gain"] for name in ("H1", "H2", "H3")) >= 0,
            min(candidates[index]["splits"][name]["rescue_harm_vs_safe"]["net"] for name in ("H1", "H2", "H3")),
            sum(candidates[index]["splits"][name]["rescue_harm_vs_safe"]["net"] for name in ("H1", "H2", "H3")),
            -sum(candidates[index]["splits"][name]["rescue_harm_vs_safe"]["harm"] for name in ("H1", "H2", "H3")),
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]

    test_ids, test_probability_value, test_safe = test_probability()
    test = test_protocol(test_ids, test_probability_value, test_safe)
    test_features, test_present = align_feature_file(TEST_FEATURES, test_ids)
    test_prediction, test_grouping = decode(
        test,
        test_probability_value,
        test_safe,
        test_features,
        test_present,
        selected["configuration"]["gate_source"],
        selected["configuration"]["gate_kind"],
        selected["configuration"]["threshold"],
    )
    changed = np.flatnonzero(test_prediction != test_safe)
    eligible = (
        min(selected["splits"][name]["minimum_user_gain"] for name in ("H1", "H2", "H3")) >= 0
        and min(selected["splits"][name]["rescue_harm_vs_safe"]["net"] for name in ("H1", "H2", "H3")) > 0
    )
    np.savez_compressed(
        OUTPUT / "validation_and_test_predictions.npz",
        h1_sample_ids=h1[0], h1_prediction=predictions[selected_index]["H1"],
        h2_sample_ids=h2[0], h2_prediction=predictions[selected_index]["H2"],
        h3_sample_ids=h3[0], h3_prediction=predictions[selected_index]["H3"],
        test_sample_ids=test_ids, test_prediction=test_prediction,
    )
    report = {
        "stage": "P89_visual_verified_peer_supported_exact_triple_v1",
        "protocol": (
            "Repair only exact-three equal-length takes and only at positions whose three "
            "frozen IR VideoMAE embeddings pass a pairwise similarity gate. Threshold/gate "
            "kind are selected jointly over three subject-disjoint Train partitions (15 users). "
            "No Test labels or leaderboard feedback are used."
        ),
        "selected": selected,
        "test": {
            "grouping": test_grouping,
            "changes_vs_0.85572_safe": int(len(changed)),
            "changed_ids": test_ids[changed].tolist(),
            "changed_labels": [
                {"sample_id": test_ids[index], "safe": int(test_safe[index]), "candidate": int(test_prediction[index])}
                for index in changed
            ],
        },
        "deployment_decision": "eligible_no_submission_csv" if eligible else "reject",
        "all_candidates": [candidates[index] for index in order],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "all_candidates"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
