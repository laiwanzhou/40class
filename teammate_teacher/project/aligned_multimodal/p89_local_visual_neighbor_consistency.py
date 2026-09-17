from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_build_dual_consensus_submission as submission_io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import align_metadata, classification_metrics
from p88_train_depth_residual import rescue_harm
from p89_deterministic_triple_repeat import TEST_SAFE
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import softmax


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_local_visual_neighbor_consistency_v1"
TRAIN_FEATURES = PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
TEST_FEATURES = PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz"
SAFE_VALIDATION = PROJECT_DIR / "runs/p89_imu_probability_blend_v1/validation_predictions.npz"


def unit_features(values: np.ndarray) -> np.ndarray:
    features = np.asarray(values, dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
    features = features.reshape(len(features), -1)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    return features


def align_feature_file(path: Path, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as source:
        source_ids = source["sample_ids"].astype(str)
        source_features = unit_features(source["features"])
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
    present = np.asarray([sample_id in lookup for sample_id in sample_ids.astype(str)])
    output = np.zeros((len(sample_ids), source_features.shape[1]), dtype=np.float32)
    rows = np.flatnonzero(present)
    output[rows] = source_features[
        np.asarray([lookup[sample_ids[index]] for index in rows], dtype=np.int64)
    ]
    return output, present


def local_neighbors(
    features: np.ndarray,
    present: np.ndarray,
    metadata,
    maximum_gap_seconds: float,
    neighbors: int,
    mutual: bool,
    minimum_similarity: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float | int]]:
    similarity = features @ features.T
    selected_rows: list[np.ndarray] = []
    selected_similarity: list[np.ndarray] = []
    preliminary: list[np.ndarray] = []
    for row in range(len(features)):
        if not present[row]:
            preliminary.append(np.empty(0, dtype=np.int64))
            continue
        time_gap = np.abs(metadata.starts - metadata.starts[row])
        eligible = (
            present
            & (metadata.dates == metadata.dates[row])
            & (time_gap > 30.0)
            & (time_gap <= maximum_gap_seconds)
        )
        eligible[row] = False
        candidates = np.flatnonzero(eligible)
        if len(candidates):
            order = np.argsort(-similarity[row, candidates], kind="stable")
            candidates = candidates[order[:neighbors]]
            candidates = candidates[similarity[row, candidates] >= minimum_similarity]
        preliminary.append(candidates)
    for row, candidates in enumerate(preliminary):
        if mutual and len(candidates):
            candidates = np.asarray(
                [candidate for candidate in candidates if row in preliminary[int(candidate)]],
                dtype=np.int64,
            )
        selected_rows.append(candidates)
        selected_similarity.append(similarity[row, candidates])
    covered = np.asarray([len(rows) > 0 for rows in selected_rows])
    similarities = np.concatenate(
        [values for values in selected_similarity if len(values)], axis=0
    ) if np.any(covered) else np.empty(0, dtype=np.float32)
    return selected_rows, selected_similarity, {
        "covered_rows": int(covered.sum()),
        "edges": int(sum(map(len, selected_rows))),
        "mean_similarity": float(np.mean(similarities)) if len(similarities) else 0.0,
        "minimum_similarity": float(np.min(similarities)) if len(similarities) else 0.0,
    }


def smooth(
    probability: np.ndarray,
    neighbor_rows: list[np.ndarray],
    weight: float,
) -> np.ndarray:
    output = np.asarray(probability, dtype=np.float64).copy()
    source = np.asarray(probability, dtype=np.float64)
    for row, neighbors in enumerate(neighbor_rows):
        if len(neighbors) == 0:
            continue
        nearby = source[neighbors].mean(axis=0)
        output[row] = (1.0 - weight) * source[row] + weight * nearby
        output[row] /= output[row].sum()
    return output


def decode(
    protocol_value,
    probability: np.ndarray,
    safe: np.ndarray,
    grouping: GlobalRepeatConfig,
) -> np.ndarray:
    adjusted_protocol = list(protocol_value)
    adjusted_protocol[2] = probability
    return joint_decode(
        probability,
        protocol_value[3],
        tuple(adjusted_protocol),
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
        initial_prediction=safe,
        grouping_prediction=protocol_value[3],
    )[0]


def evaluate(protocol_value, probability, safe, features, present, grouping, config):
    neighbor_rows, _, graph = local_neighbors(
        features,
        present,
        protocol_value[4],
        config["maximum_gap_seconds"],
        config["neighbors"],
        config["mutual"],
        config["minimum_similarity"],
    )
    prediction = decode(
        protocol_value,
        smooth(probability, neighbor_rows, config["weight"]),
        safe,
        grouping,
    )
    users = protocol_value[4].users.astype(str)
    per_user = {
        user: int(
            np.sum(prediction[users == user] == protocol_value[1][users == user])
            - np.sum(safe[users == user] == protocol_value[1][users == user])
        )
        for user in sorted(set(users.tolist()))
    }
    return {
        "configuration": config,
        "metrics": classification_metrics(protocol_value[1], prediction),
        "rescue_harm_vs_safe": rescue_harm(protocol_value[1], safe, prediction),
        "per_user_gain": per_user,
        "minimum_user_gain": int(min(per_user.values())),
        "graph": graph,
    }, prediction


def load_test_probability() -> tuple[np.ndarray, np.ndarray]:
    with np.load(PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz") as source:
        sample_ids = source["sample_ids"].astype(str)
        probability = np.asarray(source["base_probability"], dtype=np.float64)
    with np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz") as source:
        lookup = {sample_id: index for index, sample_id in enumerate(source["sample_ids"].astype(str))}
        rows = np.asarray([lookup[sample_id] for sample_id in sample_ids], dtype=np.int64)
        imu_probability = softmax(np.asarray(source["imu_logits"], dtype=np.float64)[rows], 3.0)
    probability = 0.95 * probability + 0.05 * imu_probability
    probability /= probability.sum(axis=1, keepdims=True)
    return sample_ids, probability


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouping_source = json.loads(
        (PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json").read_text(encoding="utf-8")
    )
    grouping = GlobalRepeatConfig(**grouping_source["H1_selected"]["configuration"])
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(SAFE_VALIDATION) as source:
        h1_safe = source["h1_prediction"].astype(np.int64)
        h2_safe = source["h2_prediction"].astype(np.int64)
    from p89_deterministic_triple_repeat import prepared_probability
    h1_probability = prepared_probability(h1)
    h2_probability = prepared_probability(h2)
    if not np.array_equal(decode(h1, h1_probability, h1_safe, grouping), h1_safe):
        raise RuntimeError("zero-neighbor path does not reproduce H1 safe prediction")
    if not np.array_equal(decode(h2, h2_probability, h2_safe, grouping), h2_safe):
        raise RuntimeError("zero-neighbor path does not reproduce H2 safe prediction")
    h1_features, h1_present = align_feature_file(TRAIN_FEATURES, h1[0])
    h2_features, h2_present = align_feature_file(TRAIN_FEATURES, h2[0])
    configurations = [
        {
            "maximum_gap_seconds": maximum_gap,
            "neighbors": neighbors,
            "mutual": mutual,
            "minimum_similarity": minimum_similarity,
            "weight": weight,
        }
        for maximum_gap in (180.0, 300.0, 600.0)
        for neighbors in (1, 2)
        for mutual in (False, True)
        for minimum_similarity in (0.70, 0.80, 0.90)
        for weight in (0.05, 0.10, 0.20)
    ]
    h1_results = []
    h1_predictions = []
    for config in configurations:
        result, prediction = evaluate(
            h1,
            h1_probability,
            h1_safe,
            h1_features,
            h1_present,
            grouping,
            config,
        )
        h1_results.append(result)
        h1_predictions.append(prediction)
    order = sorted(
        range(len(h1_results)),
        key=lambda index: (
            h1_results[index]["minimum_user_gain"] >= 0,
            h1_results[index]["rescue_harm_vs_safe"]["net"],
            -h1_results[index]["rescue_harm_vs_safe"]["harm"],
            -h1_results[index]["rescue_harm_vs_safe"]["changed"],
        ),
        reverse=True,
    )
    selected = h1_results[order[0]]
    h2_result, h2_prediction = evaluate(
        h2,
        h2_probability,
        h2_safe,
        h2_features,
        h2_present,
        grouping,
        selected["configuration"],
    )

    test_ids, test_probability = load_test_probability()
    test_safe = submission_io.read_prediction(TEST_SAFE)
    test_features, test_present = align_feature_file(TEST_FEATURES, test_ids)
    test_metadata = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv", test_ids
    )
    test_protocol = (
        test_ids, None, test_probability, test_safe, test_metadata,
        np.arange(len(test_ids), dtype=np.int64), None, h2[7], h2[8], None,
    )
    test_neighbors, _, test_graph = local_neighbors(
        test_features,
        test_present,
        test_metadata,
        selected["configuration"]["maximum_gap_seconds"],
        selected["configuration"]["neighbors"],
        selected["configuration"]["mutual"],
        selected["configuration"]["minimum_similarity"],
    )
    adjusted_test = smooth(test_probability, test_neighbors, selected["configuration"]["weight"])
    test_prediction = decode(test_protocol, adjusted_test, test_safe, grouping)
    changed = np.flatnonzero(test_prediction != test_safe)
    np.savez_compressed(
        OUTPUT / "validation_and_test_predictions.npz",
        h1_sample_ids=h1[0], h1_prediction=h1_predictions[order[0]],
        h2_sample_ids=h2[0], h2_prediction=h2_prediction,
        test_sample_ids=test_ids, test_prediction=test_prediction,
    )
    report = {
        "stage": "P89_local_multiclip_visual_neighbor_consistency_v1",
        "protocol": (
            "Within each anonymous date, use only IR VideoMAE similarity among recordings "
            "30 to N seconds apart. Select N/k/mutual/weight on H1 and transfer once to H2. "
            "No labels are used by the graph and no Test labels or LB feedback are used."
        ),
        "H1_selected": selected,
        "H2_confirmation": h2_result,
        "test": {
            "graph": test_graph,
            "changes_vs_0.85572_safe": int(len(changed)),
            "changed_ids": test_ids[changed].tolist(),
            "changed_labels": [
                {"sample_id": test_ids[index], "safe": int(test_safe[index]), "candidate": int(test_prediction[index])}
                for index in changed
            ],
        },
        "grid_size": len(configurations),
        "deployment_decision": "validation_only_no_submission_csv",
        "all_H1_candidates": [h1_results[index] for index in order],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "all_H1_candidates"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
