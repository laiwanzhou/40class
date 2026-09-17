from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

import p89_build_dual_consensus_submission as submission_io
import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import align_metadata, classification_metrics
from p88_train_depth_residual import rescue_harm
from p89_deterministic_triple_repeat import TEST_SAFE


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_pairwise_confusion_repair_v1"
SAFE_VALIDATION = PROJECT_DIR / "runs/p89_imu_probability_blend_v1/validation_predictions.npz"
PAIRS = ((6, 37), (17, 38), (24, 26))
PAIR_NAMES = {
    (6, 37): "drink_water_vs_take_medicine",
    (17, 38): "tap_keyboard_vs_massage",
    (24, 26): "mobile_phone_vs_play_games",
}
FEATURE_SOURCES = {
    "videomae_fullwindow": (
        PROJECT_DIR / "runs/p85_videomae_large_fullwindow_full40_v1/complete_features.npz",
        PROJECT_DIR / "runs/p85_videomae_large_fullwindow_test_v1/complete_features.npz",
        "features",
    ),
    "videomae_multiclip": (
        PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz",
        PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz",
        "features",
    ),
    "skeleton_invariant": (
        PROJECT_DIR / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz",
        PROJECT_DIR / "runs/p89_skeleton_invariant_expert_v1/test_logits.npz",
        "external_npy",
    ),
    "ir_pose_dynamics": (
        PROJECT_DIR / "runs/p89_ir_pose_dynamics_expert_v1/oof_probabilities.npz",
        PROJECT_DIR / "runs/p89_ir_pose_dynamics_expert_v1/test_probabilities.npz",
        "external_npy",
    ),
}


def flatten_unit(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), 1e-8)
    output = output.reshape(len(output), -1)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-8)
    return output


def load_source(name: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path, test_path, key = FEATURE_SOURCES[name]
    path = train_path if split == "train" else test_path
    with np.load(path) as source:
        sample_ids = source["sample_ids"].astype(str)
        present = source["present"].astype(bool) if "present" in source.files else np.ones(len(sample_ids), dtype=bool)
        if key != "external_npy":
            features = flatten_unit(source[key])
            return sample_ids, features, present
    if name == "skeleton_invariant":
        feature_path = PROJECT_DIR / "runs/p89_skeleton_invariant_expert_v1" / (
            "train_features.npy" if split == "train" else "test_features.npy"
        )
    elif name == "ir_pose_dynamics":
        feature_path = PROJECT_DIR / "runs/p89_ir_pose_dynamics_expert_v1" / (
            "train_features.npy" if split == "train" else "test_features.npy"
        )
        explicit_present = feature_path.with_suffix(".present.npy")
        if explicit_present.exists():
            present = np.load(explicit_present).astype(bool)
    else:
        raise ValueError(name)
    features = np.load(feature_path, mmap_mode="r")
    if len(features) != len(sample_ids):
        raise RuntimeError(f"{name} feature/id alignment changed")
    return sample_ids, features, present


def aligned_rows(source_ids: np.ndarray, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids.astype(str))}
    present = np.asarray([sample_id in lookup for sample_id in target_ids.astype(str)])
    rows = np.full(len(target_ids), -1, dtype=np.int64)
    positions = np.flatnonzero(present)
    rows[positions] = np.asarray([lookup[target_ids[index]] for index in positions], dtype=np.int64)
    return rows, present


def fit_pair(
    features: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    excluded_users: list[str],
    pair: tuple[int, int],
    alpha: float,
) -> tuple[StandardScaler, RidgeClassifier]:
    fit = (~np.isin(users, excluded_users)) & np.isin(labels, pair)
    scaler = StandardScaler()
    values = scaler.fit_transform(np.asarray(features[fit], dtype=np.float32))
    estimator = RidgeClassifier(alpha=alpha, class_weight="balanced", solver="lsqr")
    estimator.fit(values, labels[fit])
    return scaler, estimator


def repair(
    baseline: np.ndarray,
    target_features: np.ndarray,
    target_present: np.ndarray,
    scaler: StandardScaler,
    estimator: RidgeClassifier,
    pair: tuple[int, int],
    minimum_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = baseline.copy()
    eligible = np.isin(baseline, pair) & target_present
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return prediction, np.zeros(len(baseline), dtype=bool), np.zeros(len(baseline))
    transformed = scaler.transform(np.asarray(target_features[rows], dtype=np.float32))
    candidate = estimator.predict(transformed).astype(np.int64)
    decision = np.abs(np.asarray(estimator.decision_function(transformed), dtype=np.float64))
    accepted_local = (candidate != baseline[rows]) & (decision >= minimum_margin)
    accepted = np.zeros(len(baseline), dtype=bool)
    accepted[rows[accepted_local]] = True
    prediction[rows[accepted_local]] = candidate[accepted_local]
    margin = np.zeros(len(baseline), dtype=np.float64)
    margin[rows] = decision
    return prediction, accepted, margin


def result(protocol_value, baseline, prediction, accepted, pair):
    labels = protocol_value[1]
    users = protocol_value[4].users.astype(str)
    true_pair = np.isin(labels, pair)
    per_user = {
        user: int(
            np.sum(prediction[users == user] == labels[users == user])
            - np.sum(baseline[users == user] == labels[users == user])
        )
        for user in sorted(set(users.tolist()))
    }
    pair_rows = int(true_pair.sum())
    return {
        "metrics": classification_metrics(labels, prediction),
        "rescue_harm_vs_safe": rescue_harm(labels, baseline, prediction),
        "accepted": int(accepted.sum()),
        "true_pair_rows": pair_rows,
        "true_pair_accuracy": (
            float(np.mean(prediction[true_pair] == labels[true_pair])) if pair_rows else None
        ),
        "safe_true_pair_accuracy": (
            float(np.mean(baseline[true_pair] == labels[true_pair])) if pair_rows else None
        ),
        "per_user_gain": per_user,
        "minimum_user_gain": int(min(per_user.values())),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    teacher = np.load(PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz")
    all_ids = teacher["oof_sample_ids"].astype(str)
    all_labels = teacher["oof_labels"].astype(np.int64)
    all_users = align_metadata(
        PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv", all_ids
    ).users.astype(str)
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(SAFE_VALIDATION) as source:
        h1_safe = source["h1_prediction"].astype(np.int64)
        h2_safe = source["h2_prediction"].astype(np.int64)

    selected = {}
    h1_prediction = h1_safe.copy()
    h2_prediction = h2_safe.copy()
    pair_reports = {}
    test_payload = {}
    test_ids = None
    test_safe = submission_io.read_prediction(TEST_SAFE)
    for pair in PAIRS:
        candidates = []
        cached_train = {}
        for source_name in FEATURE_SOURCES:
            source_ids, features, source_present = load_source(source_name, "train")
            rows_h1, present_h1 = aligned_rows(source_ids, h1[0])
            present_h1 &= source_present[np.maximum(rows_h1, 0)]
            target_h1 = np.zeros((len(h1[0]), features.shape[1]), dtype=np.float32)
            valid_h1 = np.flatnonzero(present_h1)
            target_h1[valid_h1] = features[rows_h1[valid_h1]]
            all_rows, all_present = aligned_rows(source_ids, all_ids)
            if not np.all(all_present):
                # IMU-like partial sources are intentionally not in this v1 source list.
                raise RuntimeError(f"{source_name} misses Train rows")
            ordered_features = features[all_rows]
            cached_train[source_name] = (ordered_features, source_present[all_rows])
            for alpha in (1.0, 10.0, 100.0, 1000.0):
                scaler, estimator = fit_pair(
                    ordered_features, all_labels, all_users, full40.H1_USERS, pair, alpha
                )
                for minimum_margin in (0.0, 0.25, 0.50, 0.75):
                    prediction, accepted, _ = repair(
                        h1_safe,
                        target_h1,
                        present_h1,
                        scaler,
                        estimator,
                        pair,
                        minimum_margin,
                    )
                    item = result(h1, h1_safe, prediction, accepted, pair)
                    item["configuration"] = {
                        "feature_source": source_name,
                        "alpha": alpha,
                        "minimum_margin": minimum_margin,
                    }
                    candidates.append(item)
        candidates.sort(
            key=lambda item: (
                item["minimum_user_gain"] >= 0,
                item["rescue_harm_vs_safe"]["net"],
                item["true_pair_accuracy"],
                -item["rescue_harm_vs_safe"]["harm"],
                -item["accepted"],
            ),
            reverse=True,
        )
        chosen = candidates[0]
        config = chosen["configuration"]
        selected[pair] = config
        source_name = config["feature_source"]
        ordered_features, _ = cached_train[source_name]
        scaler_h1, estimator_h1 = fit_pair(
            ordered_features, all_labels, all_users, full40.H1_USERS, pair, config["alpha"]
        )
        source_ids, features, source_present = load_source(source_name, "train")
        rows_h1, present_h1 = aligned_rows(source_ids, h1[0])
        present_h1 &= source_present[np.maximum(rows_h1, 0)]
        target_h1 = np.zeros((len(h1[0]), features.shape[1]), dtype=np.float32)
        valid_h1 = np.flatnonzero(present_h1)
        target_h1[valid_h1] = features[rows_h1[valid_h1]]
        proposal_h1, accepted_h1, _ = repair(
            h1_safe, target_h1, present_h1, scaler_h1, estimator_h1, pair, config["minimum_margin"]
        )
        h1_prediction[accepted_h1] = proposal_h1[accepted_h1]

        scaler_h2, estimator_h2 = fit_pair(
            ordered_features, all_labels, all_users, full40.H2_USERS, pair, config["alpha"]
        )
        rows_h2, present_h2 = aligned_rows(source_ids, h2[0])
        present_h2 &= source_present[np.maximum(rows_h2, 0)]
        target_h2 = np.zeros((len(h2[0]), features.shape[1]), dtype=np.float32)
        valid_h2 = np.flatnonzero(present_h2)
        target_h2[valid_h2] = features[rows_h2[valid_h2]]
        proposal_h2, accepted_h2, _ = repair(
            h2_safe, target_h2, present_h2, scaler_h2, estimator_h2, pair, config["minimum_margin"]
        )
        h2_prediction[accepted_h2] = proposal_h2[accepted_h2]

        test_source_ids, test_features, test_present_source = load_source(source_name, "test")
        if test_ids is None:
            test_ids = np.load(
                PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2/test_probabilities.npz"
            )["sample_ids"].astype(str)
        test_rows, test_present = aligned_rows(test_source_ids, test_ids)
        test_present &= test_present_source[np.maximum(test_rows, 0)]
        target_test = np.zeros((len(test_ids), test_features.shape[1]), dtype=np.float32)
        valid_test = np.flatnonzero(test_present)
        target_test[valid_test] = test_features[test_rows[valid_test]]
        scaler_test = StandardScaler()
        pair_fit = np.isin(all_labels, pair)
        train_scaled = scaler_test.fit_transform(np.asarray(ordered_features[pair_fit], dtype=np.float32))
        estimator_test = RidgeClassifier(
            alpha=config["alpha"], class_weight="balanced", solver="lsqr"
        ).fit(train_scaled, all_labels[pair_fit])
        proposal_test, accepted_test, margin_test = repair(
            test_safe, target_test, test_present, scaler_test, estimator_test, pair, config["minimum_margin"]
        )
        test_payload[pair] = (proposal_test, accepted_test, margin_test)
        pair_reports[PAIR_NAMES[pair]] = {
            "pair": list(pair),
            "H1_selected": chosen,
            "H2_confirmation": result(h2, h2_safe, proposal_h2, accepted_h2, pair),
            "test_changes": int(accepted_test.sum()),
        }

    assert test_ids is not None
    test_prediction = test_safe.copy()
    for pair in PAIRS:
        proposal, accepted, _ = test_payload[pair]
        test_prediction[accepted] = proposal[accepted]
    h1_accepted = h1_prediction != h1_safe
    h2_accepted = h2_prediction != h2_safe
    test_changed = np.flatnonzero(test_prediction != test_safe)
    report = {
        "stage": "P89_subject_disjoint_pairwise_confusion_repair_v1",
        "protocol": (
            "Pairs are the three non-overlapping confusion families selected from H1 errors. "
            "For each pair, H1 selects one frozen pretrained feature source, Ridge strength, "
            "and margin under subject exclusion; H2 refits the frozen configuration while "
            "excluding all H2 users. Test fits on all labeled Train rows. No Test labels or LB feedback."
        ),
        "pairs": pair_reports,
        "H1_combined": result(h1, h1_safe, h1_prediction, h1_accepted, (-1, -1)),
        "H2_combined_confirmation": result(h2, h2_safe, h2_prediction, h2_accepted, (-1, -1)),
        "test": {
            "changes_vs_0.85572_safe": int(len(test_changed)),
            "changed_ids": test_ids[test_changed].tolist(),
            "changed_labels": [
                {"sample_id": test_ids[index], "safe": int(test_safe[index]), "candidate": int(test_prediction[index])}
                for index in test_changed
            ],
        },
        "deployment_decision": "validation_only_no_submission_csv",
    }
    np.savez_compressed(
        OUTPUT / "validation_and_test_predictions.npz",
        h1_sample_ids=h1[0], h1_prediction=h1_prediction,
        h2_sample_ids=h2[0], h2_prediction=h2_prediction,
        test_sample_ids=test_ids, test_prediction=test_prediction,
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
