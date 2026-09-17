from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import align_metadata, classification_metrics
from p88_train_depth_residual import log_softmax_numpy, rescue_harm
from p89_deploy_cohort_prior import test_groups
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import aligned_imu
from train_p46_videomae_head import l2_normalize, make_model
from train_p85_videomae_full40_head import (
    aligned_scores_40,
    sample_weights,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_cohort_centered_videomae_v1"
FEATURES = PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
IMU = PROJECT_DIR / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"


def source_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(FEATURES, allow_pickle=False) as source:
        sample_ids = source["sample_ids"].astype(str)
        users = source["users"].astype(str)
        labels = source["labels"].astype(np.int64)
        views = l2_normalize(np.asarray(source["features"], dtype=np.float32))
    # A compact action feature that does not privilege scene/person/workspace.
    values = l2_normalize(views.mean(axis=(1, 2)))
    return sample_ids, users, labels, values


def centered_scores(target_ids: np.ndarray, holdout_users: list[str]) -> np.ndarray:
    sample_ids, users, labels, values = source_features()
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    target = np.asarray([lookup[sample_id] for sample_id in target_ids], dtype=np.int64)
    fit = ~np.isin(users, holdout_users)
    if np.any(fit[target]):
        raise RuntimeError("cohort-centred expert train/evaluation overlap")
    global_center = values[fit].mean(axis=0)
    corrected = values.copy()
    for user in sorted(set(users[fit].tolist())):
        rows = fit & (users == user)
        corrected[rows] -= values[rows].mean(axis=0) - global_center

    target_metadata = align_metadata(METADATA, target_ids)
    groups = test_groups(target_metadata)
    for group in sorted(set(groups.tolist())):
        local = groups == group
        # Missing timestamps are not assumed to share one acquisition domain.
        if group == "unknown" or int(np.sum(local)) < 2:
            continue
        rows = target[local]
        corrected[rows] -= values[rows].mean(axis=0) - global_center

    model = make_model(300.0)
    model.fit(
        corrected[fit],
        labels[fit],
        ridge__sample_weight=sample_weights(labels[fit], 0.75),
    )
    return aligned_scores_40(model, corrected[target])


def safe_probability(protocol_value, grouping: GlobalRepeatConfig):
    with np.load(IMU, allow_pickle=False) as source:
        imu_probability, present = aligned_imu(
            source["sample_ids"].astype(str),
            np.asarray(source["imu_logits"], dtype=np.float64),
            protocol_value[0],
            3.0,
        )
    probability = protocol_value[2].copy()
    probability[present] = 0.95 * probability[present] + 0.05 * imu_probability[present]
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = joint_decode(
        probability,
        protocol_value[3],
        protocol_value,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )[0]
    return probability, prediction


def evaluate(
    protocol_value,
    grouping: GlobalRepeatConfig,
    base_probability: np.ndarray,
    safe_prediction: np.ndarray,
    scores: np.ndarray,
    temperature: float,
    weight: float,
):
    expert = np.exp(log_softmax_numpy(scores / temperature))
    probability = (1.0 - weight) * base_probability + weight * expert
    probability /= probability.sum(axis=1, keepdims=True)
    prediction, audit = joint_decode(
        probability,
        protocol_value[3],
        protocol_value,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
        initial_prediction=safe_prediction,
        grouping_prediction=protocol_value[3],
    )
    users = protocol_value[4].users.astype(str)
    labels = protocol_value[1]
    per_user = {
        user: int(
            np.sum(prediction[users == user] == labels[users == user])
            - np.sum(safe_prediction[users == user] == labels[users == user])
        )
        for user in sorted(set(users.tolist()))
    }
    return {
        "configuration": {"temperature": temperature, "weight": weight},
        "metrics": classification_metrics(labels, prediction),
        "rescue_harm_vs_safe": rescue_harm(labels, safe_prediction, prediction),
        "per_user_gain_vs_safe": per_user,
        "minimum_user_gain": int(min(per_user.values())),
        "positive_users": int(sum(value > 0 for value in per_user.values())),
        "grouping": audit,
    }, prediction


def main() -> None:
    grouping_source = json.loads(
        (PROJECT_DIR / "runs/p89_global_joint_grouping_tuned_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )
    grouping = GlobalRepeatConfig(**grouping_source["H1_selected"]["configuration"])
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    h1_probability, h1_safe = safe_probability(h1, grouping)
    h2_probability, h2_safe = safe_probability(h2, grouping)
    h1_scores = centered_scores(h1[0], full40.H1_USERS)

    candidates = []
    predictions = []
    for temperature in (0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0):
        for weight in (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15):
            result, prediction = evaluate(
                h1,
                grouping,
                h1_probability,
                h1_safe,
                h1_scores,
                temperature,
                weight,
            )
            candidates.append(result)
            predictions.append(prediction)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["minimum_user_gain"] >= 0,
            candidates[index]["metrics"]["correct"],
            candidates[index]["positive_users"],
            candidates[index]["metrics"]["balanced_accuracy"],
            candidates[index]["rescue_harm_vs_safe"]["net"],
            -candidates[index]["rescue_harm_vs_safe"]["harm"],
            -candidates[index]["configuration"]["weight"],
        ),
        reverse=True,
    )
    selected_index = order[0]
    selected = candidates[selected_index]
    h2_scores = centered_scores(h2[0], full40.H2_USERS)
    confirmation, h2_prediction = evaluate(
        h2,
        grouping,
        h2_probability,
        h2_safe,
        h2_scores,
        float(selected["configuration"]["temperature"]),
        float(selected["configuration"]["weight"]),
    )
    raw = {
        "H1": classification_metrics(h1[1], h1_scores.argmax(axis=1)),
        "H2": classification_metrics(h2[1], h2_scores.argmax(axis=1)),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_safe_prediction=h1_safe,
        h2_safe_prediction=h2_safe,
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
        h1_expert_scores=h1_scores.astype(np.float32),
        h2_expert_scores=h2_scores.astype(np.float32),
    )
    report = {
        "stage": "P89_transductive_recording_cohort_centered_VideoMAE_v1",
        "protocol": (
            "Remove each labelled training user's unlabeled feature mean and each held "
            "date-contiguous cohort's unlabeled feature mean, fit Ridge without held "
            "subjects, select blend calibration on H1, and transfer unchanged to H2."
        ),
        "feature": "L2 mean of 2 windows x 3 VideoMAE-Large views",
        "ridge_alpha": 300.0,
        "class_weight_power": 0.75,
        "raw_expert": raw,
        "safe": {
            "H1": classification_metrics(h1[1], h1_safe),
            "H2": classification_metrics(h2[1], h2_safe),
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
