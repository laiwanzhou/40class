from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import log_softmax_numpy, rescue_harm
from p89_cross_subject_teacher_stacker import expert_features
from p89_global_joint_repeat_decoder import joint_decode
from p89_global_repeat_decoder import GlobalRepeatConfig
from p89_imu_rescue_gate import aligned_imu


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_stacker_on_imu_safe_v1"
TEACHER = PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
IMU = PROJECT_DIR / "runs/p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax_numpy(values))


def prepare(protocol_value, grouping: GlobalRepeatConfig):
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
    safe = joint_decode(
        probability,
        protocol_value[3],
        protocol_value,
        grouping,
        evidence_weight=0.25,
        transition_scale=1.0,
    )[0]
    return probability, safe


def fit_stacker(eval_ids: np.ndarray, holdout_users: list[str], c_value: float) -> np.ndarray:
    with np.load(TEACHER, allow_pickle=False) as source:
        all_ids = source["oof_sample_ids"].astype(str)
        all_labels = source["oof_labels"].astype(np.int64)
    all_x, _ = expert_features(all_ids)
    lookup = {sample_id: index for index, sample_id in enumerate(all_ids)}
    evaluation = np.asarray([lookup[sample_id] for sample_id in eval_ids], dtype=np.int64)
    fit = np.asarray(
        [
            index
            for index, sample_id in enumerate(all_ids)
            if not any(f"__{user}__" in sample_id for user in holdout_users)
        ],
        dtype=np.int64,
    )
    if np.intersect1d(fit, evaluation).size:
        raise RuntimeError("stacker train/evaluation overlap")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            tol=2e-4,
        ),
    )
    model.fit(all_x[fit], all_labels[fit])
    return np.asarray(model.decision_function(all_x[evaluation]), dtype=np.float64)


def evaluate(
    protocol_value,
    grouping: GlobalRepeatConfig,
    safe_probability: np.ndarray,
    safe_prediction: np.ndarray,
    decision: np.ndarray,
    temperature: float,
    weight: float,
):
    stacked = softmax(decision / temperature)
    probability = (1.0 - weight) * safe_probability + weight * stacked
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
        "configuration": {
            "regularization": 0.03,
            "temperature": temperature,
            "weight": weight,
        },
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
    h1_probability, h1_safe = prepare(h1, grouping)
    h2_probability, h2_safe = prepare(h2, grouping)
    h1_decision = fit_stacker(h1[0], full40.H1_USERS, 0.03)

    candidates = []
    predictions = []
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for weight in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15):
            item, prediction = evaluate(
                h1, grouping, h1_probability, h1_safe, h1_decision, temperature, weight
            )
            candidates.append(item)
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
    configuration = selected["configuration"]
    h2_decision = fit_stacker(h2[0], full40.H2_USERS, 0.03)
    confirmation, h2_prediction = evaluate(
        h2,
        grouping,
        h2_probability,
        h2_safe,
        h2_decision,
        float(configuration["temperature"]),
        float(configuration["weight"]),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        h1_sample_ids=h1[0],
        h2_sample_ids=h2[0],
        h1_safe_prediction=h1_safe,
        h2_safe_prediction=h2_safe,
        h1_prediction=predictions[selected_index],
        h2_prediction=h2_prediction,
    )
    report = {
        "stage": "P89_stacker_on_frozen_IMU_global_safe_v1",
        "protocol": (
            "Fit the 19-expert stacker without each held-out subject set, blend it only "
            "after the frozen 5% IMU posterior, select on H1, and transfer unchanged to H2."
        ),
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
