from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from imu_data import DEVICES, read_index


PROJECT_DIR = Path(__file__).resolve().parent
CACHE = PROJECT_DIR / "cache/imu_32"
SOURCE_FEATURES = PROJECT_DIR / "runs/p89_imu_orientation_expert_v1/features.npy"
OUTPUT = PROJECT_DIR / "runs/p89_imu_sensor_attention_expert_v1"
FOLDS = PROJECT_DIR / "data/subject_folds/folds_summary.json"
SEED = 20260816
DEVICE_FEATURES = 36 * 25 + 112 + 2
METHODS = (
    "equal_probability",
    "entropy_probability",
    "class_log_t1",
    "class_log_t2",
    "class_entropy_log_t2",
)


def estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=250,
        max_depth=22,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def aligned_probability(model: ExtraTreesClassifier, features: np.ndarray) -> np.ndarray:
    result = np.full((len(features), 40), 1e-6, dtype=np.float64)
    partial = model.predict_proba(features)
    result[:, model.classes_.astype(np.int64)] = partial
    result /= result.sum(axis=1, keepdims=True)
    return result


def device_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    device_present: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    seed: int,
) -> np.ndarray:
    output = np.full((len(predict), len(DEVICES), 40), 1.0 / 40.0, dtype=np.float64)
    for device in range(len(DEVICES)):
        fit_rows = fit[device_present[fit, device]]
        if len(fit_rows) == 0:
            continue
        model = estimator(seed + device)
        left = device * DEVICE_FEATURES
        right = left + DEVICE_FEATURES
        model.fit(features[fit_rows, left:right], labels[fit_rows])
        present = device_present[predict, device]
        if np.any(present):
            output[present, device] = aligned_probability(
                model, features[predict[present], left:right]
            )
    return output


def inner_oof_reliability(
    features: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    device_present: np.ndarray,
    outer_fit: np.ndarray,
    seed: int,
) -> np.ndarray:
    fit_users = sorted(set(users[outer_fit].tolist()))
    inner_probability = np.full(
        (len(outer_fit), len(DEVICES), 40), 1.0 / 40.0, dtype=np.float64
    )
    local_users = users[outer_fit]
    for inner_fold in range(3):
        held_users = fit_users[inner_fold::3]
        held_local = np.isin(local_users, held_users)
        inner_fit = outer_fit[~held_local]
        inner_held = outer_fit[held_local]
        inner_probability[held_local] = device_probabilities(
            features,
            labels,
            device_present,
            inner_fit,
            inner_held,
            seed + 100 * inner_fold,
        )
    reliability = np.zeros((40, len(DEVICES)), dtype=np.float64)
    local_labels = labels[outer_fit]
    local_present = device_present[outer_fit]
    for class_id in range(40):
        class_rows = local_labels == class_id
        for device in range(len(DEVICES)):
            selected = class_rows & local_present[:, device]
            if np.any(selected):
                reliability[class_id, device] = float(
                    np.mean(
                        np.log(
                            np.maximum(
                                inner_probability[selected, device, class_id], 1e-6
                            )
                        )
                    )
                )
            else:
                reliability[class_id, device] = -np.log(40.0)
    return reliability


def softmax_axis(value: np.ndarray, axis: int) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    result = np.exp(shifted)
    return result / np.maximum(result.sum(axis=axis, keepdims=True), 1e-12)


def fuse(
    probability: np.ndarray,
    present: np.ndarray,
    reliability: np.ndarray,
) -> dict[str, np.ndarray]:
    availability = present.astype(np.float64)
    no_device = availability.sum(axis=1) == 0
    availability[no_device] = 1.0
    equal_weight = availability / availability.sum(axis=1, keepdims=True)
    equal = np.einsum("nd,ndc->nc", equal_weight, probability)

    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=2)
    confidence_weight = availability * np.exp(-entropy)
    confidence_weight /= np.maximum(confidence_weight.sum(axis=1, keepdims=True), 1e-12)
    entropy_probability = np.einsum("nd,ndc->nc", confidence_weight, probability)

    output = {
        "equal_probability": equal,
        "entropy_probability": entropy_probability,
    }
    log_probability = np.log(np.maximum(probability, 1e-8))
    for temperature, name in ((1.0, "class_log_t1"), (2.0, "class_log_t2")):
        class_weight = softmax_axis(reliability / temperature, axis=1)
        weights = availability[:, :, None] * class_weight.T[None]
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        score = np.sum(weights * log_probability, axis=1)
        output[name] = softmax_axis(score, axis=1)

    class_weight = softmax_axis(reliability / 2.0, axis=1)
    weights = availability[:, :, None] * class_weight.T[None]
    weights *= np.exp(-entropy)[:, :, None]
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    score = np.sum(weights * log_probability, axis=1)
    output["class_entropy_log_t2"] = softmax_axis(score, axis=1)
    for name in output:
        output[name][no_device] = 1.0 / 40.0
    return output


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = read_index(CACHE / "index.csv")
    features = np.load(SOURCE_FEATURES, mmap_mode="r")
    device_mask = np.load(CACHE / "device_mask_uint8.npy", mmap_mode="r") > 0
    if len(rows) != len(features) or len(rows) != len(device_mask):
        raise RuntimeError("IMU cache alignment changed")
    labels_all = np.asarray([row.class_id for row in rows], dtype=np.int64)
    users_all = np.asarray([row.user_id for row in rows]).astype(str)
    train = np.asarray(
        [index for index, row in enumerate(rows) if row.split == "train" and row.usable],
        dtype=np.int64,
    )
    test = np.asarray(
        [index for index, row in enumerate(rows) if row.split == "test"], dtype=np.int64
    )
    labels = labels_all[train]
    users = users_all[train]
    sample_ids = np.asarray([rows[index].sample_id for index in train])
    test_ids = np.asarray([rows[index].sample_id for index in test])
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    oof = {name: np.zeros((len(train), 40), dtype=np.float64) for name in METHODS}
    per_fold = {name: [] for name in METHODS}
    for fold in folds:
        fit_local = np.flatnonzero(np.isin(users, fold["train_users"]))
        held_local = np.flatnonzero(np.isin(users, fold["val_users"]))
        fit_global = train[fit_local]
        held_global = train[held_local]
        reliability = inner_oof_reliability(
            features,
            labels_all,
            users_all,
            device_mask,
            fit_global,
            SEED + 1000 * int(fold["fold"]),
        )
        probability = device_probabilities(
            features,
            labels_all,
            device_mask,
            fit_global,
            held_global,
            SEED + 10000 + 1000 * int(fold["fold"]),
        )
        fused = fuse(probability, device_mask[held_global], reliability)
        for name in METHODS:
            oof[name][held_local] = fused[name]
            per_fold[name].append(
                {"fold": int(fold["fold"]), **metrics(labels[held_local], fused[name].argmax(1))}
            )
        print(f"sensor-attention outer fold {fold['fold']} complete", flush=True)

    reliability = inner_oof_reliability(
        features, labels_all, users_all, device_mask, train, SEED + 50000
    )
    test_device_probability = device_probabilities(
        features,
        labels_all,
        device_mask,
        train,
        test,
        SEED + 60000,
    )
    test_fused = fuse(test_device_probability, device_mask[test], reliability)
    results = {
        name: {"overall": metrics(labels, oof[name].argmax(1)), "folds": per_fold[name]}
        for name in METHODS
    }
    selected = max(
        METHODS,
        key=lambda name: (
            results[name]["overall"]["accuracy"],
            results[name]["overall"]["balanced_accuracy"],
        ),
    )
    np.savez_compressed(
        OUTPUT / "oof_probabilities.npz",
        sample_ids=sample_ids,
        labels=labels,
        users=users,
        **{name: oof[name] for name in METHODS},
    )
    np.savez_compressed(
        OUTPUT / "test_probabilities.npz",
        sample_ids=test_ids,
        device_present=device_mask[test],
        **{name: test_fused[name] for name in METHODS},
    )
    report = {
        "stage": "P89_sensorwise_local_global_IMU_attention_v1",
        "protocol": "Five wearable locations are modeled independently from local plus quaternion-derived global signals. Class-specific sensor reliability is estimated by three-fold subject-disjoint inner OOF inside every outer fold; no outer labels or Test data tune fusion.",
        "devices": list(DEVICES),
        "device_feature_dimension": DEVICE_FEATURES,
        "methods": results,
        "selected_overall_for_diagnostics": selected,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
