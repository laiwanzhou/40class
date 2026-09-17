from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imu_data import read_index


PROJECT_DIR = Path(__file__).resolve().parent
CACHE = PROJECT_DIR / "cache/imu_32"
OUTPUT = PROJECT_DIR / "runs/p89_imu_rocket_expert_v1"
SEED = 20260816
KERNELS_PER_CHANNEL = 8
DILATIONS = (1, 2, 3)


def derived_signals(values: np.ndarray) -> np.ndarray:
    """Make synchronized univariate signals before per-clip normalization."""

    # values [N,device,time,raw10]
    raw = np.asarray(values, dtype=np.float32)
    blocks = [raw.transpose(0, 1, 3, 2).reshape(len(raw), 50, 32)]
    # Body-pair differences retain synchronized coordination which independent
    # per-device summaries cannot express.
    for first, second in ((1, 2), (3, 4), (0, 1), (0, 2), (0, 3), (0, 4)):
        blocks.append((raw[:, first, :, :6] - raw[:, second, :, :6]).transpose(0, 2, 1))
    magnitude = np.concatenate(
        (
            np.linalg.norm(raw[..., :3], axis=3),
            np.linalg.norm(raw[..., 3:6], axis=3),
        ),
        axis=1,
    )
    blocks.append(magnitude)
    signal = np.concatenate(blocks, axis=1)
    mean = signal.mean(axis=2, keepdims=True)
    std = signal.std(axis=2, keepdims=True)
    signal = (signal - mean) / np.maximum(std, 1e-4)
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def rocket_features(signal: np.ndarray) -> np.ndarray:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = signal.shape[1]
    rng = np.random.default_rng(SEED)
    output = []
    batch_size = 256 if device.type == "cuda" else 32
    for dilation in DILATIONS:
        weight = rng.normal(
            size=(channels * KERNELS_PER_CHANNEL, 1, 9)
        ).astype(np.float32)
        weight -= weight.mean(axis=2, keepdims=True)
        weight /= np.maximum(np.linalg.norm(weight, axis=2, keepdims=True), 1e-6)
        bias = rng.uniform(
            -1.25, 1.25, size=(1, channels * KERNELS_PER_CHANNEL, 1)
        ).astype(np.float32)
        weight_tensor = torch.from_numpy(weight).to(device)
        bias_tensor = torch.from_numpy(bias).to(device)
        dilation_output = []
        padding = dilation * 4
        for start in range(0, len(signal), batch_size):
            batch = torch.from_numpy(signal[start : start + batch_size]).to(device)
            convolution = F.conv1d(
                batch,
                weight_tensor,
                padding=padding,
                dilation=dilation,
                groups=channels,
            )
            features = torch.cat(
                (
                    convolution.amax(dim=2),
                    convolution.amin(dim=2),
                    (convolution > bias_tensor).float().mean(dim=2),
                ),
                dim=1,
            )
            dilation_output.append(features.cpu().numpy().astype(np.float32))
        output.append(np.concatenate(dilation_output, axis=0))
        print(f"ROCKET dilation={dilation} complete", flush=True)
    return np.concatenate(output, axis=1).astype(np.float32)


def softmax_decision(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    adjusted = np.asarray(values, dtype=np.float64) / temperature
    adjusted -= adjusted.max(axis=1, keepdims=True)
    probability = np.exp(adjusted)
    return probability / probability.sum(axis=1, keepdims=True)


def model(alpha: float):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(
            alpha=alpha,
            class_weight="balanced",
            solver="lsqr",
            tol=1e-4,
        ),
    )


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
    feature_path = OUTPUT / "features.npy"
    if feature_path.is_file():
        features = np.load(feature_path, mmap_mode="r")
    else:
        values = np.asarray(np.load(CACHE / "imu_float32.npy", mmap_mode="r"))
        signal = derived_signals(values)
        features = rocket_features(signal)
        np.save(feature_path, features)
    if len(features) != len(rows):
        raise RuntimeError("ROCKET feature row count changed")

    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.split == "train" and row.usable],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.split == "test"], dtype=np.int64
    )
    labels = np.asarray([row.class_id for row in rows], dtype=np.int64)
    users = np.asarray([row.user_id for row in rows]).astype(str)
    folds = json.loads(
        (PROJECT_DIR / "data/subject_folds/folds_summary.json").read_text(encoding="utf-8")
    )["folds"]
    results = {}
    probabilities = {}
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        oof = np.zeros((len(train_indices), 40), dtype=np.float64)
        per_fold = []
        for fold in folds:
            fit = np.isin(users[train_indices], fold["train_users"])
            held = np.isin(users[train_indices], fold["val_users"])
            estimator = model(alpha)
            estimator.fit(features[train_indices[fit]], labels[train_indices[fit]])
            target = np.flatnonzero(held)
            decision = estimator.decision_function(features[train_indices[held]])
            partial = softmax_decision(decision)
            classes = estimator[-1].classes_.astype(np.int64)
            oof[np.ix_(target, classes)] = partial
            item = {
                "fold": int(fold["fold"]),
                **metrics(labels[train_indices[held]], oof[target].argmax(axis=1)),
            }
            per_fold.append(item)
            print(f"alpha={alpha:g}", item, flush=True)
        results[str(alpha)] = {
            "overall": metrics(labels[train_indices], oof.argmax(axis=1)),
            "folds": per_fold,
        }
        probabilities[str(alpha)] = oof
        print(f"alpha={alpha:g}", results[str(alpha)]["overall"], flush=True)

    selected = max(
        results,
        key=lambda key: (
            results[key]["overall"]["accuracy"],
            results[key]["overall"]["balanced_accuracy"],
        ),
    )
    final = model(float(selected))
    final.fit(features[train_indices], labels[train_indices])
    test_partial = softmax_decision(final.decision_function(features[test_indices]))
    test_probability = np.zeros((len(test_indices), 40), dtype=np.float64)
    test_probability[:, final[-1].classes_.astype(np.int64)] = test_partial
    np.savez_compressed(
        OUTPUT / "oof_logits.npz",
        sample_ids=np.asarray([rows[index].sample_id for index in train_indices]),
        labels=labels[train_indices],
        imu_logits=np.log(np.maximum(probabilities[selected], 1e-12)),
    )
    np.savez_compressed(
        OUTPUT / "test_logits.npz",
        sample_ids=np.asarray([rows[index].sample_id for index in test_indices]),
        imu_logits=np.log(np.maximum(test_probability, 1e-12)),
    )
    report = {
        "stage": "P89_multidevice_ROCKET_IMU_expert_v1",
        "protocol": (
            "Fixed subject-disjoint folds; label-independent synchronized random "
            "convolution features over device, body-pair and magnitude signals."
        ),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "feature_dimension": int(features.shape[1]),
        "selected_alpha": float(selected),
        "models": results,
        "train_usable": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
