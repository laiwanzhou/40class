"""Refit the fixed P128 hierarchical multimodal teacher and infer Test."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import softmax

from p91_hierarchical_multimodal_teacher import (
    FusionData,
    Preprocessor,
    build_data,
    infer_full,
    train_model,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "runs/p176_p128_hierarchical_test_v1"
VMAE = REPO / "runs/p90_videomaev2_distilled_test_v1/complete_features.npz"
IV2 = REPO / "runs/p90_internvideo2_l_k400_test_v1/complete_features.npz"
DEPTH = HERE / "runs/p89_videomae_depth_test_v1/complete_features.npz"
THERMAL = HERE / "runs/p89_videomae_thermal_test_v1/complete_features.npz"
HAND = REPO / "runs/p175_videomaev2_hand_test_v1/complete_features.npz"
MOTIONBERT = HERE / "runs/a18_test_features_v1/motionbert_pretrain_front_t81.npz"
HDGCN = HERE / "runs/a18_test_features_v1/hdgcn_six_stream_tokens.npz"
CROSS = HERE / "runs/a18_test_features_v1/crossmodal_statistics.npz"
MOTION = HERE / "runs/p87s_test_motion_window_t16_v1"
SEEDS = (10017, 10043, 10071)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def motion_ids() -> np.ndarray:
    with (MOTION / "rows.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray([row["sample_id"] for row in csv.DictReader(handle)], dtype=str)


def align_fill(
    source_ids: np.ndarray,
    values: np.ndarray,
    target_ids: np.ndarray,
    fill_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {value: row for row, value in enumerate(source_ids.astype(str))}
    output = np.empty((len(target_ids), *values.shape[1:]), dtype=np.float32)
    output[...] = fill_vector
    present = np.asarray([value in lookup for value in target_ids.astype(str)])
    rows = np.flatnonzero(present)
    source_rows = np.asarray([lookup[target_ids[row]] for row in rows], dtype=np.int64)
    output[rows] = values[source_rows].astype(np.float32)
    return output, present


def align_exact(source_ids, values, target_ids):
    lookup = {value: row for row, value in enumerate(source_ids.astype(str))}
    missing = [value for value in target_ids.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"P176 exact source misses {len(missing)} rows")
    rows = np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)
    return np.asarray(values[rows], dtype=np.float32)


def visual_fill(train_values: np.ndarray, token_count: int) -> np.ndarray:
    mean = train_values.reshape(-1, train_values.shape[-1]).mean(axis=0)
    return np.broadcast_to(mean, (token_count, train_values.shape[-1])).astype(np.float32)


def build_test(train: FusionData) -> FusionData:
    motion_id = motion_ids()
    ids = motion_id.astype(str)
    if len(ids) != 405 or len(set(ids.tolist())) != 405:
        raise RuntimeError("P176 Test ID contract changed")
    vmae = load(VMAE); iv2 = load(IV2); depth = load(DEPTH); thermal = load(THERMAL)
    hand = load(HAND); motionbert = load(MOTIONBERT); hdgcn = load(HDGCN); cross = load(CROSS)
    streams = {}
    present = {}
    for name, source, raw in (
        ("vmae", vmae, vmae["features"].reshape(-1, 6, 768)),
        ("iv2", iv2, iv2["features"].reshape(-1, 6, 768)),
        ("depth", depth, depth["features"].reshape(-1, 3, 768)),
        ("thermal", thermal, thermal["features"].reshape(-1, 3, 768)),
        ("hand", hand, hand["features"].reshape(-1, 6, 768)),
    ):
        streams[name], present[name] = align_fill(
            source["sample_ids"].astype(str),
            raw,
            ids,
            visual_fill(train.streams[name], raw.shape[1]),
        )
    streams["motionbert"] = align_exact(
        motionbert["sample_ids"], motionbert["features"].reshape(-1, 12, 768), ids
    )
    streams["hdgcn"] = np.tile(
        align_exact(hdgcn["sample_ids"], hdgcn["features"], ids), (1, 1, 3)
    )
    cross_values = align_exact(cross["sample_ids"], cross["features"], ids)
    skeleton_sequence = np.load(MOTION / "skeleton_features.npy", mmap_mode="r").reshape(
        len(ids), 32, 17, 13
    ).astype(np.float32)
    skeleton_mask = np.load(MOTION / "skeleton_joint_mask.npy", mmap_mode="r").reshape(
        len(ids), 32, 17
    ).astype(np.float32)
    imu_sequence = np.load(MOTION / "imu_sequences.npy", mmap_mode="r").reshape(
        len(ids), 32, 5, 4, 16
    ).astype(np.float32)
    imu_mask = np.load(MOTION / "imu_sequence_mask.npy", mmap_mode="r").reshape(
        len(ids), 32, 5, 4
    ).astype(np.float32)
    return FusionData(
        sample_ids=ids,
        labels=np.zeros(len(ids), dtype=np.int64),
        users=np.full(len(ids), "anonymous", dtype=str),
        teacher_prediction=np.zeros(len(ids), dtype=np.int64),
        expert_probability=np.zeros((len(ids), train.expert_probability.shape[1], 40), dtype=np.float32),
        streams=streams,
        statistics={
            "skeleton": np.concatenate((cross_values[:, :2629], cross_values[:, 5729:5740]), axis=1),
            "imu": np.concatenate((cross_values[:, 2629:5729], cross_values[:, 5740:5795]), axis=1),
            "relation": cross_values[:, 5795:6195],
        },
        skeleton_sequence=skeleton_sequence,
        skeleton_mask=skeleton_mask,
        imu_sequence=imu_sequence,
        imu_mask=imu_mask,
        thermal_available=present["thermal"].astype(np.float32),
        boundaries={"Test": np.arange(len(ids), dtype=np.int64)},
    )


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    args = SimpleNamespace(
        epochs=21,
        patience=22,
        batch_size=64,
        model_dim=192,
        layers=3,
        heads=8,
        dropout=0.22,
        learning_rate=3e-4,
        weight_decay=5e-3,
        statistics_dim=64,
        device="cuda",
    )
    raw_train = build_data()
    raw_test = build_test(raw_train)
    train_indices = np.arange(len(raw_train.labels), dtype=np.int64)
    test_indices = np.arange(len(raw_test.labels), dtype=np.int64)
    preprocessor = Preprocessor(args.statistics_dim, 17600).fit(raw_train, train_indices)
    train = preprocessor.transform(raw_train)
    test = preprocessor.transform(raw_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = []
    reliability = []
    audits = []
    for seed in SEEDS:
        print(json.dumps({"stage": "P176_train", "seed": seed, "epochs": 21}), flush=True)
        model, audit, _, _ = train_model(
            args, train, train_indices, None, seed, fixed_epochs=21
        )
        values, reliability_values = infer_full(
            model, test, test_indices, device, args.batch_size
        )
        logits.append(values)
        reliability.append(reliability_values)
        audits.append(audit)
        del model
        torch.cuda.empty_cache()
    ensemble = np.mean(logits, axis=0)
    reliability_ensemble = np.mean(reliability, axis=0)
    probability = softmax(ensemble, axis=1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "test_predictions.npz"
    np.savez_compressed(
        path,
        sample_ids=test.sample_ids,
        logits=ensemble.astype(np.float32),
        probabilities=probability.astype(np.float32),
        reliability_logits=reliability_ensemble.astype(np.float32),
        prediction=probability.argmax(axis=1).astype(np.int64),
    )
    report = {
        "stage": "P176_P128_hierarchical_all2914_to_Test",
        "status": "complete",
        "protocol": {
            "train_rows": len(train.labels),
            "test_rows": len(test.labels),
            "fixed_epochs": 21,
            "seeds": list(SEEDS),
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "preprocessor": preprocessor.summary(),
        "seed_audits": audits,
        "mean_confidence": float(probability.max(axis=1).mean()),
        "output": str(path.resolve()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
