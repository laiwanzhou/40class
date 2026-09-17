"""Refit the fixed P158 LaViLa frame-token head on all Train and infer Test."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from p142_vjepa_token_transformer_oof import train_fold
from p90_teacher_common import load_protocol


HERE = Path(__file__).resolve().parent
TRAIN = HERE / "runs/p157_lavila_frame_token_cache_v1/frame_tokens.npy"
TEST = HERE / "runs/p197_lavila_frame_token_test_v1/frame_tokens.npy"
ROWS = HERE / "runs/p87s_test_pixel_cache_t16_r160_v1/rows.csv"
OUTPUT = HERE / "runs/p228_lavila_three_seed_test_v1"


def main() -> None:
    protocol = load_protocol()
    train = np.load(TRAIN, mmap_mode="r")
    test = np.load(TEST, mmap_mode="r")
    with ROWS.open("r", encoding="utf-8-sig", newline="") as handle:
        test_ids = np.asarray(
            [row["sample_id"] for row in csv.DictReader(handle)], dtype=str
        )
    if train.shape != (2914, 48, 768) or test.shape != (405, 48, 768):
        raise RuntimeError(f"P198 feature shape changed: {train.shape}/{test.shape}")
    values = np.concatenate(
        (np.asarray(train, dtype=np.float16), np.asarray(test, dtype=np.float16)),
        axis=0,
    )
    labels = np.concatenate(
        (protocol.labels, np.zeros(len(test), dtype=np.int64))
    )
    domains = np.concatenate(
        (protocol.fold_id, np.zeros(len(test), dtype=np.int64))
    )
    config = SimpleNamespace(
        hidden_dim=192,
        heads=6,
        layers=2,
        dropout=0.20,
        view_dropout=0.15,
        epochs=35,
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=0.05,
        mixup_alpha=0.20,
        repeat_consistency_weight=0.0,
        repeat_embedding_weight=0.0,
        repeat_same_label_only=False,
        class_triplet_weight=0.0,
        triplet_margin=0.20,
        teacher_weight=0.0,
        domain_adversarial_weight=0.0,
    )
    members = []
    for seed in (15801, 15817, 15833):
        members.append(
            train_fold(
                values,
                labels,
                np.arange(len(train), dtype=np.int64),
                np.arange(len(train), len(values), dtype=np.int64),
                domains,
                seed,
                config,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                None,
                None,
            )
        )
    logits = np.mean(np.stack(members), axis=0).astype(np.float64)
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "test_predictions.npz"
    np.savez_compressed(
        path,
        sample_ids=test_ids,
        logits=logits.astype(np.float32),
        probability=probability.astype(np.float32),
    )
    report = {
        "stage": "P228_LaViLa_three_seed_all2914_to_Test",
        "status": "complete",
        "train_rows": len(train),
        "test_rows": len(test),
        "seeds": [15801, 15817, 15833],
        "epochs": 35,
        "mean_confidence": float(probability.max(axis=1).mean()),
        "test_labels_read": False,
        "output": str(path.resolve()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

