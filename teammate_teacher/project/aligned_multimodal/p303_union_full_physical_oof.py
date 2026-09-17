"""Strict paired P238 OOF with the 10 recovered D/IR/Thermal rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import train_fold
from p238_physical_token_transformer_oof import PATHS, SEEDS, cfg


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p303_union_full_physical_oof_v1"
EXTRA = HERE / "runs/p302_union_full_physical_features_v1/features.npz"
CONTROL = HERE / "runs/p238_physical_token_transformer_oof_v1/oof_predictions.npz"


def main() -> None:
    protocol = load_protocol()
    sources = [np.load(path) for path in PATHS]
    for source in sources:
        if not np.array_equal(source["sample_ids"].astype(str), protocol.sample_ids):
            raise RuntimeError("P303 main feature order differs from P238")
    main_features = np.concatenate(
        [source["features"].astype(np.float16).reshape(len(protocol.labels), -1, 768)
         for source in sources],
        axis=1,
    )
    extra = np.load(EXTRA)
    extra_features = np.concatenate(
        [
            extra["ir_videomaev2"].astype(np.float16),
            extra["ir_internvideo2"].astype(np.float16),
            extra["depth_videomaev2"].astype(np.float16),
            extra["thermal_videomaev2"].astype(np.float16),
        ],
        axis=1,
    )
    if extra_features.shape != (10, 18, 768):
        raise RuntimeError(f"unexpected recovered feature shape {extra_features.shape}")
    features = np.concatenate((main_features, extra_features), axis=0)
    labels = np.concatenate((protocol.labels, extra["labels"].astype(int)))
    users = np.concatenate((protocol.users, extra["users"].astype(str)))
    domain = np.zeros(len(labels), dtype=np.int64)
    output = np.zeros((len(protocol.labels), 40), dtype=np.float64)
    folds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extra_indices = np.arange(len(protocol.labels), len(labels))
    for fold in range(3):
        held = protocol.val_indices(fold)
        held_users = set(protocol.users[held].tolist())
        allowed_extra = extra_indices[
            ~np.isin(users[extra_indices], list(held_users))
        ]
        train = np.concatenate((protocol.train_indices(fold), allowed_extra))
        members = [
            train_fold(
                features,
                labels,
                train,
                held,
                domain,
                seed + fold * 1000,
                cfg(),
                device,
                None,
                None,
            )
            for seed in SEEDS
        ]
        output[held] = np.mean(np.stack(members), axis=0)
        folds.append(
            {
                "fold": fold,
                "rows": int(len(held)),
                "main_train_rows": int(len(protocol.train_indices(fold))),
                "extra_train_rows": int(len(allowed_extra)),
                "held_extra_subject_rows_excluded": int(len(extra_indices) - len(allowed_extra)),
                "correct": int(np.sum(output[held].argmax(1) == protocol.labels[held])),
            }
        )
        print(json.dumps(folds[-1]), flush=True)
    probability = np.exp(output - output.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    control = np.load(CONTROL)["probability"]
    control_prediction = control.argmax(1)
    prediction = probability.argmax(1)
    changed = prediction != control_prediction
    report = {
        "stage": "P303_union_full_physical_OOF",
        "status": "complete",
        "protocol": {
            "paired_control": "P238 identical architecture, seeds, epochs and folds",
            "main_rows": int(len(protocol.labels)),
            "recovered_extra_rows": int(len(extra_indices)),
            "extra_subjects": sorted(set(extra["users"].astype(str).tolist())),
            "held_subject_extra_rows_excluded": True,
            "strict_subject_folds": True,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "control": {
            "correct": int(np.sum(control_prediction == protocol.labels)),
            "accuracy": float(np.mean(control_prediction == protocol.labels)),
        },
        "augmented": {
            "correct": int(np.sum(prediction == protocol.labels)),
            "accuracy": float(np.mean(prediction == protocol.labels)),
            "folds": folds,
            "changed": int(changed.sum()),
            "rescued": int(np.sum(changed & (control_prediction != protocol.labels) & (prediction == protocol.labels))),
            "harmed": int(np.sum(changed & (control_prediction == protocol.labels) & (prediction != protocol.labels))),
        },
    }
    report["augmented"]["net_vs_control"] = (
        report["augmented"]["correct"] - report["control"]["correct"]
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        sample_ids=protocol.sample_ids,
        labels=protocol.labels,
        users=protocol.users,
        fold_id=protocol.fold_id,
        logits=output.astype(np.float32),
        probability=probability.astype(np.float32),
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
