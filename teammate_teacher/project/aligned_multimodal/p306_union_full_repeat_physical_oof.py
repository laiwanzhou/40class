"""P253 repeat-regularized physical expert with 10 recovered union rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from p90_teacher_common import load_protocol
from p142_vjepa_token_transformer_oof import build_repeat_pairs, train_fold
from p253_repeat_physical_transformer_oof import PATHS, SEEDS, cfg


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p306_union_full_repeat_physical_oof_v1"
EXTRA = HERE / "runs/p302_union_full_physical_features_v1/features.npz"
CONTROL = HERE / "runs/p253_repeat_physical_transformer_oof_v1/oof_predictions.npz"


def main() -> None:
    protocol = load_protocol()
    sources = [np.load(path) for path in PATHS]
    main_features = np.concatenate(
        [source["features"].astype(np.float16).reshape(len(protocol.labels), -1, 768)
         for source in sources], axis=1
    )
    extra = np.load(EXTRA)
    extra_features = np.concatenate(
        [extra["ir_videomaev2"], extra["ir_internvideo2"],
         extra["depth_videomaev2"], extra["thermal_videomaev2"]], axis=1
    ).astype(np.float16)
    features = np.concatenate((main_features, extra_features), axis=0)
    labels = np.concatenate((protocol.labels, extra["labels"].astype(int)))
    users = np.concatenate((protocol.users, extra["users"].astype(str)))
    domain = np.zeros(len(labels), dtype=np.int64)
    pairs = build_repeat_pairs(protocol)
    extra_indices = np.arange(len(protocol.labels), len(labels))
    logits = np.zeros((len(protocol.labels), 40), dtype=np.float64)
    folds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in range(3):
        held = protocol.val_indices(fold)
        held_users = set(protocol.users[held].tolist())
        allowed_extra = extra_indices[~np.isin(users[extra_indices], list(held_users))]
        train = np.concatenate((protocol.train_indices(fold), allowed_extra))
        members = [
            train_fold(features, labels, train, held, domain, seed + fold * 1000,
                       cfg(), device, pairs, None)
            for seed in SEEDS
        ]
        logits[held] = np.mean(np.stack(members), axis=0)
        folds.append({
            "fold": fold,
            "extra_train_rows": int(len(allowed_extra)),
            "correct": int(np.sum(logits[held].argmax(1) == protocol.labels[held])),
            "rows": int(len(held)),
        })
        print(json.dumps(folds[-1]), flush=True)
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    old_probability = np.load(CONTROL)["probability"]
    old = old_probability.argmax(1)
    new = probability.argmax(1)
    changed = old != new
    report = {
        "stage": "P306_union_full_repeat_physical_OOF",
        "status": "complete",
        "protocol": {
            "paired_control": "P253 identical seeds/config/repeat pairs",
            "recovered_extra_rows": 10,
            "repeat_pairs": int(len(pairs)),
            "repeat_pairs_include_extra": False,
            "held_subject_extra_rows_excluded": True,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "control_correct": int(np.sum(old == protocol.labels)),
        "augmented_correct": int(np.sum(new == protocol.labels)),
        "augmented_accuracy": float(np.mean(new == protocol.labels)),
        "net_vs_control": int(np.sum(new == protocol.labels) - np.sum(old == protocol.labels)),
        "changed": int(changed.sum()),
        "rescued": int(np.sum(changed & (old != protocol.labels) & (new == protocol.labels))),
        "harmed": int(np.sum(changed & (old == protocol.labels) & (new != protocol.labels))),
        "folds": folds,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        sample_ids=protocol.sample_ids, labels=protocol.labels,
        probability=probability.astype(np.float32), logits=logits.astype(np.float32),
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
