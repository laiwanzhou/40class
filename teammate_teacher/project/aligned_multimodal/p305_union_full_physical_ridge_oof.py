"""Strict P231 IR+Thermal Ridge with the 10 recovered physical rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p90_teacher_common import load_protocol, softmax
from p231_depth_thermal_ir_oof import IR1, IR2, T, fit, flat
from train_p46_videomae_head import l2_normalize


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p305_union_full_physical_ridge_oof_v1"
EXTRA = HERE / "runs/p302_union_full_physical_features_v1/features.npz"
CONTROL = HERE / "runs/p231_depth_thermal_ir_oof_v1/oof_predictions.npz"


def flatten(values: np.ndarray) -> np.ndarray:
    return l2_normalize(values.astype(np.float32)).reshape(len(values), -1)


def main() -> None:
    protocol = load_protocol()
    ir1, ir2, thermal = np.load(IR1), np.load(IR2), np.load(T)
    main = np.concatenate((flat(ir1), flat(ir2), flat(thermal)), axis=1)
    extra = np.load(EXTRA)
    extra_values = np.concatenate(
        (
            flatten(extra["ir_videomaev2"]),
            flatten(extra["ir_internvideo2"]),
            flatten(extra["thermal_videomaev2"]),
        ),
        axis=1,
    )
    values = np.concatenate((main, extra_values), axis=0)
    labels = np.concatenate((protocol.labels, extra["labels"].astype(int)))
    users = np.concatenate((protocol.users, extra["users"].astype(str)))
    extra_indices = np.arange(len(protocol.labels), len(labels))
    logits = np.zeros((len(protocol.labels), 40), dtype=np.float64)
    folds = []
    for fold in range(3):
        held = protocol.val_indices(fold)
        held_users = set(protocol.users[held].tolist())
        allowed_extra = extra_indices[
            ~np.isin(users[extra_indices], list(held_users))
        ]
        train = np.concatenate((protocol.train_indices(fold), allowed_extra))
        logits[held] = fit(values, labels, train, held)
        folds.append(
            {
                "fold": fold,
                "extra_train_rows": int(len(allowed_extra)),
                "correct": int(np.sum(logits[held].argmax(1) == protocol.labels[held])),
                "rows": int(len(held)),
            }
        )
    probability = softmax(logits)
    control = np.load(CONTROL)["ir_thermal_probability"]
    old = control.argmax(1)
    new = probability.argmax(1)
    changed = old != new
    report = {
        "stage": "P305_union_full_physical_ridge_OOF",
        "status": "complete",
        "protocol": {
            "paired_control": "P231 ir_thermal alpha=3000 weight_power=.75",
            "recovered_extra_rows": 10,
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
        sample_ids=protocol.sample_ids,
        labels=protocol.labels,
        probability=probability.astype(np.float32),
        logits=logits.astype(np.float32),
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
