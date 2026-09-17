"""Complete three-fold OOF for the frozen P96 V-JEPA2 dense24 features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p90_teacher_common import load_protocol, softmax
from p96_vjepa2_dense24_teacher_h1h2 import fit_scores
from train_p46_videomae_head import l2_normalize, row_standardize


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
P96 = REPO / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
VMAE = REPO / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IV2 = REPO / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"
OUTPUT = HERE / "runs/p123_vjepa2_dense24_full_oof_v1"


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(labels)),
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
    }


def load_features() -> dict[str, tuple[np.ndarray, float]]:
    dense = l2_normalize(
        np.asarray(np.load(P96 / "features.npy", mmap_mode="r"), dtype=np.float32)
    )
    action = row_standardize(
        np.asarray(np.load(P96 / "ssv2_logits.npy", mmap_mode="r"), dtype=np.float32)
    )
    dense_group = l2_normalize(dense.reshape(len(dense), 8, 3, 1024).mean(axis=2))
    action_group = row_standardize(action.reshape(len(action), 8, 3, 174).mean(axis=2))
    dense_flat = dense_group.reshape(len(dense), -1)
    dense_action = np.concatenate(
        (dense_flat, action_group.reshape(len(dense), -1)), axis=1
    ).astype(np.float32)
    with np.load(VMAE) as vmae, np.load(IV2) as iv2:
        protocol = load_protocol()
        if not np.array_equal(vmae["sample_ids"].astype(str), protocol.sample_ids):
            raise ValueError("VMAE order mismatch")
        if not np.array_equal(iv2["sample_ids"].astype(str), protocol.sample_ids):
            raise ValueError("IV2 order mismatch")
        old_ir = np.concatenate(
            (
                l2_normalize(vmae["features"].astype(np.float32)).reshape(len(dense), -1),
                l2_normalize(iv2["features"].astype(np.float32)).reshape(len(dense), -1),
            ),
            axis=1,
        ).astype(np.float32)
    return {
        "dense24_group8": (dense_flat, 3000.0),
        "dense24_group8_ssv2": (dense_action, 4000.0),
        "old_ir_plus_dense24_group8_ssv2": (
            np.concatenate((old_ir, dense_action), axis=1).astype(np.float32),
            9000.0,
        ),
    }


def main() -> None:
    protocol = load_protocol()
    candidates = load_features()
    saved: dict[str, np.ndarray] = {
        "sample_ids": protocol.sample_ids,
        "labels": protocol.labels,
        "users": protocol.users,
        "fold_id": protocol.fold_id,
    }
    variants = {}
    for name, (feature, alpha) in candidates.items():
        logits = np.zeros((len(protocol.labels), 40), dtype=np.float64)
        fold_results = []
        for fold in range(3):
            train = protocol.train_indices(fold)
            held = protocol.val_indices(fold)
            logits[held] = fit_scores(
                feature[train], protocol.labels[train], feature[held], alpha
            )
            fold_results.append(
                {"fold": fold, **metrics(protocol.labels[held], logits[held].argmax(1))}
            )
        prediction = logits.argmax(1)
        variants[name] = {
            "feature_dim": int(feature.shape[1]),
            "alpha": alpha,
            "metrics": metrics(protocol.labels, prediction),
            "folds": fold_results,
        }
        saved[f"{name}_logits"] = logits.astype(np.float32)
        saved[f"{name}_probability"] = softmax(logits).astype(np.float32)
        print(json.dumps({"candidate": name, **variants[name]}), flush=True)
    report = {
        "stage": "P123_VJEPA2_dense24_complete_threefold_OOF",
        "status": "complete",
        "protocol": {
            "features_frozen": True,
            "recipes_and_alpha": "frozen from P96 source-only study",
            "held_fold_used_for_recipe_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "variants": variants,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "oof_predictions.npz", **saved)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
