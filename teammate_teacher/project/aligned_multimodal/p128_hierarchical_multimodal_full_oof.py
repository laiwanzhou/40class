"""Fixed-recipe complete OOF for the P91 hierarchical multimodal teacher."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import softmax

from p91_hierarchical_multimodal_teacher import (
    Preprocessor,
    build_data,
    infer_full,
    train_model,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs/p128_hierarchical_multimodal_full_oof_v1"
FOLDS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
EMBARGO = "E0_p87_sequence_source"
SEEDS = (10017, 10043, 10071)
FIXED_EPOCHS = 21


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(labels)),
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
    }


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    args = SimpleNamespace(
        epochs=FIXED_EPOCHS,
        patience=FIXED_EPOCHS + 1,
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
    raw = build_data()
    logits = np.zeros((len(raw.labels), 40), dtype=np.float64)
    reliability = np.zeros(len(raw.labels), dtype=np.float64)
    fold_reports = []
    for fold_index, held_name in enumerate(FOLDS):
        source_names = [name for name in FOLDS if name != held_name] + [EMBARGO]
        train_indices = np.concatenate([raw.boundaries[name] for name in source_names])
        held_indices = raw.boundaries[held_name]
        preprocessor = Preprocessor(
            args.statistics_dim, 12800 + fold_index * 100
        ).fit(raw, train_indices)
        data = preprocessor.transform(raw)
        seed_logits = []
        seed_reliability = []
        seed_audits = []
        for seed in SEEDS:
            actual_seed = seed + fold_index * 1000
            print(
                json.dumps(
                    {
                        "stage": "P128_train",
                        "held": held_name,
                        "seed": actual_seed,
                        "epochs": FIXED_EPOCHS,
                    }
                ),
                flush=True,
            )
            model, audit, _, _ = train_model(
                args,
                data,
                train_indices,
                None,
                actual_seed,
                fixed_epochs=FIXED_EPOCHS,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            values, reliability_values = infer_full(
                model, data, held_indices, device, args.batch_size
            )
            seed_logits.append(values)
            seed_reliability.append(reliability_values)
            seed_audits.append(audit)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fold_logits = np.mean(seed_logits, axis=0)
        fold_reliability = np.mean(seed_reliability, axis=0)
        logits[held_indices] = fold_logits
        reliability[held_indices] = fold_reliability
        fold_reports.append(
            {
                "held": held_name,
                "source": source_names,
                "train_rows": int(len(train_indices)),
                "metrics": metrics(
                    data.labels[held_indices], fold_logits.argmax(axis=1)
                ),
                "seed_audits": seed_audits,
                "preprocessor": preprocessor.summary(),
            }
        )
    scored = np.concatenate([raw.boundaries[name] for name in FOLDS])
    prediction = logits.argmax(axis=1)
    report = {
        "stage": "P128_hierarchical_multimodal_complete_outer_OOF_v1",
        "status": "complete",
        "protocol": {
            "architecture": "P91 hierarchical multimodal v3 frozen",
            "fixed_epochs": FIXED_EPOCHS,
            "seeds_per_fold": list(SEEDS),
            "source_per_fold": "other two scored cohorts plus E0 embargo",
            "held_fold_used_for_model_or_epoch_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "scored_metrics": metrics(raw.labels[scored], prediction[scored]),
        "folds": fold_reports,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "oof_predictions.npz",
        sample_ids=raw.sample_ids[scored],
        labels=raw.labels[scored],
        users=raw.users[scored],
        logits=logits[scored].astype(np.float32),
        probabilities=softmax(logits[scored], axis=1).astype(np.float32),
        reliability_logits=reliability[scored].astype(np.float32),
        prediction=prediction[scored],
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
