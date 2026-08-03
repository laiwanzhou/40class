from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import collect_predictions
from src.train_unimodal import build_datasets, build_model, load_config, set_seed


RUN_ID = "depth_ir_pose_roi_expert_fold0_14train_4val"
RUN_DIR = PROJECT_ROOT / "outputs/depth_ir_pose_roi_expert_fold0/depth_ir" / RUN_ID
SHUFFLE_SEED = 20260803


def config_args() -> argparse.Namespace:
    return argparse.Namespace(
        data_root=None, manifest=None, fold=None, output_root=None, device=None, seed=None,
        smoke_test=False, max_epochs=None, num_workers=0, max_train_batches=None,
        max_val_batches=None, run_id=None,
    )


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs/experiments/depth_ir_pose_roi_expert.yaml", config_args())
    set_seed(int(config["seed"]))
    _, dataset = build_datasets(config)
    model = build_model(config, dataset[0]).cuda()
    checkpoint = torch.load(RUN_DIR / "best_model.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    results: dict[str, dict[str, object]] = {}
    predictions_by_mode: dict[str, np.ndarray] = {}
    logits_by_mode: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for mode in ("normal", "masked", "shuffled"):
        dataset.set_ir_mode(mode, seed=SHUFFLE_SEED)
        if mode == "shuffled" and np.any(dataset.ir_permutation == np.arange(len(dataset))):
            raise RuntimeError("Shuffled IR permutation contains a self-pair.")
        loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, pin_memory=True)
        output = collect_predictions(model, loader, criterion, torch.device("cuda"), amp_enabled=bool(config["amp"]))
        metrics = output["metrics"]
        labels = np.asarray(output["labels"])
        logits = np.asarray(output["logits"])
        predictions = logits.argmax(axis=1)
        predictions_by_mode[mode] = predictions
        logits_by_mode[mode] = logits
        results[mode] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "loss": metrics["loss"],
            "per_class_f1": metrics["per_class_f1"],
            "per_class_recall": metrics["per_class_recall"],
            "prediction_change_rate_vs_normal": 0.0 if mode == "normal" else float(np.mean(predictions != predictions_by_mode["normal"])),
            "prediction_change_count_vs_normal": 0 if mode == "normal" else int(np.sum(predictions != predictions_by_mode["normal"])),
            "logit_mean_absolute_change_vs_normal": 0.0 if mode == "normal" else float(np.mean(np.abs(logits - logits_by_mode["normal"]))),
        }
        for label, action in enumerate(dataset.class_names):
            rows.append(
                {
                    "mode": mode,
                    "expert_label": label,
                    "original_class_id": dataset.original_class_ids[label],
                    "action_name": action,
                    "support": int(np.sum(labels == label)),
                    "f1": metrics["per_class_f1"][label],
                    "recall": metrics["per_class_recall"][label],
                }
            )
        print(mode, json.dumps(results[mode]))
    payload = {
        "checkpoint": str(RUN_DIR / "best_model.pt"),
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_has_self_pair": bool(np.any(dataset.ir_permutation == np.arange(len(dataset)))),
        "sample_count": len(dataset),
        "modes": results,
    }
    (RUN_DIR / "ir_ablation_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(RUN_DIR / "ir_ablation_per_class.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        RUN_DIR / "ir_ablation_predictions.npz",
        sample_ids=np.asarray([sample["sample_id"] for sample in dataset.samples]),
        labels=labels,
        normal_predictions=predictions_by_mode["normal"],
        masked_predictions=predictions_by_mode["masked"],
        shuffled_predictions=predictions_by_mode["shuffled"],
        normal_logits=logits_by_mode["normal"],
        masked_logits=logits_by_mode["masked"],
        shuffled_logits=logits_by_mode["shuffled"],
        shuffled_source_indices=dataset.ir_permutation,
    )


if __name__ == "__main__":
    main()
