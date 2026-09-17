"""Refit frozen P142/P144/P149 token heads on all Train and infer Test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from p142_vjepa_token_transformer_oof import build_repeat_pairs, train_fold
from p90_teacher_common import load_protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN = REPO / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1/features.npy"
TEST = REPO / "runs/p171_vjepa2_dense24_test_v1"
OUTPUT = HERE / "runs/p172_vjepa_token_heads_test_v1"
SEEDS = (14201, 14217, 14233)
SUBSETS = {
    "all": np.arange(24),
    "hand_interaction": np.asarray((14, 17, 20, 23)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, default=TRAIN)
    parser.add_argument("--test-dir", type=Path, default=TEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def variant_args(repeat_weight: float, token_subset: str) -> SimpleNamespace:
    return SimpleNamespace(
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
        repeat_consistency_weight=repeat_weight,
        repeat_embedding_weight=0.0,
        repeat_same_label_only=False,
        class_triplet_weight=0.0,
        triplet_margin=0.20,
        teacher_weight=0.0,
        domain_adversarial_weight=0.0,
        token_subset=token_subset,
    )


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    train = np.load(args.train_cache.resolve(), mmap_mode="r")
    test = np.load(args.test_dir.resolve() / "features.npy", mmap_mode="r")
    done = np.load(args.test_dir.resolve() / "done.npy").astype(bool)
    test_ids = np.load(args.test_dir.resolve() / "sample_ids.npy").astype(str)
    if train.shape != (2914, 24, 1024):
        raise RuntimeError(f"P172 Train feature shape changed: {train.shape}")
    if test.shape != (401, 24, 1024) or not done.all() or len(test_ids) != 401:
        raise RuntimeError("P172 requires the complete 401-row P171 cache")
    repeat_pairs = build_repeat_pairs(protocol)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        "p142_all": ("all", 0.0),
        "p144_hand_interaction": ("hand_interaction", 0.0),
        "p149_repeat_consistency": ("all", 0.20),
    }
    saved = {"sample_ids": test_ids}
    report = {
        "stage": "P172_VJEPA_token_heads_all2914_to_Test",
        "status": "complete",
        "protocol": {
            "train_rows": len(protocol.labels),
            "test_rows": len(test_ids),
            "test_labels_read": False,
            "backbone_frozen": True,
            "seeds": list(SEEDS),
            "epochs": 35,
        },
        "variants": {},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, (subset_name, repeat_weight) in variants.items():
        subset = SUBSETS[subset_name]
        # train_fold indexes one shared matrix; concatenate the frozen Train and
        # Test token rows while limiting train_indices to Train only.
        values = np.concatenate(
            (
                np.asarray(train[:, subset], dtype=np.float16),
                np.asarray(test[:, subset], dtype=np.float16),
            ),
            axis=0,
        )
        labels = np.concatenate(
            (protocol.labels.astype(np.int64), np.zeros(len(test_ids), dtype=np.int64))
        )
        domains = np.concatenate(
            (protocol.fold_id.astype(np.int64), np.zeros(len(test_ids), dtype=np.int64))
        )
        train_indices = np.arange(len(protocol.labels), dtype=np.int64)
        test_indices = np.arange(len(protocol.labels), len(labels), dtype=np.int64)
        config = variant_args(repeat_weight, subset_name)
        members = []
        for seed in SEEDS:
            members.append(
                train_fold(
                    values,
                    labels,
                    train_indices,
                    test_indices,
                    domains,
                    seed,
                    config,
                    device,
                    repeat_pairs if repeat_weight > 0 else None,
                    None,
                )
            )
        logits = np.mean(np.stack(members), axis=0).astype(np.float64)
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        saved[f"{name}_logits"] = logits.astype(np.float32)
        saved[f"{name}_probability"] = probability.astype(np.float32)
        report["variants"][name] = {
            "token_subset": subset_name,
            "token_count": len(subset),
            "repeat_consistency_weight": repeat_weight,
            "anonymous_repeat_pairs": int(len(repeat_pairs)) if repeat_weight else 0,
            "mean_confidence": float(probability.max(axis=1).mean()),
        }
        del values
    np.savez_compressed(output / "test_predictions.npz", **saved)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
