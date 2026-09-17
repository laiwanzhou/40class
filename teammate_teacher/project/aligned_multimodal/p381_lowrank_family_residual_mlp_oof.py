"""Low-rank family residual MLP guarded against the P310 base."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from p361_bidirectional_existing_teacher_gate_oof import COHORTS
from p363_target_class_family_boost_oof import family_parts
from p372_nonnegative_class_family_stacker_oof import choose_threshold


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p381_lowrank_family_residual_mlp_oof_v1"
HIDDEN = 64
BOTTLENECK = 16
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.10
DROPOUT = 0.35
RESIDUAL_SCALE = 0.15
SEEDS = (38101,)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def entropy(probability):
    value = np.clip(probability, 1e-7, 1.0)
    return -(value * np.log(value)).sum(axis=2, keepdims=True) / np.log(40.0)


def make_features(part, family_indices):
    group = np.clip(part["group_probability"].astype(np.float32), 1e-7, 1.0)
    families = np.clip(part["families"][:, family_indices, :].astype(np.float32), 1e-7, 1.0)
    family_log_delta = np.log(families) - np.log(group[:, None, :])
    ordered = np.sort(families, axis=2)[:, :, ::-1]
    scalar = np.concatenate(
        (
            ordered[:, :, :1],
            ordered[:, :, :1] - ordered[:, :, 1:2],
            entropy(families),
        ),
        axis=2,
    )
    base_one_hot = np.eye(40, dtype=np.float32)[part["base"]]
    return np.concatenate(
        (
            np.sqrt(families).reshape(len(group), -1),
            np.clip(family_log_delta, -5.0, 5.0).reshape(len(group), -1),
            scalar.reshape(len(group), -1),
            np.sqrt(group),
            base_one_hot,
        ),
        axis=1,
    ).astype(np.float32)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, family_count):
        super().__init__()
        self.family_count = family_count
        self.norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, BOTTLENECK),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.head = nn.Linear(BOTTLENECK, 40)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, anchor_log_probability):
        residual = torch.tanh(self.head(self.encoder(self.norm(features))))
        return anchor_log_probability + RESIDUAL_SCALE * residual


def class_weights(labels):
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    weight = (len(labels) / np.maximum(40.0 * counts, 1.0)) ** 0.25
    return (weight / weight.mean()).astype(np.float32)


def train_predict(train_part, predict_part, family_indices, seed, device):
    seed_all(seed)
    train_x = make_features(train_part, family_indices)
    predict_x = make_features(predict_part, family_indices)
    train_anchor = np.log(np.clip(train_part["group_probability"], 1e-7, 1.0)).astype(np.float32)
    predict_anchor = np.log(np.clip(predict_part["group_probability"], 1e-7, 1.0)).astype(np.float32)
    labels = train_part["labels"].astype(np.int64)
    model = ResidualMLP(train_x.shape[1], len(family_indices)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(
        TensorDataset(torch.arange(len(labels))),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS * math.ceil(len(labels) / BATCH_SIZE), 1), eta_min=1e-5
    )
    weight = torch.from_numpy(class_weights(labels)).to(device)
    for epoch in range(EPOCHS):
        model.train()
        for (indices,) in loader:
            index = indices.numpy()
            features = torch.from_numpy(train_x[index]).to(device)
            anchor = torch.from_numpy(train_anchor[index]).to(device)
            target = torch.from_numpy(labels[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(features, anchor)
                loss = F.cross_entropy(logits.float(), target, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(predict_x), BATCH_SIZE * 2):
            features = torch.from_numpy(predict_x[start : start + BATCH_SIZE * 2]).to(device)
            anchor = torch.from_numpy(predict_anchor[start : start + BATCH_SIZE * 2]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs.append(model(features, anchor).float().cpu().numpy())
    return np.concatenate(outputs)


def concatenate(items):
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in ("ids", "users", "labels", "base", "group_probability", "families", "cohort")
    }


def softmax(logits):
    value = logits.astype(np.float64)
    value -= value.max(axis=1, keepdims=True)
    probability = np.exp(value)
    return probability / probability.sum(axis=1, keepdims=True)


def main():
    print(
        "P381 tests a ~40k-parameter low-rank residual MLP over six teacher families. "
        "P307 is an immutable log-probability skip and P310 remains the gated base.",
        flush=True,
    )
    parts, family_names, family_members = family_parts()
    family_indices = [index for index, name in enumerate(family_names) if name.endswith("__geometric")]
    selected_names = [family_names[index] for index in family_indices]
    for cohort in COHORTS:
        parts[cohort]["cohort"] = np.full(len(parts[cohort]["labels"]), cohort, dtype=object)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "stage": "P381_lowrank_family_residual_MLP_OOF",
        "status": "complete",
        "protocol": {
            "base": "P310/P315 label-identical teacher",
            "anchor": "P307 group log probability immutable skip",
            "families": selected_names,
            "hidden": HIDDEN,
            "bottleneck": BOTTLENECK,
            "residual_scale": RESIDUAL_SCALE,
            "dropout": DROPOUT,
            "family_dropout": False,
            "seeds": list(SEEDS),
            "source_inner_models_trained_separately": True,
            "held_model_trained_on_pooled_sources": True,
            "held_labels_used_for_training_or_threshold": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    outputs = []
    probabilities = []
    for held_index, held in enumerate(COHORTS):
        source_names = [cohort for cohort in COHORTS if cohort != held]
        source = concatenate([parts[cohort] for cohort in source_names])
        source_probability = np.empty((len(source["labels"]), 40), dtype=float)
        offsets = {
            source_names[0]: (0, len(parts[source_names[0]]["labels"])),
            source_names[1]: (len(parts[source_names[0]]["labels"]), len(source["labels"])),
        }
        for train_name, validation_name in (
            (source_names[0], source_names[1]),
            (source_names[1], source_names[0]),
        ):
            members = [
                train_predict(
                    parts[train_name], parts[validation_name], family_indices,
                    seed + held_index * 1000, device,
                )
                for seed in SEEDS
            ]
            probability = softmax(np.mean(np.stack(members), axis=0))
            lo, hi = offsets[validation_name]
            source_probability[lo:hi] = probability
        threshold = choose_threshold(source, source_probability)

        held_members = [
            train_predict(
                source, parts[held], family_indices,
                seed + held_index * 1000, device,
            )
            for seed in SEEDS
        ]
        held_probability = softmax(np.mean(np.stack(held_members), axis=0))
        rows = np.arange(len(parts[held]["base"]))
        proposal = held_probability.argmax(axis=1)
        score = held_probability[rows, proposal] - held_probability[rows, parts[held]["base"]]
        route = (proposal != parts[held]["base"]) & (score >= float(threshold["threshold"]))
        output = parts[held]["base"].copy()
        output[route] = proposal[route]
        outputs.append(output)
        probabilities.append(held_probability.astype(np.float32))
        labels = parts[held]["labels"]
        base = parts[held]["base"]
        report["cohorts"][held] = {
            "source": source_names,
            "source_threshold": threshold,
            "held": {
                "rows": len(labels),
                "base_correct": int(np.sum(base == labels)),
                "correct": int(np.sum(output == labels)),
                "net": int(np.sum(output == labels) - np.sum(base == labels)),
                "changed": int(route.sum()),
                "rescue": int(np.sum(route & (base != labels) & (output == labels))),
                "harm": int(np.sum(route & (base == labels) & (output != labels))),
            },
        }
        print(json.dumps({"held": held, "threshold": threshold, "result": report["cohorts"][held]["held"]}, ensure_ascii=False), flush=True)

    labels = np.concatenate([parts[cohort]["labels"] for cohort in COHORTS])
    base = np.concatenate([parts[cohort]["base"] for cohort in COHORTS])
    prediction = np.concatenate(outputs)
    fold_nets = [report["cohorts"][cohort]["held"]["net"] for cohort in COHORTS]
    strict_pass = bool(
        all(net > 0 for net in fold_nets)
        or (sum(net > 0 for net in fold_nets) >= 2 and min(fold_nets) >= -1)
    )
    report["aggregate"] = {
        "rows": len(labels),
        "base_correct": int(np.sum(base == labels)),
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "net_vs_p310": int(np.sum(prediction == labels) - np.sum(base == labels)),
        "fold_nets": fold_nets,
        "strict_gate_pass": strict_pass,
        "decision": "eligible_for_test_refit" if strict_pass else "reject_before_test",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "oof_predictions.npz",
        labels=labels,
        base_prediction=base,
        prediction=prediction,
        probability=np.concatenate(probabilities),
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: low-rank bounded residual MLP over fixed teacher families.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
