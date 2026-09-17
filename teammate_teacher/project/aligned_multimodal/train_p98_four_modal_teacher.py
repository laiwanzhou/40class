"""Train the source-only configurable IR/Depth/Skeleton/IMU teacher.

Stages are intentionally limited to:

* ``dry_run``: validate caches, preprocessing, model construction and forward;
* ``h1_cv``: user-disjoint exploration inside H1 with E0 as source-only data;
* ``h2_confirmation``: train a frozen H1 recipe for fixed epochs and evaluate H2.

There is no H3 stage in this entry point.  A later final evaluator must be a
separate, explicit program after the complete recipe has been frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from p91_hierarchical_multimodal_teacher import FAMILIES
from p98_four_modal_teacher_data import (
    FourModalData,
    FourModalPreprocessor,
    build_four_modal_source_data,
    counterfactual_data,
    validate_modalities,
)
from p98_four_modal_teacher_model import FourModalTeacher, FourModalTeacherConfig


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p98_four_modal_teacher_baseline.json"
DEFAULT_OUTPUT = PROJECT / "runs/p98_four_modal_teacher_h1_v1"

AUDIT_PAIRS = {
    "phone_game": (24, 26),
    "phone_call_headphones": (19, 23),
    "drink_medicine": (6, 37),
    "read_turn_pages": (21, 22),
    "tableware_wipe_bowls": (8, 14),
    "tableware_pour": (8, 9),
    "tableware_stir": (8, 10),
}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    patience: int = 14
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 5e-3
    statistics_dim: int = 64
    seeds: tuple[int, ...] = (17, 43, 71)
    label_smoothing: float = 0.025
    family_weight: float = 0.18
    modality_aux_weight: float = 0.10
    reliability_weight: float = 0.08
    anchor_correct_weight: float = 0.20
    anchor_wrong_weight: float = 0.025

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("epochs, patience and batch_size must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage", choices=("dry_run", "h1_cv", "h2_confirmation"), required=True
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-epochs", type=int)
    parser.add_argument("--h1-summary", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold-limit", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], FourModalTeacherConfig, TrainingConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    modalities = validate_modalities(raw.get("modalities", ("ir", "depth", "skeleton", "imu")))
    model_values = dict(raw.get("model", {}))
    model = FourModalTeacherConfig(modalities=modalities, **model_values)
    training_values = dict(raw.get("training", {}))
    if "seeds" in training_values:
        training_values["seeds"] = tuple(int(value) for value in training_values["seeds"])
    training = TrainingConfig(**training_values)
    normalized_model = asdict(model)
    # Keep the original T0 digest stable when later structural fields are absent.
    extension_fields = {
        "expert_residual",
        "expert_mixture",
        "anchor_margin",
        "residual_scale",
    }
    normalized_model = {
        key: value
        for key, value in normalized_model.items()
        if key not in extension_fields or key in model_values
    }
    normalized = {
        "name": str(raw.get("name", path.stem)),
        "modalities": list(modalities),
        "model": normalized_model,
        "training": asdict(training),
    }
    normalized["model"]["modalities"] = list(modalities)
    normalized["training"]["seeds"] = list(training.seeds)
    return normalized, model, training


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class FourModalDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, data: FourModalData, indices: np.ndarray, weights: np.ndarray) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        item = {
            name: torch.from_numpy(values[row]) for name, values in self.data.streams.items()
        }
        item.update(
            {
                name: torch.from_numpy(values[row])
                for name, values in self.data.statistics.items()
            }
        )
        item.update(
            {
                "skeleton_sequence": torch.from_numpy(self.data.skeleton_sequence[row]),
                "skeleton_mask": torch.from_numpy(self.data.skeleton_mask[row]),
                "imu_sequence": torch.from_numpy(self.data.imu_sequence[row]),
                "imu_mask": torch.from_numpy(self.data.imu_mask[row]),
                "label": torch.tensor(self.data.labels[row], dtype=torch.long),
                "family": torch.tensor(FAMILIES[self.data.labels[row]], dtype=torch.long),
                "base_prediction": torch.tensor(
                    self.data.base_prediction[row], dtype=torch.long
                ),
                "expert_probability": torch.from_numpy(
                    self.data.expert_probability[row]
                ),
                "weight": torch.tensor(self.weights[row], dtype=torch.float32),
                "row": torch.tensor(row, dtype=torch.long),
            }
        )
        item.update(
            {
                f"{name}_available": torch.tensor(values[row], dtype=torch.float32)
                for name, values in self.data.modality_available.items()
            }
        )
        return item


def sample_weights(data: FourModalData, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    counts = np.bincount(data.labels[indices], minlength=40).astype(np.float64)
    class_weight = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    user_values, user_counts = np.unique(data.users[indices], return_counts=True)
    user_weight = {
        user: 1.0 / np.sqrt(count) for user, count in zip(user_values, user_counts)
    }
    output = np.ones(len(data.labels), dtype=np.float32)
    output[indices] = np.asarray(
        [class_weight[data.labels[row]] * user_weight[data.users[row]] for row in indices],
        dtype=np.float32,
    )
    output[indices] /= output[indices].mean()
    output[indices] *= np.where(
        data.base_prediction[indices] == data.labels[indices], 1.0, 1.30
    ).astype(np.float32)
    return output


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_model(config: FourModalTeacherConfig, data: FourModalData) -> FourModalTeacher:
    return FourModalTeacher(
        config,
        stream_dims={name: int(values.shape[-1]) for name, values in data.streams.items()},
        stream_groups=data.stream_groups,
        statistic_dims={
            name: int(values.shape[-1]) for name, values in data.statistics.items()
        },
        statistic_groups=data.statistic_groups,
        expert_names=data.expert_names,
        expert_groups=data.expert_groups,
    )


@torch.no_grad()
def infer_outputs(
    model: FourModalTeacher,
    data: FourModalData,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(
        FourModalDataset(data, indices, np.ones(len(data.labels), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    logits: list[np.ndarray] = []
    reliability: list[np.ndarray] = []
    importance: list[np.ndarray] = []
    representation: list[np.ndarray] = []
    anchor_logits: list[np.ndarray] = []
    residual_logits: list[np.ndarray] = []
    expert_gate: list[np.ndarray] = []
    modality_logits: dict[str, list[np.ndarray]] = {
        name: [] for name in model.modalities
    }
    modality_embeddings: dict[str, list[np.ndarray]] = {
        name: [] for name in model.modalities
    }
    rows: list[np.ndarray] = []
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(batch)
        logits.append(output["logits"].float().cpu().numpy())
        reliability.append(output["reliability_logits"].float().cpu().numpy())
        importance.append(output["modality_importance"].float().cpu().numpy())
        representation.append(output["representation"].float().cpu().numpy())
        anchor_logits.append(output["anchor_logits"].float().cpu().numpy())
        residual_logits.append(output["residual_logits"].float().cpu().numpy())
        expert_gate.append(output["expert_gate"].float().cpu().numpy())
        for name in model.modalities:
            modality_logits[name].append(
                output["modality_logits"][name].float().cpu().numpy()
            )
            modality_embeddings[name].append(
                output["modality_embeddings"][name].float().cpu().numpy()
            )
        rows.append(batch["row"].cpu().numpy())
    observed = np.concatenate(rows)
    if not np.array_equal(observed, np.asarray(indices, dtype=np.int64)):
        raise ValueError("inference order changed")
    return {
        "logits": np.concatenate(logits),
        "reliability_logits": np.concatenate(reliability),
        "modality_importance": np.concatenate(importance),
        "representation": np.concatenate(representation),
        "anchor_logits": np.concatenate(anchor_logits),
        "residual_logits": np.concatenate(residual_logits),
        "expert_gate": np.concatenate(expert_gate),
        "modality_logits": {
            name: np.concatenate(values) for name, values in modality_logits.items()
        },
        "modality_embeddings": {
            name: np.concatenate(values) for name, values in modality_embeddings.items()
        },
    }


def train_model(
    model_config: FourModalTeacherConfig,
    training: TrainingConfig,
    data: FourModalData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    device: torch.device,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[FourModalTeacher, dict[str, Any], dict[str, Any] | None]:
    set_seed(seed)
    model = build_model(model_config, data).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    epochs = int(fixed_epochs or training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weights = sample_weights(data, train_indices)
    loader = DataLoader(
        FourModalDataset(data, train_indices, weights),
        batch_size=training.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    if validation_indices is not None and model.config.expert_residual:
        initial_validation = infer_outputs(
            model, data, validation_indices, device, training.batch_size
        )
        best_accuracy = float(
            np.mean(
                initial_validation["logits"].argmax(axis=1)
                == data.labels[validation_indices]
            )
        )
        best_state = copy.deepcopy(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
        )
        history.append({"epoch": 0.0, "validation_accuracy": best_accuracy})
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(batch)
                main = F.cross_entropy(
                    output["logits"],
                    batch["label"],
                    reduction="none",
                    label_smoothing=training.label_smoothing,
                )
                main = (main * batch["weight"]).mean()
                family = F.cross_entropy(output["family_logits"], batch["family"])
                modality_aux = torch.stack(
                    [
                        F.cross_entropy(values, batch["label"])
                        for values in output["modality_logits"].values()
                    ]
                ).mean()
                base_wrong = (batch["base_prediction"] != batch["label"]).float()
                reliability = F.binary_cross_entropy_with_logits(
                    output["reliability_logits"], base_wrong
                )
                anchor_ce = F.cross_entropy(
                    output["logits"], batch["base_prediction"], reduction="none"
                )
                anchor_weight = torch.where(
                    batch["base_prediction"] == batch["label"],
                    training.anchor_correct_weight,
                    training.anchor_wrong_weight,
                )
                anchor = (anchor_ce * anchor_weight).mean()
                if output["expert_gate"].shape[1]:
                    gate_entropy = -(
                        output["expert_gate"].clamp_min(1e-8).log()
                        * output["expert_gate"]
                    ).sum(dim=1).mean()
                else:
                    gate_entropy = torch.zeros((), device=device)
                loss = (
                    main
                    + training.family_weight * family
                    + training.modality_aux_weight * modality_aux
                    + training.reliability_weight * reliability
                    + anchor
                    - 0.006 * gate_entropy
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        record: dict[str, float] = {"epoch": float(epoch), "loss": float(np.mean(losses))}
        if validation_indices is not None:
            validation = infer_outputs(
                model, data, validation_indices, device, training.batch_size
            )
            accuracy = float(
                np.mean(validation["logits"].argmax(axis=1) == data.labels[validation_indices])
            )
            record["validation_accuracy"] = accuracy
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = accuracy
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                stale = 0
            else:
                stale += 1
            if stale >= training.patience:
                history.append(record)
                break
        history.append(record)
    if validation_indices is not None:
        if best_state is None:
            raise RuntimeError("training produced no validation checkpoint")
        model.load_state_dict(best_state)
        final_validation = infer_outputs(
            model, data, validation_indices, device, training.batch_size
        )
    else:
        best_epoch = epochs
        final_validation = None
    audit = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_accuracy": float(best_accuracy),
        "epochs_ran": int(sum(item["epoch"] > 0 for item in history)),
        "parameter_count": model.parameter_count,
        "history": history,
    }
    return model, audit, final_validation


def numpy_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def subset_audit(
    labels: np.ndarray,
    logits: np.ndarray,
    base_prediction: np.ndarray,
    users: np.ndarray,
) -> dict[str, Any]:
    probability = numpy_softmax(logits)
    prediction = probability.argmax(axis=1)
    correct = prediction == labels
    base_correct = base_prediction == labels
    result: dict[str, Any] = {
        "samples": int(len(labels)),
        "top1_accuracy": float(accuracy_score(labels, prediction)),
        "top1_correct": int(correct.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "base_accuracy": float(np.mean(base_correct)),
        "base_correct": int(base_correct.sum()),
        "rescue": int(np.sum(~base_correct & correct)),
        "harm": int(np.sum(base_correct & ~correct)),
        "changed": int(np.sum(prediction != base_prediction)),
    }
    result["net"] = result["top1_correct"] - result["base_correct"]
    for k in (1, 3, 5, 10):
        top = np.argpartition(probability, -k, axis=1)[:, -k:]
        result[f"top{k}_coverage"] = float(
            np.mean(np.any(top == labels[:, None], axis=1))
        )
    confidence = probability.max(axis=1)
    low = confidence < 0.95
    result["confidence_lt_095"] = {
        "samples": int(low.sum()),
        "accuracy": float(correct[low].mean()) if low.any() else None,
        "errors": int(np.sum(low & ~correct)),
    }
    base_wrong = ~base_correct
    result["base_error_pool"] = {
        "samples": int(base_wrong.sum()),
        "teacher_top1_accuracy": float(correct[base_wrong].mean()) if base_wrong.any() else None,
        "teacher_top5_coverage": float(
            np.mean(
                np.any(
                    np.argpartition(probability[base_wrong], -5, axis=1)[:, -5:]
                    == labels[base_wrong, None],
                    axis=1,
                )
            )
        )
        if base_wrong.any()
        else None,
    }
    result["per_user"] = {}
    for user in sorted(np.unique(users).tolist()):
        selected = users == user
        result["per_user"][user] = {
            "samples": int(selected.sum()),
            "accuracy": float(correct[selected].mean()),
            "base_accuracy": float(base_correct[selected].mean()),
            "net": int(correct[selected].sum() - base_correct[selected].sum()),
        }
    result["confusion_pairs"] = {}
    for name, pair in AUDIT_PAIRS.items():
        selected = np.isin(labels, pair)
        result["confusion_pairs"][name] = {
            "class_ids": list(pair),
            "samples": int(selected.sum()),
            "accuracy": float(correct[selected].mean()) if selected.any() else None,
            "base_accuracy": float(base_correct[selected].mean()) if selected.any() else None,
            "net": int(correct[selected].sum() - base_correct[selected].sum()),
        }
    return result


def mean_output(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    modalities = tuple(outputs[0]["modality_logits"])
    return {
        "logits": np.mean([value["logits"] for value in outputs], axis=0),
        "reliability_logits": np.mean(
            [value["reliability_logits"] for value in outputs], axis=0
        ),
        "modality_importance": np.mean(
            [value["modality_importance"] for value in outputs], axis=0
        ),
        "representation": np.mean(
            [value["representation"] for value in outputs], axis=0
        ),
        "anchor_logits": np.mean(
            [value["anchor_logits"] for value in outputs], axis=0
        ),
        "residual_logits": np.mean(
            [value["residual_logits"] for value in outputs], axis=0
        ),
        "expert_gate": np.mean(
            [value["expert_gate"] for value in outputs], axis=0
        ),
        "modality_logits": {
            name: np.mean([value["modality_logits"][name] for value in outputs], axis=0)
            for name in modalities
        },
        "modality_embeddings": {
            name: np.mean(
                [value["modality_embeddings"][name] for value in outputs], axis=0
            )
            for name in modalities
        },
    }


def save_predictions(
    path: Path,
    data: FourModalData,
    indices: np.ndarray,
    output: dict[str, Any],
) -> None:
    payload: dict[str, np.ndarray] = {
        "sample_ids": data.sample_ids[indices],
        "labels": data.labels[indices],
        "users": data.users[indices],
        "base_prediction": data.base_prediction[indices],
        "direct_logits": output["logits"].astype(np.float32),
        "direct_prediction": output["logits"].argmax(axis=1).astype(np.int64),
        "reliability_logits": output["reliability_logits"].astype(np.float32),
        "modality_importance": output["modality_importance"].astype(np.float32),
        "fused_embedding": output["representation"].astype(np.float32),
        "anchor_logits": output["anchor_logits"].astype(np.float32),
        "residual_logits": output["residual_logits"].astype(np.float32),
    }
    for name, values in output["modality_logits"].items():
        payload[f"{name}_logits"] = values.astype(np.float32)
    for name, values in output["modality_embeddings"].items():
        payload[f"{name}_embedding"] = values.astype(np.float32)
    if output["expert_gate"].shape[1]:
        payload["expert_gate"] = output["expert_gate"].astype(np.float32)
    np.savez_compressed(path, **payload)


def dry_run(
    output_dir: Path,
    raw_config: dict[str, Any],
    model_config: FourModalTeacherConfig,
    training: TrainingConfig,
    device: torch.device,
    smoke: bool,
) -> None:
    raw = build_four_modal_source_data(model_config.modalities, include_h2=False)
    train_indices = np.concatenate(
        (raw.boundaries["H1_selection"], raw.boundaries["E0_source_only"])
    )
    if smoke:
        train_indices = train_indices[: max(80, training.statistics_dim + 1)]
    preprocessor = FourModalPreprocessor(training.statistics_dim, training.seeds[0]).fit(
        raw, train_indices
    )
    data = preprocessor.transform(raw)
    model = build_model(model_config, data).to(device).eval()
    target = raw.boundaries["H1_selection"][:4]
    batch = next(
        iter(
            DataLoader(
                FourModalDataset(
                    data, target, np.ones(len(data.labels), dtype=np.float32)
                ),
                batch_size=len(target),
                shuffle=False,
            )
        )
    )
    batch = move(batch, device)
    with torch.no_grad():
        result = model(batch)
    contract = {
        "protocol": "source-only H1/E0 contract; H2/H3 unavailable in this stage",
        "config": raw_config,
        "config_sha256": config_digest(raw_config),
        "data": raw.summary(),
        "preprocessor": preprocessor.summary(),
        "model": {
            "parameter_count": model.parameter_count,
            "logits": list(result["logits"].shape),
            "fused_embedding": list(result["representation"].shape),
            "modality_importance": list(result["modality_importance"].shape),
            "modality_logits": {
                name: list(values.shape)
                for name, values in result["modality_logits"].items()
            },
            "expert_gate": list(result["expert_gate"].shape),
        },
    }
    write_json(output_dir / "contract.json", contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)


def run_h1_cv(
    output_dir: Path,
    raw_config: dict[str, Any],
    model_config: FourModalTeacherConfig,
    training: TrainingConfig,
    device: torch.device,
    smoke: bool,
    fold_limit: int | None,
) -> None:
    raw = build_four_modal_source_data(model_config.modalities, include_h2=False)
    h1 = raw.boundaries["H1_selection"]
    e0 = raw.boundaries["E0_source_only"]
    users = sorted(np.unique(raw.users[h1]).tolist())
    if fold_limit is not None:
        users = users[:fold_limit]
    seeds = training.seeds[:1] if smoke else training.seeds
    oof_logits = np.full((len(raw.labels), 40), np.nan, dtype=np.float32)
    oof_reliability = np.full(len(raw.labels), np.nan, dtype=np.float32)
    oof_importance = np.full(
        (len(raw.labels), len(model_config.modalities)), np.nan, dtype=np.float32
    )
    oof_anchor_logits = np.full((len(raw.labels), 40), np.nan, dtype=np.float32)
    oof_residual_logits = np.full((len(raw.labels), 40), np.nan, dtype=np.float32)
    oof_expert_gate = np.full(
        (len(raw.labels), len(raw.expert_names)), np.nan, dtype=np.float32
    )
    oof_modality_logits = {
        name: np.full((len(raw.labels), 40), np.nan, dtype=np.float32)
        for name in model_config.modalities
    }
    seed_audits: list[dict[str, Any]] = []
    selected_epochs: list[int] = []
    for fold_index, held_user in enumerate(users):
        validation = h1[raw.users[h1] == held_user]
        train = np.concatenate((h1[raw.users[h1] != held_user], e0))
        if smoke:
            train = train[: max(96, training.statistics_dim + 1)]
            validation = validation[:32]
        if set(raw.users[train]) & set(raw.users[validation]):
            raise RuntimeError(f"user leakage in H1 fold {held_user}")
        preprocessor = FourModalPreprocessor(
            training.statistics_dim, training.seeds[0] + fold_index
        ).fit(raw, train)
        data = preprocessor.transform(raw)
        fold_outputs: list[dict[str, Any]] = []
        for seed in seeds:
            print(
                f"[H1] fold={fold_index + 1}/{len(users)} held_user={held_user} "
                f"seed={seed} train={len(train)} validation={len(validation)}",
                flush=True,
            )
            model, audit, values = train_model(
                model_config,
                training,
                data,
                train,
                validation,
                device,
                seed + fold_index * 1000,
                fixed_epochs=1 if smoke else None,
            )
            if values is None:
                raise RuntimeError("H1 fold returned no validation output")
            audit.update(
                {
                    "fold": fold_index,
                    "held_user": held_user,
                    "train_samples": int(len(train)),
                    "validation_samples": int(len(validation)),
                }
            )
            seed_audits.append(audit)
            selected_epochs.append(int(audit["best_epoch"]))
            fold_outputs.append(values)
            print(
                f"[H1] completed held_user={held_user} seed={seed} "
                f"best_epoch={audit['best_epoch']} "
                f"best_accuracy={audit.get('best_accuracy', float('nan')):.5f}",
                flush=True,
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "preprocessor": preprocessor,
                    "audit": audit,
                },
                output_dir / f"h1_fold{fold_index}_{held_user}_seed{seed}.pt",
            )
        ensemble = mean_output(fold_outputs)
        oof_logits[validation] = ensemble["logits"]
        oof_reliability[validation] = ensemble["reliability_logits"]
        oof_importance[validation] = ensemble["modality_importance"]
        oof_anchor_logits[validation] = ensemble["anchor_logits"]
        oof_residual_logits[validation] = ensemble["residual_logits"]
        if ensemble["expert_gate"].shape[1]:
            oof_expert_gate[validation] = ensemble["expert_gate"]
        for name in model_config.modalities:
            oof_modality_logits[name][validation] = ensemble["modality_logits"][name]
    evaluated = h1[np.isfinite(oof_logits[h1]).all(axis=1)]
    if not smoke and len(evaluated) != len(h1):
        raise RuntimeError("formal H1 CV did not predict every H1 row")
    audit = subset_audit(
        raw.labels[evaluated],
        oof_logits[evaluated],
        raw.base_prediction[evaluated],
        raw.users[evaluated],
    )
    fixed_epochs = max(1, int(round(float(np.median(selected_epochs)))))
    modality_audits = {
        name: subset_audit(
            raw.labels[evaluated],
            values[evaluated],
            raw.base_prediction[evaluated],
            raw.users[evaluated],
        )
        for name, values in oof_modality_logits.items()
    }
    summary = {
        "stage": "h1_cv",
        "protocol": "H1 leave-one-user-out exploration with E0 source-only training rows; H2/H3 unread",
        "config": raw_config,
        "config_sha256": config_digest(raw_config),
        "data": raw.summary(),
        "evaluated_users": users,
        "evaluated_samples": int(len(evaluated)),
        "fixed_epochs_for_h2": fixed_epochs,
        "seed_audits": seed_audits,
        "audit": audit,
        "modality_heads": modality_audits,
        "mean_modality_importance": {
            name: float(oof_importance[evaluated, index].mean())
            for index, name in enumerate(model_config.modalities)
        },
        "residual_rms": float(np.sqrt(np.mean(oof_residual_logits[evaluated] ** 2))),
    }
    if np.isfinite(oof_expert_gate[evaluated]).all():
        summary["mean_expert_gate"] = {
            name: float(oof_expert_gate[evaluated, index].mean())
            for index, name in enumerate(raw.expert_names)
        }
    prediction_payload: dict[str, np.ndarray] = {
        "sample_ids": raw.sample_ids[evaluated],
        "labels": raw.labels[evaluated],
        "users": raw.users[evaluated],
        "base_prediction": raw.base_prediction[evaluated],
        "direct_logits": oof_logits[evaluated],
        "reliability_logits": oof_reliability[evaluated],
        "modality_importance": oof_importance[evaluated],
        "anchor_logits": oof_anchor_logits[evaluated],
        "residual_logits": oof_residual_logits[evaluated],
    }
    if np.isfinite(oof_expert_gate[evaluated]).all():
        prediction_payload["expert_gate"] = oof_expert_gate[evaluated]
    for name, values in oof_modality_logits.items():
        prediction_payload[f"{name}_logits"] = values[evaluated]
    np.savez_compressed(
        output_dir / "h1_oof_predictions.npz",
        **prediction_payload,
    )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def resolve_fixed_epochs(args: argparse.Namespace, digest: str) -> int:
    if args.h1_summary is not None:
        summary = json.loads(args.h1_summary.read_text(encoding="utf-8"))
        if summary.get("stage") != "h1_cv":
            raise ValueError("--h1-summary is not an H1 CV summary")
        if summary.get("config_sha256") != digest:
            raise ValueError("H1 summary config differs from the frozen H2 config")
        frozen = int(summary["fixed_epochs_for_h2"])
        if args.fixed_epochs is not None and int(args.fixed_epochs) != frozen:
            raise ValueError("--fixed-epochs conflicts with the H1-frozen epoch count")
        return frozen
    if args.fixed_epochs is None:
        raise ValueError("H2 confirmation requires --h1-summary or --fixed-epochs")
    if args.fixed_epochs <= 0:
        raise ValueError("--fixed-epochs must be positive")
    return int(args.fixed_epochs)


def run_h2_confirmation(
    output_dir: Path,
    raw_config: dict[str, Any],
    model_config: FourModalTeacherConfig,
    training: TrainingConfig,
    device: torch.device,
    fixed_epochs: int,
    smoke: bool,
) -> None:
    raw = build_four_modal_source_data(model_config.modalities, include_h2=True)
    train = np.concatenate(
        (raw.boundaries["H1_selection"], raw.boundaries["E0_source_only"])
    )
    target = raw.boundaries["H2_confirmation"]
    if set(raw.users[train]) & set(raw.users[target]):
        raise RuntimeError("H2 confirmation is not subject-disjoint")
    if smoke:
        train = train[: max(96, training.statistics_dim + 1)]
        target = target[:32]
        fixed_epochs = 1
    preprocessor = FourModalPreprocessor(training.statistics_dim, training.seeds[0]).fit(
        raw, train
    )
    data = preprocessor.transform(raw)
    seeds = training.seeds[:1] if smoke else training.seeds
    outputs: list[dict[str, Any]] = []
    model_audits: list[dict[str, Any]] = []
    counterfactual_outputs: dict[str, list[dict[str, Any]]] = {
        f"{mode}_{modality}": []
        for modality in model_config.modalities
        for mode in ("zero", "shuffle")
    }
    for seed in seeds:
        print(
            f"[H2] seed={seed} train={len(train)} target={len(target)} "
            f"fixed_epochs={fixed_epochs}",
            flush=True,
        )
        model, audit, _ = train_model(
            model_config,
            training,
            data,
            train,
            None,
            device,
            seed,
            fixed_epochs=fixed_epochs,
        )
        audit.update(
            {
                "train_samples": int(len(train)),
                "target_samples": int(len(target)),
                "target_used_for_early_stopping": False,
            }
        )
        model_audits.append(audit)
        outputs.append(infer_outputs(model, data, target, device, training.batch_size))
        for modality in model_config.modalities:
            for mode in ("zero", "shuffle"):
                changed = counterfactual_data(data, modality, mode, seed=17)
                counterfactual_outputs[f"{mode}_{modality}"].append(
                    infer_outputs(model, changed, target, device, training.batch_size)
                )
        print(f"[H2] completed seed={seed}", flush=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": asdict(model_config),
                "preprocessor": preprocessor,
                "audit": audit,
            },
            output_dir / f"h2_seed{seed}_teacher.pt",
        )
    ensemble = mean_output(outputs)
    direct_audit = subset_audit(
        raw.labels[target],
        ensemble["logits"],
        raw.base_prediction[target],
        raw.users[target],
    )
    modality_audits = {
        name: subset_audit(
            raw.labels[target],
            values,
            raw.base_prediction[target],
            raw.users[target],
        )
        for name, values in ensemble["modality_logits"].items()
    }
    counterfactual_audit: dict[str, Any] = {}
    for name, values in counterfactual_outputs.items():
        changed = mean_output(values)
        audit = subset_audit(
            raw.labels[target],
            changed["logits"],
            raw.base_prediction[target],
            raw.users[target],
        )
        audit["delta_correct_vs_direct"] = (
            audit["top1_correct"] - direct_audit["top1_correct"]
        )
        counterfactual_audit[name] = audit
    summary = {
        "stage": "h2_confirmation",
        "protocol": "frozen H1 recipe; H1+E0 train; H2 evaluated once without early stopping; H3 unavailable",
        "config": raw_config,
        "config_sha256": config_digest(raw_config),
        "data": raw.summary(),
        "fixed_epochs": int(fixed_epochs),
        "model_audits": model_audits,
        "direct": direct_audit,
        "modality_heads": modality_audits,
        "mean_modality_importance": {
            name: float(ensemble["modality_importance"][:, index].mean())
            for index, name in enumerate(model_config.modalities)
        },
        "counterfactuals": counterfactual_audit,
    }
    save_predictions(output_dir / "h2_predictions.npz", raw, target, ensemble)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    raw_config, model_config, training = load_config(args.config.resolve())
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "stage": args.stage,
            "config_path": str(args.config.resolve()),
            "config": raw_config,
            "config_sha256": config_digest(raw_config),
            "device": str(device),
            "smoke": bool(args.smoke),
            "h3_access": False,
        },
    )
    if args.stage == "dry_run":
        dry_run(output_dir, raw_config, model_config, training, device, args.smoke)
        return
    if args.stage == "h1_cv":
        run_h1_cv(
            output_dir,
            raw_config,
            model_config,
            training,
            device,
            args.smoke,
            args.fold_limit,
        )
        return
    fixed_epochs = resolve_fixed_epochs(args, config_digest(raw_config))
    run_h2_confirmation(
        output_dir,
        raw_config,
        model_config,
        training,
        device,
        fixed_epochs,
        args.smoke,
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
