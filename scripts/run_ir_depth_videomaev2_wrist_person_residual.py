from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/ir_depth_videomaev2_wrist_person_residual.yaml"
)

from scripts.cache_ir_depth_videomaev2_p2a import _atomic_npz, _atomic_write_text, _metrics
from scripts.cache_ir_depth_videomaev2_p3r1 import (
    _project_path,
    validate_cache_membership,
)
from src.models.ir_depth_videomaev2_teacher import sha256_file
from src.models.wrist_person_residual_fusion import (
    WristPersonResidualFusion,
    wrist_person_loss,
)


class CachedWristPersonDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        return {
            "view_embeddings": torch.from_numpy(
                self.cache["fused_view_embeddings"][row]
            ).float(),
            "anchor_logits": torch.from_numpy(self.cache["route_logits"][row, 0]).float(),
            "anchor_view_weights": torch.from_numpy(
                self.cache["route_view_weights"][row, 0]
            ).float(),
            "availability": torch.from_numpy(
                self.cache["availability"][row].any(axis=0)
            ).bool(),
            "num_frames": torch.tensor(float(self.cache["num_frames"][row])),
            "label": torch.tensor(int(self.cache["labels"][row]), dtype=torch.long),
            "index": torch.tensor(row, dtype=torch.long),
        }


def load_wrist_person_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("wrist-person config must be a mapping")
    split, cv = config.get("split", {}), config.get("cv", {})
    model, policy = config.get("model", {}), config.get("policy", {})
    if config.get("stage") != "P4-WP0":
        raise ValueError("wrist-person stage changed")
    train_users = [str(value) for value in split.get("train_user_ids", [])]
    validation_users = [str(value) for value in split.get("validation_user_ids", [])]
    if set(train_users) & set(validation_users) or validation_users != ["user6", "user7"]:
        raise ValueError("wrist-person user boundary changed")
    fold_users = [str(user) for fold in cv.get("folds", []) for user in fold]
    if len(fold_users) != len(set(fold_users)) or set(fold_users) != set(train_users):
        raise ValueError("wrist-person folds must partition train users")
    expected_candidates = [
        "person_fixed10",
        "person_no_margin",
        "person_margin",
        "person_margin_no_aux",
    ]
    if model.get("candidates") != expected_candidates:
        raise ValueError("wrist-person candidates changed")
    for key in (
        "update_videomae",
        "validation_users_enter_gradient",
        "validation_users_enter_sampler",
        "validation_users_enter_cv_selection",
    ):
        if policy.get(key) is not False:
            raise ValueError("wrist-person isolation policy changed")
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _candidate_contract(candidate: str) -> tuple[str, bool]:
    if candidate == "person_fixed10":
        return "fixed10", True
    if candidate == "person_no_margin":
        return "no_margin", True
    if candidate == "person_margin":
        return "margin", True
    if candidate == "person_margin_no_aux":
        return "margin", False
    raise ValueError(f"unknown wrist-person candidate: {candidate}")


def _model_factory(candidate: str, config: dict[str, Any], classes: int) -> nn.Module:
    gating_mode, _ = _candidate_contract(candidate)
    model = config["model"]
    return WristPersonResidualFusion(
        embedding_dim=int(model["embedding_dim"]),
        num_classes=classes,
        class_embedding_dim=int(model["class_embedding_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        gating_mode=gating_mode,
        maximum_logit_delta=float(model["maximum_logit_delta"]),
    )


def _forward(
    model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return model(
        view_embeddings=batch["view_embeddings"].to(device),
        anchor_logits=batch["anchor_logits"].to(device),
        anchor_view_weights=batch["anchor_view_weights"].to(device),
        availability=batch["availability"].to(device),
        num_frames=batch["num_frames"].to(device),
    )


def _weighted_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _optimizer_step(
    *, loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    loss.backward()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("non-finite wrist-person gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()


def _person_aux_weight(candidate: str, config: dict[str, Any]) -> float:
    _, use_auxiliary = _candidate_contract(candidate)
    return float(config["training"]["person_aux_weight"]) if use_auxiliary else 0.0


@torch.no_grad()
def _predict(
    *,
    model: nn.Module,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        CachedWristPersonDataset(cache, indices), batch_size=batch_size, shuffle=False
    )
    logits, person_logits, gates, deltas, ordered = [], [], [], [], []
    for batch in loader:
        output = _forward(model, batch, device)
        logits.append(output["logits"].float().cpu())
        person_logits.append(output["person_logits"].float().cpu())
        gates.append(output["person_gate"].float().cpu())
        deltas.append(output["delta_logits"].float().cpu())
        ordered.append(batch["index"].long())
    order = torch.cat(ordered).numpy()
    expected = np.asarray(indices, dtype=np.int64)
    if not np.array_equal(order, expected):
        raise RuntimeError("wrist-person prediction order changed")
    logits_np = torch.cat(logits).numpy()
    person_np = torch.cat(person_logits).numpy()
    gate_np = torch.cat(gates).numpy()
    delta_np = torch.cat(deltas).numpy()
    metrics = _metrics(cache["labels"][order], logits_np, cache["user_ids"][order])
    person_metrics = _metrics(
        cache["labels"][order], person_np, cache["user_ids"][order]
    )
    metrics.update(
        {
            "mean_person_gate": float(gate_np.mean()),
            "person_gate_quantiles": np.quantile(
                gate_np, [0.0, 0.25, 0.5, 0.75, 1.0]
            ).tolist(),
            "mean_absolute_delta": float(np.abs(delta_np).mean()),
            "maximum_absolute_delta": float(np.abs(delta_np).max()),
            "person_only_accuracy": float(person_metrics["accuracy"]),
            "person_only_macro_f1": float(person_metrics["macro_f1"]),
        }
    )
    return {
        "metrics": metrics,
        "logits": logits_np,
        "person_logits": person_np,
        "gates": gate_np,
        "deltas": delta_np,
        "indices": order,
    }


def _train_fixed(
    *,
    candidate: str,
    config: dict[str, Any],
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    _set_seed(seed)
    classes = int(cache["route_logits"].shape[2])
    model = _model_factory(candidate, config, classes).to(device)
    training = config["training"]
    dataset = CachedWristPersonDataset(cache, indices)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        sampler=_weighted_sampler(cache["labels"][indices], seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, int(training["fixed_epochs"]) + 1):
        model.train()
        totals = {"loss": [], "fused_ce": [], "person_ce": [], "guard_loss": []}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch, device)
            losses = wrist_person_loss(
                output=output,
                labels=batch["label"].to(device),
                person_available=batch["availability"][:, 1].to(device),
                person_aux_weight=_person_aux_weight(candidate, config),
                guard_weight=float(training["guard_weight"]),
            )
            _optimizer_step(loss=losses["loss"], model=model, optimizer=optimizer)
            for key in totals:
                totals[key].append(float(losses[key].detach()))
        history.append(
            {"epoch": float(epoch), **{key: float(np.mean(value)) for key, value in totals.items()}}
        )
    return model, history


def train_cv_fold(
    *,
    candidate: str,
    config: dict[str, Any],
    cache: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model, history = _train_fixed(
        candidate=candidate,
        config=config,
        cache=cache,
        indices=train_indices,
        seed=seed,
        device=device,
    )
    prediction = _predict(
        model=model,
        cache=cache,
        indices=validation_indices,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
    )
    return {
        "candidate": candidate,
        "epochs": int(config["training"]["fixed_epochs"]),
        "metrics": prediction["metrics"],
        "logits": prediction["logits"],
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "history": history,
        "validation_indices": np.asarray(validation_indices, dtype=np.int64),
        "validation_sample_ids": cache["sample_ids"][validation_indices]
        .astype(str)
        .tolist(),
    }


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    required = {
        "sample_ids",
        "user_ids",
        "labels",
        "num_frames",
        "availability",
        "fused_view_embeddings",
        "route_logits",
        "route_view_weights",
    }
    if not required.issubset(cache):
        raise RuntimeError(f"wrist-person cache misses {sorted(required - set(cache))}")
    return cache


def _comparison(
    *, labels: np.ndarray, anchor_logits: np.ndarray, candidate_logits: np.ndarray
) -> dict[str, int]:
    anchor = anchor_logits.argmax(axis=1)
    candidate = candidate_logits.argmax(axis=1)
    return {
        "rescued": int(((candidate == labels) & (anchor != labels)).sum()),
        "harmed": int(((candidate != labels) & (anchor == labels)).sum()),
        "net": int((candidate == labels).sum() - (anchor == labels).sum()),
        "disagreement": int((candidate != anchor).sum()),
    }


def _per_class_delta(
    *, labels: np.ndarray, anchor_logits: np.ndarray, candidate_logits: np.ndarray
) -> list[dict[str, float | int]]:
    anchor = anchor_logits.argmax(axis=1)
    candidate = candidate_logits.argmax(axis=1)
    rows = []
    for class_id in range(anchor_logits.shape[1]):
        mask = labels == class_id
        anchor_recall = float((anchor[mask] == class_id).mean())
        candidate_recall = float((candidate[mask] == class_id).mean())
        rows.append(
            {
                "class_id": class_id,
                "support": int(mask.sum()),
                "anchor_recall": anchor_recall,
                "candidate_recall": candidate_recall,
                "recall_delta": candidate_recall - anchor_recall,
            }
        )
    return rows


def _validate_membership(
    *, train: dict[str, np.ndarray], validation: dict[str, np.ndarray], config: dict[str, Any]
) -> None:
    for partition, cache in (("train", train), ("validation", validation)):
        validate_cache_membership(
            partition=partition,
            sample_ids=cache["sample_ids"],
            user_ids=cache["user_ids"],
            labels=cache["labels"],
            expected_samples=int(config["split"][f"{partition}_samples"]),
            config=config,
        )


def run(config_path: Path) -> dict[str, Any]:
    config = load_wrist_person_config(config_path)
    cache_report_path = _project_path(str(config["source"]["cache_report"]))
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    caches: dict[str, dict[str, np.ndarray]] = {}
    for partition in ("train", "validation"):
        path = _project_path(str(config["source"][f"{partition}_cache"]))
        artifact = cache_report["artifacts"][partition]
        if path.stat().st_size != int(artifact["bytes"]) or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"{partition} wrist-person cache provenance changed")
        caches[partition] = _load_cache(path)
    train, validation = caches["train"], caches["validation"]
    _validate_membership(train=train, validation=validation, config=config)

    device = torch.device("cpu")
    candidates = tuple(str(value) for value in config["model"]["candidates"])
    cv_results: dict[str, Any] = {}
    for candidate_index, candidate in enumerate(candidates):
        pooled = np.full_like(train["route_logits"][:, 0], np.nan, dtype=np.float32)
        folds = []
        for fold_index, fold_users in enumerate(config["cv"]["folds"]):
            held_out = np.isin(train["user_ids"].astype(str), np.asarray(fold_users))
            validation_indices = np.flatnonzero(held_out)
            train_indices = np.flatnonzero(~held_out)
            result = train_cv_fold(
                candidate=candidate,
                config=config,
                cache=train,
                train_indices=train_indices,
                validation_indices=validation_indices,
                seed=int(config["training"]["seed"]) + candidate_index * 100 + fold_index,
                device=device,
            )
            pooled[validation_indices] = result["logits"]
            folds.append(
                {
                    "users": [str(value) for value in fold_users],
                    "epochs": result["epochs"],
                    "metrics": result["metrics"],
                    "history": result["history"],
                }
            )
        if not np.isfinite(pooled).all():
            raise RuntimeError(f"{candidate} grouped predictions incomplete")
        metrics = _metrics(train["labels"], pooled, train["user_ids"])
        cv_results[candidate] = {
            "metrics": metrics,
            "comparison_to_anchor": _comparison(
                labels=train["labels"],
                anchor_logits=train["route_logits"][:, 0],
                candidate_logits=pooled,
            ),
            "folds": folds,
        }
        print(
            json.dumps(
                {
                    "candidate": candidate,
                    "oof_accuracy": metrics["accuracy"],
                    "oof_macro_f1": metrics["macro_f1"],
                }
            ),
            flush=True,
        )
    selected = max(
        candidates,
        key=lambda name: (
            cv_results[name]["metrics"]["accuracy"],
            cv_results[name]["metrics"]["macro_f1"],
            cv_results[name]["metrics"]["worst_user_accuracy"],
            -cv_results[name]["metrics"]["nll"],
        ),
    )

    output_root = _project_path(str(config["outputs"]["root"]))
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    train_indices = np.arange(len(train["labels"]), dtype=np.int64)
    validation_indices = np.arange(len(validation["labels"]), dtype=np.int64)
    final_results: dict[str, Any] = {}
    for candidate_index, candidate in enumerate(candidates):
        model, history = _train_fixed(
            candidate=candidate,
            config=config,
            cache=train,
            indices=train_indices,
            seed=int(config["training"]["seed"]) + 1000 + candidate_index,
            device=device,
        )
        train_prediction = _predict(
            model=model,
            cache=train,
            indices=train_indices,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
        )
        validation_prediction = _predict(
            model=model,
            cache=validation,
            indices=validation_indices,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
        )
        checkpoint = output_root / f"{candidate}.pt"
        torch.save(
            {
                "candidate": candidate,
                "epochs": int(config["training"]["fixed_epochs"]),
                "model_state_dict": model.state_dict(),
                "config": config,
                "grouped_metrics": cv_results[candidate]["metrics"],
            },
            checkpoint,
        )
        _atomic_npz(
            output_root / f"{candidate}_validation_predictions.npz",
            sample_ids=validation["sample_ids"],
            user_ids=validation["user_ids"],
            labels=validation["labels"],
            logits=validation_prediction["logits"],
            person_logits=validation_prediction["person_logits"],
            person_gates=validation_prediction["gates"],
            delta_logits=validation_prediction["deltas"],
        )
        final_results[candidate] = {
            "train_metrics": train_prediction["metrics"],
            "validation_metrics": validation_prediction["metrics"],
            "comparison_to_anchor": _comparison(
                labels=validation["labels"],
                anchor_logits=validation["route_logits"][:, 0],
                candidate_logits=validation_prediction["logits"],
            ),
            "per_class_delta": _per_class_delta(
                labels=validation["labels"],
                anchor_logits=validation["route_logits"][:, 0],
                candidate_logits=validation_prediction["logits"],
            ),
            "history": history,
            "checkpoint": str(Path(config["outputs"]["root"]) / f"{candidate}.pt"),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    selected_metrics = final_results[selected]["validation_metrics"]
    passes = (
        float(selected_metrics["accuracy"])
        > float(config["references"]["fixed_both_wrists_accuracy"])
        and float(selected_metrics["worst_user_accuracy"])
        >= float(config["references"]["minimum_worst_user_accuracy"])
    )
    report = {
        "stage": "P4-WP0",
        "status": "completed",
        "selected_by_train_user_grouped_cv": selected,
        "decision": "promote_to_full_finetune" if passes else "reject_cached_fusion",
        "cache_provenance": {
            "report": str(config["source"]["cache_report"]),
            "artifacts": cache_report["artifacts"],
            "selected_prediction_reproduction": cache_report[
                "selected_prediction_reproduction"
            ],
        },
        "cv_results": cv_results,
        "final_results": final_results,
        "validation_users_entered_training": False,
        "validation_users_entered_cv_selection": False,
        "videomae_updated": False,
        "limitations": [
            "The frozen base checkpoint saw all train12 samples; grouped folds isolate only the new fusion module.",
            "The cached experiment cannot determine whether end-to-end auxiliary person training would improve the person embedding itself.",
            "user6/user7 remain a repeatedly inspected development boundary, not untouched final evidence.",
        ],
    }
    report_json = _project_path(str(config["outputs"]["report_json"]))
    report_markdown = _project_path(str(config["outputs"]["report_markdown"]))
    _atomic_write_text(report_json, json.dumps(report, indent=2) + "\n")
    lines = [
        "# VideoMAE wrist-person residual result",
        "",
        f"- Selected by train-user grouped CV: `{selected}`",
        f"- Decision: `{report['decision']}`",
        "- Validation users entered training/CV selection: `False`",
        "",
        "| Candidate | Grouped Acc | Val Acc | Macro-F1 | Worst-user | Person-only | Gate | Rescue/Harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        metrics = final_results[candidate]["validation_metrics"]
        comparison = final_results[candidate]["comparison_to_anchor"]
        lines.append(
            f"| {candidate} | {cv_results[candidate]['metrics']['accuracy']:.6f} | "
            f"{metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['worst_user_accuracy']:.6f} | {metrics['person_only_accuracy']:.6f} | "
            f"{metrics['mean_person_gate']:.6f} | {comparison['rescued']}/{comparison['harmed']} |"
        )
    _atomic_write_text(report_markdown, "\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached wrist-person residual fusion")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    selected = report["selected_by_train_user_grouped_cv"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": selected,
                "validation_accuracy": report["final_results"][selected][
                    "validation_metrics"
                ]["accuracy"],
                "decision": report["decision"],
            }
        )
    )


if __name__ == "__main__":
    main()

