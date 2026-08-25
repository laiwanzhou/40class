from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_depth_videomaev2_p3r1.yaml"

from scripts.cache_ir_depth_videomaev2_p2a import _atomic_npz, _atomic_write_text, _metrics
from scripts.cache_ir_depth_videomaev2_p3r1 import (
    _project_path,
    load_p3r1_config,
    validate_cache_membership,
)
from src.models.ir_anchored_top5_reranker import Top5LogitReranker
from src.models.ir_depth_videomaev2_teacher import sha256_file
from src.models.margin_conditioned_top3_routing import MarginConditionedTop3Reranker


class CachedRoutes(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        return {
            "route_logits": torch.from_numpy(self.cache["route_logits"][row]).float(),
            "num_frames": torch.tensor(float(self.cache["num_frames"][row])),
            "label": torch.tensor(int(self.cache["labels"][row]), dtype=torch.long),
            "index": torch.tensor(row, dtype=torch.long),
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_factory(
    candidate: str, *, config: dict[str, Any], classes: int, routes: int
) -> nn.Module:
    settings = config["reranker"]
    if candidate == "logit_only":
        return Top5LogitReranker(
            num_classes=classes,
            hidden_dim=int(settings["hidden_dim"]),
            class_top_k=int(settings["class_top_k"]),
            dropout=0.2,
        )
    if candidate in {"routes_no_margin", "margin_routes"}:
        return MarginConditionedTop3Reranker(
            num_classes=classes,
            route_count=routes,
            class_embedding_dim=int(settings["class_embedding_dim"]),
            hidden_dim=int(settings["hidden_dim"]),
            use_margin_gate=candidate == "margin_routes",
        )
    raise ValueError(f"unknown P3-R1 candidate: {candidate}")


def _forward(
    model: nn.Module, candidate: str, batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    routes = batch["route_logits"].to(device)
    anchor = routes[:, 0]
    if candidate == "logit_only":
        output = model(base_logits=anchor)
        output["margin_gate"] = torch.ones(anchor.shape[0], device=device)
        return output
    return model(
        anchor_logits=anchor,
        route_logits=routes,
        num_frames=batch["num_frames"].to(device),
    )


def _guarded_loss(
    *, logits: torch.Tensor, anchor_logits: torch.Tensor, labels: torch.Tensor, weight: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = nn.functional.cross_entropy(logits, labels, reduction="none")
    anchor_ce = nn.functional.cross_entropy(anchor_logits.detach(), labels, reduction="none")
    guard = torch.relu(ce - anchor_ce)
    return ce.mean() + float(weight) * guard.mean(), ce.mean(), guard.mean()


def _optimizer_step(
    *, loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    loss.backward()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("non-finite P3-R1 gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()


def _weighted_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    weights = 1.0 / counts[labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


@torch.no_grad()
def _predict(
    model: nn.Module,
    candidate: str,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(CachedRoutes(cache, indices), batch_size=batch_size, shuffle=False)
    logits, ordered_indices, gates = [], [], []
    for batch in loader:
        output = _forward(model, candidate, batch, device)
        logits.append(output["logits"].float().cpu())
        ordered_indices.append(batch["index"].long())
        gates.append(output["margin_gate"].float().cpu())
    logits_np = torch.cat(logits).numpy()
    order = torch.cat(ordered_indices).numpy()
    expected = np.asarray(indices, dtype=np.int64)
    if not np.array_equal(order, expected):
        raise RuntimeError("P3-R1 prediction order changed")
    metrics = _metrics(cache["labels"][order], logits_np, cache["user_ids"][order])
    metrics["mean_margin_gate"] = float(torch.cat(gates).mean())
    return metrics, logits_np, order


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
    _set_seed(seed)
    classes = int(cache["route_logits"].shape[2])
    routes = int(cache["route_logits"].shape[1])
    model = _model_factory(candidate, config=config, classes=classes, routes=routes).to(device)
    settings = config["reranker"]
    dataset = CachedRoutes(cache, train_indices)
    sampler = _weighted_sampler(cache["labels"][train_indices], seed)
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), sampler=sampler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    history: list[dict[str, object]] = []
    fixed_epochs = int(settings["fixed_epochs"])
    for epoch in range(1, fixed_epochs + 1):
        model.train()
        losses, guards = [], []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, candidate, batch, device)
            labels = batch["label"].to(device)
            anchor = batch["route_logits"][:, 0].to(device)
            loss, _, guard = _guarded_loss(
                logits=output["logits"],
                anchor_logits=anchor,
                labels=labels,
                weight=float(settings["guard_weight"]),
            )
            _optimizer_step(loss=loss, model=model, optimizer=optimizer)
            losses.append(float(loss.detach()))
            guards.append(float(guard.detach()))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "guard_loss": float(np.mean(guards)),
            }
        )
    metrics, logits, _ = _predict(
        model,
        candidate,
        cache,
        validation_indices,
        batch_size=int(settings["batch_size"]),
        device=device,
    )
    return {
        "candidate": candidate,
        "best_epoch": fixed_epochs,
        "epochs_completed": len(history),
        "best_metrics": metrics,
        "best_logits": logits,
        "best_state": copy.deepcopy(model.state_dict()),
        "history": history,
        "validation_indices": np.asarray(validation_indices, dtype=np.int64),
        "validation_sample_ids": cache["sample_ids"][validation_indices].astype(str).tolist(),
    }


def _fit_fixed_epochs(
    *,
    candidate: str,
    config: dict[str, Any],
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    _set_seed(seed)
    settings = config["reranker"]
    model = _model_factory(
        candidate,
        config=config,
        classes=int(cache["route_logits"].shape[2]),
        routes=int(cache["route_logits"].shape[1]),
    ).to(device)
    dataset = CachedRoutes(cache, indices)
    sampler = _weighted_sampler(cache["labels"][indices], seed)
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), sampler=sampler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, candidate, batch, device)
            labels = batch["label"].to(device)
            anchor = batch["route_logits"][:, 0].to(device)
            loss, _, _ = _guarded_loss(
                logits=output["logits"],
                anchor_logits=anchor,
                labels=labels,
                weight=float(settings["guard_weight"]),
            )
            _optimizer_step(loss=loss, model=model, optimizer=optimizer)
    return model


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    required = {
        "sample_ids", "user_ids", "labels", "num_frames", "route_names",
        "route_logits", "route_view_weights",
    }
    if not required.issubset(cache):
        raise RuntimeError(f"P3-R1 cache misses {sorted(required - set(cache))}")
    return cache


def _route_controls(cache: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    controls: dict[str, dict[str, object]] = {}
    anchor = cache["route_logits"][:, 0]
    for index, name in enumerate(cache["route_names"].astype(str)):
        logits = cache["route_logits"][:, index]
        controls[str(name)] = {
            **_metrics(cache["labels"], logits, cache["user_ids"]),
            "comparison_to_anchor": _comparison(
                labels=cache["labels"], anchor_logits=anchor, candidate_logits=logits
            ),
            "mean_view_weights": cache["route_view_weights"][:, index]
            .astype(np.float64)
            .mean(axis=(0, 1))
            .tolist(),
        }
    return controls


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


def compare_uniform_reference(
    *,
    current: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    action_names: tuple[str, ...],
) -> dict[str, Any]:
    current_ids = current["sample_ids"].astype(str)
    reference_ids = reference["sample_ids"].astype(str)
    if np.unique(current_ids).size != len(current_ids) or np.unique(reference_ids).size != len(reference_ids):
        raise ValueError("uniform comparison sample IDs must be unique")
    if set(current_ids.tolist()) != set(reference_ids.tolist()):
        raise ValueError("uniform comparison sample membership differs")
    lookup = {sample_id: index for index, sample_id in enumerate(reference_ids)}
    order = np.asarray([lookup[sample_id] for sample_id in current_ids], dtype=np.int64)
    labels = current["labels"].astype(np.int64)
    if not np.array_equal(labels, reference["labels"][order].astype(np.int64)):
        raise ValueError("uniform comparison labels differ")
    current_logits = current["route_logits"][:, 0].astype(np.float32)
    uniform_logits = reference["full_logits"][order].astype(np.float32)
    classes = current_logits.shape[1]
    if len(action_names) != classes:
        raise ValueError("uniform comparison action names differ")
    current_predictions = current_logits.argmax(axis=1)
    uniform_predictions = uniform_logits.argmax(axis=1)
    per_class = []
    for class_id in range(classes):
        mask = labels == class_id
        current_correct = current_predictions[mask] == class_id
        uniform_correct = uniform_predictions[mask] == class_id
        per_class.append(
            {
                "class_id": class_id,
                "action_name": action_names[class_id],
                "support": int(mask.sum()),
                "uniform_recall": float(uniform_correct.mean()),
                "current_recall": float(current_correct.mean()),
                "recall_delta": float(current_correct.mean() - uniform_correct.mean()),
                "rescued": int((current_correct & ~uniform_correct).sum()),
                "harmed": int((~current_correct & uniform_correct).sum()),
            }
        )
    weights = reference["class_view_weights"][order].astype(np.float64)
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=(2, 3))
    return {
        "uniform_accuracy": float((uniform_predictions == labels).mean()),
        "current_accuracy": float((current_predictions == labels).mean()),
        "uniform_correct": int((uniform_predictions == labels).sum()),
        "current_correct": int((current_predictions == labels).sum()),
        "current_vs_uniform": _comparison(
            labels=labels,
            anchor_logits=uniform_logits,
            candidate_logits=current_logits,
        ),
        "uniform_mean_view_weights": weights.mean(axis=(0, 1)).tolist(),
        "uniform_mean_view_entropy": float(entropy.mean()),
        "per_class": per_class,
    }


def _action_names() -> tuple[str, ...]:
    names: dict[int, str] = {}
    with (PROJECT_ROOT / "metadata/manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            names[int(row["class_id"])] = str(row["action_name"])
    return tuple(names[index] for index in range(len(names)))


def _jsonable_fold(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"best_state", "best_logits", "validation_indices"}
    }


def validate_loaded_cache_contracts(
    *,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    config: dict[str, Any],
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
    config = load_p3r1_config(config_path)
    cache_report_path = _project_path(str(config["cache"]["report"]))
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    caches: dict[str, dict[str, np.ndarray]] = {}
    for partition in ("train", "validation"):
        artifact = cache_report["artifacts"][partition]
        path = _project_path(str(config["cache"][partition]))
        if path.stat().st_size != int(artifact["bytes"]) or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"{partition} P3-R1 cache provenance changed")
        caches[partition] = _load_cache(path)
    train, validation = caches["train"], caches["validation"]
    validate_loaded_cache_contracts(train=train, validation=validation, config=config)
    if not np.array_equal(train["route_names"], validation["route_names"]):
        raise RuntimeError("P3-R1 cache route order differs")
    if set(validation["user_ids"].astype(str).tolist()) != {"user6", "user7"}:
        raise RuntimeError("P3-R1 final validation membership changed")
    uniform_path = _project_path(str(config["source"]["uniform_validation_cache"]))
    if (
        uniform_path.stat().st_size != int(config["source"]["uniform_validation_cache_bytes"])
        or sha256_file(uniform_path) != config["source"]["uniform_validation_cache_sha256"]
    ):
        raise RuntimeError("P3-R1 uniform reference provenance changed")
    with np.load(uniform_path, allow_pickle=False) as archive:
        uniform_reference = {key: archive[key] for key in archive.files}
    uniform_comparison = compare_uniform_reference(
        current=validation,
        reference=uniform_reference,
        action_names=_action_names(),
    )

    device = torch.device("cpu")
    candidates = ("logit_only", "routes_no_margin", "margin_routes")
    cv_results: dict[str, Any] = {}
    oof_logits: dict[str, np.ndarray] = {}
    for candidate_index, candidate in enumerate(candidates):
        pooled = np.full_like(train["route_logits"][:, 0], np.nan, dtype=np.float32)
        folds = []
        for fold_index, fold_users in enumerate(config["cv"]["folds"]):
            validation_mask = np.isin(train["user_ids"].astype(str), np.asarray(fold_users))
            validation_indices = np.flatnonzero(validation_mask)
            train_indices = np.flatnonzero(~validation_mask)
            result = train_cv_fold(
                candidate=candidate,
                config=config,
                cache=train,
                train_indices=train_indices,
                validation_indices=validation_indices,
                seed=int(config["reranker"]["seed"]) + candidate_index * 100 + fold_index,
                device=device,
            )
            pooled[validation_indices] = result["best_logits"]
            folds.append(result)
        if not np.isfinite(pooled).all():
            raise RuntimeError(f"{candidate} OOF predictions incomplete")
        metrics = _metrics(train["labels"], pooled, train["user_ids"])
        oof_logits[candidate] = pooled
        cv_results[candidate] = {
            "oof_metrics": metrics,
            "comparison_to_anchor": _comparison(
                labels=train["labels"],
                anchor_logits=train["route_logits"][:, 0],
                candidate_logits=pooled,
            ),
            "folds": [_jsonable_fold(result) for result in folds],
            "final_epoch": int(config["reranker"]["fixed_epochs"]),
        }
        print(
            json.dumps(
                {
                    "candidate": candidate,
                    "oof_accuracy": metrics["accuracy"],
                    "oof_macro_f1": metrics["macro_f1"],
                    "final_epoch": cv_results[candidate]["final_epoch"],
                }
            ),
            flush=True,
        )

    selected = max(
        candidates,
        key=lambda name: (
            cv_results[name]["oof_metrics"]["accuracy"],
            cv_results[name]["oof_metrics"]["macro_f1"],
            cv_results[name]["oof_metrics"]["worst_user_accuracy"],
            -cv_results[name]["oof_metrics"]["nll"],
        ),
    )
    output_root = _project_path(str(config["outputs"]["root"]))
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    final_results: dict[str, Any] = {}
    all_train_indices = np.arange(len(train["labels"]), dtype=np.int64)
    all_validation_indices = np.arange(len(validation["labels"]), dtype=np.int64)
    for candidate_index, candidate in enumerate(candidates):
        model = _fit_fixed_epochs(
            candidate=candidate,
            config=config,
            cache=train,
            indices=all_train_indices,
            epochs=int(cv_results[candidate]["final_epoch"]),
            seed=int(config["reranker"]["seed"]) + 1000 + candidate_index,
            device=device,
        )
        train_metrics, train_logits, _ = _predict(
            model,
            candidate,
            train,
            all_train_indices,
            batch_size=int(config["reranker"]["batch_size"]),
            device=device,
        )
        validation_metrics, validation_logits, _ = _predict(
            model,
            candidate,
            validation,
            all_validation_indices,
            batch_size=int(config["reranker"]["batch_size"]),
            device=device,
        )
        torch.save(
            {
                "candidate": candidate,
                "epoch": int(cv_results[candidate]["final_epoch"]),
                "model_state_dict": model.state_dict(),
                "config": config,
                "oof_metrics": cv_results[candidate]["oof_metrics"],
            },
            output_root / f"{candidate}.pt",
        )
        checkpoint_path = output_root / f"{candidate}.pt"
        _atomic_npz(
            output_root / f"{candidate}_validation_predictions.npz",
            sample_ids=validation["sample_ids"],
            user_ids=validation["user_ids"],
            labels=validation["labels"],
            logits=validation_logits,
            predictions=validation_logits.argmax(axis=1),
        )
        final_results[candidate] = {
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "comparison_to_anchor": _comparison(
                labels=validation["labels"],
                anchor_logits=validation["route_logits"][:, 0],
                candidate_logits=validation_logits,
            ),
            "checkpoint": str(Path(config["outputs"]["root"]) / f"{candidate}.pt"),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }

    route_controls = {
        "train": _route_controls(train),
        "validation": _route_controls(validation),
    }
    selected_accuracy = float(final_results[selected]["validation_metrics"]["accuracy"])
    report = {
        "stage": "P3-R1",
        "status": "completed",
        "selected_by_train_user_grouped_cv": selected,
        "route_names": train["route_names"].astype(str).tolist(),
        "route_controls": route_controls,
        "cache_provenance": {
            "report": str(config["cache"]["report"]),
            "artifacts": cache_report["artifacts"],
            "selected_prediction_reproduction": cache_report.get(
                "selected_prediction_reproduction"
            ),
        },
        "wrist_fallback_samples": {
            "train": int((~train["availability"].any(axis=1)[:, 2:].any(axis=1)).sum()),
            "validation": int(
                (~validation["availability"].any(axis=1)[:, 2:].any(axis=1)).sum()
            ),
        },
        "uniform_reference_comparison": uniform_comparison,
        "cv_results": cv_results,
        "final_results": final_results,
        "decision": (
            "passes_0.75" if selected_accuracy >= float(config["references"]["minimum_accuracy"])
            else "beats_fixed_wrists" if selected_accuracy >= float(config["references"]["fixed_both_wrists_accuracy"])
            else "reject"
        ),
        "validation_users_entered_training": False,
        "validation_users_entered_cv_selection": False,
        "base_checkpoint_updated": False,
        "limitations": [
            "The frozen base checkpoint saw all train12 samples before reranker grouped CV; OOF applies to the reranker, not the base representation.",
            "user6/user7 are a repeatedly inspected development boundary and not untouched final evidence.",
        ],
    }
    report_json = _project_path(str(config["outputs"]["report_json"]))
    report_markdown = _project_path(str(config["outputs"]["report_markdown"]))
    _atomic_write_text(report_json, json.dumps(report, indent=2) + "\n")
    lines = [
        "# IR + Depth VideoMAE P3-R1 result",
        "",
        f"- Selected by train-user grouped CV: `{selected}`",
        f"- Validation Accuracy: `{selected_accuracy:.6f}`",
        f"- Decision: `{report['decision']}`",
        "- Validation users entered training/CV selection: `False`",
        "",
        "## Route controls",
        "",
        "| Route | Accuracy | Macro-F1 | Worst-user | G/P/L/R weights | Rescue/Harm |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name, metrics in route_controls["validation"].items():
        weights = "/".join(f"{value:.3f}" for value in metrics["mean_view_weights"])
        comparison = metrics["comparison_to_anchor"]
        lines.append(
            f"| {name} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['worst_user_accuracy']:.6f} | {weights} | "
            f"{comparison['rescued']}/{comparison['harmed']} |"
        )
    lines.extend(
        [
            "",
            "## Old uniform fusion versus current hard Top-2",
            "",
            f"- Uniform Accuracy: `{uniform_comparison['uniform_accuracy']:.6f}`",
            f"- Current Accuracy: `{uniform_comparison['current_accuracy']:.6f}`",
            f"- Rescue/Harm: `{uniform_comparison['current_vs_uniform']['rescued']}/"
            f"{uniform_comparison['current_vs_uniform']['harmed']}`",
            "",
            "| Class | Support | Uniform recall | Current recall | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in uniform_comparison["per_class"]:
        lines.append(
            f"| {row['action_name']} | {row['support']} | {row['uniform_recall']:.6f} | "
            f"{row['current_recall']:.6f} | {row['recall_delta']:+.6f} |"
        )
    lines.extend(["", "## Final rerankers", "", "| Candidate | OOF Acc | Val Acc | Rescue/Harm |", "|---|---:|---:|---:|"])
    for name in candidates:
        comparison = final_results[name]["comparison_to_anchor"]
        lines.append(
            f"| {name} | {cv_results[name]['oof_metrics']['accuracy']:.6f} | "
            f"{final_results[name]['validation_metrics']['accuracy']:.6f} | "
            f"{comparison['rescued']}/{comparison['harmed']} |"
        )
    _atomic_write_text(report_markdown, "\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached VideoMAE P3-R1 reranking")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    selected = report["selected_by_train_user_grouped_cv"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": selected,
                "validation_accuracy": report["final_results"][selected]["validation_metrics"]["accuracy"],
                "decision": report["decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
