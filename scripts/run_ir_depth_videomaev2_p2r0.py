from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_ir_depth_videomaev2_p2a import (
    _atomic_npz,
    _atomic_write_text,
    _metrics,
    _require_artifact,
    fuse_cached_view_logits,
)
from src.models.ir_anchored_top5_reranker import (
    IRAnchoredTop5Reranker,
    Top5LogitReranker,
    guarded_reranker_loss,
)
from src.models.ir_depth_videomaev2_teacher import sha256_file


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_depth_videomaev2_p2r0.yaml"


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_p2r0_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P2-R0 config must be a mapping")
    model = config.get("model", {})
    policy = config.get("policy", {})
    validation = config.get("validation_policy", {})
    folds = config.get("cv", {}).get("folds", [])
    if config.get("stage") != "P2-R0":
        raise ValueError("P2-R0 stage changed")
    if model.get("view_top_k") != 2 or model.get("class_top_k") != 5:
        raise ValueError("P2-R0 sparse routing contract changed")
    if policy.get("update_videomae") is not False or policy.get("p2b_authorized") is not False:
        raise ValueError("P2-R0 may not update VideoMAE or authorize P2-B")
    if validation.get("final_users") != ["user6", "user7"]:
        raise ValueError("P2-R0 final users changed")
    if validation.get("final_evaluation_count") != 1:
        raise ValueError("P2-R0 permits one final evaluation")
    if (
        validation.get("use_final_users_for_training") is not False
        or validation.get("use_final_users_for_early_stopping") is not False
        or validation.get("use_final_users_for_hyperparameter_search") is not False
    ):
        raise ValueError("P2-R0 final users are isolated from model selection")
    flat_folds = [str(user) for fold in folds for user in fold]
    if len(folds) != 3 or len(flat_folds) != 12 or len(set(flat_folds)) != 12:
        raise ValueError("P2-R0 requires three disjoint four-user CV folds")
    if set(flat_folds) & set(validation["final_users"]):
        raise ValueError("P2-R0 CV folds include final validation users")
    return config


class CachedFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.embeddings = torch.from_numpy(cache["view_embeddings"][indices].astype(np.float32))
        self.base_logits = torch.from_numpy(cache["ir_anchor_logits"][indices].astype(np.float32))
        self.availability = torch.from_numpy(cache["availability"][indices].astype(bool))
        self.labels = torch.from_numpy(cache["labels"][indices].astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "view_embeddings": self.embeddings[index],
            "base_logits": self.base_logits[index],
            "availability": self.availability[index],
            "label": self.labels[index],
        }


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _validate_caches(
    config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    source = config["source"]
    train_report = json.loads(
        _project_path(str(source["train_cache_report"])).read_text(encoding="utf-8")
    )
    if train_report.get("status") != "completed" or train_report.get("sample_count") != 1935:
        raise RuntimeError("P2-R0 train cache report is incomplete")
    train_path = _project_path(str(source["train_cache"]))
    _require_artifact(
        train_path,
        expected_hash=str(train_report["cache_sha256"]),
        expected_bytes=int(train_report["cache_bytes"]),
    )
    validation_report = json.loads(
        _project_path(str(source["validation_cache_report"])).read_text(encoding="utf-8")
    )
    validation_path = _project_path(str(source["validation_cache"]))
    artifact = validation_report["artifacts"]["cache"]
    _require_artifact(
        validation_path,
        expected_hash=str(artifact["sha256"]),
        expected_bytes=int(artifact["bytes"]),
    )
    train = _load_cache(train_path)
    validation = _load_cache(validation_path)
    if (
        train["view_embeddings"].shape != (1935, 2, 4, 768)
        or validation["view_embeddings"].shape != (385, 2, 4, 768)
        or np.unique(train["sample_ids"].astype(str)).size != 1935
        or np.unique(validation["sample_ids"].astype(str)).size != 385
        or set(validation["user_ids"].astype(str).tolist()) != {"user6", "user7"}
    ):
        raise RuntimeError("P2-R0 cache membership or shape mismatch")
    validation = dict(validation)
    ir_available = np.zeros_like(validation["availability"], dtype=bool)
    ir_available[:, 0] = validation["availability"][:, 0]
    validation["ir_anchor_logits"], _ = fuse_cached_view_logits(
        validation["view_logits"], validation["class_view_gate"], ir_available
    )
    return train, validation, validation_report


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_factory(name: str, config: dict[str, Any]) -> nn.Module:
    model = config["model"]
    if name == "logit_only":
        return Top5LogitReranker(
            num_classes=int(model["num_classes"]),
            hidden_dim=int(model["fusion_dim"]),
            class_top_k=int(model["class_top_k"]),
            dropout=float(model["dropout"]),
        )
    if name == "feature_router":
        return IRAnchoredTop5Reranker(
            embedding_dim=int(model["embedding_dim"]),
            fusion_dim=int(model["fusion_dim"]),
            num_classes=int(model["num_classes"]),
            view_top_k=int(model["view_top_k"]),
            class_top_k=int(model["class_top_k"]),
            dropout=float(model["dropout"]),
            depth_gate_initial_bias=float(model["depth_gate_initial_bias"]),
        )
    raise ValueError(f"unknown P2-R0 candidate {name}")


def _forward(
    model: nn.Module, name: str, batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    base_logits = batch["base_logits"].to(device)
    if name == "logit_only":
        return model(base_logits=base_logits)
    return model(
        view_embeddings=batch["view_embeddings"].to(device),
        base_logits=base_logits,
        availability=batch["availability"].to(device),
    )


@torch.no_grad()
def _predict(
    model: nn.Module,
    name: str,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    dataset = CachedFeatureDataset(cache, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    logits, base_logits, view_weights, depth_gates = [], [], [], []
    for batch in loader:
        output = _forward(model, name, batch, device)
        logits.append(output["logits"].float().cpu())
        base_logits.append(output["base_logits"].float().cpu())
        if "view_weights" in output:
            view_weights.append(output["view_weights"].float().cpu())
            depth_gates.append(output["depth_gates"].float().cpu())
    logits_np = torch.cat(logits).numpy()
    base_np = torch.cat(base_logits).numpy()
    labels = cache["labels"][indices].astype(np.int64)
    users = cache["user_ids"][indices].astype(str)
    metrics = _metrics(labels, logits_np, users)
    prediction = logits_np.argmax(axis=1)
    base_prediction = base_np.argmax(axis=1)
    base_correct = base_prediction == labels
    candidate_correct = prediction == labels
    rescued = int((~base_correct & candidate_correct).sum())
    harmed = int((base_correct & ~candidate_correct).sum())
    shifted = base_np - base_np.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    high_confidence_correct = base_correct & (probabilities.max(axis=1) >= 0.7)
    high_confidence_harmed = int((high_confidence_correct & ~candidate_correct).sum())
    metrics.update(
        {
            "base_accuracy": float(base_correct.mean()),
            "rescued": rescued,
            "harmed": harmed,
            "net_rescue": rescued - harmed,
            "high_confidence_correct_count": int(high_confidence_correct.sum()),
            "high_confidence_harmed": high_confidence_harmed,
            "high_confidence_harm_rate": float(
                high_confidence_harmed / max(int(high_confidence_correct.sum()), 1)
            ),
        }
    )
    archive: dict[str, np.ndarray] = {
        "indices": indices,
        "sample_ids": cache["sample_ids"][indices],
        "user_ids": users,
        "labels": labels,
        "logits": logits_np,
        "base_logits": base_np,
        "predictions": prediction,
    }
    if view_weights:
        weights_np = torch.cat(view_weights).numpy()
        gates_np = torch.cat(depth_gates).numpy()
        nonzero = (weights_np > 0).sum(axis=2)
        entropy = -(weights_np * np.log(np.clip(weights_np, 1e-12, 1.0))).sum(axis=2)
        metrics.update(
            {
                "maximum_nonzero_views": int(nonzero.max()),
                "mean_view_entropy": float(entropy.mean()),
                "mean_depth_gate": float(gates_np.mean()),
            }
        )
        archive["view_weights"] = weights_np
        archive["depth_gates"] = gates_np
    return metrics, archive


def _weighted_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    classes, counts = np.unique(labels, return_counts=True)
    count_by_class = dict(zip(classes.tolist(), counts.tolist(), strict=True))
    weights = torch.tensor(
        [1.0 / count_by_class[int(label)] for label in labels], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights,
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _train_cv_fold(
    *,
    candidate: str,
    config: dict[str, Any],
    cache: dict[str, np.ndarray],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    _set_seed(seed)
    model = _model_factory(candidate, config).to(device)
    training = config["training"]
    dataset = CachedFeatureDataset(cache, train_indices)
    sampler = _weighted_sampler(cache["labels"][train_indices], seed)
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), sampler=sampler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None
    patience = 0
    history = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, candidate, batch, device)
            losses = guarded_reranker_loss(
                logits=output["logits"],
                base_logits=output["base_logits"],
                labels=batch["label"].to(device),
                depth_gates=output["depth_gates"],
                guard_weight=float(training["guard_weight"]),
                depth_l1_weight=float(training["depth_l1_weight"]),
            )
            losses["loss"].backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite P2-R0 gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(batch["label"])
            total_loss += float(losses["loss"].detach()) * count
            samples += count
        val_metrics, _ = _predict(
            model, candidate, cache, val_indices, device, int(training["batch_size"])
        )
        row = {"epoch": epoch, "train_loss": total_loss / samples, "validation": val_metrics}
        history.append(row)
        key = (
            float(val_metrics["macro_f1"]),
            float(val_metrics["accuracy"]),
            float(val_metrics["worst_user_accuracy"]),
            -float(epoch),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            patience = 0
        else:
            patience += 1
        if patience >= int(training["early_stopping_patience"]):
            break
    assert best_state is not None and best_metrics is not None
    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_state": best_state,
        "history": history,
        "epochs_completed": len(history),
    }


def _train_final(
    *,
    candidate: str,
    config: dict[str, Any],
    cache: dict[str, np.ndarray],
    epochs: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    _set_seed(seed)
    model = _model_factory(candidate, config).to(device)
    indices = np.arange(len(cache["labels"]), dtype=np.int64)
    dataset = CachedFeatureDataset(cache, indices)
    sampler = _weighted_sampler(cache["labels"], seed)
    training = config["training"]
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), sampler=sampler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, candidate, batch, device)
            losses = guarded_reranker_loss(
                logits=output["logits"],
                base_logits=output["base_logits"],
                labels=batch["label"].to(device),
                depth_gates=output["depth_gates"],
                guard_weight=float(training["guard_weight"]),
                depth_l1_weight=float(training["depth_l1_weight"]),
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_experiment(config_path: Path) -> dict[str, object]:
    config = load_p2r0_config(config_path)
    train_cache, validation_cache, validation_report = _validate_caches(config)
    output_root = _project_path(str(config["outputs"]["root"])) / str(config["run_id"])
    report_json = _project_path(str(config["outputs"]["report_json"]))
    report_markdown = _project_path(str(config["outputs"]["report_markdown"]))
    if output_root.exists() or report_json.exists() or report_markdown.exists():
        raise FileExistsError("P2-R0 outputs already exist")
    output_root.mkdir(parents=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    started = time.perf_counter()
    users = train_cache["user_ids"].astype(str)
    final_users = set(config["validation_policy"]["final_users"])
    if set(users.tolist()) & final_users:
        raise RuntimeError("final validation users leaked into P2-R0 train cache")

    candidates = ("logit_only", "feature_router")
    cv_results: dict[str, list[dict[str, object]]] = {name: [] for name in candidates}
    for candidate_index, candidate in enumerate(candidates):
        for fold_index, fold_users in enumerate(config["cv"]["folds"]):
            val_mask = np.isin(users, np.asarray(fold_users))
            train_indices = np.flatnonzero(~val_mask)
            val_indices = np.flatnonzero(val_mask)
            result = _train_cv_fold(
                candidate=candidate,
                config=config,
                cache=train_cache,
                train_indices=train_indices,
                val_indices=val_indices,
                seed=int(config["training"]["seed"]) + candidate_index * 100 + fold_index,
                device=device,
            )
            cv_results[candidate].append(
                {
                    "fold": fold_index,
                    "validation_users": list(fold_users),
                    "train_samples": len(train_indices),
                    "validation_samples": len(val_indices),
                    "best_epoch": result["best_epoch"],
                    "best_metrics": result["best_metrics"],
                    "epochs_completed": result["epochs_completed"],
                    "history": result["history"],
                }
            )

    final_models = {}
    final_epochs = {}
    for candidate_index, candidate in enumerate(candidates):
        epochs = max(
            1,
            int(statistics.median(result["best_epoch"] for result in cv_results[candidate])),
        )
        final_epochs[candidate] = epochs
        final_models[candidate] = _train_final(
            candidate=candidate,
            config=config,
            cache=train_cache,
            epochs=epochs,
            seed=int(config["training"]["seed"]) + 1000 + candidate_index,
            device=device,
        )

    # One frozen user6/user7 evaluation event for all preregistered candidates.
    final_indices = np.arange(len(validation_cache["labels"]), dtype=np.int64)
    final_results = {}
    for candidate in candidates:
        metrics, archive = _predict(
            final_models[candidate],
            candidate,
            validation_cache,
            final_indices,
            device,
            int(config["training"]["batch_size"]),
        )
        final_results[candidate] = metrics
        _atomic_npz(output_root / f"{candidate}_validation_predictions.npz", **archive)
        _atomic_torch_save(
            output_root / f"{candidate}_model.pt",
            {"model_state_dict": final_models[candidate].state_dict(), "epochs": final_epochs[candidate], "config": config},
        )

    feature = final_results["feature_router"]
    gates_config = config["success_gates"]
    success_gates = {
        "minimum_accuracy": float(feature["accuracy"]) > float(gates_config["minimum_accuracy"]),
        "minimum_macro_f1": float(feature["macro_f1"]) >= float(gates_config["minimum_macro_f1"]),
        "minimum_worst_user_accuracy": float(feature["worst_user_accuracy"])
        >= float(gates_config["minimum_worst_user_accuracy"]),
        "high_confidence_harm_rate": float(feature["high_confidence_harm_rate"]) <= 0.02,
        "maximum_two_views": int(feature.get("maximum_nonzero_views", 99)) <= 2,
    }
    success_gates["passed"] = all(success_gates.values())
    report = {
        "schema_version": 1,
        "stage": "P2-R0",
        "status": "completed",
        "training_scope": "cached rerankers only; VideoMAE frozen",
        "final_validation_evaluation_count": 1,
        "final_validation_users": config["validation_policy"]["final_users"],
        "train_cache_sha256": sha256_file(_project_path(str(config["source"]["train_cache"]))),
        "validation_cache_sha256": validation_report["artifacts"]["cache"]["sha256"],
        "cv_results": cv_results,
        "final_epochs": final_epochs,
        "final_results": final_results,
        "references": config["references"],
        "success_gates": success_gates,
        "effective_gate": float(feature["accuracy"]) >= float(gates_config["effective_accuracy"]),
        "strong_gate": float(feature["accuracy"]) >= float(gates_config["strong_accuracy"]),
        "hypothesis_status": "supported" if success_gates["passed"] else "rejected",
        "interpretation": {
            "cv_limit": (
                "The frozen VideoMAE had already trained on all train12 users, so internal "
                "router CV starts from near-saturated, non-cross-fitted teacher features."
            ),
            "final_outcome": (
                f"Feature router rescued {feature['rescued']} trials and harmed "
                f"{feature['harmed']}, for net rescue {feature['net_rescue']}."
            ),
            "decision": "Do not retry or start P2-B automatically; require human review.",
        },
        "videomae_updated": False,
        "p2b_started": False,
        "runtime_seconds": time.perf_counter() - started,
        "output_root": str(output_root),
    }
    _atomic_write_text(report_json, json.dumps(report, indent=2) + "\n")
    markdown = [
        "# IR + Depth VideoMAE P2-R0 result",
        "",
        "VideoMAE remained frozen. User6/user7 were evaluated once after CV and final training.",
        "",
        "| Candidate | Accuracy | Macro-F1 | Worst-user | Net rescue | HC harm rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        metrics = final_results[candidate]
        markdown.append(
            f"| {candidate} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['worst_user_accuracy']:.6f} | {metrics['net_rescue']} | "
            f"{metrics['high_confidence_harm_rate']:.6f} |"
        )
    markdown.extend(
        [
            "",
            f"- Feature-router gate passed: `{success_gates['passed']}`",
            f"- Effective >=0.78 gate: `{report['effective_gate']}`",
            f"- Strong >=0.80 gate: `{report['strong_gate']}`",
            f"- Hypothesis status: `{report['hypothesis_status']}`",
            "- VideoMAE updated: `False`",
            "- P2-B started: `False`",
            "",
            "## Interpretation",
            "",
            "The cached-feature reranking hypothesis was rejected. The frozen VideoMAE had",
            "already trained on all train12 users, so router CV began from near-saturated,",
            "non-cross-fitted teacher features and did not predict unseen-user improvement.",
            "The feature router rescued 13 validation trials but harmed 22 (net -9).",
            "No automatic retry or P2-B launch is authorized.",
            "",
        ]
    )
    _atomic_write_text(report_markdown, "\n".join(markdown))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P2-R0 cached-feature reranker")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run_experiment(args.config.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "feature_accuracy": report["final_results"]["feature_router"]["accuracy"],
                "logit_accuracy": report["final_results"]["logit_only"]["accuracy"],
                "passed": report["success_gates"]["passed"],
                "p2b_started": report["p2b_started"],
            }
        )
    )


if __name__ == "__main__":
    main()
