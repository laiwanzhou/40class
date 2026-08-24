from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_depth_videomaev2_p2a.yaml"

from scripts.probe_ir_depth_videomaev2_teacher import load_probe_config
from scripts.run_ir_depth_videomaev2_teacher import _make_dataset, load_training_config
from src.models.ir_depth_videomaev2_teacher import (
    IRDepthVideoMAEV2Teacher,
    build_official_videomaev2_vit_b,
    sha256_file,
)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_p2a_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P2-A config must be a mapping")
    policy = config.get("policy", {})
    inputs = config.get("input_contract", {})
    deferred = config.get("deferred_p2b", {})
    execution = config.get("execution", {})
    if config.get("stage") != "P2-A":
        raise ValueError("P2-A stage changed")
    if inputs.get("modalities") != ["ir", "depth"]:
        raise ValueError("P2-A modalities changed")
    if inputs.get("views") != [
        "global", "person_context", "left_hand_object", "right_hand_object"
    ]:
        raise ValueError("P2-A views changed")
    if inputs.get("frames") != 16 or inputs.get("image_size") != 224:
        raise ValueError("P2-A input shape changed")
    if inputs.get("deterministic") is not True:
        raise ValueError("P2-A must be deterministic")
    if policy.get("training_allowed") is not False or policy.get("p2b_authorized") is not False:
        raise ValueError("P2-A must not authorize training or P2-B")
    if deferred.get("long_trial_frames") != 32 or deferred.get("motion_peak_sampling") is not False:
        raise ValueError("deferred P2-B candidate changed")
    if execution.get("seed") != 20260715 or execution.get("deterministic_algorithms") is not True:
        raise ValueError("P2-A deterministic execution contract changed")
    if execution.get("cublas_workspace_config") != ":4096:8":
        raise ValueError("P2-A CuBLAS deterministic workspace changed")
    if execution.get("maximum_reference_logit_delta") != 0.00001:
        raise ValueError("P2-A reference tolerance changed")
    if policy.get("exploratory_validation_diagnostics") is not True:
        raise ValueError("P2-A diagnostics must be marked exploratory")
    if policy.get("may_select_p2b_without_new_evaluation_boundary") is not False:
        raise ValueError("P2-A may not directly select P2-B")
    return config


def validate_reference_membership(
    current_ids: np.ndarray, reference_ids: np.ndarray
) -> np.ndarray:
    current = np.asarray(current_ids).astype(str)
    reference = np.asarray(reference_ids).astype(str)
    if np.unique(current).size != len(current) or np.unique(reference).size != len(reference):
        raise ValueError("current and reference sample IDs must both be unique")
    if len(current) != len(reference) or set(current.tolist()) != set(reference.tolist()):
        raise ValueError("current and reference sample ID sets must exactly match")
    lookup = {sample_id: index for index, sample_id in enumerate(reference)}
    return np.asarray([lookup[sample_id] for sample_id in current], dtype=np.int64)


def fuse_cached_view_logits(
    view_logits: np.ndarray,
    class_view_gate: np.ndarray,
    availability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    view_logits = np.asarray(view_logits, dtype=np.float32)
    class_view_gate = np.asarray(class_view_gate, dtype=np.float32)
    availability = np.asarray(availability, dtype=bool)
    if view_logits.ndim != 4:
        raise ValueError("view_logits must have shape [N,M,V,C]")
    samples, modalities, views, classes = view_logits.shape
    if class_view_gate.shape != (classes, modalities, views):
        raise ValueError("class_view_gate must have shape [C,M,V]")
    if availability.shape != (samples, modalities, views):
        raise ValueError("availability must have shape [N,M,V]")
    if bool((~availability.any(axis=(1, 2))).any()):
        raise ValueError("every sample requires at least one available stream")
    gate = np.broadcast_to(class_view_gate[None], (samples, classes, modalities, views))
    masked = np.where(availability[:, None], gate, -np.inf)
    maximum = np.max(masked, axis=(2, 3), keepdims=True)
    exponent = np.where(availability[:, None], np.exp(masked - maximum), 0.0)
    weights = exponent / exponent.sum(axis=(2, 3), keepdims=True)
    logits = (view_logits.transpose(0, 3, 1, 2) * weights).sum(axis=(2, 3))
    return logits.astype(np.float32), weights.astype(np.float32)


def _metrics(labels: np.ndarray, logits: np.ndarray, users: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    users = np.asarray(users).astype(str)
    predictions = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    classes = logits.shape[1]
    per_class_recall = []
    for class_id in range(classes):
        selected = labels == class_id
        per_class_recall.append(
            float(np.mean(predictions[selected] == class_id)) if selected.any() else 0.0
        )
    user_accuracy = {
        user: float(accuracy_score(labels[users == user], predictions[users == user]))
        for user in sorted(set(users.tolist()))
    }
    order = np.argsort(-logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels, predictions, labels=np.arange(classes), average="macro", zero_division=0
            )
        ),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(classes))),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "predicted_class_count": int(np.unique(predictions).size),
        "zero_recall_classes": int(np.sum(np.asarray(per_class_recall) == 0.0)),
        "per_class_recall": per_class_recall,
        "top3_accuracy": float(np.mean(np.any(order[:, : min(3, classes)] == labels[:, None], axis=1))),
        "top5_accuracy": float(np.mean(np.any(order[:, : min(5, classes)] == labels[:, None], axis=1))),
    }


def evaluate_cached_ablation(
    *,
    view_logits: np.ndarray,
    class_view_gate: np.ndarray,
    availability: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    modality_names: tuple[str, ...],
    view_names: tuple[str, ...],
) -> dict[str, object]:
    modalities, views = view_logits.shape[1:3]
    if len(modality_names) != modalities or len(view_names) != views:
        raise ValueError("stream names do not match cached dimensions")

    def evaluate(mask: np.ndarray) -> dict[str, object]:
        logits, _ = fuse_cached_view_logits(
            view_logits, class_view_gate, availability & mask[None]
        )
        return _metrics(labels, logits, users)

    full_mask = np.ones((modalities, views), dtype=bool)
    report: dict[str, object] = {"full": evaluate(full_mask)}
    stream_only: dict[str, object] = {}
    stream_drop: dict[str, object] = {}
    for modality_index, modality in enumerate(modality_names):
        for view_index, view in enumerate(view_names):
            name = f"{modality}:{view}"
            only = np.zeros_like(full_mask)
            only[modality_index, view_index] = True
            drop = full_mask.copy()
            drop[modality_index, view_index] = False
            stream_only[name] = evaluate(only)
            stream_drop[name] = evaluate(drop)
    report["stream_only"] = stream_only
    report["stream_drop"] = stream_drop
    modality_only: dict[str, object] = {}
    modality_drop: dict[str, object] = {}
    for modality_index, modality in enumerate(modality_names):
        only = np.zeros_like(full_mask)
        only[modality_index] = True
        drop = full_mask.copy()
        drop[modality_index] = False
        modality_only[modality] = evaluate(only)
        modality_drop[modality] = evaluate(drop)
    report["modality_only"] = modality_only
    report["modality_drop"] = modality_drop
    view_only: dict[str, object] = {}
    view_drop: dict[str, object] = {}
    for view_index, view in enumerate(view_names):
        only = np.zeros_like(full_mask)
        only[:, view_index] = True
        drop = full_mask.copy()
        drop[:, view_index] = False
        view_only[view] = evaluate(only)
        view_drop[view] = evaluate(drop)
    report["view_only"] = view_only
    report["view_drop"] = view_drop
    baseline = float(report["full"]["accuracy"])
    for group in (
        stream_only, stream_drop, modality_only, modality_drop, view_only, view_drop
    ):
        for metrics in group.values():
            metrics["accuracy_delta_vs_full"] = float(metrics["accuracy"]) - baseline
    return report


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _require_artifact(path: Path, *, expected_hash: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"artifact byte mismatch: {path}")
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"artifact SHA256 mismatch: {path}")


def _dataset_coverage(p1_config: dict[str, Any]) -> list[dict[str, object]]:
    data = p1_config["data"]
    manifest = pd.read_csv(
        _project_path(str(data["manifest"])), encoding="utf-8-sig", dtype={"user_id": str}
    )
    split = json.loads(_project_path(str(data["split"])).read_text(encoding="utf-8"))
    audit = pd.read_csv(_project_path(str(data["pairing_audit"])), encoding="utf-8-sig")
    complete = audit["complete_pairing"].astype(str).str.casefold().eq("true")
    valid_ids = set(audit.loc[complete, "sample_id"].astype(str))
    train = manifest[
        manifest["user_id"].isin(split["train_user_ids"])
        & manifest["sample_id"].astype(str).isin(valid_ids)
        & manifest["depth_color_path"].fillna("").astype(str).str.strip().ne("")
        & manifest["ir_path"].fillna("").astype(str).str.strip().ne("")
    ]
    rows = []
    for class_id, frame in train.groupby("class_id"):
        rows.append(
            {
                "class_id": int(class_id),
                "action_name": str(frame["action_name"].iloc[0]),
                "train_trials": int(len(frame)),
                "train_users": int(frame["user_id"].nunique()),
                "status": "deferred_dataset_coverage" if frame["user_id"].nunique() <= 5 else "observed",
            }
        )
    return sorted(rows, key=lambda row: int(row["class_id"]))


def _per_class_frame(
    *,
    labels: np.ndarray,
    diagnostics: dict[str, object],
    coverage: list[dict[str, object]],
) -> pd.DataFrame:
    coverage_by_class = {int(row["class_id"]): row for row in coverage}
    rows: list[dict[str, object]] = []
    full = diagnostics["full"]
    for class_id, baseline_recall in enumerate(full["per_class_recall"]):
        coverage_row = coverage_by_class[class_id]
        row: dict[str, object] = {
            **coverage_row,
            "validation_support": int(np.sum(labels == class_id)),
            "full_recall": float(baseline_recall),
        }
        for group_name in (
            "stream_only", "stream_drop", "modality_only", "modality_drop", "view_only", "view_drop"
        ):
            for experiment, metrics in diagnostics[group_name].items():
                recall = float(metrics["per_class_recall"][class_id])
                row[f"{group_name}__{experiment}__recall"] = recall
                row[f"{group_name}__{experiment}__delta"] = recall - float(baseline_recall)
        rows.append(row)
    return pd.DataFrame(rows)


def _ablation_table(markdown: list[str], title: str, group: dict[str, object]) -> None:
    markdown.extend(
        [
            f"## {title}",
            "",
            "| Experiment | Accuracy | Delta | Macro-F1 | Worst-user | Zero recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in sorted(
        group.items(), key=lambda item: float(item[1]["accuracy"]), reverse=True
    ):
        markdown.append(
            f"| {name} | {metrics['accuracy']:.6f} | "
            f"{metrics['accuracy_delta_vs_full']:+.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['worst_user_accuracy']:.6f} | {metrics['zero_recall_classes']} |"
        )
    markdown.append("")


def run_p2a(config_path: Path, *, refresh: bool = False) -> dict[str, object]:
    config = load_p2a_config(config_path)
    source = config["source"]
    checkpoint_path = _project_path(str(source["checkpoint"]))
    reference_path = _project_path(str(source["reference_predictions"]))
    _require_artifact(
        checkpoint_path,
        expected_hash=str(source["checkpoint_sha256"]),
        expected_bytes=int(source["checkpoint_bytes"]),
    )
    _require_artifact(
        reference_path, expected_hash=str(source["reference_predictions_sha256"])
    )
    output_paths = {name: _project_path(str(value)) for name, value in config["outputs"].items()}
    if output_paths["report_json"].is_file() and not refresh:
        completed = json.loads(output_paths["report_json"].read_text(encoding="utf-8"))
        if completed.get("status") != "completed":
            raise RuntimeError("existing P2-A report is not complete")
        for name in ("cache", "per_class_csv", "report_markdown"):
            artifact = completed["artifacts"][name]
            path = Path(str(artifact["path"]))
            _require_artifact(
                path,
                expected_hash=str(artifact["sha256"]),
                expected_bytes=int(artifact["bytes"]),
            )
        return completed

    seed = int(config["execution"]["seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        config["execution"]["cublas_workspace_config"]
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    p1_config = load_training_config(_project_path(str(source["p1_config"])))
    p0_config = load_probe_config(_project_path(str(p1_config["p0_config"])))
    dataset = _make_dataset(p1_config, partition="validation", training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    backbone, provenance = build_official_videomaev2_vit_b(
        checkpoint_path=Path(str(p0_config["checkpoint"]["path"])),
        num_classes=40,
        with_cp=True,
    )
    model = IRDepthVideoMAEV2Teacher(backbone=backbone, num_classes=40)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()

    sample_ids: list[str] = []
    users: list[str] = []
    labels: list[torch.Tensor] = []
    availability_all: list[torch.Tensor] = []
    sampled_indices: list[torch.Tensor] = []
    view_logits_all: list[torch.Tensor] = []
    view_embeddings_all: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            clips = batch["clips"].to(device, non_blocking=True)
            batch_view_logits = []
            batch_view_embeddings = []
            for modality_index in range(clips.shape[1]):
                modality_logits = []
                modality_embeddings = []
                for view_index in range(clips.shape[2]):
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        embedding = model.backbone.forward_features(
                            clips[:, modality_index, view_index]
                        )
                        logits = model.backbone.head(
                            model.backbone.head_dropout(embedding)
                        )
                    modality_logits.append(logits.float().cpu())
                    modality_embeddings.append(embedding.float().cpu())
                batch_view_logits.append(torch.stack(modality_logits, dim=1))
                batch_view_embeddings.append(torch.stack(modality_embeddings, dim=1))
            view_logits_all.append(torch.stack(batch_view_logits, dim=1))
            view_embeddings_all.append(torch.stack(batch_view_embeddings, dim=1))
            availability_all.append(batch["availability"].bool().cpu())
            labels.append(batch["label"].long().cpu())
            sampled_indices.append(batch["sampled_indices"].long().cpu())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            users.extend(str(value) for value in batch["user_id"])

    view_logits = torch.cat(view_logits_all).numpy().astype(np.float32)
    view_embeddings = torch.cat(view_embeddings_all).numpy().astype(np.float16)
    availability = torch.cat(availability_all).numpy().astype(bool)
    labels_np = torch.cat(labels).numpy().astype(np.int64)
    sampled_indices_np = torch.cat(sampled_indices).numpy().astype(np.int64)
    sample_ids_np = np.asarray(sample_ids)
    users_np = np.asarray(users)
    num_frames = sampled_indices_np[:, -1] + 1
    class_view_gate = model.class_view_gate.detach().float().cpu().numpy()
    full_logits, full_weights = fuse_cached_view_logits(
        view_logits, class_view_gate, availability
    )

    reference = np.load(reference_path, allow_pickle=False)
    reference_ids = reference["sample_ids"].astype(str)
    take = validate_reference_membership(sample_ids_np, reference_ids)
    reference_logits = reference["logits"][take].astype(np.float32)
    reference_labels = reference["labels"][take].astype(np.int64)
    reference_users = reference["users"][take].astype(str)
    max_logit_delta = float(np.max(np.abs(full_logits - reference_logits)))
    argmax_exact = bool(
        np.array_equal(full_logits.argmax(axis=1), reference_logits.argmax(axis=1))
    )
    maximum_delta = float(config["execution"]["maximum_reference_logit_delta"])
    if (
        not np.array_equal(labels_np, reference_labels)
        or not np.array_equal(users_np, reference_users)
        or not argmax_exact
        or max_logit_delta > maximum_delta
    ):
        raise RuntimeError("P2-A cache does not reproduce selected validation predictions")

    diagnostics = evaluate_cached_ablation(
        view_logits=view_logits,
        class_view_gate=class_view_gate,
        availability=availability,
        labels=labels_np,
        users=users_np,
        modality_names=tuple(config["input_contract"]["modalities"]),
        view_names=tuple(config["input_contract"]["views"]),
    )
    entropy = -(
        full_weights * np.log(np.clip(full_weights, 1e-12, 1.0))
    ).sum(axis=(2, 3))
    stream_predictions = view_logits.argmax(axis=3)
    stream_oracle = np.any(stream_predictions == labels_np[:, None, None], axis=(1, 2))
    coverage = _dataset_coverage(p1_config)
    per_class = _per_class_frame(
        labels=labels_np, diagnostics=diagnostics, coverage=coverage
    )

    output_paths["cache"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output_paths["cache"],
        sample_ids=sample_ids_np,
        user_ids=users_np,
        labels=labels_np,
        num_frames=num_frames.astype(np.int64),
        sampled_indices=sampled_indices_np,
        availability=availability,
        view_logits=view_logits,
        view_embeddings=view_embeddings,
        class_view_gate=class_view_gate,
        class_view_weights=full_weights,
        full_logits=full_logits,
    )
    output_paths["per_class_csv"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_paths["per_class_csv"], per_class)
    diagnostic_sources = (
        config_path.resolve(),
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/models/ir_depth_videomaev2_teacher.py",
        PROJECT_ROOT / "src/data/ir_depth_videomaev2_dataset.py",
        PROJECT_ROOT / "scripts/run_ir_depth_videomaev2_teacher.py",
    )
    report = {
        "schema_version": 1,
        "stage": "P2-A",
        "status": "completed",
        "training_performed": False,
        "exploratory_validation_diagnostics": True,
        "source": {
            "selected_epoch": int(checkpoint["epoch"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "reference_predictions": str(reference_path),
            "reference_predictions_sha256": sha256_file(reference_path),
            "backbone_provenance": provenance,
            "diagnostic_source_sha256": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
                for path in diagnostic_sources
            },
        },
        "execution": {
            "seed": seed,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "claim_boundary": "deterministic sampling and algorithms within the recorded stack",
        },
        "cache": {
            "path": str(output_paths["cache"]),
            "sha256": sha256_file(output_paths["cache"]),
            "bytes": output_paths["cache"].stat().st_size,
            "sample_count": len(labels_np),
            "view_logits_shape": list(view_logits.shape),
            "view_embeddings_shape": list(view_embeddings.shape),
            "embedding_dtype": str(view_embeddings.dtype),
            "finite": bool(
                np.isfinite(view_logits).all()
                and np.isfinite(view_embeddings).all()
                and np.isfinite(full_logits).all()
            ),
        },
        "reproduction": {
            "sample_count_exact": len(sample_ids_np) == len(reference_ids),
            "unique_sample_ids": np.unique(sample_ids_np).size == len(sample_ids_np),
            "sample_id_set_exact": set(sample_ids_np.tolist()) == set(reference_ids.tolist()),
            "labels_exact": True,
            "users_exact": True,
            "argmax_exact": argmax_exact,
            "maximum_logit_delta": max_logit_delta,
            "maximum_allowed_logit_delta": maximum_delta,
            "reference_accuracy": float(
                accuracy_score(labels_np, reference_logits.argmax(axis=1))
            ),
            "cache_accuracy": float(diagnostics["full"]["accuracy"]),
        },
        "gate": {
            "mean_entropy": float(entropy.mean()),
            "minimum_entropy": float(entropy.min()),
            "maximum_entropy": float(entropy.max()),
            "uniform_entropy": float(np.log(view_logits.shape[1] * view_logits.shape[2])),
            "mean_absolute_weight_deviation_from_uniform": float(
                np.abs(full_weights - 1.0 / (view_logits.shape[1] * view_logits.shape[2])).mean()
            ),
            "maximum_absolute_weight_deviation_from_uniform": float(
                np.abs(full_weights - 1.0 / (view_logits.shape[1] * view_logits.shape[2])).max()
            ),
        },
        "stream_oracle_accuracy": float(stream_oracle.mean()),
        "diagnostics": diagnostics,
        "dataset_coverage": coverage,
        "deferred_p2b": config["deferred_p2b"],
        "p2b_started": False,
        "may_select_p2b_without_new_evaluation_boundary": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    markdown = [
        "# IR + Depth VideoMAE V2 P2-A diagnostics",
        "",
        f"- Full Accuracy: `{diagnostics['full']['accuracy']:.6f}`",
        f"- Full Macro-F1: `{diagnostics['full']['macro_f1']:.6f}`",
        f"- Eight-stream prediction oracle: `{stream_oracle.mean():.6f}`",
        f"- Gate entropy: `{entropy.mean():.6f}` / uniform `{np.log(8):.6f}`",
        f"- Cached logits/embeddings finite: `{report['cache']['finite']}`",
        f"- Reproduced selected predictions exactly: `{argmax_exact}`",
        "",
        "Current fusion is static class-conditioned late-logit fusion. It is not sample-conditioned",
        "and does not perform IR/Depth feature interaction before the classifier.",
        "",
        "> These user6/user7 ablations are exploratory diagnostics. They must not be used",
        "> to fit or select a gate/P2-B recipe without a new user-grouped evaluation boundary.",
        "",
    ]
    _ablation_table(markdown, "Modality only", diagnostics["modality_only"])
    _ablation_table(markdown, "Modality drop", diagnostics["modality_drop"])
    _ablation_table(markdown, "View type only", diagnostics["view_only"])
    _ablation_table(markdown, "View type drop", diagnostics["view_drop"])
    _ablation_table(markdown, "Individual stream only", diagnostics["stream_only"])
    _ablation_table(markdown, "Individual stream drop", diagnostics["stream_drop"])
    markdown.extend(
        [
            "## Dataset coverage deferred",
            "",
            "Competition-provided classes with five or fewer train12 users are recorded as",
            "`deferred_dataset_coverage`; this diagnostic does not attempt to repair them.",
            "",
            "## P2-B boundary",
            "",
            "P2-B was not started. The only recorded candidate is fixed 32 frames for long trials;",
            "motion-peak sampling is explicitly disabled pending this report's review.",
            "",
        ]
    )
    deferred_rows = [row for row in coverage if row["status"] == "deferred_dataset_coverage"]
    if deferred_rows:
        markdown.extend(
            [
                "Deferred classes: "
                + ", ".join(
                    f"`{row['class_id']} {row['action_name']}` ({row['train_users']} users)"
                    for row in deferred_rows
                ),
                "",
            ]
        )
    _atomic_write_text(output_paths["report_markdown"], "\n".join(markdown))
    report["artifacts"] = {
        name: {
            "path": str(output_paths[name]),
            "sha256": sha256_file(output_paths[name]),
            "bytes": output_paths[name].stat().st_size,
        }
        for name in ("cache", "per_class_csv", "report_markdown")
    }
    _atomic_write_text(
        output_paths["report_json"], json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache P2-A per-view VideoMAE diagnostics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute and atomically replace an existing completed P2-A diagnostic.",
    )
    args = parser.parse_args()
    report = run_p2a(args.config.resolve(), refresh=args.refresh)
    print(
        json.dumps(
            {
                "status": report["status"],
                "full_accuracy": report["diagnostics"]["full"]["accuracy"],
                "stream_oracle_accuracy": report["stream_oracle_accuracy"],
                "p2b_started": report["p2b_started"],
            }
        )
    )


if __name__ == "__main__":
    main()
