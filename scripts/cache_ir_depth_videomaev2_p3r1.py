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
import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_depth_videomaev2_p3r1.yaml"

from scripts.run_ir_depth_videomaev2_aggressive_dev import (
    _model,
    load_aggressive_config,
)
from scripts.probe_ir_depth_videomaev2_teacher import load_probe_config
from src.data.ir_depth_videomaev2_dataset import IRDepthVideoMAEV2Dataset
from src.models.ir_depth_videomaev2_teacher import sha256_file
from src.models.margin_conditioned_top3_routing import build_route_bank


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_p3r1_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P3-R1 config must be a mapping")
    split, source = config.get("split", {}), config.get("source", {})
    policy, cv = config.get("policy", {}), config.get("cv", {})
    if config.get("stage") != "P3-R1":
        raise ValueError("P3-R1 stage changed")
    train_users = [str(value) for value in split.get("train_user_ids", [])]
    validation_users = [str(value) for value in split.get("validation_user_ids", [])]
    if set(train_users) & set(validation_users):
        raise ValueError("P3-R1 train and validation users overlap")
    if validation_users != ["user6", "user7"]:
        raise ValueError("P3-R1 validation users changed")
    fold_users = [str(user) for fold in cv.get("folds", []) for user in fold]
    if len(fold_users) != len(set(fold_users)) or set(fold_users) != set(train_users):
        raise ValueError("P3-R1 CV folds must partition train users")
    required_false = (
        "update_videomae",
        "validation_users_enter_gradient",
        "validation_users_enter_normalization",
        "validation_users_enter_sampler",
        "validation_users_enter_cv_selection",
    )
    if any(policy.get(key) is not False for key in required_false):
        raise ValueError("P3-R1 validation isolation policy changed")
    if int(source.get("selected_epoch", -1)) != 14:
        raise ValueError("P3-R1 selected checkpoint epoch changed")
    return config


def validate_cache_membership(
    *,
    partition: str,
    sample_ids: np.ndarray,
    user_ids: np.ndarray,
    labels: np.ndarray,
    expected_samples: int,
    config: dict[str, Any],
) -> None:
    samples = np.asarray(sample_ids).astype(str)
    users = np.asarray(user_ids).astype(str)
    targets = np.asarray(labels, dtype=np.int64)
    if len(samples) != expected_samples or len(users) != expected_samples:
        raise ValueError(f"{partition} sample count changed")
    if np.unique(samples).size != len(samples):
        raise ValueError(f"{partition} sample IDs are not unique")
    expected_users = set(config["split"][f"{partition}_user_ids"])
    if set(users.tolist()) != expected_users:
        raise ValueError(f"{partition} user membership changed")
    if targets.shape != (expected_samples,) or bool(
        ((targets < 0) | (targets >= int(config["split"]["class_count"]))).any()
    ):
        raise ValueError(f"{partition} labels changed")


def validate_reference_predictions(
    *,
    current: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    maximum_logit_delta: float,
) -> dict[str, float | int]:
    current_ids = current["sample_ids"].astype(str)
    reference_ids = reference["sample_ids"].astype(str)
    if set(current_ids.tolist()) != set(reference_ids.tolist()):
        raise RuntimeError("selected prediction sample membership changed")
    lookup = {sample_id: index for index, sample_id in enumerate(reference_ids)}
    order = np.asarray([lookup[sample_id] for sample_id in current_ids], dtype=np.int64)
    if not np.array_equal(current["labels"], reference["labels"][order]):
        raise RuntimeError("selected prediction labels changed")
    anchor = current["route_logits"][:, 0].astype(np.float32)
    expected = reference["logits"][order].astype(np.float32)
    delta = float(np.max(np.abs(anchor - expected)))
    disagreement = int((anchor.argmax(axis=1) != expected.argmax(axis=1)).sum())
    if delta > float(maximum_logit_delta) or disagreement:
        raise RuntimeError(
            f"cached anchor logits changed: max_delta={delta}, disagreements={disagreement}"
        )
    return {"maximum_logit_delta": delta, "prediction_disagreement": disagreement}


def _dataset(config: dict[str, Any], *, partition: str) -> IRDepthVideoMAEV2Dataset:
    aggressive = load_aggressive_config(
        _project_path(str(config["source"]["aggressive_config"]))
    )
    data = aggressive["data"]
    p0 = load_probe_config(_project_path(str(aggressive["p0_input_config"])))
    return IRDepthVideoMAEV2Dataset(
        manifest_path=_project_path(str(data["manifest"])),
        split_path=_project_path(str(aggressive["split"]["path"])),
        data_root=Path(str(data["root"])),
        pose_cache_path=Path(str(data["pose_cache"])),
        pairing_audit_path=_project_path(str(data["pairing_audit"])),
        partition=partition,
        training=False,
        frames=int(data["frames"]),
        image_size=int(data["image_size"]),
        temporal_jitter=0.0,
        interaction_config=dict(p0["roi"]),
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _cache_partition(
    *,
    model: torch.nn.Module,
    dataset: IRDepthVideoMAEV2Dataset,
    partition: str,
    config: dict[str, Any],
    device: torch.device,
    smoke_test: bool,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    fused_embeddings, ir_embeddings, route_logits, route_weights = [], [], [], []
    labels, users, sample_ids, availability_rows, num_frames = [], [], [], [], []
    gates, delta_norms = [], []
    route_names: tuple[str, ...] | None = None
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            clips = batch["clips"].to(device, non_blocking=True)
            availability = batch["availability"].to(device, non_blocking=True)
            ir_availability = availability.clone()
            ir_availability[:, 1] = False
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                full = model(
                    ir=clips[:, 0], depth=clips[:, 1], availability=availability
                )
                ir_only = model(
                    ir=clips[:, 0], depth=clips[:, 1], availability=ir_availability
                )
                bank = build_route_bank(
                    fused_view_embeddings=full["view_embeddings"],
                    ir_view_embeddings=ir_only["view_embeddings"],
                    class_queries=model.class_queries,
                    class_view_bias=model.class_view_bias,
                    head_weight=model.backbone.head.weight,
                    head_bias=model.backbone.head.bias,
                    availability=availability.any(dim=1),
                )
            route_anchor_delta = float(
                (bank["route_logits"][:, 0] - full["logits"]).abs().max().float().cpu()
            )
            if route_anchor_delta > 0.00001:
                raise RuntimeError(
                    f"offline hard Top-2 failed to reproduce model logits: {route_anchor_delta}"
                )
            current_names = tuple(str(value) for value in bank["route_names"])
            if route_names is None:
                route_names = current_names
            elif current_names != route_names:
                raise RuntimeError("route bank order changed within cache")
            fused_embeddings.append(full["view_embeddings"].half().cpu())
            ir_embeddings.append(ir_only["view_embeddings"].half().cpu())
            route_logits.append(bank["route_logits"].float().cpu())
            route_weights.append(bank["route_view_weights"].half().cpu())
            gates.append(full["depth_gates"].float().cpu())
            delta_norms.append(full["depth_delta_norm"].float().cpu())
            labels.append(batch["label"].long().cpu())
            users.extend(str(value) for value in batch["user_id"])
            sample_ids.extend(str(value) for value in batch["sample_id"])
            availability_rows.append(availability.cpu())
            sampled = batch["sampled_indices"].long()
            num_frames.extend(int(row.max().item()) + 1 for row in sampled)
            if step % 100 == 0 or step == len(loader):
                print(
                    json.dumps(
                        {
                            "partition": partition,
                            "cached": step,
                            "total": len(loader),
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            if smoke_test and step >= 2:
                break
    assert route_names is not None
    arrays = {
        "sample_ids": np.asarray(sample_ids),
        "user_ids": np.asarray(users),
        "labels": torch.cat(labels).numpy(),
        "num_frames": np.asarray(num_frames, dtype=np.int64),
        "availability": torch.cat(availability_rows).numpy(),
        "fused_view_embeddings": torch.cat(fused_embeddings).numpy(),
        "ir_view_embeddings": torch.cat(ir_embeddings).numpy(),
        "route_names": np.asarray(route_names),
        "route_logits": torch.cat(route_logits).numpy(),
        "route_view_weights": torch.cat(route_weights).numpy(),
        "depth_gates": torch.cat(gates).numpy(),
        "depth_delta_norm": torch.cat(delta_norms).numpy(),
    }
    expected = len(arrays["labels"]) if smoke_test else int(
        config["split"][f"{partition}_samples"]
    )
    if not smoke_test:
        validate_cache_membership(
            partition=partition,
            sample_ids=arrays["sample_ids"],
            user_ids=arrays["user_ids"],
            labels=arrays["labels"],
            expected_samples=expected,
            config=config,
        )
        if np.unique(arrays["labels"]).size != int(config["split"]["class_count"]):
            raise ValueError(f"{partition} cache does not contain all classes")
    if arrays["route_logits"].shape[:2] != (expected, len(route_names)):
        raise RuntimeError(f"{partition} route cache shape changed")
    if not all(np.isfinite(arrays[key]).all() for key in (
        "fused_view_embeddings", "ir_view_embeddings", "route_logits",
        "route_view_weights", "depth_gates", "depth_delta_norm",
    )):
        raise FloatingPointError(f"non-finite values in {partition} route cache")
    return arrays, route_names


def run(config_path: Path, *, smoke_test: bool = False) -> dict[str, Any]:
    config = load_p3r1_config(config_path)
    source = config["source"]
    checkpoint = _project_path(str(source["checkpoint"]))
    if checkpoint.stat().st_size != int(source["checkpoint_bytes"]):
        raise RuntimeError("P3-R1 checkpoint byte count changed")
    if sha256_file(checkpoint) != str(source["checkpoint_sha256"]):
        raise RuntimeError("P3-R1 checkpoint hash changed")
    seed = int(config["reranker"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    aggressive = load_aggressive_config(
        _project_path(str(source["aggressive_config"]))
    )
    model = _model(aggressive).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != int(source["selected_epoch"]):
        raise RuntimeError("P3-R1 checkpoint epoch changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)

    artifacts: dict[str, dict[str, Any]] = {}
    partition_arrays: dict[str, dict[str, np.ndarray]] = {}
    canonical_routes: tuple[str, ...] | None = None
    for partition in ("train", "validation"):
        output_path = _project_path(str(config["cache"][partition]))
        if not smoke_test and output_path.exists():
            raise FileExistsError(output_path)
        arrays, route_names = _cache_partition(
            model=model,
            dataset=_dataset(config, partition=partition),
            partition=partition,
            config=config,
            device=device,
            smoke_test=smoke_test,
        )
        if canonical_routes is None:
            canonical_routes = route_names
        elif route_names != canonical_routes:
            raise RuntimeError("train and validation route order differs")
        partition_arrays[partition] = arrays
        artifacts[partition] = {"samples": len(arrays["labels"])}
    reference_validation: dict[str, np.ndarray] | None = None
    reference_result: dict[str, float | int] | None = None
    if not smoke_test:
        reference_path = _project_path(str(source["selected_validation_predictions"]))
        if (
            reference_path.stat().st_size != int(source["selected_validation_predictions_bytes"])
            or sha256_file(reference_path) != source["selected_validation_predictions_sha256"]
        ):
            raise RuntimeError("selected validation prediction provenance changed")
        with np.load(reference_path, allow_pickle=False) as archive:
            reference_validation = {key: archive[key] for key in archive.files}
        reference_result = validate_reference_predictions(
            current=partition_arrays["validation"],
            reference=reference_validation,
            maximum_logit_delta=float(source["maximum_anchor_logit_delta"]),
        )
        for partition in ("train", "validation"):
            output_path = _project_path(str(config["cache"][partition]))
            _atomic_npz(output_path, **partition_arrays[partition])
            artifacts[partition] = {
                "path": str(config["cache"][partition]),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "samples": len(partition_arrays[partition]["labels"]),
            }
    report = {
        "stage": "P3-R1-cache",
        "status": "smoke_passed" if smoke_test else "completed",
        "checkpoint": {
            "path": str(source["checkpoint"]),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "epoch": int(source["selected_epoch"]),
        },
        "route_names": list(canonical_routes or ()),
        "artifacts": artifacts,
        "selected_prediction_reproduction": reference_result,
        "validation_users_entered_training": False,
    }
    if not smoke_test:
        report_path = _project_path(str(config["cache"]["report"]))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache aggressive VideoMAE P3-R1 routes")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), smoke_test=args.smoke_test)))


if __name__ == "__main__":
    main()
