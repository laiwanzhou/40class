from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_ir_depth_videomaev2_p2a import (
    _atomic_npz,
    _atomic_write_text,
    _require_artifact,
    fuse_cached_view_logits,
)
from scripts.probe_ir_depth_videomaev2_teacher import load_probe_config
from scripts.run_ir_depth_videomaev2_p2r0 import load_p2r0_config
from scripts.run_ir_depth_videomaev2_teacher import _make_dataset, load_training_config
from src.models.ir_depth_videomaev2_teacher import (
    IRDepthVideoMAEV2Teacher,
    build_official_videomaev2_vit_b,
    sha256_file,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/ir_depth_videomaev2_p2r0.yaml"


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_ir_anchor_logits(
    view_logits: np.ndarray,
    class_view_gate: np.ndarray,
    availability: np.ndarray,
) -> np.ndarray:
    ir_only = np.zeros_like(availability, dtype=bool)
    ir_only[:, 0] = availability[:, 0]
    logits, _ = fuse_cached_view_logits(
        view_logits, class_view_gate, ir_only
    )
    return logits


def run_cache(config_path: Path) -> dict[str, object]:
    config = load_p2r0_config(config_path)
    source = config["source"]
    checkpoint_path = _project_path(str(source["checkpoint"]))
    output_path = _project_path(str(source["train_cache"]))
    report_path = _project_path(str(source["train_cache_report"]))
    if output_path.exists() or report_path.exists():
        raise FileExistsError("P2-R0 train cache output already exists")
    _require_artifact(
        checkpoint_path,
        expected_hash=str(source["checkpoint_sha256"]),
        expected_bytes=345_104_224,
    )
    seed = int(config["training"]["seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    p1_config = load_training_config(_project_path(str(source["p1_config"])))
    p0_config = load_probe_config(_project_path(str(p1_config["p0_config"])))
    dataset = _make_dataset(p1_config, partition="train", training=False)
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
            modality_logits = []
            modality_embeddings = []
            for modality_index in range(2):
                logits_by_view = []
                embeddings_by_view = []
                for view_index in range(4):
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        embedding = model.backbone.forward_features(
                            clips[:, modality_index, view_index]
                        )
                        logits = model.backbone.head(model.backbone.head_dropout(embedding))
                    logits_by_view.append(logits.float().cpu())
                    embeddings_by_view.append(embedding.float().cpu())
                modality_logits.append(torch.stack(logits_by_view, dim=1))
                modality_embeddings.append(torch.stack(embeddings_by_view, dim=1))
            view_logits_all.append(torch.stack(modality_logits, dim=1))
            view_embeddings_all.append(torch.stack(modality_embeddings, dim=1))
            sample_ids.extend(str(value) for value in batch["sample_id"])
            users.extend(str(value) for value in batch["user_id"])
            labels.append(batch["label"].long().cpu())
            availability_all.append(batch["availability"].bool().cpu())
            sampled_indices.append(batch["sampled_indices"].long().cpu())

    sample_ids_np = np.asarray(sample_ids)
    users_np = np.asarray(users)
    labels_np = torch.cat(labels).numpy().astype(np.int64)
    availability = torch.cat(availability_all).numpy().astype(bool)
    sampled_indices_np = torch.cat(sampled_indices).numpy().astype(np.int64)
    view_logits = torch.cat(view_logits_all).numpy().astype(np.float32)
    view_embeddings = torch.cat(view_embeddings_all).numpy().astype(np.float16)
    class_view_gate = model.class_view_gate.detach().float().cpu().numpy()
    ir_anchor_logits = build_ir_anchor_logits(
        view_logits, class_view_gate, availability
    )
    full_logits, _ = fuse_cached_view_logits(
        view_logits, class_view_gate, availability
    )
    split = json.loads(
        _project_path(str(p1_config["data"]["split"])).read_text(encoding="utf-8")
    )
    if (
        len(sample_ids_np) != 1935
        or np.unique(sample_ids_np).size != len(sample_ids_np)
        or np.unique(labels_np).size != 40
        or set(users_np.tolist()) != set(split["train_user_ids"])
    ):
        raise RuntimeError("unexpected P2-R0 train12 cache membership")
    if not (
        np.isfinite(view_logits).all()
        and np.isfinite(view_embeddings).all()
        and np.isfinite(ir_anchor_logits).all()
        and np.isfinite(full_logits).all()
    ):
        raise FloatingPointError("non-finite P2-R0 train cache")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output_path,
        sample_ids=sample_ids_np,
        user_ids=users_np,
        labels=labels_np,
        num_frames=(sampled_indices_np[:, -1] + 1).astype(np.int64),
        sampled_indices=sampled_indices_np,
        availability=availability,
        view_logits=view_logits,
        view_embeddings=view_embeddings,
        class_view_gate=class_view_gate,
        ir_anchor_logits=ir_anchor_logits,
        full_logits=full_logits,
    )
    report = {
        "schema_version": 1,
        "stage": "P2-R0-train-cache",
        "status": "completed",
        "training_performed": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cache": str(output_path),
        "cache_sha256": sha256_file(output_path),
        "cache_bytes": output_path.stat().st_size,
        "sample_count": len(sample_ids_np),
        "unique_sample_ids": np.unique(sample_ids_np).size == len(sample_ids_np),
        "class_count": int(np.unique(labels_np).size),
        "users": sorted(set(users_np.tolist())),
        "view_logits_shape": list(view_logits.shape),
        "view_embeddings_shape": list(view_embeddings.shape),
        "finite": True,
        "runtime_seconds": time.perf_counter() - started,
        "backbone_provenance": provenance,
        "execution": {
            "seed": seed,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache train12 VideoMAE views for P2-R0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run_cache(args.config.resolve())
    print(json.dumps({"status": report["status"], "samples": report["sample_count"]}))


if __name__ == "__main__":
    main()
