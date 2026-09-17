from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from p86_mc3_visual_model import P86MC3VisualStudent
from p86_visual_pixel_data import P86VisualPixelDataset, collate_p86_pixels
from p86_visual_pixel_model import P86TrainableVisualStudent
from train_p86_visual_pixel_oof import (
    DEFAULT_TEACHER_FEATURES,
    DEFAULT_TEACHER_LOGITS,
    batch_to_device,
    load_npz,
    split_universe,
)


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract subject-pure frozen visual logits/embeddings for P86 modality probes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pixel-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("outer_train", "all"), default="outer_train"
    )
    parser.add_argument("--outer-fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint: dict[str, Any]) -> nn.Module:
    config = dict(checkpoint["model_config"])
    backbone = str(config.pop("backbone", "resnet18_tsm"))
    config.pop("freeze_through", None)
    if backbone == "mc3_18":
        model: nn.Module = P86MC3VisualStudent(
            **config, kinetics_pretrained=False
        )
    elif backbone == "resnet18_tsm":
        model = P86TrainableVisualStudent(**config)
    else:
        raise ValueError(f"unknown checkpoint backbone: {backbone}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, args.outer_fold, args.seed)
    split_ids = np.asarray(split["sample_ids"]).astype(str)
    role_lookup: dict[str, str] = {}
    for role in ("inner_train", "inner_dev", "outer_held"):
        for sample_id in split_ids[split[role]]:
            role_lookup[str(sample_id)] = role

    reference = P86VisualPixelDataset(
        args.pixel_cache, args.teacher_features, args.teacher_logits
    )
    if args.scope == "outer_train":
        requested = split_ids[split["outer_train"]].tolist()
    else:
        requested = [row["sample_id"] for row in reference.rows]
    indices = np.asarray(
        [reference.index_lookup[sample_id] for sample_id in requested], dtype=np.int64
    )
    dataset = P86VisualPixelDataset(
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        indices=indices,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_p86_pixels,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint).to(device).eval()
    sample_ids: list[str] = []
    users: list[str] = []
    labels: list[int] = []
    logits: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    view_weights: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(
                    batch["images"], batch["view_valid"], batch["view_quality"]
                )
            sample_ids.extend(batch["sample_id"])
            users.extend(batch["user_id"])
            labels.extend(batch["label"].cpu().tolist())
            logits.append(output["logits"].float().cpu().numpy())
            embeddings.append(output["visual_embedding"].float().cpu().numpy())
            view_weights.append(output["view_weight"].float().cpu().numpy())

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roles = np.asarray([role_lookup.get(sample_id, "outer_held") for sample_id in sample_ids])
    np.savez_compressed(
        output_path,
        sample_ids=np.asarray(sample_ids),
        users=np.asarray(users),
        labels=np.asarray(labels, dtype=np.int64),
        split_roles=roles,
        logits=np.concatenate(logits).astype(np.float32),
        visual_embeddings=np.concatenate(embeddings).astype(np.float32),
        view_weights=np.concatenate(view_weights).astype(np.float32),
        source_checkpoint=np.asarray(str(checkpoint_path)),
        source_checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        training_scope=np.asarray(str(checkpoint.get("training_scope", checkpoint.get("stage", "unknown")))),
        outer_fold=np.asarray(args.outer_fold, dtype=np.int64),
    )
    summary = {
        "stage": "P86_frozen_visual_anchor_cache",
        "scope": args.scope,
        "samples": len(sample_ids),
        "roles": {
            role: int((roles == role).sum()) for role in sorted(set(roles.tolist()))
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "output": str(output_path),
        "large_videomae_required_at_inference": False,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
