from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset
from evaluate_p27_strong_joint_imu import P12_PROTOCOLS, align_imu
from probe_p27r3_incremental_information import metric_bundle, write_csv
from residual_logit_fusion import build_model
from train import move_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label-free temporal/flip multi-view evaluation of an aligned model"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--imu-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-fold", type=int, default=0)
    parser.add_argument(
        "--temporal-views", type=float, nargs="+", default=[0.2, 0.5, 0.8]
    )
    parser.add_argument("--include-flip", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(values: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, metrics in values.items()
        for key, value in metrics.items()
    }


def infer_view(
    model: torch.nn.Module,
    config: dict[str, Any],
    manifest: Path,
    device: torch.device,
    temporal_view: float,
    flip: bool,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modalities = list(config["modalities"])
    dataset = AlignedMultimodalDataset(
        manifest_path=manifest,
        split="val",
        modalities=modalities,
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=False,
        cache_dir=config.get("cache_dir"),
        skeleton_strategy=config.get("skeleton_strategy", "first"),
        depth_representation=config.get("depth_representation", "jet_rgb"),
        visual_normalization=config.get("visual_normalization", "legacy"),
        skeleton_representation=config.get(
            "skeleton_representation", "frame_joint"
        ),
        skeleton_raw_cache_dir=config.get("skeleton_raw_cache_dir"),
        ir_motion_mode=config.get("ir_motion_mode", "none"),
        ir_roi_csv=config.get("ir_roi_csv"),
        ir_roi_context=float(config.get("ir_roi_context", 0.3)),
        temporal_view=float(temporal_view),
        force_horizontal_flip=bool(flip),
        temporal_sampling=str(config.get("temporal_sampling", "uniform")),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    sample_ids: list[str] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = move_inputs(batch, modalities, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(inputs)
            if not isinstance(output, torch.Tensor):
                raise TypeError("aligned checkpoint must return a logits tensor")
            sample_ids.extend(batch["sample_id"])
            labels.append(batch["label"])
            logits.append(output.float().cpu())
    return (
        np.asarray(sample_ids),
        torch.cat(labels).numpy(),
        torch.cat(logits).numpy(),
    )


def main() -> None:
    args = parse_args()
    fold = int(args.inner_fold)
    if fold not in P12_PROTOCOLS:
        raise ValueError(f"unsupported inner fold {fold}")
    views = sorted(set(float(value) for value in args.temporal_views))
    if any(not 0.0 <= value <= 1.0 for value in views):
        raise ValueError("temporal views must lie in [0, 1]")
    flips = [False, True] if args.include_flip else [False]
    checkpoint_path = args.checkpoint.resolve()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device)
    model.eval()
    reference_ids: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    view_logits: dict[str, np.ndarray] = {}
    for temporal_view in views:
        for flip in flips:
            key = f"t{temporal_view:.2f}_{'flip' if flip else 'plain'}"
            sample_ids, labels, logits = infer_view(
                model,
                config,
                manifest,
                device,
                temporal_view,
                flip,
                int(args.num_workers),
            )
            if reference_ids is None:
                reference_ids = sample_ids
                reference_labels = labels
            elif not (
                np.array_equal(reference_ids, sample_ids)
                and np.array_equal(reference_labels, labels)
            ):
                raise RuntimeError("multi-view sample alignment changed")
            view_logits[key] = logits.astype(np.float32)
    assert reference_ids is not None and reference_labels is not None
    with np.load(args.imu_archive.resolve(), allow_pickle=False) as imu_archive:
        imu_logits, device_counts, imu_present = align_imu(
            imu_archive, reference_ids, reference_labels
        )
    joint_temperature, imu_temperature, base_weight = P12_PROTOCOLS[fold]
    weights = (
        base_weight
        * np.clip(device_counts.astype(np.float32) / 5.0, 0.0, 1.0)
    )[:, None]

    center_key = min(
        (key for key in view_logits if key.endswith("plain")),
        key=lambda key: abs(float(key[1:5]) - 0.5),
    )
    plain_logits = np.mean(
        [value for key, value in view_logits.items() if key.endswith("plain")],
        axis=0,
    )
    all_logits = np.mean(list(view_logits.values()), axis=0)
    joint_methods = {
        "single_center": view_logits[center_key],
        "temporal_multiview": plain_logits,
        "temporal_flip_multiview": all_logits,
    }
    rows: list[dict[str, Any]] = []
    methods: dict[str, np.ndarray] = {}
    for name, joint_logits in joint_methods.items():
        methods[name] = joint_logits
        methods[f"{name}_rfimu_fixed"] = (
            (1.0 - weights) * joint_logits / joint_temperature
            + weights * imu_logits / imu_temperature
        )
    for name, logits in methods.items():
        rows.append(
            {
                "method": name,
                **flatten(
                    metric_bundle(reference_labels, logits.argmax(axis=1))
                ),
            }
        )
    write_csv(output_dir / "metrics.csv", rows)
    subjects_by_id: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "val":
                subjects_by_id[row["sample_id"]] = row["user_id"]
    subjects = np.asarray([subjects_by_id[str(value)] for value in reference_ids])
    subject_rows: list[dict[str, Any]] = []
    for name, logits in methods.items():
        predictions = logits.argmax(axis=1)
        for subject in sorted(set(subjects.tolist())):
            selected = subjects == subject
            subject_rows.append(
                {
                    "method": name,
                    "subject": subject,
                    **metric_bundle(
                        reference_labels[selected], predictions[selected]
                    )["overall"],
                }
            )
    write_csv(output_dir / "per_subject.csv", subject_rows)
    np.savez_compressed(
        output_dir / "multiview_logits.npz",
        protocol=np.asarray("p27-aligned-label-free-multiview-inner-v1"),
        sample_ids=reference_ids,
        labels=reference_labels,
        subjects=subjects,
        imu_logits=imu_logits,
        imu_present=imu_present,
        imu_device_counts=device_counts,
        **{f"{name}_logits": value for name, value in methods.items()},
        outer_held_predictions_generated=np.asarray(False),
    )
    summary = {
        "protocol": "p27-aligned-label-free-multiview-inner-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "temporal_views": views,
        "horizontal_flips": flips,
        "view_count": len(view_logits),
        "selection": "standard fixed 3-bin within-segment views; no label fitting",
        "metrics": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
