from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel
from imu_data import DEVICES, read_index
from imu_model import DeviceAwareIMUStudent
from run_imu_stat_baseline import feature_vector
from thermal_baseline.thermal_oof_data import ThermalOOFDataset
from thermal_baseline.thermal_tsm_model import ThermalResNetTSM


MODALITY_ORDER = ("skeleton", "depth", "thermal", "imu")
EMBEDDING_DIMS = {
    "skeleton": 512,
    "depth": 1024,
    "thermal": 1024,
    "imu": 320,
}
FEATURES_PER_DEVICE = 48
MASKS_PER_DEVICE = 2
NUM_CLASSES = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build three fixed-final, outer-fold-conditioned pooled-feature "
            "caches. This script performs inference only and never trains."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "p22_joint_pooled_fusion.json",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=PROJECT_DIR / "configs" / "p22_artifact_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p22_joint_pooled_fusion" / "cache",
    )
    parser.add_argument(
        "--equivalence-samples",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Build one fold only. Omit to build all three folds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only P22 cache files in output-dir; never touches existing OOF.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def resolve_repo_path(path: str) -> Path:
    return (REPO_ROOT / path).resolve()


def verify_file_identity(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in expected and path.stat().st_size != int(expected["bytes"]):
        raise ValueError(f"Size mismatch: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected["sha256"]:
        raise ValueError(
            f"SHA256 mismatch: {path}\nexpected={expected['sha256']}\nactual={actual_hash}"
        )


def verify_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    for fold_info in manifest["folds"]:
        for modality, checkpoint in fold_info["checkpoints"].items():
            path = resolve_repo_path(checkpoint["path"])
            if "best_accuracy" in path.name:
                raise ValueError(f"Forbidden best_accuracy checkpoint: {path}")
            verify_file_identity(path, checkpoint)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload["epoch"]) != int(checkpoint["epoch"]):
                raise ValueError(f"Epoch mismatch: {path}")
            if modality == "imu" and not bool(payload.get("fixed_final_epoch")):
                raise ValueError(f"Tiny IMU is not fixed-final: {path}")
            checked.append(
                {
                    "fold": int(fold_info["fold"]),
                    "modality": modality,
                    "path": str(path),
                    "sha256": checkpoint["sha256"],
                    "epoch": int(payload["epoch"]),
                }
            )
    for name, file_info in manifest["manifests"].items():
        path = resolve_repo_path(file_info["path"])
        verify_file_identity(path, file_info)
        checked.append(
            {
                "asset": name,
                "path": str(path),
                "sha256": file_info["sha256"],
            }
        )
    return {"checked_assets": checked, "checked_count": len(checked)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_metadata() -> dict[str, np.ndarray]:
    reference_path = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
    with np.load(reference_path, allow_pickle=False) as reference:
        sample_ids = reference["sample_ids"].astype(str)
        labels = reference["labels"].astype(np.int64)
        folds = reference["folds"].astype(np.int64)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("p12_complete_oof sample_ids are not unique")

    rows = read_csv_rows(PROJECT_DIR / "data" / "manifest.csv")
    lookup = {row["sample_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Visual manifest sample_ids are not unique")
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise KeyError(f"Visual manifest misses {len(missing)} P12 ids: {missing[0]}")

    subjects = np.asarray([lookup[sample_id]["user_id"] for sample_id in sample_ids])
    manifest_labels = np.asarray(
        [int(lookup[sample_id]["class_id"]) for sample_id in sample_ids],
        dtype=np.int64,
    )
    if not np.array_equal(labels, manifest_labels):
        raise ValueError("P12 labels disagree with visual manifest")

    fold_summary = load_json(PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json")
    subject_to_fold: dict[str, int] = {}
    for fold_info in fold_summary["folds"]:
        fold = int(fold_info["fold"])
        for subject in fold_info["val_users"]:
            if subject in subject_to_fold:
                raise ValueError(f"Subject occurs in multiple held folds: {subject}")
            subject_to_fold[subject] = fold
    expected_folds = np.asarray(
        [subject_to_fold[subject] for subject in subjects],
        dtype=np.int64,
    )
    if not np.array_equal(folds, expected_folds):
        raise ValueError("P12 folds disagree with subject fold summary")
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "subjects": subjects,
        "folds": folds,
    }


def state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def state_unchanged(
    before: dict[str, torch.Tensor], model: torch.nn.Module
) -> bool:
    after = model.state_dict()
    return all(torch.equal(before[name], after[name].detach().cpu()) for name in before)


def update_equivalence(
    result: dict[str, Any],
    original_logits: torch.Tensor,
    reconstructed_logits: torch.Tensor,
    fp16_reloaded_logits: torch.Tensor,
) -> None:
    original = original_logits.detach().float()
    reconstructed = reconstructed_logits.detach().float()
    fp16_reloaded = fp16_reloaded_logits.detach().float()
    result["fp32_max_abs_error"] = max(
        float(result["fp32_max_abs_error"]),
        float((original - reconstructed).abs().max().item()),
    )
    result["fp16_max_abs_error"] = max(
        float(result["fp16_max_abs_error"]),
        float((original - fp16_reloaded).abs().max().item()),
    )
    result["argmax_total"] += int(len(original))
    result["argmax_matches"] += int(
        (original.argmax(1) == fp16_reloaded.argmax(1)).sum().item()
    )


def empty_equivalence(fold: int, modality: str) -> dict[str, Any]:
    return {
        "fold": fold,
        "modality": modality,
        "real_samples": 0,
        "fp32_max_abs_error": 0.0,
        "fp16_max_abs_error": 0.0,
        "argmax_matches": 0,
        "argmax_total": 0,
        "argmax_consistency": 0.0,
        "eval_mode": True,
        "inference_mode": True,
        "state_unchanged": False,
    }


def load_aligned_model(
    modality: str, checkpoint_path: Path, device: torch.device
) -> AlignedMultimodalModel:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = AlignedMultimodalModel(
        [modality],
        num_classes=NUM_CLASSES,
        dropout=float(config["dropout"]),
        imagenet_pretrained=False,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def make_visual_loader(
    manifest_path: Path,
    split: str,
    config: dict[str, Any],
    device: torch.device,
) -> DataLoader:
    cache_config = config["cache"]
    dataset = AlignedMultimodalDataset(
        manifest_path=manifest_path,
        split=split,
        modalities=["depth", "skeleton"],
        num_frames=12,
        image_height=144,
        image_width=192,
        augment=False,
        cache_dir=PROJECT_DIR / "cache" / "aligned_192x144",
        skeleton_strategy="first",
        depth_representation="jet_rgb",
        visual_normalization="legacy",
        skeleton_representation="frame_joint",
    )
    return DataLoader(
        dataset,
        batch_size=int(cache_config["visual_batch_size"]),
        shuffle=False,
        num_workers=int(cache_config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def fill_visual(
    fold: int,
    checkpoint_info: dict[str, Any],
    config: dict[str, Any],
    canonical: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    device: torch.device,
    equivalence_samples: int,
) -> list[dict[str, Any]]:
    skeleton_path = resolve_repo_path(checkpoint_info["skeleton"]["path"])
    depth_path = resolve_repo_path(checkpoint_info["depth"]["path"])
    skeleton_model = load_aligned_model("skeleton", skeleton_path, device)
    depth_model = load_aligned_model("depth", depth_path, device)
    skeleton_before = state_snapshot(skeleton_model)
    depth_before = state_snapshot(depth_model)
    lookup = {
        sample_id: index
        for index, sample_id in enumerate(canonical["sample_ids"].tolist())
    }
    seen: set[str] = set()
    skeleton_eq = empty_equivalence(fold, "skeleton")
    depth_eq = empty_equivalence(fold, "depth")
    fold_manifest = PROJECT_DIR / "data" / "subject_folds" / f"fold_{fold}.csv"

    with torch.inference_mode():
        for split in ("train", "val"):
            loader = make_visual_loader(fold_manifest, split, config, device)
            for batch_index, batch in enumerate(loader):
                sample_ids = [str(value) for value in batch["sample_id"]]
                indices = np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)
                batch_labels = batch["label"].numpy().astype(np.int64)
                if not np.array_equal(batch_labels, canonical["labels"][indices]):
                    raise ValueError(f"Visual label mismatch in fold {fold}")
                overlap = seen.intersection(sample_ids)
                if overlap:
                    raise ValueError(f"Duplicate visual sample in fold {fold}: {next(iter(overlap))}")
                seen.update(sample_ids)

                skeleton = batch["skeleton"].to(device, non_blocking=True)
                depth = batch["depth"].to(device, non_blocking=True)
                skeleton_embedding = skeleton_model.skeleton.encode_pooled(skeleton)
                depth_embedding = depth_model.visual.encode_pooled({"depth": depth})
                skeleton_logits = skeleton_model.classifier(skeleton_embedding)
                depth_logits = depth_model.classifier(depth_embedding)

                arrays["skeleton_embedding"][indices] = (
                    skeleton_embedding.detach().cpu().numpy().astype(np.float16)
                )
                arrays["depth_embedding"][indices] = (
                    depth_embedding.detach().cpu().numpy().astype(np.float16)
                )
                arrays["per_modality_logits"][indices, 0] = (
                    skeleton_logits.detach().cpu().numpy().astype(np.float16)
                )
                arrays["per_modality_logits"][indices, 1] = (
                    depth_logits.detach().cpu().numpy().astype(np.float16)
                )

                for model, inputs, embedding, logits, result in (
                    (
                        skeleton_model,
                        {"skeleton": skeleton},
                        skeleton_embedding,
                        skeleton_logits,
                        skeleton_eq,
                    ),
                    (
                        depth_model,
                        {"depth": depth},
                        depth_embedding,
                        depth_logits,
                        depth_eq,
                    ),
                ):
                    remaining = equivalence_samples - int(result["real_samples"])
                    if remaining <= 0:
                        continue
                    count = min(remaining, len(sample_ids))
                    sliced_inputs = {key: value[:count] for key, value in inputs.items()}
                    original = model(sliced_inputs)
                    if result["modality"] == "skeleton":
                        pooled = model.skeleton.encode_pooled(sliced_inputs["skeleton"])
                    else:
                        pooled = model.visual.encode_pooled(sliced_inputs)
                    reconstructed = model.classifier(pooled)
                    fp16_reloaded = model.classifier(pooled.half().float())
                    update_equivalence(result, original, reconstructed, fp16_reloaded)
                    result["real_samples"] += count
                if batch_index % 25 == 0:
                    print(
                        f"fold={fold} visual split={split} batch={batch_index} "
                        f"seen={len(seen)}",
                        flush=True,
                    )

    expected = set(canonical["sample_ids"].tolist())
    if seen != expected:
        raise ValueError(
            f"Visual coverage mismatch fold={fold}: seen={len(seen)} expected={len(expected)}"
        )
    arrays["presence"][:, 0:2] = 1
    skeleton_eq["state_unchanged"] = state_unchanged(skeleton_before, skeleton_model)
    depth_eq["state_unchanged"] = state_unchanged(depth_before, depth_model)
    for result, model in ((skeleton_eq, skeleton_model), (depth_eq, depth_model)):
        result["eval_mode"] = not model.training
        result["argmax_consistency"] = (
            float(result["argmax_matches"]) / max(int(result["argmax_total"]), 1)
        )
    del skeleton_model, depth_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return [skeleton_eq, depth_eq]


def make_thermal_loader(
    manifest_path: Path,
    split: str,
    config: dict[str, Any],
    device: torch.device,
    allowed_ids: set[str],
) -> DataLoader:
    cache_config = config["cache"]
    dataset = ThermalOOFDataset(
        manifest_path=manifest_path,
        split=split,
        num_frames=12,
        image_height=144,
        image_width=192,
        augment=False,
        normalization="legacy",
    )
    dataset.samples = [
        sample for sample in dataset.samples if sample.sample_id in allowed_ids
    ]
    return DataLoader(
        dataset,
        batch_size=int(cache_config["thermal_batch_size"]),
        shuffle=False,
        num_workers=int(cache_config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def fill_thermal(
    fold: int,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    canonical: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    device: torch.device,
    equivalence_samples: int,
) -> dict[str, Any]:
    path = resolve_repo_path(checkpoint["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ThermalResNetTSM(
        num_classes=NUM_CLASSES,
        dropout=float(payload["config"]["dropout"]),
        imagenet_pretrained=False,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    before = state_snapshot(model)
    result = empty_equivalence(fold, "thermal")
    lookup = {
        sample_id: index
        for index, sample_id in enumerate(canonical["sample_ids"].tolist())
    }
    allowed_ids = set(lookup)
    seen: set[str] = set()
    manifest_path = (
        REPO_ROOT
        / "thermal_baseline"
        / "data"
        / "subject_folds"
        / f"fold_{fold}.csv"
    )

    with torch.inference_mode():
        for split in ("train", "val"):
            loader = make_thermal_loader(
                manifest_path, split, config, device, allowed_ids
            )
            for batch_index, batch in enumerate(loader):
                sample_ids = [str(value) for value in batch["sample_id"]]
                indices = np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)
                batch_labels = batch["label"].numpy().astype(np.int64)
                if not np.array_equal(batch_labels, canonical["labels"][indices]):
                    raise ValueError(f"Thermal label mismatch in fold {fold}")
                overlap = seen.intersection(sample_ids)
                if overlap:
                    raise ValueError(f"Duplicate Thermal sample in fold {fold}")
                seen.update(sample_ids)
                clip = batch["clip"].to(device, non_blocking=True)
                pooled = model.encode_pooled(clip)
                logits = model.classifier(pooled)
                arrays["thermal_embedding"][indices] = (
                    pooled.detach().cpu().numpy().astype(np.float16)
                )
                arrays["per_modality_logits"][indices, 2] = (
                    logits.detach().cpu().numpy().astype(np.float16)
                )
                arrays["presence"][indices, 2] = 1

                remaining = equivalence_samples - int(result["real_samples"])
                if remaining > 0:
                    count = min(remaining, len(sample_ids))
                    original = model(clip[:count])
                    small_pooled = model.encode_pooled(clip[:count])
                    reconstructed = model.classifier(small_pooled)
                    fp16_reloaded = model.classifier(small_pooled.half().float())
                    update_equivalence(
                        result, original, reconstructed, fp16_reloaded
                    )
                    result["real_samples"] += count
                if batch_index % 25 == 0:
                    print(
                        f"fold={fold} thermal split={split} batch={batch_index} "
                        f"seen={len(seen)}",
                        flush=True,
                    )
    if len(seen) != 2776:
        raise ValueError(f"Thermal/P12 intersection changed: {len(seen)} != 2776")
    result["state_unchanged"] = state_unchanged(before, model)
    result["eval_mode"] = not model.training
    result["argmax_consistency"] = (
        float(result["argmax_matches"]) / max(int(result["argmax_total"]), 1)
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_imu_statistics(
    canonical: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = PROJECT_DIR / "cache" / "imu_32"
    rows = [
        row for row in read_index(cache / "index.csv")
        if row.split == "train" and row.usable
    ]
    values = np.load(cache / "imu_float32.npy", allow_pickle=False)
    time_mask = np.load(cache / "time_mask_uint8.npy", allow_pickle=False)
    device_mask = np.load(cache / "device_mask_uint8.npy", allow_pickle=False)
    lookup = {
        sample_id: index
        for index, sample_id in enumerate(canonical["sample_ids"].tolist())
    }
    features = np.zeros(
        (len(lookup), len(DEVICES), FEATURES_PER_DEVICE), dtype=np.float32
    )
    masks = np.zeros(
        (len(lookup), len(DEVICES), MASKS_PER_DEVICE), dtype=np.float32
    )
    present = np.zeros(len(lookup), dtype=np.uint8)
    seen: set[str] = set()
    for row in rows:
        if row.sample_id not in lookup:
            continue
        if row.sample_id in seen:
            raise ValueError(f"Duplicate parsed IMU sample: {row.sample_id}")
        index = lookup[row.sample_id]
        if int(row.class_id) != int(canonical["labels"][index]):
            raise ValueError(f"IMU label mismatch: {row.sample_id}")
        if str(row.user_id) != str(canonical["subjects"][index]):
            raise ValueError(f"IMU subject mismatch: {row.sample_id}")
        feature, mask = feature_vector(
            values[row.cache_index],
            time_mask[row.cache_index],
            device_mask[row.cache_index],
        )
        features[index] = feature.reshape(len(DEVICES), FEATURES_PER_DEVICE)
        masks[index] = mask.reshape(len(DEVICES), MASKS_PER_DEVICE)
        present[index] = 1
        seen.add(row.sample_id)
    if len(seen) != 2855:
        raise ValueError(f"IMU/P12 parser intersection changed: {len(seen)} != 2855")
    return features, masks, present


def fill_imu(
    fold: int,
    checkpoint: dict[str, Any],
    canonical: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    raw_features: np.ndarray,
    raw_masks: np.ndarray,
    parsed_presence: np.ndarray,
    batch_size: int,
    device: torch.device,
    equivalence_samples: int,
) -> dict[str, Any]:
    path = resolve_repo_path(checkpoint["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DeviceAwareIMUStudent(
        dropout=float(payload["config"]["dropout"])
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    before = state_snapshot(model)

    mean = np.asarray(payload["normalizer_mean"], dtype=np.float32)
    std = np.asarray(payload["normalizer_std"], dtype=np.float32)
    normalized = (raw_features - mean[None]) / std[None]
    normalized *= (raw_masks[..., :1] > 0).astype(np.float32)
    inputs = np.concatenate([normalized, raw_masks], axis=2).astype(np.float32)
    present_indices = np.flatnonzero(parsed_presence > 0)
    result = empty_equivalence(fold, "imu")

    with torch.inference_mode():
        for start in range(0, len(present_indices), batch_size):
            indices = present_indices[start : start + batch_size]
            batch = torch.from_numpy(inputs[indices]).to(device)
            pooled = model.encode_pooled(batch)
            logits = model.classifier(pooled)
            arrays["imu_embedding"][indices] = (
                pooled.detach().cpu().numpy().astype(np.float16)
            )
            arrays["per_modality_logits"][indices, 3] = (
                logits.detach().cpu().numpy().astype(np.float16)
            )
            arrays["presence"][indices, 3] = 1
            remaining = equivalence_samples - int(result["real_samples"])
            if remaining > 0:
                count = min(remaining, len(indices))
                small = batch[:count]
                original = model(small)
                small_pooled = model.encode_pooled(small)
                reconstructed = model.classifier(small_pooled)
                fp16_reloaded = model.classifier(small_pooled.half().float())
                update_equivalence(
                    result, original, reconstructed, fp16_reloaded
                )
                result["real_samples"] += count
    result["state_unchanged"] = state_unchanged(before, model)
    result["eval_mode"] = not model.training
    result["argmax_consistency"] = (
        float(result["argmax_matches"]) / max(int(result["argmax_total"]), 1)
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def new_cache_arrays(samples: int) -> dict[str, np.ndarray]:
    return {
        "presence": np.zeros((samples, len(MODALITY_ORDER)), dtype=np.uint8),
        "skeleton_embedding": np.zeros(
            (samples, EMBEDDING_DIMS["skeleton"]), dtype=np.float16
        ),
        "depth_embedding": np.zeros(
            (samples, EMBEDDING_DIMS["depth"]), dtype=np.float16
        ),
        "thermal_embedding": np.zeros(
            (samples, EMBEDDING_DIMS["thermal"]), dtype=np.float16
        ),
        "imu_embedding": np.zeros(
            (samples, EMBEDDING_DIMS["imu"]), dtype=np.float16
        ),
        "per_modality_logits": np.zeros(
            (samples, len(MODALITY_ORDER), NUM_CLASSES), dtype=np.float16
        ),
    }


def validate_equivalence(result: dict[str, Any]) -> None:
    if int(result["real_samples"]) < 1:
        raise ValueError(f"No real equivalence samples: {result}")
    if float(result["fp32_max_abs_error"]) > 1e-6:
        raise ValueError(f"FP32 equivalence failed: {result}")
    if float(result["fp16_max_abs_error"]) > 1e-2:
        raise ValueError(f"FP16 equivalence failed: {result}")
    if float(result["argmax_consistency"]) < 0.9999:
        raise ValueError(f"FP16 argmax consistency failed: {result}")
    if not bool(result["eval_mode"]):
        raise ValueError(f"Model not in eval mode: {result}")
    if not bool(result["state_unchanged"]):
        raise ValueError(f"Parameters or buffers changed: {result}")


def cache_metadata_arrays(
    canonical: dict[str, np.ndarray],
    fold: int,
    arrays: dict[str, np.ndarray],
    checkpoint_info: dict[str, Any],
    artifact_manifest: dict[str, Any],
    config: dict[str, Any],
    commit: str,
) -> dict[str, np.ndarray]:
    checkpoint_hashes = {
        modality: checkpoint_info[modality]["sha256"]
        for modality in MODALITY_ORDER
    }
    manifest_hashes = {
        name: value["sha256"]
        for name, value in artifact_manifest["manifests"].items()
    }
    return {
        "sample_ids": canonical["sample_ids"],
        "labels": canonical["labels"],
        "subjects": canonical["subjects"],
        "folds": canonical["folds"],
        "outer_fold": np.asarray(fold, dtype=np.int64),
        "is_outer_train": (canonical["folds"] != fold).astype(np.uint8),
        **arrays,
        "modality_order": np.asarray(MODALITY_ORDER),
        "checkpoint_sha256": np.asarray(
            json.dumps(checkpoint_hashes, sort_keys=True)
        ),
        "manifest_sha256": np.asarray(
            json.dumps(manifest_hashes, sort_keys=True)
        ),
        "preprocess_config": np.asarray(
            json.dumps(artifact_manifest["preprocess"], sort_keys=True)
        ),
        "fixed_config": np.asarray(json.dumps(config, sort_keys=True)),
        "code_commit": np.asarray(commit),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    config = load_json(args.config)
    artifact_manifest = load_json(args.artifact_manifest)
    if config["protocol_version"] != "p22-a-v1-fixed-before-training":
        raise ValueError("Unexpected P22 protocol version")
    print("preflight: verifying checkpoint and manifest identities", flush=True)
    artifact_audit = verify_artifacts(artifact_manifest)
    print(
        f"preflight: verified {artifact_audit['checked_count']} assets",
        flush=True,
    )
    canonical = canonical_metadata()
    print(
        f"preflight: canonical sample identity passed ({len(canonical['sample_ids'])})",
        flush=True,
    )
    commit = current_commit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P22 cache extraction requires CUDA on this node")
    torch.manual_seed(int(config["training"]["seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [args.fold] if args.fold is not None else [0, 1, 2]
    print("preflight: loading parsed IMU statistics", flush=True)
    raw_imu_features, raw_imu_masks, parsed_imu_presence = load_imu_statistics(
        canonical
    )
    print(
        f"preflight: parsed IMU/P12 intersection={int(parsed_imu_presence.sum())}",
        flush=True,
    )
    fold_summaries: list[dict[str, Any]] = []
    all_equivalence: list[dict[str, Any]] = []

    for fold in folds:
        assert fold is not None
        fold_started = time.time()
        output_path = output_dir / f"fold_{fold}_cache.npz"
        if output_path.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite P22 cache without --force: {output_path}"
            )
        fold_info = next(
            item for item in artifact_manifest["folds"]
            if int(item["fold"]) == int(fold)
        )
        print(f"preflight: starting fold={fold} visual extraction", flush=True)
        checkpoint_info = fold_info["checkpoints"]
        arrays = new_cache_arrays(len(canonical["sample_ids"]))
        equivalence = fill_visual(
            fold,
            checkpoint_info,
            config,
            canonical,
            arrays,
            device,
            int(args.equivalence_samples),
        )
        equivalence.append(
            fill_thermal(
                fold,
                checkpoint_info["thermal"],
                config,
                canonical,
                arrays,
                device,
                int(args.equivalence_samples),
            )
        )
        equivalence.append(
            fill_imu(
                fold,
                checkpoint_info["imu"],
                canonical,
                arrays,
                raw_imu_features,
                raw_imu_masks,
                parsed_imu_presence,
                int(config["cache"]["imu_batch_size"]),
                device,
                int(args.equivalence_samples),
            )
        )
        for result in equivalence:
            validate_equivalence(result)
        all_equivalence.extend(equivalence)

        presence_counts = {
            modality: int(arrays["presence"][:, index].sum())
            for index, modality in enumerate(MODALITY_ORDER)
        }
        expected_counts = {
            "skeleton": 2914,
            "depth": 2914,
            "thermal": 2776,
            "imu": 2855,
        }
        if presence_counts != expected_counts:
            raise ValueError(
                f"Presence counts changed fold={fold}: {presence_counts}"
            )
        for index, modality in enumerate(MODALITY_ORDER):
            missing = arrays["presence"][:, index] == 0
            embedding = arrays[f"{modality}_embedding"]
            if not np.all(embedding[missing] == 0):
                raise ValueError(f"Missing {modality} embeddings are not zero")
            if not np.all(arrays["per_modality_logits"][missing, index] == 0):
                raise ValueError(f"Missing {modality} logits are not zero")

        payload = cache_metadata_arrays(
            canonical,
            fold,
            arrays,
            checkpoint_info,
            artifact_manifest,
            config,
            commit,
        )
        np.savez_compressed(output_path, **payload)
        with np.load(output_path, allow_pickle=False) as saved:
            if saved["sample_ids"].shape != (2914,):
                raise ValueError(f"Saved cache shape invalid: {output_path}")
            if str(saved["code_commit"].item()) != commit:
                raise ValueError(f"Saved cache commit mismatch: {output_path}")
        fold_summary = {
            "fold": fold,
            "cache_path": str(output_path),
            "cache_bytes": output_path.stat().st_size,
            "cache_sha256": sha256_file(output_path),
            "samples": 2914,
            "outer_train": int((canonical["folds"] != fold).sum()),
            "held_fold": int((canonical["folds"] == fold).sum()),
            "presence_counts": presence_counts,
            "equivalence": equivalence,
            "elapsed_seconds": round(time.time() - fold_started, 3),
        }
        fold_summaries.append(fold_summary)
        print(json.dumps(fold_summary, ensure_ascii=False, indent=2), flush=True)

    preflight = {
        "status": "passed",
        "training_started": False,
        "protocol": config["protocol_version"],
        "code_commit": commit,
        "device": str(device),
        "samples": len(canonical["sample_ids"]),
        "folds_built": folds,
        "artifact_audit": artifact_audit,
        "folds": fold_summaries,
        "equivalence": all_equivalence,
        "thresholds": {
            "fp32_max_abs_error": 1e-6,
            "fp16_max_abs_error": 1e-2,
            "fp16_argmax_consistency_min": 0.9999,
        },
        "total_cache_bytes": sum(
            int(summary["cache_bytes"]) for summary in fold_summaries
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
