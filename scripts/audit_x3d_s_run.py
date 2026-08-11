from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml

from src.data.x3d_clip_dataset import X3DClipDataset, _transform_clip_frames
from src.data.ir_primary_full_sequence_dataset import class_map_hash
from src.inference.x3d_s_ir_context_pipeline import X3DIRContextPipeline
from src.models.expert_contract import (
    ExpertBatchResult,
    ExpertOutput,
    align_expert_batch,
    calibrated_probability_mixture,
)
from src.models.x3d_s_visual_expert import X3DSVisualExpert, build_x3d_s_feature_backbone
from src.roi.pose_locator import UltralyticsPoseLocator
from src.train_x3d_s_visual_expert import prepare_run_directory, train_partition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/x3d_s_ir_context_fold0.yaml"
PARITY_SAMPLE_IDS = (
    "train__c02__user8__7-1-3",
    "train__c22__user1__6-2-2",
    "train__c12__user2__3-1-3",
    "train__c28__user2__5-1-1",
    "train__c23__user18__1-2-1",
    "train__c12__user6__3-1-1",
)


def select_overfit_sample_ids(trials: pd.DataFrame) -> tuple[str, ...]:
    required = {"sample_id", "class_id", "frames"}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"Overfit trial table is missing columns: {sorted(missing)}")
    classes = sorted(trials["class_id"].astype(int).unique())[:8]
    if len(classes) < 8:
        raise ValueError("Overfit selection requires at least eight classes")

    selected: list[str] = []
    for class_position, class_id in enumerate(classes):
        candidates = trials[trials["class_id"].astype(int) == class_id].sort_values(
            ["frames", "sample_id"]
        )
        if len(candidates) < 2:
            raise ValueError(f"Class {class_id} has fewer than two trials")
        first = candidates.iloc[0]
        if class_position == 0:
            multi = candidates[candidates["frames"].astype(int) > 32]
            if multi.empty:
                raise ValueError("Overfit selection requires a multi-clip trial")
            second = multi.iloc[0]
        else:
            second = candidates.iloc[1]
        selected.extend((str(first["sample_id"]), str(second["sample_id"])))
    if len(set(selected)) != 16:
        raise ValueError("Overfit selection did not produce sixteen unique trials")
    return tuple(selected)


class SelectedTrialDataset(Dataset[Mapping[str, object]]):
    def __init__(self, dataset: X3DClipDataset, sample_ids: Sequence[str]) -> None:
        lookup = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
        missing = set(sample_ids) - set(lookup)
        if missing:
            raise ValueError(f"Selected trials are absent from dataset: {sorted(missing)}")
        self.dataset = dataset
        self.indices = tuple(lookup[sample_id] for sample_id in sample_ids)
        self.num_clips = [dataset.num_clips[index] for index in self.indices]
        self.class_names = dataset.class_names
        self.class_map_hash = dataset.class_map_hash

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self.dataset[self.indices[index]]


def _archive_result(path: Path) -> tuple[ExpertBatchResult, dict[str, Any]]:
    with np.load(path) as data:
        values = {name: data[name].copy() for name in data.files}
    result = ExpertBatchResult(
        sample_ids=tuple(values["sample_ids"].astype(str)),
        class_map_hash=str(values["class_map_hash"].item()),
        output=ExpertOutput(
            main_logits=torch.from_numpy(values["logits"]).float(),
            embedding=torch.from_numpy(values["embeddings"]).float(),
            quality=torch.from_numpy(values["quality"]).float(),
            quality_mask=torch.from_numpy(values["quality_mask"]).bool(),
            availability=torch.from_numpy(values["availability"]).bool(),
        ),
    )
    result.validate()
    return result, values


def audit_smoke_run(config: Mapping[str, Any], smoke_directory: Path) -> dict[str, Any]:
    checkpoint_path = smoke_directory / "best_accuracy.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing deployable X3D checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint has no model_state_dict")
    parameter_bytes = sum(
        int(value.numel() * value.element_size())
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    head_parameter_bytes = sum(
        int(value.numel() * value.element_size())
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
        and (str(name).startswith("embedding_head.") or str(name).startswith("classifier."))
    )

    deployment = config.get("deployment_artifacts")
    if not isinstance(deployment, Mapping):
        raise ValueError("Raw IR inference requires deployment_artifacts")
    yolo_value = deployment.get("yolo_checkpoint")
    yolo_path = Path(str(yolo_value)) if yolo_value else None
    if yolo_path is None or not yolo_path.is_file():
        raise FileNotFoundError("Raw IR inference requires YOLO pose weights")
    size_gate = config.get("size_gate")
    if not isinstance(size_gate, Mapping):
        raise ValueError("Config has no size gate")
    size_limit = int(size_gate["internal_limit_bytes"])
    route_bytes = checkpoint_path.stat().st_size + yolo_path.stat().st_size

    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    class_rows = manifest[["class_id", "action_name"]].drop_duplicates()
    expected_hash = class_map_hash(class_rows)
    accuracy_result, accuracy_values = _archive_result(
        smoke_directory / "val_predictions_best_accuracy.npz"
    )
    macro_result, _ = _archive_result(smoke_directory / "val_predictions_best_macro_f1.npz")
    summary = json.loads((smoke_directory / "run_summary.json").read_text(encoding="utf-8"))
    expected_rows = int(summary["val_samples_evaluated_last_epoch"])
    archive_complete = (
        len(accuracy_result.sample_ids) == expected_rows
        and len(set(accuracy_result.sample_ids)) == expected_rows
        and accuracy_result.sample_ids == macro_result.sample_ids
        and accuracy_result.output.main_logits.shape == (expected_rows, 40)
    )
    if accuracy_result.class_map_hash != expected_hash or macro_result.class_map_hash != expected_hash:
        raise ValueError("Prediction archive class_map_hash differs from visual expert map")
    if not archive_complete:
        raise ValueError("Validation prediction archives are incomplete or inconsistent")

    reversed_indices = torch.arange(expected_rows - 1, -1, -1)
    shuffled = ExpertBatchResult(
        sample_ids=tuple(reversed(accuracy_result.sample_ids)),
        class_map_hash=accuracy_result.class_map_hash,
        output=ExpertOutput(
            main_logits=accuracy_result.output.main_logits[reversed_indices],
            embedding=accuracy_result.output.embedding[reversed_indices],
            quality=accuracy_result.output.quality[reversed_indices],
            quality_mask=accuracy_result.output.quality_mask[reversed_indices],
            availability=accuracy_result.output.availability[reversed_indices],
        ),
    )
    alignment = align_expert_batch(accuracy_result, shuffled)
    alignment_passed = torch.equal(alignment, reversed_indices)
    sensor_logits = torch.flip(accuracy_result.output.main_logits, dims=(1,))
    recovered = calibrated_probability_mixture(
        accuracy_result.output.main_logits,
        sensor_logits,
        0.0,
    )
    visual_probabilities = torch.softmax(accuracy_result.output.main_logits, dim=-1)
    alpha_zero_passed = torch.equal(recovered, visual_probabilities)
    finite_archive = bool(np.isfinite(accuracy_values["logits"]).all())
    report = {
        "status": "passed",
        "x3d_checkpoint": str(checkpoint_path.resolve()),
        "x3d_checkpoint_bytes": checkpoint_path.stat().st_size,
        "x3d_plus_custom_head_parameter_bytes": parameter_bytes,
        "custom_head_parameter_bytes": head_parameter_bytes,
        "custom_head_embedded_in_x3d_checkpoint": True,
        "custom_head_counted_twice": False,
        "yolo_required_at_inference": True,
        "yolo_checkpoint": str(yolo_path.resolve()),
        "yolo_checkpoint_bytes": yolo_path.stat().st_size,
        "learned_preprocessing_or_calibration_bytes": 0,
        "deployable_file_count": 2,
        "ir_route_serialized_weight_subtotal": route_bytes,
        "internal_size_limit_bytes": size_limit,
        "ir_route_provisional_size_gate_passed": route_bytes < size_limit,
        "validation_sample_count": expected_rows,
        "validation_archive_complete": archive_complete,
        "validation_logits_finite": finite_archive,
        "class_map_hash": expected_hash,
        "alignment_gate_passed": alignment_passed,
        "alpha_zero_gate_passed": alpha_zero_passed,
        "complete_submission_package_claimed": False,
    }
    if not all(
        (
            report["ir_route_provisional_size_gate_passed"],
            finite_archive,
            alignment_passed,
            alpha_zero_passed,
        )
    ):
        raise RuntimeError("One or more smoke audit gates failed")
    return report


def _latency_bucket(num_frames: int) -> str:
    if num_frames <= 13:
        return "<=13"
    if num_frames <= 32:
        return "14-32"
    if num_frames <= 64:
        return "33-64"
    return ">64"


def crop_parity_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError("Crop parity expects matching [frames,height,width] arrays")
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    mean_absolute_error = float(np.mean(difference))
    p99_absolute_error = float(np.percentile(difference, 99))
    mean_squared_error = float(np.mean(np.square(difference)))
    psnr_db = (
        float("inf")
        if mean_squared_error == 0.0
        else float(10.0 * np.log10((255.0**2) / mean_squared_error))
    )
    worst_frame_mae = float(np.max(np.mean(difference, axis=(1, 2))))
    result = {
        "max_absolute_error": int(np.max(difference)),
        "mae": mean_absolute_error,
        "p99_absolute_error": p99_absolute_error,
        "psnr_db": psnr_db,
        "worst_frame_mae": worst_frame_mae,
    }
    result["gate_passed"] = bool(
        mean_absolute_error <= 1.0
        and p99_absolute_error <= 8.0
        and psnr_db >= 40.0
        and worst_frame_mae <= 2.0
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rebuild_normalized_clips(
    crops: np.ndarray,
    source_indices: torch.Tensor,
) -> torch.Tensor:
    tensors = [
        torch.from_numpy(crop.copy()).unsqueeze(0).to(torch.float32).div_(255.0)
        for crop in crops
    ]
    clips = []
    for window_index, indices in enumerate(source_indices[:, 0]):
        clip = _transform_clip_frames(
            [tensors[int(index)] for index in indices.tolist()],
            training=False,
            generator=torch.Generator().manual_seed(window_index),
        )
        clips.append(clip.unsqueeze(0))
    return torch.stack(clips)


def _model_sensitivity(
    online: ExpertOutput,
    offline: ExpertOutput,
) -> dict[str, Any]:
    online_probability = torch.exp(online.main_logits.float()).clamp_min(1e-12)
    offline_probability = torch.exp(offline.main_logits.float()).clamp_min(1e-12)
    online_probability = online_probability / online_probability.sum(dim=-1, keepdim=True)
    offline_probability = offline_probability / offline_probability.sum(dim=-1, keepdim=True)
    midpoint = 0.5 * (online_probability + offline_probability)
    js_divergence = 0.5 * (
        torch.sum(online_probability * torch.log(online_probability / midpoint), dim=-1)
        + torch.sum(offline_probability * torch.log(offline_probability / midpoint), dim=-1)
    )
    return {
        "embedding_cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                online.embedding.float(), offline.embedding.float(), dim=-1
            ).item()
        ),
        "probability_l1_distance": float(
            torch.sum(torch.abs(online_probability - offline_probability)).item()
        ),
        "jensen_shannon_divergence": float(js_divergence.item()),
        "max_class_probability_delta": float(
            torch.max(torch.abs(online_probability - offline_probability)).item()
        ),
        "top1_agreement": bool(
            torch.argmax(online_probability, dim=-1).item()
            == torch.argmax(offline_probability, dim=-1).item()
        ),
    }


def run_online_parity(
    config: Mapping[str, Any],
    smoke_directory: Path,
) -> dict[str, Any]:
    manifest = pd.read_csv(Path(str(config["input_manifest"])), encoding="utf-8-sig")
    selected_frame = manifest[manifest["sample_id"].astype(str).isin(PARITY_SAMPLE_IDS)].copy()
    if set(selected_frame["sample_id"].astype(str).unique()) != set(PARITY_SAMPLE_IDS):
        raise ValueError("Parity sample set is incomplete")
    if set(selected_frame["split"].astype(str)) != {"train"}:
        raise ValueError("Online parity may use training data only")

    first_export = Path(str(selected_frame.iloc[0]["ir_context_path"]))
    export_root = first_export.parents[4]
    roi_audit_path = export_root / "roi_frame_audit.csv"
    if not roi_audit_path.is_file():
        raise FileNotFoundError(f"Missing exported ROI audit: {roi_audit_path}")
    roi_audit = pd.read_csv(
        roi_audit_path,
        encoding="utf-8-sig",
        usecols=(
            "sample_id",
            "source_frame_index",
            "view_name",
            "x1",
            "y1",
            "x2",
            "y2",
        ),
    )
    roi_audit = roi_audit[
        roi_audit["sample_id"].astype(str).isin(PARITY_SAMPLE_IDS)
        & roi_audit["view_name"].astype(str).eq("ir_context")
    ]

    export_metadata = json.loads((export_root / "export_metadata.json").read_text(encoding="utf-8"))
    pose_cache_path = Path(str(export_metadata["pose_cache"]))
    if not pose_cache_path.is_file():
        raise FileNotFoundError(f"Missing historical pose cache: {pose_cache_path}")
    with np.load(pose_cache_path, allow_pickle=False) as pose_cache:
        cache_ids = pose_cache["sample_ids"].astype(str)
        cache_recovery = pose_cache["person_crop_low_confidence_recovery"].astype(bool)
        expected_recovery = {
            sample_id: bool(np.any(cache_recovery[cache_ids == sample_id]))
            for sample_id in PARITY_SAMPLE_IDS
        }

    checkpoint_path = smoke_directory / "best_accuracy.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = X3DSVisualExpert(
        backbone=build_x3d_s_feature_backbone(pretrained=False),
        num_classes=int(config["num_classes"]),
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
        update_backbone_bn_running_stats=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    deployment = config.get("deployment_artifacts")
    if not isinstance(deployment, Mapping):
        raise ValueError("Parity config has no deployment artifacts")
    locator = UltralyticsPoseLocator(
        Path(str(deployment["yolo_checkpoint"])),
        device=0,
        image_size=640,
        detection_confidence=0.25,
    )
    device = torch.device(str(config["device"]))
    pipeline = X3DIRContextPipeline(pose_locator=locator, model=model, device=device)
    offline_dataset = X3DClipDataset(manifest, split="train", training=True, augmentation_enabled=False)
    offline_lookup = {sample_id: index for index, sample_id in enumerate(offline_dataset.sample_ids)}

    max_box_error = 0.0
    max_crop_error = 0
    max_crop_mae = 0.0
    max_crop_p99 = 0.0
    min_crop_psnr = float("inf")
    max_worst_frame_crop_mae = 0.0
    max_clip_error = 0.0
    recovered = 0
    clip_counts = []
    latency_rows: dict[str, list[dict[str, float]]] = {
        "<=13": [],
        "14-32": [],
        "33-64": [],
        ">64": [],
    }
    frame_order_gate = True
    temporal_gate = True
    recovery_gate = True
    normalization_gate = True
    sample_rows = []
    for sample_id in PARITY_SAMPLE_IDS:
        trial = selected_frame[selected_frame["sample_id"].astype(str) == sample_id].sort_values(
            "source_frame_index"
        )
        trial_indices = trial["source_frame_index"].to_numpy(np.int64)
        frame_order_gate &= bool(np.array_equal(trial_indices, np.arange(len(trial))))
        raw_paths = tuple(Path(path) for path in trial["source_ir_path"].astype(str))
        preprocessed = pipeline.preprocess_trial(raw_paths)
        expected_audit = roi_audit[roi_audit["sample_id"].astype(str) == sample_id].sort_values(
            "source_frame_index"
        )
        expected_audit_indices = expected_audit["source_frame_index"].to_numpy(np.int64)
        frame_order_gate &= bool(np.array_equal(expected_audit_indices, trial_indices))
        expected_boxes = expected_audit[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        if len(expected_boxes) != len(raw_paths):
            raise ValueError(f"ROI audit row count differs for {sample_id}")
        box_error = float(np.max(np.abs(preprocessed.boxes - expected_boxes)))
        offline_crop_rows = []
        for path in trial["ir_context_path"].astype(str):
            offline_crop = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if offline_crop is None:
                raise ValueError(f"Could not decode exported IR context: {path}")
            if offline_crop.ndim == 3 and offline_crop.shape[2] == 1:
                offline_crop = offline_crop[:, :, 0]
            offline_crop_rows.append(offline_crop)
        offline_crops = np.stack(offline_crop_rows)
        crop_metrics = crop_parity_metrics(preprocessed.crops, offline_crops)
        crop_error = int(crop_metrics["max_absolute_error"])
        offline = offline_dataset[offline_lookup[sample_id]]
        temporal_gate &= bool(torch.equal(preprocessed.window_bounds, offline["window_bounds"]))
        temporal_gate &= bool(torch.equal(preprocessed.source_indices, offline["source_indices"]))
        recovery_matches = preprocessed.recovery_used == expected_recovery[sample_id]
        recovery_gate &= recovery_matches
        online_rebuilt = _rebuild_normalized_clips(
            preprocessed.crops, preprocessed.source_indices
        )
        offline_rebuilt = _rebuild_normalized_clips(offline_crops, offline["source_indices"])
        sample_normalization_gate = bool(
            torch.equal(preprocessed.clips, online_rebuilt)
            and torch.equal(offline["clips"], offline_rebuilt)
        )
        normalization_gate &= sample_normalization_gate
        clip_error = float(torch.max(torch.abs(preprocessed.clips - offline["clips"])).item())
        online_prediction = pipeline.predict_preprocessed(preprocessed, num_frames=len(raw_paths))
        offline_preprocessed = replace(preprocessed, crops=offline_crops, clips=offline["clips"])
        offline_prediction = pipeline.predict_preprocessed(
            offline_preprocessed, num_frames=len(raw_paths)
        )
        if online_prediction.expert.main_logits.shape != (1, 40):
            raise ValueError("Online X3D prediction shape is not [1,40]")
        sensitivity = _model_sensitivity(
            online_prediction.expert,
            offline_prediction.expert,
        )
        bucket = _latency_bucket(len(raw_paths))
        latency_rows[bucket].append(
            {
                "pose_roi_preprocessing": preprocessed.pose_roi_seconds,
                "x3d_per_clip": online_prediction.x3d_seconds
                / int(preprocessed.clips.shape[0]),
                "complete_trial": preprocessed.pose_roi_seconds
                + online_prediction.x3d_seconds,
            }
        )
        recovered += int(preprocessed.recovery_used)
        clip_counts.append(int(preprocessed.clips.shape[0]))
        max_box_error = max(max_box_error, box_error)
        max_crop_error = max(max_crop_error, crop_error)
        max_crop_mae = max(max_crop_mae, float(crop_metrics["mae"]))
        max_crop_p99 = max(max_crop_p99, float(crop_metrics["p99_absolute_error"]))
        min_crop_psnr = min(min_crop_psnr, float(crop_metrics["psnr_db"]))
        max_worst_frame_crop_mae = max(
            max_worst_frame_crop_mae, float(crop_metrics["worst_frame_mae"])
        )
        max_clip_error = max(max_clip_error, clip_error)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "num_frames": len(raw_paths),
                "num_clips": int(preprocessed.clips.shape[0]),
                "recovery_used": preprocessed.recovery_used,
                "expected_recovery_used": expected_recovery[sample_id],
                "person_selection_recovery_path_matches": recovery_matches,
                "box_max_abs_error": box_error,
                "crop_max_abs_error": crop_error,
                "crop_mae": crop_metrics["mae"],
                "crop_p99_absolute_error": crop_metrics["p99_absolute_error"],
                "crop_psnr_db": crop_metrics["psnr_db"],
                "worst_frame_crop_mae": crop_metrics["worst_frame_mae"],
                "crop_gate_passed": crop_metrics["gate_passed"],
                "normalized_clip_max_abs_error": clip_error,
                "normalization_algorithm_matches": sample_normalization_gate,
                "model_sensitivity": sensitivity,
            }
        )

    latency = {
        bucket: {
            "sample_count": len(rows),
            "pose_roi_preprocessing_mean": float(
                np.mean([row["pose_roi_preprocessing"] for row in rows])
            ),
            "x3d_per_clip_mean": float(np.mean([row["x3d_per_clip"] for row in rows])),
            "complete_trial_mean": float(np.mean([row["complete_trial"] for row in rows])),
        }
        for bucket, rows in latency_rows.items()
        if rows
    }

    clip_count_gate = clip_counts == [1, 1, 2, 4, 8, 8]
    box_gate = max_box_error <= 1.0
    crop_gate = bool(
        max_crop_mae <= 1.0
        and max_crop_p99 <= 8.0
        and min_crop_psnr >= 40.0
        and max_worst_frame_crop_mae <= 2.0
    )
    input_parity_gate = bool(
        frame_order_gate
        and temporal_gate
        and recovery_gate
        and normalization_gate
        and clip_count_gate
        and box_gate
        and crop_gate
    )

    report = {
        "status": "passed" if input_parity_gate else "failed",
        "sample_count": len(sample_rows),
        "training_data_only": True,
        "sample_rows": sample_rows,
        "clip_counts": clip_counts,
        "recovered_sample_count": recovered,
        "max_box_absolute_error": max_box_error,
        "max_crop_absolute_error": max_crop_error,
        "max_crop_mae": max_crop_mae,
        "max_crop_p99_absolute_error": max_crop_p99,
        "min_crop_psnr_db": min_crop_psnr,
        "max_worst_frame_crop_mae": max_worst_frame_crop_mae,
        "max_normalized_clip_absolute_error": max_clip_error,
        "latency_seconds_by_length_bucket": latency,
        "x3d_checkpoint": str(checkpoint_path.resolve()),
        "x3d_checkpoint_sha256": _sha256(checkpoint_path),
        "model_sensitivity_thresholds_applied": False,
        "missing_weights_fails": True,
        "silent_full_frame_fallback_forbidden": True,
        "frame_order_gate_passed": frame_order_gate,
        "ordering_gate_passed": frame_order_gate,
        "temporal_gate_passed": temporal_gate,
        "person_selection_recovery_gate_passed": recovery_gate,
        "clip_count_gate_passed": clip_count_gate,
        "normalization_contract": "shared_x3d_clip_transform",
        "normalization_gate_passed": normalization_gate,
        "box_gate_passed": box_gate,
        "crop_gate_passed": crop_gate,
        "input_parity_gate_passed": input_parity_gate,
    }
    if (
        not input_parity_gate
        or recovered < 1
        or set(latency) != {"<=13", "14-32", "33-64", ">64"}
    ):
        raise RuntimeError(f"Online parity gate failed: {json.dumps(report, indent=2)}")
    return report


def run_overfit(config: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    resolved["training"] = {"epochs": 20, "warmup_epochs": 2, "patience": 20}
    resolved["optimizer"] = {
        **resolved["optimizer"],
        "backbone_lr": 3e-4,
        "head_lr": 3e-3,
    }
    resolved["dropout"] = 0.0
    frame = pd.read_csv(Path(str(resolved["input_manifest"])), encoding="utf-8-sig")
    train_frame = frame[frame["split"].astype(str) == "train"]
    trials = (
        train_frame.groupby(["sample_id", "class_id"], as_index=False)
        .size()
        .rename(columns={"size": "frames"})
    )
    sample_ids = select_overfit_sample_ids(trials)
    base_dataset = X3DClipDataset(
        frame,
        split="train",
        training=True,
        augmentation_enabled=False,
        seed=int(resolved["seed"]),
    )
    selected = SelectedTrialDataset(base_dataset, sample_ids)
    device = torch.device(str(resolved["device"]))
    model = X3DSVisualExpert(
        backbone=build_x3d_s_feature_backbone(pretrained=bool(resolved["pretrained"])),
        num_classes=int(resolved["num_classes"]),
        embedding_dim=int(resolved["embedding_dim"]),
        dropout=float(resolved["dropout"]),
        update_backbone_bn_running_stats=False,
    )
    run_directory = prepare_run_directory(Path(str(resolved["output_root"])), run_id)
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (run_directory / "overfit_sample_ids.json").write_text(
        json.dumps(list(sample_ids), indent=2) + "\n", encoding="utf-8"
    )
    summary = train_partition(
        model=model,
        train_dataset=selected,
        validation_dataset=selected,
        config=resolved,
        run_directory=run_directory,
        device=device,
        max_train_batches=None,
        max_val_batches=None,
    )
    history = pd.read_csv(run_directory / "history.csv")
    first_loss = float(history.iloc[0]["train_loss"])
    final_loss = float(history.iloc[-1]["train_loss"])
    final_accuracy = float(history.iloc[-1]["train_accuracy"])
    summary.update(
        {
            "role": "implementation_overfit",
            "augmentation_enabled": False,
            "selected_sample_count": len(sample_ids),
            "selected_class_count": 8,
            "contains_one_clip_trial": any(count == 1 for count in selected.num_clips),
            "contains_multi_clip_trial": any(count > 1 for count in selected.num_clips),
            "first_train_loss": first_loss,
            "final_train_loss": final_loss,
            "final_train_accuracy": final_accuracy,
            "loss_fell": final_loss < first_loss,
            "accuracy_gate_passed": final_accuracy > 0.80,
            "scientific_evidence": False,
        }
    )
    (run_directory / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not summary["loss_fell"] or not summary["accuracy_gate_passed"]:
        raise RuntimeError(
            f"Overfit gate failed: loss {first_loss:.4f}->{final_loss:.4f}, "
            f"accuracy={final_accuracy:.4f}"
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an adaptive multi-clip X3D-S run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-overfit", action="store_true")
    parser.add_argument("--run-online-parity", action="store_true")
    parser.add_argument("--run-id", default="x3d_s_ir_context_overfit16")
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Config root must be a mapping")
    if args.run_overfit:
        summary = run_overfit(config, run_id=args.run_id)
        print(json.dumps(summary, indent=2))
        return
    if args.run_online_parity:
        if args.smoke_run is None:
            raise ValueError("--run-online-parity requires --smoke-run")
        report = run_online_parity(config, args.smoke_run)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if args.smoke_run is not None:
        report = audit_smoke_run(config, args.smoke_run)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    raise ValueError("Select an audit operation")


if __name__ == "__main__":
    main()
