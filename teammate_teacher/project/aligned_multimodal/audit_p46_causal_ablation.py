from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from p46_event_data import P46EventDataset, collate_p46_events
from p46_protocol import HARD_CLASS_IDS
from p46_step10_model import P46Step10Model
from train_p46_step10 import FrameBudgetBatchSampler, move_batch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_DIR / "runs" / "p46_step10_detail21_fullcoverage_v2"
DEFAULT_OUTPUT = DEFAULT_RUN / "causal_ablation_v1"
SEED = 20260807


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-training causal audit for P46 Step10")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--frame-budget", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--maximum-batches", type=int, default=0)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clone_with_zeros(batch: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    output = dict(batch)
    for key in keys:
        value = batch[key]
        output[key] = torch.zeros_like(value)
    return output


SKELETON_VALUE_KEYS = (
    "skeleton_features",
    "skeleton_relations",
    "skeleton_frame_quality",
    "body_axes_camera",
)
SKELETON_MASK_KEYS = (
    "skeleton_feature_mask",
    "skeleton_joint_mask",
    "skeleton_relation_mask",
    "body_axes_raw_valid",
)
IMU_VALUE_KEYS = (
    "imu_values",
    "imu_raw_vectors",
    "imu_time_seconds",
    "imu_interval_counts",
)
IMU_MASK_KEYS = ("imu_point_mask", "imu_device_mask")
LOCAL_VISUAL_VALUE_KEYS = (
    "arm_spatial_features",
    "detail_spatial_features",
    "local_geometry_features",
    "oriented_roi_geometry",
    "local_roi_quality",
    "local_roi_source",
    "local_roi_clipped_ratio",
    "pose_quality_factor",
)
LOCAL_VISUAL_MASK_KEYS = ("oriented_angle_valid", "local_roi_valid")
CONTEXT_VALUE_KEYS = ("context_features", "context_quality")
CONTEXT_MASK_KEYS = ("context_valid",)


def remove_skeleton(batch: dict[str, Any]) -> dict[str, Any]:
    return clone_with_zeros(batch, SKELETON_VALUE_KEYS + SKELETON_MASK_KEYS)


def remove_imu(batch: dict[str, Any]) -> dict[str, Any]:
    output = clone_with_zeros(batch, IMU_VALUE_KEYS + IMU_MASK_KEYS)
    output["imu_frame_index"] = torch.full_like(batch["imu_frame_index"], -1)
    return output


def remove_motion(batch: dict[str, Any]) -> dict[str, Any]:
    return remove_imu(remove_skeleton(batch))


def remove_local_visual(batch: dict[str, Any]) -> dict[str, Any]:
    return clone_with_zeros(
        batch, LOCAL_VISUAL_VALUE_KEYS + LOCAL_VISUAL_MASK_KEYS
    )


def remove_all_visual(batch: dict[str, Any]) -> dict[str, Any]:
    return clone_with_zeros(
        remove_local_visual(batch), CONTEXT_VALUE_KEYS + CONTEXT_MASK_KEYS
    )


def swapped_order(size: int, pairs: tuple[tuple[int, int], ...], device: torch.device) -> torch.Tensor:
    order = torch.arange(size, device=device)
    for left, right in pairs:
        order[left], order[right] = order[right].clone(), order[left].clone()
    return order


def swap_axis(value: torch.Tensor, axis: int, pairs: tuple[tuple[int, int], ...]) -> torch.Tensor:
    order = swapped_order(value.shape[axis], pairs, value.device)
    return value.index_select(axis, order)


def swap_left_right(batch: dict[str, Any]) -> dict[str, Any]:
    """Swap labelled left/right streams without geometrically mirroring them."""
    output = dict(batch)
    joint_pairs = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))
    for key in ("skeleton_features", "skeleton_feature_mask", "skeleton_joint_mask"):
        output[key] = swap_axis(batch[key], 2, joint_pairs)
    relation_pairs = ((0, 1), (3, 4), (5, 6), (7, 8), (15, 16))
    for key in ("skeleton_relations", "skeleton_relation_mask"):
        output[key] = swap_axis(batch[key], 2, relation_pairs)

    device_pairs = ((1, 2), (3, 4))
    for key in (
        "imu_values",
        "imu_raw_vectors",
        "imu_time_seconds",
        "imu_frame_index",
        "imu_point_mask",
        "imu_device_mask",
    ):
        output[key] = swap_axis(batch[key], 1, device_pairs)
    output["imu_interval_counts"] = swap_axis(
        batch["imu_interval_counts"], 2, device_pairs
    )

    output["arm_spatial_features"] = swap_axis(
        batch["arm_spatial_features"], 3, ((0, 1),)
    )
    output["detail_spatial_features"] = swap_axis(
        batch["detail_spatial_features"], 3, ((0, 1),)
    )
    region_pairs = ((0, 1), (2, 3))
    for key in (
        "local_geometry_features",
        "oriented_roi_geometry",
        "oriented_angle_valid",
        "local_roi_valid",
        "local_roi_quality",
        "local_roi_source",
        "local_roi_clipped_ratio",
    ):
        output[key] = swap_axis(batch[key], 2, region_pairs)
    return output


def copy_tensor_dict(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in source.items()
    }


def remove_soft_objects(visual: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Remove learned object/surface candidates and their contact proxy only.

    Candidate positions follow LocalVisualObjectEncoder._pad_sources.  Assertions
    below make this audit fail loudly if the tensor contract changes.
    """
    output = copy_tensor_dict(visual)
    context_count = int(batch["context_features"].shape[2] * batch["context_features"].shape[3])
    arm_count = int(
        batch["arm_spatial_features"].shape[2]
        * batch["arm_spatial_features"].shape[4]
        * batch["arm_spatial_features"].shape[5]
    )
    detail_count = int(
        batch["detail_spatial_features"].shape[2]
        * batch["detail_spatial_features"].shape[4]
        * batch["detail_spatial_features"].shape[5]
    )
    arm_object = 1 + context_count + arm_count + detail_count
    hand_object = 1 + context_count + detail_count + detail_count
    workspace_left = 1 + context_count + 3 * detail_count
    workspace_right = workspace_left + 1
    workspace_surface = workspace_left + 2
    positions = {
        3: (arm_object,),
        4: (arm_object,),
        5: (hand_object,),
        6: (hand_object,),
        9: (workspace_left, workspace_right, workspace_surface),
    }
    candidates = output["part_sources"].shape[3]
    if max(max(values) for values in positions.values()) >= candidates:
        raise RuntimeError(
            f"object candidate contract changed: candidates={candidates}, positions={positions}"
        )
    for part, indices in positions.items():
        for index in indices:
            output["part_sources"][:, :, part, index] = 0
            output["part_source_mask"][:, :, part, index] = False
    for key in ("left_object_token", "right_object_token", "surface_token"):
        output[key].zero_()
    for key in ("left_object_quality", "right_object_quality", "surface_quality"):
        output[key].zero_()
    output["contact_proxy"].zero_()
    return output


def valid_time_permutations(
    frame_mask: torch.Tensor, mode: str, generator: torch.Generator
) -> list[torch.Tensor]:
    permutations: list[torch.Tensor] = []
    for index in range(frame_mask.shape[0]):
        count = int(frame_mask[index].sum())
        if mode == "reverse":
            permutation = torch.arange(count - 1, -1, -1, device=frame_mask.device)
        elif mode == "shuffle":
            permutation = torch.randperm(count, generator=generator, device="cpu").to(
                frame_mask.device
            )
        elif mode == "shift":
            offset = max(1, count // 3)
            permutation = torch.roll(
                torch.arange(count, device=frame_mask.device), shifts=offset
            )
        else:
            raise ValueError(mode)
        permutations.append(permutation)
    return permutations


def permute_time(
    source: dict[str, Any], frame_mask: torch.Tensor, mode: str, generator: torch.Generator
) -> dict[str, Any]:
    output = copy_tensor_dict(source)
    permutations = valid_time_permutations(frame_mask, mode, generator)
    batch, steps = frame_mask.shape
    for key, value in source.items():
        if not torch.is_tensor(value) or value.ndim < 2 or tuple(value.shape[:2]) != (batch, steps):
            continue
        changed = value.clone()
        for row, permutation in enumerate(permutations):
            count = len(permutation)
            changed[row, :count] = value[row, permutation]
        output[key] = changed
    return output


def different_label_permutation(labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    count = len(labels)
    if count <= 1:
        return torch.arange(count, device=labels.device), 0
    best = torch.roll(torch.arange(count, device=labels.device), shifts=1)
    best_different = int((labels[best] != labels).sum())
    for shift in range(2, count):
        candidate = torch.roll(torch.arange(count, device=labels.device), shifts=shift)
        different = int((labels[candidate] != labels).sum())
        if different > best_different:
            best, best_different = candidate, different
    return best, best_different


def permute_batch(source: dict[str, Any], permutation: torch.Tensor) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in source.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == len(permutation):
            output[key] = value.index_select(0, permutation)
        else:
            output[key] = copy.deepcopy(value)
    return output


def classify_components(
    model: P46Step10Model,
    motion: dict[str, torch.Tensor],
    visual: dict[str, torch.Tensor],
    frame_mask: torch.Tensor,
    fused_mode: str | None,
    generator: torch.Generator,
) -> torch.Tensor:
    fusion = model.encoder.fusion(motion, visual, frame_mask)
    contact = visual["contact_proxy"]
    if fused_mode is not None:
        fused = {
            "event_tokens": fusion["event_tokens"],
            "event_mask": fusion["event_mask"],
            "soft_event_gate": fusion["soft_event_gate"],
            "contact_proxy": contact,
        }
        fused = permute_time(fused, frame_mask, fused_mode, generator)
        event_tokens = fused["event_tokens"]
        event_mask = fused["event_mask"]
        soft_gate = fused["soft_event_gate"]
        contact = fused["contact_proxy"]
    else:
        event_tokens = fusion["event_tokens"]
        event_mask = fusion["event_mask"]
        soft_gate = fusion["soft_event_gate"]
    temporal = model.encoder.temporal(
        event_tokens, event_mask, soft_gate, contact, frame_mask
    )
    return model.detail_head(temporal["trial_embedding"])


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(len(HARD_CLASS_IDS)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "frozen_config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        run / "best_macro_f1.pt", map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    validation = P46EventDataset(
        Path(config["event_run"]), Path(config["context_run"]), split="val"
    )
    sampler = FrameBudgetBatchSampler(
        validation.frame_lengths,
        maximum_batch_size=args.batch_size,
        frame_budget=args.frame_budget,
    )
    loader = DataLoader(
        validation,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_p46_events,
    )
    model = P46Step10Model(subjects=len(config["train_subjects"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    digest_before = state_digest(model)

    condition_names = (
        "baseline",
        "event_time_reverse",
        "event_time_shuffle",
        "visual_time_shift_one_third",
        "visual_wrong_trial",
        "soft_object_surface_contact_removed",
        "local_roi_removed_keep_context",
        "all_visual_removed",
        "skeleton_removed",
        "imu_removed",
        "skeleton_and_imu_removed",
        "left_right_streams_swapped",
    )
    all_logits: dict[str, list[np.ndarray]] = {name: [] for name in condition_names}
    all_labels: list[np.ndarray] = []
    all_sources: list[str] = []
    all_users: list[str] = []
    wrong_trial_different = 0
    wrong_trial_total = 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if args.maximum_batches and batch_index >= args.maximum_batches:
                break
            batch = move_batch(raw_batch, device)
            labels = batch["detail_index"]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                motion = model.encoder.motion(batch)
                visual = model.encoder.visual(batch)
                condition_logits: dict[str, torch.Tensor] = {}
                condition_logits["baseline"] = classify_components(
                    model, motion, visual, batch["frame_mask"], None, generator
                )
                condition_logits["event_time_reverse"] = classify_components(
                    model, motion, visual, batch["frame_mask"], "reverse", generator
                )
                condition_logits["event_time_shuffle"] = classify_components(
                    model, motion, visual, batch["frame_mask"], "shuffle", generator
                )
                shifted_visual = permute_time(
                    visual, batch["frame_mask"], "shift", generator
                )
                condition_logits["visual_time_shift_one_third"] = classify_components(
                    model, motion, shifted_visual, batch["frame_mask"], None, generator
                )
                donor_permutation, different = different_label_permutation(labels)
                donor_visual = permute_batch(visual, donor_permutation)
                wrong_trial_different += different
                wrong_trial_total += len(labels)
                condition_logits["visual_wrong_trial"] = classify_components(
                    model, motion, donor_visual, batch["frame_mask"], None, generator
                )
                no_objects = remove_soft_objects(visual, batch)
                condition_logits["soft_object_surface_contact_removed"] = classify_components(
                    model, motion, no_objects, batch["frame_mask"], None, generator
                )
                local_batch = remove_local_visual(batch)
                local_visual = model.encoder.visual(local_batch)
                condition_logits["local_roi_removed_keep_context"] = classify_components(
                    model, motion, local_visual, batch["frame_mask"], None, generator
                )
                no_visual_batch = remove_all_visual(batch)
                no_visual = model.encoder.visual(no_visual_batch)
                condition_logits["all_visual_removed"] = classify_components(
                    model, motion, no_visual, batch["frame_mask"], None, generator
                )
                no_skeleton_motion = model.encoder.motion(remove_skeleton(batch))
                condition_logits["skeleton_removed"] = classify_components(
                    model, no_skeleton_motion, visual, batch["frame_mask"], None, generator
                )
                no_imu_motion = model.encoder.motion(remove_imu(batch))
                condition_logits["imu_removed"] = classify_components(
                    model, no_imu_motion, visual, batch["frame_mask"], None, generator
                )
                no_motion = model.encoder.motion(remove_motion(batch))
                condition_logits["skeleton_and_imu_removed"] = classify_components(
                    model, no_motion, visual, batch["frame_mask"], None, generator
                )
                swapped_batch = swap_left_right(batch)
                swapped_motion = model.encoder.motion(swapped_batch)
                swapped_visual = model.encoder.visual(swapped_batch)
                condition_logits["left_right_streams_swapped"] = classify_components(
                    model,
                    swapped_motion,
                    swapped_visual,
                    batch["frame_mask"],
                    None,
                    generator,
                )
            for name in condition_names:
                all_logits[name].append(condition_logits[name].float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_sources.extend(batch["source_id"])
            all_users.extend(batch["user_id"])
            print(
                json.dumps(
                    {
                        "stage": "ablation_batch",
                        "batch": batch_index + 1,
                        "batches": len(loader),
                        "samples_seen": len(all_sources),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    label_array = np.concatenate(all_labels)
    logit_arrays = {name: np.concatenate(values) for name, values in all_logits.items()}
    if len(set(all_sources)) != len(all_sources):
        raise RuntimeError("validation source IDs are not unique")
    if not args.maximum_batches and len(all_sources) != len(validation):
        raise RuntimeError(
            f"validation coverage failed: {len(all_sources)} != {len(validation)}"
        )
    baseline_prediction = logit_arrays["baseline"].argmax(1)
    saved_rows: dict[str, dict[str, str]] = {}
    with (run / "best_macro_f1_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            saved_rows[row["source_id"]] = row
    if not args.maximum_batches:
        saved_prediction = np.asarray(
            [int(saved_rows[source]["predicted_detail_index"]) for source in all_sources]
        )
        saved_label = np.asarray(
            [int(saved_rows[source]["true_detail_index"]) for source in all_sources]
        )
        if not np.array_equal(label_array, saved_label):
            raise RuntimeError("baseline labels do not reproduce the saved evaluation")
        if not np.array_equal(baseline_prediction, saved_prediction):
            mismatches = int((baseline_prediction != saved_prediction).sum())
            raise RuntimeError(
                f"baseline predictions do not reproduce checkpoint CSV: {mismatches}"
            )

    baseline_correct = baseline_prediction == label_array
    baseline_metrics = metrics(label_array, logit_arrays["baseline"])
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    per_subject_rows: list[dict[str, Any]] = []
    for name in condition_names:
        logits = logit_arrays[name]
        prediction = logits.argmax(1)
        current_correct = prediction == label_array
        result = metrics(label_array, logits)
        changed = prediction != baseline_prediction
        metric_rows.append(
            {
                "condition": name,
                **result,
                "accuracy_delta_pp": 100.0 * (result["accuracy"] - baseline_metrics["accuracy"]),
                "balanced_accuracy_delta_pp": 100.0
                * (result["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]),
                "macro_f1_delta_pp": 100.0
                * (result["macro_f1"] - baseline_metrics["macro_f1"]),
                "prediction_changes": int(changed.sum()),
                "baseline_correct_to_wrong": int((baseline_correct & ~current_correct).sum()),
                "baseline_wrong_to_correct": int((~baseline_correct & current_correct).sum()),
                "mean_absolute_logit_change": float(
                    np.abs(logits - logit_arrays["baseline"]).mean()
                ),
            }
        )
        for index, source in enumerate(all_sources):
            prediction_rows.append(
                {
                    "condition": name,
                    "source_id": source,
                    "user_id": all_users[index],
                    "true_detail_index": int(label_array[index]),
                    "true_class_id": int(HARD_CLASS_IDS[int(label_array[index])]),
                    "predicted_detail_index": int(prediction[index]),
                    "predicted_class_id": int(HARD_CLASS_IDS[int(prediction[index])]),
                    "correct": int(current_correct[index]),
                    "baseline_prediction_changed": int(changed[index]),
                }
            )
        for detail_index, class_id in enumerate(HARD_CLASS_IDS):
            mask = label_array == detail_index
            per_class_rows.append(
                {
                    "condition": name,
                    "class_id": int(class_id),
                    "samples": int(mask.sum()),
                    "accuracy": float(current_correct[mask].mean()) if mask.any() else 0.0,
                    "baseline_accuracy": float(baseline_correct[mask].mean()) if mask.any() else 0.0,
                    "delta_pp": 100.0
                    * (
                        float(current_correct[mask].mean())
                        - float(baseline_correct[mask].mean())
                    )
                    if mask.any()
                    else 0.0,
                    "prediction_changes": int(changed[mask].sum()),
                }
            )
        for user in sorted(set(all_users)):
            mask = np.asarray([value == user for value in all_users])
            per_subject_rows.append(
                {
                    "condition": name,
                    "user_id": user,
                    "samples": int(mask.sum()),
                    "accuracy": float(current_correct[mask].mean()),
                    "baseline_accuracy": float(baseline_correct[mask].mean()),
                    "delta_pp": 100.0
                    * (
                        float(current_correct[mask].mean())
                        - float(baseline_correct[mask].mean())
                    ),
                    "prediction_changes": int(changed[mask].sum()),
                }
            )

    digest_after = state_digest(model)
    if digest_before != digest_after:
        raise RuntimeError("model weights changed during the inference-only audit")
    summary = {
        "stage": "P46_zero_training_causal_ablation_v1",
        "checkpoint": str((run / "best_macro_f1.pt").resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_performed": False,
        "human_labels_used": False,
        "validation_samples": len(all_sources),
        "validation_unique_samples": len(set(all_sources)),
        "validation_subjects": sorted(set(all_users)),
        "baseline": baseline_metrics,
        "conditions": metric_rows,
        "wrong_trial_visual_donor": {
            "total": wrong_trial_total,
            "different_class": wrong_trial_different,
            "different_class_fraction": wrong_trial_different
            / max(wrong_trial_total, 1),
        },
        "weight_digest_before": digest_before,
        "weight_digest_after": digest_after,
        "weights_unchanged": digest_before == digest_after,
        "maximum_batches": args.maximum_batches,
        "seed": SEED,
    }
    write_csv(output / "metrics.csv", metric_rows)
    write_csv(output / "predictions.csv", prediction_rows)
    write_csv(output / "per_class.csv", per_class_rows)
    write_csv(output / "per_subject.csv", per_subject_rows)
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
