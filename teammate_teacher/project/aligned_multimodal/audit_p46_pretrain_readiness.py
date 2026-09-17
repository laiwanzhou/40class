from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p30_shared_dir_roi_model import SharedResNet18Pyramid, model_size_mib as backbone_size_mib
from p46_event_data import P46EventDataset, collate_p46_events
from p46_event_model import P46EventTokenEncoder, model_size_mib, parameter_count
from p46_protocol import EXPECTED, HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_RUN = PROJECT_DIR / "runs" / "p46_event_inputs_full"
DEFAULT_CONTEXT_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P46 [0]-[9] before any Step 10 training.")
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--context-run", type=Path, default=DEFAULT_CONTEXT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_representatives(rows: list[dict[str, str]]) -> list[str]:
    selected: list[str] = []
    for split in ("train", "val"):
        candidates = [row for row in rows if row["p46_split"] == split]
        if candidates:
            selected.append(max(candidates, key=lambda row: int(row["frames"]))["source_id"])
    missing_imu = next((row for row in rows if int(row["imu_points"]) == 0), None)
    if missing_imu is not None:
        selected.append(missing_imu["source_id"])
    class25 = next((row for row in rows if int(row["class_id"]) == 25), None)
    if class25 is not None:
        selected.append(class25["source_id"])
    return list(dict.fromkeys(selected))


def finite_cache_audit(dataset: P46EventDataset) -> dict[str, Any]:
    representatives = choose_representatives(dataset.rows)
    failures: list[str] = []
    frame_count_failures: list[str] = []
    keys = (
        "arm_spatial_features",
        "detail_spatial_features",
        "local_geometry_features",
        "oriented_roi_geometry",
        "skeleton_features",
        "skeleton_relations",
        "body_axes_camera",
        "imu_values",
        "context_features",
    )
    for index, row in enumerate(dataset.rows):
        # Loading through the real Dataset audits every local cache, the exact
        # P30 context join and all frame-id contracts used by Step 10.
        item = dataset[index]
        if len(item["frame_ids"]) != int(row["frames"]):
            frame_count_failures.append(item["source_id"])
        for key in (
            *keys,
        ):
            if not torch.isfinite(item[key]).all():
                failures.append(f"{item['source_id']}:{key}")
    return {
        "audited_trials": len(dataset),
        "audited_local_and_context_frame_joins": len(dataset),
        "representative_source_ids": representatives,
        "frame_count_failures": frame_count_failures,
        "nonfinite_failures": failures,
    }


def forward_audit(
    dataset: P46EventDataset, representative_ids: list[str], device: torch.device
) -> dict[str, Any]:
    by_id = {row["source_id"]: index for index, row in enumerate(dataset.rows)}
    batch = collate_p46_events([dataset[by_id[source_id]] for source_id in representative_ids])
    model = P46EventTokenEncoder().to(device).eval()
    tensor_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    use_amp = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=use_amp
    ):
        output = model(tensor_batch)
    floating_keys = (
        "event_tokens",
        "soft_event_gate",
        "trial_embedding",
        "phase_logits",
        "explicit_event_statistics",
    )
    nonfinite = [key for key in floating_keys if not torch.isfinite(output[key]).all()]
    frozen_backbone = SharedResNet18Pyramid(imagenet_pretrained=False)
    return {
        "batch_size": len(representative_ids),
        "maximum_frames": int(batch["frame_mask"].shape[1]),
        "event_tokens_shape": list(output["event_tokens"].shape),
        "phase_logits_shape": list(output["phase_logits"].shape),
        "trial_embedding_shape": list(output["trial_embedding"].shape),
        "nonfinite_outputs": nonfinite,
        "classification_logits_present": "logits" in output,
        "event_encoder_parameters": parameter_count(model),
        "event_encoder_fp32_mib": model_size_mib(model),
        "event_encoder_fp16_mib": model_size_mib(model, bytes_per_parameter=2),
        "frozen_resnet18_fp16_mib": backbone_size_mib(
            frozen_backbone, bytes_per_parameter=2
        ),
        "combined_feature_and_event_encoder_fp16_mib": (
            model_size_mib(model, bytes_per_parameter=2)
            + backbone_size_mib(frozen_backbone, bytes_per_parameter=2)
        ),
    }


def main() -> None:
    args = parse_args()
    event_run = args.event_run.resolve()
    context_run = args.context_run.resolve()
    rows = read_csv(event_run / "trial_summary.csv")
    train = [row for row in rows if row["p46_split"] == "train"]
    val = [row for row in rows if row["p46_split"] == "val"]
    observed_classes = sorted({int(row["class_id"]) for row in rows})
    train_subjects = sorted({row["user_id"] for row in train})
    val_subjects = sorted({row["user_id"] for row in val})
    if len(train) != EXPECTED["train_detail_trials"]:
        raise RuntimeError(f"P46 train detail count changed: {len(train)}")
    if len(val) != EXPECTED["val_detail_trials"]:
        raise RuntimeError(f"P46 val detail count changed: {len(val)}")
    total_frames = sum(int(row["frames"]) for row in rows)
    expected_frames = EXPECTED["train_detail_frames"] + EXPECTED["val_detail_frames"]
    if total_frames != expected_frames:
        raise RuntimeError(
            f"P46 all-frame contract changed: expected={expected_frames}, observed={total_frames}"
        )
    if observed_classes != list(HARD_CLASS_IDS):
        raise RuntimeError(f"P46 hard classes changed: {observed_classes}")
    overlap = sorted(set(train_subjects) & set(val_subjects))
    if overlap:
        raise RuntimeError(f"P46 subject leakage: {overlap}")
    dataset = P46EventDataset(event_run, context_run)
    finite = finite_cache_audit(dataset)
    if finite["frame_count_failures"] or finite["nonfinite_failures"]:
        raise RuntimeError(f"P46 cache audit failed: {finite}")
    forward = forward_audit(
        dataset, finite["representative_source_ids"], torch.device(args.device)
    )
    if forward["nonfinite_outputs"] or forward["classification_logits_present"]:
        raise RuntimeError(f"P46 forward contract failed: {forward}")
    report = {
        "stage": "P46_steps_0_to_9_pretrain_readiness",
        "step10_training_started": False,
        "event_run": str(event_run),
        "context_run": str(context_run),
        "train_detail_trials": len(train),
        "val_detail_trials": len(val),
        "hard_class_ids": observed_classes,
        "train_subject_ids": train_subjects,
        "val_subject_ids": val_subjects,
        "subject_overlap": overlap,
        "total_frames": total_frames,
        "missing_imu_trials": sum(int(row["imu_points"]) == 0 for row in rows),
        "cache_bytes": sum(int(row["cache_bytes"]) for row in rows),
        "mean_body_axes_raw_valid_rate": float(
            np.mean([float(row["body_axes_raw_valid_rate"]) for row in rows])
        ),
        "mean_oriented_angle_valid_rate": float(
            np.mean([float(row["oriented_angle_valid_rate"]) for row in rows])
        ),
        "finite_cache_audit": finite,
        "forward_audit": forward,
        "known_coordinate_limit": (
            "IMU is device-relative because no IMU-to-Skeleton body-frame extrinsic is provided"
        ),
        "known_depth_limit": (
            "decoded JET index is ordered and temporally stable but is not interpreted as metric metres"
        ),
    }
    output = args.output.resolve() if args.output else event_run / "pretrain_readiness.json"
    temporary = output.with_suffix(output.suffix + ".building")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
