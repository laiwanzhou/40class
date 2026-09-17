from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p86_cached_motion_data import P86CachedSequenceMotionDataset
from p86_mobind_lite_model import P86JointMotionEncoder
from train_p86_cached_motion_proxy import load_npz
from train_p86_mobind_fusion_proxy import (
    build_model,
    evaluate,
    loader,
    make_dataset,
    seed_all,
)
from train_p86_visual_student_oof import split_universe


PATH_ARGUMENTS = {
    "visual_checkpoint",
    "sequence_cache",
    "pretrain_checkpoint",
    "motion_cache",
    "pixel_cache",
    "teacher_features",
    "teacher_logits",
    "imu_teacher_logits",
    "imu_event_features",
    "output_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Disable the IMU token residual inside a trained P86 joint model and "
            "measure its causal proxy contribution without retraining."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(
    records: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    output = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                field: key,
                "samples": len(values),
                "full_correct": sum(value["full_correct"] for value in values),
                "ablated_correct": sum(value["ablated_correct"] for value in values),
                "imu_rescues": sum(value["imu_rescue"] for value in values),
                "imu_harms": sum(value["imu_harm"] for value in values),
            }
        )
    return output


def runtime_args(config: dict[str, Any]) -> argparse.Namespace:
    values = dict(config)
    for key in PATH_ARGUMENTS:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return argparse.Namespace(**values)


def main() -> None:
    cli = parse_args()
    run = cli.run_dir.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["modality"] != "joint":
        raise ValueError("IMU residual ablation requires a joint run")
    args = runtime_args(summary["config"])
    seed_all(args.seed)

    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, outer_fold=0, seed=args.seed)
    proxy_indices = np.asarray(split["outer_held"], dtype=np.int64)
    forbidden_indices = np.asarray(split["inner_dev"], dtype=np.int64)
    if (len(proxy_indices), len(forbidden_indices)) != (973, 444):
        raise RuntimeError("P86 fixed proxy/permanent counts changed")
    full_data = P86CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        imu_teacher_logits=args.imu_teacher_logits,
        imu_event_features=args.imu_event_features,
    )
    proxy = make_dataset(
        args,
        full_data,
        split["sample_ids"][proxy_indices],
        temporal_augment=False,
    )

    model, _, _ = build_model(args)
    checkpoint = torch.load(
        run / "unified_student.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    joint = model.motion_residual.encoder
    if not isinstance(joint, P86JointMotionEncoder):
        raise TypeError("loaded model does not contain P86JointMotionEncoder")
    trained_maximum = joint.maximum_imu_residual
    joint.maximum_imu_residual = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ablated_metrics, ablated_rows, ablated_logits = evaluate(
        model, loader(proxy, args, False), device, args.max_eval_batches
    )

    full_rows = read_rows(run / "proxy_validation_predictions.csv")
    records = []
    for row in ablated_rows:
        full_row = full_rows[row["sample_id"]]
        label = int(row["label"])
        full_prediction = int(full_row["prediction"])
        ablated_prediction = int(row["prediction"])
        full_correct = full_prediction == label
        ablated_correct = ablated_prediction == label
        records.append(
            {
                "sample_id": row["sample_id"],
                "user_id": row["user_id"],
                "class_id": label,
                "full_prediction": full_prediction,
                "ablated_prediction": ablated_prediction,
                "full_correct": int(full_correct),
                "ablated_correct": int(ablated_correct),
                "changed": int(full_prediction != ablated_prediction),
                "imu_rescue": int(full_correct and not ablated_correct),
                "imu_harm": int(not full_correct and ablated_correct),
            }
        )
    output = {
        "protocol": (
            "Load the trained joint checkpoint and set only its maximum IMU token "
            "residual to zero. No retraining and no permanent-validation access."
        ),
        "trained_maximum_imu_residual": trained_maximum,
        "full_metrics": summary["proxy_metrics"],
        "ablated_metrics": ablated_metrics,
        "changed": sum(record["changed"] for record in records),
        "imu_rescues": sum(record["imu_rescue"] for record in records),
        "imu_harms": sum(record["imu_harm"] for record in records),
    }
    write_rows(run / "joint_imu_ablation_predictions.csv", records)
    write_rows(
        run / "joint_imu_ablation_per_user.csv",
        grouped_summary(records, "user_id"),
    )
    write_rows(
        run / "joint_imu_ablation_per_class.csv",
        grouped_summary(records, "class_id"),
    )
    np.save(run / "joint_imu_ablation_logits.npy", ablated_logits)
    (run / "joint_imu_ablation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
