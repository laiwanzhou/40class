from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from probe_p27r3_incremental_information import metric_bundle, write_csv


P12_PROTOCOLS = {
    0: (0.9167410586298098, 0.6363677656203999, 0.4),
    1: (0.8908255030742397, 0.5918757644479288, 0.4),
    2: (0.9401831574570029, 0.6312607866282122, 0.4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse fold-pure precomputed model logits with the exact-fold RF-IMU expert"
    )
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--imu-logits", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inner-fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flatten(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def main() -> None:
    args = parse_args()
    fold = int(args.inner_fold)
    if fold not in P12_PROTOCOLS:
        raise ValueError(f"Unsupported fold: {fold}")
    base = np.load(args.base_logits.resolve(), allow_pickle=False)
    imu = np.load(args.imu_logits.resolve(), allow_pickle=False)
    if bool(base["outer_held_predictions_generated"]) or bool(
        imu["outer_held_predictions_generated"]
    ):
        raise RuntimeError("Outer-held predictions are forbidden")
    sample_ids = base["sample_ids"].astype(str)
    labels = base["labels"].astype(np.int64)
    base_logits = base["logits"].astype(np.float32)
    position = {
        str(sample_id): index
        for index, sample_id in enumerate(imu["sample_ids"].astype(str))
    }
    indices = np.asarray([position[sample_id] for sample_id in sample_ids])
    if not np.array_equal(imu["labels"][indices], labels):
        raise RuntimeError("Base and IMU labels do not align")
    imu_logits = imu["imu_logits"][indices].astype(np.float32)
    device_counts = imu["imu_device_counts"][indices].astype(np.int64)
    subject_by_id = {
        row["sample_id"]: row["user_id"]
        for row in read_csv(args.manifest.resolve())
        if row["split"] == "val"
    }
    subjects = np.asarray([subject_by_id[sample_id] for sample_id in sample_ids])
    base_temperature, imu_temperature, maximum_weight = P12_PROTOCOLS[fold]
    weights = (
        maximum_weight
        * np.clip(device_counts.astype(np.float32) / 5.0, 0.0, 1.0)
    )[:, None]
    fused_logits = (
        (1.0 - weights) * base_logits / base_temperature
        + weights * imu_logits / imu_temperature
    )
    methods = {
        "base": base_logits,
        "base_rfimu_fixed": fused_logits,
    }
    rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for method, logits in methods.items():
        predictions = logits.argmax(axis=1)
        rows.append(
            {"method": method, **flatten(metric_bundle(labels, predictions))}
        )
        for subject in sorted(set(subjects.tolist())):
            selected = subjects == subject
            subject_rows.append(
                {
                    "method": method,
                    "subject": subject,
                    **metric_bundle(
                        labels[selected], predictions[selected]
                    )["overall"],
                }
            )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "per_subject.csv", subject_rows)
    np.savez_compressed(
        output / f"fold_{fold}_logits.npz",
        protocol=np.asarray("p27-precomputed-rfimu-fixed-inner-v1"),
        sample_ids=sample_ids,
        labels=labels,
        subjects=subjects,
        base_logits=base_logits,
        imu_logits=imu_logits,
        fused_logits=fused_logits.astype(np.float32),
        imu_device_counts=device_counts,
        outer_held_predictions_generated=np.asarray(False),
    )
    summary = {
        "protocol": "outer-fold-0 train subjects only; fixed external P12 RF-IMU calibration",
        "outer_held_predictions_generated": False,
        "inner_fold": fold,
        "fixed_parameters": {
            "base_temperature": base_temperature,
            "imu_temperature": imu_temperature,
            "maximum_weight": maximum_weight,
        },
        "metrics": {row["method"]: row for row in rows},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
