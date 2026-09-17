from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align fold-pure P27 base logits with the P20 tiny IMU student."
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--tiny-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for fold in range(3):
        base = np.load(
            args.base_dir.resolve() / f"fold_{fold}_logits.npz",
            allow_pickle=False,
        )
        tiny = np.load(
            args.tiny_dir.resolve() / f"fold_{fold}_logits.npz",
            allow_pickle=False,
        )
        if bool(base["outer_held_predictions_generated"]) or bool(
            tiny["outer_held_predictions_generated"]
        ):
            raise RuntimeError("outer-held predictions are forbidden")
        tiny_location = {
            str(sample_id): index
            for index, sample_id in enumerate(tiny["sample_ids"])
        }
        imu_logits = np.zeros_like(base["joint_logits"], dtype=np.float32)
        device_counts = np.zeros(len(base["sample_ids"]), dtype=np.float32)
        present = np.zeros(len(base["sample_ids"]), dtype=np.float32)
        for index, sample_id in enumerate(base["sample_ids"]):
            source = tiny_location.get(str(sample_id))
            if source is None:
                continue
            if int(base["labels"][index]) != int(tiny["labels"][source]):
                raise RuntimeError(f"label mismatch: {sample_id}")
            imu_logits[index] = tiny["imu_logits"][source]
            device_counts[index] = tiny["imu_device_counts"][source]
            present[index] = 1.0
        np.savez_compressed(
            output / f"fold_{fold}_logits.npz",
            protocol=np.asarray("p27-ir-skeleton-tiny-imu-inner-v1"),
            sample_ids=base["sample_ids"],
            labels=base["labels"],
            subjects=base["subjects"],
            joint_logits=base["joint_logits"],
            imu_logits=imu_logits,
            imu_present=present,
            imu_device_counts=device_counts,
            outer_held_predictions_generated=np.asarray(False),
        )
        print(
            f"fold {fold}: {int(present.sum())}/{len(present)} IMU-present samples"
        )


if __name__ == "__main__":
    main()
