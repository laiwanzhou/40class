from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from probe_p27r3_incremental_information import metric_bundle


PROJECT_DIR = Path(__file__).resolve().parent
P12_PROTOCOLS = {
    0: (0.9167410586298098, 0.6363677656203999, 0.4),
    1: (0.8908255030742397, 0.5918757644479288, 0.4),
    2: (0.9401831574570029, 0.6312607866282122, 0.4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fold-0-fixed 0.75 2D / 0.25 S3D joint-logit blend "
            "on outer-train subject-disjoint inner folds"
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p27_strong_inner" / "ir_skeleton_imu",
    )
    parser.add_argument(
        "--s3d-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p27_strong_inner" / "ir_s3d_skeleton",
    )
    parser.add_argument(
        "--s3d-template",
        default="fold_{fold}_fixed9/fixed_epoch_09_held_logits.npz",
    )
    parser.add_argument(
        "--s3d-fold0",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p27_strong_inner"
            / "ir_s3d_skeleton"
            / "fold_0_b16"
            / "best_accuracy_held_logits.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p27_strong_inner"
            / "ir_s3d_fixed_blend"
        ),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--s3d-weight", type=float, default=0.25)
    parser.add_argument(
        "--fixed-epoch-metadata",
        type=int,
        default=9,
        help="Frozen reproduction epoch recorded in the protocol manifest.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def align_logits(
    archive: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    if bool(archive["outer_held_predictions_generated"]):
        raise RuntimeError("Archive unexpectedly contains outer-held predictions")
    positions = {
        str(sample_id): index
        for index, sample_id in enumerate(archive["sample_ids"].astype(str))
    }
    missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in positions]
    if missing:
        raise RuntimeError(f"S3D archive misses {len(missing)} samples")
    indices = np.asarray([positions[str(sample_id)] for sample_id in sample_ids])
    if not np.array_equal(archive["labels"][indices], labels):
        raise RuntimeError("S3D labels do not align with the legal 2D archive")
    return archive["logits"][indices].astype(np.float32)


def main() -> None:
    args = parse_args()
    alpha = float(args.s3d_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--s3d-weight must be in [0, 1]")
    folds = sorted(set(int(fold) for fold in args.folds))
    if any(fold not in P12_PROTOCOLS for fold in folds):
        raise ValueError(f"Unsupported folds: {folds}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    with (PROJECT_DIR / "data" / "manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        class_names = {
            int(row["class_id"]): row["class_name"]
            for row in csv.DictReader(handle)
        }
    summary_folds: dict[str, Any] = {}
    for fold in folds:
        base_path = args.base_dir.resolve() / f"fold_{fold}_logits.npz"
        s3d_path = (
            args.s3d_fold0.resolve()
            if fold == 0
            else args.s3d_dir.resolve()
            / str(args.s3d_template).format(fold=fold)
        )
        base = np.load(base_path, allow_pickle=False)
        if bool(base["outer_held_predictions_generated"]):
            raise RuntimeError("Base archive unexpectedly contains outer-held predictions")
        sample_ids = base["sample_ids"].astype(str)
        labels = base["labels"].astype(np.int64)
        subjects = base["subjects"].astype(str)
        joint = base["joint_logits"].astype(np.float32)
        imu = base["imu_logits"].astype(np.float32)
        device_counts = base["imu_device_counts"].astype(np.float32)
        s3d_archive = np.load(s3d_path, allow_pickle=False)
        s3d = align_logits(s3d_archive, sample_ids, labels)

        joint_temperature, imu_temperature, maximum_weight = P12_PROTOCOLS[fold]
        imu_weight = (
            maximum_weight * np.clip(device_counts / 5.0, 0.0, 1.0)
        )[:, None]
        base_fused = (
            (1.0 - imu_weight) * joint / joint_temperature
            + imu_weight * imu / imu_temperature
        )
        s3d_fused = (
            (1.0 - imu_weight) * s3d / joint_temperature
            + imu_weight * imu / imu_temperature
        )
        blended_joint = (1.0 - alpha) * joint + alpha * s3d
        blended_fused = (
            (1.0 - imu_weight) * blended_joint / joint_temperature
            + imu_weight * imu / imu_temperature
        )
        methods = {
            "base_2d_rfimu_fixed": base_fused,
            "s3d_rfimu_fixed": s3d_fused,
            "fixed_2d_s3d_rfimu_blend": blended_fused,
        }
        predictions = {
            method: logits.argmax(axis=1) for method, logits in methods.items()
        }
        for method, predicted in predictions.items():
            metric_rows.append(
                {
                    "inner_fold": fold,
                    "method": method,
                    **flatten(metric_bundle(labels, predicted)),
                }
            )
            for subject in sorted(set(subjects.tolist())):
                selected = subjects == subject
                subject_rows.append(
                    {
                        "inner_fold": fold,
                        "method": method,
                        "subject": subject,
                        **metric_bundle(
                            labels[selected], predicted[selected]
                        )["overall"],
                    }
                )
            for class_id in range(40):
                selected_class = labels == class_id
                per_class_rows.append(
                    {
                        "inner_fold": fold,
                        "method": method,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "samples": int(selected_class.sum()),
                        "recall": float(
                            np.mean(predicted[selected_class] == labels[selected_class])
                        )
                        if selected_class.any()
                        else float("nan"),
                    }
                )

        base_correct = predictions["base_2d_rfimu_fixed"] == labels
        blend_correct = predictions["fixed_2d_s3d_rfimu_blend"] == labels
        for subject in ["__all__", *sorted(set(subjects.tolist()))]:
            selected = (
                np.ones(len(labels), dtype=bool)
                if subject == "__all__"
                else subjects == subject
            )
            rescue = int((~base_correct[selected] & blend_correct[selected]).sum())
            new_error = int((base_correct[selected] & ~blend_correct[selected]).sum())
            transition_rows.append(
                {
                    "inner_fold": fold,
                    "subject": subject,
                    "samples": int(selected.sum()),
                    "rescues": rescue,
                    "new_errors": new_error,
                    "net_rescues": rescue - new_error,
                }
            )
        np.savez_compressed(
            output / f"fold_{fold}_logits.npz",
            protocol=np.asarray("p27-s3d-fixed-blend-inner-v1"),
            sample_ids=sample_ids,
            labels=labels,
            subjects=subjects,
            base_fused_logits=base_fused.astype(np.float32),
            s3d_logits=s3d,
            blended_fused_logits=blended_fused.astype(np.float32),
            outer_held_predictions_generated=np.asarray(False),
        )
        summary_folds[str(fold)] = {
            "base_archive": str(base_path),
            "base_sha256": sha256(base_path),
            "s3d_archive": str(s3d_path),
            "s3d_sha256": sha256(s3d_path),
            "samples": int(len(labels)),
            "outer_held_predictions_generated": False,
        }

    write_csv(output / "fold_metrics.csv", metric_rows)
    write_csv(output / "per_subject.csv", subject_rows)
    write_csv(output / "transitions.csv", transition_rows)
    write_csv(output / "per_class.csv", per_class_rows)
    mean_rows: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in metric_rows}):
        selected = [row for row in metric_rows if row["method"] == method]
        numeric_keys = [
            key
            for key, value in selected[0].items()
            if key not in {"inner_fold", "method"} and isinstance(value, (int, float))
        ]
        mean_rows.append(
            {
                "method": method,
                **{
                    key: float(np.mean([float(row[key]) for row in selected]))
                    for key in numeric_keys
                },
            }
        )
    write_csv(output / "mean_metrics.csv", mean_rows)
    summary = {
        "protocol": "p27-s3d-fixed-blend-inner-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "development_choice": {
            "source": "inner fold 0 only",
            "s3d_weight": alpha,
            "fixed_s3d_epoch_for_reproduction": args.fixed_epoch_metadata,
        },
        "independent_reproduction_folds": [
            fold for fold in folds if fold in (1, 2)
        ],
        "folds": summary_folds,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
