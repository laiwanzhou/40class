from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from analyze_thermal_oof_fusion import (
    align_thermal,
    cross_fitted_thermal_residual,
    load_thermal,
    metric_dict,
)
from evaluate_conditional_expert_routing import (
    bootstrap_delta,
    route_cross_fitted,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_SD_ROOT = PROJECT_DIR / "runs" / "p11_fp16_oof"
DEFAULT_IMU_OOF = (
    PROJECT_DIR
    / "runs"
    / "p3_imu_oof"
    / "stat_random_forest_device_dropout_aligned_oof.npz"
)
DEFAULT_IMU_SUMMARY = PROJECT_DIR / "runs" / "p3_imu_oof" / "summary.json"
DEFAULT_THERMAL_ROOT = (
    REPO_DIR / "thermal_baseline" / "runs" / "p11_thermal_imagenet_fp16"
)
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p12_complete_oof"
SMALL_ACTION_IDS = np.asarray(
    (1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39),
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete 2914-row missing-aware OOF baseline. IMU-missing "
            "samples fall back to calibrated S+D; Thermal fusion and routing are "
            "re-fitted outside every held subject fold."
        )
    )
    parser.add_argument("--sd-root", type=Path, default=DEFAULT_SD_ROOT)
    parser.add_argument("--imu-oof", type=Path, default=DEFAULT_IMU_OOF)
    parser.add_argument("--imu-summary", type=Path, default=DEFAULT_IMU_SUMMARY)
    parser.add_argument("--thermal-root", type=Path, default=DEFAULT_THERMAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def load_sd(root: Path) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "labels": [],
        "folds": [],
        "skeleton_logits": [],
        "depth_logits": [],
    }
    for fold in range(3):
        path = root / f"fold_{fold}" / "logits_fp16.npz"
        with np.load(path, allow_pickle=False) as data:
            count = len(data["labels"])
            result["sample_ids"].append(data["sample_ids"].astype(str))
            result["labels"].append(data["labels"].astype(np.int64))
            result["folds"].append(np.full(count, fold, dtype=np.int64))
            result["skeleton_logits"].append(
                data["skeleton_logits"].astype(np.float32)
            )
            result["depth_logits"].append(data["depth_logits"].astype(np.float32))
    combined = {key: np.concatenate(value) for key, value in result.items()}
    if len(set(combined["sample_ids"].tolist())) != len(combined["sample_ids"]):
        raise ValueError("Duplicate S+D OOF sample IDs")
    order = np.argsort(combined["sample_ids"])
    return {key: value[order] for key, value in combined.items()}


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["sample_id"]: row for row in rows}


def load_imu_protocols(path: Path) -> dict[int, dict[str, float]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = summary["sources"]["stat_random_forest_device_dropout"][
        "cross_fitted_protocols"
    ]
    return {
        int(row["target_fold"]): {
            "sd_temperature": float(row["sd_temperature"]),
            "imu_temperature": float(row["imu_temperature"]),
            "imu_weight": float(row["selected_imu_weight"]),
        }
        for row in rows
    }


def align_imu(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    sd_logits: np.ndarray,
    imu_path: Path,
    protocols: dict[int, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(imu_path, allow_pickle=False) as data:
        source_ids = data["sample_ids"].astype(str)
        source_labels = data["labels"].astype(np.int64)
        source_folds = data["folds"].astype(np.int64)
        source_imu = data["imu_logits"].astype(np.float32)
        source_fused = data["fused_logits"].astype(np.float32)
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
    present = np.asarray([sample_id in lookup for sample_id in sample_ids], dtype=bool)
    imu_logits = np.zeros_like(sd_logits, dtype=np.float32)
    fused_logits = np.empty_like(sd_logits, dtype=np.float32)

    # Missing IMU has an explicit, fold-external calibrated S+D fallback.
    for fold in range(3):
        target = folds == fold
        fused_logits[target] = (
            sd_logits[target] / protocols[fold]["sd_temperature"]
        )

    for target_index, sample_id in enumerate(sample_ids):
        if not present[target_index]:
            continue
        source_index = lookup[sample_id]
        if int(source_labels[source_index]) != int(labels[target_index]):
            raise ValueError(f"IMU label mismatch: {sample_id}")
        if int(source_folds[source_index]) != int(folds[target_index]):
            raise ValueError(f"IMU fold mismatch: {sample_id}")
        imu_logits[target_index] = source_imu[source_index]
        # Preserve the already audited P3 fusion on all rows where it exists.
        fused_logits[target_index] = source_fused[source_index]
    return present, imu_logits, fused_logits


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
            "macro_f1": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
        }


def summarize(
    labels: np.ndarray,
    folds: np.ndarray,
    logits: np.ndarray,
) -> dict[str, object]:
    predictions = logits.argmax(1)
    small = np.isin(labels, SMALL_ACTION_IDS)
    return {
        "all": metrics(labels, predictions),
        "small_actions": metrics(labels[small], predictions[small]),
        "per_fold": {
            str(fold): metrics(
                labels[folds == fold],
                predictions[folds == fold],
            )
            for fold in range(3)
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sd = load_sd(args.sd_root.resolve())
    sample_ids = sd["sample_ids"].astype(str)
    labels = sd["labels"].astype(np.int64)
    folds = sd["folds"].astype(np.int64)
    skeleton_logits = sd["skeleton_logits"].astype(np.float32)
    depth_logits = sd["depth_logits"].astype(np.float32)
    sd_logits = 0.6 * skeleton_logits + 0.4 * depth_logits
    if len(sample_ids) != 2914:
        raise ValueError(f"Expected 2914 S+D rows, got {len(sample_ids)}")

    manifest = load_manifest(args.manifest.resolve())
    if set(sample_ids.tolist()) != set(manifest):
        missing = sorted(set(manifest) - set(sample_ids.tolist()))
        extra = sorted(set(sample_ids.tolist()) - set(manifest))
        raise ValueError(
            f"S+D/manifest mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    imu_protocols = load_imu_protocols(args.imu_summary.resolve())
    imu_present, imu_logits, sd_imu_logits = align_imu(
        sample_ids,
        labels,
        folds,
        sd_logits,
        args.imu_oof.resolve(),
        imu_protocols,
    )

    thermal = load_thermal(
        args.thermal_root.resolve(),
        "val_logits_fp16.npz",
    )
    thermal_present, thermal_logits = align_thermal(sample_ids, labels, thermal)
    thermal_result = cross_fitted_thermal_residual(
        "complete_sd_imu_plus_thermal",
        sample_ids,
        labels,
        folds,
        sd_imu_logits,
        thermal,
    )
    thermal_candidate_logits = thermal_result["logits"].astype(np.float32)

    router = route_cross_fitted(
        labels,
        folds,
        sd_imu_logits,
        thermal_candidate_logits,
        thermal_present,
        include_class=True,
    )
    route_to_thermal = router["route_candidate"].astype(bool)
    final_logits = sd_imu_logits.copy()
    final_logits[route_to_thermal] = thermal_candidate_logits[route_to_thermal]
    final_predictions = final_logits.argmax(1)

    np.savez_compressed(
        output_dir / "complete_oof.npz",
        sample_ids=sample_ids,
        labels=labels,
        folds=folds,
        skeleton_logits=skeleton_logits.astype(np.float16),
        depth_logits=depth_logits.astype(np.float16),
        sd_logits=sd_logits.astype(np.float16),
        imu_logits=imu_logits.astype(np.float16),
        imu_present=imu_present.astype(np.uint8),
        sd_imu_logits=sd_imu_logits.astype(np.float16),
        thermal_logits=thermal_logits.astype(np.float16),
        thermal_present=thermal_present.astype(np.uint8),
        thermal_candidate_logits=thermal_candidate_logits.astype(np.float16),
        route_probability=np.asarray(router["probability"], dtype=np.float32),
        route_to_thermal=route_to_thermal.astype(np.uint8),
        final_logits=final_logits.astype(np.float16),
        final_predictions=final_predictions.astype(np.int64),
    )

    fieldnames = [
        "sample_id",
        "fold",
        "class_id",
        "class_name",
        "user_id",
        "trial_id",
        "imu_present",
        "thermal_present",
        "sd_prediction",
        "sd_imu_prediction",
        "thermal_candidate_prediction",
        "route_to_thermal",
        "route_probability",
        "final_prediction",
        "final_correct",
        "final_confidence",
        "final_margin",
    ]
    shifted = final_logits - final_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    top2 = np.partition(probabilities, -2, axis=1)[:, -2:]
    with (output_dir / "complete_oof_rows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            row = manifest[sample_id]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "fold": int(folds[index]),
                    "class_id": int(labels[index]),
                    "class_name": row["class_name"],
                    "user_id": row["user_id"],
                    "trial_id": row["trial_id"],
                    "imu_present": int(imu_present[index]),
                    "thermal_present": int(thermal_present[index]),
                    "sd_prediction": int(sd_logits[index].argmax()),
                    "sd_imu_prediction": int(sd_imu_logits[index].argmax()),
                    "thermal_candidate_prediction": int(
                        thermal_candidate_logits[index].argmax()
                    ),
                    "route_to_thermal": int(route_to_thermal[index]),
                    "route_probability": float(router["probability"][index]),
                    "final_prediction": int(final_predictions[index]),
                    "final_correct": int(final_predictions[index] == labels[index]),
                    "final_confidence": float(probabilities[index].max()),
                    "final_margin": float(top2[index, 1] - top2[index, 0]),
                }
            )

    common_imu = imu_present
    baseline_predictions = sd_imu_logits.argmax(1)
    summary = {
        "protocol": (
            "Complete subject-disjoint OOF over all 2914 Depth+Skeleton samples. "
            "Existing audited S+D+IMU logits are preserved where IMU is available; "
            "missing IMU falls back to fold-external temperature-scaled S+D. "
            "Thermal weight/temperatures and the scalar+predicted-class router are "
            "re-fitted on the other two folds for every held fold."
        ),
        "counts": {
            "samples": int(len(labels)),
            "imu_present": int(imu_present.sum()),
            "imu_missing": int((~imu_present).sum()),
            "thermal_present": int(thermal_present.sum()),
            "thermal_missing": int((~thermal_present).sum()),
            "both_present": int((imu_present & thermal_present).sum()),
            "routed_to_thermal": int(route_to_thermal.sum()),
        },
        "metrics": {
            "sd": summarize(labels, folds, sd_logits),
            "sd_imu_missing_aware": summarize(labels, folds, sd_imu_logits),
            "thermal_candidate": summarize(
                labels, folds, thermal_candidate_logits
            ),
            "thermal_routed_final": summarize(labels, folds, final_logits),
        },
        "imu_common_subset": {
            "samples": int(common_imu.sum()),
            "sd_imu_accuracy": float(
                accuracy_score(
                    labels[common_imu],
                    baseline_predictions[common_imu],
                )
            ),
            "routed_final_accuracy": float(
                accuracy_score(
                    labels[common_imu],
                    final_predictions[common_imu],
                )
            ),
        },
        "thermal_protocols": thermal_result["protocols"],
        "router_protocols": router["protocols"],
        "routed_vs_sd_imu": bootstrap_delta(
            labels,
            final_predictions,
            baseline_predictions,
            sample_ids,
            args.bootstrap_repeats,
            args.seed,
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
