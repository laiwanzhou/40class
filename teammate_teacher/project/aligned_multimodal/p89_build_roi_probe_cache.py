from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_FEATURE_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p89_roi_probe_cache_v1"

# The three crops with the clearest object/hand semantics.  The full-body and
# global crops are deliberately excluded: P89 is a complementary local expert,
# not another copy of the P87 visual branch.
SELECTED_REGIONS = ("left_hand", "right_hand", "hand_workspace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact P89 trial-level local-ROI probe features.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-run", type=Path, default=DEFAULT_FEATURE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_moments(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.maximum(weight.astype(np.float32), 1e-3)
    weight = weight / np.maximum(weight.sum(), 1e-6)
    mean = np.sum(values * weight[:, None], axis=0)
    variance = np.sum(np.square(values - mean) * weight[:, None], axis=0)
    return mean, np.sqrt(np.maximum(variance, 1e-8))


def aggregate(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    with np.load(path, allow_pickle=False) as source:
        features = source["features"].astype(np.float32)
        region_names = source["region_names"].astype(str).tolist()
        quality = source["roi_quality"].astype(np.float32)
        valid = source["roi_valid"].astype(np.float32)

    region_indices = [region_names.index(name) for name in SELECTED_REGIONS]
    vectors: list[np.ndarray] = []
    for modality in range(features.shape[1]):
        for region in region_indices:
            values = features[:, modality, region]
            # Per-frame direction removes much of the subject/background scale
            # while retaining the ImageNet semantic direction of each crop.
            normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
            weight = quality[:, region] * valid[:, region]
            mean, std = weighted_moments(normalized, weight)
            midpoint = max(1, len(normalized) // 2)
            early = normalized[:midpoint].mean(axis=0)
            late = normalized[midpoint:].mean(axis=0) if midpoint < len(normalized) else early
            vectors.extend((mean, std, late - early))

    diagnostics = {
        "frames": float(features.shape[0]),
        "mean_selected_roi_quality": float(quality[:, region_indices].mean()),
        "valid_selected_roi_rate": float(valid[:, region_indices].mean()),
    }
    return np.concatenate(vectors).astype(np.float32), diagnostics


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_root = args.feature_run.resolve() / "trial_feature_cache"
    rows = [
        row
        for row in read_rows(args.manifest.resolve())
        if row["split"] == "train"
        and feature_root.joinpath(*row["sample_id"].split("/")).with_suffix(".npz").is_file()
    ]

    sample_ids: list[str] = []
    labels: list[int] = []
    users: list[str] = []
    trial_features: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        path = feature_root.joinpath(*sample_id.split("/")).with_suffix(".npz")
        vector, diagnostic = aggregate(path)
        sample_ids.append(sample_id)
        labels.append(int(row["class_id"]))
        users.append(row["user_id"])
        trial_features.append(vector)
        diagnostics.append(diagnostic)
        if index % 250 == 0 or index == len(rows):
            print(f"aggregated {index}/{len(rows)}", flush=True)

    matrix = np.stack(trial_features).astype(np.float16)
    np.savez_compressed(
        output / "roi_probe_features.npz",
        sample_ids=np.asarray(sample_ids),
        labels=np.asarray(labels, dtype=np.int64),
        users=np.asarray(users),
        features=matrix,
    )
    summary = {
        "stage": "P89_local_ROI_probe_cache_v1",
        "status": "complete",
        "source_feature_run": str(args.feature_run.resolve()),
        "samples": len(rows),
        "feature_dim": int(matrix.shape[1]),
        "selected_regions": list(SELECTED_REGIONS),
        "modalities": ["depth", "ir"],
        "statistics": ["quality_weighted_l2_normalized_mean", "temporal_std", "late_minus_early"],
        "mean_frames": float(np.mean([item["frames"] for item in diagnostics])),
        "mean_selected_roi_quality": float(np.mean([item["mean_selected_roi_quality"] for item in diagnostics])),
        "mean_valid_selected_roi_rate": float(np.mean([item["valid_selected_roi_rate"] for item in diagnostics])),
        "storage_dtype": str(matrix.dtype),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
