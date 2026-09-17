from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
MODALITIES = ("depth_color", "skeleton", "imu", "thermal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only feasibility audit for pooled multimodal feature fusion. "
            "The script does not train, modify checkpoints, or write caches."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output. Omit this argument for a strictly read-only run.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_union(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def usable(row: dict[str, str], modality: str) -> bool:
    return row[f"{modality}_usable"] == "1"


def availability_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = {
        modality: sum(usable(row, modality) for row in rows)
        for modality in MODALITIES
    }
    all_four = sum(
        all(usable(row, modality) for modality in MODALITIES) for row in rows
    )
    pattern_counts = Counter(
        "".join("1" if usable(row, modality) else "0" for modality in MODALITIES)
        for row in rows
    )
    return {
        "samples": len(rows),
        "usable": counts,
        "all_four_usable": all_four,
        "all_four_missing_or_unusable": len(rows) - all_four,
        "pattern_order": list(MODALITIES),
        "usable_patterns": dict(sorted(pattern_counts.items())),
    }


def parsed_imu_availability(
    index_path: Path, union_rows: list[dict[str, str]], split: str
) -> dict[str, int]:
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        imu_rows = list(csv.DictReader(handle))
    imu_ids = {
        row["source_sample_id"]
        for row in imu_rows
        if row["split"] == split and row["usable"] == "1"
    }
    thermal_ids = {
        row["sample_id"] for row in union_rows if usable(row, "thermal")
    }
    return {
        "imu_parser_usable": len(imu_ids),
        "thermal_manifest_usable": len(thermal_ids),
        "imu_and_thermal_usable": len(imu_ids & thermal_ids),
    }


def state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    result = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(result, dict):
        raise TypeError("Checkpoint has no model_state_dict/state_dict")
    return result


def checkpoint_summary(path: Path, modality: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = state_dict(checkpoint)
    classifier_key = (
        "classifier.1.weight" if modality == "imu" else "classifier.2.weight"
    )
    classifier_weight = state[classifier_key]
    if modality == "imu":
        # Five 64-D device embeddings are flattened before the 320->128 classifier.
        pooled_embedding_dim = int(classifier_weight.shape[1])
        token_shape = None
    elif modality == "skeleton":
        pooled_embedding_dim = int(classifier_weight.shape[1])
        token_shape = [12, 256]
    else:
        pooled_embedding_dim = int(classifier_weight.shape[1])
        token_shape = [12, 512]
    config = checkpoint.get("config", {})
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_mib": path.stat().st_size / 1024**2,
        "epoch": checkpoint.get("epoch"),
        "config_fold": config.get("fold"),
        "train_users": config.get("train_users"),
        "val_users": config.get("val_users"),
        "pooled_embedding_shape": ["B", pooled_embedding_dim],
        "available_temporal_token_shape": (
            ["B", *token_shape] if token_shape is not None else None
        ),
        "has_formal_embedding_api": modality == "skeleton",
        "classifier_weight_shape": list(classifier_weight.shape),
    }


def id_index(values: np.ndarray) -> dict[str, int]:
    return {value: index for index, value in enumerate(values.astype(str).tolist())}


def assert_unique(name: str, values: np.ndarray) -> None:
    text = values.astype(str)
    if len(np.unique(text)) != len(text):
        raise ValueError(f"{name} contains duplicate sample_ids")


def intersection_audit(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    candidate_fold_key: str,
) -> dict[str, Any]:
    assert_unique("reference", reference["sample_ids"])
    assert_unique("candidate", candidate["sample_ids"])
    reference_lookup = id_index(reference["sample_ids"])
    candidate_lookup = id_index(candidate["sample_ids"])
    common = sorted(set(reference_lookup) & set(candidate_lookup))
    label_mismatches = 0
    fold_mismatches = 0
    for sample_id in common:
        reference_index = reference_lookup[sample_id]
        candidate_index = candidate_lookup[sample_id]
        label_mismatches += int(
            reference["labels"][reference_index] != candidate["labels"][candidate_index]
        )
        fold_mismatches += int(
            reference["folds"][reference_index]
            != candidate[candidate_fold_key][candidate_index]
        )
    return {
        "reference_samples": len(reference["sample_ids"]),
        "candidate_samples": len(candidate["sample_ids"]),
        "intersection": len(common),
        "reference_only": len(reference_lookup) - len(common),
        "candidate_only": len(candidate_lookup) - len(common),
        "label_mismatches": label_mismatches,
        "fold_mismatches": fold_mismatches,
    }


def thermal_oof_audit(reference: dict[str, np.ndarray]) -> dict[str, Any]:
    pieces: list[dict[str, np.ndarray]] = []
    per_fold: list[dict[str, Any]] = []
    for fold in range(3):
        path = (
            REPO_ROOT
            / "thermal_baseline"
            / "runs"
            / "p11_thermal_imagenet_fp16"
            / f"fold_{fold}"
            / "val_logits_fp16.npz"
        )
        piece = load_npz(path)
        piece["held_fold"] = np.full(
            len(piece["sample_ids"]), fold, dtype=np.int64
        )
        pieces.append(piece)
        per_fold.append(
            {
                "fold": fold,
                "samples": len(piece["sample_ids"]),
                "path": str(path.resolve()),
            }
        )
    combined = {
        "sample_ids": np.concatenate([piece["sample_ids"] for piece in pieces]),
        "labels": np.concatenate([piece["labels"] for piece in pieces]),
        "held_fold": np.concatenate([piece["held_fold"] for piece in pieces]),
    }
    return {
        "per_fold": per_fold,
        "alignment": intersection_audit(reference, combined, "held_fold"),
    }


def cache_estimate(
    samples: int, embedding_dims: list[int], folds: int = 3
) -> dict[str, float | int]:
    elements = samples * sum(embedding_dims)
    return {
        "samples": samples,
        "fold_conditioned_caches": folds,
        "raw_embedding_dims": embedding_dims,
        "raw_embedding_dim_total": sum(embedding_dims),
        "one_cache_fp16_mib": elements * 2 / 1024**2,
        "three_cache_fp16_mib": elements * 2 * folds / 1024**2,
        "three_cache_fp32_mib": elements * 4 * folds / 1024**2,
    }


def main() -> None:
    args = parse_args()
    train_rows = read_union(
        PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
    )
    test_rows = read_union(
        PROJECT_DIR / "data" / "six_modality_audit" / "test_union_manifest.csv"
    )
    reference = load_npz(
        PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
    )
    student = load_npz(
        PROJECT_DIR / "runs" / "p20_tiny_imu_student_oof" / "oof_logits.npz"
    )

    checkpoint_roots = {
        "skeleton": (
            PROJECT_DIR
            / "runs"
            / "p11_fp16_oof"
            / "fold_0"
            / "skeleton_best_accuracy_fp16.pt"
        ),
        "depth": (
            PROJECT_DIR
            / "runs"
            / "p11_fp16_oof"
            / "fold_0"
            / "depth_best_accuracy_fp16.pt"
        ),
        "thermal": (
            REPO_ROOT
            / "thermal_baseline"
            / "runs"
            / "p11_thermal_imagenet_fp16"
            / "fold_0"
            / "best_accuracy_fp16.pt"
        ),
        "imu": (
            PROJECT_DIR
            / "runs"
            / "p20_tiny_imu_student_oof"
            / "fold_0"
            / "final.pt"
        ),
    }
    checkpoints = {
        modality: checkpoint_summary(path, modality)
        for modality, path in checkpoint_roots.items()
    }
    embedding_dims = [
        int(checkpoints[modality]["pooled_embedding_shape"][1])
        for modality in ("skeleton", "depth", "thermal", "imu")
    ]

    reference_ids = reference["sample_ids"].astype(str)
    report = {
        "status": "read_only_audit_complete",
        "training_started": False,
        "availability": {
            "train_union": availability_summary(train_rows),
            "test_union": availability_summary(test_rows),
            "parser_level": {
                "train": parsed_imu_availability(
                    PROJECT_DIR / "cache" / "imu_32" / "index.csv",
                    train_rows,
                    "train",
                ),
                "test": parsed_imu_availability(
                    PROJECT_DIR / "cache" / "imu_32" / "index.csv",
                    test_rows,
                    "test",
                ),
                "warning": (
                    "The union manifest checks IMU file presence, while the IMU "
                    "cache index checks successful parsing. Deployment masks must "
                    "use parser-level availability."
                ),
            },
            "p12_visual_base": {
                "samples": len(reference_ids),
                "unique_sample_ids": len(np.unique(reference_ids)),
                "imu_present": int(reference["imu_present"].sum()),
                "thermal_present": int(reference["thermal_present"].sum()),
                "imu_and_thermal_present": int(
                    np.logical_and(
                        reference["imu_present"], reference["thermal_present"]
                    ).sum()
                ),
                "label_range": [
                    int(reference["labels"].min()),
                    int(reference["labels"].max()),
                ],
                "fold_counts": {
                    str(key): int(value)
                    for key, value in zip(
                        *np.unique(reference["folds"], return_counts=True)
                    )
                },
            },
        },
        "checkpoints_fold0": checkpoints,
        "oof_alignment": {
            "tiny_imu_vs_p12": intersection_audit(
                reference, student, "held_fold"
            ),
            "thermal_vs_p12": thermal_oof_audit(reference),
        },
        "cache_estimate": cache_estimate(len(reference_ids), embedding_dims),
        "protocol_warning": (
            "One OOF embedding per sample is not sufficient for an unbiased "
            "cross-fitted fusion head. For each outer fold k, checkpoint k must "
            "encode both the outer-train rows and held-fold rows, producing three "
            "fold-conditioned full-sample caches."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.resolve().write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
