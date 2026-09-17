from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from aligned_data import frame_map
from aligned_model import AlignedMultimodalModel
from local_roi_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    sample_positions,
    standardize_box,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_REVIEW = (
    PROJECT_DIR
    / "data"
    / "local_roi_annotation_v2"
    / "blind60_review_annotations.csv"
)
DEFAULT_LOCAL_ROOT = PROJECT_DIR / "runs" / "p12_local_depth_oof_v2"
DEFAULT_BASE = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p12_reviewed60_manual_roi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use held-fold Local models to compare manual vs locator ROI on reviewed60"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--base-oof", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ReviewedManualROIDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        review_path: Path,
        held_fold: int,
    ) -> None:
        manifest = {row["sample_id"]: row for row in read_csv(manifest_path)}
        self.rows = [
            row for row in read_csv(review_path) if int(row["fold"]) == held_fold
        ]
        self.rows.sort(key=lambda row: row["sample_id"])
        self.manifest = manifest
        self.held_fold = held_fold

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        review = self.rows[index]
        sample_id = review["sample_id"]
        row = self.manifest[sample_id]
        maps = {
            "depth": frame_map(Path(row["depth_dir"]), "depth"),
            "ir": frame_map(Path(row["ir_dir"]), "ir"),
            "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
        }
        common_ids = sorted(set.intersection(*(set(value) for value in maps.values())))
        frame_ids = [
            common_ids[position]
            for position in sample_positions(len(common_ids), 12, augment=False)
        ]
        raw_box = tuple(
            float(review[key]) for key in ("x0", "y0", "x1", "y1")
        )
        frames: list[torch.Tensor] = []
        raw_size: tuple[int, int] | None = None
        crop_box: tuple[int, int, int, int] | None = None
        for frame_id in frame_ids:
            with Image.open(maps["depth"][frame_id]) as image:
                image = image.convert("RGB")
                if raw_size is None:
                    raw_size = image.size
                    crop_box = standardize_box(
                        raw_box,
                        raw_size[0],
                        raw_size[1],
                        context=0.15,
                    )
                elif image.size != raw_size:
                    raise ValueError(f"{sample_id}: inconsistent Depth sizes")
                assert crop_box is not None
                image = image.crop(crop_box).resize(
                    (192, 144), Image.Resampling.BILINEAR
                )
                array = np.asarray(image, dtype=np.uint8).copy()
            frames.append(
                torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            )
        tensor = torch.stack(frames)
        tensor = (tensor - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[
            None, :, None, None
        ]
        return {
            "depth": tensor,
            "label": int(review["class_id"]),
            "sample_id": sample_id,
            "machine_box_assessment": review["machine_box_assessment"],
            "final_box_source": review["final_box_source"],
            "original_fallback": int(review["original_fallback"]),
        }


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


def infer_fold(
    dataset: ReviewedManualROIDataset,
    checkpoint_path: Path,
    batch_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AlignedMultimodalModel(["depth"], num_classes=40, dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    assessment: list[str] = []
    source: list[str] = []
    fallback: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            depth = batch["depth"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model({"depth": depth})
            logits.append(output.float().cpu().numpy())
            labels.append(batch["label"].numpy())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            assessment.extend(str(value) for value in batch["machine_box_assessment"])
            source.extend(str(value) for value in batch["final_box_source"])
            fallback.extend(int(value) for value in batch["original_fallback"])
    return {
        "sample_ids": np.asarray(sample_ids),
        "labels": np.concatenate(labels),
        "logits": np.concatenate(logits),
        "machine_box_assessment": np.asarray(assessment),
        "final_box_source": np.asarray(source),
        "original_fallback": np.asarray(fallback, dtype=np.int64),
        "held_fold": np.full(len(sample_ids), dataset.held_fold, dtype=np.int64),
    }


def grouped_summary(
    labels: np.ndarray,
    locator_predictions: np.ndarray,
    manual_predictions: np.ndarray,
    group: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in sorted(set(group.tolist()), key=str):
        selected = group == value
        locator = locator_predictions[selected]
        manual = manual_predictions[selected]
        selected_labels = labels[selected]
        result[str(value)] = {
            "samples": int(selected.sum()),
            "locator_roi": metrics(selected_labels, locator),
            "manual_roi": metrics(selected_labels, manual),
            "manual_accuracy_delta_pp": float(
                100
                * (
                    accuracy_score(selected_labels, manual)
                    - accuracy_score(selected_labels, locator)
                )
            ),
            "locator_wrong_manual_right": int(
                np.sum((locator != selected_labels) & (manual == selected_labels))
            ),
            "locator_right_manual_wrong": int(
                np.sum((locator == selected_labels) & (manual != selected_labels))
            ),
            "prediction_changed": int(np.sum(locator != manual)),
        }
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    parts = []
    for held_fold in range(3):
        dataset = ReviewedManualROIDataset(
            args.manifest.resolve(),
            args.review.resolve(),
            held_fold,
        )
        parts.append(
            infer_fold(
                dataset,
                args.local_root.resolve() / f"fold_{held_fold}" / "best.pt",
                int(args.batch_size),
                int(args.workers),
            )
        )
    manual = {
        key: np.concatenate([part[key] for part in parts])
        for key in parts[0]
    }
    order = np.argsort(manual["sample_ids"])
    manual = {key: value[order] for key, value in manual.items()}
    with np.load(
        args.local_root.resolve() / "oof_logits.npz", allow_pickle=False
    ) as local:
        local_lookup = {
            sample_id: index
            for index, sample_id in enumerate(local["sample_ids"].astype(str))
        }
        indices = np.asarray(
            [local_lookup[sample_id] for sample_id in manual["sample_ids"]],
            dtype=np.int64,
        )
        locator_logits = local["logits"][indices].astype(np.float32)
        if not np.array_equal(local["labels"][indices], manual["labels"]):
            raise ValueError("Manual/locator labels differ")
    with np.load(args.base_oof.resolve(), allow_pickle=False) as base:
        base_lookup = {
            sample_id: index
            for index, sample_id in enumerate(base["sample_ids"].astype(str))
        }
        base_indices = np.asarray(
            [base_lookup[sample_id] for sample_id in manual["sample_ids"]],
            dtype=np.int64,
        )
        final_predictions = base["final_logits"][base_indices].argmax(1)

    labels = manual["labels"].astype(np.int64)
    locator_predictions = locator_logits.argmax(1)
    manual_predictions = manual["logits"].argmax(1)
    summary = {
        "protocol": (
            "Each reviewed sample is inferred with the Local classifier from its "
            "held subject fold. The classifier never trained on that subject fold. "
            "Manual ROI uses the final second-review box; locator ROI uses the "
            "strict fold-pure locator output. Frame sampling and preprocessing are identical."
        ),
        "samples": int(len(labels)),
        "locator_roi": metrics(labels, locator_predictions),
        "manual_roi": metrics(labels, manual_predictions),
        "manual_accuracy_delta_pp": float(
            100
            * (
                accuracy_score(labels, manual_predictions)
                - accuracy_score(labels, locator_predictions)
            )
        ),
        "locator_wrong_manual_right": int(
            np.sum((locator_predictions != labels) & (manual_predictions == labels))
        ),
        "locator_right_manual_wrong": int(
            np.sum((locator_predictions == labels) & (manual_predictions != labels))
        ),
        "prediction_changed": int(np.sum(locator_predictions != manual_predictions)),
        "current_final_accuracy": float(accuracy_score(labels, final_predictions)),
        "groups": {
            "machine_box_assessment": grouped_summary(
                labels,
                locator_predictions,
                manual_predictions,
                manual["machine_box_assessment"],
            ),
            "final_box_source": grouped_summary(
                labels,
                locator_predictions,
                manual_predictions,
                manual["final_box_source"],
            ),
            "original_fallback": grouped_summary(
                labels,
                locator_predictions,
                manual_predictions,
                manual["original_fallback"],
            ),
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "manual_vs_locator_logits.npz",
        sample_ids=manual["sample_ids"],
        labels=labels,
        folds=manual["held_fold"],
        locator_logits=locator_logits,
        manual_logits=manual["logits"].astype(np.float32),
        machine_box_assessment=manual["machine_box_assessment"],
        final_box_source=manual["final_box_source"],
        original_fallback=manual["original_fallback"],
    )
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "fold",
                "label",
                "locator_prediction",
                "manual_prediction",
                "current_final_prediction",
                "machine_box_assessment",
                "final_box_source",
                "original_fallback",
            ]
        )
        writer.writerows(
            zip(
                manual["sample_ids"],
                manual["held_fold"],
                labels,
                locator_predictions,
                manual_predictions,
                final_predictions,
                manual["machine_box_assessment"],
                manual["final_box_source"],
                manual["original_fallback"],
            )
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
