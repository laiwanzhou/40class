from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel
from probe_p27r3_incremental_information import metric_bundle


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a label-free wide IR Local view with a frozen fold-pure "
            "IR+Skeleton encoder."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--motion-boxes",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p12_fold_pure_locator_predictions"
        / "motion_boxes_all2914.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class MotionBoxDataset(AlignedMultimodalDataset):
    def __init__(
        self,
        *,
        motion_boxes: Path,
        padding: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.padding = float(padding)
        self.boxes = {
            row["sample_id"]: (
                np.asarray(
                    [
                        float(row["x0"]) / 640.0,
                        float(row["y0"]) / 480.0,
                        float(row["x1"]) / 640.0,
                        float(row["y1"]) / 480.0,
                    ],
                    dtype=np.float32,
                )
                if int(row["fallback"]) == 0
                else np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
            )
            for row in read_csv(motion_boxes)
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        result = super().__getitem__(index)
        sample_id = str(result["sample_id"])
        result["user_id"] = self.samples[index].user_id
        box = self.boxes.get(
            sample_id, np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        ).copy()
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        box[[0, 2]] += np.asarray([-1.0, 1.0]) * self.padding * width
        box[[1, 3]] += np.asarray([-1.0, 1.0]) * self.padding * height
        result["roi"] = torch.from_numpy(np.clip(box, 0.0, 1.0))
        return result


def build_model(checkpoint: dict, device: torch.device) -> AlignedMultimodalModel:
    config = checkpoint["config"]
    model = AlignedMultimodalModel(
        list(config["modalities"]),
        dropout=float(config["dropout"]),
        use_aux_heads=bool(config.get("use_aux_heads", False)),
        use_modality_masks=bool(config.get("use_modality_masks", False)),
        use_cross_attention=bool(config.get("use_cross_attention", False)),
        cross_attention_grid=tuple(config.get("cross_attention_grid", [2, 3])),
        cross_attention_heads=int(config.get("cross_attention_heads", 4)),
        depth_input_channels=int(config.get("depth_input_channels", 3)),
        use_layer3_spatial=bool(config.get("use_layer3_spatial", False)),
        imagenet_pretrained=False,
        ir_stem_initialization=str(config.get("ir_stem_initialization", "mean")),
        visual_stem_fusion=str(config.get("visual_stem_fusion", "concat")),
        skeleton_input_dim=int(config.get("skeleton_input_dim", 4)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def crop_ir(ir: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    batch_size, time_steps, channels, height, width = ir.shape
    repeated = boxes[:, None].expand(batch_size, time_steps, 4).reshape(-1, 4)
    scaled = repeated.to(device=ir.device, dtype=ir.dtype).clone()
    scaled[:, [0, 2]] *= float(width)
    scaled[:, [1, 3]] *= float(height)
    indices = torch.arange(
        batch_size * time_steps, device=ir.device, dtype=ir.dtype
    ).unsqueeze(1)
    rois = torch.cat([indices, scaled], dim=1)
    local = roi_align(
        ir.reshape(batch_size * time_steps, channels, height, width),
        rois,
        output_size=(height, width),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    return local.reshape(batch_size, time_steps, channels, height, width)


@torch.no_grad()
def extract(
    model: AlignedMultimodalModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    sample_ids: list[str] = []
    labels: list[np.ndarray] = []
    subjects: list[str] = []
    full_features: list[np.ndarray] = []
    local_features: list[np.ndarray] = []
    skeleton_features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    assert model.visual is not None and model.skeleton is not None
    for batch in loader:
        ir = batch["ir"].to(device, non_blocking=True)
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        local_ir = crop_ir(ir, batch["roi"])
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            full = model.visual({"ir": ir})
            local = model.visual({"ir": local_ir})
            body = model.skeleton(skeleton)
            output = model({"ir": ir, "skeleton": skeleton})
        full_features.append(full.float().cpu().numpy())
        local_features.append(local.float().cpu().numpy())
        skeleton_features.append(body.float().cpu().numpy())
        logits.append(output.float().cpu().numpy())
        labels.append(batch["label"].numpy())
        sample_ids.extend(str(value) for value in batch["sample_id"])
        subjects.extend(str(value) for value in batch["user_id"])
    return {
        "sample_ids": np.asarray(sample_ids),
        "labels": np.concatenate(labels).astype(np.int64),
        "subjects": np.asarray(subjects),
        "full": np.concatenate(full_features).astype(np.float32),
        "local": np.concatenate(local_features).astype(np.float32),
        "skeleton": np.concatenate(skeleton_features).astype(np.float32),
        "base_logits": np.concatenate(logits).astype(np.float32),
    }


def features(part: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            part["full"],
            part["local"],
            part["local"] - part["full"],
            part["skeleton"],
        ],
        axis=1,
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if list(config["modalities"]) != ["ir", "skeleton"]:
        raise ValueError("checkpoint must be an IR+Skeleton model")
    common = {
        "manifest_path": args.manifest.resolve(),
        "modalities": ["ir", "skeleton"],
        "num_frames": int(config["num_frames"]),
        "image_height": int(config["image_height"]),
        "image_width": int(config["image_width"]),
        "cache_dir": config.get("cache_dir"),
        "skeleton_strategy": config.get("skeleton_strategy", "first"),
        "visual_normalization": config.get("visual_normalization", "legacy"),
        "skeleton_representation": config.get(
            "skeleton_representation", "frame_joint"
        ),
        "skeleton_raw_cache_dir": config.get("skeleton_raw_cache_dir"),
        "motion_boxes": args.motion_boxes.resolve(),
        "padding": args.padding,
        "augment": False,
    }
    train_dataset = MotionBoxDataset(split="train", **common)
    held_dataset = MotionBoxDataset(split="val", **common)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device)
    train = extract(
        model,
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        device,
    )
    held = extract(
        model,
        DataLoader(
            held_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        device,
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.10,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=27083,
        ),
    )
    classifier.fit(features(train), train["labels"])
    held_logits = classifier.decision_function(features(held)).astype(np.float32)
    methods = {
        "frozen_base": held["base_logits"],
        "wide_local_linear": held_logits,
    }
    metrics = {
        name: metric_bundle(held["labels"], values.argmax(axis=1))
        for name, values in methods.items()
    }
    np.savez_compressed(
        output / "held_logits.npz",
        sample_ids=held["sample_ids"],
        labels=held["labels"],
        subjects=held["subjects"],
        base_logits=held["base_logits"],
        local_logits=held_logits,
        outer_held_predictions_generated=np.asarray(False),
    )
    joblib.dump(classifier, output / "wide_local_linear.joblib")
    summary = {
        "protocol": "outer-fold-0 train subjects only; one subject-disjoint inner fold",
        "outer_held_predictions_generated": False,
        "checkpoint": str(checkpoint_path),
        "padding": float(args.padding),
        "motion_box_fallback_policy": "full frame; no learned locator prediction",
        "feature_dim": int(features(train).shape[1]),
        "train_samples": int(len(train["labels"])),
        "held_samples": int(len(held["labels"])),
        "metrics": metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
