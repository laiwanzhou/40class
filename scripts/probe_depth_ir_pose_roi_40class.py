from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_unimodal import build_datasets, build_model, load_config, set_seed


def config_args() -> argparse.Namespace:
    return argparse.Namespace(
        data_root=None, manifest=None, fold=None, output_root=None, device=None, seed=None,
        smoke_test=False, max_epochs=None, num_workers=0, max_train_batches=None,
        max_val_batches=None, run_id=None,
    )


def roi_quality(datasets: list[object]) -> None:
    detection = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_pose_quality.csv")
    accumulators: dict[int, dict[str, object]] = {}
    for dataset in datasets:
        for sample in dataset.samples:
            class_id = int(sample["original_class_id"])
            values = accumulators.setdefault(
                class_id,
                {"upper_fallback": 0, "left_fallback": 0, "right_fallback": 0, "frames": 0, "areas": [0.0, 0.0, 0.0]},
            )
            sources = sample["roi_sources"]
            boxes = sample["roi_boxes"]
            path = sample["paths"][0]
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
            frames = len(sources)
            values["frames"] += frames
            values["upper_fallback"] += int(np.sum(sources[:, 0] != "keypoints"))
            values["left_fallback"] += int(np.sum(~np.isin(sources[:, 1], ["wrist", "interpolated_wrist"])))
            values["right_fallback"] += int(np.sum(~np.isin(sources[:, 2], ["wrist", "interpolated_wrist"])))
            area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]) / (width * height)
            for view in range(3):
                values["areas"][view] += float(area[:, view].sum())
    rows = []
    for class_id, values in sorted(accumulators.items()):
        frames = int(values["frames"])
        detector = detection.loc[detection["class_id"] == class_id].iloc[0]
        rows.append(
            {
                **detector.to_dict(),
                "upper_body_fallback": values["upper_fallback"] / frames,
                "left_hand_fallback": values["left_fallback"] / frames,
                "right_hand_fallback": values["right_fallback"] / frames,
                "upper_body_area_ratio": values["areas"][0] / frames,
                "left_hand_area_ratio": values["areas"][1] / frames,
                "right_hand_area_ratio": values["areas"][2] / frames,
            }
        )
    frame_weights = np.asarray([row["frames"] for row in rows], dtype=np.float64)
    overall = {"class_id": -1, "action_name": "ALL", "frames": int(frame_weights.sum())}
    metric_columns = [column for column in rows[0] if column not in {"class_id", "action_name", "frames"}]
    for column in metric_columns:
        overall[column] = float(np.average([row[column] for row in rows], weights=frame_weights))
    pd.DataFrame([overall, *rows]).to_csv(
        PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_pose_quality.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs/experiments/depth_ir_pose_roi_40class.yaml", config_args())
    set_seed(int(config["seed"]))
    train_dataset, val_dataset = build_datasets(config)
    roi_quality([train_dataset, val_dataset])
    if train_dataset.original_class_ids != list(range(40)) or val_dataset.original_class_ids != list(range(40)):
        raise ValueError("Original class mapping is not exactly 0-39.")
    sample = train_dataset[0]
    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model = build_model(config, sample).cuda().train()
    reference = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    checks: dict[str, object] = {
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "all_40_classes_present": len(train_dataset.class_names) == 40,
        "depth_stem_exact_pretrained": bool(torch.equal(model.depth_stem[0].weight.detach().cpu(), reference.features[0][0].weight)),
        "ir_stem_exact_rgb_mean": bool(torch.equal(model.ir_stem[0].weight.detach().cpu(), reference.features[0][0].weight.mean(dim=1, keepdim=True))),
    }
    torch.cuda.reset_peak_memory_stats()
    inputs = {"depth_input": batch["depth_input"].cuda(), "ir_input": batch["ir_input"].cuda()}
    mask = batch["temporal_mask"].cuda()
    labels = batch["label"].cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs, temporal_mask=mask)
    loss = nn.CrossEntropyLoss()(output["logits"], labels)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"])))
    optimizer.step()
    checks.update(
        {
            "depth_shape": list(inputs["depth_input"].shape),
            "ir_shape": list(inputs["ir_input"].shape),
            "logits_shape": list(output["logits"].shape),
            "embedding_shape": list(output["embedding"].shape),
            "roi_attention_shape": list(output["roi_attention"].shape),
            "modality_gate_shape": list(output["modality_gate"].shape),
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": gradient_norm,
            "optimizer_step_completed": True,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    if checks["logits_shape"] != [4, 40] or not checks["all_40_classes_present"]:
        raise RuntimeError(json.dumps(checks, indent=2))
    path = PROJECT_ROOT / "outputs/depth_ir_pose_roi_40class_probe/training_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
