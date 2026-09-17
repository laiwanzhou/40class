"""Extract the matched P238 physical tokens for the 10 recoverable union rows."""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor

from p90_internvideo2_l_teacher import (
    EightFrameVideoCollator,
    PROCESSOR_SNAPSHOT,
    build_model as build_internvideo,
)
from p90_videomae_lora_teacher import EvalTrialDataset, VideoCollator
from p90_videomaev2_distilled_teacher import (
    PROCESSOR_REPO,
    build_model as build_videomaev2,
)
from p91_videomaev2_modality_teacher import ModalityTrialDataset


HERE = Path(__file__).resolve().parent
UNION = HERE / "data/six_modality_audit/train_union_manifest.csv"
MAIN = HERE / "data/manifest.csv"
OUTPUT = HERE / "runs/p302_union_full_physical_features_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rows() -> list[dict[str, str]]:
    main_keys = {
        (row["class_name"], row["user_id"], row["trial_id"])
        for row in read_csv(MAIN)
    }
    selected = []
    for source in read_csv(UNION):
        key = (source["class_name"], source["user_id"], source["trial_id"])
        if key in main_keys or not source["usable_pattern"].startswith("111"):
            continue
        selected.append(
            {
                "sample_id": source["sample_id"],
                "source_id": source["sample_id"],
                "user_id": source["user_id"],
                "class_id": source["class_id"],
                "ir_dir": source["ir_path"],
                "depth_dir": source["depth_color_path"],
            }
        )
    selected.sort(key=lambda row: row["sample_id"])
    if len(selected) != 10:
        raise RuntimeError(f"expected 10 recoverable D/IR/Thermal rows, found {len(selected)}")
    return selected


@torch.inference_mode()
def videomaev2_features(
    selected: list[dict[str, str]], labels: np.ndarray, modality: str
) -> np.ndarray:
    device = torch.device("cuda")
    model, _ = build_videomaev2(device)
    processor_path = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(processor_path, local_files_only=True)
    if modality == "ir":
        dataset = EvalTrialDataset(selected, labels, "ir")
        views = 6
    else:
        dataset = ModalityTrialDataset(selected, labels, modality)
        views = 3
    loader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        num_workers=0,
        collate_fn=VideoCollator(processor),
        pin_memory=True,
    )
    parts = []
    for batch in loader:
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4).to(
            device=device, dtype=torch.float16
        )
        feature = model.forward_features(pixels)
        parts.append(feature.reshape(len(batch["sample_ids"]), views, 768).half().cpu().numpy())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(parts)


@torch.inference_mode()
def internvideo_features(selected: list[dict[str, str]], labels: np.ndarray) -> np.ndarray:
    device = torch.device("cuda")
    model, head = build_internvideo(device)
    processor = VideoMAEImageProcessor.from_pretrained(
        PROCESSOR_SNAPSHOT, local_files_only=True
    )
    loader = DataLoader(
        EvalTrialDataset(selected, labels, "ir"),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=EightFrameVideoCollator(processor),
        pin_memory=True,
    )
    parts = []
    for batch in loader:
        pixels = batch["pixel_values"].permute(0, 2, 1, 3, 4).to(
            device=device, dtype=torch.bfloat16
        )
        feature = model(pixels)
        parts.append(feature.reshape(len(batch["sample_ids"]), 6, 768).half().cpu().numpy())
    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(parts)


def main() -> None:
    selected = rows()
    labels = np.asarray([int(row["class_id"]) for row in selected], dtype=np.int64)
    ids = np.asarray([row["sample_id"] for row in selected])
    users = np.asarray([row["user_id"] for row in selected])
    ir_vmae = videomaev2_features(selected, labels, "ir")
    print(json.dumps({"feature": "ir_videomaev2", "shape": ir_vmae.shape}), flush=True)
    ir_iv2 = internvideo_features(selected, labels)
    print(json.dumps({"feature": "ir_internvideo2", "shape": ir_iv2.shape}), flush=True)
    depth = videomaev2_features(selected, labels, "depth")
    print(json.dumps({"feature": "depth_videomaev2", "shape": depth.shape}), flush=True)
    thermal = videomaev2_features(selected, labels, "thermal")
    print(json.dumps({"feature": "thermal_videomaev2", "shape": thermal.shape}), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "features.npz",
        sample_ids=ids,
        labels=labels,
        users=users,
        ir_videomaev2=ir_vmae,
        ir_internvideo2=ir_iv2,
        depth_videomaev2=depth,
        thermal_videomaev2=thermal,
    )
    report = {
        "stage": "P302_union_full_physical_features",
        "status": "complete",
        "rows": len(selected),
        "classes": sorted(set(labels.tolist())),
        "users": sorted(set(users.tolist())),
        "tokens": {
            "ir_videomaev2": list(ir_vmae.shape),
            "ir_internvideo2": list(ir_iv2.shape),
            "depth_videomaev2": list(depth.shape),
            "thermal_videomaev2": list(thermal.shape),
        },
        "test_rows_loaded": 0,
        "test_labels_read": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
