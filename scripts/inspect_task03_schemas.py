from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.common import load_modality_frames, sorted_files


DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
MODALITIES = {
    "Depth_Color": "depth_color_path",
    "IR": "ir_path",
    "Thermal": "thermal_path",
    "IMU": "imu_path",
    "Skeleton": "skeleton_path",
    "Radar": "radar_path",
}


def main() -> None:
    manifest = PROJECT_ROOT / "metadata" / "manifest.csv"
    fold = PROJECT_ROOT / "metadata" / "splits" / "fold_0.json"
    for modality, path_column in MODALITIES.items():
        train, val = load_modality_frames(manifest, fold, DATA_ROOT, path_column)
        print(f"{modality}: train={len(train)} val={len(val)}")
        for _, row in train.head(2).iterrows():
            trial = Path(row["trial_path"])
            files = sorted_files(trial, {".png", ".jpg", ".jpeg", ".csv", ".json"}, recursive=True)
            print(f"  {row['sample_id']}: {trial} ({len(files)} files)")
            if modality in {"Depth_Color", "IR", "Thermal"}:
                with Image.open(files[0]) as image:
                    print(f"    image={image.mode} {image.size} {files[0].suffix.lower()}")
            elif modality in {"IMU", "Radar"}:
                table = pd.read_csv(files[0])
                print(f"    csv_shape={table.shape} columns={list(table.columns)}")
            else:
                value = json.loads(files[0].read_text(encoding="utf-8"))
                print(f"    json_type={type(value).__name__} people={len(value) if isinstance(value, list) else 'n/a'}")


if __name__ == "__main__":
    main()
