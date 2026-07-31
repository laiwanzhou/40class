from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.common import resolve_manifest_path, sorted_files
from src.data.visual_six_patch_dataset import six_patch_boxes


DEFAULT_SAMPLE_IDS = (
    "train__c00__user4__1-1-1",
    "train__c25__user24__2-2-1",
    "train__c36__user23__2-2-1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata/manifest.csv")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/depth_six_patch_fold0_14train_4val/visual_checks",
    )
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    return parser.parse_args()


def render_sample(row: pd.Series, data_root: Path, output_dir: Path) -> Path:
    trial_path = resolve_manifest_path(data_root, str(row["depth_color_path"]))
    files = sorted_files(trial_path, {".png", ".jpg", ".jpeg"})
    image_path = files[len(files) // 2]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    boxes = six_patch_boxes(*image.size)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    colors = ("red", "lime", "cyan", "yellow", "magenta", "orange")
    for index, (box, color) in enumerate(zip(boxes, colors, strict=True), start=1):
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=color, width=4)
        draw.text((box[0] + 6, box[1] + 6), str(index), fill=color, stroke_width=2, stroke_fill="black")

    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels = [("Global", image), ("Overlay", overlay)] + [
        (f"Patch {index}", image.crop(box)) for index, box in enumerate(boxes, start=1)
    ]
    for axis, (title, panel) in zip(axes.ravel(), panels, strict=True):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"{row['sample_id']} | frame={image_path.name} | source={image.width}x{image.height}")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row['sample_id']}.png"
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    sample_ids = tuple(args.sample_ids or DEFAULT_SAMPLE_IDS)
    selected = manifest.set_index("sample_id").reindex(sample_ids)
    if selected["depth_color_path"].isna().any():
        missing = selected.index[selected["depth_color_path"].isna()].tolist()
        raise ValueError(f"Unknown or missing Depth_Color samples: {missing}")
    for sample_id, row in selected.reset_index().set_index("sample_id").iterrows():
        row["sample_id"] = sample_id
        print(render_sample(row, args.data_root, args.output_dir))


if __name__ == "__main__":
    main()
