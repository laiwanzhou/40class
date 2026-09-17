from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from depth_encoding import decode_jet_rgb, jet_palette_rgb


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "depth_encoding_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 Depth_Color 是否可由 OpenCV JET 调色板反解")
    parser.add_argument("--training-root", type=Path, default=REPO_DIR / "Training" / "data" / "HAR" / "data" / "Depth_Color")
    parser.add_argument("--testing-root", type=Path, default=REPO_DIR / "Testing" / "data")
    parser.add_argument("--sample-files", type=int, default=256)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def evenly_sample(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    positions = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[index] for index in positions]


def audit_split(root: Path, sample_files: int, pixel_stride: int) -> dict[str, object]:
    paths = sorted(root.rglob("Depth_*_Color.png"))
    selected = evenly_sample(paths, sample_files)
    total = valid = exact = unmatched = 0
    for path in selected:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)[::pixel_stride, ::pixel_stride]
        _, mask, repaired = decode_jet_rgb(rgb)
        pixels = int(mask.size)
        valid_pixels = int(mask.sum())
        total += pixels
        valid += valid_pixels
        unmatched += repaired
        exact += valid_pixels - repaired
    return {
        "root": str(root.resolve()),
        "all_files": len(paths),
        "sampled_files": len(selected),
        "sampled_pixels": total,
        "black_invalid_fraction": (total - valid) / total if total else None,
        "exact_jet_fraction_of_valid": exact / valid if valid else None,
        "repaired_nonblack_pixels": unmatched,
    }


def temporal_stability(training_root: Path) -> dict[str, object]:
    trial_dirs = sorted({path.parent for path in training_root.rglob("Depth_*_Color.png")})
    trial = next((path for path in trial_dirs if len(list(path.glob("Depth_*_Color.png"))) >= 20), None)
    if trial is None:
        return {"available": False}
    frames: list[np.ndarray] = []
    for path in sorted(trial.glob("Depth_*_Color.png")):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)[::8, ::8]
        decoded, valid, _ = decode_jet_rgb(rgb)
        frames.append(np.where(valid, decoded.astype(np.float32), np.nan))
    sequence = np.stack(frames)
    always_valid = np.isfinite(sequence).all(axis=0)
    standard_deviation = sequence[:, always_valid].std(axis=0)
    return {
        "available": True,
        "trial": str(trial.resolve()),
        "frames": len(frames),
        "always_valid_sampled_pixels": int(always_valid.sum()),
        "median_temporal_std": float(np.median(standard_deviation)),
        "p90_temporal_std": float(np.quantile(standard_deviation, 0.9)),
        "fraction_std_le_2": float(np.mean(standard_deviation <= 2.0)),
        "note": "仅用于确认同一静态场景中的调色板索引是否稳定，不能证明跨场景为绝对米制深度。",
    }


def main() -> None:
    args = parse_args()
    training = audit_split(args.training_root.resolve(), args.sample_files, args.pixel_stride)
    testing = audit_split(args.testing_root.resolve(), args.sample_files, args.pixel_stride)
    summary = {
        "palette": "OpenCV COLORMAP_JET",
        "palette_unique_colors": int(len(np.unique(jet_palette_rgb(), axis=0))),
        "training": training,
        "testing": testing,
        "temporal_stability_example": temporal_stability(args.training_root.resolve()),
        "conclusion": (
            "非黑像素若几乎全部精确命中 JET，可先在原始分辨率反解为有序深度索引，"
            "并将黑色单独作为 invalid mask；不可把该索引未经额外验证解释为绝对米制深度。"
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
