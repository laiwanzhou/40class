from __future__ import annotations

import json
from pathlib import Path

import torch

from aligned_model import AlignedMultimodalModel
from residual_logit_fusion import build_model
from train import load_residual_modality_initialization


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = (
    PROJECT_DIR
    / "runs"
    / "p27_strong_inner"
    / "ir_skeleton"
    / "fold_0"
    / "best_accuracy.pt"
)
OUTPUT = (
    PROJECT_DIR
    / "runs"
    / "p27_strong_inner"
    / "depth_residual_on_ir_skeleton"
    / "zero_residual_verification.json"
)


def main() -> None:
    torch.manual_seed(27087)
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source = build_model(checkpoint, torch.device("cpu"))
    target = AlignedMultimodalModel(
        ["depth", "ir", "skeleton"],
        dropout=float(checkpoint["config"]["dropout"]),
        visual_stem_fusion="ir_depth_residual",
        skeleton_input_dim=int(checkpoint["config"]["skeleton_input_dim"]),
    )
    load_residual_modality_initialization(target, str(SOURCE))
    source.eval()
    target.eval()
    inputs = {
        "ir": torch.randn(2, 12, 1, 144, 192),
        "depth": torch.randn(2, 12, 3, 144, 192),
        "skeleton": torch.randn(2, 12, 17, 10),
    }
    with torch.inference_mode():
        source_logits = source(
            {"ir": inputs["ir"], "skeleton": inputs["skeleton"]}
        )
        target_logits = target(inputs)
    maximum_absolute_difference = float(
        (source_logits - target_logits).abs().max()
    )
    result = {
        "source_checkpoint": str(SOURCE.resolve()),
        "depth_residual_scale": float(
            target.visual.depth_residual_scale.detach()
        ),
        "maximum_absolute_logit_difference": maximum_absolute_difference,
        "numerically_equivalent": maximum_absolute_difference <= 1e-6,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["numerically_equivalent"]:
        raise RuntimeError("Zero-residual initialization changed base logits")


if __name__ == "__main__":
    main()
