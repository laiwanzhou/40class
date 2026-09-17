from __future__ import annotations

import torch

from aligned_model import AlignedMultimodalModel


CHECKPOINT = "runs/p27_strong_inner/ir_skeleton/fold_0/best_accuracy.pt"


def make_model(use_ir_local: bool) -> AlignedMultimodalModel:
    return AlignedMultimodalModel(
        ["ir", "skeleton"],
        dropout=0.3,
        skeleton_input_dim=10,
        use_ir_local=use_ir_local,
    ).eval()


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    base = make_model(False)
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    candidate = make_model(True)
    incompatible = candidate.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    expected = {
        "visual.ir_local_stem.weight",
        "visual.ir_local_norm.weight",
        "visual.ir_local_norm.bias",
        "visual.ir_local_residual_scale",
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(incompatible)
    assert candidate.visual is not None
    assert candidate.visual.ir_stem is not None
    assert candidate.visual.ir_local_stem is not None
    with torch.no_grad():
        candidate.visual.ir_local_stem.weight.copy_(
            candidate.visual.ir_stem.weight
        )
    torch.manual_seed(27131)
    inputs = {
        "ir": torch.randn(2, 12, 1, 144, 192),
        "ir_local": torch.randn(2, 12, 1, 144, 192),
        "ir_local_quality": torch.tensor([1.0, 0.65]),
        "skeleton": torch.randn(2, 12, 17, 10),
    }
    with torch.inference_mode():
        base_logits = base(
            {"ir": inputs["ir"], "skeleton": inputs["skeleton"]}
        )
        candidate_logits = candidate(inputs)
    difference = float((base_logits - candidate_logits).abs().max())
    print(
        {
            "max_abs_logit_difference": difference,
            "local_scale": float(
                candidate.visual.ir_local_residual_scale.detach()
            ),
            "missing_parameters": sorted(incompatible.missing_keys),
        }
    )
    if difference > 1e-6:
        raise RuntimeError(f"initialization is not protected: {difference}")


if __name__ == "__main__":
    main()
