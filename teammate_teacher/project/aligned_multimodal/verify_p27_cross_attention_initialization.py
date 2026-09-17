from __future__ import annotations

import torch

from aligned_model import AlignedMultimodalModel


def model(cross_attention: bool) -> AlignedMultimodalModel:
    return AlignedMultimodalModel(
        ["ir", "skeleton"],
        dropout=0.3,
        use_cross_attention=cross_attention,
        cross_attention_grid=(3, 4),
        skeleton_input_dim=10,
    ).eval()


def main() -> None:
    checkpoint = torch.load(
        "runs/p27_strong_inner/ir_skeleton/fold_0/best_accuracy.pt",
        map_location="cpu",
        weights_only=False,
    )
    base = model(False)
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    candidate = model(True)
    incompatible = candidate.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    expected = {
        "cross_attention_scale",
        "depth_token_project.weight",
        "depth_token_project.bias",
        "cross_attention.in_proj_weight",
        "cross_attention.in_proj_bias",
        "cross_attention.out_proj.weight",
        "cross_attention.out_proj.bias",
        "cross_attention_norm.weight",
        "cross_attention_norm.bias",
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(incompatible)
    torch.manual_seed(27083)
    inputs = {
        "ir": torch.randn(1, 2, 1, 64, 64),
        "skeleton": torch.randn(1, 2, 17, 10),
    }
    with torch.no_grad():
        base_logits = base(inputs)
        candidate_logits = candidate(inputs)
    difference = float((base_logits - candidate_logits).abs().max())
    print(
        {
            "max_abs_logit_difference": difference,
            "cross_attention_scale": float(candidate.cross_attention_scale.detach()),
            "missing_parameters": sorted(incompatible.missing_keys),
        }
    )
    if difference > 1e-6:
        raise RuntimeError(f"initialization is not protected: {difference}")


if __name__ == "__main__":
    main()
