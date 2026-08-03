from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs/experiments/depth_ir_pose_roi_expert.yaml", config_args())
    set_seed(int(config["seed"]))
    train_dataset, _ = build_datasets(config)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model = build_model(config, train_dataset[0]).cuda().train()
    reference = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    reference_weight = reference.features[0][0].weight.detach()
    depth_weight = model.depth_stem[0].weight.detach().cpu()
    ir_weight = model.ir_stem[0].weight.detach().cpu()
    checks = {
        "depth_stem_exact_pretrained": bool(torch.equal(depth_weight, reference_weight)),
        "ir_stem_exact_rgb_mean": bool(torch.equal(ir_weight, reference_weight.mean(dim=1, keepdim=True))),
        "gate_weight_zero": bool(torch.count_nonzero(model.modality_gate.weight.detach()).item() == 0),
        "gate_bias_zero": bool(torch.count_nonzero(model.modality_gate.bias.detach()).item() == 0),
        "one_shared_body": len([name for name, _ in model.named_modules() if name == "shared_body"]) == 1,
    }
    inputs = {
        "depth_input": batch["depth_input"].cuda(non_blocking=True),
        "ir_input": batch["ir_input"].cuda(non_blocking=True),
    }
    mask = batch["temporal_mask"].cuda(non_blocking=True)
    labels = batch["label"].cuda(non_blocking=True)
    output = model(inputs, temporal_mask=mask)
    checks.update(
        {
            "depth_shape": list(inputs["depth_input"].shape),
            "ir_shape": list(inputs["ir_input"].shape),
            "logits_shape": list(output["logits"].shape),
            "embedding_shape": list(output["embedding"].shape),
            "roi_attention_shape": list(output["roi_attention"].shape),
            "modality_gate_shape": list(output["modality_gate"].shape),
            "initial_gate_mean": float(output["modality_gate"].mean().detach()),
        }
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(output["logits"], labels)
    loss.backward()
    gradient_groups = {
        "depth_stem": model.depth_stem,
        "ir_stem": model.ir_stem,
        "modality_gate": model.modality_gate,
        "shared_body": model.shared_body,
        "roi_scorer": model.local_scorer,
        "frame_projection": model.frame_projection,
        "gru": model.temporal,
        "classifier": model.classifier,
    }
    checks["gradient_norms"] = {
        name: float(sum(parameter.grad.detach().abs().sum().item() for parameter in module.parameters() if parameter.grad is not None))
        for name, module in gradient_groups.items()
    }
    checks["all_required_gradients_nonzero"] = all(value > 0 for value in checks["gradient_norms"].values())
    optimizer.step()
    checks["optimizer_step_completed"] = True
    checks["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    checks["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    if not all(boolean_checks) or abs(checks["initial_gate_mean"] - 0.5) > 1e-7:
        raise RuntimeError(json.dumps(checks, indent=2))
    output_path = PROJECT_ROOT / "outputs/depth_ir_pose_roi_probe/initialization_and_batch_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
