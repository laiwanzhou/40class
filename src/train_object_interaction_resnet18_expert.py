from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.models.object_interaction_resnet18_tcn_expert import ObjectInteractionResNet18TCNExpert
from src.train_object_interaction_tcn_expert import PROJECT_ROOT, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/depth_ir_object_interaction_resnet18_expert.yaml",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-id")
    return parser.parse_args()


def model_factory(config: dict[str, Any], targets: list[int]) -> ObjectInteractionResNet18TCNExpert:
    return ObjectInteractionResNet18TCNExpert(
        targets,
        frame_feature_dim=int(config["frame_feature_dim"]),
        tcn_channels=int(config["tcn_channels"]),
        embedding_dim=int(config["expert_embedding_dim"]),
        kernel_size=int(config["tcn_kernel_size"]),
        dilations=tuple(int(value) for value in config["tcn_dilations"]),
        dropout=float(config["dropout"]),
        encoder_chunk_size=int(config["encoder_chunk_size"]),
    )


def main() -> None:
    run(parse_args(), model_factory=model_factory)


if __name__ == "__main__":
    main()
