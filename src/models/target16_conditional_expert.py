from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from src.models.depth_ir_pose_roi_expert import DepthIRPoseROIExpert


class Target16ConditionalExpert(DepthIRPoseROIExpert):
    """B2-compatible conditional classifier for a fixed subset of actions."""

    def __init__(
        self,
        target_class_ids: Sequence[int],
        embedding_dim: int = 192,
        frame_feature_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = False,
        frozen_shared_blocks: int = 6,
    ) -> None:
        target_ids = tuple(int(value) for value in target_class_ids)
        if len(target_ids) != 16 or len(set(target_ids)) != 16:
            raise ValueError("Target expert requires exactly 16 unique class IDs.")
        if tuple(sorted(target_ids)) != target_ids:
            raise ValueError("Target class IDs must follow ascending dataset label order.")
        super().__init__(
            num_classes=len(target_ids),
            expected_views=4,
            embedding_dim=embedding_dim,
            frame_feature_dim=frame_feature_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
        if not 0 <= frozen_shared_blocks <= len(self.shared_body):
            raise ValueError("Invalid frozen shared-body block count.")
        self.target_class_ids = target_ids
        self.frozen_shared_blocks = int(frozen_shared_blocks)
        self._frozen_modules: tuple[nn.Module, ...] = ()

    def initialize_from_b2(self, checkpoint: Mapping[str, Any]) -> None:
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("B2 checkpoint is missing model_state_dict.")
        weight = state.get("classifier.weight")
        bias = state.get("classifier.bias")
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise ValueError("B2 checkpoint is missing classifier tensors.")
        if weight.shape[0] != 40 or bias.shape[0] != 40:
            raise ValueError(f"Expected a 40-class B2 head, got {tuple(weight.shape)} / {tuple(bias.shape)}.")

        mapped = dict(state)
        indices = torch.tensor(self.target_class_ids, dtype=torch.long, device=weight.device)
        mapped["classifier.weight"] = weight.index_select(0, indices).clone()
        mapped["classifier.bias"] = bias.index_select(0, indices).clone()
        self.load_state_dict(mapped, strict=True)

    def freeze_b2_early_layers(self) -> dict[str, int]:
        frozen = (
            self.depth_stem,
            self.ir_stem,
            self.modality_gate,
            *tuple(self.shared_body[: self.frozen_shared_blocks]),
        )
        for module in frozen:
            module.requires_grad_(False)
            module.eval()
        self._frozen_modules = frozen
        return self.parameter_audit()

    def parameter_audit(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable, "frozen_parameters": total - trainable}

    def train(self, mode: bool = True) -> "Target16ConditionalExpert":
        super().train(mode)
        if mode:
            for module in self._frozen_modules:
                module.eval()
        return self
