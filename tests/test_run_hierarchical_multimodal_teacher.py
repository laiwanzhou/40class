from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from src.models.body_motion_segment_encoder import BodyMotionSegmentEncoder
from src.models.hierarchical_action_query_fusion import HierarchicalActionQueryFusion
from src.models.hierarchical_multimodal_teacher import HierarchicalMultimodalTeacher
from src.models.structured_ir_depth_visual_encoder import StructuredIRDepthVisualEncoder
from src.train_hierarchical_multimodal_teacher import run_smoke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/hierarchical_multimodal_midfusion_stage1.yaml"


class TinyTemporalBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 12
        self.proj = nn.Conv3d(3, 12, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.tail = nn.Linear(12, 12)

    def encode_prefix(self, clips: torch.Tensor) -> torch.Tensor:
        return self.proj(clips).mean(dim=(3, 4)).transpose(1, 2)[:, :, None]

    def encode_tail(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tail(tokens.mean(dim=2))


def tiny_model(_: dict) -> HierarchicalMultimodalTeacher:
    dim = 32
    return HierarchicalMultimodalTeacher(
        visual_encoder=StructuredIRDepthVisualEncoder(
            backbone=TinyTemporalBackbone(), output_dim=dim
        ),
        body_encoder=BodyMotionSegmentEncoder(output_dim=dim, heads=4),
        fusion=HierarchicalActionQueryFusion(
            dim=dim, classes=40, heads=4, layers=2
        ),
        dim=dim,
        classes=40,
    )


class TinyDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        torch.manual_seed(100 + index)
        complete = index < 2
        return {
            "visual": torch.randn(2, 4, 3, 16, 4, 4),
            "visual_view_availability": torch.full(
                (2, 4), complete, dtype=torch.bool
            ),
            "skeleton": torch.randn(8, 17, 6),
            "skeleton_mask": torch.full((8,), complete, dtype=torch.bool),
            "imu": torch.randn(8, 5, 16),
            "imu_role_mask": torch.full((8, 5), complete, dtype=torch.bool),
            "availability": torch.full((4,), complete, dtype=torch.bool),
            "core_available": torch.tensor(complete),
            "sample_id": f"sample_{index}",
            "user_id": "user1",
            "label": index,
        }


def test_smoke_uses_train_users_only_and_updates_every_group(tmp_path: Path) -> None:
    report = run_smoke(
        CONFIG,
        output_root=tmp_path / "smoke",
        model_factory=tiny_model,
        dataset_factory=lambda _: TinyDataset(),
        device=torch.device("cpu"),
    )

    assert report["sample_users_entered_gradient"] == ["user1"]
    assert set(report["sample_users_entered_gradient"]).isdisjoint({"user6", "user7"})
    assert report["finite_gradients"] is True
    assert report["changed_parameter_groups"] == [
        "visual", "skeleton", "imu", "fusion"
    ]
    assert report["body_only_finite"] is True
    assert report["no_core_finite"] is True


def test_smoke_refuses_existing_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()

    try:
        run_smoke(
            CONFIG,
            output_root=output,
            model_factory=tiny_model,
            dataset_factory=lambda _: TinyDataset(),
            device=torch.device("cpu"),
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("smoke must refuse an existing output directory")
