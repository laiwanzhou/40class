from __future__ import annotations

import numpy as np
import torch

from src.data.dual_spatial_full_sequence_dataset import DualSpatialFullSequenceDataset
from src.models.depth_ir_pose_roi_expert import DepthIRPoseROIExpert
from src.models.dual_spatial_full_sequence_tcn import DualSpatialFullSequenceTCN


def test_full_sequence_window_covers_endpoints_and_padding() -> None:
    dataset = object.__new__(DualSpatialFullSequenceDataset)
    dataset.num_frames = 6
    indices, mask = dataset._window(13)
    assert indices[0] == 0
    assert indices[-1] == 12
    assert len(np.unique(indices)) == 6
    assert mask.tolist() == [True] * 6
    indices, mask = dataset._window(3)
    assert indices.tolist() == [0, 1, 2, 2, 2, 2]
    assert mask.tolist() == [True, True, True, False, False, False]


def test_view_feature_refactor_preserves_original_encoder_output() -> None:
    torch.manual_seed(4)
    model = DepthIRPoseROIExpert(
        num_classes=40, frame_feature_dim=16, embedding_dim=12, dropout=0.0, pretrained=False,
    ).eval()
    inputs = {
        "depth_input": torch.randn(1, 2, 4, 3, 64, 64),
        "ir_input": torch.randn(1, 2, 4, 1, 64, 64),
    }
    direct = model.encode_frames(inputs)
    views = model.encode_view_features(inputs)["view_features"]
    local_attention = torch.softmax(model.local_scorer(views[:, :, 1:]).squeeze(-1), dim=-1)
    local_summary = (views[:, :, 1:] * local_attention.unsqueeze(-1)).sum(dim=2)
    expected = model.frame_projection(torch.cat((views[:, :, 0], local_summary), dim=-1))
    torch.testing.assert_close(direct["frame_features"], expected)
    torch.testing.assert_close(direct["roi_attention"], local_attention)


def test_zero_initialized_global_residual_starts_from_interaction_features() -> None:
    model = DualSpatialFullSequenceTCN(
        frame_feature_dim=16,
        channels=16,
        embedding_dim=24,
        short_dilations=(1,),
        long_dilations=(1, 2),
        dropout=0.0,
        pretrained=False,
    ).eval()
    interaction = torch.randn(2, 5, 16)
    global_features = torch.randn(2, 5, model.raw_spatial_dim)
    fused, gate, residual = model.fuse_spatial(interaction, global_features)
    torch.testing.assert_close(fused, interaction)
    assert torch.count_nonzero(residual) == 0
    torch.testing.assert_close(gate, torch.full_like(gate, 0.5))


def test_cached_dual_spatial_forward_shapes_and_spatial_freeze() -> None:
    model = DualSpatialFullSequenceTCN(
        frame_feature_dim=16,
        channels=16,
        embedding_dim=24,
        short_dilations=(1,),
        long_dilations=(1, 2),
        dropout=0.0,
        pretrained=False,
    )
    output = model.forward_cached(
        torch.randn(3, 7, 16),
        torch.randn(3, 7, model.raw_spatial_dim),
        torch.tensor([[1] * 7, [1] * 5 + [0] * 2, [1] * 3 + [0] * 4], dtype=torch.bool),
    )
    assert output["logits"].shape == (3, 40)
    assert output["embedding"].shape == (3, 24)
    assert output["spatial_gate"].shape == (3, 7)
    model.freeze_spatial()
    assert not any(parameter.requires_grad for parameter in model.spatial_encoder.parameters())
