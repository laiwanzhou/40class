from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p30_shared_dir_roi_model import (
    PYRAMID_FEATURE_DIM,
    P30SharedDIRTemporalEncoder,
    SharedResNet18Pyramid,
    model_size_mib,
)


def test_shared_backbone_pyramid_shape() -> None:
    model = SharedResNet18Pyramid(imagenet_pretrained=False).eval()
    with torch.inference_mode():
        features = model(torch.randn(2, 3, 160, 160))
    assert features.shape == (2, PYRAMID_FEATURE_DIM)


def test_variable_all_frame_temporal_shapes_and_budget() -> None:
    model = P30SharedDIRTemporalEncoder(dropout=0.0).eval()
    features = torch.randn(2, 4, 2, 7, PYRAMID_FEATURE_DIM)
    roi_valid = torch.ones(2, 4, 7, dtype=torch.bool)
    roi_valid[1, 2:] = False
    # The second sample has two real frames; padded frames carry no regions.
    frame_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    roi_quality = roi_valid.float() * 0.8
    roi_source = roi_valid.long()
    clipped = torch.zeros(2, 4, 7)
    pose_factor = frame_mask.float()
    time_position = torch.tensor(
        [[0.0, 0.33, 0.67, 1.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
    )
    with torch.inference_mode():
        output = model(
            features,
            roi_quality,
            roi_valid,
            roi_source,
            clipped,
            pose_factor,
            frame_mask,
            time_position,
        )
    assert output["logits"].shape == (2, 40)
    assert output["visual_embedding"].shape == (2, 384)
    assert output["frame_sequence"].shape == (2, 4, 256)
    assert output["region_sequence"].shape == (2, 4, 7, 256)
    assert output["modality_gate"].shape == (2, 4, 7, 2)
    assert torch.isfinite(output["logits"]).all()
    assert torch.count_nonzero(output["frame_sequence"][1, 2:]) == 0

    backbone = SharedResNet18Pyramid(imagenet_pretrained=False)
    assert model_size_mib(backbone, 2) + model_size_mib(model, 2) < 40.0
