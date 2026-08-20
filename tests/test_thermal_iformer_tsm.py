from __future__ import annotations

import torch
from torch import nn

from src.models.expert_contract import ExpertOutput
from src.models.thermal_iformer_tsm import ThermalIFormerExpert
from src.models.thermal_tsm import IFormerTSM


class _FakeIFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_bn = True
        self.last_proj = False
        self.downsample_layers = nn.ModuleList(
            [
                nn.Conv2d(3, 8, 3, stride=2, padding=1),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.Conv2d(16, 24, 3, stride=2, padding=1),
                nn.Conv2d(24, 32, 3, stride=2, padding=1),
            ]
        )
        self.stages = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.classifier = nn.Linear(32, 40)


def test_thermal_expert_returns_common_trial_surface() -> None:
    model = ThermalIFormerExpert(IFormerTSM(backbone=_FakeIFormer())).eval()
    clips = torch.zeros(2, 16, 3, 224, 224)
    quality = torch.ones(2, 5)
    quality_mask = torch.ones_like(quality, dtype=torch.bool)
    availability = torch.tensor([True, False])

    with torch.inference_mode():
        output = model(clips, quality, quality_mask, availability)

    assert isinstance(output, ExpertOutput)
    assert output.main_logits.shape == (2, 40)
    assert output.embedding.shape == (2, 32)
    assert torch.isfinite(output.main_logits).all()
    assert output.availability is availability
    assert output.quality is quality
    assert output.quality_mask is quality_mask
