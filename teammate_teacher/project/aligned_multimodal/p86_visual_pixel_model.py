from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from aligned_model import temporal_shift
from p86_visual_student_model import TemporalResidualBlock


class ClipPool(nn.Module):
    def __init__(self, width: int, dropout: float, frames: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(width, dilation, dropout) for dilation in (1, 2, 4)
        )
        if frames < 2:
            raise ValueError("frames must be at least two")
        self.time_position = nn.Parameter(torch.zeros(1, frames, width))
        nn.init.trunc_normal_(self.time_position, std=0.02)
        self.score = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width * 3), nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        steps = sequence.shape[1]
        if steps != self.time_position.shape[1]:
            raise ValueError(f"expected {self.time_position.shape[1]} frames, got {steps}")
        mask = torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
        value = sequence + self.time_position
        for block in self.blocks:
            value = block(value, mask)
        attention = torch.softmax(self.score(value).squeeze(-1), dim=1)
        attended = (value * attention.unsqueeze(-1)).sum(dim=1)
        maximum = value.amax(dim=1)
        mean = value.mean(dim=1)
        return self.output(torch.cat((attended, maximum, mean), dim=1))


class P86TrainableVisualStudent(nn.Module):
    """Shared trainable ResNet18 over early/late x scene/person/workspace clips."""

    def __init__(
        self,
        classes: int = 40,
        width: int = 512,
        dropout: float = 0.18,
        fusion_mode: str = "gated",
        enable_distillation_projection: bool = False,
        frames: int = 8,
    ) -> None:
        super().__init__()
        if width != 512:
            raise ValueError("ResNet18 output width is fixed at 512")
        if fusion_mode not in {"gated", "structured", "gated_residual"}:
            raise ValueError(f"unknown visual fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.frames = int(frames)
        self.freeze_through = "none"
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(backbone.conv1.weight.sum(dim=1, keepdim=True))
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.clip_pool = ClipPool(width, dropout, frames=self.frames)
        self.view_embedding = nn.Parameter(torch.zeros(3, width))
        self.window_embedding = nn.Parameter(torch.zeros(2, width))
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        nn.init.trunc_normal_(self.window_embedding, std=0.02)
        token_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=8,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=1)
        if fusion_mode in {"gated", "gated_residual"}:
            self.quality_gate = nn.Sequential(
                nn.LayerNorm(width + 2),
                nn.Linear(width + 2, width // 2),
                nn.GELU(),
                nn.Linear(width // 2, 1),
            )
            self.stage_fusion = nn.Sequential(
                nn.LayerNorm(width * 4),
                nn.Linear(width * 4, width),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.quality_gate = None
            self.stage_fusion = None
        if fusion_mode in {"structured", "gated_residual"}:
            # Preserve all six early/late x scene/person/workspace tokens until
            # the final visual decision instead of forcing views to compete.
            self.structured_fusion = nn.Sequential(
                nn.LayerNorm(width * 6),
                nn.Linear(width * 6, width),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.structured_fusion = None
        if fusion_mode == "gated_residual":
            self.structured_residual_norm = nn.LayerNorm(width)
            self.structured_residual_scale = nn.Parameter(torch.tensor(0.05))
        else:
            self.structured_residual_norm = None
            self.register_parameter("structured_residual_scale", None)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, classes)
        )
        # Training-only aligned feature head.  It does not feed classification
        # logits and can be stripped from a deployment checkpoint.
        self.distillation_projection = (
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024))
            if enable_distillation_projection
            else None
        )

    def backbone_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def head_parameters(self) -> list[nn.Parameter]:
        backbone_ids = {id(parameter) for parameter in self.backbone_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in backbone_ids]

    def freeze_low_level(self, freeze_through: str = "layer2") -> None:
        if freeze_through not in {"layer1", "layer2"}:
            raise ValueError(f"unknown freeze boundary: {freeze_through}")
        self.freeze_through = freeze_through
        modules = [self.conv1, self.bn1, self.layer1]
        if freeze_through == "layer2":
            modules.append(self.layer2)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        # Frozen BN statistics are part of the rollback-stable ImageNet representation.
        self.bn1.eval()
        frozen_blocks = [self.layer1]
        if freeze_through == "layer2":
            frozen_blocks.append(self.layer2)
        for module in frozen_blocks:
            for child in module.modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.eval()

    def train(self, mode: bool = True) -> "P86TrainableVisualStudent":
        super().train(mode)
        if mode:
            self.bn1.eval()
            frozen_blocks = [self.layer1]
            if self.freeze_through == "layer2":
                frozen_blocks.append(self.layer2)
            for module in frozen_blocks:
                for child in module.modules():
                    if isinstance(child, nn.BatchNorm2d):
                        child.eval()
        return self

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6 or images.shape[1:4] != (2, self.frames, 3):
            raise ValueError(
                f"expected [B,2,{self.frames},3,H,W], got {tuple(images.shape)}"
            )
        batch, windows, steps, views, height, width = images.shape
        clip_batch = batch * windows * views
        source = images.permute(0, 1, 3, 2, 4, 5).reshape(
            clip_batch * steps, 1, height, width
        )
        source = source.float().div_(255.0).sub_(0.449).div_(0.226)
        source = self.maxpool(self.relu(self.bn1(self.conv1(source))))
        source = self.layer1(temporal_shift(source, clip_batch, steps))
        source = self.layer2(temporal_shift(source, clip_batch, steps))
        source = self.layer3(temporal_shift(source, clip_batch, steps))
        source = self.layer4(temporal_shift(source, clip_batch, steps))
        source = source.mean(dim=(-2, -1)).reshape(batch, windows, views, steps, 512)
        return source.permute(0, 1, 3, 2, 4)

    def forward(
        self,
        images: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sequence = self.encode_frames(images)
        batch = sequence.shape[0]
        clips = sequence.permute(0, 1, 3, 2, 4).reshape(
            batch * 6, self.frames, 512
        )
        clip_embeddings = self.clip_pool(clips).reshape(batch, 2, 3, 512)
        clip_embeddings = (
            clip_embeddings
            + self.window_embedding.view(1, 2, 1, 512)
            + self.view_embedding.view(1, 1, 3, 512)
        )
        clip_mask = view_valid.any(dim=2)
        clip_quality = view_quality.mean(dim=2)
        flat = clip_embeddings.reshape(batch, 6, 512)
        flat_mask = clip_mask.reshape(batch, 6)
        encoded = self.token_encoder(flat, src_key_padding_mask=~flat_mask)
        encoded = encoded.reshape(batch, 2, 3, 512) * clip_mask.unsqueeze(-1)
        gated_embedding: torch.Tensor | None = None
        if self.fusion_mode in {"gated", "gated_residual"}:
            assert self.quality_gate is not None and self.stage_fusion is not None
            gate_input = torch.cat(
                (encoded, clip_quality.unsqueeze(-1), clip_mask.to(encoded.dtype).unsqueeze(-1)),
                dim=-1,
            )
            gate = self.quality_gate(gate_input).squeeze(-1)
            gate = gate.masked_fill(~clip_mask, -1e4)
            view_weight = torch.softmax(gate, dim=2) * clip_mask
            view_weight = view_weight / view_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
            windows = (encoded * view_weight.unsqueeze(-1)).sum(dim=2)
            early, late = windows[:, 0], windows[:, 1]
            gated_embedding = self.stage_fusion(
                torch.cat((early, late, 0.5 * (early + late), late - early), dim=1)
            )
        if self.fusion_mode in {"structured", "gated_residual"}:
            assert self.structured_fusion is not None
            if self.fusion_mode == "structured":
                # Quality weights are diagnostic only here; no valid view is
                # allowed to suppress another before classification.
                diagnostic_score = torch.log(clip_quality.clamp_min(1e-3)).masked_fill(
                    ~clip_mask, -1e4
                )
                view_weight = torch.softmax(diagnostic_score, dim=2) * clip_mask
                view_weight = view_weight / view_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
                windows = (encoded * view_weight.unsqueeze(-1)).sum(dim=2)
            structured_embedding = self.structured_fusion(encoded.flatten(1))
        if self.fusion_mode == "gated":
            assert gated_embedding is not None
            embedding = gated_embedding
        elif self.fusion_mode == "structured":
            embedding = structured_embedding
        else:
            assert gated_embedding is not None
            assert self.structured_residual_norm is not None
            assert self.structured_residual_scale is not None
            embedding = gated_embedding + torch.tanh(
                self.structured_residual_scale
            ) * self.structured_residual_norm(structured_embedding)
        output = {
            "logits": self.classifier(embedding),
            "visual_embedding": embedding,
            "clip_embeddings": encoded,
            "clip_mask": clip_mask,
            "window_embeddings": windows,
            "view_weight": view_weight,
        }
        if self.distillation_projection is not None:
            output["projected_clip_embeddings"] = self.distillation_projection(encoded)
        return output


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def model_size_mib(module: nn.Module, bytes_per_parameter: int = 4) -> float:
    return parameter_count(module) * bytes_per_parameter / (1024**2)
