from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

from build_p46_videomae_cache import restore_legacy_attention_biases


DEFAULT_VIDEOMAE_SMALL = "MCG-NJU/videomae-small-finetuned-kinetics"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_snapshot(model_name_or_path: str | Path) -> Path:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return Path(
        snapshot_download(
            str(model_name_or_path),
            allow_patterns=("config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors"),
            local_files_only=True,
        )
    ).resolve()


class P86VideoMAESmallVisualStudent(nn.Module):
    """One deployable VideoMAE-Small student shared by all six IR clips.

    The public Kinetics checkpoint initializes the patch encoder and temporal
    Transformer. Large VideoMAE arrays are training targets only and never
    enter this model's forward pass.
    """

    def __init__(
        self,
        classes: int = 40,
        width: int = 384,
        dropout: float = 0.18,
        fusion_mode: str = "gated",
        enable_distillation_projection: bool = False,
        frames: int = 12,
        resolution: int = 112,
        pretrained_model: str | Path = DEFAULT_VIDEOMAE_SMALL,
        kinetics_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if width != 384:
            raise ValueError("VideoMAE-Small output width is fixed at 384")
        if fusion_mode != "gated":
            raise ValueError("the controlled VideoMAE-Small trial supports gated fusion only")
        if frames < 2 or frames % 2:
            raise ValueError("VideoMAE frames must be an even integer >= 2")
        if resolution < 32 or resolution % 16:
            raise ValueError("VideoMAE resolution must be a multiple of 16")
        self.fusion_mode = fusion_mode
        self.frames = int(frames)
        self.resolution = int(resolution)
        self.width = width
        self.freeze_through = "none"
        self.pretrained_load_audit: dict[str, Any]

        if kinetics_pretrained:
            snapshot = resolve_snapshot(pretrained_model)
            config = VideoMAEConfig.from_pretrained(snapshot, local_files_only=True)
            if (
                config.hidden_size != width
                or config.num_hidden_layers != 12
                or config.patch_size != 16
                or config.tubelet_size != 2
            ):
                raise RuntimeError("the requested checkpoint is not the expected VideoMAE-Small architecture")
            # VideoMAE uses fixed (not learned) sin/cos positions. Rebuilding
            # them for the controlled P86 geometry preserves every learned
            # patch/encoder weight while allowing the existing 12x112 cache.
            config.image_size = self.resolution
            config.num_frames = self.frames
            pretrained = VideoMAEForVideoClassification.from_pretrained(
                snapshot,
                config=config,
                local_files_only=True,
            )
            self.pretrained_load_audit = restore_legacy_attention_biases(pretrained, snapshot)
            self.pretrained_load_audit.update(
                {
                    "source": str(snapshot),
                    "source_model": str(pretrained_model),
                    "adapted_frames": self.frames,
                    "adapted_resolution": self.resolution,
                    "position_encoding": "fixed sinusoidal table regenerated for target geometry",
                }
            )
        else:
            config = VideoMAEConfig(
                image_size=self.resolution,
                patch_size=16,
                num_channels=3,
                num_frames=self.frames,
                tubelet_size=2,
                hidden_size=width,
                num_hidden_layers=12,
                num_attention_heads=16,
                intermediate_size=1536,
                num_labels=400,
                use_mean_pooling=True,
            )
            pretrained = VideoMAEForVideoClassification(config)
            self.pretrained_load_audit = {"source_model": None, "random_initialization": True}

        self.backbone = pretrained.videomae
        self.backbone_norm = pretrained.fc_norm or nn.Identity()
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
        self.classifier = nn.Sequential(
            nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, classes)
        )
        self.distillation_projection = (
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024))
            if enable_distillation_projection
            else None
        )

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.backbone_norm.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        backbone_ids = {id(parameter) for parameter in self.backbone_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in backbone_ids]

    def freeze_low_level(self, freeze_through: str = "layer8") -> None:
        if freeze_through not in {"layer6", "layer8", "layer10"}:
            raise ValueError(f"unknown VideoMAE freeze boundary: {freeze_through}")
        boundary = int(freeze_through.removeprefix("layer"))
        self.freeze_through = freeze_through
        for parameter in self.backbone.embeddings.parameters():
            parameter.requires_grad_(False)
        for layer in self.backbone.encoder.layer[:boundary]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)

    def encode_clips(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6 or images.shape[1:4] != (2, self.frames, 3):
            raise ValueError(
                f"expected [B,2,{self.frames},3,H,W], got {tuple(images.shape)}"
            )
        batch, windows, steps, views, height, width = images.shape
        if (height, width) != (self.resolution, self.resolution):
            raise ValueError(
                f"expected {self.resolution}x{self.resolution} pixels, got {height}x{width}"
            )
        clips = images.permute(0, 1, 3, 2, 4, 5).reshape(
            batch * windows * views, steps, 1, height, width
        )
        unit = clips.float().div_(255.0).repeat(1, 1, 3, 1, 1)
        mean = unit.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = unit.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        sequence = self.backbone(pixel_values=(unit - mean) / std).last_hidden_state
        return self.backbone_norm(sequence.mean(dim=1)).reshape(batch, windows, views, self.width)

    def forward(
        self,
        images: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        clip_embeddings = self.encode_clips(images)
        batch = clip_embeddings.shape[0]
        clip_embeddings = (
            clip_embeddings
            + self.window_embedding.view(1, 2, 1, self.width)
            + self.view_embedding.view(1, 1, 3, self.width)
        )
        clip_mask = view_valid.any(dim=2)
        clip_quality = view_quality.mean(dim=2)
        encoded = self.token_encoder(
            clip_embeddings.reshape(batch, 6, self.width),
            src_key_padding_mask=~clip_mask.reshape(batch, 6),
        ).reshape(batch, 2, 3, self.width)
        encoded = encoded * clip_mask.unsqueeze(-1)
        gate_input = torch.cat(
            (
                encoded,
                clip_quality.unsqueeze(-1),
                clip_mask.to(encoded.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        gate = self.quality_gate(gate_input).squeeze(-1).masked_fill(~clip_mask, -1e4)
        view_weight = torch.softmax(gate, dim=2) * clip_mask
        view_weight = view_weight / view_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        windows = (encoded * view_weight.unsqueeze(-1)).sum(dim=2)
        early, late = windows[:, 0], windows[:, 1]
        embedding = self.stage_fusion(
            torch.cat((early, late, 0.5 * (early + late), late - early), dim=1)
        )
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
