from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint


MODALITY_NAMES = ("ir", "depth")
VIEW_NAMES = ("global", "person_context", "left_hand_object", "right_hand_object")

OFFICIAL_VIDEOMAEV2_VIT_B = {
    "architecture": "vit_base_patch16_224",
    "repository": "https://github.com/OpenGVLab/VideoMAEv2",
    "repository_revision": "29eab1e8a588d1b3ec0cdec7b03a86cca491b74b",
    "license": "MIT",
    "checkpoint_url": (
        "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/"
        "distill/vit_b_k710_dl_from_giant.pth"
    ),
    "checkpoint_sha256": "8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec",
    "checkpoint_bytes": 173_574_417,
    "pretraining_dataset": "Kinetics-710 distilled from VideoMAE V2 ViT-giant",
    "frames": 16,
    "image_size": 224,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(inputs))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = inputs.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(inputs, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale @ key.transpose(-2, -1)).softmax(dim=-1)
        output = (self.attn_drop(attention) @ value).transpose(1, 2).reshape(
            batch, tokens, channels
        )
        return self.proj_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        return output + self.drop_path(self.mlp(self.norm2(output)))


class PatchEmbed(nn.Module):
    def __init__(self, *, image_size: int, frames: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = (image_size, image_size)
        self.tubelet_size = 2
        self.patch_size = (16, 16)
        self.num_patches = (frames // 2) * (image_size // 16) ** 2
        self.proj = nn.Conv3d(
            3, embed_dim, kernel_size=(2, 16, 16), stride=(2, 16, 16)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[2:] != (16, 224, 224):
            raise ValueError("VideoMAE V2 ViT-B requires [B,3,16,224,224]")
        return self.proj(inputs).flatten(2).transpose(1, 2)


def sinusoid_encoding(tokens: int, dim: int) -> torch.Tensor:
    positions = np.arange(tokens, dtype=np.float64)[:, None]
    dimensions = np.arange(dim, dtype=np.float64)[None, :]
    angles = positions / np.power(10000, 2 * np.floor(dimensions / 2) / dim)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.tensor(angles, dtype=torch.float32).unsqueeze(0)


class VideoMAEV2ViTBase(nn.Module):
    """Checkpoint-compatible subset of the official VideoMAE V2 fine-tune model.

    The tensor/module names intentionally match upstream revision
    29eab1e8a588d1b3ec0cdec7b03a86cca491b74b.
    """

    def __init__(self, *, num_classes: int = 40, with_cp: bool = True) -> None:
        super().__init__()
        embed_dim = 768
        self.num_classes = int(num_classes)
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(image_size=224, frames=16, embed_dim=embed_dim)
        self.pos_embed = sinusoid_encoding(self.patch_embed.num_patches, embed_dim)
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(embed_dim, 12) for _ in range(12)])
        self.norm = nn.Identity()
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head_dropout = nn.Dropout(0.0)
        self.head = nn.Linear(embed_dim, self.num_classes)
        self.with_cp = bool(with_cp)

    def forward_features(self, clips: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(clips)
        features = self.pos_drop(
            features
            + self.pos_embed.to(device=features.device, dtype=features.dtype).expand(
                features.shape[0], -1, -1
            )
        )
        for block in self.blocks:
            if self.with_cp and self.training and torch.is_grad_enabled():
                features = checkpoint.checkpoint(block, features, use_reentrant=True)
            else:
                features = block(features)
        return self.fc_norm(features.mean(dim=1))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.head(self.head_dropout(self.forward_features(clips)))


def build_official_videomaev2_vit_b(
    *, checkpoint_path: Path, num_classes: int = 40, with_cp: bool = True
) -> tuple[VideoMAEV2ViTBase, dict[str, object]]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    size = checkpoint_path.stat().st_size
    digest = sha256_file(checkpoint_path)
    if size != OFFICIAL_VIDEOMAEV2_VIT_B["checkpoint_bytes"]:
        raise RuntimeError(f"official checkpoint byte mismatch: {size}")
    if digest != OFFICIAL_VIDEOMAEV2_VIT_B["checkpoint_sha256"]:
        raise RuntimeError(f"official checkpoint SHA256 mismatch: {digest}")

    archive = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(archive, dict) or not isinstance(archive.get("module"), dict):
        raise ValueError("official checkpoint must contain a module state mapping")
    state = dict(archive["module"])
    source_head_shape = list(state["head.weight"].shape)
    del state["head.weight"], state["head.bias"]
    model = VideoMAEV2ViTBase(num_classes=num_classes, with_cp=with_cp)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {"head.weight", "head.bias"}:
        raise RuntimeError(
            "strict backbone load failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    provenance = {
        **OFFICIAL_VIDEOMAEV2_VIT_B,
        "checkpoint_path": str(checkpoint_path),
        "strict_backbone_load": True,
        "source_head_shape": source_head_shape,
        "replacement_head_shape": list(model.head.weight.shape),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "activation_checkpointing": bool(with_cp),
    }
    return model, provenance


class IRDepthVideoMAEV2Teacher(nn.Module):
    """Training-only shared video backbone with class-conditioned view fusion."""

    def __init__(self, *, backbone: nn.Module, num_classes: int = 40) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.class_view_gate = nn.Parameter(
            torch.zeros(self.num_classes, len(MODALITY_NAMES), len(VIEW_NAMES))
        )
        self.last_execution_trace: list[str] = []

    def fuse_view_logits(
        self, view_logits: torch.Tensor, availability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if view_logits.ndim != 4 or view_logits.shape[1:] != (
            len(MODALITY_NAMES), len(VIEW_NAMES), self.num_classes
        ):
            raise ValueError("view_logits must have shape [B,2,4,C]")
        if availability.shape != view_logits.shape[:3] or availability.dtype != torch.bool:
            raise ValueError("availability must be bool [B,2,4]")
        if bool((~availability.any(dim=(1, 2))).any()):
            raise ValueError("every trial requires at least one available modality view")

        gate = self.class_view_gate.unsqueeze(0).expand(view_logits.shape[0], -1, -1, -1)
        gate = gate.masked_fill(
            ~availability[:, None], torch.finfo(view_logits.dtype).min
        )
        weights = torch.softmax(gate, dim=-1)
        weights = torch.softmax(
            torch.logsumexp(gate, dim=-1), dim=-1
        ).unsqueeze(-1) * weights
        weights = weights.masked_fill(~availability[:, None], 0.0)
        logits = (view_logits.permute(0, 3, 1, 2) * weights).sum(dim=(2, 3))
        return logits, weights

    def forward(
        self, *, clips: torch.Tensor, availability: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        expected = (len(MODALITY_NAMES), len(VIEW_NAMES), 3)
        if clips.ndim != 7 or clips.shape[1:4] != expected:
            raise ValueError("clips must have shape [B,2,4,3,T,H,W]")
        if availability.shape != clips.shape[:3] or availability.dtype != torch.bool:
            raise ValueError("availability must be bool [B,2,4]")

        logits_by_view: list[torch.Tensor] = []
        trace: list[str] = []
        for modality_index, modality in enumerate(MODALITY_NAMES):
            modality_logits: list[torch.Tensor] = []
            for view_index, view in enumerate(VIEW_NAMES):
                logits = self.backbone(clips[:, modality_index, view_index])
                if logits.shape != (clips.shape[0], self.num_classes):
                    raise ValueError("backbone must emit [B,C] logits")
                modality_logits.append(logits)
                trace.append(f"{modality}:{view}")
            logits_by_view.append(torch.stack(modality_logits, dim=1))
        view_logits = torch.stack(logits_by_view, dim=1)
        logits, weights = self.fuse_view_logits(view_logits, availability)
        self.last_execution_trace = trace
        return {
            "logits": logits,
            "view_logits": view_logits,
            "class_view_weights": weights,
        }


def sequential_multiview_backward(
    *,
    model: IRDepthVideoMAEV2Teacher,
    clips: torch.Tensor,
    availability: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    """Backpropagate one trial while retaining one VideoMAE clip graph at a time."""
    if clips.shape[0] != 1 or labels.shape != (1,):
        raise ValueError("sequential multiview backward requires one physical trial")
    with torch.no_grad():
        detached_view_logits = model(
            clips=clips, availability=availability
        )["view_logits"].detach()

    proxy = detached_view_logits.requires_grad_(True)
    logits, weights = model.fuse_view_logits(proxy, availability)
    loss = F.cross_entropy(logits.float(), labels, label_smoothing=label_smoothing)
    loss.backward()
    if proxy.grad is None:
        raise RuntimeError("proxy view logits did not receive gradients")
    proxy_gradient = proxy.grad.detach()

    trace: list[str] = []
    for modality_index, modality in enumerate(MODALITY_NAMES):
        for view_index, view in enumerate(VIEW_NAMES):
            gradient = proxy_gradient[:, modality_index, view_index]
            if bool(gradient.abs().sum()):
                replay_clips = clips[:, modality_index, view_index].detach().requires_grad_(True)
                replay_logits = model.backbone(replay_clips)
                torch.autograd.backward(replay_logits, grad_tensors=gradient)
            trace.append(f"{modality}:{view}")
    model.last_execution_trace = trace
    return {
        "loss": loss.detach(),
        "logits": logits.detach(),
        "view_logits": detached_view_logits,
        "class_view_weights": weights.detach(),
    }
