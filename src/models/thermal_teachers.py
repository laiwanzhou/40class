from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torchvision.models.video import r2plus1d_18


OFFICIAL_R2PLUS1D18 = {
    "paper_name": "A Closer Look at Spatiotemporal Convolutions for Action Recognition",
    "repository": "https://github.com/pytorch/vision",
    "repository_revision": "9eb57cd5c96be7fe31923eb65399c3819d064587",
    "torchvision_version": "0.22.0",
    "license": "BSD-3-Clause",
    "license_scope": "torchvision source code; pretrained weights may inherit training-dataset terms",
    "checkpoint_url": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
    "checkpoint_sha256": "91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8",
    "checkpoint_bytes": 126_162_996,
    "pretraining_dataset": "Kinetics-400",
    "weights_enum": "R2Plus1D_18_Weights.KINETICS400_V1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_checkpoint_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / "r2plus1d_18-91a641e6.pth"


def build_official_r2plus1d18(
    *, checkpoint_path: Path | None = None, num_classes: int = 40
) -> tuple[nn.Module, dict[str, object]]:
    path = (checkpoint_path or default_checkpoint_path()).resolve()
    if not path.is_file():
        torch.hub.load_state_dict_from_url(
            OFFICIAL_R2PLUS1D18["checkpoint_url"],
            model_dir=str(path.parent),
            file_name=path.name,
            check_hash=True,
            map_location="cpu",
        )
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != OFFICIAL_R2PLUS1D18["checkpoint_bytes"]:
        raise RuntimeError(f"official checkpoint byte mismatch: {size}")
    if digest != OFFICIAL_R2PLUS1D18["checkpoint_sha256"]:
        raise RuntimeError(f"official checkpoint SHA256 mismatch: {digest}")
    model = r2plus1d_18(weights=None)
    state = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    provenance = {
        **OFFICIAL_R2PLUS1D18,
        "checkpoint_path": str(path),
        "strict_load": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "classifier_replaced_after_strict_load": True,
        "classifier_in_features": in_features,
        "classifier_out_features": num_classes,
        "preprocessing_identity": "thermal_v2_train12_rgb_not_kinetics_transform",
        "competition_use_basis": "user-confirmed training-only large pretrained teachers are permitted",
    }
    return model, provenance


class ThermalR2Plus1D18Teacher(nn.Module):
    """Training-only C1 teacher with sequential trial clip aggregation."""

    def __init__(
        self,
        *,
        backbone: nn.Module | None = None,
        checkpoint_path: Path | None = None,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone, provenance = build_official_r2plus1d18(
                checkpoint_path=checkpoint_path, num_classes=num_classes
            )
        else:
            provenance = {
                "injected_test_backbone": True,
                "preprocessing_identity": "thermal_v2_train12_rgb_not_kinetics_transform",
            }
        self.backbone = backbone
        self.initialization_provenance = provenance
        self.last_execution_trace: list[str] = []

    def iter_available_clips(
        self,
        full_rgb: torch.Tensor,
        crop_rgb: torch.Tensor,
        *,
        window_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        if full_rgb.ndim != 6 or crop_rgb.shape != full_rgb.shape:
            raise ValueError("teacher raster inputs must share [B,W,C,T,H,W]")
        batch, windows = full_rgb.shape[:2]
        if window_mask.shape != (batch, windows) or availability.shape != (batch, 2):
            raise ValueError("teacher masks have incompatible shapes")
        for window in range(windows):
            active_full = window_mask[:, window] & availability[:, 0]
            if bool(active_full.any()):
                yield f"full:{window}", full_rgb[:, window], active_full
            active_crop = window_mask[:, window] & availability[:, 1]
            if bool(active_crop.any()):
                yield f"crop:{window}", crop_rgb[:, window], active_crop

    def forward(
        self,
        full_rgb: torch.Tensor,
        crop_rgb: torch.Tensor,
        *,
        window_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits_sum: torch.Tensor | None = None
        counts = torch.zeros(full_rgb.shape[0], device=full_rgb.device, dtype=torch.long)
        trace: list[str] = []
        for name, clip, active in self.iter_available_clips(
            full_rgb, crop_rgb, window_mask=window_mask, availability=availability
        ):
            clip_logits = self.backbone(clip)
            if clip_logits.ndim != 2:
                raise ValueError("teacher backbone must return [B,C] logits")
            weighted = clip_logits * active[:, None].to(clip_logits.dtype)
            logits_sum = weighted if logits_sum is None else logits_sum + weighted
            counts += active.long()
            trace.append(name)
        if logits_sum is None or bool((counts == 0).any()):
            raise ValueError("every teacher trial requires at least one available clip")
        logits = logits_sum / counts[:, None].to(logits_sum.dtype)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite teacher logits")
        self.last_execution_trace = trace
        return {"logits": logits, "clip_count": counts}

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        classifier = getattr(self.backbone, "fc", None)
        classifier_params = list(classifier.parameters()) if isinstance(classifier, nn.Module) else []
        classifier_ids = {id(parameter) for parameter in classifier_params}
        backbone_params = [
            parameter for parameter in self.backbone.parameters()
            if id(parameter) not in classifier_ids
        ]
        return backbone_params, classifier_params
