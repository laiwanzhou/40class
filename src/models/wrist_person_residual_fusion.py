from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class WristPersonResidualFusion(nn.Module):
    """Keep the wrist classifier as anchor and add a bounded person residual."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_classes: int,
        class_embedding_dim: int = 16,
        hidden_dim: int = 128,
        gating_mode: str = "margin",
        maximum_logit_delta: float = 2.0,
    ) -> None:
        super().__init__()
        if gating_mode not in {"fixed10", "no_margin", "margin"}:
            raise ValueError("unknown wrist-person gating mode")
        if embedding_dim <= 0 or num_classes < 3 or maximum_logit_delta <= 0:
            raise ValueError("invalid wrist-person dimensions")
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.gating_mode = str(gating_mode)
        self.maximum_logit_delta = float(maximum_logit_delta)
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        sample_features = 6
        residual_features = embedding_dim * 3 + class_embedding_dim + sample_features
        self.residual = nn.Sequential(
            nn.LayerNorm(residual_features),
            nn.Linear(residual_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.person_adapter_norm = nn.LayerNorm(embedding_dim)
        self.person_adapter = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.person_adapter.weight)
        nn.init.zeros_(self.person_adapter.bias)
        self.person_head = nn.Linear(embedding_dim, num_classes)
        self.person_norm = nn.LayerNorm(embedding_dim)
        self.gate_context = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, max(8, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(8, hidden_dim // 4), num_classes),
        )
        nn.init.zeros_(self.gate_context[-1].weight)
        nn.init.zeros_(self.gate_context[-1].bias)
        self.gate_intercept = nn.Parameter(torch.zeros(num_classes))
        self.gate_raw_slope = nn.Parameter(torch.full((num_classes,), 0.5413248546))

    def forward(
        self,
        *,
        view_embeddings: torch.Tensor,
        anchor_logits: torch.Tensor,
        anchor_view_weights: torch.Tensor,
        availability: torch.Tensor,
        num_frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if view_embeddings.ndim != 3 or view_embeddings.shape[1:] != (
            4,
            self.embedding_dim,
        ):
            raise ValueError("view_embeddings must have shape [B,4,D]")
        batch = view_embeddings.shape[0]
        if anchor_logits.shape != (batch, self.num_classes):
            raise ValueError("anchor_logits must have shape [B,C]")
        if anchor_view_weights.shape != (batch, self.num_classes, 4):
            raise ValueError("anchor_view_weights must have shape [B,C,4]")
        if availability.shape != (batch, 4) or availability.dtype != torch.bool:
            raise ValueError("availability must be bool [B,4]")
        if num_frames.shape != (batch,):
            raise ValueError("num_frames must have shape [B]")

        wrist_weights = anchor_view_weights[:, :, 2:].clamp_min(0)
        wrist_available = availability[:, 2:][:, None].to(wrist_weights.dtype)
        wrist_weights = wrist_weights * wrist_available
        wrist_mass = wrist_weights.sum(dim=2, keepdim=True)
        normalized_wrist = wrist_weights / wrist_mass.clamp_min(1e-12)
        hand_features = torch.einsum(
            "bcv,bvd->bcd", normalized_wrist, view_embeddings[:, 2:]
        )
        raw_person_features = view_embeddings[:, 1]
        person_features = raw_person_features + self.person_adapter(
            self.person_adapter_norm(raw_person_features)
        )
        no_wrist = wrist_mass.squeeze(2) == 0
        hand_features = torch.where(
            no_wrist[:, :, None], person_features[:, None], hand_features
        )

        top_values = anchor_logits.topk(3, dim=1).values
        margin_12 = top_values[:, 0] - top_values[:, 1]
        margin_13 = top_values[:, 0] - top_values[:, 2]
        probabilities = torch.softmax(anchor_logits, dim=1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)
        hand_summary = hand_features.mean(dim=1)
        cosine_distance = 1.0 - F.cosine_similarity(hand_summary, person_features, dim=1)
        norm_ratio = person_features.norm(dim=1) / hand_summary.norm(dim=1).clamp_min(1e-6)
        log_frames = torch.log1p(num_frames.to(anchor_logits.dtype)) / 6.0
        sample_features = torch.stack(
            (
                torch.log1p(margin_12.clamp_min(0)),
                torch.log1p(margin_13.clamp_min(0)),
                entropy / math.log(self.num_classes),
                cosine_distance,
                torch.log1p(norm_ratio),
                log_frames,
            ),
            dim=1,
        )

        person_by_class = person_features[:, None].expand(-1, self.num_classes, -1)
        class_ids = torch.arange(self.num_classes, device=view_embeddings.device)
        class_features = self.class_embedding(class_ids)[None].expand(batch, -1, -1)
        expanded_sample = sample_features[:, None].expand(-1, self.num_classes, -1)
        residual_inputs = torch.cat(
            (
                hand_features,
                person_by_class,
                person_by_class - hand_features,
                class_features,
                expanded_sample,
            ),
            dim=2,
        )
        raw_delta = self.residual(residual_inputs).squeeze(2)
        bounded_delta = self.maximum_logit_delta * torch.tanh(raw_delta)

        person_available = availability[:, 1]
        if self.gating_mode == "fixed10":
            person_gate = anchor_logits.new_full(anchor_logits.shape, 0.10)
        elif self.gating_mode == "no_margin":
            person_gate = torch.ones_like(anchor_logits)
        else:
            gate_inputs = torch.stack(
                (cosine_distance, torch.log1p(norm_ratio), log_frames), dim=1
            )
            context = self.gate_context(gate_inputs)
            slope = F.softplus(self.gate_raw_slope)[None]
            person_gate = torch.sigmoid(
                self.gate_intercept[None]
                + context
                - slope * torch.log1p(margin_13.clamp_min(0))[:, None]
            )
        person_gate = person_gate * person_available[:, None].to(person_gate.dtype)
        delta_logits = person_gate * bounded_delta
        logits = anchor_logits + delta_logits
        person_logits = self.person_head(self.person_norm(person_features))
        return {
            "logits": logits,
            "anchor_logits": anchor_logits,
            "person_logits": person_logits,
            "person_gate": person_gate,
            "delta_logits": delta_logits,
            "bounded_person_delta": bounded_delta,
            "person_available": person_available,
            "wrist_fallback": no_wrist.all(dim=1),
        }


def wrist_person_loss(
    *,
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    person_available: torch.Tensor,
    person_aux_weight: float,
    guard_weight: float,
) -> dict[str, torch.Tensor]:
    fused_ce = F.cross_entropy(output["logits"], labels, reduction="none")
    anchor_ce = F.cross_entropy(
        output["anchor_logits"].detach(), labels, reduction="none"
    )
    person_ce = F.cross_entropy(output["person_logits"], labels, reduction="none")
    available = person_available.to(person_ce.dtype)
    auxiliary = (person_ce * available).sum() / available.sum().clamp_min(1.0)
    guard = torch.relu(fused_ce - anchor_ce).mean()
    loss = fused_ce.mean() + float(person_aux_weight) * auxiliary + float(
        guard_weight
    ) * guard
    return {
        "loss": loss,
        "fused_ce": fused_ce.mean(),
        "person_ce": auxiliary,
        "guard_loss": guard,
        "anchor_ce": anchor_ce.mean(),
    }
