from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def reroute_cached_embeddings(
    *,
    view_embeddings: torch.Tensor,
    class_queries: torch.Tensor,
    class_view_bias: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor | None,
    availability: torch.Tensor,
    mode: str,
    top_k: int = 2,
    context_share: float = 0.25,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Rebuild class logits from cached [global, person, left, right] embeddings."""

    if view_embeddings.ndim != 3 or view_embeddings.shape[1] != 4:
        raise ValueError("view_embeddings must have shape [B,4,D]")
    batch, _, embedding_dim = view_embeddings.shape
    if class_queries.ndim != 2 or class_queries.shape[1] != embedding_dim:
        raise ValueError("class_queries must have shape [C,D]")
    classes = class_queries.shape[0]
    if class_view_bias.shape != (classes, 4):
        raise ValueError("class_view_bias must have shape [C,4]")
    if head_weight.shape != (classes, embedding_dim):
        raise ValueError("head_weight must have shape [C,D]")
    if head_bias is not None and head_bias.shape != (classes,):
        raise ValueError("head_bias must have shape [C]")
    if availability.shape != (batch, 4) or availability.dtype != torch.bool:
        raise ValueError("availability must be bool [B,4]")
    if bool((~availability.any(dim=1)).any()):
        raise ValueError("every sample needs an available view")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scores = torch.einsum("bvd,cd->bcv", view_embeddings, class_queries)
    scores = scores / math.sqrt(embedding_dim) + class_view_bias[None]
    minimum = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~availability[:, None], minimum)

    selected_views: torch.Tensor
    if mode == "hard":
        if not 1 <= top_k <= 4:
            raise ValueError("top_k must be in [1,4]")
        selected_scores, selected_views = scores.topk(top_k, dim=2)
        sparse_scores = torch.full_like(scores, minimum)
        sparse_scores.scatter_(2, selected_views, selected_scores)
        view_weights = torch.softmax(sparse_scores / temperature, dim=2)
    elif mode == "soft":
        view_weights = torch.softmax(scores / temperature, dim=2)
        selected_views = scores.argmax(dim=2, keepdim=True)
    elif mode == "grouped":
        if not 0.0 <= context_share <= 1.0:
            raise ValueError("context_share must be in [0,1]")
        if bool((~availability[:, :2].any(dim=1)).any()):
            raise ValueError("every sample needs a context view")
        context_weights = torch.softmax(scores[:, :, :2] / temperature, dim=2)
        wrist_weights = torch.softmax(scores[:, :, 2:] / temperature, dim=2)
        wrist_available = availability[:, 2:].any(dim=1)[:, None, None]
        context_mass = torch.where(
            wrist_available,
            scores.new_full((batch, 1, 1), float(context_share)),
            scores.new_ones((batch, 1, 1)),
        )
        view_weights = torch.cat(
            (
                context_weights * context_mass,
                wrist_weights * (1.0 - context_mass),
            ),
            dim=2,
        )
        selected_views = torch.stack(
            (scores[:, :, :2].argmax(dim=2), scores[:, :, 2:].argmax(dim=2) + 2),
            dim=2,
        )
    else:
        raise ValueError(f"unknown routing mode: {mode}")

    class_features = torch.einsum("bcv,bvd->bcd", view_weights, view_embeddings)
    logits = torch.einsum("bcd,cd->bc", class_features, head_weight)
    if head_bias is not None:
        logits = logits + head_bias[None]
    return {
        "logits": logits,
        "view_weights": view_weights,
        "selected_views": selected_views,
        "class_features": class_features,
    }


def build_route_bank(
    *,
    fused_view_embeddings: torch.Tensor,
    ir_view_embeddings: torch.Tensor,
    class_queries: torch.Tensor,
    class_view_bias: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor | None,
    availability: torch.Tensor,
) -> dict[str, torch.Tensor | tuple[str, ...]]:
    """Build the fixed routing controls used by the cached P3-R1 experiment."""

    if ir_view_embeddings.shape != fused_view_embeddings.shape:
        raise ValueError("IR and fused view embeddings must have identical shapes")
    common = {
        "class_queries": class_queries,
        "class_view_bias": class_view_bias,
        "head_weight": head_weight,
        "head_bias": head_bias,
        "availability": availability,
    }
    route_specs = (
        ("full_hard2", fused_view_embeddings, {"mode": "hard", "top_k": 2}),
        ("ir_hard2", ir_view_embeddings, {"mode": "hard", "top_k": 2}),
        ("full_hard3", fused_view_embeddings, {"mode": "hard", "top_k": 3}),
        ("full_hard4", fused_view_embeddings, {"mode": "hard", "top_k": 4}),
        ("full_soft", fused_view_embeddings, {"mode": "soft"}),
        (
            "full_group_context10",
            fused_view_embeddings,
            {"mode": "grouped", "context_share": 0.10},
        ),
        (
            "full_group_context25",
            fused_view_embeddings,
            {"mode": "grouped", "context_share": 0.25},
        ),
        (
            "full_group_context50",
            fused_view_embeddings,
            {"mode": "grouped", "context_share": 0.50},
        ),
        (
            "ir_group_context25",
            ir_view_embeddings,
            {"mode": "grouped", "context_share": 0.25},
        ),
    )
    names: list[str] = []
    logits: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for name, embeddings, options in route_specs:
        output = reroute_cached_embeddings(
            view_embeddings=embeddings, **common, **options
        )
        names.append(name)
        logits.append(output["logits"])
        weights.append(output["view_weights"])

    context_availability = availability.clone()
    context_availability[:, 2:] = False
    wrist_availability = availability.clone()
    wrist_availability[:, :2] = False
    no_wrist = ~wrist_availability.any(dim=1)
    wrist_availability[no_wrist] = availability[no_wrist]
    for name, masked_availability in (
        ("full_context_only", context_availability),
        ("full_wrists_only", wrist_availability),
    ):
        output = reroute_cached_embeddings(
            view_embeddings=fused_view_embeddings,
            class_queries=class_queries,
            class_view_bias=class_view_bias,
            head_weight=head_weight,
            head_bias=head_bias,
            availability=masked_availability,
            mode="hard",
            top_k=2,
        )
        names.append(name)
        logits.append(output["logits"])
        weights.append(output["view_weights"])

    return {
        "route_names": tuple(names),
        "route_logits": torch.stack(logits, dim=1),
        "route_view_weights": torch.stack(weights, dim=1),
    }


class MarginConditionedTop3Reranker(nn.Module):
    """A bounded residual over anchor Top-3 classes with a monotonic confidence gate."""

    def __init__(
        self,
        *,
        num_classes: int,
        route_count: int,
        class_embedding_dim: int = 16,
        hidden_dim: int = 64,
        use_margin_gate: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 5 or route_count < 2:
            raise ValueError("reranker requires at least five classes and two routes")
        self.num_classes = int(num_classes)
        self.route_count = int(route_count)
        self.use_margin_gate = bool(use_margin_gate)
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        sample_feature_dim = 5
        candidate_feature_dim = 1 + route_count * 2 + class_embedding_dim + sample_feature_dim
        self.residual = nn.Sequential(
            nn.LayerNorm(candidate_feature_dim),
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.gate_intercept = nn.Parameter(torch.tensor(0.0))
        self.gate_raw_slope = nn.Parameter(torch.tensor(0.5413248546))

    def forward(
        self,
        *,
        anchor_logits: torch.Tensor,
        route_logits: torch.Tensor,
        num_frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if anchor_logits.ndim != 2 or anchor_logits.shape[1] != self.num_classes:
            raise ValueError("anchor_logits must have shape [B,C]")
        batch = anchor_logits.shape[0]
        if route_logits.shape != (batch, self.route_count, self.num_classes):
            raise ValueError("route_logits must have shape [B,R,C]")
        if num_frames.shape != (batch,):
            raise ValueError("num_frames must have shape [B]")

        top_values, top_indices = anchor_logits.topk(5, dim=1)
        candidate_indices = top_indices[:, :3]
        candidate_mask = torch.zeros_like(anchor_logits, dtype=torch.bool)
        candidate_mask.scatter_(1, candidate_indices, True)

        probabilities = torch.softmax(anchor_logits, dim=1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)
        margin_12 = top_values[:, 0] - top_values[:, 1]
        margin_13 = top_values[:, 0] - top_values[:, 2]
        margin_15 = top_values[:, 0] - top_values[:, 4]
        sample_features = torch.stack(
            (
                torch.log1p(margin_12.clamp_min(0)),
                torch.log1p(margin_13.clamp_min(0)),
                torch.log1p(margin_15.clamp_min(0)),
                entropy / math.log(self.num_classes),
                torch.log1p(num_frames.to(anchor_logits.dtype)) / 6.0,
            ),
            dim=1,
        )

        route_by_class = route_logits.transpose(1, 2)
        anchor_feature = anchor_logits[:, :, None]
        differences = route_by_class - anchor_feature
        class_ids = torch.arange(self.num_classes, device=anchor_logits.device)
        class_features = self.class_embedding(class_ids)[None].expand(batch, -1, -1)
        expanded_sample = sample_features[:, None].expand(-1, self.num_classes, -1)
        features = torch.cat(
            (anchor_feature, route_by_class, differences, class_features, expanded_sample),
            dim=2,
        )
        raw_delta = self.residual(features).squeeze(2)
        if self.use_margin_gate:
            slope = F.softplus(self.gate_raw_slope)
            margin_gate = torch.sigmoid(
                self.gate_intercept - slope * torch.log1p(margin_13.clamp_min(0))
            )
        else:
            margin_gate = torch.ones_like(margin_13)
        proposed_delta = raw_delta * margin_gate[:, None] * candidate_mask.to(raw_delta.dtype)
        proposed_logits = anchor_logits + proposed_delta
        outside_maximum = anchor_logits.masked_fill(
            candidate_mask, torch.finfo(anchor_logits.dtype).min
        ).max(dim=1).values
        candidate_floor = torch.nextafter(
            outside_maximum, torch.full_like(outside_maximum, float("inf"))
        )
        corrected_logits = torch.where(
            candidate_mask,
            torch.maximum(proposed_logits, candidate_floor[:, None]),
            anchor_logits,
        )
        delta_logits = corrected_logits - anchor_logits
        return {
            "logits": corrected_logits,
            "delta_logits": delta_logits,
            "candidate_mask": candidate_mask,
            "candidate_indices": candidate_indices,
            "margin_gate": margin_gate,
        }
