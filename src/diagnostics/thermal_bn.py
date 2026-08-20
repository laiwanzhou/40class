from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def apply_bn_linear(
    features: torch.Tensor,
    *,
    bn: nn.BatchNorm1d,
    linear: nn.Linear,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
) -> torch.Tensor:
    """Replay an eval-mode BatchNorm1d followed by Linear on the last axis."""
    if features.shape[-1] != bn.num_features:
        raise ValueError("Feature width does not match classifier BatchNorm")
    if running_mean.shape != (bn.num_features,) or running_var.shape != (
        bn.num_features,
    ):
        raise ValueError("Running statistics do not match classifier BatchNorm")
    if bool((running_var < 0).any()):
        raise ValueError("BatchNorm variance must be non-negative")
    normalized = (features - running_mean) * torch.rsqrt(running_var + bn.eps)
    if bn.affine:
        normalized = normalized * bn.weight + bn.bias
    return nn.functional.linear(normalized, linear.weight, linear.bias)


def population_moments(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim < 2 or not features.shape[0]:
        raise ValueError("Population features must be non-empty [...,D]")
    flattened = features.reshape(-1, features.shape[-1]).to(dtype=torch.float64)
    return flattened.mean(dim=0), flattened.var(dim=0, unbiased=False)


def simulate_bn_running_stats(
    embeddings: torch.Tensor,
    *,
    batch_size: int,
    epochs: int,
    seeds: Iterable[int],
    initial_mean: torch.Tensor,
    initial_var: torch.Tensor,
    momentum: float = 0.1,
) -> list[dict[str, torch.Tensor | int]]:
    """Replay PyTorch BN running-stat updates without gradients or model mutation."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("Embeddings must be [N,D] with N >= 2")
    if batch_size < 2 or epochs < 1:
        raise ValueError("Batch size must be >=2 and epochs must be positive")
    if not 0.0 < momentum <= 1.0:
        raise ValueError("Momentum must be in (0,1]")
    width = embeddings.shape[1]
    if initial_mean.shape != (width,) or initial_var.shape != (width,):
        raise ValueError("Initial running statistics do not match embeddings")

    values = embeddings.detach().to(dtype=torch.float64, device="cpu")
    results: list[dict[str, torch.Tensor | int]] = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(int(seed))
        running_mean = initial_mean.detach().to(dtype=torch.float64, device="cpu").clone()
        running_var = initial_var.detach().to(dtype=torch.float64, device="cpu").clone()
        for _ in range(epochs):
            order = torch.randperm(len(values), generator=generator)
            for start in range(0, len(values), batch_size):
                batch = values.index_select(0, order[start : start + batch_size])
                if len(batch) < 2:
                    continue
                batch_mean = batch.mean(dim=0)
                batch_var = batch.var(dim=0, unbiased=True)
                running_mean.lerp_(batch_mean, momentum)
                running_var.lerp_(batch_var, momentum)
        results.append(
            {"seed": int(seed), "running_mean": running_mean, "running_var": running_var}
        )
    if not results:
        raise ValueError("At least one simulation seed is required")
    return results


def summarize_seed_dispersion(
    simulations: list[dict[str, torch.Tensor | int]],
) -> dict[str, float]:
    if len(simulations) < 2:
        raise ValueError("At least two simulations are required")
    means = torch.stack([row["running_mean"] for row in simulations])
    variances = torch.stack([row["running_var"] for row in simulations])
    mean_std = means.std(dim=0, unbiased=False)
    variance_std = variances.std(dim=0, unbiased=False)
    return {
        "running_mean_cross_seed_std_rms": float(mean_std.square().mean().sqrt()),
        "running_mean_cross_seed_std_max": float(mean_std.max()),
        "running_var_cross_seed_std_rms": float(variance_std.square().mean().sqrt()),
        "running_var_cross_seed_std_max": float(variance_std.max()),
    }
