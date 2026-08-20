from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.diagnostics.thermal_bn import (
    apply_bn_linear,
    simulate_bn_running_stats,
    summarize_seed_dispersion,
)


def test_analytic_bn_linear_replay_matches_eval_and_temporal_consensus() -> None:
    torch.manual_seed(7)
    bn = nn.BatchNorm1d(6)
    linear = nn.Linear(6, 4)
    bn.running_mean.copy_(torch.linspace(-1.0, 1.0, 6))
    bn.running_var.copy_(torch.linspace(0.5, 1.5, 6))
    bn.eval()
    frames = torch.randn(5, 16, 6)

    expected = linear(bn(frames.mean(dim=1)))
    replayed = apply_bn_linear(
        frames.mean(dim=1),
        bn=bn,
        linear=linear,
        running_mean=bn.running_mean,
        running_var=bn.running_var,
    )
    frame_consensus = apply_bn_linear(
        frames,
        bn=bn,
        linear=linear,
        running_mean=bn.running_mean,
        running_var=bn.running_var,
    ).mean(dim=1)

    torch.testing.assert_close(replayed, expected)
    torch.testing.assert_close(frame_consensus, expected)


def test_small_physical_batch_has_higher_bn_order_sensitivity() -> None:
    generator = torch.Generator().manual_seed(11)
    low = torch.randn(128, 12, generator=generator) * 0.2 - 4.0
    high = torch.randn(128, 12, generator=generator) * 0.2 + 4.0
    embeddings = torch.cat([low, high], dim=0)
    initial_mean = torch.zeros(12)
    initial_var = torch.ones(12)

    batch4 = simulate_bn_running_stats(
        embeddings,
        batch_size=4,
        epochs=4,
        seeds=range(12),
        initial_mean=initial_mean,
        initial_var=initial_var,
    )
    batch64 = simulate_bn_running_stats(
        embeddings,
        batch_size=64,
        epochs=4,
        seeds=range(12),
        initial_mean=initial_mean,
        initial_var=initial_var,
    )

    dispersion4 = summarize_seed_dispersion(batch4)
    dispersion64 = summarize_seed_dispersion(batch64)

    assert dispersion4["running_mean_cross_seed_std_rms"] > (
        1.5 * dispersion64["running_mean_cross_seed_std_rms"]
    )
    assert dispersion4["running_var_cross_seed_std_rms"] > (
        1.5 * dispersion64["running_var_cross_seed_std_rms"]
    )


def test_diagnostic_entrypoint_contains_no_training_operations() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts/diagnose_thermal_t1b1_bn.py").read_text(
        encoding="utf-8"
    )

    assert "build_optimizer" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "model.train(" not in source


def test_committed_t1b1_report_keeps_training_stopped_and_trials_canonical() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (project_root / "reports/thermal_t1b1_bn_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "root_cause_not_confirmed_training_still_stopped"
    assert not any(report["safety_boundary"].values())
    assert report["root_cause_gate"]["verdict"] == "not_confirmed"
    assert report["root_cause_gate"]["bn_free_short_run_authorized"] is False
    assert report["root_cause_gate"]["bn_free_short_run_recommended"] is False
    assert "no canonical trial removed" in report["embedding_outliers"]["robust_rule"]
    assert report["embedding_outliers"][
        "largest_spike_persists_without_bfloat16_autocast"
    ]
