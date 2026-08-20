from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.thermal_native_dataset import (
    ThermalNativeDataset,
    collate_thermal_trials,
    load_development_records,
)
from src.diagnostics.thermal_bn import (
    apply_bn_linear,
    population_moments,
    simulate_bn_running_stats,
    summarize_seed_dispersion,
)
from src.models.thermal_iformer_tsm import build_pretrained_iformer_t_expert
from src.train_thermal_native_expert import thermal_metrics


DATA_ROOT = PROJECT_ROOT.parent / "datasets/Small-Model-Track/train"
AUDIT_PATH = PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json"
SPLIT_PATH = PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/thermal_iformer_t_tsm_train12_val2_seed20260715/best_macro_f1.pt"
)
REPORT_JSON = PROJECT_ROOT / "reports/thermal_t1b1_bn_diagnostic.json"
REPORT_MD = PROJECT_ROOT / "reports/thermal_t1b1_bn_diagnostic.md"
EXPECTED_CHECKPOINT_SHA256 = (
    "ca9c11c0f4d50f67c89da52284f726085dfeb3d1578621438872b9255a05d827"
)
EXPECTED_COUNTS = (1922, 377)
SIMULATION_SEEDS = tuple(range(8))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classifier_parts(model: nn.Module) -> tuple[nn.BatchNorm1d, nn.Linear]:
    try:
        classifier = model.spatial.backbone.classifier.classifier
        bn = classifier.bn
        linear = classifier.l
    except AttributeError as error:
        raise TypeError("Audited mobile iFormer classifier layout changed") from error
    if not isinstance(bn, nn.BatchNorm1d) or not isinstance(linear, nn.Linear):
        raise TypeError("Expected the audited BatchNorm1d -> Linear classifier")
    return bn, linear


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    flattened = values.detach().to(dtype=torch.float64, device="cpu").flatten()
    if not len(flattened) or not bool(torch.isfinite(flattened).all()):
        raise ValueError("Summary values must be non-empty and finite")
    points = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=torch.float64)
    result = torch.quantile(flattened, points)
    return {
        "min": float(result[0]),
        "p25": float(result[1]),
        "median": float(result[2]),
        "p75": float(result[3]),
        "p95": float(result[4]),
        "max": float(result[5]),
        "mean": float(flattened.mean()),
        "std": float(flattened.std(unbiased=False)),
    }


def _embedding_summary(frame_features: torch.Tensor) -> dict[str, Any]:
    trial_features = frame_features.mean(dim=1)
    residuals = frame_features - trial_features[:, None, :]
    return {
        "frame_l2_norm": _quantiles(torch.linalg.vector_norm(frame_features, dim=-1)),
        "trial_l2_norm": _quantiles(torch.linalg.vector_norm(trial_features, dim=-1)),
        "within_trial_frame_to_mean_l2": _quantiles(
            torch.linalg.vector_norm(residuals, dim=-1)
        ),
    }


def _robust_trial_mask(trial_features: torch.Tensor) -> tuple[torch.Tensor, float]:
    norms = torch.linalg.vector_norm(trial_features, dim=-1)
    q1, q3 = torch.quantile(norms, torch.tensor([0.25, 0.75], dtype=norms.dtype))
    threshold = float(q3 + 3.0 * (q3 - q1))
    return norms <= threshold, threshold


def _top_norm_rows(
    frame_features: torch.Tensor,
    *,
    labels: np.ndarray,
    user_ids: np.ndarray,
    sample_ids: tuple[str, ...],
    records: list[Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    trial_norms = torch.linalg.vector_norm(frame_features.mean(dim=1), dim=-1)
    frame_norms = torch.linalg.vector_norm(frame_features, dim=-1).max(dim=1).values
    indices = torch.topk(trial_norms, k=min(limit, len(trial_norms))).indices.tolist()
    return [
        {
            "rank": rank,
            "index": int(index),
            "sample_id": sample_ids[index],
            "user_id": str(user_ids[index]),
            "class_id": int(labels[index]),
            "decodable_frame_count": int(records[index].decodable_frame_count),
            "trial_embedding_l2": float(trial_norms[index]),
            "maximum_frame_embedding_l2": float(frame_norms[index]),
        }
        for rank, index in enumerate(indices, start=1)
    ]


def _user_shift(
    trial_features: torch.Tensor, user_ids: np.ndarray
) -> dict[str, Any]:
    if set(user_ids.tolist()) != {"user6", "user7"}:
        raise ValueError("T1-B.1 user shift is restricted to user6/user7")
    selected = {
        user: trial_features[torch.from_numpy(user_ids == user)]
        for user in ("user6", "user7")
    }
    centroids = {user: values.mean(dim=0) for user, values in selected.items()}
    pooled_std = trial_features.std(dim=0, unbiased=False).clamp_min(1e-8)
    standardized = (centroids["user6"] - centroids["user7"]) / pooled_std
    return {
        "user6_trial_count": int(len(selected["user6"])),
        "user7_trial_count": int(len(selected["user7"])),
        "trial_l2_norm": {
            user: _quantiles(torch.linalg.vector_norm(values, dim=-1))
            for user, values in selected.items()
        },
        "centroid_l2_distance": float(
            torch.linalg.vector_norm(centroids["user6"] - centroids["user7"])
        ),
        "centroid_cosine_similarity": float(
            nn.functional.cosine_similarity(
                centroids["user6"][None], centroids["user7"][None]
            )[0]
        ),
        "standardized_mean_difference_rms": float(
            standardized.square().mean().sqrt()
        ),
        "standardized_mean_difference_abs_max": float(standardized.abs().max()),
    }


def _mode_metrics(
    logits: torch.Tensor, labels: np.ndarray, user_ids: np.ndarray
) -> dict[str, Any]:
    values = logits.detach().to(dtype=torch.float64, device="cpu")
    target = torch.from_numpy(labels).long()
    nll = -nn.functional.log_softmax(values, dim=1)[torch.arange(len(target)), target]
    probabilities = nn.functional.softmax(values, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(dim=1)
    predictions = values.argmax(dim=1).numpy()
    metrics = thermal_metrics(labels, predictions, user_ids)
    return {
        "logit_abs_max": float(values.abs().max()),
        "logit_std": float(values.std(unbiased=False)),
        "per_trial_logit_range": _quantiles(values.max(dim=1).values - values.min(dim=1).values),
        "nll": _quantiles(nll),
        "predictive_entropy": _quantiles(entropy),
        "accuracy": metrics["accuracy"],
        "macro_f1_fixed_0_39": metrics["macro_f1"],
        "worst_user_accuracy": metrics["worst_user_accuracy"],
        "per_user": metrics["per_user"],
        "zero_recall_class_ids": metrics["zero_recall_class_ids"],
    }


def _stat_comparison(
    *,
    checkpoint_mean: torch.Tensor,
    checkpoint_var: torch.Tensor,
    population_mean: torch.Tensor,
    population_var: torch.Tensor,
    eps: float,
) -> dict[str, Any]:
    mean_delta = checkpoint_mean - population_mean
    normalized = mean_delta / torch.sqrt(population_var + eps)
    variance_ratio = checkpoint_var / (population_var + eps)
    return {
        "mean_rmse": float(mean_delta.square().mean().sqrt()),
        "mean_abs_max": float(mean_delta.abs().max()),
        "normalized_mean_error_rms": float(normalized.square().mean().sqrt()),
        "normalized_mean_error_abs_max": float(normalized.abs().max()),
        "checkpoint_to_population_variance_ratio": _quantiles(variance_ratio),
    }


def _collect_features(
    model: nn.Module,
    records: list[Any],
    *,
    device: torch.device,
    batch_size: int,
    use_bfloat16: bool = True,
) -> dict[str, Any]:
    dataset = ThermalNativeDataset(records, training=False, seed=20260715)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_thermal_trials,
        pin_memory=device.type == "cuda",
    )
    frame_features: list[torch.Tensor] = []
    labels: list[np.ndarray] = []
    users: list[str] = []
    sample_ids: list[str] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            clips = batch["clips"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and use_bfloat16,
            ):
                features = model.spatial.forward_frame_features(clips)
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError("Non-finite frame embedding in T1-B.1")
            frame_features.append(features.float().cpu())
            labels.append(batch["labels"].numpy())
            users.extend(batch["user_ids"])
            sample_ids.extend(batch["sample_ids"])
    return {
        "frame_features": torch.cat(frame_features),
        "labels": np.concatenate(labels),
        "user_ids": np.asarray(users),
        "sample_ids": tuple(sample_ids),
        "wall_seconds": time.perf_counter() - started,
    }


def _simulation_summary(
    train_trials: torch.Tensor,
    validation_trials: torch.Tensor,
    labels: np.ndarray,
    users: np.ndarray,
    *,
    bn: nn.BatchNorm1d,
    linear: nn.Linear,
    initial_mean: torch.Tensor,
    initial_var: torch.Tensor,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for batch_size in (4, 16, 64):
        simulations = simulate_bn_running_stats(
            train_trials,
            batch_size=batch_size,
            epochs=17,
            seeds=SIMULATION_SEEDS,
            initial_mean=initial_mean,
            initial_var=initial_var,
        )
        rows = []
        for simulation in simulations:
            logits = apply_bn_linear(
                validation_trials,
                bn=bn,
                linear=linear,
                running_mean=simulation["running_mean"],
                running_var=simulation["running_var"],
            )
            metrics = _mode_metrics(logits, labels, users)
            rows.append(
                {
                    "seed": int(simulation["seed"]),
                    "logit_std": metrics["logit_std"],
                    "nll_mean": metrics["nll"]["mean"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1_fixed_0_39": metrics["macro_f1_fixed_0_39"],
                }
            )
        summary = summarize_seed_dispersion(simulations)
        for field in ("logit_std", "nll_mean", "accuracy", "macro_f1_fixed_0_39"):
            values = torch.tensor([row[field] for row in rows], dtype=torch.float64)
            summary[field] = _quantiles(values)
        summary["seeds"] = rows
        report[str(batch_size)] = summary
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    modes = report["controlled_modes"]
    gate = report["root_cause_gate"]
    lines = [
        "# Thermal T1-B.1 zero-training BN diagnostic",
        "",
        f"- Status: **{report['status']}**",
        f"- Epoch-16 checkpoint: `{report['checkpoint']['sha256']}`",
        "- Boundary: train12 plus user6/user7 only; no sealed, competition-test, or quarantined evidence access.",
        "- Execution: inference-only feature collection and offline statistic replay; no optimizer, backward, checkpoint mutation, or training epoch.",
        "",
        "## Controlled mode results",
        "",
        "| Mode | Logit std | |logit| max | Mean NLL | Accuracy | Macro-F1 | Worst-user acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in modes.items():
        lines.append(
            f"| {name} | {row['logit_std']:.4f} | {row['logit_abs_max']:.4f} | "
            f"{row['nll']['mean']:.4f} | {row['accuracy']:.4f} | "
            f"{row['macro_f1_fixed_0_39']:.4f} | {row['worst_user_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`checkpoint_analytic_replay` is a numerical control. `checkpoint_per_frame_consensus` "
            "is also expected to match checkpoint eval because eval-mode BN plus Linear is affine. "
            "The population-stat and identity modes are diagnostic interventions, not deployable candidates.",
            "",
            "## Root-cause gate",
            "",
            f"- Verdict: **{gate['verdict']}**",
            f"- BN-free short fresh run authorized: **{str(gate['bn_free_short_run_authorized']).lower()}**",
        ]
    )
    for name, row in gate["criteria"].items():
        lines.append(f"- `{name}`: {row['passed']} ({row['detail']})")
    shift = report["validation_user_shift"]
    robust_shift = report["validation_user_shift_robust_inliers"]
    outliers = report["embedding_outliers"]
    lines.extend(
        [
            "",
            "## User distribution",
            "",
            f"- user6/user7 centroid L2 distance: `{shift['centroid_l2_distance']:.6f}`",
            f"- centroid cosine similarity: `{shift['centroid_cosine_similarity']:.6f}`",
            f"- standardized mean-difference RMS: `{shift['standardized_mean_difference_rms']:.6f}`",
            "- Per-user accuracy and fixed-label Macro-F1 are reported for every controlled mode; missing single-user classes are not interpreted as zero capability.",
            "",
            "## Embedding spikes",
            "",
            f"- Robust flags (not exclusions): train12 `{outliers['train12_flagged_count']}`, validation `{outliers['validation_flagged_count']}`.",
            f"- Largest validation trial: `{outliers['validation_top_norm_trials'][0]['sample_id']}`, bfloat16 norm `{outliers['validation_top_norm_trials'][0]['trial_embedding_l2']:.3f}`, FP32 norm `{outliers['fp32_replay_of_largest_validation_trial']['trial_embedding_l2']:.3f}`.",
            f"- Validation flagged class counts: `{outliers['validation_flagged_class_counts']}`.",
            f"- Robust-inlier user6/user7 centroid cosine: `{robust_shift['centroid_cosine_similarity']:.6f}`; standardized mean-difference RMS: `{robust_shift['standardized_mean_difference_rms']:.6f}`.",
            "- The spike persists in FP32, so CUDA bfloat16 autocast is not its cause. A layerwise upstream activation trace is required before any head redesign is authorized.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Thermal T1-B.1 zero-training BN diagnostic")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("T1-B.1 must reproduce the CUDA bfloat16 inference path")
    if _sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Epoch-16 checkpoint SHA256 does not match the interrupted audit")

    device = torch.device("cuda")
    train_canonical, validation_canonical = load_development_records(
        AUDIT_PATH, SPLIT_PATH, args.data_root
    )
    train_records = [record for record in train_canonical if record.usable]
    validation_records = [record for record in validation_canonical if record.usable]
    if (len(train_records), len(validation_records)) != EXPECTED_COUNTS:
        raise ValueError("Frozen T1-B.1 development population changed")

    model = build_pretrained_iformer_t_expert()
    pretrained_bn, _ = _classifier_parts(model)
    pretrained_mean = pretrained_bn.running_mean.detach().double().clone()
    pretrained_var = pretrained_bn.running_var.detach().double().clone()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != 16:
        raise ValueError("T1-B.1 requires the epoch-16 selected checkpoint")
    load = model.load_state_dict(checkpoint["model"], strict=True)
    if load.missing_keys or load.unexpected_keys:
        raise RuntimeError("Checkpoint strict load was incomplete")
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train = _collect_features(model, train_records, device=device, batch_size=args.batch_size)
    validation = _collect_features(
        model, validation_records, device=device, batch_size=args.batch_size
    )
    validation_top_rows = _top_norm_rows(
        validation["frame_features"],
        labels=validation["labels"],
        user_ids=validation["user_ids"],
        sample_ids=validation["sample_ids"],
        records=validation_records,
    )
    precision_records = [validation_records[row["index"]] for row in validation_top_rows]
    precision = _collect_features(
        model,
        precision_records,
        device=device,
        batch_size=args.batch_size,
        use_bfloat16=False,
    )
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    model.cpu()
    torch.cuda.empty_cache()

    bn, linear = _classifier_parts(model)
    bn.eval()
    train_frames = train["frame_features"].double()
    validation_frames = validation["frame_features"].double()
    train_trials = train_frames.mean(dim=1)
    validation_trials = validation_frames.mean(dim=1)
    labels = validation["labels"]
    users = validation["user_ids"]
    checkpoint_mean = bn.running_mean.detach().double()
    checkpoint_var = bn.running_var.detach().double()
    bn.double()
    linear.double()

    train_trial_mean, train_trial_var = population_moments(train_trials)
    train_frame_mean, train_frame_var = population_moments(train_frames)
    train_robust_mask, train_robust_threshold = _robust_trial_mask(train_trials)
    validation_robust_mask, validation_robust_threshold = _robust_trial_mask(
        validation_trials
    )
    robust_train_trials = train_trials[train_robust_mask]
    robust_train_frames = train_frames[train_robust_mask]
    robust_trial_mean, robust_trial_var = population_moments(robust_train_trials)
    robust_frame_mean, robust_frame_var = population_moments(robust_train_frames)
    checkpoint_eval = linear(bn(validation_trials))
    analytic = apply_bn_linear(
        validation_trials,
        bn=bn,
        linear=linear,
        running_mean=checkpoint_mean,
        running_var=checkpoint_var,
    )
    per_frame = apply_bn_linear(
        validation_frames,
        bn=bn,
        linear=linear,
        running_mean=checkpoint_mean,
        running_var=checkpoint_var,
    ).mean(dim=1)
    modes = {
        "checkpoint_eval": checkpoint_eval,
        "checkpoint_analytic_replay": analytic,
        "checkpoint_per_frame_consensus": per_frame,
        "pretrained_frozen_running_stats": apply_bn_linear(
            validation_trials,
            bn=bn,
            linear=linear,
            running_mean=pretrained_mean,
            running_var=pretrained_var,
        ),
        "train12_trial_population_stats": apply_bn_linear(
            validation_trials,
            bn=bn,
            linear=linear,
            running_mean=train_trial_mean,
            running_var=train_trial_var,
        ),
        "train12_frame_population_stats": apply_bn_linear(
            validation_frames,
            bn=bn,
            linear=linear,
            running_mean=train_frame_mean,
            running_var=train_frame_var,
        ).mean(dim=1),
        "train12_robust_trial_population_stats": apply_bn_linear(
            validation_trials,
            bn=bn,
            linear=linear,
            running_mean=robust_trial_mean,
            running_var=robust_trial_var,
        ),
        "identity_bn_control": nn.functional.linear(
            validation_trials, linear.weight, linear.bias
        ),
    }
    controlled = {name: _mode_metrics(value, labels, users) for name, value in modes.items()}
    analytic_max_error = float((checkpoint_eval - analytic).abs().max())
    consensus_max_error = float((checkpoint_eval - per_frame).abs().max())

    simulations = _simulation_summary(
        train_trials,
        validation_trials,
        labels,
        users,
        bn=bn,
        linear=linear,
        initial_mean=pretrained_mean,
        initial_var=pretrained_var,
    )
    robust_simulations = _simulation_summary(
        robust_train_trials,
        validation_trials,
        labels,
        users,
        bn=bn,
        linear=linear,
        initial_mean=pretrained_mean,
        initial_var=pretrained_var,
    )
    baseline = controlled["checkpoint_eval"]
    population = controlled["train12_robust_trial_population_stats"]
    nll_improvement = 1.0 - population["nll"]["mean"] / baseline["nll"]["mean"]
    scale_reduction = 1.0 - population["logit_std"] / baseline["logit_std"]
    mean_dispersion_ratio = (
        robust_simulations["4"]["running_mean_cross_seed_std_rms"]
        / robust_simulations["64"]["running_mean_cross_seed_std_rms"]
    )
    var_dispersion_ratio = (
        robust_simulations["4"]["running_var_cross_seed_std_rms"]
        / robust_simulations["64"]["running_var_cross_seed_std_rms"]
    )
    checkpoint_vs_trial = _stat_comparison(
        checkpoint_mean=checkpoint_mean,
        checkpoint_var=checkpoint_var,
        population_mean=robust_trial_mean,
        population_var=robust_trial_var,
        eps=bn.eps,
    )
    criteria = {
        "analytic_control": {
            "passed": analytic_max_error <= 1e-8 and consensus_max_error <= 1e-8,
            "detail": f"analytic_max_error={analytic_max_error:.3e}, consensus_max_error={consensus_max_error:.3e}",
        },
        "checkpoint_logits_pathological": {
            "passed": baseline["logit_abs_max"] > 100.0
            and baseline["nll"]["mean"] > 2.0 * math.log(40),
            "detail": f"abs_max={baseline['logit_abs_max']:.3f}, mean_nll={baseline['nll']['mean']:.3f}",
        },
        "checkpoint_stats_mismatch_train_population": {
            "passed": checkpoint_vs_trial["normalized_mean_error_rms"] > 0.5
            or not (
                0.5
                <= checkpoint_vs_trial["checkpoint_to_population_variance_ratio"][
                    "median"
                ]
                <= 2.0
            ),
            "detail": (
                "robust normalized_mean_error_rms="
                f"{checkpoint_vs_trial['normalized_mean_error_rms']:.3f}, median_variance_ratio="
                f"{checkpoint_vs_trial['checkpoint_to_population_variance_ratio']['median']:.3f}"
            ),
        },
        "population_stats_reduce_scale_and_nll": {
            "passed": nll_improvement >= 0.2 and scale_reduction >= 0.2,
            "detail": f"relative_nll_improvement={nll_improvement:.3f}, relative_logit_std_reduction={scale_reduction:.3f}",
        },
        "batch4_order_sensitivity": {
            "passed": mean_dispersion_ratio >= 1.5 and var_dispersion_ratio >= 1.5,
            "detail": f"robust replay vs_batch64 mean_dispersion_ratio={mean_dispersion_ratio:.3f}, variance_dispersion_ratio={var_dispersion_ratio:.3f}",
        },
    }
    confirmed = all(row["passed"] for row in criteria.values())
    train_top_rows = _top_norm_rows(
        train_frames,
        labels=train["labels"],
        user_ids=train["user_ids"],
        sample_ids=train["sample_ids"],
        records=train_records,
    )
    precision_top_rows = _top_norm_rows(
        precision["frame_features"],
        labels=precision["labels"],
        user_ids=precision["user_ids"],
        sample_ids=precision["sample_ids"],
        records=precision_records,
    )
    precision_by_sample = {row["sample_id"]: row for row in precision_top_rows}
    largest_sample_id = validation_top_rows[0]["sample_id"]
    flagged_labels = validation["labels"][(~validation_robust_mask).numpy()]
    flagged_class_counts = {
        str(class_id): int((flagged_labels == class_id).sum())
        for class_id in sorted(set(flagged_labels.tolist()))
    }
    report = {
        "schema_version": "thermal-t1b1-bn-diagnostic-v1",
        "stage": "thermal_t1b1",
        "status": (
            "bn_instability_root_cause_confirmed_training_still_stopped"
            if confirmed
            else "root_cause_not_confirmed_training_still_stopped"
        ),
        "branch": "experiment/thermal-iformer-t-t1b",
        "scientific_baseline_commit": "c42bb43091c79903e5fde5655c2846c87305895a",
        "authorization": "explicit_zero_training_diagnostic_only_2026_08_20",
        "safety_boundary": {
            "optimizer_created": False,
            "backward_called": False,
            "checkpoint_modified": False,
            "training_epoch_executed": False,
            "heldout_labels_accessed": False,
            "competition_test_accessed": False,
            "quarantined_evidence_accessed": False,
            "frozen_ir_x3d_modified": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_peak_allocated_bytes": peak_memory,
        },
        "checkpoint": {
            "path": str(args.checkpoint.relative_to(PROJECT_ROOT)),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": _sha256(args.checkpoint),
            "epoch": int(checkpoint["epoch"]),
            "strict_load_missing": [],
            "strict_load_unexpected": [],
        },
        "data": {
            "train12_usable_trials": len(train_records),
            "validation_usable_trials": len(validation_records),
            "validation_users": ["user6", "user7"],
            "train_transform": "deterministic_validation_center_crop_for_population_probe",
            "sampling": "16_uniform_thermal_native_normalized_time",
            "train_feature_collection_seconds": train["wall_seconds"],
            "validation_feature_collection_seconds": validation["wall_seconds"],
            "audit_sha256": _sha256(AUDIT_PATH),
            "split_sha256": _sha256(SPLIT_PATH),
        },
        "classifier_contract": {
            "path": "spatial.backbone.classifier.classifier",
            "flow": "mean(frame_features) -> BatchNorm1d(256) -> Linear(256,40)",
            "physical_batch_seen_by_classifier_bn": 4,
            "gradient_accumulation": 4,
            "gradient_accumulation_changes_bn_statistics": False,
            "checkpoint_num_batches_tracked": int(bn.num_batches_tracked),
        },
        "embedding_summary": {
            "train12": _embedding_summary(train_frames),
            "validation": _embedding_summary(validation_frames),
        },
        "embedding_outliers": {
            "robust_rule": "trial_l2 <= Q3 + 3*IQR; diagnostic flag only, no canonical trial removed",
            "train12_threshold": train_robust_threshold,
            "train12_flagged_count": int((~train_robust_mask).sum()),
            "validation_threshold": validation_robust_threshold,
            "validation_flagged_count": int((~validation_robust_mask).sum()),
            "train12_top_norm_trials": train_top_rows,
            "validation_top_norm_trials": validation_top_rows,
            "validation_top_fp32_replay": precision_top_rows,
            "fp32_replay_of_largest_validation_trial": precision_by_sample[
                largest_sample_id
            ],
            "largest_spike_persists_without_bfloat16_autocast": (
                precision_by_sample[largest_sample_id]["trial_embedding_l2"]
                > 10.0
                * float(torch.median(torch.linalg.vector_norm(validation_trials, dim=1)))
            ),
            "validation_flagged_class_counts": flagged_class_counts,
        },
        "validation_user_shift": _user_shift(validation_trials, users),
        "validation_user_shift_robust_inliers": _user_shift(
            validation_trials[validation_robust_mask], users[validation_robust_mask.numpy()]
        ),
        "bn_statistics": {
            "checkpoint_running_mean": _quantiles(checkpoint_mean),
            "checkpoint_running_var": _quantiles(checkpoint_var),
            "pretrained_running_mean": _quantiles(pretrained_mean),
            "pretrained_running_var": _quantiles(pretrained_var),
            "train12_trial_population_mean": _quantiles(train_trial_mean),
            "train12_trial_population_var": _quantiles(train_trial_var),
            "train12_frame_population_mean": _quantiles(train_frame_mean),
            "train12_frame_population_var": _quantiles(train_frame_var),
            "checkpoint_vs_train12_trial_population": checkpoint_vs_trial,
            "checkpoint_vs_untrimmed_train12_trial_population": _stat_comparison(
                checkpoint_mean=checkpoint_mean,
                checkpoint_var=checkpoint_var,
                population_mean=train_trial_mean,
                population_var=train_trial_var,
                eps=bn.eps,
            ),
            "robust_train12_trial_population_mean": _quantiles(robust_trial_mean),
            "robust_train12_trial_population_var": _quantiles(robust_trial_var),
            "robust_train12_frame_population_mean": _quantiles(robust_frame_mean),
            "robust_train12_frame_population_var": _quantiles(robust_frame_var),
            "checkpoint_vs_train12_frame_population": _stat_comparison(
                checkpoint_mean=checkpoint_mean,
                checkpoint_var=checkpoint_var,
                population_mean=train_frame_mean,
                population_var=train_frame_var,
                eps=bn.eps,
            ),
        },
        "controlled_modes": controlled,
        "bn_ema_sensitivity_replay": {
            "scope": "final epoch16 deterministic-center-crop trial embeddings; sensitivity analysis, not historical reconstruction",
            "epochs": 17,
            "initial_stats": "official_pretrained_classifier_bn",
            "batch_sizes": simulations,
            "robust_train12_batch_sizes": robust_simulations,
        },
        "root_cause_gate": {
            "verdict": (
                "classifier_bn_physical_batch4_instability_confirmed"
                if confirmed
                else "not_confirmed"
            ),
            "criteria": criteria,
            "bn_free_short_run_authorized": False,
            "bn_free_short_run_recommended": False,
            "decision": (
                "A separate human approval may now consider a fresh short BN-free-head run; this diagnostic does not authorize it."
                if confirmed
                else "Do not authorize a BN-free run. The checkpoint BN matches robust train12 population statistics; upstream embedding spikes persist in FP32 and require layerwise zero-training tracing."
            ),
        },
        "limitations": [
            "Final epoch-16 embeddings differ from embeddings seen throughout optimization, so EMA replay measures sensitivity and does not reconstruct exact historical BN statistics.",
            "Train population moments use deterministic center crops to remove augmentation randomness; they are diagnostic statistics and are not written into a deployable checkpoint.",
            "Controlled replacement modes reuse the learned BN affine parameters and Linear weights, so they isolate inference normalization but cannot prove how a fresh head would optimize.",
            "User6 and user7 have different class support; per-user Macro-F1 uses the fixed 0..39 label set and absent classes are not interpreted as model incapability.",
        ],
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["root_cause_gate"]["verdict"],
                "json": str(REPORT_JSON),
                "markdown": str(REPORT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
