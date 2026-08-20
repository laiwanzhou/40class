from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_thermal_backbones import DeploymentAsset, build_deployment_ledger
from src.data.ir_primary_full_sequence_dataset import class_map_hash
from src.data.thermal_native_dataset import (
    ThermalNativeDataset,
    collate_thermal_trials,
    load_development_records,
)
from src.diagnostics.activation_trace import (
    ReadOnlyActivationTracer,
    activation_summary,
    state_dict_digest,
)
from src.models.thermal_mobilenet_tsm import build_pretrained_mobilenet_expert
from src.train_thermal_native_expert import (
    T1BRecipe,
    build_optimizer,
    run_epoch,
    seed_everything,
    train_development,
)


DATA_ROOT = PROJECT_ROOT.parent / "datasets/Small-Model-Track/train"
AUDIT_PATH = PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json"
SPLIT_PATH = PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json"
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/thermal_mobilenetv3_tsm_train12_val2.yaml"
T1A_PATH = PROJECT_ROOT / "reports/thermal_backbone_environment_probe.json"
T1B1_PATH = PROJECT_ROOT / "reports/thermal_t1b1_bn_diagnostic.json"
T1B3_PATH = PROJECT_ROOT / "reports/thermal_t1b3_block_attribution.json"
T1B4_PATH = PROJECT_ROOT / "reports/thermal_t1b4_head_only_probe.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs/thermal_mobilenetv3_tsm_train12_val2_seed20260715"
REPORT_JSON = PROJECT_ROOT / "reports/thermal_mobilenetv3_tsm_train12_val2.json"
REPORT_MD = PROJECT_ROOT / "reports/thermal_mobilenetv3_tsm_train12_val2.md"
WEIGHT_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
PRECISIONS = ("fp32", "bfloat16")
AMPLIFICATION_THRESHOLD = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_hash(records: list[Any]) -> str:
    rows = pd.DataFrame(
        sorted({(record.class_id, record.action_name) for record in records}),
        columns=["class_id", "action_name"],
    )
    if len(rows) != 40:
        raise ValueError("Thermal development inventory must contain all 40 classes")
    return class_map_hash(rows)


def _load_clip(record: Any) -> torch.Tensor:
    item = ThermalNativeDataset([record], training=False, seed=20260715)[0]
    if item["route"] != "full_frame" or not bool(item["availability"]):
        raise ValueError("Matched control trace requires usable full-frame Thermal input")
    clips = item["clips"].unsqueeze(0)
    if clips.shape != (1, 16, 3, 224, 224):
        raise ValueError("Matched control input contract changed")
    return clips


def _forward(
    model: torch.nn.Module, clips: torch.Tensor, *, precision: str
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(
        device_type=clips.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bfloat16",
    ):
        embedding = model.spatial.forward_features(clips)
        logits = model.spatial.classifier(embedding)
    return embedding, logits


def _trace_one(
    model: torch.nn.Module, clips: torch.Tensor, *, precision: str
) -> dict[str, Any]:
    module_names = [
        f"spatial.features.{index}" for index in range(len(model.spatial.features))
    ]
    with torch.inference_mode():
        baseline_embedding, baseline_logits = _forward(model, clips, precision=precision)
        with ReadOnlyActivationTracer(model, module_names) as tracer:
            traced_embedding, traced_logits = _forward(model, clips, precision=precision)
    logits_exact = torch.equal(baseline_logits, traced_logits)
    embeddings_exact = torch.equal(baseline_embedding, traced_embedding)
    if not logits_exact or not embeddings_exact:
        raise RuntimeError("MobileNet hooks changed logits or embeddings")
    return {
        "precision": precision,
        "embedding": activation_summary(traced_embedding),
        "logits": activation_summary(traced_logits),
        "predicted_class_id": int(traced_logits.argmax(dim=1)[0]),
        "logits_exact_with_and_without_hooks": logits_exact,
        "embedding_exact_with_and_without_hooks": embeddings_exact,
        "feature_blocks": tracer.records,
    }


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _matched_stability_trace(
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    records_by_id: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    sample_ids = sorted(
        {pair["spike_sample_id"] for pair in pairs}
        | {pair["control_sample_id"] for pair in pairs}
    )
    traces: dict[str, Any] = {}
    for sample_id in sample_ids:
        clips = _load_clip(records_by_id[sample_id]).to(device)
        traces[sample_id] = {
            precision: _trace_one(model, clips, precision=precision)
            for precision in PRECISIONS
        }
    comparisons: dict[str, Any] = {}
    for precision in PRECISIONS:
        rows: list[dict[str, Any]] = []
        for pair in pairs:
            spike = traces[pair["spike_sample_id"]][precision]
            control = traces[pair["control_sample_id"]][precision]
            if len(spike["feature_blocks"]) != len(control["feature_blocks"]):
                raise RuntimeError("MobileNet feature trace lengths differ")
            block_ratios = []
            for spike_block, control_block in zip(
                spike["feature_blocks"], control["feature_blocks"], strict=True
            ):
                if spike_block["point"] != control_block["point"]:
                    raise RuntimeError("MobileNet feature trace points differ")
                block_ratios.append(
                    {
                        "point": spike_block["point"],
                        "rms_ratio": spike_block["rms"] / max(control_block["rms"], 1e-12),
                        "abs_max_ratio": spike_block["abs_max"]
                        / max(control_block["abs_max"], 1e-12),
                    }
                )
            rows.append(
                {
                    **pair,
                    "feature_block_ratios": block_ratios,
                    "maximum_feature_block_rms_ratio": max(
                        row["rms_ratio"] for row in block_ratios
                    ),
                    "embedding_rms_ratio": spike["embedding"]["rms"]
                    / max(control["embedding"]["rms"], 1e-12),
                    "embedding_abs_max_ratio": spike["embedding"]["abs_max"]
                    / max(control["embedding"]["abs_max"], 1e-12),
                    "logit_rms_ratio": spike["logits"]["rms"]
                    / max(control["logits"]["rms"], 1e-12),
                    "logit_abs_max_ratio": spike["logits"]["abs_max"]
                    / max(control["logits"]["abs_max"], 1e-12),
                }
            )
        block_points = [row["point"] for row in rows[0]["feature_block_ratios"]]
        block_aggregate = {
            point: {
                "median_spike_control_rms_ratio": _median(
                    [
                        row["feature_block_ratios"][index]["rms_ratio"]
                        for row in rows
                    ]
                ),
                "maximum_spike_control_rms_ratio": max(
                    row["feature_block_ratios"][index]["rms_ratio"] for row in rows
                ),
            }
            for index, point in enumerate(block_points)
        }
        comparisons[precision] = {
            "pairs": rows,
            "feature_block_aggregate": block_aggregate,
            "aggregate": {
                "median_maximum_feature_block_rms_ratio": _median(
                    [row["maximum_feature_block_rms_ratio"] for row in rows]
                ),
                "maximum_feature_block_rms_ratio": max(
                    row["maximum_feature_block_rms_ratio"] for row in rows
                ),
                "median_embedding_rms_ratio": _median(
                    [row["embedding_rms_ratio"] for row in rows]
                ),
                "maximum_embedding_rms_ratio": max(
                    row["embedding_rms_ratio"] for row in rows
                ),
                "median_logit_rms_ratio": _median([row["logit_rms_ratio"] for row in rows]),
                "maximum_logit_rms_ratio": max(row["logit_rms_ratio"] for row in rows),
                "feature_or_embedding_10x_event": any(
                    row["maximum_feature_block_rms_ratio"] >= AMPLIFICATION_THRESHOLD
                    or row["embedding_rms_ratio"] >= AMPLIFICATION_THRESHOLD
                    for row in rows
                ),
            },
        }
    hooks_exact = all(
        row["logits_exact_with_and_without_hooks"]
        and row["embedding_exact_with_and_without_hooks"]
        for precision_rows in traces.values()
        for row in precision_rows.values()
    )
    return {
        "cohort": {
            "spike_count": len(pairs),
            "matched_control_count": len(pairs),
            "match_tier_counts": dict(Counter(pair["match_tier"] for pair in pairs)),
            "source": str(T1B3_PATH.relative_to(PROJECT_ROOT)),
        },
        "comparisons": comparisons,
        "hooks_preserved_logits_and_embeddings_exactly": hooks_exact,
    }


def _validation_embedding_scan(
    model: torch.nn.Module,
    records: list[Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        ThermalNativeDataset(records, training=False, seed=20260715),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_thermal_trials,
        pin_memory=True,
    )
    sample_ids: list[str] = []
    norms: list[float] = []
    logit_abs_max: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            clips = batch["clips"].to(device, non_blocking=True)
            embedding, logits = _forward(model, clips, precision="bfloat16")
            sample_ids.extend(batch["sample_ids"])
            norms.extend(torch.linalg.vector_norm(embedding.float(), dim=1).cpu().tolist())
            logit_abs_max.extend(logits.float().abs().max(dim=1).values.cpu().tolist())
    values = np.asarray(norms, dtype=np.float64)
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    threshold = float(q3 + 3.0 * (q3 - q1))
    outliers = [
        {"sample_id": sample_id, "embedding_l2": norm}
        for sample_id, norm in zip(sample_ids, norms, strict=True)
        if norm > threshold
    ]
    return {
        "trial_count": len(records),
        "precision": "bfloat16",
        "embedding_l2": {
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "max": float(values.max()),
            "robust_upper_threshold_q3_plus_3iqr": threshold,
            "robust_outlier_count": len(outliers),
            "outliers": outliers,
        },
        "logit_abs_max": {
            "median": float(np.median(logit_abs_max)),
            "p95": float(np.quantile(logit_abs_max, 0.95)),
            "max": float(max(logit_abs_max)),
        },
    }


def _deployment_ledger(checkpoint_path: Path) -> dict[str, Any]:
    prior = json.loads(T1A_PATH.read_text(encoding="utf-8"))
    retained = [
        DeploymentAsset(**asset)
        for asset in prior["deployment_ledger"]["current_retained_inference_assets"]["assets"]
    ]
    retained.extend(
        [
            DeploymentAsset.from_file("thermal_mobilenetv3_small_tsm", checkpoint_path),
            DeploymentAsset(
                name="calibration_fusion_reserved_upper_bound",
                identity="f" * 64,
                serialized_bytes=3_000_000,
                sha256="f" * 64,
                status="reserved_not_yet_serialized",
            ),
        ]
    )
    return build_deployment_ledger(retained)


def _render_markdown(report: dict[str, Any]) -> str:
    mobile = report["training"]["best_validation"]
    comparison = report["comparison"]
    fp32 = report["stability"]["matched_pairs"]["comparisons"]["fp32"]["aggregate"]
    bf16 = report["stability"]["matched_pairs"]["comparisons"]["bfloat16"]["aggregate"]
    scan = report["stability"]["all_validation_scan"]["embedding_l2"]
    ledger = report["deployment_ledger"]
    lines = [
        "# Thermal pretrained MobileNetV3-Small + TSM matched control",
        "",
        f"- Status: **{report['status']}**",
        f"- Best epoch: `{report['training']['best_epoch']}` of `30`.",
        f"- MobileNet validation Accuracy / Macro-F1 / worst-user Accuracy: `{mobile['accuracy']:.6f}` / `{mobile['macro_f1']:.6f}` / `{mobile['worst_user_accuracy']:.6f}`.",
        f"- Serialized checkpoint: `{report['training']['checkpoint_bytes']}` bytes, SHA256 `{report['training']['checkpoint_sha256']}`.",
        f"- Provisional deduplicated package: `{ledger['total_serialized_bytes']}` bytes; strict `<95,000,000` pass `{ledger['passes_strict_limit']}`.",
        "",
        "## Matched comparison",
        "",
        "| Model | Accuracy | Macro-F1 | Worst-user Accuracy | Activation status |",
        "|---|---:|---:|---:|---|",
        f"| iFormer-T epoch16 | {comparison['iformer_epoch16']['accuracy']:.6f} | {comparison['iformer_epoch16']['macro_f1']:.6f} | {comparison['iformer_epoch16']['worst_user_accuracy']:.6f} | abnormal fine-tuned tail |",
        f"| frozen iFormer-T head-only | {comparison['frozen_iformer_t']['accuracy']:.6f} | {comparison['frozen_iformer_t']['macro_f1']:.6f} | {comparison['frozen_iformer_t']['worst_user_accuracy']:.6f} | stable, insufficient representation |",
        f"| MobileNetV3-Small + TSM | {mobile['accuracy']:.6f} | {mobile['macro_f1']:.6f} | {mobile['worst_user_accuracy']:.6f} | {report['conclusion']['activation_interpretation']} |",
        "",
        "## Stability",
        "",
        f"- FP32 matched-pair maximum feature-block / embedding RMS ratios: `{fp32['maximum_feature_block_rms_ratio']:.3f}` / `{fp32['maximum_embedding_rms_ratio']:.3f}`; any 10x event `{fp32['feature_or_embedding_10x_event']}`.",
        f"- bfloat16 matched-pair maximum feature-block / embedding RMS ratios: `{bf16['maximum_feature_block_rms_ratio']:.3f}` / `{bf16['maximum_embedding_rms_ratio']:.3f}`; any 10x event `{bf16['feature_or_embedding_10x_event']}`.",
        f"- All-validation embedding L2 median / max: `{scan['median']:.3f}` / `{scan['max']:.3f}`; robust outliers `{scan['robust_outlier_count']}` of `377`.",
        f"- Read-only hooks preserved logits and embeddings exactly: `{report['stability']['matched_pairs']['hooks_preserved_logits_and_embeddings_exactly']}`; trace state unchanged: `{report['stability']['state_unchanged']}`.",
        "",
        "## Decision",
        "",
        report["conclusion"]["summary"],
        "",
        "This development result cannot automatically promote MobileNet. Formal retention remains gated by shared train-14 OOF, IR unique-correct/oracle-pair evidence, exact deployment bytes, and latency. Training is stopped.",
        "",
        "No YOLO crop, heldout label, competition test, quarantined evidence, or frozen IR/X3D modification was used.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched Thermal MobileNet control")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Matched MobileNet recipe requires CUDA bfloat16 AMP")
    recipe = T1BRecipe()
    if recipe.epochs != 30:
        raise RuntimeError("Matched control hard stop must remain 30 epochs")
    seed_everything(recipe.seed)
    train_canonical, validation_canonical = load_development_records(
        AUDIT_PATH, SPLIT_PATH, args.data_root
    )
    train = [record for record in train_canonical if record.usable]
    validation = [record for record in validation_canonical if record.usable]
    counts = (len(train_canonical), len(train), len(validation_canonical), len(validation))
    if counts != (2039, 1922, 388, 377):
        raise ValueError(f"Frozen matched-control population changed: {counts}")
    class_hash = _class_hash(train_canonical + validation_canonical)
    device = torch.device("cuda")

    smoke_model = build_pretrained_mobilenet_expert().to(device)
    if sum(parameter.numel() for parameter in smoke_model.spatial.temporal_shift.parameters()):
        raise RuntimeError("TSM gained trainable parameters")
    smoke_loader = DataLoader(
        ThermalNativeDataset(train[:4], training=True, seed=recipe.seed),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_thermal_trials,
    )
    smoke_optimizer = build_optimizer(smoke_model, recipe)
    torch.cuda.reset_peak_memory_stats(device)
    smoke_started = time.perf_counter()
    smoke = run_epoch(
        smoke_model,
        smoke_loader,
        device=device,
        recipe=recipe,
        optimizer=smoke_optimizer,
        max_batches=1,
    )
    smoke_report = {
        "passed": True,
        "batch_shape": [4, 16, 3, 224, 224],
        "finite_loss": bool(np.isfinite(smoke.loss)),
        "loss": smoke.loss,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "wall_seconds": time.perf_counter() - smoke_started,
        "tsm_parameter_count": 0,
        "model_parameter_count": sum(parameter.numel() for parameter in smoke_model.parameters()),
        "head_parameter_count": sum(
            parameter.numel() for parameter in smoke_model.head_parameters()
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cuda_smoke.json").write_text(
        json.dumps(smoke_report, indent=2), encoding="utf-8"
    )
    del smoke_model, smoke_optimizer
    torch.cuda.empty_cache()
    if args.smoke_only:
        print(json.dumps(smoke_report, indent=2))
        return

    model = build_pretrained_mobilenet_expert()
    training = train_development(
        model,
        ThermalNativeDataset(train, training=True, seed=recipe.seed),
        ThermalNativeDataset(validation, training=False, seed=recipe.seed),
        device=device,
        output_dir=args.output_dir,
        class_map_hash=class_hash,
        recipe=recipe,
    )
    training["history"] = json.loads(
        (args.output_dir / "history.json").read_text(encoding="utf-8")
    )
    checkpoint_path = Path(training["checkpoint_path"])
    checkpoint_sha_before = _sha256(checkpoint_path)
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    traced_model = build_pretrained_mobilenet_expert()
    loaded = traced_model.load_state_dict(best["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("MobileNet best checkpoint strict load incomplete")
    traced_model.to(device).eval()
    state_before = state_dict_digest(traced_model.state_dict())
    t1b3 = json.loads(T1B3_PATH.read_text(encoding="utf-8"))
    pairs = t1b3["control_matching"]["pairs"]
    if len(pairs) != 23 or Counter(pair["match_tier"] for pair in pairs) != {
        "same_user_same_class": 23
    }:
        raise ValueError("T1-B.3 matched cohort changed")
    records_by_id = {record.sample_id: record for record in validation}
    trace_started = time.perf_counter()
    matched_trace = _matched_stability_trace(
        traced_model, pairs, records_by_id, device=device
    )
    validation_scan = _validation_embedding_scan(traced_model, validation, device=device)
    trace_seconds = time.perf_counter() - trace_started
    state_after = state_dict_digest(traced_model.state_dict())
    stability = {
        "matched_pairs": matched_trace,
        "all_validation_scan": validation_scan,
        "state_digest_before": state_before,
        "state_digest_after": state_after,
        "state_unchanged": state_before == state_after,
        "checkpoint_file_unchanged": checkpoint_sha_before == _sha256(checkpoint_path),
        "trace_seconds": trace_seconds,
    }
    if not all(
        (
            stability["matched_pairs"]["hooks_preserved_logits_and_embeddings_exactly"],
            stability["state_unchanged"],
            stability["checkpoint_file_unchanged"],
        )
    ):
        raise RuntimeError("MobileNet read-only stability trace failed")

    t1b1 = json.loads(T1B1_PATH.read_text(encoding="utf-8"))
    epoch16 = t1b1["controlled_modes"]["checkpoint_eval"]
    t1b4 = json.loads(T1B4_PATH.read_text(encoding="utf-8"))
    frozen = t1b4["training"]["best_validation"]
    mobile = training["best_validation"]
    comparison = {
        "iformer_epoch16": {
            "accuracy": epoch16["accuracy"],
            "macro_f1": epoch16["macro_f1_fixed_0_39"],
            "worst_user_accuracy": epoch16["worst_user_accuracy"],
            "activation_tail": "finetuning_induced_extreme_tail",
        },
        "frozen_iformer_t": {
            "accuracy": frozen["accuracy"],
            "macro_f1": frozen["macro_f1"],
            "worst_user_accuracy": frozen["worst_user_accuracy"],
            "activation_tail": "absent_under_preregistered_gate",
        },
        "mobilenetv3_small_tsm": {
            "accuracy": mobile["accuracy"],
            "macro_f1": mobile["macro_f1"],
            "worst_user_accuracy": mobile["worst_user_accuracy"],
        },
        "mobilenet_minus_iformer_epoch16": {
            "accuracy": mobile["accuracy"] - epoch16["accuracy"],
            "macro_f1": mobile["macro_f1"] - epoch16["macro_f1_fixed_0_39"],
            "worst_user_accuracy": mobile["worst_user_accuracy"]
            - epoch16["worst_user_accuracy"],
        },
        "mobilenet_minus_frozen_iformer": {
            "accuracy": mobile["accuracy"] - frozen["accuracy"],
            "macro_f1": mobile["macro_f1"] - frozen["macro_f1"],
            "worst_user_accuracy": mobile["worst_user_accuracy"]
            - frozen["worst_user_accuracy"],
        },
    }
    any_10x = any(
        matched_trace["comparisons"][precision]["aggregate"][
            "feature_or_embedding_10x_event"
        ]
        for precision in PRECISIONS
    )
    if any_10x:
        activation_interpretation = "matched 10x activation event observed"
        architecture_interpretation = (
            "The MobileNet control also develops an extreme activation event, so the "
            "instability cannot be attributed specifically to iFormer fine-tuning."
        )
    else:
        activation_interpretation = "no matched 10x activation event"
        architecture_interpretation = (
            "MobileNet does not reproduce the iFormer epoch16 extreme activation tail on "
            "the fixed cohort, supporting an iFormer-specific fine-tuning response rather "
            "than an unavoidable property of the Thermal inputs or shared recipe."
        )
    report = {
        "schema_version": "thermal-mobilenet-matched-control-v1",
        "stage": "thermal_mobilenet_matched_control",
        "status": "completed_development_control_training_stopped_waiting_human_decision",
        "branch": "experiment/thermal-iformer-t-t1b",
        "scientific_baseline_commit": "c42bb43091c79903e5fde5655c2846c87305895a",
        "preregistered_parent_commit": "TO_BE_FILLED_FROM_GIT_AT_RUNTIME",
        "authorization": "explicit_user_instruction_2026_08_20_18_11",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
        },
        "initialization": {
            "model": "torchvision MobileNetV3-Small",
            "source_revision": "torchvision-v0.22.0",
            "pretrained_weights": "IMAGENET1K_V1",
            "pretrained_weight_sha256": WEIGHT_SHA256,
            "strict_pretrained_load_before_classifier_replacement": True,
            "random_initialization_fallback": False,
        },
        "matched_contract": {
            "only_model_family_changed_from_t1b": True,
            "classifier": "batchnorm1d_576_then_linear40",
            "classifier_structure_matches_iformer_t": True,
            "route": "full_frame",
            "segments": 16,
            "timeline": "thermal_native_normalized_time",
            "split": "train12_fit_user6_user7_validation_only",
            "same_recipe_seed_metrics_checkpoint_rule": True,
            "yolo_crop_used": False,
        },
        "data_access": {
            "counts": {
                "train_canonical": counts[0],
                "train_thermal_usable": counts[1],
                "validation_canonical": counts[2],
                "validation_thermal_usable": counts[3],
            },
            "heldout_labels_accessed": False,
            "competition_test_accessed": False,
            "quarantined_evidence_accessed": False,
            "ir_x3d_modified": False,
        },
        "inputs": {
            "audit_sha256": _sha256(AUDIT_PATH),
            "split_sha256": _sha256(SPLIT_PATH),
            "config_sha256": _sha256(CONFIG_PATH),
            "t1b1_sha256": _sha256(T1B1_PATH),
            "t1b3_sha256": _sha256(T1B3_PATH),
            "t1b4_sha256": _sha256(T1B4_PATH),
            "class_map_hash": class_hash,
        },
        "cuda_smoke": smoke_report,
        "training": training,
        "stability": stability,
        "comparison": comparison,
        "deployment_ledger": _deployment_ledger(checkpoint_path),
        "conclusion": {
            "activation_interpretation": activation_interpretation,
            "architecture_interpretation": architecture_interpretation,
            "automatic_promotion_authorized": False,
            "shared_train14_oof_required_for_retention": True,
            "ir_unique_correct_oracle_pair_required": True,
            "training_remains_stopped": True,
            "summary": (
                f"{architecture_interpretation} MobileNet's development performance is "
                "reported as a matched lightweight control, not as formal retention evidence."
            ),
        },
        "limitations": [
            "Single train12/user6-user7 development run; no shared train-14 OOF evidence.",
            "IR unique-correct and oracle-pair cannot be computed from development-only logits.",
            "Pretrained-weight permission remains the deployer's responsibility per the T1-A caveat.",
        ],
    }
    import subprocess

    report["preregistered_parent_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON),
                "best_epoch": training["best_epoch"],
                "accuracy": mobile["accuracy"],
                "macro_f1": mobile["macro_f1"],
                "any_10x_activation": any_10x,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
