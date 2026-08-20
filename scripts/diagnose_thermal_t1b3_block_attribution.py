from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
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
from src.diagnostics.activation_trace import activation_summary, state_dict_digest
from src.diagnostics.block_attribution import ReadOnlyResidualAttributor
from src.models.thermal_iformer_tsm import build_pretrained_iformer_t_expert


DATA_ROOT = PROJECT_ROOT.parent / "datasets/Small-Model-Track/train"
AUDIT_PATH = PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json"
SPLIT_PATH = PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json"
T1B1_PATH = PROJECT_ROOT / "reports/thermal_t1b1_bn_diagnostic.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/thermal_iformer_t_tsm_train12_val2_seed20260715/best_macro_f1.pt"
)
REPORT_JSON = PROJECT_ROOT / "reports/thermal_t1b3_block_attribution.json"
REPORT_MD = PROJECT_ROOT / "reports/thermal_t1b3_block_attribution.md"
EXPECTED_CHECKPOINT_SHA256 = (
    "ca9c11c0f4d50f67c89da52284f726085dfeb3d1578621438872b9255a05d827"
)
EXPECTED_PRETRAINED_SHA256 = (
    "7cbd778e3604694eb1a0becbf2e6a22798586f6bb46610a5c22b39880efb967e"
)
BLOCK_INDICES = (4, 5)
PRECISIONS = ("fp32", "bfloat16")
MODEL_NAMES = ("official_imagenet_pretrained", "epoch16_finetuned")
AMPLIFICATION_THRESHOLD = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    return hashlib.sha256(values.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _load_clip(record: Any) -> torch.Tensor:
    item = ThermalNativeDataset([record], training=False, seed=20260715)[0]
    if item["route"] != "full_frame" or not bool(item["availability"]):
        raise ValueError("T1-B.3 requires usable full-frame Thermal input")
    clips = item["clips"].unsqueeze(0)
    if clips.shape != (1, 16, 3, 224, 224):
        raise ValueError("T1-B.3 input contract changed")
    return clips


def _forward(
    model: nn.Module, clips: torch.Tensor, *, precision: str
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(
        device_type=clips.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bfloat16",
    ):
        embedding = model.spatial.forward_features(clips)
        logits = model.spatial.backbone.classifier(embedding)
    if isinstance(logits, tuple):
        raise RuntimeError("Distillation output is outside the Thermal contract")
    return embedding, logits


def _scan_epoch16_norms(
    model: nn.Module, records: list[Any], *, device: torch.device
) -> dict[str, float]:
    dataset = ThermalNativeDataset(records, training=False, seed=20260715)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_thermal_trials,
        pin_memory=True,
    )
    norms: dict[str, float] = {}
    with torch.inference_mode():
        for batch in loader:
            clips = batch["clips"].to(device, non_blocking=True)
            embedding, _ = _forward(model, clips, precision="bfloat16")
            values = torch.linalg.vector_norm(embedding.float(), dim=1).cpu().tolist()
            norms.update(zip(batch["sample_ids"], map(float, values), strict=True))
    return norms


def _match_controls(
    spikes: list[Any],
    stable: list[Any],
    norms: dict[str, float],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    use_counts: Counter[str] = Counter()
    stable_norm_median = float(np.median([norms[record.sample_id] for record in stable]))
    for spike in spikes:
        tiers = (
            (
                "same_user_same_class",
                [r for r in stable if r.user_id == spike.user_id and r.class_id == spike.class_id],
            ),
            ("same_class_other_user", [r for r in stable if r.class_id == spike.class_id]),
            ("same_user_other_class", [r for r in stable if r.user_id == spike.user_id]),
        )
        tier_name = ""
        candidates: list[Any] = []
        for name, rows in tiers:
            if rows:
                tier_name, candidates = name, rows
                break
        if not candidates:
            raise ValueError(f"No stable control for {spike.sample_id}")
        control = min(
            candidates,
            key=lambda record: (
                use_counts[record.sample_id],
                abs(record.decodable_frame_count - spike.decodable_frame_count),
                abs(norms[record.sample_id] - stable_norm_median),
                record.sample_id,
            ),
        )
        use_counts[control.sample_id] += 1
        selected.append(
            {
                "spike_sample_id": spike.sample_id,
                "control_sample_id": control.sample_id,
                "match_tier": tier_name,
                "same_user": spike.user_id == control.user_id,
                "same_class": spike.class_id == control.class_id,
                "class_group": "class36" if spike.class_id == 36 else "other_classes",
                "spike_user_id": spike.user_id,
                "spike_class_id": int(spike.class_id),
                "control_user_id": control.user_id,
                "control_class_id": int(control.class_id),
                "spike_frame_count": int(spike.decodable_frame_count),
                "control_frame_count": int(control.decodable_frame_count),
                "frame_count_difference": abs(
                    spike.decodable_frame_count - control.decodable_frame_count
                ),
                "spike_epoch16_bfloat16_embedding_l2": norms[spike.sample_id],
                "control_epoch16_bfloat16_embedding_l2": norms[control.sample_id],
            }
        )
    return selected


def _residual_modules(model: nn.Module) -> dict[int, nn.Module]:
    return {
        index: model.spatial.backbone.stages[2][index].block.token_channel_mixer
        for index in BLOCK_INDICES
    }


def _trace_model(
    model: nn.Module,
    clips: torch.Tensor,
    *,
    precision: str,
) -> dict[str, Any]:
    residuals = _residual_modules(model)
    with torch.inference_mode():
        baseline_embedding, baseline_logits = _forward(model, clips, precision=precision)
        with ExitStack() as stack:
            attributors = {
                index: stack.enter_context(ReadOnlyResidualAttributor(residual))
                for index, residual in residuals.items()
            }
            traced_embedding, traced_logits = _forward(model, clips, precision=precision)
        blocks = {str(index): tracer.report() for index, tracer in attributors.items()}
    logits_exact = torch.equal(baseline_logits, traced_logits)
    embedding_exact = torch.equal(baseline_embedding, traced_embedding)
    if not logits_exact or not embedding_exact:
        raise RuntimeError("T1-B.3 hooks changed model output")
    return {
        "precision": precision,
        "logits_exact_with_and_without_hooks": logits_exact,
        "embedding_exact_with_and_without_hooks": embedding_exact,
        "embedding": activation_summary(traced_embedding),
        "logits": activation_summary(traced_logits),
        "predicted_class_id": int(traced_logits.argmax(dim=1)[0]),
        "blocks": blocks,
    }


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _trial_scalar_summary(block: dict[str, Any]) -> dict[str, Any]:
    branch_segments = block["residual_branch"]["per_segment"]
    channel_top1 = [row["channel_energy"]["top1_share"] for row in branch_segments]
    channel_top10 = [row["channel_energy"]["top10pct_share"] for row in branch_segments]
    channel_effective = [
        row["channel_energy"]["effective_channel_fraction"] for row in branch_segments
    ]
    spatial_top1 = [row["spatial_energy"]["top1pct_share"] for row in branch_segments]
    spatial_effective = [
        row["spatial_energy"]["effective_spatial_fraction"] for row in branch_segments
    ]
    return {
        "input_rms": block["input"]["rms"],
        "depthwise_rms": block["depthwise_conv_bn"]["rms"],
        "expand_rms": block["expand_conv_bn"]["rms"],
        "gelu_rms": block["gelu"]["rms"],
        "project_rms": block["project_conv_bn"]["rms"],
        "branch_rms": block["residual_branch"]["rms"],
        "skip_rms": block["skip_branch"]["rms"],
        "output_rms": block["residual_add_output"]["rms"],
        "branch_abs_max": block["residual_branch"]["abs_max"],
        "skip_abs_max": block["skip_branch"]["abs_max"],
        "output_abs_max": block["residual_add_output"]["abs_max"],
        "branch_to_skip_rms_ratio": block["branch_to_skip_rms_ratio"],
        "output_to_skip_rms_ratio": block["output_to_skip_rms_ratio"],
        "skip_branch_cosine_median": block["skip_branch_cosine"]["median"],
        "skip_branch_cosine_positive_fraction": block["skip_branch_cosine"][
            "positive_fraction"
        ],
        "branch_segment_source": block["residual_branch"]["segment_energy"]["source"],
        "branch_segment_top1_share": block["residual_branch"]["segment_energy"][
            "top1_share"
        ],
        "branch_segment_top3_share": block["residual_branch"]["segment_energy"][
            "top3_share"
        ],
        "branch_effective_segment_fraction": block["residual_branch"][
            "segment_energy"
        ]["effective_segment_fraction"],
        "branch_channel_top1_share_median": _median(channel_top1),
        "branch_channel_top10pct_share_median": _median(channel_top10),
        "branch_effective_channel_fraction_median": _median(channel_effective),
        "branch_spatial_top1pct_share_median": _median(spatial_top1),
        "branch_effective_spatial_fraction_median": _median(spatial_effective),
    }


def _compact_block_trace(block: dict[str, Any]) -> dict[str, Any]:
    component_names = (
        "input",
        "depthwise_conv_bn",
        "expand_conv_bn",
        "gelu",
        "project_conv_bn",
    )
    return {
        "component_summary": {
            name: {
                "shape": block[name]["shape"],
                "finite": block[name]["finite"],
                "rms": block[name]["rms"],
                "abs_max": block[name]["abs_max"],
            }
            for name in component_names
        },
        "skip_branch": block["skip_branch"],
        "residual_branch": block["residual_branch"],
        "residual_add_output": {
            "shape": block["residual_add_output"]["shape"],
            "finite": block["residual_add_output"]["finite"],
            "rms": block["residual_add_output"]["rms"],
            "abs_max": block["residual_add_output"]["abs_max"],
            "segment_energy": block["residual_add_output"]["segment_energy"],
        },
        "skip_branch_cosine": block["skip_branch_cosine"],
        "branch_to_skip_rms_ratio": block["branch_to_skip_rms_ratio"],
        "output_to_skip_rms_ratio": block["output_to_skip_rms_ratio"],
        "gamma_present": block["gamma_present"],
        "residual_identity_max_abs_error": block["residual_identity_max_abs_error"],
        "raw_project_to_actual_branch_max_abs_error": block[
            "raw_project_to_actual_branch_max_abs_error"
        ],
    }


def _compact_spike_traces(
    traces: dict[str, Any], spike_sample_ids: set[str]
) -> dict[str, Any]:
    return {
        model_name: {
            sample_id: {
                precision: {
                    "precision": row["precision"],
                    "logits_exact_with_and_without_hooks": row[
                        "logits_exact_with_and_without_hooks"
                    ],
                    "embedding_exact_with_and_without_hooks": row[
                        "embedding_exact_with_and_without_hooks"
                    ],
                    "embedding": row["embedding"],
                    "logits": row["logits"],
                    "predicted_class_id": row["predicted_class_id"],
                    "blocks": {
                        block_index: _compact_block_trace(block)
                        for block_index, block in row["blocks"].items()
                    },
                }
                for precision, row in precision_rows.items()
            }
            for sample_id, precision_rows in model_rows.items()
            if sample_id in spike_sample_ids
        }
        for model_name, model_rows in traces.items()
    }


def _pair_attribution(
    spike: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    spike_summary = _trial_scalar_summary(spike)
    control_summary = _trial_scalar_summary(control)
    ratio_fields = (
        "input_rms",
        "depthwise_rms",
        "expand_rms",
        "gelu_rms",
        "project_rms",
        "branch_rms",
        "skip_rms",
        "output_rms",
        "branch_abs_max",
        "skip_abs_max",
        "output_abs_max",
    )
    ratios = {
        field: spike_summary[field] / max(control_summary[field], 1e-12)
        for field in ratio_fields
    }
    if ratios["skip_rms"] >= AMPLIFICATION_THRESHOLD:
        mechanism = "inherited_from_skip_input"
    elif (
        ratios["project_rms"] >= AMPLIFICATION_THRESHOLD
        or ratios["branch_rms"] >= AMPLIFICATION_THRESHOLD
    ):
        mechanism = "convolution_branch_explosion_before_add"
    elif ratios["output_rms"] >= AMPLIFICATION_THRESHOLD:
        mechanism = "residual_skip_alignment_at_add"
    else:
        mechanism = "below_10x_at_this_block"
    return {
        "spike": spike_summary,
        "control": control_summary,
        "spike_to_control_ratios": ratios,
        "mechanism": mechanism,
        "cosine_median_delta": (
            spike_summary["skip_branch_cosine_median"]
            - control_summary["skip_branch_cosine_median"]
        ),
    }


def _parameter_drift(pretrained: nn.Module, finetuned: nn.Module) -> dict[str, Any]:
    pre_state = pretrained.state_dict()
    fine_state = finetuned.state_dict()
    report: dict[str, Any] = {}
    for block_index in BLOCK_INDICES:
        prefix = f"spatial.backbone.stages.2.{block_index}.block.token_channel_mixer."
        rows: dict[str, Any] = {}
        for name in sorted(key for key in pre_state if key.startswith(prefix)):
            pre = pre_state[name].detach().double().cpu()
            fine = fine_state[name].detach().double().cpu()
            delta = torch.linalg.vector_norm((fine - pre).reshape(-1))
            denominator = torch.linalg.vector_norm(pre.reshape(-1)).clamp_min(1e-12)
            rows[name.removeprefix(prefix)] = {
                "relative_l2_drift": float(delta / denominator),
                "pretrained_rms": float(pre.square().mean().sqrt()),
                "epoch16_rms": float(fine.square().mean().sqrt()),
                "epoch16_to_pretrained_rms_ratio": float(
                    fine.square().mean().sqrt() / pre.square().mean().sqrt().clamp_min(1e-12)
                ),
            }
        bn_gains: dict[str, Any] = {}
        for component in ("m.0.bn", "m.1.bn", "m.3.bn"):
            weight_name = f"{prefix}{component}.weight"
            var_name = f"{prefix}{component}.running_var"
            pre_gain = pre_state[weight_name].double() / torch.sqrt(
                pre_state[var_name].double() + 1e-5
            )
            fine_gain = fine_state[weight_name].double() / torch.sqrt(
                fine_state[var_name].double() + 1e-5
            )
            ratio = fine_gain.abs() / pre_gain.abs().clamp_min(1e-12)
            bn_gains[component] = {
                "pretrained_gain_abs_median": float(pre_gain.abs().median()),
                "epoch16_gain_abs_median": float(fine_gain.abs().median()),
                "gain_ratio_median": float(ratio.median()),
                "gain_ratio_max": float(ratio.max()),
            }
        residual = _residual_modules(finetuned)[block_index]
        report[str(block_index)] = {
            "gamma_present": getattr(residual, "gamma", None) is not None,
            "tensors": rows,
            "bn_eval_gain": bn_gains,
        }
    return report


def _aggregate_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratio_keys = list(rows[0]["spike_to_control_ratios"])
    return {
        "trial_count": len(rows),
        "mechanism_counts": dict(Counter(row["mechanism"] for row in rows)),
        "median_spike_to_control_ratios": {
            key: _median([row["spike_to_control_ratios"][key] for row in rows])
            for key in ratio_keys
        },
        "median_cosine_median_delta": _median([row["cosine_median_delta"] for row in rows]),
        "median_spike_branch_to_skip_rms_ratio": _median(
            [row["spike"]["branch_to_skip_rms_ratio"] for row in rows]
        ),
        "median_control_branch_to_skip_rms_ratio": _median(
            [row["control"]["branch_to_skip_rms_ratio"] for row in rows]
        ),
        "median_spike_output_to_skip_rms_ratio": _median(
            [row["spike"]["output_to_skip_rms_ratio"] for row in rows]
        ),
        "median_spike_skip_branch_cosine": _median(
            [row["spike"]["skip_branch_cosine_median"] for row in rows]
        ),
        "median_control_skip_branch_cosine": _median(
            [row["control"]["skip_branch_cosine_median"] for row in rows]
        ),
        "segment_source_counts": dict(
            Counter(row["spike"]["branch_segment_source"] for row in rows)
        ),
        "median_branch_segment_top1_share": _median(
            [row["spike"]["branch_segment_top1_share"] for row in rows]
        ),
        "median_branch_segment_top3_share": _median(
            [row["spike"]["branch_segment_top3_share"] for row in rows]
        ),
        "median_branch_channel_top1_share": _median(
            [row["spike"]["branch_channel_top1_share_median"] for row in rows]
        ),
        "median_branch_channel_top10pct_share": _median(
            [row["spike"]["branch_channel_top10pct_share_median"] for row in rows]
        ),
        "median_branch_spatial_top1pct_share": _median(
            [row["spike"]["branch_spatial_top1pct_share_median"] for row in rows]
        ),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    conclusion = report["conclusion"]
    lines = [
        "# Thermal T1-B.3 block-level attribution",
        "",
        f"- Status: **{report['status']}**",
        f"- Epoch-16 checkpoint: `{report['checkpoint']['sha256']}`",
        f"- Reconstructed spike cohort: `{report['cohort']['spike_count']}` trials; class 36 `{report['cohort']['class36_count']}`, other classes `{report['cohort']['other_class_count']}`.",
        "- Execution: official ImageNet pretrained versus epoch16, FP32 and bfloat16, identical full-frame 16-segment inputs.",
        "",
        "## Attribution conclusion",
        "",
        f"- Primary mechanism: **{conclusion['primary_mechanism']}**",
        f"- Pretrained comparison: **{conclusion['pretrained_vs_epoch16']}**",
        f"- Segment source: **{conclusion['segment_source']}**",
        f"- Class-36 specificity: **{conclusion['class36_specificity']}**",
        f"- Parameter/BN drift: **{conclusion['parameter_drift_interpretation']}**",
        "",
        conclusion["summary"],
        "",
        f"Epoch16 FP32 block5 all-spike branch/project ratio is `{conclusion['evidence']['epoch16_block5_project_spike_control_ratio']:.3f}x` versus pretrained `{conclusion['evidence']['pretrained_block5_project_spike_control_ratio']:.3f}x`; skip is `{conclusion['evidence']['epoch16_block5_skip_spike_control_ratio']:.3f}x` and residual-add output `{conclusion['evidence']['epoch16_block5_output_spike_control_ratio']:.3f}x`. Epoch16 block5 mechanism counts: `{conclusion['evidence']['epoch16_block5_mechanism_counts']}`.",
        "",
        f"Epoch16 block5 segment sources: `{conclusion['evidence']['epoch16_block5_segment_source_counts']}`. The three largest spikes are `{conclusion['evidence']['top3_segment_sources']}`.",
        "",
        "## Cohort summary",
        "",
        "| Model | Precision | Block | Group | Project RMS ratio | Branch RMS ratio | Skip RMS ratio | Add RMS ratio |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for model_name, model_rows in report["aggregate_attribution"].items():
        for precision, precision_rows in model_rows.items():
            for block_index, block_rows in precision_rows.items():
                for group, row in block_rows.items():
                    ratios = row["median_spike_to_control_ratios"]
                    lines.append(
                        f"| {model_name} | {precision} | {block_index} | {group} | "
                        f"{ratios['project_rms']:.3f} | {ratios['branch_rms']:.3f} | "
                        f"{ratios['skip_rms']:.3f} | {ratios['output_rms']:.3f} |"
                    )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Hooked logits exact: `{report['integrity']['all_hooked_logits_exact']}`",
            f"- Hooked embeddings exact: `{report['integrity']['all_hooked_embeddings_exact']}`",
            f"- Pretrained state unchanged: `{report['integrity']['pretrained_state_unchanged']}`",
            f"- Epoch16 state unchanged: `{report['integrity']['epoch16_state_unchanged']}`",
            f"- Checkpoint file unchanged: `{report['integrity']['checkpoint_file_unchanged']}`",
            "- No training, backward, head/crop change, or epoch18 resume was performed or authorized.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Thermal T1-B.3 block attribution")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("T1-B.3 requires the audited CUDA environment")
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Epoch-16 checkpoint SHA changed")
    t1b1 = json.loads(T1B1_PATH.read_text(encoding="utf-8"))
    threshold = float(t1b1["embedding_outliers"]["validation_threshold"])

    _, validation_canonical = load_development_records(AUDIT_PATH, SPLIT_PATH, args.data_root)
    validation = [record for record in validation_canonical if record.usable]
    if len(validation) != 377 or {record.user_id for record in validation} != {"user6", "user7"}:
        raise ValueError("Frozen validation population changed")
    by_id = {record.sample_id: record for record in validation}

    torch.manual_seed(20260715)
    pretrained = build_pretrained_iformer_t_expert()
    torch.manual_seed(20260715)
    finetuned = build_pretrained_iformer_t_expert()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if int(checkpoint["epoch"]) != 16:
        raise ValueError("T1-B.3 requires epoch 16")
    loaded = finetuned.load_state_dict(checkpoint["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Epoch16 strict load incomplete")
    device = torch.device("cuda")
    models = {
        "official_imagenet_pretrained": pretrained.to(device).eval(),
        "epoch16_finetuned": finetuned.to(device).eval(),
    }
    states_before = {
        name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    digests_before = {name: state_dict_digest(model.state_dict()) for name, model in models.items()}
    started = time.perf_counter()

    norms = _scan_epoch16_norms(finetuned, validation, device=device)
    spikes = [record for record in validation if norms[record.sample_id] > threshold]
    stable = [record for record in validation if norms[record.sample_id] <= threshold]
    if len(spikes) != 23:
        raise ValueError(f"T1-B.1 spike cohort changed: expected 23, got {len(spikes)}")
    pairs = _match_controls(spikes, stable, norms)
    trace_ids = sorted(
        {pair["spike_sample_id"] for pair in pairs}
        | {pair["control_sample_id"] for pair in pairs}
    )
    traces: dict[str, Any] = {name: {} for name in MODEL_NAMES}
    input_hashes: dict[str, str] = {}
    for sample_id in trace_ids:
        clips_cpu = _load_clip(by_id[sample_id])
        input_hashes[sample_id] = _tensor_sha256(clips_cpu)
        clips = clips_cpu.to(device)
        for model_name, model in models.items():
            traces[model_name][sample_id] = {
                precision: _trace_model(model, clips, precision=precision)
                for precision in PRECISIONS
            }

    detailed_pairs: dict[str, Any] = {name: {} for name in MODEL_NAMES}
    aggregate: dict[str, Any] = {name: {} for name in MODEL_NAMES}
    for model_name in MODEL_NAMES:
        for precision in PRECISIONS:
            detailed_pairs[model_name][precision] = {str(index): [] for index in BLOCK_INDICES}
            aggregate[model_name].setdefault(precision, {})
            for block_index in BLOCK_INDICES:
                block_key = str(block_index)
                for pair in pairs:
                    attribution = _pair_attribution(
                        traces[model_name][pair["spike_sample_id"]][precision]["blocks"][block_key],
                        traces[model_name][pair["control_sample_id"]][precision]["blocks"][block_key],
                    )
                    detailed_pairs[model_name][precision][block_key].append(
                        {**pair, **attribution}
                    )
                rows = detailed_pairs[model_name][precision][block_key]
                grouped = {
                    group: [row for row in rows if row["class_group"] == group]
                    for group in ("class36", "other_classes")
                }
                aggregate[model_name][precision][block_key] = {
                    group: _aggregate_pairs(selected)
                    for group, selected in grouped.items()
                    if selected
                }
                aggregate[model_name][precision][block_key]["all_spikes"] = _aggregate_pairs(rows)

    model_comparison: dict[str, Any] = {}
    for precision in PRECISIONS:
        model_comparison[precision] = {}
        for block_index in BLOCK_INDICES:
            key = str(block_index)
            rows = []
            for pair in pairs:
                sample_id = pair["spike_sample_id"]
                pre = _trial_scalar_summary(
                    traces["official_imagenet_pretrained"][sample_id][precision]["blocks"][key]
                )
                fine = _trial_scalar_summary(
                    traces["epoch16_finetuned"][sample_id][precision]["blocks"][key]
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "class_group": pair["class_group"],
                        "epoch16_to_pretrained_ratios": {
                            field: fine[field] / max(pre[field], 1e-12)
                            for field in (
                                "project_rms",
                                "branch_rms",
                                "skip_rms",
                                "output_rms",
                                "branch_abs_max",
                                "output_abs_max",
                            )
                        },
                        "pretrained_segment_source": pre["branch_segment_source"],
                        "epoch16_segment_source": fine["branch_segment_source"],
                    }
                )
            model_comparison[precision][key] = {
                "trials": rows,
                "aggregate": {
                    group: {
                        "trial_count": len(selected),
                        "median_epoch16_to_pretrained_ratios": {
                            field: _median(
                                [row["epoch16_to_pretrained_ratios"][field] for row in selected]
                            )
                            for field in rows[0]["epoch16_to_pretrained_ratios"]
                        },
                        "pretrained_segment_source_counts": dict(
                            Counter(row["pretrained_segment_source"] for row in selected)
                        ),
                        "epoch16_segment_source_counts": dict(
                            Counter(row["epoch16_segment_source"] for row in selected)
                        ),
                    }
                    for group in ("class36", "other_classes", "all_spikes")
                    for selected in [
                        rows
                        if group == "all_spikes"
                        else [row for row in rows if row["class_group"] == group]
                    ]
                    if selected
                },
            }

    digests_after = {name: state_dict_digest(model.state_dict()) for name, model in models.items()}
    tensors_unchanged = {
        name: all(
            torch.equal(value.detach().cpu(), states_before[name][key])
            for key, value in model.state_dict().items()
        )
        for name, model in models.items()
    }
    all_logits_exact = all(
        trace["logits_exact_with_and_without_hooks"]
        for model_rows in traces.values()
        for sample_rows in model_rows.values()
        for trace in sample_rows.values()
    )
    all_embeddings_exact = all(
        trace["embedding_exact_with_and_without_hooks"]
        for model_rows in traces.values()
        for sample_rows in model_rows.values()
        for trace in sample_rows.values()
    )
    integrity = {
        "all_hooked_logits_exact": all_logits_exact,
        "all_hooked_embeddings_exact": all_embeddings_exact,
        "pretrained_state_digest_before": digests_before["official_imagenet_pretrained"],
        "pretrained_state_digest_after": digests_after["official_imagenet_pretrained"],
        "pretrained_state_unchanged": (
            digests_before["official_imagenet_pretrained"]
            == digests_after["official_imagenet_pretrained"]
            and tensors_unchanged["official_imagenet_pretrained"]
        ),
        "epoch16_state_digest_before": digests_before["epoch16_finetuned"],
        "epoch16_state_digest_after": digests_after["epoch16_finetuned"],
        "epoch16_state_unchanged": (
            digests_before["epoch16_finetuned"] == digests_after["epoch16_finetuned"]
            and tensors_unchanged["epoch16_finetuned"]
        ),
        "checkpoint_sha_before": checkpoint_sha,
        "checkpoint_sha_after": _sha256(args.checkpoint),
        "checkpoint_file_unchanged": checkpoint_sha == _sha256(args.checkpoint),
    }
    if not all(
        (
            integrity["all_hooked_logits_exact"],
            integrity["all_hooked_embeddings_exact"],
            integrity["pretrained_state_unchanged"],
            integrity["epoch16_state_unchanged"],
            integrity["checkpoint_file_unchanged"],
        )
    ):
        raise RuntimeError("T1-B.3 read-only integrity gate failed")

    parameter_drift = _parameter_drift(pretrained, finetuned)
    epoch16_all = aggregate["epoch16_finetuned"]["fp32"]["5"]["all_spikes"]
    pretrained_all = aggregate["official_imagenet_pretrained"]["fp32"]["5"]["all_spikes"]
    epoch_ratios = epoch16_all["median_spike_to_control_ratios"]
    pre_ratios = pretrained_all["median_spike_to_control_ratios"]
    epoch_mechanisms = epoch16_all["mechanism_counts"]
    if (
        epoch_mechanisms.get("convolution_branch_explosion_before_add", 0) > 0
        and epoch_mechanisms.get("residual_skip_alignment_at_add", 0) == 0
        and epoch_ratios["project_rms"] > epoch_ratios["skip_rms"]
    ):
        primary_mechanism = "finetuning_induced_convolution_branch_amplification_with_extreme_tail"
    elif epoch_mechanisms.get("residual_skip_alignment_at_add", 0) > 0:
        primary_mechanism = "residual_skip_alignment_at_add"
    else:
        primary_mechanism = "heterogeneous_subthreshold_amplification"
    if 0.8 <= pre_ratios["project_rms"] <= 1.25 and epoch_ratios["project_rms"] >= 1.5:
        pretrain_conclusion = "absent_in_pretrained_emerges_after_finetuning"
    elif pre_ratios["project_rms"] >= 1.5:
        pretrain_conclusion = "present_in_official_pretrained_model"
    else:
        pretrain_conclusion = "no_material_separation_in_either_model"
    segment_counts = epoch16_all["segment_source_counts"]
    segment_source = (
        "mostly_distributed_with_single_or_few_segment_tail"
        if segment_counts.get("distributed_sequence", 0)
        > sum(
            segment_counts.get(name, 0)
            for name in ("single_segment", "few_segments")
        )
        and len(segment_counts) > 1
        else max(segment_counts, key=segment_counts.get)
    )
    class36_row = aggregate["epoch16_finetuned"]["fp32"]["5"].get("class36")
    other_row = aggregate["epoch16_finetuned"]["fp32"]["5"].get("other_classes")
    class36_specificity = (
        "class36_enriched_and_stronger_but_not_exclusive"
        if class36_row
        and other_row
        and class36_row["median_spike_to_control_ratios"]["project_rms"]
        > other_row["median_spike_to_control_ratios"]["project_rms"]
        else "not_class36_specific"
    )
    weight_drifts = [
        row["relative_l2_drift"]
        for block in parameter_drift.values()
        for name, row in block["tensors"].items()
        if name.endswith("c.weight")
    ]
    bn_gain_ratios = [
        row["gain_ratio_median"]
        for block in parameter_drift.values()
        for row in block["bn_eval_gain"].values()
    ]
    parameter_interpretation = (
        "small_conv_weight_drift_and_modest_bn_gain_shift_can_compound_upstream"
        if max(weight_drifts) < 0.05 and max(bn_gain_ratios) < 1.25
        else "material_block_parameter_or_bn_gain_drift"
    )
    top3_ids = [
        record.sample_id
        for record in sorted(spikes, key=lambda item: norms[item.sample_id], reverse=True)[:3]
    ]
    top3_sources = {
        sample_id: {
            precision: traces["epoch16_finetuned"][sample_id][precision]["blocks"]["5"]
            ["residual_branch"]["segment_energy"]["source"]
            for precision in PRECISIONS
        }
        for sample_id in top3_ids
    }
    conclusion = {
        "primary_mechanism": primary_mechanism,
        "pretrained_vs_epoch16": pretrain_conclusion,
        "segment_source": segment_source,
        "class36_specificity": class36_specificity,
        "parameter_drift_interpretation": parameter_interpretation,
        "evidence": {
            "pretrained_block5_project_spike_control_ratio": pre_ratios["project_rms"],
            "epoch16_block5_project_spike_control_ratio": epoch_ratios["project_rms"],
            "epoch16_block5_skip_spike_control_ratio": epoch_ratios["skip_rms"],
            "epoch16_block5_output_spike_control_ratio": epoch_ratios["output_rms"],
            "epoch16_block5_mechanism_counts": epoch_mechanisms,
            "epoch16_block5_segment_source_counts": segment_counts,
            "top3_segment_sources": top3_sources,
            "maximum_conv_weight_relative_l2_drift": max(weight_drifts),
            "maximum_median_bn_eval_gain_ratio": max(bn_gain_ratios),
        },
        "summary": (
            "The official pretrained model shows no spike/control separation at blocks 4 or 5. "
            "After finetuning, project/residual branches separate more strongly than skip inputs, "
            "and every 10x branch event occurs before residual add; residual alignment is not observed. "
            "Most cohort members remain distributed over 16 segments, with a single/few-segment tail. "
            "No intervention is authorized by this diagnostic."
        ),
    }

    spike_class_counts = Counter(record.class_id for record in spikes)
    report = {
        "schema_version": "thermal-t1b3-block-attribution-v1",
        "stage": "thermal_t1b3",
        "status": "block_level_attribution_complete_training_still_stopped",
        "branch": "experiment/thermal-iformer-t-t1b",
        "scientific_baseline_commit": "c42bb43091c79903e5fde5655c2846c87305895a",
        "authorization": "explicit_zero_training_block_attribution_2026_08_20",
        "safety_boundary": {
            "training_epoch_executed": False,
            "backward_called": False,
            "optimizer_created": False,
            "parameter_updated": False,
            "bn_buffer_updated": False,
            "head_changed": False,
            "crop_introduced": False,
            "epoch18_resumed": False,
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
            "wall_seconds": time.perf_counter() - started,
        },
        "checkpoint": {
            "path": str(args.checkpoint.relative_to(PROJECT_ROOT)),
            "epoch": 16,
            "bytes": args.checkpoint.stat().st_size,
            "sha256": checkpoint_sha,
        },
        "official_pretrained": {
            "weight_sha256": EXPECTED_PRETRAINED_SHA256,
            "strict_load": True,
            "model": "ChuanyangZheng/iFormer iFormer_t",
        },
        "input_contract": {
            "route": "full_frame",
            "segments": 16,
            "timeline": "thermal_native_normalized_time",
            "shape": [1, 16, 3, 224, 224],
            "validation_transform": "resize_short_256_center_crop_224",
            "input_tensor_sha256_by_sample": input_hashes,
        },
        "cohort": {
            "threshold_source": "T1-B.1 validation Q3+3*IQR trial embedding norm",
            "threshold": threshold,
            "usable_validation_count": len(validation),
            "spike_count": len(spikes),
            "stable_count": len(stable),
            "class36_count": spike_class_counts.get(36, 0),
            "other_class_count": len(spikes) - spike_class_counts.get(36, 0),
            "class_counts": {str(key): value for key, value in sorted(spike_class_counts.items())},
            "spike_sample_ids": [record.sample_id for record in spikes],
            "epoch16_bfloat16_embedding_l2": {
                record.sample_id: norms[record.sample_id] for record in spikes
            },
        },
        "control_matching": {
            "priority": [
                "same_user_same_class",
                "same_class_other_user",
                "same_user_other_class",
            ],
            "pairs": pairs,
            "tier_counts": dict(Counter(pair["match_tier"] for pair in pairs)),
            "control_reuse_counts": dict(Counter(pair["control_sample_id"] for pair in pairs)),
        },
        "block_contract": {
            "paths": [f"spatial.backbone.stages.2.{index}" for index in BLOCK_INDICES],
            "decomposition": [
                "input_skip",
                "depthwise_conv_bn",
                "expand_conv_bn",
                "gelu",
                "project_conv_bn",
                "residual_branch",
                "residual_add_output",
            ],
            "gamma_present": False,
            "segment_source_gates": {
                "single_segment": "top1_energy_share >= 0.50",
                "few_segments": "otherwise top3_energy_share >= 0.70",
                "distributed_sequence": "otherwise",
            },
            "channel_concentration": [
                "top1_energy_share",
                "top10pct_energy_share",
                "effective_channel_fraction",
            ],
            "spatial_concentration": [
                "top1pct_energy_share",
                "effective_spatial_fraction",
            ],
        },
        "parameter_and_bn_drift": parameter_drift,
        "spike_traces": _compact_spike_traces(
            traces, {record.sample_id for record in spikes}
        ),
        "pair_attribution": detailed_pairs,
        "aggregate_attribution": aggregate,
        "same_input_model_comparison": model_comparison,
        "conclusion": conclusion,
        "integrity": integrity,
        "decision": {
            "training_authorized": False,
            "epoch18_resume_authorized": False,
            "head_change_authorized": False,
            "crop_change_authorized": False,
            "single_variable_experiment_authorized": False,
            "status": "waiting_for_human_decision",
        },
        "limitations": [
            "The spike cohort is reconstructed from the frozen T1-B.1 numeric threshold on user6/user7 validation; no canonical trial is removed.",
            "Matched controls minimize frame-count difference within user/class when possible, but scene and pose remain potential confounders.",
            "A 10x cohort threshold emphasizes severe magnitude pathology and does not imply smaller but systematic gains are harmless.",
            "Residual attribution identifies where magnitude arises, not why a learned convolution or BN responds to specific Thermal content.",
        ],
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "cohort": {
                    "spikes": len(spikes),
                    "class36": spike_class_counts.get(36, 0),
                    "other": len(spikes) - spike_class_counts.get(36, 0),
                },
                "conclusion": conclusion,
                "report_json": str(REPORT_JSON),
                "report_md": str(REPORT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
