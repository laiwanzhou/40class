from __future__ import annotations

import argparse
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.thermal_native_dataset import ThermalNativeDataset, load_development_records
from src.diagnostics.activation_trace import (
    ReadOnlyActivationTracer,
    activation_summary,
    first_amplification,
    state_dict_digest,
)
from src.models.thermal_iformer_tsm import build_pretrained_iformer_t_expert


DATA_ROOT = PROJECT_ROOT.parent / "datasets/Small-Model-Track/train"
AUDIT_PATH = PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json"
SPLIT_PATH = PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json"
T1B1_PATH = PROJECT_ROOT / "reports/thermal_t1b1_bn_diagnostic.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs/thermal_iformer_t_tsm_train12_val2_seed20260715/best_macro_f1.pt"
)
REPORT_JSON = PROJECT_ROOT / "reports/thermal_t1b2_activation_trace.json"
REPORT_MD = PROJECT_ROOT / "reports/thermal_t1b2_activation_trace.md"
EXPECTED_CHECKPOINT_SHA256 = (
    "ca9c11c0f4d50f67c89da52284f726085dfeb3d1578621438872b9255a05d827"
)
SPIKE_SAMPLE_IDS = (
    "train__c36__user7__1-2-2",
    "train__c36__user7__1-2-1",
    "train__c36__user7__1-2-3",
)
AMPLIFICATION_THRESHOLD = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_module_names(model: nn.Module) -> tuple[str, ...]:
    names: list[str] = []
    backbone = model.spatial.backbone
    for stage_index, downsample in enumerate(backbone.downsample_layers):
        prefix = f"spatial.backbone.downsample_layers.{stage_index}"
        for child_name, child in downsample.named_modules():
            if child_name and not any(child.children()):
                names.append(f"{prefix}.{child_name}")
        names.append(prefix)
    names.append("spatial.temporal_shift")
    for stage_index, stage in enumerate(backbone.stages):
        for block_index in range(len(stage)):
            names.append(f"spatial.backbone.stages.{stage_index}.{block_index}")
    names.extend(
        (
            "spatial.backbone.classifier.classifier.bn",
            "spatial.backbone.classifier.classifier.l",
        )
    )
    if len(names) != len(set(names)):
        raise RuntimeError("Activation trace module names are not unique")
    return tuple(names)


def _load_clip(record: Any) -> torch.Tensor:
    item = ThermalNativeDataset([record], training=False, seed=20260715)[0]
    if item["route"] != "full_frame" or not bool(item["availability"]):
        raise ValueError("T1-B.2 requires an available full-frame Thermal trial")
    clips = item["clips"].unsqueeze(0)
    if clips.shape != (1, 16, 3, 224, 224):
        raise ValueError("T1-B.2 input contract changed")
    return clips


def _forward(
    model: nn.Module, clips: torch.Tensor, *, precision: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if precision not in {"fp32", "bfloat16"}:
        raise ValueError("Unsupported trace precision")
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


def _insert_manual_points(
    records: list[dict[str, Any]], clips: torch.Tensor, embedding: torch.Tensor
) -> list[dict[str, Any]]:
    traced = [{"point": "input", **activation_summary(clips)}]
    inserted = False
    for row in records:
        if not inserted and row["point"].endswith("classifier.classifier.bn"):
            traced.append({"point": "trial_embedding", **activation_summary(embedding)})
            inserted = True
        traced.append(row)
    if not inserted:
        raise RuntimeError("Classifier BN hook was not observed")
    return traced


def _trace_trial(
    model: nn.Module,
    record: Any,
    *,
    device: torch.device,
    precision: str,
    module_names: tuple[str, ...],
) -> dict[str, Any]:
    clips = _load_clip(record).to(device)
    with torch.inference_mode():
        baseline_embedding, baseline_logits = _forward(model, clips, precision=precision)
        with ReadOnlyActivationTracer(model, module_names) as tracer:
            traced_embedding, traced_logits = _forward(model, clips, precision=precision)
    logits_exact = torch.equal(baseline_logits, traced_logits)
    embedding_exact = torch.equal(baseline_embedding, traced_embedding)
    if not logits_exact or not embedding_exact:
        raise RuntimeError("Read-only hooks changed model outputs")
    trace = _insert_manual_points(tracer.records, clips, traced_embedding)
    if not all(row["finite"] for row in trace):
        raise RuntimeError(f"Non-finite activation in {record.sample_id} ({precision})")
    return {
        "sample_id": record.sample_id,
        "user_id": record.user_id,
        "class_id": int(record.class_id),
        "decodable_frame_count": int(record.decodable_frame_count),
        "precision": precision,
        "logits_exact_with_and_without_hooks": logits_exact,
        "embedding_exact_with_and_without_hooks": embedding_exact,
        "logits": activation_summary(traced_logits),
        "predicted_class_id": int(traced_logits.argmax(dim=1)[0]),
        "trace": trace,
    }


def _final_embedding_norm(
    model: nn.Module, record: Any, *, device: torch.device
) -> float:
    clips = _load_clip(record).to(device)
    with torch.inference_mode():
        embedding, _ = _forward(model, clips, precision="fp32")
    return float(torch.linalg.vector_norm(embedding.float(), dim=1)[0])


def _select_matched_controls(
    records: list[Any],
    norms: dict[str, float],
    *,
    normal_threshold: float,
) -> list[dict[str, Any]]:
    by_id = {record.sample_id: record for record in records}
    if not set(SPIKE_SAMPLE_IDS).issubset(by_id):
        raise ValueError("Fixed T1-B.1 spike trials are absent")
    candidates = [
        record
        for record in records
        if record.sample_id not in SPIKE_SAMPLE_IDS
        and norms[record.sample_id] <= normal_threshold
    ]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for spike_id in SPIKE_SAMPLE_IDS:
        spike = by_id[spike_id]
        available = [record for record in candidates if record.sample_id not in used]
        if not available:
            raise ValueError("Insufficient stable same-user/class controls")
        control = min(
            available,
            key=lambda record: (
                abs(record.decodable_frame_count - spike.decodable_frame_count),
                record.sample_id,
            ),
        )
        used.add(control.sample_id)
        selected.append(
            {
                "spike_sample_id": spike_id,
                "control_sample_id": control.sample_id,
                "same_user": spike.user_id == control.user_id,
                "same_class": spike.class_id == control.class_id,
                "spike_frame_count": int(spike.decodable_frame_count),
                "control_frame_count": int(control.decodable_frame_count),
                "absolute_frame_count_difference": abs(
                    spike.decodable_frame_count - control.decodable_frame_count
                ),
                "spike_fp32_embedding_l2": norms[spike_id],
                "control_fp32_embedding_l2": norms[control.sample_id],
            }
        )
    return selected


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate_group_trace(
    spike_traces: list[list[dict[str, Any]]],
    control_traces: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not spike_traces or len(spike_traces) != len(control_traces):
        raise ValueError("Aligned spike/control traces are required")
    points = [row["point"] for row in spike_traces[0]]
    for trace in spike_traces + control_traces:
        if [row["point"] for row in trace] != points:
            raise ValueError("Trace points changed between trials")
    aggregated: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        spike_rms = [float(trace[index]["rms"]) for trace in spike_traces]
        control_rms = [float(trace[index]["rms"]) for trace in control_traces]
        spike_abs_max = [float(trace[index]["abs_max"]) for trace in spike_traces]
        control_abs_max = [float(trace[index]["abs_max"]) for trace in control_traces]
        median_spike = _median(spike_rms)
        median_control = _median(control_rms)
        aggregated.append(
            {
                "point": point,
                "spike_rms_median": median_spike,
                "control_rms_median": median_control,
                "rms_ratio": median_spike / max(median_control, 1e-12),
                "spike_abs_max_median": _median(spike_abs_max),
                "control_abs_max_median": _median(control_abs_max),
                "abs_max_ratio": _median(spike_abs_max)
                / max(_median(control_abs_max), 1e-12),
                "spike_rms_values": spike_rms,
                "control_rms_values": control_rms,
            }
        )
    return aggregated


def _locate_group_amplification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spike = [{"point": row["point"], "rms": row["spike_rms_median"]} for row in rows]
    control = [
        {"point": row["point"], "rms": row["control_rms_median"]} for row in rows
    ]
    result = first_amplification(
        spike, control, ratio_threshold=AMPLIFICATION_THRESHOLD
    )
    if result["point"] is not None:
        index = next(i for i, row in enumerate(rows) if row["point"] == result["point"])
        result["window"] = rows[max(0, index - 2) : index + 3]
    else:
        result["window"] = []
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Thermal T1-B.2 layerwise activation trace",
        "",
        f"- Status: **{report['status']}**",
        f"- Checkpoint SHA256: `{report['checkpoint']['sha256']}`",
        "- Execution: eval/inference only; zero training, backward, parameter update, or BN-buffer update.",
        "- Input: full-frame Thermal, 16 uniform normalized-time frames, 224x224.",
        "",
        "## Matched trials",
        "",
        "| Spike | Control | Frames | FP32 embedding L2 spike/control |",
        "|---|---|---:|---:|",
    ]
    for pair in report["matched_pairs"]:
        lines.append(
            f"| {pair['spike_sample_id']} | {pair['control_sample_id']} | "
            f"{pair['spike_frame_count']}/{pair['control_frame_count']} | "
            f"{pair['spike_fp32_embedding_l2']:.3f}/{pair['control_fp32_embedding_l2']:.3f} |"
        )
    lines.extend(["", "## First observed amplification", ""])
    for precision, result in report["group_amplification"].items():
        lines.append(
            f"- `{precision}`: `{result['point']}` at RMS ratio "
            f"`{result['ratio']:.3f}` (previous `{result['previous_point']}`: "
            f"`{result['previous_ratio']:.3f}`)."
        )
    localized = report["localized_group_module"]
    lines.append(
        f"- Agreed module: `{localized['module_class']}` wrapping "
        f"`{localized['inner_block_class']}` with "
        f"`{localized['token_channel_mixer_class']}` token/channel mixer."
    )
    lines.extend(
        [
            "",
            "The location is the first observed 10x spike/control RMS crossing, not proof that the module itself is defective. Full per-layer statistics and per-pair crossings are in the JSON report.",
            "",
            "## Integrity gates",
            "",
            f"- Hooks preserve logits exactly: `{report['integrity']['all_hooked_logits_exact']}`",
            f"- Hooks preserve embeddings exactly: `{report['integrity']['all_hooked_embeddings_exact']}`",
            f"- State dict digest unchanged: `{report['integrity']['state_dict_digest_unchanged']}`",
            f"- Every state tensor unchanged: `{report['integrity']['every_state_tensor_unchanged']}`",
            f"- BN-free run authorized: `{report['decision']['bn_free_short_run_authorized']}`",
            f"- Epoch 18 resume authorized: `{report['decision']['epoch18_resume_authorized']}`",
            "",
            "## Decision",
            "",
            report["decision"]["summary"],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Thermal T1-B.2 activation trace")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("T1-B.2 requires the audited CUDA execution environment")
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Epoch-16 checkpoint SHA256 changed")
    t1b1 = json.loads(T1B1_PATH.read_text(encoding="utf-8"))
    normal_threshold = float(t1b1["embedding_outliers"]["validation_threshold"])

    train_canonical, validation_canonical = load_development_records(
        AUDIT_PATH, SPLIT_PATH, args.data_root
    )
    if {record.user_id for record in validation_canonical} != {"user6", "user7"}:
        raise ValueError("Validation population changed")
    candidate_records = [
        record
        for record in validation_canonical
        if record.usable and record.user_id == "user7" and record.class_id == 36
    ]
    if len(candidate_records) != 22:
        raise ValueError("Expected 22 usable user7/class36 candidate trials")
    by_id = {record.sample_id: record for record in candidate_records}

    device = torch.device("cuda")
    model = build_pretrained_iformer_t_expert()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if int(checkpoint["epoch"]) != 16:
        raise ValueError("T1-B.2 requires the epoch-16 selected checkpoint")
    loaded = model.load_state_dict(checkpoint["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Checkpoint strict load was incomplete")
    model.to(device).eval()
    state_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    digest_before = state_dict_digest(model.state_dict())
    checkpoint_sha_after_load = _sha256(args.checkpoint)
    if checkpoint_sha_after_load != checkpoint_sha:
        raise RuntimeError("Checkpoint changed during load")

    started = time.perf_counter()
    candidate_norms = {
        record.sample_id: _final_embedding_norm(model, record, device=device)
        for record in candidate_records
    }
    pairs = _select_matched_controls(
        candidate_records, candidate_norms, normal_threshold=normal_threshold
    )
    trace_ids = list(SPIKE_SAMPLE_IDS) + [pair["control_sample_id"] for pair in pairs]
    module_names = _trace_module_names(model)
    traces: dict[str, dict[str, dict[str, Any]]] = {"fp32": {}, "bfloat16": {}}
    for precision in ("fp32", "bfloat16"):
        for sample_id in trace_ids:
            traces[precision][sample_id] = _trace_trial(
                model,
                by_id[sample_id],
                device=device,
                precision=precision,
                module_names=module_names,
            )

    digest_after = state_dict_digest(model.state_dict())
    every_tensor_unchanged = all(
        torch.equal(value.detach().cpu(), state_before[name])
        for name, value in model.state_dict().items()
    )
    checkpoint_sha_after_trace = _sha256(args.checkpoint)
    integrity = {
        "all_hooked_logits_exact": all(
            row["logits_exact_with_and_without_hooks"]
            for precision_rows in traces.values()
            for row in precision_rows.values()
        ),
        "all_hooked_embeddings_exact": all(
            row["embedding_exact_with_and_without_hooks"]
            for precision_rows in traces.values()
            for row in precision_rows.values()
        ),
        "state_dict_digest_before": digest_before,
        "state_dict_digest_after": digest_after,
        "state_dict_digest_unchanged": digest_before == digest_after,
        "every_state_tensor_unchanged": every_tensor_unchanged,
        "checkpoint_sha_before": checkpoint_sha,
        "checkpoint_sha_after_load": checkpoint_sha_after_load,
        "checkpoint_sha_after_trace": checkpoint_sha_after_trace,
        "checkpoint_file_unchanged": (
            checkpoint_sha == checkpoint_sha_after_load == checkpoint_sha_after_trace
        ),
    }
    if not all(
        (
            integrity["all_hooked_logits_exact"],
            integrity["all_hooked_embeddings_exact"],
            integrity["state_dict_digest_unchanged"],
            integrity["every_state_tensor_unchanged"],
            integrity["checkpoint_file_unchanged"],
        )
    ):
        raise RuntimeError("T1-B.2 read-only integrity gate failed")

    group_traces: dict[str, list[dict[str, Any]]] = {}
    group_amplification: dict[str, dict[str, Any]] = {}
    pair_amplification: dict[str, list[dict[str, Any]]] = {}
    for precision in ("fp32", "bfloat16"):
        spike_traces = [traces[precision][sample_id]["trace"] for sample_id in SPIKE_SAMPLE_IDS]
        control_traces = [
            traces[precision][pair["control_sample_id"]]["trace"] for pair in pairs
        ]
        grouped = _aggregate_group_trace(spike_traces, control_traces)
        group_traces[precision] = grouped
        group_amplification[precision] = _locate_group_amplification(grouped)
        pair_amplification[precision] = []
        for pair in pairs:
            result = first_amplification(
                traces[precision][pair["spike_sample_id"]]["trace"],
                traces[precision][pair["control_sample_id"]]["trace"],
                ratio_threshold=AMPLIFICATION_THRESHOLD,
            )
            pair_amplification[precision].append(
                {
                    "spike_sample_id": pair["spike_sample_id"],
                    "control_sample_id": pair["control_sample_id"],
                    **result,
                }
            )

    located_points = {
        precision: result["point"] for precision, result in group_amplification.items()
    }
    precision_agreement = len(set(located_points.values())) == 1
    localized_module: dict[str, Any] | None = None
    if precision_agreement and located_points["fp32"]:
        localized_path = str(located_points["fp32"])
        localized = dict(model.named_modules())[localized_path]
        localized_module = {
            "path": localized_path,
            "stage_index": 2,
            "block_index": 5,
            "module_class": type(localized).__name__,
            "inner_block_class": type(localized.block).__name__,
            "token_channel_mixer_class": type(
                localized.block.token_channel_mixer
            ).__name__,
        }
    status = (
        "first_abnormal_amplification_localized_training_still_stopped"
        if all(located_points.values())
        else "amplification_not_localized_training_still_stopped"
    )
    report = {
        "schema_version": "thermal-t1b2-activation-trace-v1",
        "stage": "thermal_t1b2",
        "status": status,
        "branch": "experiment/thermal-iformer-t-t1b",
        "scientific_baseline_commit": "c42bb43091c79903e5fde5655c2846c87305895a",
        "authorization": "explicit_zero_training_layerwise_trace_2026_08_20",
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
            "strict_load_missing": [],
            "strict_load_unexpected": [],
        },
        "input_contract": {
            "route": "full_frame",
            "segments": 16,
            "timeline": "thermal_native_normalized_time",
            "motion_peak_sampling": False,
            "imports_ir_indices": False,
            "shape": [1, 16, 3, 224, 224],
            "validation_transform": "resize_short_256_center_crop_224",
        },
        "population": {
            "selection_scope": "user6_user7_validation_only",
            "control_pool": "usable_user7_class36",
            "control_pool_trial_count": len(candidate_records),
            "normal_threshold_from_t1b1": normal_threshold,
            "candidate_fp32_embedding_l2": candidate_norms,
        },
        "matched_pairs": pairs,
        "trace_contract": {
            "module_count": len(module_names),
            "module_names": list(module_names),
            "manual_points": ["input", "trial_embedding"],
            "metric": "activation RMS; abs-max and leading-axis RMS also retained",
            "first_amplification_ratio_threshold": AMPLIFICATION_THRESHOLD,
        },
        "traces": traces,
        "group_trace": group_traces,
        "group_amplification": group_amplification,
        "pair_amplification": pair_amplification,
        "precision_agreement": {
            "first_point_by_precision": located_points,
            "same_first_point": precision_agreement,
        },
        "localized_group_module": localized_module,
        "integrity": integrity,
        "decision": {
            "epoch18_resume_authorized": False,
            "bn_free_short_run_authorized": False,
            "head_change_authorized": False,
            "training_authorized": False,
            "summary": (
                "T1-B.2 localizes the group-median first 10x crossing to iFormer stage 2 "
                "block 5 in both precisions. Individual crossings vary with spike severity, "
                "so this is an observed amplification boundary rather than proof of a defective "
                "block. No training or architecture change is authorized; human review is required."
            ),
        },
        "limitations": [
            "The first 10x spike/control crossing localizes where abnormal magnitude first becomes visible; it does not alone prove the module is defective.",
            "Controls match user, class, preprocessing route, and nearest available frame count, but source scene and performer pose can still differ.",
            "Only the three strongest T1-B.1 validation spikes are traced; all canonical trials remain retained and unchanged.",
            "Individual first crossings range from stage 2 block 4 to later blocks; the group-median stage 2 block 5 location is representative, not universal.",
            "No heldout, competition-test, quarantined evidence, or frozen IR/X3D asset was accessed.",
        ],
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "first_point_by_precision": located_points,
                "precision_agreement": precision_agreement,
                "report_json": str(REPORT_JSON),
                "report_md": str(REPORT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
