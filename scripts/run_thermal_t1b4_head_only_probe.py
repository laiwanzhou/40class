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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_thermal_t1b3_block_attribution import (
    _compact_block_trace,
    _load_clip,
    _trace_model,
    _trial_scalar_summary,
)
from src.data.ir_primary_full_sequence_dataset import class_map_hash
from src.data.thermal_native_dataset import ThermalNativeDataset, load_development_records
from src.diagnostics.activation_trace import state_dict_digest
from src.models.thermal_iformer_tsm import build_pretrained_iformer_t_expert
from src.train_thermal_head_only_probe import (
    T1B4HeadOnlyRecipe,
    _partitioned_state,
    freeze_backbone_for_head_only,
    train_head_only_probe,
)
from src.train_thermal_native_expert import seed_everything


DATA_ROOT = PROJECT_ROOT.parent / "datasets/Small-Model-Track/train"
AUDIT_PATH = PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json"
SPLIT_PATH = PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json"
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/thermal_iformer_t_tsm_head_only_probe.yaml"
T1B3_PATH = PROJECT_ROOT / "reports/thermal_t1b3_block_attribution.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs/thermal_iformer_t_tsm_head_only_probe_seed20260715"
REPORT_JSON = PROJECT_ROOT / "reports/thermal_t1b4_head_only_probe.json"
REPORT_MD = PROJECT_ROOT / "reports/thermal_t1b4_head_only_probe.md"
PRECISIONS = ("fp32", "bfloat16")
EXPECTED_PRETRAINED_SHA256 = (
    "7cbd778e3604694eb1a0becbf2e6a22798586f6bb46610a5c22b39880efb967e"
)
MEDIAN_RATIO_LIMIT = 1.25
INDIVIDUAL_RATIO_LIMIT = 2.0


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


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def evaluate_tail_gate(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(pair_rows) != 23:
        raise ValueError("T1-B.4 tail gate requires all 23 preregistered pairs")
    fields = ("block5_output_rms_ratio", "embedding_rms_ratio")
    medians = {field: _median([row[field] for row in pair_rows]) for field in fields}
    maxima = {field: max(float(row[field]) for row in pair_rows) for field in fields}
    passed = all(value <= MEDIAN_RATIO_LIMIT for value in medians.values()) and all(
        value <= INDIVIDUAL_RATIO_LIMIT for value in maxima.values()
    )
    return {
        "passed": passed,
        "median_ratio_limit": MEDIAN_RATIO_LIMIT,
        "individual_ratio_limit": INDIVIDUAL_RATIO_LIMIT,
        "median_ratios": medians,
        "max_ratios": maxima,
    }


def _trace_pairs(
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    records_by_id: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_ids = sorted(
        {str(pair["spike_sample_id"]) for pair in pairs}
        | {str(pair["control_sample_id"]) for pair in pairs}
    )
    traces: dict[str, Any] = {}
    for sample_id in sample_ids:
        clips = _load_clip(records_by_id[sample_id]).to(device)
        traces[sample_id] = {
            precision: _trace_model(model, clips, precision=precision)
            for precision in PRECISIONS
        }
    pair_results: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for precision in PRECISIONS:
        rows: list[dict[str, Any]] = []
        for pair in pairs:
            spike_id = str(pair["spike_sample_id"])
            control_id = str(pair["control_sample_id"])
            spike = traces[spike_id][precision]
            control = traces[control_id][precision]
            spike_block = _trial_scalar_summary(spike["blocks"]["5"])
            control_block = _trial_scalar_summary(control["blocks"]["5"])
            rows.append(
                {
                    **pair,
                    "block5_output_rms_ratio": spike_block["output_rms"]
                    / max(control_block["output_rms"], 1e-12),
                    "block5_abs_max_ratio": spike_block["output_abs_max"]
                    / max(control_block["output_abs_max"], 1e-12),
                    "embedding_rms_ratio": spike["embedding"]["rms"]
                    / max(control["embedding"]["rms"], 1e-12),
                    "embedding_abs_max_ratio": spike["embedding"]["abs_max"]
                    / max(control["embedding"]["abs_max"], 1e-12),
                    "logit_rms_ratio": spike["logits"]["rms"]
                    / max(control["logits"]["rms"], 1e-12),
                    "logit_abs_max_ratio": spike["logits"]["abs_max"]
                    / max(control["logits"]["abs_max"], 1e-12),
                    "spike_embedding": spike["embedding"],
                    "control_embedding": control["embedding"],
                    "spike_logits": spike["logits"],
                    "control_logits": control["logits"],
                }
            )
        pair_results[precision] = {
            "pairs": rows,
            "aggregate": {
                group: {
                    "trial_count": len(selected),
                    "median_block5_output_rms_ratio": _median(
                        [row["block5_output_rms_ratio"] for row in selected]
                    ),
                    "max_block5_output_rms_ratio": max(
                        row["block5_output_rms_ratio"] for row in selected
                    ),
                    "median_embedding_rms_ratio": _median(
                        [row["embedding_rms_ratio"] for row in selected]
                    ),
                    "max_embedding_rms_ratio": max(
                        row["embedding_rms_ratio"] for row in selected
                    ),
                    "median_logit_rms_ratio": _median(
                        [row["logit_rms_ratio"] for row in selected]
                    ),
                    "max_logit_rms_ratio": max(row["logit_rms_ratio"] for row in selected),
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
        gates[precision] = evaluate_tail_gate(rows)
    compact_traces = {
        sample_id: {
            precision: {
                "embedding": row["embedding"],
                "logits": row["logits"],
                "predicted_class_id": row["predicted_class_id"],
                "logits_exact_with_and_without_hooks": row[
                    "logits_exact_with_and_without_hooks"
                ],
                "embedding_exact_with_and_without_hooks": row[
                    "embedding_exact_with_and_without_hooks"
                ],
                "block5": _compact_block_trace(row["blocks"]["5"]),
            }
            for precision, row in precision_rows.items()
        }
        for sample_id, precision_rows in traces.items()
    }
    return {"pair_attribution": pair_results, "sample_traces": compact_traces}, gates


def _render_markdown(report: dict[str, Any]) -> str:
    best = report["training"]["best_validation"]
    lines = [
        "# Thermal T1-B.4 frozen-backbone head-only probe",
        "",
        f"- Status: **{report['status']}**",
        f"- Fixed execution: `{report['training']['epochs_completed']}` of `8` epochs; hard stop honored `{report['training']['hard_stop_honored']}`.",
        f"- Best epoch: `{report['training']['best_epoch']}`; validation Macro-F1 `{best['macro_f1']:.6f}`, Accuracy `{best['accuracy']:.6f}`, worst-user Accuracy `{best['worst_user_accuracy']:.6f}`.",
        f"- Frozen backbone exact before/after: `{report['training']['frozen_backbone_exactly_unchanged']}`.",
        f"- Post-training embedding tail disappeared under preregistered gate: **{report['conclusion']['embedding_tail_disappeared']}**.",
        "",
        "## Causal answer",
        "",
        report["conclusion"]["summary"],
        "",
        "| Precision | Block5 median/max | Embedding median/max | Logit median/max | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for precision in PRECISIONS:
        aggregate = report["post_training_trace"]["pair_attribution"][precision]["aggregate"][
            "all_spikes"
        ]
        gate = report["post_training_trace"]["tail_gate"][precision]
        lines.append(
            f"| {precision} | {aggregate['median_block5_output_rms_ratio']:.3f}/{aggregate['max_block5_output_rms_ratio']:.3f} | "
            f"{aggregate['median_embedding_rms_ratio']:.3f}/{aggregate['max_embedding_rms_ratio']:.3f} | "
            f"{aggregate['median_logit_rms_ratio']:.3f}/{aggregate['max_logit_rms_ratio']:.3f} | {gate['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity and scope",
            "",
            f"- Official pretrained SHA: `{report['initialization']['pretrained_weight_sha256']}`.",
            "- The epoch16 model was not loaded. The prior T1-B.3 JSON supplied only the frozen spike/control sample IDs and historical comparison statistics.",
            "- Full-frame, 16-frame normalized-time preprocessing, seed, loss, batch size, and classifier structure were unchanged.",
            "- No crop, BN-free head, activation clamp, residual scaling, heldout labels, competition test, or quarantined evidence was used.",
            "- This is a causal probe only and cannot be promoted from user6/user7 development metrics.",
            "- Training is stopped pending a new human decision.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preregistered Thermal T1-B.4 probe")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("T1-B.4 requires the audited CUDA bfloat16 environment")
    if EXPECTED_PRETRAINED_SHA256 not in CONFIG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("Preregistered official checkpoint identity changed")
    recipe = T1B4HeadOnlyRecipe()
    seed_everything(recipe.seed)
    train_canonical, validation_canonical = load_development_records(
        AUDIT_PATH, SPLIT_PATH, args.data_root
    )
    train = [record for record in train_canonical if record.usable]
    validation = [record for record in validation_canonical if record.usable]
    counts = (len(train_canonical), len(train), len(validation_canonical), len(validation))
    if counts != (2039, 1922, 388, 377):
        raise ValueError(f"Frozen T1-B.4 population changed: {counts}")
    class_hash = _class_hash(train_canonical + validation_canonical)
    t1b3 = json.loads(T1B3_PATH.read_text(encoding="utf-8"))
    pairs = t1b3["control_matching"]["pairs"]
    if len(pairs) != 23 or Counter(pair["match_tier"] for pair in pairs) != {
        "same_user_same_class": 23
    }:
        raise ValueError("T1-B.3 frozen spike/control cohort changed")

    device = torch.device("cuda")
    model = build_pretrained_iformer_t_expert()
    freeze_backbone_for_head_only(model)
    official_backbone, _ = _partitioned_state(model)
    official_backbone_digest = state_dict_digest(official_backbone)
    train_dataset = ThermalNativeDataset(train, training=True, seed=recipe.seed)
    validation_dataset = ThermalNativeDataset(validation, training=False, seed=recipe.seed)
    training = train_head_only_probe(
        model,
        train_dataset,
        validation_dataset,
        device=device,
        output_dir=args.output_dir,
        class_map_hash=class_hash,
        recipe=recipe,
    )
    checkpoint_path = Path(training["checkpoint_path"])
    checkpoint_sha_before_trace = _sha256(checkpoint_path)
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if best.get("epoch16_backbone_loaded") is not False or best.get("probe_only") is not True:
        raise RuntimeError("T1-B.4 checkpoint provenance gate failed")

    traced_model = build_pretrained_iformer_t_expert()
    loaded = traced_model.load_state_dict(best["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("T1-B.4 best checkpoint strict load incomplete")
    freeze_backbone_for_head_only(traced_model)
    traced_backbone, _ = _partitioned_state(traced_model)
    if state_dict_digest(traced_backbone) != official_backbone_digest:
        raise RuntimeError("Best checkpoint backbone differs from official pretrained state")
    traced_model.to(device).eval()
    trace_state_before = state_dict_digest(traced_model.state_dict())
    records_by_id = {record.sample_id: record for record in validation}
    started_trace = time.perf_counter()
    post_trace, gates = _trace_pairs(
        traced_model, pairs, records_by_id, device=device
    )
    trace_seconds = time.perf_counter() - started_trace
    trace_state_after = state_dict_digest(traced_model.state_dict())
    hooks_exact = all(
        row["logits_exact_with_and_without_hooks"]
        and row["embedding_exact_with_and_without_hooks"]
        for precision_rows in post_trace["sample_traces"].values()
        for row in precision_rows.values()
    )
    integrity = {
        "official_backbone_digest": official_backbone_digest,
        "training_backbone_before_matches_official": training[
            "frozen_backbone_state_digest_before"
        ]
        == official_backbone_digest,
        "training_backbone_after_matches_official": training[
            "frozen_backbone_state_digest_after"
        ]
        == official_backbone_digest,
        "trace_state_digest_before": trace_state_before,
        "trace_state_digest_after": trace_state_after,
        "trace_state_unchanged": trace_state_before == trace_state_after,
        "hooks_preserved_logits_and_embeddings_exactly": hooks_exact,
        "checkpoint_sha_before_trace": checkpoint_sha_before_trace,
        "checkpoint_sha_after_trace": _sha256(checkpoint_path),
        "checkpoint_file_unchanged_by_trace": checkpoint_sha_before_trace
        == _sha256(checkpoint_path),
    }
    if not all(
        (
            training["frozen_backbone_exactly_unchanged"],
            integrity["training_backbone_before_matches_official"],
            integrity["training_backbone_after_matches_official"],
            integrity["trace_state_unchanged"],
            integrity["hooks_preserved_logits_and_embeddings_exactly"],
            integrity["checkpoint_file_unchanged_by_trace"],
        )
    ):
        raise RuntimeError("T1-B.4 integrity gate failed")
    tail_disappeared = all(gate["passed"] for gate in gates.values())
    logit_max = max(
        post_trace["pair_attribution"][precision]["aggregate"]["all_spikes"][
            "max_logit_rms_ratio"
        ]
        for precision in PRECISIONS
    )
    if tail_disappeared:
        summary = (
            "With the official pretrained backbone held exactly fixed, the preregistered "
            "block5/final-embedding spike tail disappears. This strengthens the T1-B.3 "
            "conclusion that epoch16 backbone fine-tuning created the abnormal tail. "
            "The development score measures how far the unchanged pretrained Thermal "
            "representation can go with only the original BN-to-Linear head fitted."
        )
    else:
        summary = (
            "The abnormal block5/final-embedding tail remains despite an exactly frozen "
            "official pretrained backbone. The next decision must revisit input/head "
            "interaction before any backbone fine-tuning intervention."
        )
    report = {
        "schema_version": "thermal-t1b4-head-only-probe-v1",
        "stage": "thermal_t1b4",
        "status": "completed_causal_probe_training_stopped_waiting_human_decision",
        "branch": "experiment/thermal-iformer-t-t1b",
        "scientific_baseline_commit": "c42bb43091c79903e5fde5655c2846c87305895a",
        "preregistered_parent_commit": "TO_BE_FILLED_FROM_GIT_AT_RUNTIME",
        "authorization": "explicit_user_instruction_2026_08_20",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "trace_seconds": trace_seconds,
        },
        "initialization": {
            "model": "ChuanyangZheng/iFormer iFormer_t",
            "source_revision": "2a87540fcb345afe9d950a58d0eb3873b938c3dc",
            "pretrained_weight_sha256": EXPECTED_PRETRAINED_SHA256,
            "strict_pretrained_load": True,
            "epoch16_backbone_loaded": False,
            "random_initialization_fallback": False,
        },
        "data_access": {
            "population": "train12_fit_user6_user7_validation_only",
            "counts": {
                "train_canonical": counts[0],
                "train_thermal_usable": counts[1],
                "validation_canonical": counts[2],
                "validation_thermal_usable": counts[3],
            },
            "route": "full_frame",
            "segments": 16,
            "timeline": "thermal_native_normalized_time",
            "heldout_labels_accessed": False,
            "competition_test_accessed": False,
            "quarantined_evidence_accessed": False,
            "ir_x3d_modified": False,
        },
        "inputs": {
            "audit_sha256": _sha256(AUDIT_PATH),
            "split_sha256": _sha256(SPLIT_PATH),
            "config_sha256": _sha256(CONFIG_PATH),
            "t1b3_cohort_report_sha256": _sha256(T1B3_PATH),
            "class_map_hash": class_hash,
        },
        "training": training,
        "post_training_trace": {
            "cohort": {
                "spike_count": 23,
                "matched_control_count": 23,
                "class36_spike_count": sum(
                    pair["class_group"] == "class36" for pair in pairs
                ),
                "other_class_spike_count": sum(
                    pair["class_group"] == "other_classes" for pair in pairs
                ),
                "match_tier_counts": dict(Counter(pair["match_tier"] for pair in pairs)),
            },
            **post_trace,
            "tail_gate": gates,
            "integrity": integrity,
        },
        "conclusion": {
            "embedding_tail_disappeared": tail_disappeared,
            "maximum_matched_logit_rms_ratio": logit_max,
            "head_or_input_interaction_flag": bool(tail_disappeared and logit_max > 2.0),
            "backbone_finetuning_tail_causality_strengthened": tail_disappeared,
            "promotion_authorized": False,
            "training_remains_stopped": True,
            "summary": summary,
        },
        "limitations": [
            "Causal development probe only; user6/user7 metrics cannot promote a candidate.",
            "The unchanged pretrained representation was not selected using shared train-14 OOF.",
            "Any next partial-finetuning or lower-backbone-LR run requires a new human decision.",
        ],
    }
    try:
        import subprocess

        report["preregistered_parent_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("Could not record preregistration commit") from None
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON),
                "best_epoch": training["best_epoch"],
                "best_macro_f1": training["best_validation"]["macro_f1"],
                "best_accuracy": training["best_validation"]["accuracy"],
                "embedding_tail_disappeared": tail_disappeared,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
