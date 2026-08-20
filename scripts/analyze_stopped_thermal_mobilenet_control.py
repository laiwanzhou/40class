from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_thermal_mobilenet_matched_control import (
    AUDIT_PATH,
    DATA_ROOT,
    OUTPUT_DIR,
    PRECISIONS,
    SPLIT_PATH,
    T1B1_PATH,
    T1B3_PATH,
    T1B4_PATH,
    _deployment_ledger,
    _matched_stability_trace,
    _validation_embedding_scan,
)
from src.data.thermal_native_dataset import load_development_records
from src.diagnostics.activation_trace import state_dict_digest
from src.models.thermal_mobilenet_tsm import build_pretrained_mobilenet_expert


REPORT_JSON = PROJECT_ROOT / "reports/thermal_mobilenetv3_tsm_epoch14_stopped_audit.json"
REPORT_MD = PROJECT_ROOT / "reports/thermal_mobilenetv3_tsm_epoch14_stopped_audit.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render(report: dict) -> str:
    best = report["best_checkpoint"]["validation"]
    comparison = report["comparison"]
    fp32 = report["stability"]["matched_pairs"]["comparisons"]["fp32"]["aggregate"]
    bf16 = report["stability"]["matched_pairs"]["comparisons"]["bfloat16"]["aggregate"]
    scan = report["stability"]["all_validation_scan"]["embedding_l2"]
    return "\n".join(
        [
            "# Thermal MobileNetV3-Small + TSM stopped-run audit",
            "",
            "- Status: **stopped after epoch 14 by explicit human decision**",
            f"- Best checkpoint: epoch `{report['best_checkpoint']['epoch']}`, SHA256 `{report['best_checkpoint']['sha256']}`.",
            f"- Accuracy / Macro-F1 / worst-user Accuracy: `{best['accuracy']:.6f}` / `{best['macro_f1']:.6f}` / `{best['worst_user_accuracy']:.6f}`.",
            f"- user6 Accuracy/Macro-F1: `{best['per_user']['user6']['accuracy']:.6f}` / `{best['per_user']['user6']['macro_f1_fixed_0_39']:.6f}`.",
            f"- user7 Accuracy/Macro-F1: `{best['per_user']['user7']['accuracy']:.6f}` / `{best['per_user']['user7']['macro_f1_fixed_0_39']:.6f}`.",
            f"- Zero-recall validation classes: `{len(best['zero_recall_class_ids'])}/40`.",
            "",
            "## Stop assessment",
            "",
            report["stop_assessment"]["summary"],
            "",
            "The run did not satisfy the preregistered 30-epoch completion contract and is not represented as complete. Epochs 15-30 may in principle contain a later stochastic improvement, but the observed generalization and loss trends do not justify that compute for this development control.",
            "",
            "## Matched comparison",
            "",
            "| Model | Accuracy | Macro-F1 | Worst-user Accuracy | Stability |",
            "|---|---:|---:|---:|---|",
            f"| iFormer-T epoch16 | {comparison['iformer_epoch16']['accuracy']:.6f} | {comparison['iformer_epoch16']['macro_f1']:.6f} | {comparison['iformer_epoch16']['worst_user_accuracy']:.6f} | abnormal fine-tuned activation tail |",
            f"| frozen iFormer-T | {comparison['frozen_iformer_t']['accuracy']:.6f} | {comparison['frozen_iformer_t']['macro_f1']:.6f} | {comparison['frozen_iformer_t']['worst_user_accuracy']:.6f} | stable, insufficient representation |",
            f"| MobileNet epoch12/14-stop | {best['accuracy']:.6f} | {best['macro_f1']:.6f} | {best['worst_user_accuracy']:.6f} | {report['conclusion']['activation_interpretation']} |",
            "",
            f"MobileNet minus epoch16 iFormer: Accuracy `{comparison['mobilenet_minus_iformer_epoch16']['accuracy']:+.6f}`, Macro-F1 `{comparison['mobilenet_minus_iformer_epoch16']['macro_f1']:+.6f}`, worst-user `{comparison['mobilenet_minus_iformer_epoch16']['worst_user_accuracy']:+.6f}`.",
            "",
            f"MobileNet minus frozen iFormer: Accuracy `{comparison['mobilenet_minus_frozen_iformer']['accuracy']:+.6f}`, Macro-F1 `{comparison['mobilenet_minus_frozen_iformer']['macro_f1']:+.6f}`, worst-user `{comparison['mobilenet_minus_frozen_iformer']['worst_user_accuracy']:+.6f}`.",
            "",
            "## Activation stability",
            "",
            f"- FP32 maximum matched feature-block / embedding RMS ratios: `{fp32['maximum_feature_block_rms_ratio']:.3f}` / `{fp32['maximum_embedding_rms_ratio']:.3f}`; 10x event `{fp32['feature_or_embedding_10x_event']}`.",
            f"- bfloat16 maximum matched feature-block / embedding RMS ratios: `{bf16['maximum_feature_block_rms_ratio']:.3f}` / `{bf16['maximum_embedding_rms_ratio']:.3f}`; 10x event `{bf16['feature_or_embedding_10x_event']}`.",
            f"- All-validation embedding L2 median/max: `{scan['median']:.3f}` / `{scan['max']:.3f}`; robust outliers `{scan['robust_outlier_count']}/377`.",
            f"- Hooks exact: `{report['stability']['matched_pairs']['hooks_preserved_logits_and_embeddings_exactly']}`; state unchanged: `{report['stability']['state_unchanged']}`.",
            "",
            "## Decision",
            "",
            report["conclusion"]["summary"],
            "",
            f"Checkpoint bytes: `{report['best_checkpoint']['bytes']}`. Provisional deduplicated package: `{report['deployment_ledger']['total_serialized_bytes']}` bytes; strict limit pass `{report['deployment_ledger']['passes_strict_limit']}`.",
            "",
            "This stopped development control cannot auto-promote. Any formal retention requires shared train-14 OOF, IR unique-correct/oracle-pair evidence, deployment bytes, and latency. Training remains stopped.",
            "",
        ]
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Read-only stability audit requires CUDA")
    history_path = OUTPUT_DIR / "history.json"
    checkpoint_path = OUTPUT_DIR / "best_macro_f1.pt"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if len(history) != 14 or [row["epoch"] for row in history] != list(range(1, 15)):
        raise ValueError("Stopped MobileNet history must contain exact epochs 1..14")
    checkpoint_sha = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["epoch"]) != 12:
        raise ValueError("Expected epoch 12 to remain the stopped-run best")
    best = checkpoint["validation_metrics"]

    _, validation_canonical = load_development_records(AUDIT_PATH, SPLIT_PATH, DATA_ROOT)
    validation = [record for record in validation_canonical if record.usable]
    if len(validation) != 377:
        raise ValueError("Frozen validation population changed")
    model = build_pretrained_mobilenet_expert()
    loaded = model.load_state_dict(checkpoint["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Stopped checkpoint strict load incomplete")
    device = torch.device("cuda")
    model.to(device).eval()
    state_before = state_dict_digest(model.state_dict())
    t1b3 = json.loads(T1B3_PATH.read_text(encoding="utf-8"))
    pairs = t1b3["control_matching"]["pairs"]
    records_by_id = {record.sample_id: record for record in validation}
    started = time.perf_counter()
    matched = _matched_stability_trace(model, pairs, records_by_id, device=device)
    scan = _validation_embedding_scan(model, validation, device=device)
    state_after = state_dict_digest(model.state_dict())
    stability = {
        "matched_pairs": matched,
        "all_validation_scan": scan,
        "state_digest_before": state_before,
        "state_digest_after": state_after,
        "state_unchanged": state_before == state_after,
        "checkpoint_file_unchanged": checkpoint_sha == _sha256(checkpoint_path),
        "seconds": time.perf_counter() - started,
    }
    if not all(
        (
            matched["hooks_preserved_logits_and_embeddings_exactly"],
            stability["state_unchanged"],
            stability["checkpoint_file_unchanged"],
        )
    ):
        raise RuntimeError("Stopped-run read-only integrity gate failed")

    t1b1 = json.loads(T1B1_PATH.read_text(encoding="utf-8"))
    epoch16 = t1b1["controlled_modes"]["checkpoint_eval"]
    frozen = json.loads(T1B4_PATH.read_text(encoding="utf-8"))["training"]["best_validation"]
    comparison = {
        "iformer_epoch16": {
            "accuracy": epoch16["accuracy"],
            "macro_f1": epoch16["macro_f1_fixed_0_39"],
            "worst_user_accuracy": epoch16["worst_user_accuracy"],
        },
        "frozen_iformer_t": {
            "accuracy": frozen["accuracy"],
            "macro_f1": frozen["macro_f1"],
            "worst_user_accuracy": frozen["worst_user_accuracy"],
        },
        "mobilenet_minus_iformer_epoch16": {
            "accuracy": best["accuracy"] - epoch16["accuracy"],
            "macro_f1": best["macro_f1"] - epoch16["macro_f1_fixed_0_39"],
            "worst_user_accuracy": best["worst_user_accuracy"] - epoch16["worst_user_accuracy"],
        },
        "mobilenet_minus_frozen_iformer": {
            "accuracy": best["accuracy"] - frozen["accuracy"],
            "macro_f1": best["macro_f1"] - frozen["macro_f1"],
            "worst_user_accuracy": best["worst_user_accuracy"] - frozen["worst_user_accuracy"],
        },
    }
    by_epoch = {row["epoch"]: row for row in history}
    best_row = by_epoch[12]
    final_row = by_epoch[14]
    min_loss_row = min(history, key=lambda row: row["validation"]["loss"])
    stop_assessment = {
        "epochs_completed": 14,
        "planned_epochs": 30,
        "user_approved_early_stop": True,
        "best_epoch": 12,
        "validation_loss_min_epoch": min_loss_row["epoch"],
        "validation_loss_min": min_loss_row["validation"]["loss"],
        "epoch13_macro_f1_delta_from_best": by_epoch[13]["validation"]["macro_f1"]
        - best["macro_f1"],
        "epoch14_macro_f1_delta_from_best": final_row["validation"]["macro_f1"]
        - best["macro_f1"],
        "best_train_validation_accuracy_gap": best_row["train"]["accuracy"]
        - best_row["validation"]["accuracy"],
        "final_train_validation_accuracy_gap": final_row["train"]["accuracy"]
        - final_row["validation"]["accuracy"],
        "later_best_cannot_be_excluded": True,
        "summary": (
            "Epoch 12 is the observed best. Epochs 13 and 14 regress on the primary metric "
            "while train accuracy continues rising, the train-validation gap widens, and "
            f"validation loss had already reached its minimum at epoch {min_loss_row['epoch']}. "
            "A later best cannot be excluded, but the observed curve supports the approved "
            "compute stop because this control is not near an automatic-retention threshold."
        ),
    }
    any_10x = any(
        matched["comparisons"][precision]["aggregate"]["feature_or_embedding_10x_event"]
        for precision in PRECISIONS
    )
    activation = (
        "matched 10x activation event observed"
        if any_10x
        else "no matched 10x activation event"
    )
    architecture = (
        "MobileNet reproduces an extreme tail, so instability is not iFormer-specific."
        if any_10x
        else "MobileNet does not reproduce the iFormer epoch16 extreme tail, supporting an iFormer-specific fine-tuning response under this recipe."
    )
    report = {
        "schema_version": "thermal-mobilenet-stopped-audit-v1",
        "stage": "thermal_mobilenet_matched_control",
        "status": "stopped_after_epoch14_by_human_decision_analysis_complete",
        "branch": "experiment/thermal-iformer-t-t1b",
        "preregistered_parent_commit": "ed525d867b1b50f4601a8ae0903756fec28e5421",
        "analysis_commit_parent": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "safety": {
            "analysis_training_performed": False,
            "backward_called": False,
            "optimizer_created": False,
            "restart_stopped": True,
            "heldout_labels_accessed": False,
            "competition_test_accessed": False,
            "quarantined_evidence_accessed": False,
            "yolo_crop_used": False,
            "ir_x3d_modified": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
        "history": history,
        "stop_assessment": stop_assessment,
        "best_checkpoint": {
            "epoch": int(checkpoint["epoch"]),
            "path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha,
            "validation": best,
            "latency": best_row["latency"],
        },
        "stability": stability,
        "comparison": comparison,
        "deployment_ledger": _deployment_ledger(checkpoint_path),
        "conclusion": {
            "activation_interpretation": activation,
            "architecture_interpretation": architecture,
            "automatic_promotion_authorized": False,
            "formal_retention_requires_shared_train14_oof": True,
            "training_remains_stopped": True,
            "summary": (
                f"{architecture} The stopped MobileNet control is substantially stronger "
                "than frozen iFormer and more balanced than epoch16 iFormer, but its 0.286 "
                "development accuracy and 17 zero-recall classes do not justify automatic "
                "promotion or further development compute."
            ),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_render(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON),
                "best_epoch": 12,
                "accuracy": best["accuracy"],
                "macro_f1": best["macro_f1"],
                "any_10x_activation": any_10x,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
