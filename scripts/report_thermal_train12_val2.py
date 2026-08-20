from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "reports/thermal_iformer_t_tsm_train12_val2.json"
REPORT_PATH = PROJECT_ROOT / "reports/thermal_iformer_t_tsm_train12_val2.md"


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    training = result["training"]
    metrics = training["best_validation"]
    ledger = result["deployment_ledger"]
    latency = training["best_latency"]
    lines = [
        "# Thermal iFormer-T + TSM T1-B Development Result",
        "",
        "## Boundary",
        "",
        f"This authorized development run used train12 for fitting and combined user6+user7 for selection on branch `{result['branch']}`. It did not access heldout labels, competition test data, quarantined evidence, or modify frozen IR/X3D assets.",
        "",
        "## Frozen Input And Recipe",
        "",
        "- View: full Thermal frame only; resize short side to 256, then one shared 224 crop across all trial frames.",
        "- Time: 16 uniform normalized Thermal-native targets; no IR indices and no motion-peak sampling.",
        "- Model: official pretrained ChuanyangZheng iFormer-T with TSM at four native stage boundaries.",
        "- Selection: fixed-label 0..39 Macro-F1, then Accuracy, then worst-user Accuracy, then lower epoch.",
        f"- Population: {result['population']['train_thermal_usable']} usable train trials and {result['population']['validation_thermal_usable']} usable validation trials. Canonical unavailable rows remain inventoried but do not enter BatchNorm, loss, or selection metrics.",
        "",
        "## Best Checkpoint",
        "",
        f"- Epoch: **{training['best_epoch']} / {training['epochs_completed']}**",
        f"- Combined Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Combined Macro-F1 (labels 0..39): **{metrics['macro_f1']:.4f}**",
        f"- Worst-user Accuracy: **{metrics['worst_user_accuracy']:.4f}**",
    ]
    for user_id, row in metrics["per_user"].items():
        lines.append(
            f"- {user_id}: Accuracy **{row['accuracy']:.4f}**, fixed-label Macro-F1 **{row['macro_f1_fixed_0_39']:.4f}**, present classes {row['present_class_count']}/40, usable trials {row['trial_count']}"
        )
    lines.extend(
        [
            f"- Zero-recall class IDs: `{metrics['zero_recall_class_ids']}`",
            "",
            "A single user's absent classes are reporting support gaps, not evidence of zero model capability. The ten low-support validation classes remain protected from architecture retuning based on one or two errors.",
            "",
            "## Resources",
            "",
            f"- Checkpoint: `{training['checkpoint_path']}`",
            f"- Serialized checkpoint bytes: **{training['checkpoint_bytes']:,}**",
            f"- Checkpoint SHA256: `{training['checkpoint_sha256']}`",
            f"- CUDA peak allocated: **{training['cuda_peak_allocated_bytes']:,} bytes**",
            f"- Mean preprocessing latency: **{latency['preprocessing_ms_per_trial']:.3f} ms/trial**",
            f"- Mean model latency: **{latency['model_ms_per_trial']:.3f} ms/trial**",
            f"- Mean end-to-end latency proxy: **{latency['preprocessing_ms_per_trial'] + latency['model_ms_per_trial']:.3f} ms/trial**",
            f"- Training wall time: **{training['training_seconds'] / 60.0:.2f} min**",
            "",
            "## Deployment Ledger",
            "",
            f"Known retained IR/X3D, shared YOLO, this Thermal checkpoint, and the calibration/fusion reserve total **{ledger['total_serialized_bytes']:,} bytes**, leaving **{ledger['headroom_bytes']:,} bytes** below the strict 95,000,000-byte ceiling. Each asset identity is counted once. This is still a provisional known-asset ledger, not a complete six-modal package claim.",
            "",
            "## Decision Boundary",
            "",
            "This result is development evidence only. Architecture retention remains controlled by shared train-14 OOF, and no sealed heldout action is authorized by T1-B.",
            "",
            "## Per-Class Validation",
            "",
            "| Class | Support | Recall | F1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for class_id, (support, recall, f1) in enumerate(
        zip(metrics["per_class_support"], metrics["per_class_recall"], metrics["per_class_f1"])
    ):
        lines.append(f"| {class_id} | {support} | {recall:.4f} | {f1:.4f} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
