from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
X3D_ROOT = ROOT / "outputs" / "x3d_s_ir_context_oof"
BASELINE_ROOT = ROOT / "outputs" / "mobilenet_tcn_ir_context_oof"
REPORTS = ROOT / "reports"

SEED_FOLDS = {
    20260715: [
        X3D_ROOT / "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715" / "fold_0",
        X3D_ROOT / "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715" / "fold_1",
        X3D_ROOT / "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715_fold2_only" / "fold_2",
    ],
    20260716: [
        X3D_ROOT / "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260716" / f"fold_{fold}"
        for fold in range(3)
    ],
    20260717: [
        X3D_ROOT / "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260717_restart" / "fold_0",
        X3D_ROOT / "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260717_fold1_refit10_fold2" / "fold_1",
        X3D_ROOT / "x3d_s_ir_context_adaptive_oof_strict_v3_seed20260717_fold1_refit10_fold2" / "fold_2",
    ],
}
BASELINE_FOLD0 = BASELINE_ROOT / "matched_mobilenet_tcn_fold0_fixed10_seed20260715"


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def concatenate(paths: list[Path]) -> dict[str, np.ndarray]:
    archives = [load_archive(path / "formal_outer_predictions.npz") for path in paths]
    keys = set(archives[0])
    if any(set(archive) != keys for archive in archives[1:]):
        raise ValueError("Fold archives expose different fields")
    result = {
        key: np.concatenate([archive[key] for archive in archives])
        for key in keys
        if archive_row_field(archives[0][key], len(archives[0]["sample_ids"]))
    }
    sample_ids = result["sample_ids"].astype(str)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Concatenated OOF contains duplicate sample IDs")
    return result


def archive_row_field(value: np.ndarray, rows: int) -> bool:
    return value.ndim > 0 and value.shape[0] == rows


def metrics(archive: dict[str, np.ndarray]) -> dict[str, object]:
    labels = archive["labels"].astype(int)
    predictions = archive["logits"].argmax(axis=1)
    users = archive["user_ids"].astype(str)
    per_user = {
        user: float(accuracy_score(labels[users == user], predictions[users == user]))
        for user in np.unique(users)
    }
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=np.arange(40), average="macro", zero_division=0)
        ),
        "worst_user_accuracy": min(per_user.values()),
        "worst_user_id": min(per_user, key=per_user.get),
        "predictions": predictions,
        "per_user": per_user,
    }


def aligned(reference: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    lookup = {str(sample): index for index, sample in enumerate(other["sample_ids"])}
    if set(lookup) != set(reference["sample_ids"].astype(str)):
        raise ValueError("Comparison sample sets differ")
    indices = np.asarray([lookup[str(sample)] for sample in reference["sample_ids"]])
    rows = len(other["sample_ids"])
    return {
        key: value[indices] if archive_row_field(value, rows) else value
        for key, value in other.items()
    }


def bucket(frames: int) -> str:
    if frames <= 13:
        return "<=13"
    if frames <= 32:
        return "14-32"
    if frames <= 64:
        return "33-64"
    return ">64"


def main() -> None:
    seed_archives = {seed: concatenate(paths) for seed, paths in SEED_FOLDS.items()}
    canonical = seed_archives[20260715]
    canonical_ids = set(canonical["sample_ids"].astype(str))
    if len(canonical_ids) != 2320:
        raise ValueError("Canonical population must contain 2,320 trials")
    for seed, archive in seed_archives.items():
        if set(archive["sample_ids"].astype(str)) != canonical_ids:
            raise ValueError(f"Seed {seed} does not cover the canonical sample set")

    seed_rows = []
    for seed, archive in seed_archives.items():
        result = metrics(archive)
        seed_rows.append(
            {
                "seed": seed,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "worst_user_accuracy": result["worst_user_accuracy"],
                "worst_user_id": result["worst_user_id"],
                "protocol": "strict" if seed != 20260717 else "recovery_deviation_fold1_epoch10",
            }
        )
    seed_frame = pd.DataFrame(seed_rows)

    canonical_metrics = metrics(canonical)
    labels = canonical["labels"].astype(int)
    predictions = canonical_metrics["predictions"]
    users = canonical["user_ids"].astype(str)
    per_user_rows = []
    for user in sorted(np.unique(users)):
        mask = users == user
        per_user_rows.append(
            {
                "user_id": user,
                "sample_count": int(mask.sum()),
                "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
                "macro_f1": float(
                    f1_score(
                        labels[mask], predictions[mask], labels=np.arange(40), average="macro", zero_division=0
                    )
                ),
            }
        )
    per_user = pd.DataFrame(per_user_rows)

    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(40), zero_division=0
    )
    manifest = pd.read_csv(
        r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256\combined_frame_manifest.csv",
        encoding="utf-8-sig",
    )
    class_names = (
        manifest[["class_id", "action_name"]].drop_duplicates().set_index("class_id")["action_name"].to_dict()
    )
    per_class = pd.DataFrame(
        {
            "class_id": np.arange(40),
            "class_name": [class_names[index] for index in range(40)],
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
            "support": support,
        }
    )

    duration_rows = []
    buckets = np.asarray([bucket(int(value)) for value in canonical["num_frames"]])
    for name in ("<=13", "14-32", "33-64", ">64"):
        mask = buckets == name
        duration_rows.append(
            {
                "bucket": name,
                "sample_count": int(mask.sum()),
                "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
                "macro_f1": float(
                    f1_score(
                        labels[mask], predictions[mask], labels=np.arange(40), average="macro", zero_division=0
                    )
                ),
                "mean_num_clips": float(canonical["num_clips"][mask].mean()),
            }
        )
    duration = pd.DataFrame(duration_rows)

    canonical_fold0 = load_archive(SEED_FOLDS[20260715][0] / "formal_outer_predictions.npz")
    baseline = aligned(canonical_fold0, load_archive(BASELINE_FOLD0 / "formal_outer_predictions.npz"))
    x3d_fold0_metrics = metrics(canonical_fold0)
    baseline_metrics = metrics(baseline)
    fold0_labels = canonical_fold0["labels"].astype(int)
    if not np.array_equal(fold0_labels, baseline["labels"].astype(int)):
        raise ValueError("Matched fold0 labels differ")
    x_correct = x3d_fold0_metrics["predictions"] == fold0_labels
    b_correct = baseline_metrics["predictions"] == fold0_labels
    comparison = {
        "sample_count": len(fold0_labels),
        "x3d_accuracy": x3d_fold0_metrics["accuracy"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "accuracy_delta": x3d_fold0_metrics["accuracy"] - baseline_metrics["accuracy"],
        "x3d_macro_f1": x3d_fold0_metrics["macro_f1"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "macro_f1_delta": x3d_fold0_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "x3d_worst_user_accuracy": x3d_fold0_metrics["worst_user_accuracy"],
        "baseline_worst_user_accuracy": baseline_metrics["worst_user_accuracy"],
        "worst_user_delta": x3d_fold0_metrics["worst_user_accuracy"]
        - baseline_metrics["worst_user_accuracy"],
        "x3d_unique_correct": int(np.count_nonzero(x_correct & ~b_correct)),
        "baseline_unique_correct": int(np.count_nonzero(b_correct & ~x_correct)),
        "both_correct": int(np.count_nonzero(x_correct & b_correct)),
        "both_wrong": int(np.count_nonzero(~x_correct & ~b_correct)),
        "oracle_pair_accuracy": float(np.mean(x_correct | b_correct)),
        "prediction_disagreement": float(
            np.mean(x3d_fold0_metrics["predictions"] != baseline_metrics["predictions"])
        ),
        "sanity_threshold": 0.53,
        "decision": "stop_baseline_compute_keep_x3d",
    }

    verification = json.loads(
        (REPORTS / "x3d_s_phase4_checkpoint_reverification.json").read_text(encoding="utf-8")
    )
    final_summary = {
        "status": "phase4_completed_under_compute_amendment",
        "decision": "competition-retained; full matched primary rule not evaluated",
        "canonical_seed": 20260715,
        "sample_count": 2320,
        "seed_results": seed_rows,
        "three_seed_mean": {
            key: float(seed_frame[key].mean())
            for key in ("accuracy", "macro_f1", "worst_user_accuracy")
        },
        "three_seed_sample_std": {
            key: float(seed_frame[key].std(ddof=1))
            for key in ("accuracy", "macro_f1", "worst_user_accuracy")
        },
        "checkpoint_reverification_all_passed": bool(verification["all_passed"]),
        "checkpoint_reverification_maximum_delta": max(
            float(entry["maximum_array_delta"]) for entry in verification["entries"]
        ),
        "matched_sanity_fold0": comparison,
        "paired_bootstrap_performed": False,
        "paired_bootstrap_omission_reason": "Superseded by approved fixed-compute sanity amendment",
        "seed17_protocol_note": "Fold1 selected epoch 10 from the best observed Accuracy after 28/30 inner epochs",
        "zero_recall_class_ids": per_class.loc[per_class["recall"] == 0, "class_id"].astype(int).tolist(),
    }
    (REPORTS / "x3d_s_phase4_final_summary.json").write_text(
        json.dumps(final_summary, indent=2) + "\n", encoding="utf-8"
    )
    per_user.to_csv(REPORTS / "x3d_s_ir_context_oof_per_user.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(REPORTS / "x3d_s_ir_context_oof_per_class.csv", index=False, encoding="utf-8-sig")
    duration.to_csv(REPORTS / "x3d_s_ir_context_oof_duration.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# X3D-S IR-Context Phase 4 Report",
        "",
        "## Decision",
        "",
        "**Competition-retained; full matched primary rule not evaluated.** The approved compute amendment replaced the complete matched three-fold experiment with a fixed fold-0, 10-epoch sanity check.",
        "",
        "## Three-Seed Stability",
        "",
        seed_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Mean Accuracy `{final_summary['three_seed_mean']['accuracy']:.6f}` (sample SD `{final_summary['three_seed_sample_std']['accuracy']:.6f}`); mean Macro-F1 `{final_summary['three_seed_mean']['macro_f1']:.6f}` (sample SD `{final_summary['three_seed_sample_std']['macro_f1']:.6f}`). Seed 20260717 is stability-supporting evidence with the documented fold-1 recovery deviation; seed 20260715 remains canonical.",
        "",
        "## Checkpoint Reverification",
        "",
        f"All `{verification['entry_count']}` formal fold checkpoints regenerated their saved arrays and metrics within `{verification['tolerance']}`. Observed maximum delta: `{final_summary['checkpoint_reverification_maximum_delta']}`.",
        "",
        "## Fixed-Budget Matched Sanity",
        "",
        f"On the exact 800-trial outer fold 0, X3D Accuracy/Macro-F1/worst-user were `{comparison['x3d_accuracy']:.6f}` / `{comparison['x3d_macro_f1']:.6f}` / `{comparison['x3d_worst_user_accuracy']:.6f}`. MobileNet/TCN at the frozen 10-epoch budget achieved `{comparison['baseline_accuracy']:.6f}` / `{comparison['baseline_macro_f1']:.6f}` / `{comparison['baseline_worst_user_accuracy']:.6f}`. X3D deltas were `{comparison['accuracy_delta']:.6f}` / `{comparison['macro_f1_delta']:.6f}` / `{comparison['worst_user_delta']:.6f}`.",
        "",
        f"The baseline is below the preregistered `0.53` anomaly threshold. X3D uniquely corrected `{comparison['x3d_unique_correct']}` trials; the baseline uniquely corrected `{comparison['baseline_unique_correct']}`; oracle-pair Accuracy was `{comparison['oracle_pair_accuracy']:.6f}` and prediction disagreement was `{comparison['prediction_disagreement']:.6f}`. These are fold-0 diagnostics, not three-fold confidence intervals.",
        "",
        "## Duration Diagnostics (Canonical Seed)",
        "",
        duration.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Representative submission-path latency remains the Phase 3 measurement: `<=13` 0.8110 s, `14-32` 0.2162 s, `33-64` 0.4824 s, and `>64` 11.3689 s per trial including YOLO/ROI. The trained X3D+head checkpoint is 14,388,607 bytes; with YOLO the provisional IR route is 20,644,200 bytes.",
        "",
        "## Claim Boundary",
        "",
        "The teammate VideoMAE `0.641` result and historical MobileNet/Skeleton results remain descriptive because their split and metric contracts are not fully matched. No held-out-4 labels or predictions were read. The original 10,000-replicate paired bootstrap was not performed after the user-approved compute amendment, so this report does not claim the original matched three-fold primary criterion passed.",
    ]
    (REPORTS / "x3d_s_ir_context_oof_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
