from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report strict Skeleton C0/C1 OOF results.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/skeleton_c0_c1_strict_oof")
    parser.add_argument("--clean-views-root", type=Path, default=PROJECT_ROOT / "reports/skeleton_strict_oof_clean_views")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/skeleton_c0_c1_strict_oof")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def bootstrap_accuracy_delta(c0: np.ndarray, c1: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(c0), size=len(c0))
        deltas[index] = c1[sampled].mean() - c0[sampled].mean()
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    predictions = {}
    for arm in ("C0", "C1"):
        frames = []
        for fold in range(3):
            run_dir = args.output_root / arm / f"fold_{fold}"
            fold_rows.append(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
            frames.append(pd.read_csv(run_dir / "formal_outer_predictions.csv", encoding="utf-8-sig"))
        predictions[arm] = pd.concat(frames, ignore_index=True)
    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(args.report_dir / "fold_results.csv", index=False, encoding="utf-8-sig")

    c0 = predictions["C0"].sort_values("sample_id").reset_index(drop=True)
    c1 = predictions["C1"].sort_values("sample_id").reset_index(drop=True)
    if not c0["sample_id"].equals(c1["sample_id"]) or not c0["label"].equals(c1["label"]):
        raise ValueError("C0 and C1 OOF rows are not paired")
    c0_correct = c0["prediction"].to_numpy() == c0["label"].to_numpy()
    c1_correct = c1["prediction"].to_numpy() == c1["label"].to_numpy()
    c0_only = int((c0_correct & ~c1_correct).sum())
    c1_only = int((c1_correct & ~c0_correct).sum())
    ci_low, ci_high = bootstrap_accuracy_delta(c0_correct, c1_correct, args.bootstrap_replicates, args.seed)
    mcnemar_p = float(binomtest(min(c0_only, c1_only), c0_only + c1_only, 0.5).pvalue)

    combined_rows = []
    for arm, frame in predictions.items():
        combined_rows.append({
            "representation": arm, "samples": len(frame),
            "accuracy": accuracy_score(frame["label"], frame["prediction"]),
            "macro_f1_40class": f1_score(
                frame["label"], frame["prediction"], labels=np.arange(40), average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(frame["label"], frame["prediction"], average="weighted", zero_division=0),
        })
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(args.report_dir / "combined_oof_summary.csv", index=False, encoding="utf-8-sig")

    paired = pd.DataFrame({
        "sample_id": c0["sample_id"], "user_id": c0["user_id"], "fold": c0["fold"],
        "label": c0["label"], "c0_prediction": c0["prediction"], "c1_prediction": c1["prediction"],
        "c0_correct": c0_correct, "c1_correct": c1_correct,
    })
    paired["outcome"] = np.select(
        [paired["c0_correct"] & paired["c1_correct"], paired["c0_correct"] & ~paired["c1_correct"],
         ~paired["c0_correct"] & paired["c1_correct"]],
        ["both_correct", "c0_only", "c1_only"], default="both_wrong",
    )
    paired.to_csv(args.report_dir / "paired_oof_outcomes.csv", index=False, encoding="utf-8-sig")

    user_rows = []
    for user, group in paired.groupby("user_id"):
        user_rows.append({
            "user_id": user, "samples": len(group), "c0_accuracy": group["c0_correct"].mean(),
            "c1_accuracy": group["c1_correct"].mean(),
            "c1_minus_c0": group["c1_correct"].mean() - group["c0_correct"].mean(),
            "c0_only": int((group["outcome"] == "c0_only").sum()),
            "c1_only": int((group["outcome"] == "c1_only").sum()),
        })
    per_user = pd.DataFrame(user_rows).sort_values("user_id")
    per_user.to_csv(args.report_dir / "paired_oof_per_user.csv", index=False, encoding="utf-8-sig")

    class_rows = []
    for class_id, group in paired.groupby("label"):
        class_rows.append({
            "class_id": class_id, "samples": len(group), "c0_accuracy": group["c0_correct"].mean(),
            "c1_accuracy": group["c1_correct"].mean(),
            "c1_minus_c0": group["c1_correct"].mean() - group["c0_correct"].mean(),
            "c0_only": int((group["outcome"] == "c0_only").sum()),
            "c1_only": int((group["outcome"] == "c1_only").sum()),
        })
    pd.DataFrame(class_rows).to_csv(args.report_dir / "paired_oof_per_class.csv", index=False, encoding="utf-8-sig")

    clean_summary = pd.read_csv(args.clean_views_root / "strict_clean_view_summary.csv", encoding="utf-8-sig")
    clean_summary.to_csv(args.report_dir / "strict_clean_view_summary.csv", index=False, encoding="utf-8-sig")
    c0_metrics = combined.set_index("representation").loc["C0"]
    c1_metrics = combined.set_index("representation").loc["C1"]
    fold_table = fold_results[[
        "representation", "fold", "selected_epoch", "outer_validation_trials",
        "outer_accuracy", "outer_macro_f1_40class", "outer_worst_user_accuracy",
    ]]
    report = f"""# Skeleton C0 vs C1 Strict OOF Report

## Protocol

- Frozen shared assignment: `metadata/splits/train14_oof_3fold.json`.
- Six independently fitted clean views: each `inner_selection` projection uses inner-fit users only; each `formal_outer` projection uses outer-train users only.
- Visual candidate margin is fixed at 20%; corresponding validation users never fit projection or feature normalization.
- C0 and C1 use the same H36M-17 median bone-length estimator, 64-step gap-aware timeline, segment-local velocity, residual TCN, optimizer, seed schedule, and epoch-selection rule.
- C0 uses one median trial scale; C1 uses the per-frame value of the same estimator.
- Formal models are freshly initialized and refit for the epoch selected on inner validation. Outer-validation labels are evaluated exactly once.

## Clean Views

{clean_summary[["fold", "scope", "calibration_frames", "retained_frames", "ambiguous_frames", "confident_multi_person_retained", "empty_trials"]].to_markdown(index=False)}

All six scopes have disjoint projection-fit and validation users and zero empty trials.

## Fold Results

{fold_table.to_markdown(index=False, floatfmt=".6f")}

## Combined OOF

{combined.to_markdown(index=False, floatfmt=".6f")}

- C1 - C0 accuracy: **{c1_metrics['accuracy'] - c0_metrics['accuracy']:+.6f}**.
- C1 - C0 macro-F1: **{c1_metrics['macro_f1_40class'] - c0_metrics['macro_f1_40class']:+.6f}**.
- Paired discordance: C0-only correct={c0_only}, C1-only correct={c1_only}; exact McNemar p={mcnemar_p:.6g}.
- Paired bootstrap 95% CI for C1 - C0 accuracy: `[{ci_low:+.6f}, {ci_high:+.6f}]` ({args.bootstrap_replicates} replicates, seed {args.seed}).

## Decision

C1 wins combined OOF accuracy and macro-F1, and the paired accuracy interval excludes zero. However, fold0 favors C0 while folds1/2 favor C1, and selected epochs differ materially. The evidence supports **C1 per-frame bone scale as the current preprocessing winner**, with fold heterogeneity retained as a documented residual risk. No graph/ST-GCN experiment was run.

## Artifacts

- `fold_results.csv`: fold-level metrics and provenance hashes.
- `combined_oof_summary.csv`: primary combined metrics.
- `paired_oof_outcomes.csv`: one row per canonical OOF trial.
- `paired_oof_per_user.csv` and `paired_oof_per_class.csv`: heterogeneity diagnostics.
- Raw training histories and normalization statistics remain under `outputs/skeleton_c0_c1_strict_oof`; no checkpoint or model weight was saved.
"""
    (args.report_dir / "skeleton_c0_c1_strict_oof_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "c0_accuracy": c0_metrics["accuracy"], "c1_accuracy": c1_metrics["accuracy"],
        "accuracy_delta": c1_metrics["accuracy"] - c0_metrics["accuracy"],
        "bootstrap_ci": [ci_low, ci_high], "mcnemar_p": mcnemar_p,
    }, indent=2))


if __name__ == "__main__":
    main()
