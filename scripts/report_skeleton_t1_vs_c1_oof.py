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
    parser = argparse.ArgumentParser(description="Compare frozen T1-v1 with strict C1-TCN OOF predictions.")
    parser.add_argument(
        "--t1-root", type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_segment_aware_tcn_strict_oof/T1_segment_aware_c1_tcn",
    )
    parser.add_argument(
        "--c1-root", type=Path, default=PROJECT_ROOT / "outputs/skeleton_c0_c1_strict_oof/C1",
    )
    parser.add_argument(
        "--views-root", type=Path, default=PROJECT_ROOT / "reports/skeleton_strict_oof_clean_views",
    )
    parser.add_argument(
        "--fold-assignment", type=Path,
        default=PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "reports/skeleton_segment_aware_tcn_strict_oof",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def load_arm(root: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, summaries = [], []
    for fold in range(3):
        run_dir = root / f"fold_{fold}"
        frame = pd.read_csv(run_dir / "formal_outer_predictions.csv", encoding="utf-8-sig")
        frame["model"] = name
        predictions.append(frame)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary["model"] = name
        summaries.append(summary)
    combined = pd.concat(predictions, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate OOF sample IDs for {name}")
    return combined, pd.DataFrame(summaries)


def validate_ownership(frame: pd.DataFrame, assignment: dict) -> None:
    for fold in assignment["folds"]:
        observed = set(frame.loc[frame["fold"] == int(fold["fold"]), "user_id"])
        if observed != set(fold["validation_user_ids"]):
            raise ValueError(f"Fold {fold['fold']} OOF ownership mismatch")


def metrics(frame: pd.DataFrame, model: str) -> dict[str, object]:
    return {
        "model": model, "samples": len(frame),
        "accuracy": accuracy_score(frame["label"], frame["prediction"]),
        "macro_f1_40class": f1_score(
            frame["label"], frame["prediction"], labels=np.arange(40), average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(frame["label"], frame["prediction"], average="weighted", zero_division=0),
    }


def bootstrap_delta(
    baseline: np.ndarray, candidate: np.ndarray, replicates: int, seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(baseline), size=len(baseline))
        deltas[index] = candidate[sampled].mean() - baseline[sampled].mean()
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def segment_counts(views_root: Path, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(3):
        view = pd.read_csv(
            views_root / f"fold_{fold}/formal_outer/clean_view.csv", encoding="utf-8-sig",
            usecols=["sample_id", "retained_segment_index", "use_for_frame_training"],
        )
        view = view[view["use_for_frame_training"].astype(bool)]
        expected = set(paired.loc[paired["fold"] == fold, "sample_id"])
        counts = view[view["sample_id"].isin(expected)].groupby("sample_id")["retained_segment_index"].nunique()
        if set(counts.index) != expected:
            raise ValueError(f"Missing segment provenance for fold {fold}")
        rows.extend({"sample_id": sample_id, "retained_segment_count": int(count)} for sample_id, count in counts.items())
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    assignment = json.loads(args.fold_assignment.read_text(encoding="utf-8"))
    c1, c1_folds = load_arm(args.c1_root, "C1-TCN")
    t1, t1_folds = load_arm(args.t1_root, "T1-v1-SegmentAware-TCN")
    validate_ownership(c1, assignment)
    validate_ownership(t1, assignment)
    c1 = c1.sort_values("sample_id").reset_index(drop=True)
    t1 = t1.sort_values("sample_id").reset_index(drop=True)
    identity = ["sample_id", "user_id", "label"]
    if not c1[identity].equals(t1[identity]):
        raise ValueError("C1-TCN and T1-v1 OOF rows are not exactly paired")
    if len(c1) != 2341 or c1["user_id"].nunique() != 14 or c1["label"].nunique() != 40:
        raise ValueError("Unexpected paired OOF sample space")

    c1_correct = c1["prediction"].to_numpy() == c1["label"].to_numpy()
    t1_correct = t1["prediction"].to_numpy() == t1["label"].to_numpy()
    c1_only = int((c1_correct & ~t1_correct).sum())
    t1_only = int((~c1_correct & t1_correct).sum())
    mcnemar_p = float(binomtest(min(c1_only, t1_only), c1_only + t1_only, 0.5).pvalue)
    ci_low, ci_high = bootstrap_delta(c1_correct, t1_correct, args.bootstrap_replicates, args.seed)
    combined = pd.DataFrame([metrics(c1, "C1-TCN"), metrics(t1, "T1-v1-SegmentAware-TCN")])
    combined.to_csv(args.report_dir / "t1_vs_c1_combined_oof.csv", index=False, encoding="utf-8-sig")
    fold_results = pd.concat([
        c1_folds[["model", "fold", "selected_epoch", "outer_validation_trials", "outer_accuracy", "outer_macro_f1_40class", "outer_weighted_f1"]],
        t1_folds[["model", "fold", "selected_epoch", "outer_validation_trials", "outer_accuracy", "outer_macro_f1_40class", "outer_weighted_f1"]],
    ]).sort_values(["fold", "model"])
    fold_results.to_csv(args.report_dir / "t1_vs_c1_fold_results.csv", index=False, encoding="utf-8-sig")

    paired = pd.DataFrame({
        "sample_id": c1["sample_id"], "user_id": c1["user_id"], "fold": c1["fold"],
        "label": c1["label"], "c1_prediction": c1["prediction"], "t1_prediction": t1["prediction"],
        "c1_correct": c1_correct, "t1_correct": t1_correct,
    })
    paired = paired.merge(segment_counts(args.views_root, paired), on="sample_id", validate="one_to_one")
    paired["segment_group"] = np.where(
        paired["retained_segment_count"] > 1, "multiple_segments", "single_segment"
    )
    paired["outcome"] = np.select(
        [paired["c1_correct"] & paired["t1_correct"], paired["c1_correct"] & ~paired["t1_correct"],
         ~paired["c1_correct"] & paired["t1_correct"]],
        ["both_correct", "c1_only", "t1_only"], default="both_wrong",
    )
    paired.to_csv(args.report_dir / "t1_vs_c1_paired_oof_outcomes.csv", index=False, encoding="utf-8-sig")
    improved_counts = {}
    for name, column in (("user", "user_id"), ("class", "label")):
        rows = []
        for value, group in paired.groupby(column):
            rows.append({
                column: value, "samples": len(group), "c1_accuracy": group["c1_correct"].mean(),
                "t1_accuracy": group["t1_correct"].mean(),
                "t1_minus_c1": group["t1_correct"].mean() - group["c1_correct"].mean(),
                "c1_only": int((group["outcome"] == "c1_only").sum()),
                "t1_only": int((group["outcome"] == "t1_only").sum()),
            })
        result = pd.DataFrame(rows).sort_values(column)
        result.to_csv(args.report_dir / f"t1_vs_c1_per_{name}.csv", index=False, encoding="utf-8-sig")
        improved_counts[name] = int((result["t1_minus_c1"] > 0).sum())

    segment_rows = []
    for group_name, group in paired.groupby("segment_group"):
        baseline = group["c1_correct"].to_numpy()
        candidate = group["t1_correct"].to_numpy()
        low, high = bootstrap_delta(baseline, candidate, args.bootstrap_replicates, args.seed)
        segment_rows.append({
            "segment_group": group_name, "samples": len(group), "c1_accuracy": baseline.mean(),
            "t1_accuracy": candidate.mean(), "t1_minus_c1": candidate.mean() - baseline.mean(),
            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "c1_only": int((baseline & ~candidate).sum()), "t1_only": int((~baseline & candidate).sum()),
        })
    segment_analysis = pd.DataFrame(segment_rows)
    segment_analysis.to_csv(args.report_dir / "t1_vs_c1_by_segment_count.csv", index=False, encoding="utf-8-sig")

    c1_metrics = combined.set_index("model").loc["C1-TCN"]
    t1_metrics = combined.set_index("model").loc["T1-v1-SegmentAware-TCN"]
    accuracy_delta = t1_metrics["accuracy"] - c1_metrics["accuracy"]
    macro_delta = t1_metrics["macro_f1_40class"] - c1_metrics["macro_f1_40class"]
    report = f"""# T1-v1 Segment-Aware C1-TCN Strict Cross-User OOF Development Evidence

## Frozen Contract

- Model commit: `77b4455a81ff874a1bbd28245f600e32aae2a204`.
- C1 per-frame preprocessing, joint `xyz + velocity` 102D input, T=64, and strict nested user folds are unchanged.
- Original channels `[64,128]`, kernels `5/3`, dilation schedule, residual, BatchNorm, GELU, dropout, optimizer, and seed are unchanged.
- Only temporal edges change: source and destination must both be valid and share the same retained segment ID.
- Parameter count remains 172,776. Single-segment fully valid inference is numerically equivalent after copying C1 weights and BatchNorm state.
- All formal outer folds ran consecutively after freeze; no outer fold was used as a gate.

## Fold Results

{fold_results.to_markdown(index=False, floatfmt=".6f")}

## Combined OOF

{combined.to_markdown(index=False, floatfmt=".6f")}

- T1-v1 - C1 accuracy: **{accuracy_delta:+.6f}**.
- T1-v1 - C1 macro-F1: **{macro_delta:+.6f}**.
- Paired discordance: C1-only correct={c1_only}, T1-only correct={t1_only}; exact McNemar p={mcnemar_p:.6g}.
- Paired bootstrap 95% CI for T1-v1 - C1 accuracy: `[{ci_low:+.6f}, {ci_high:+.6f}]` ({args.bootstrap_replicates} replicates, seed {args.seed}).
- T1 improves accuracy for {improved_counts['user']}/14 users and {improved_counts['class']}/40 classes.

## Segment-Stratified Evidence

{segment_analysis.to_markdown(index=False, floatfmt=".6f")}

This stratification is diagnostic, not a separately pre-registered success test. Only 88 trials contain multiple retained segments, and their point estimate favors C1. The aggregate gain comes from single-segment trials, where T1 and C1 have the same temporal-edge semantics and are numerically equivalent at inference after state copy. Their independently trained outcomes can still diverge through floating-point operation order and epoch-selection trajectories. Therefore the aggregate point estimate cannot be attributed to blocking cross-segment edges.

## Pre-Registered Decision

**T1-v1 does not pass the replacement criterion.** Accuracy and Macro-F1 point estimates improve, but the paired accuracy interval includes zero, McNemar is not significant at 0.05, only fold0 improves, and only 5/14 users improve. More importantly, the multiple-segment subgroup does not improve. The evidence does not support cross-gap temporal mixing as a major C1 bottleneck. C1-TCN remains the Skeleton expert; T1 is retained as development evidence only. Under the frozen route, the next temporal question may evaluate resolution, but it must be registered as a new experiment rather than tuned from these outer-fold outcomes.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `{t1_folds.iloc[0]['oof_assignment_sha256']}`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- Raw histories and predictions remain under `outputs/skeleton_segment_aware_tcn_strict_oof`; no checkpoint or model weight was saved.
"""
    (args.report_dir / "skeleton_segment_aware_tcn_strict_oof_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "c1_accuracy": c1_metrics["accuracy"], "t1_accuracy": t1_metrics["accuracy"],
        "accuracy_delta": accuracy_delta, "macro_f1_delta": macro_delta,
        "c1_only": c1_only, "t1_only": t1_only, "mcnemar_p": mcnemar_p,
        "bootstrap_ci": [ci_low, ci_high], "users_improved": improved_counts["user"],
        "classes_improved": improved_counts["class"],
    }, indent=2))


if __name__ == "__main__":
    main()
