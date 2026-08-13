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
    parser = argparse.ArgumentParser(description="Compare frozen T2-v1 T=96 with strict C1-TCN OOF predictions.")
    parser.add_argument(
        "--t2-root", type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_t96_tcn_strict_oof/T2_c1_tcn_96",
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
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports/skeleton_t96_tcn_strict_oof",
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
        "model": model,
        "samples": len(frame),
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


def retained_lengths(views_root: Path, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(3):
        view = pd.read_csv(
            views_root / f"fold_{fold}/formal_outer/clean_view.csv", encoding="utf-8-sig",
            usecols=["sample_id", "use_for_frame_training"],
        )
        view = view[view["use_for_frame_training"].astype(bool)]
        expected = set(paired.loc[paired["fold"] == fold, "sample_id"])
        counts = view[view["sample_id"].isin(expected)].groupby("sample_id").size()
        if set(counts.index) != expected:
            raise ValueError(f"Missing retained-length provenance for fold {fold}")
        rows.extend({"sample_id": sample_id, "retained_length": int(count)} for sample_id, count in counts.items())
    return pd.DataFrame(rows)


def length_bucket(lengths: pd.Series) -> pd.Categorical:
    return pd.cut(
        lengths,
        bins=[0, 15, 31, 63, np.inf],
        labels=["<=15", "16-31", "32-63", ">=64"],
        include_lowest=True,
        ordered=True,
    )


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    assignment = json.loads(args.fold_assignment.read_text(encoding="utf-8"))
    c1, c1_folds = load_arm(args.c1_root, "C1-TCN-T64")
    t2, t2_folds = load_arm(args.t2_root, "T2-v1-T96")
    validate_ownership(c1, assignment)
    validate_ownership(t2, assignment)
    c1 = c1.sort_values("sample_id").reset_index(drop=True)
    t2 = t2.sort_values("sample_id").reset_index(drop=True)
    identity = ["sample_id", "user_id", "label"]
    if not c1[identity].equals(t2[identity]):
        raise ValueError("C1-TCN and T2-v1 OOF rows are not exactly paired")
    if len(c1) != 2341 or c1["user_id"].nunique() != 14 or c1["label"].nunique() != 40:
        raise ValueError("Unexpected paired OOF sample space")

    c1_correct = c1["prediction"].to_numpy() == c1["label"].to_numpy()
    t2_correct = t2["prediction"].to_numpy() == t2["label"].to_numpy()
    c1_only = int((c1_correct & ~t2_correct).sum())
    t2_only = int((~c1_correct & t2_correct).sum())
    mcnemar_p = float(binomtest(min(c1_only, t2_only), c1_only + t2_only, 0.5).pvalue)
    ci_low, ci_high = bootstrap_delta(c1_correct, t2_correct, args.bootstrap_replicates, args.seed)
    combined = pd.DataFrame([metrics(c1, "C1-TCN-T64"), metrics(t2, "T2-v1-T96")])
    combined.to_csv(args.report_dir / "t2_vs_c1_combined_oof.csv", index=False, encoding="utf-8-sig")
    fold_results = pd.concat([
        c1_folds[["model", "fold", "selected_epoch", "outer_validation_trials", "outer_accuracy", "outer_macro_f1_40class", "outer_weighted_f1"]],
        t2_folds[["model", "fold", "selected_epoch", "outer_validation_trials", "outer_accuracy", "outer_macro_f1_40class", "outer_weighted_f1"]],
    ]).sort_values(["fold", "model"])
    fold_results.to_csv(args.report_dir / "t2_vs_c1_fold_results.csv", index=False, encoding="utf-8-sig")

    paired = pd.DataFrame({
        "sample_id": c1["sample_id"],
        "user_id": c1["user_id"],
        "fold": c1["fold"],
        "label": c1["label"],
        "c1_prediction": c1["prediction"],
        "t2_prediction": t2["prediction"],
        "c1_correct": c1_correct,
        "t2_correct": t2_correct,
    })
    paired = paired.merge(retained_lengths(args.views_root, paired), on="sample_id", validate="one_to_one")
    paired["length_bucket"] = length_bucket(paired["retained_length"])
    if paired["length_bucket"].isna().any():
        raise ValueError("Every paired trial must have a pre-registered retained-length bucket")
    paired["outcome"] = np.select(
        [
            paired["c1_correct"] & paired["t2_correct"],
            paired["c1_correct"] & ~paired["t2_correct"],
            ~paired["c1_correct"] & paired["t2_correct"],
        ],
        ["both_correct", "c1_only", "t2_only"],
        default="both_wrong",
    )
    paired.to_csv(args.report_dir / "t2_vs_c1_paired_oof_outcomes.csv", index=False, encoding="utf-8-sig")

    improved_counts = {}
    for name, column in (("user", "user_id"), ("class", "label")):
        rows = []
        for value, group in paired.groupby(column):
            rows.append({
                column: value,
                "samples": len(group),
                "c1_accuracy": group["c1_correct"].mean(),
                "t2_accuracy": group["t2_correct"].mean(),
                "t2_minus_c1": group["t2_correct"].mean() - group["c1_correct"].mean(),
                "c1_only": int((group["outcome"] == "c1_only").sum()),
                "t2_only": int((group["outcome"] == "t2_only").sum()),
            })
        result = pd.DataFrame(rows).sort_values(column)
        result.to_csv(args.report_dir / f"t2_vs_c1_per_{name}.csv", index=False, encoding="utf-8-sig")
        improved_counts[name] = int((result["t2_minus_c1"] > 0).sum())

    length_rows = []
    for bucket, group in paired.groupby("length_bucket", observed=False):
        if group.empty:
            continue
        baseline = group["c1_correct"].to_numpy()
        candidate = group["t2_correct"].to_numpy()
        low, high = bootstrap_delta(baseline, candidate, args.bootstrap_replicates, args.seed)
        length_rows.append({
            "length_bucket": str(bucket),
            "samples": len(group),
            "c1_accuracy": baseline.mean(),
            "t2_accuracy": candidate.mean(),
            "t2_minus_c1": candidate.mean() - baseline.mean(),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "c1_only": int((baseline & ~candidate).sum()),
            "t2_only": int((~baseline & candidate).sum()),
        })
    length_analysis = pd.DataFrame(length_rows)
    length_analysis.to_csv(args.report_dir / "t2_vs_c1_by_retained_length.csv", index=False, encoding="utf-8-sig")

    c1_metrics = combined.set_index("model").loc["C1-TCN-T64"]
    t2_metrics = combined.set_index("model").loc["T2-v1-T96"]
    accuracy_delta = t2_metrics["accuracy"] - c1_metrics["accuracy"]
    macro_delta = t2_metrics["macro_f1_40class"] - c1_metrics["macro_f1_40class"]
    fold_deltas = (
        fold_results.pivot(index="fold", columns="model", values="outer_accuracy")["T2-v1-T96"]
        - fold_results.pivot(index="fold", columns="model", values="outer_accuracy")["C1-TCN-T64"]
    )
    fold_net_correct = {
        int(fold): int(group["t2_correct"].sum() - group["c1_correct"].sum())
        for fold, group in paired.groupby("fold")
    }
    headline_support = accuracy_delta > 0 and macro_delta >= 0 and ci_low > 0
    stability_support = bool((fold_deltas > 0).sum() >= 2 and improved_counts["user"] > 7)
    passes = headline_support and stability_support
    decision = (
        "T2-v1 passes the replacement criterion; T=96 replaces C1-TCN-T64 as the Skeleton expert."
        if passes else
        "T2-v1 does not pass the replacement criterion and does not replace C1-TCN-T64. C1-TCN remains the frozen Skeleton expert, and the planned Skeleton single-modality architecture search ends here."
    )
    report = f"""# T2-v1 T=96 C1-TCN Strict Cross-User OOF Development Evidence

## Frozen Contract

- Model commit: `4e731b438fe3b50e0f799771c0324ce0b3694d25`.
- C1 per-frame preprocessing, joint `xyz + velocity` 102D input, segment-local velocity, and strict nested user folds are unchanged.
- Gap-aware sequence length is the only experimental variable: T=64 becomes T=96.
- The original `TemporalClassifier`, channels `[64,128]`, kernels `5/3`, dilation schedule, residual, BatchNorm, GELU, dropout, optimizer, scheduler, seed, and pooling remain unchanged.
- Parameter count remains 172,776. All formal outer folds ran consecutively after freeze; no outer fold was used as a gate.

## Fold Results

{fold_results.to_markdown(index=False, floatfmt=".6f")}

## Combined OOF

{combined.to_markdown(index=False, floatfmt=".6f")}

- T2-v1 - C1 accuracy: **{accuracy_delta:+.6f}**.
- T2-v1 - C1 macro-F1: **{macro_delta:+.6f}**.
- Paired discordance: C1-only correct={c1_only}, T2-only correct={t2_only}; exact McNemar p={mcnemar_p:.6g}.
- Paired bootstrap 95% CI for T2-v1 - C1 accuracy: `[{ci_low:+.6f}, {ci_high:+.6f}]` ({args.bootstrap_replicates} replicates, seed {args.seed}).
- T2 improves accuracy for {improved_counts['user']}/14 users and {improved_counts['class']}/40 classes.
- Per-fold T2 - C1 accuracy deltas: {', '.join(f'fold{int(fold)}={value:+.6f}' for fold, value in fold_deltas.items())}.
- Per-fold net additional correct trials: {', '.join(f'fold{fold}={value:+d}' for fold, value in fold_net_correct.items())}.

## Retained-Length Diagnosis

{length_analysis.to_markdown(index=False, floatfmt=".6f")}

This retained original sequence-length stratification is post-hoc diagnosis only, not a separate success criterion. Length is the number of retained, frame-training-usable rows in each fold-specific formal outer clean view before resampling. A credible resolution effect would be expected to concentrate in longer trials; random positive and negative bucket fluctuations do not establish that denser interpolation adds information.

## Pre-Registered Decision

**{decision}** The headline metrics are positive: Accuracy is +1.32pp, Macro-F1 is +2.03pp, the paired bootstrap interval narrowly excludes zero, and exact McNemar p=0.0386. However, the required stability condition fails. Fold0 contributes +39 net correct trials while the combined net gain is only +31; fold1 loses 16, fold2 gains 8, and only 5/14 users improve. The retained-length pattern also does not support the proposed mechanism: the clearest gain is in the shortest bucket, while `32-63` is slightly negative and `>=64` is inconclusive. This is positive development evidence, but it is too fold- and user-concentrated to replace C1 under the pre-registered rule.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `{t2_folds.iloc[0]['oof_assignment_sha256']}`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- T2 fold summaries report `sequence_length=96`, `model_class=TemporalClassifier`, and validation-excluding preprocessing provenance.
- Raw histories and predictions remain under `outputs/skeleton_t96_tcn_strict_oof`; no checkpoint or model weight was saved.
"""
    (args.report_dir / "skeleton_t96_tcn_strict_oof_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "c1_accuracy": c1_metrics["accuracy"],
        "t2_accuracy": t2_metrics["accuracy"],
        "accuracy_delta": accuracy_delta,
        "macro_f1_delta": macro_delta,
        "c1_only": c1_only,
        "t2_only": t2_only,
        "mcnemar_p": mcnemar_p,
        "bootstrap_ci": [ci_low, ci_high],
        "fold_deltas": {str(index): value for index, value in fold_deltas.items()},
        "fold_net_correct": {str(index): value for index, value in fold_net_correct.items()},
        "users_improved": improved_counts["user"],
        "classes_improved": improved_counts["class"],
        "headline_support": headline_support,
        "stability_support": stability_support,
        "passes_replacement": passes,
    }, indent=2))


if __name__ == "__main__":
    main()
