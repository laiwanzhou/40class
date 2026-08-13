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
    parser = argparse.ArgumentParser(description="Compare frozen D2-v1 with strict C1-TCN OOF predictions.")
    parser.add_argument(
        "--d2-root", type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_joint_bone_tcn_strict_oof/D2_joint_bone_tcn",
    )
    parser.add_argument(
        "--c1-root", type=Path, default=PROJECT_ROOT / "outputs/skeleton_c0_c1_strict_oof/C1",
    )
    parser.add_argument(
        "--fold-assignment", type=Path,
        default=PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "reports/skeleton_joint_bone_tcn_strict_oof",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def load_arm(root: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    summaries = []
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
        fold_index = int(fold["fold"])
        observed = set(frame.loc[frame["fold"] == fold_index, "user_id"])
        if observed != set(fold["validation_user_ids"]):
            raise ValueError(f"Fold {fold_index} OOF ownership mismatch")


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


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    assignment = json.loads(args.fold_assignment.read_text(encoding="utf-8"))
    c1, c1_folds = load_arm(args.c1_root, "C1-TCN")
    d2, d2_folds = load_arm(args.d2_root, "D2-v1-JointBone-TCN")
    validate_ownership(c1, assignment)
    validate_ownership(d2, assignment)
    c1 = c1.sort_values("sample_id").reset_index(drop=True)
    d2 = d2.sort_values("sample_id").reset_index(drop=True)
    identity_columns = ["sample_id", "user_id", "label"]
    if not c1[identity_columns].equals(d2[identity_columns]):
        raise ValueError("C1-TCN and D2-v1 OOF rows are not exactly paired")
    if len(c1) != 2341 or c1["user_id"].nunique() != 14 or c1["label"].nunique() != 40:
        raise ValueError("Unexpected paired OOF sample space")

    c1_correct = c1["prediction"].to_numpy() == c1["label"].to_numpy()
    d2_correct = d2["prediction"].to_numpy() == d2["label"].to_numpy()
    c1_only = int((c1_correct & ~d2_correct).sum())
    d2_only = int((~c1_correct & d2_correct).sum())
    mcnemar_p = float(binomtest(min(c1_only, d2_only), c1_only + d2_only, 0.5).pvalue)
    ci_low, ci_high = bootstrap_delta(c1_correct, d2_correct, args.bootstrap_replicates, args.seed)
    combined = pd.DataFrame([metrics(c1, "C1-TCN"), metrics(d2, "D2-v1-JointBone-TCN")])
    combined.to_csv(args.report_dir / "d2_vs_c1_combined_oof.csv", index=False, encoding="utf-8-sig")

    fold_results = pd.concat([
        c1_folds[["model", "fold", "selected_epoch", "outer_validation_trials", "outer_accuracy", "outer_macro_f1_40class", "outer_weighted_f1"]],
        d2_folds[["model", "fold", "selected_epoch", "outer_validation_trials", "outer_accuracy", "outer_macro_f1_40class", "outer_weighted_f1"]],
    ]).sort_values(["fold", "model"])
    fold_results.to_csv(args.report_dir / "d2_vs_c1_fold_results.csv", index=False, encoding="utf-8-sig")

    paired = pd.DataFrame({
        "sample_id": c1["sample_id"], "user_id": c1["user_id"], "fold": c1["fold"],
        "label": c1["label"], "c1_prediction": c1["prediction"], "d2_prediction": d2["prediction"],
        "c1_correct": c1_correct, "d2_correct": d2_correct,
    })
    paired["outcome"] = np.select(
        [paired["c1_correct"] & paired["d2_correct"], paired["c1_correct"] & ~paired["d2_correct"],
         ~paired["c1_correct"] & paired["d2_correct"]],
        ["both_correct", "c1_only", "d2_only"], default="both_wrong",
    )
    paired.to_csv(args.report_dir / "d2_vs_c1_paired_oof_outcomes.csv", index=False, encoding="utf-8-sig")
    improved_counts = {}
    for name, column in (("user", "user_id"), ("class", "label")):
        rows = []
        for value, group in paired.groupby(column):
            rows.append({
                column: value, "samples": len(group), "c1_accuracy": group["c1_correct"].mean(),
                "d2_accuracy": group["d2_correct"].mean(),
                "d2_minus_c1": group["d2_correct"].mean() - group["c1_correct"].mean(),
                "c1_only": int((group["outcome"] == "c1_only").sum()),
                "d2_only": int((group["outcome"] == "d2_only").sum()),
            })
        result = pd.DataFrame(rows).sort_values(column)
        result.to_csv(args.report_dir / f"d2_vs_c1_per_{name}.csv", index=False, encoding="utf-8-sig")
        improved_counts[name] = int((result["d2_minus_c1"] > 0).sum())

    curves = []
    for fold in range(3):
        run_dir = args.d2_root / f"fold_{fold}"
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        inner = pd.read_csv(run_dir / "inner_selection_history.csv", encoding="utf-8-sig")
        formal = pd.read_csv(run_dir / "formal_refit_history.csv", encoding="utf-8-sig")
        curves.append({
            "fold": fold, "selected_epoch": summary["selected_epoch"],
            "inner_max_train_accuracy": inner["train_accuracy"].max(),
            "inner_best_validation_accuracy": inner["validation_accuracy"].max(),
            "formal_max_train_accuracy": formal["train_accuracy"].max(),
            "outer_accuracy": summary["outer_accuracy"],
        })
    curve_summary = pd.DataFrame(curves)
    curve_summary.to_csv(args.report_dir / "d2_training_curve_summary.csv", index=False, encoding="utf-8-sig")

    c1_metrics = combined.set_index("model").loc["C1-TCN"]
    d2_metrics = combined.set_index("model").loc["D2-v1-JointBone-TCN"]
    accuracy_delta = d2_metrics["accuracy"] - c1_metrics["accuracy"]
    macro_delta = d2_metrics["macro_f1_40class"] - c1_metrics["macro_f1_40class"]
    report = f"""# D2-v1 Joint/Bone TCN Strict Cross-User OOF Development Evidence

## Frozen Contract

- Model commit: `90c9b30518f60b3900cbca1691858d424e4f981d`.
- C1 per-frame scale, 64-step gap-aware resampling, and strict nested user folds are unchanged.
- Input order: joint `17 x [xyz, velocity]` followed by H36M edge-order bone `16 x [xyz, velocity]`.
- Bone position is child minus parent after C1 scale normalization; bone velocity is child joint velocity minus parent joint velocity before resampling.
- Input has 198 independently normalized channels; original TemporalClassifier `[64,128]`, 128D embedding, 40 logits.
- Parameter count: 209,640; seed 20260812. All three formal outer folds ran consecutively without an outer-fold gate.

## Fold Results

{fold_results.to_markdown(index=False, floatfmt=".6f")}

## Combined OOF

{combined.to_markdown(index=False, floatfmt=".6f")}

- D2-v1 - C1 accuracy: **{accuracy_delta:+.6f}**.
- D2-v1 - C1 macro-F1: **{macro_delta:+.6f}**.
- Paired discordance: C1-only correct={c1_only}, D2-only correct={d2_only}; exact McNemar p={mcnemar_p:.6g}.
- Paired bootstrap 95% CI for D2-v1 - C1 accuracy: `[{ci_low:+.6f}, {ci_high:+.6f}]` ({args.bootstrap_replicates} replicates, seed {args.seed}).
- D2 improves accuracy for {improved_counts['user']}/14 users and {improved_counts['class']}/40 classes.

## Training Curves

{curve_summary.to_markdown(index=False, floatfmt=".6f")}

## Pre-Registered Decision

**D2-v1 does not pass the success criterion.** Combined accuracy is only marginally higher, its paired confidence interval includes zero, and Macro-F1 decreases, violating the protection metric. C1-TCN remains the Skeleton expert. The D2-only outcomes are recorded as complementarity evidence only; no OOF ensemble weight is fitted in this experiment. Under the frozen route, topology experiments pause here rather than advancing to D3.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `{d2_folds.iloc[0]['oof_assignment_sha256']}`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- Raw histories and predictions remain under `outputs/skeleton_joint_bone_tcn_strict_oof`; no checkpoint or model weight was saved.
"""
    (args.report_dir / "skeleton_joint_bone_tcn_strict_oof_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "c1_accuracy": c1_metrics["accuracy"], "d2_accuracy": d2_metrics["accuracy"],
        "accuracy_delta": accuracy_delta, "macro_f1_delta": macro_delta,
        "c1_only": c1_only, "d2_only": d2_only, "mcnemar_p": mcnemar_p,
        "bootstrap_ci": [ci_low, ci_high], "users_improved": improved_counts["user"],
        "classes_improved": improved_counts["class"],
    }, indent=2))


if __name__ == "__main__":
    main()
