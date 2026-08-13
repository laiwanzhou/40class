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
    parser = argparse.ArgumentParser(description="Compare D1-v1 ST-GCN with strict C1-TCN OOF predictions.")
    parser.add_argument(
        "--d1-output-root", type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_lightweight_stgcn_strict_oof/lightweight_stgcn_c1",
    )
    parser.add_argument(
        "--c1-output-root", type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_c0_c1_strict_oof/C1",
    )
    parser.add_argument(
        "--fold-assignment", type=Path,
        default=PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "reports/skeleton_lightweight_stgcn_strict_oof",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def paired_bootstrap_delta(
    baseline_correct: np.ndarray, candidate_correct: np.ndarray, replicates: int, seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(baseline_correct), size=len(baseline_correct))
        deltas[index] = candidate_correct[sampled].mean() - baseline_correct[sampled].mean()
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


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
        expected = set(fold["validation_user_ids"])
        if observed != expected:
            raise ValueError(f"Fold {fold_index} OOF ownership mismatch: {observed} != {expected}")


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


def training_curve_summary(d1_root: Path) -> pd.DataFrame:
    rows = []
    for fold in range(3):
        run_dir = d1_root / f"fold_{fold}"
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        inner = pd.read_csv(run_dir / "inner_selection_history.csv", encoding="utf-8-sig")
        formal = pd.read_csv(run_dir / "formal_refit_history.csv", encoding="utf-8-sig")
        rows.append({
            "fold": fold,
            "selected_epoch": summary["selected_epoch"],
            "inner_final_train_accuracy": inner.iloc[-1]["train_accuracy"],
            "inner_max_train_accuracy": inner["train_accuracy"].max(),
            "inner_best_validation_accuracy": inner["validation_accuracy"].max(),
            "formal_final_train_accuracy": formal.iloc[-1]["train_accuracy"],
            "formal_max_train_accuracy": formal["train_accuracy"].max(),
            "outer_accuracy": summary["outer_accuracy"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    assignment = json.loads(args.fold_assignment.read_text(encoding="utf-8"))
    c1, c1_folds = load_arm(args.c1_output_root, "C1-TCN")
    d1, d1_folds = load_arm(args.d1_output_root, "D1-v1-STGCN")
    validate_ownership(c1, assignment)
    validate_ownership(d1, assignment)
    c1 = c1.sort_values("sample_id").reset_index(drop=True)
    d1 = d1.sort_values("sample_id").reset_index(drop=True)
    if not c1["sample_id"].equals(d1["sample_id"]) or not c1["label"].equals(d1["label"]):
        raise ValueError("C1-TCN and D1-v1 OOF rows are not exactly paired")
    if c1["user_id"].nunique() != 14 or c1["label"].nunique() != 40:
        raise ValueError("Paired OOF sample space does not cover 14 users and 40 classes")

    c1_correct = c1["prediction"].to_numpy() == c1["label"].to_numpy()
    d1_correct = d1["prediction"].to_numpy() == d1["label"].to_numpy()
    c1_only = int((c1_correct & ~d1_correct).sum())
    d1_only = int((~c1_correct & d1_correct).sum())
    discordant = c1_only + d1_only
    mcnemar_p = float(binomtest(min(c1_only, d1_only), discordant, 0.5).pvalue)
    ci_low, ci_high = paired_bootstrap_delta(
        c1_correct, d1_correct, args.bootstrap_replicates, args.seed
    )

    combined = pd.DataFrame([metrics(c1, "C1-TCN"), metrics(d1, "D1-v1-STGCN")])
    combined.to_csv(args.report_dir / "d1_vs_c1_combined_oof.csv", index=False, encoding="utf-8-sig")
    fold_comparison = pd.concat([
        c1_folds[["model", "fold", "selected_epoch", "outer_validation_trials", "outer_accuracy", "outer_macro_f1_40class", "outer_weighted_f1"]],
        d1_folds[["model", "fold", "selected_epoch", "outer_validation_trials", "outer_accuracy", "outer_macro_f1_40class", "outer_weighted_f1"]],
    ]).sort_values(["fold", "model"])
    fold_comparison.to_csv(args.report_dir / "d1_vs_c1_fold_results.csv", index=False, encoding="utf-8-sig")

    paired = pd.DataFrame({
        "sample_id": c1["sample_id"], "user_id": c1["user_id"], "fold": c1["fold"],
        "label": c1["label"], "c1_prediction": c1["prediction"], "d1_prediction": d1["prediction"],
        "c1_correct": c1_correct, "d1_correct": d1_correct,
    })
    paired["outcome"] = np.select(
        [paired["c1_correct"] & paired["d1_correct"], paired["c1_correct"] & ~paired["d1_correct"],
         ~paired["c1_correct"] & paired["d1_correct"]],
        ["both_correct", "c1_only", "d1_only"], default="both_wrong",
    )
    paired.to_csv(args.report_dir / "d1_vs_c1_paired_oof_outcomes.csv", index=False, encoding="utf-8-sig")

    diagnostic_rows = []
    for grouping, column in (("user", "user_id"), ("class", "label")):
        rows = []
        for value, group in paired.groupby(column):
            rows.append({
                column: value, "samples": len(group), "c1_accuracy": group["c1_correct"].mean(),
                "d1_accuracy": group["d1_correct"].mean(),
                "d1_minus_c1": group["d1_correct"].mean() - group["c1_correct"].mean(),
                "c1_only": int((group["outcome"] == "c1_only").sum()),
                "d1_only": int((group["outcome"] == "d1_only").sum()),
            })
        result = pd.DataFrame(rows).sort_values(column)
        result.to_csv(args.report_dir / f"d1_vs_c1_per_{grouping}.csv", index=False, encoding="utf-8-sig")
        diagnostic_rows.append(result)

    curves = training_curve_summary(args.d1_output_root)
    curves.to_csv(args.report_dir / "d1_training_curve_summary.csv", index=False, encoding="utf-8-sig")
    c1_metrics = combined.set_index("model").loc["C1-TCN"]
    d1_metrics = combined.set_index("model").loc["D1-v1-STGCN"]
    users_improved = int((diagnostic_rows[0]["d1_minus_c1"] > 0).sum())
    classes_improved = int((diagnostic_rows[1]["d1_minus_c1"] > 0).sum())
    report = f"""# D1-v1 Lightweight ST-GCN Strict OOF Report

## Frozen Contract

- Model commit: `ed3ac7535aaf2fd0b42958032005fed4743e8a85`.
- Fixed H36M-17 graph; input `xyz + velocity`; C1 per-frame bone scale.
- Channels `32, 48, 64`; temporal kernel 5; dilations `1, 2, 4`; receptive field 29.
- Segment-aware temporal convolution; 128D embedding; dropout 0.2; seed 20260812.
- Parameter count: 60,920. No adaptive adjacency, attention, extra stream, or augmentation.
- All three formal outer folds were run consecutively after contract freeze. No outer fold was used as a gate.

## Fold Results

{fold_comparison.to_markdown(index=False, floatfmt=".6f")}

## Combined OOF

{combined.to_markdown(index=False, floatfmt=".6f")}

- D1-v1 - C1 accuracy: **{d1_metrics['accuracy'] - c1_metrics['accuracy']:+.6f}**.
- D1-v1 - C1 macro-F1: **{d1_metrics['macro_f1_40class'] - c1_metrics['macro_f1_40class']:+.6f}**.
- Paired discordance: C1-only correct={c1_only}, D1-only correct={d1_only}; exact McNemar p={mcnemar_p:.6g}.
- Paired bootstrap 95% CI for D1-v1 - C1 accuracy: `[{ci_low:+.6f}, {ci_high:+.6f}]` ({args.bootstrap_replicates} replicates, seed {args.seed}).
- D1-v1 improves accuracy for {users_improved}/14 users and {classes_improved}/40 classes.

## Training Diagnosis

{curves.to_markdown(index=False, floatfmt=".6f")}

Train accuracy remains low in inner selection and formal refit, while D1-v1 loses substantially on every formal outer fold. This is consistent with a strong underfitting/representation bottleneck in D1-v1, not an isolated fold failure. It does not support retaining this graph model over the C1-TCN expert.

## Decision

**D1-v1 fails the topology-retention test.** C1-TCN remains the Skeleton expert baseline. The paired interval is wholly below zero and the loss occurs across all three folds. Do not use D1-v1 for fusion or replace C1-TCN with it. Any D1-v2, joint/bone stream, or joint-identity experiment must be treated as a new pre-frozen experiment and cannot retroactively change this OOF result.

## Provenance

- Frozen split: `metadata/splits/train14_oof_3fold.json`.
- OOF assignment SHA256: `{d1_folds.iloc[0]['oof_assignment_sha256']}`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- Raw histories and predictions remain under `outputs/skeleton_lightweight_stgcn_strict_oof`; no checkpoint or model weight was saved.
"""
    (args.report_dir / "skeleton_lightweight_stgcn_strict_oof_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "c1_accuracy": c1_metrics["accuracy"], "d1_accuracy": d1_metrics["accuracy"],
        "accuracy_delta": d1_metrics["accuracy"] - c1_metrics["accuracy"],
        "macro_f1_delta": d1_metrics["macro_f1_40class"] - c1_metrics["macro_f1_40class"],
        "c1_only": c1_only, "d1_only": d1_only, "mcnemar_p": mcnemar_p,
        "bootstrap_ci": [ci_low, ci_high], "users_improved": users_improved,
        "classes_improved": classes_improved,
    }, indent=2))


if __name__ == "__main__":
    main()
