from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260812, 20260912, 20261012]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frozen E1 C1 seed ensemble with C1 strict OOF.")
    parser.add_argument(
        "--e1-root", type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_c1_seed_ensemble_strict_oof",
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
        default=PROJECT_ROOT / "reports/skeleton_c1_seed_ensemble_strict_oof",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def load_folds(root: Path, subdir: str = "") -> pd.DataFrame:
    frames = []
    for fold in range(3):
        frames.append(pd.read_csv(root / subdir / f"fold_{fold}/formal_outer_predictions.csv", encoding="utf-8-sig"))
    frame = pd.concat(frames, ignore_index=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate OOF sample IDs")
    return frame


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


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    assignment = json.loads(args.fold_assignment.read_text(encoding="utf-8"))
    c1 = load_folds(args.c1_root)
    e1 = load_folds(args.e1_root, "ensemble")
    c1 = c1.sort_values("sample_id").reset_index(drop=True)
    e1 = e1.sort_values("sample_id").reset_index(drop=True)
    identity = ["fold", "sample_id", "user_id", "label"]
    if not c1[identity].equals(e1[identity]):
        raise ValueError("C1 and E1 OOF rows are not exactly paired")
    if len(e1) != 2341 or e1["user_id"].nunique() != 14 or e1["label"].nunique() != 40:
        raise ValueError("Unexpected E1 OOF sample space")
    for fold in assignment["folds"]:
        observed = set(e1.loc[e1["fold"] == int(fold["fold"]), "user_id"])
        if observed != set(fold["validation_user_ids"]):
            raise ValueError(f"Fold {fold['fold']} OOF ownership mismatch")

    c1_correct = c1["prediction"].to_numpy() == c1["label"].to_numpy()
    e1_correct = e1["prediction"].to_numpy() == e1["label"].to_numpy()
    c1_only = int((c1_correct & ~e1_correct).sum())
    e1_only = int((~c1_correct & e1_correct).sum())
    ci_low, ci_high = bootstrap_delta(c1_correct, e1_correct, args.bootstrap_replicates, args.bootstrap_seed)
    mcnemar_p = float(binomtest(min(c1_only, e1_only), c1_only + e1_only, 0.5).pvalue)
    combined = pd.DataFrame([metrics(c1, "C1-TCN-single-seed"), metrics(e1, "E1-C1-3seed-ensemble")])
    combined.to_csv(args.report_dir / "e1_vs_c1_combined_oof.csv", index=False, encoding="utf-8-sig")

    paired = pd.DataFrame({
        "sample_id": c1["sample_id"],
        "user_id": c1["user_id"],
        "fold": c1["fold"],
        "label": c1["label"],
        "c1_prediction": c1["prediction"],
        "e1_prediction": e1["prediction"],
        "c1_correct": c1_correct,
        "e1_correct": e1_correct,
    })
    paired["outcome"] = np.select(
        [
            paired["c1_correct"] & paired["e1_correct"],
            paired["c1_correct"] & ~paired["e1_correct"],
            ~paired["c1_correct"] & paired["e1_correct"],
        ],
        ["both_correct", "c1_only", "e1_only"],
        default="both_wrong",
    )
    paired.to_csv(args.report_dir / "e1_vs_c1_paired_oof_outcomes.csv", index=False, encoding="utf-8-sig")

    improved_counts = {}
    for name, column in (("user", "user_id"), ("class", "label"), ("fold", "fold")):
        rows = []
        for value, group in paired.groupby(column):
            baseline = group["c1_correct"].to_numpy()
            candidate = group["e1_correct"].to_numpy()
            low, high = bootstrap_delta(baseline, candidate, args.bootstrap_replicates, args.bootstrap_seed)
            rows.append({
                column: value,
                "samples": len(group),
                "c1_accuracy": baseline.mean(),
                "e1_accuracy": candidate.mean(),
                "e1_minus_c1": candidate.mean() - baseline.mean(),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "c1_only": int((baseline & ~candidate).sum()),
                "e1_only": int((~baseline & candidate).sum()),
            })
        result = pd.DataFrame(rows).sort_values(column)
        result.to_csv(args.report_dir / f"e1_vs_c1_per_{name}.csv", index=False, encoding="utf-8-sig")
        improved_counts[name] = int((result["e1_minus_c1"] > 0).sum())

    member_frames = {}
    member_metrics = []
    for seed in SEEDS:
        member = load_folds(args.e1_root, f"member_seed_{seed}").sort_values("sample_id").reset_index(drop=True)
        if not e1[identity].equals(member[identity]):
            raise ValueError(f"Member {seed} is not exactly paired with ensemble")
        member_frames[seed] = member
        member_metrics.append(metrics(member, f"C1-seed-{seed}"))
    member_table = pd.DataFrame(member_metrics)
    member_table.to_csv(args.report_dir / "e1_member_combined_oof.csv", index=False, encoding="utf-8-sig")

    member_predictions = np.stack([member_frames[seed]["prediction"].to_numpy() for seed in SEEDS], axis=1)
    unanimous = np.all(member_predictions == member_predictions[:, :1], axis=1)
    disagreement_rows = []
    for group_name, selected in (("unanimous", unanimous), ("member_disagreement", ~unanimous)):
        disagreement_rows.append({
            "group": group_name,
            "samples": int(selected.sum()),
            "sample_fraction": float(selected.mean()),
            "c1_accuracy": float(c1_correct[selected].mean()),
            "e1_accuracy": float(e1_correct[selected].mean()),
            "e1_minus_c1": float(e1_correct[selected].mean() - c1_correct[selected].mean()),
        })
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement.to_csv(args.report_dir / "e1_member_disagreement.csv", index=False, encoding="utf-8-sig")

    c1_metrics = combined.set_index("model").loc["C1-TCN-single-seed"]
    e1_metrics = combined.set_index("model").loc["E1-C1-3seed-ensemble"]
    accuracy_delta = float(e1_metrics["accuracy"] - c1_metrics["accuracy"])
    macro_delta = float(e1_metrics["macro_f1_40class"] - c1_metrics["macro_f1_40class"])
    passes = accuracy_delta > 0 and macro_delta >= 0 and ci_low > 0 and improved_counts["fold"] >= 2
    decision = (
        "E1 passes the replacement criterion and becomes the preferred Skeleton expert."
        if passes else
        "E1 does not demonstrate a sufficiently stable gain; the single-seed C1 expert remains preferred."
    )
    fold_table = pd.read_csv(args.report_dir / "e1_vs_c1_per_fold.csv", encoding="utf-8-sig")
    report = f"""# E1-v1 Three-Seed C1-TCN Ensemble Strict Cross-User OOF Development Evidence

## Frozen Contract

- Frozen implementation commit: `8cf30b4f7c7af6be5dbe14f6f7206091d8592992`.
- Three pre-registered member seeds: `{SEEDS[0]}`, `{SEEDS[1]}`, `{SEEDS[2]}`; fold index is added to each training seed.
- Every member independently performs legal inner-user epoch selection and a fresh formal outer refit.
- C1 per-frame scale, joint `xyz + velocity` 102D, T=64, clean views, architecture, optimizer, scheduler, and all training settings are unchanged.
- Ensemble rule is the fixed equal mean of member softmax probabilities. No member selection, outer-fold weighting, or gate is fitted.
- Each member has 172,776 parameters; total inference parameters are 518,328.

## Member Results

{member_table.to_markdown(index=False, floatfmt=".6f")}

## Fold Results

{fold_table.to_markdown(index=False, floatfmt=".6f")}

## Combined OOF

{combined.to_markdown(index=False, floatfmt=".6f")}

- E1 - C1 accuracy: **{accuracy_delta:+.6f}**.
- E1 - C1 Macro-F1: **{macro_delta:+.6f}**.
- Paired discordance: C1-only correct={c1_only}, E1-only correct={e1_only}; exact McNemar p={mcnemar_p:.6g}.
- Paired bootstrap 95% CI for E1 - C1 accuracy: `[{ci_low:+.6f}, {ci_high:+.6f}]` ({args.bootstrap_replicates} replicates, seed {args.bootstrap_seed}).
- E1 improves accuracy for {improved_counts['fold']}/3 folds, {improved_counts['user']}/14 users, and {improved_counts['class']}/40 classes.

## Member Diversity

{disagreement.to_markdown(index=False, floatfmt=".6f")}

This is a diagnostic decomposition. It shows where fixed probability averaging changes outcomes; it was not used to select members or fit weights.

## Pre-Registered Decision

**{decision}** The decision requires positive combined Accuracy, non-decreasing Macro-F1, a paired bootstrap interval above zero, and improvement on at least two folds. Per-user and per-class results remain stability diagnostics rather than tuning inputs.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- All nine member summaries state that outer labels were not used for selection and preprocessing excluded scope-validation users.
- Raw histories and member predictions remain under `outputs/skeleton_c1_seed_ensemble_strict_oof`; no checkpoint or model weight was saved.
"""
    (args.report_dir / "skeleton_c1_seed_ensemble_strict_oof_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "c1_accuracy": c1_metrics["accuracy"],
        "e1_accuracy": e1_metrics["accuracy"],
        "accuracy_delta": accuracy_delta,
        "macro_f1_delta": macro_delta,
        "c1_only": c1_only,
        "e1_only": e1_only,
        "mcnemar_p": mcnemar_p,
        "bootstrap_ci": [ci_low, ci_high],
        "folds_improved": improved_counts["fold"],
        "users_improved": improved_counts["user"],
        "classes_improved": improved_counts["class"],
        "passes_replacement": passes,
    }, indent=2))


if __name__ == "__main__":
    main()
