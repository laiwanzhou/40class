from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260812, 20260912, 20261012]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frozen E2 balanced E1 with E1 strict OOF.")
    parser.add_argument(
        "--e2-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_c1_balanced_seed_ensemble_strict_oof",
    )
    parser.add_argument(
        "--e1-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_c1_seed_ensemble_strict_oof",
    )
    parser.add_argument(
        "--fold-assignment",
        type=Path,
        default=PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/skeleton_c1_balanced_seed_ensemble_strict_oof.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "metadata/manifest.csv",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/skeleton_c1_balanced_seed_ensemble_strict_oof",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def load_folds(root: Path, subdir: str) -> pd.DataFrame:
    paths = [root / subdir / f"fold_{fold}/formal_outer_predictions.csv" for fold in range(3)]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"Missing formal OOF predictions: {missing}")
    frame = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in paths], ignore_index=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate OOF sample IDs")
    return frame.sort_values("sample_id").reset_index(drop=True)


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


def bootstrap_deltas(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    rng = np.random.default_rng(seed)
    accuracy_deltas = np.empty(replicates, dtype=np.float64)
    macro_deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[selected]
        baseline = baseline_predictions[selected]
        candidate = candidate_predictions[selected]
        accuracy_deltas[replicate] = (candidate == sampled_labels).mean() - (baseline == sampled_labels).mean()
        macro_deltas[replicate] = f1_score(
            sampled_labels, candidate, labels=np.arange(40), average="macro", zero_division=0
        ) - f1_score(sampled_labels, baseline, labels=np.arange(40), average="macro", zero_division=0)
    return (
        (float(np.quantile(accuracy_deltas, 0.025)), float(np.quantile(accuracy_deltas, 0.975))),
        (float(np.quantile(macro_deltas, 0.025)), float(np.quantile(macro_deltas, 0.975))),
    )


def grouped_results(paired: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in paired.groupby(column):
        rows.append({
            column: value,
            "samples": len(group),
            "e1_accuracy": accuracy_score(group["label"], group["e1_prediction"]),
            "e2_accuracy": accuracy_score(group["label"], group["e2_prediction"]),
            "e1_macro_f1_40class": f1_score(
                group["label"], group["e1_prediction"], labels=np.arange(40), average="macro", zero_division=0
            ),
            "e2_macro_f1_40class": f1_score(
                group["label"], group["e2_prediction"], labels=np.arange(40), average="macro", zero_division=0
            ),
            "e1_only": int((group["e1_correct"] & ~group["e2_correct"]).sum()),
            "e2_only": int((~group["e1_correct"] & group["e2_correct"]).sum()),
        })
    result = pd.DataFrame(rows).sort_values(column)
    result["e2_minus_e1_accuracy"] = result["e2_accuracy"] - result["e1_accuracy"]
    result["e2_minus_e1_macro_f1"] = result["e2_macro_f1_40class"] - result["e1_macro_f1_40class"]
    return result


def class_results(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, prediction_column in (("e1", "e1_prediction"), ("e2", "e2_prediction")):
        precision, recall, f1, support = precision_recall_fscore_support(
            paired["label"], paired[prediction_column], labels=np.arange(40), zero_division=0
        )
        for class_id in range(40):
            rows.append({
                "model": name,
                "class_id": class_id,
                "support": int(support[class_id]),
                "precision": precision[class_id],
                "recall": recall[class_id],
                "f1": f1[class_id],
            })
    wide = pd.DataFrame(rows).pivot(index=["class_id", "support"], columns="model").reset_index()
    wide.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in wide.columns
    ]
    wide["e2_minus_e1_f1"] = wide["f1_e2"] - wide["f1_e1"]
    return wide.sort_values("class_id")


def support_bucket_results(classes: pd.DataFrame) -> pd.DataFrame:
    buckets = pd.cut(
        classes["support"],
        bins=[-1, 15, 31, 63, np.inf],
        labels=["support_le_15", "support_16_31", "support_32_63", "support_ge_64"],
    )
    frame = classes.assign(support_bucket=buckets)
    return frame.groupby("support_bucket", observed=False).agg(
        classes=("class_id", "count"),
        samples=("support", "sum"),
        e1_macro_f1=("f1_e1", "mean"),
        e2_macro_f1=("f1_e2", "mean"),
        e2_minus_e1_macro_f1=("e2_minus_e1_f1", "mean"),
    ).reset_index()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    assignment = json.loads(args.fold_assignment.read_text(encoding="utf-8"))
    e1 = load_folds(args.e1_root, "ensemble")
    e2 = load_folds(args.e2_root, "ensemble")
    identity = ["fold", "sample_id", "user_id", "label"]
    if not e1[identity].equals(e2[identity]):
        raise ValueError("E1 and E2 OOF rows are not exactly paired")
    if len(e2) != 2341 or e2["user_id"].nunique() != 14 or e2["label"].nunique() != 40:
        raise ValueError("Unexpected E2 OOF sample space")
    for fold in assignment["folds"]:
        observed = set(e2.loc[e2["fold"] == int(fold["fold"]), "user_id"])
        if observed != set(fold["validation_user_ids"]):
            raise ValueError(f"Fold {fold['fold']} OOF ownership mismatch")

    summaries = [
        json.loads((args.e2_root / f"member_seed_{seed}/fold_{fold}/summary.json").read_text(encoding="utf-8"))
        for seed in SEEDS for fold in range(3)
    ]
    if any(
        summary["outer_validation_labels_used_for_selection"]
        or not summary["sampling_fit_excludes_scope_validation_users"]
        or not summary["preprocessing_fit_excludes_scope_validation_users"]
        for summary in summaries
    ):
        raise ValueError("E2 provenance flags do not satisfy the frozen strict OOF contract")

    e1_correct = e1["prediction"].to_numpy() == e1["label"].to_numpy()
    e2_correct = e2["prediction"].to_numpy() == e2["label"].to_numpy()
    e1_only = int((e1_correct & ~e2_correct).sum())
    e2_only = int((~e1_correct & e2_correct).sum())
    labels = e1["label"].to_numpy()
    accuracy_ci, macro_ci = bootstrap_deltas(
        labels,
        e1["prediction"].to_numpy(),
        e2["prediction"].to_numpy(),
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    mcnemar_p = float(binomtest(min(e1_only, e2_only), e1_only + e2_only, 0.5).pvalue)
    combined = pd.DataFrame([metrics(e1, "E1-C1-3seed-ensemble"), metrics(e2, "E2-balanced-E1-3seed-ensemble")])
    combined.to_csv(args.report_dir / "e2_vs_e1_combined_oof.csv", index=False, encoding="utf-8-sig")

    paired = pd.DataFrame({
        "sample_id": e1["sample_id"],
        "user_id": e1["user_id"],
        "fold": e1["fold"],
        "label": labels,
        "e1_prediction": e1["prediction"],
        "e2_prediction": e2["prediction"],
        "e1_correct": e1_correct,
        "e2_correct": e2_correct,
    })
    paired["outcome"] = np.select(
        [
            paired["e1_correct"] & paired["e2_correct"],
            paired["e1_correct"] & ~paired["e2_correct"],
            ~paired["e1_correct"] & paired["e2_correct"],
        ],
        ["both_correct", "e1_only", "e2_only"],
        default="both_wrong",
    )
    paired.to_csv(args.report_dir / "e2_vs_e1_paired_oof_outcomes.csv", index=False, encoding="utf-8-sig")

    group_tables = {}
    for name, column in (("fold", "fold"), ("user", "user_id")):
        group_tables[name] = grouped_results(paired, column)
        group_tables[name].to_csv(args.report_dir / f"e2_vs_e1_per_{name}.csv", index=False, encoding="utf-8-sig")
    classes = class_results(paired)
    action_names = (
        pd.read_csv(args.manifest, encoding="utf-8-sig", usecols=["class_id", "action_name"])
        .drop_duplicates("class_id")
    )
    classes = classes.merge(action_names, on="class_id", how="left", validate="one_to_one")
    if classes["action_name"].isna().any():
        raise ValueError("Missing action names for one or more E2 classes")
    classes.to_csv(args.report_dir / "e2_vs_e1_per_class.csv", index=False, encoding="utf-8-sig")
    buckets = support_bucket_results(classes)
    buckets.to_csv(args.report_dir / "e2_vs_e1_class_support_buckets.csv", index=False, encoding="utf-8-sig")

    member_metrics = []
    for seed in SEEDS:
        member = load_folds(args.e2_root, f"member_seed_{seed}")
        if not e2[identity].equals(member[identity]):
            raise ValueError(f"E2 member {seed} is not exactly paired with its ensemble")
        member_metrics.append(metrics(member, f"E2-seed-{seed}"))
    members = pd.DataFrame(member_metrics)
    members.to_csv(args.report_dir / "e2_member_combined_oof.csv", index=False, encoding="utf-8-sig")

    baseline = combined.set_index("model").loc["E1-C1-3seed-ensemble"]
    candidate = combined.set_index("model").loc["E2-balanced-E1-3seed-ensemble"]
    accuracy_delta = float(candidate["accuracy"] - baseline["accuracy"])
    macro_delta = float(candidate["macro_f1_40class"] - baseline["macro_f1_40class"])
    folds_macro_improved = int((group_tables["fold"]["e2_minus_e1_macro_f1"] > 0).sum())
    users_accuracy_improved = int((group_tables["user"]["e2_minus_e1_accuracy"] > 0).sum())
    classes_f1_improved = int((classes["e2_minus_e1_f1"] > 0).sum())
    largest_gains = classes.nlargest(5, "e2_minus_e1_f1")[
        ["class_id", "action_name", "support", "f1_e1", "f1_e2", "e2_minus_e1_f1"]
    ]
    largest_declines = classes.nsmallest(5, "e2_minus_e1_f1")[
        ["class_id", "action_name", "support", "f1_e1", "f1_e2", "e2_minus_e1_f1"]
    ]
    passes = accuracy_delta >= 0 and macro_delta > 0 and macro_ci[0] > 0 and folds_macro_improved >= 2
    decision = (
        "E2-v1 passes the pre-registered replacement criterion and becomes the preferred Skeleton expert."
        if passes else
        "E2-v1 does not pass the pre-registered replacement criterion; E1 remains the preferred Skeleton expert."
    )
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    report = f"""# E2-v1 Balanced E1 Strict Cross-User OOF Development Evidence

## Frozen Contract

- E1 architecture, C1 preprocessing, T=64, optimizer, scheduler, CE loss, batch size, nested OOF protocol, and all three seeds are unchanged.
- The only experimental variable is train-scope sampling weight `1 / sqrt(n_c)` with replacement and exactly one train-scope-size draw per epoch.
- Class counts are fitted independently on inner-fit and formal outer-train. Validation labels are never used to fit sampling or preprocessing.
- Fixed equal mean of three member softmax probabilities; no member selection, weighting, or outer-fold gate.
- E2 config SHA256: `{config_sha}`.

## Member Results

{members.to_markdown(index=False, floatfmt=".6f")}

## Fold Results

{group_tables['fold'].to_markdown(index=False, floatfmt=".6f")}

## Combined OOF

{combined.to_markdown(index=False, floatfmt=".6f")}

- E2 - E1 Accuracy: **{accuracy_delta:+.6f}**; paired bootstrap 95% CI `[{accuracy_ci[0]:+.6f}, {accuracy_ci[1]:+.6f}]`.
- E2 - E1 Macro-F1: **{macro_delta:+.6f}**; paired bootstrap 95% CI `[{macro_ci[0]:+.6f}, {macro_ci[1]:+.6f}]`.
- Paired discordance: E1-only correct={e1_only}, E2-only correct={e2_only}; exact McNemar p={mcnemar_p:.6g}.
- E2 improves Macro-F1 for {folds_macro_improved}/3 folds, Accuracy for {users_accuracy_improved}/14 users, and F1 for {classes_f1_improved}/40 classes.

## Class-Support Diagnosis

{buckets.to_markdown(index=False, floatfmt=".6f")}

This support analysis is diagnostic and was not used to set the sampling exponent or select a model.

## Largest Class Changes

Largest F1 gains:

{largest_gains.to_markdown(index=False, floatfmt=".6f")}

Largest F1 declines:

{largest_declines.to_markdown(index=False, floatfmt=".6f")}

## Pre-Registered Decision

**{decision}** Replacement requires non-decreasing combined Accuracy, increasing Macro-F1, a Macro-F1 paired-bootstrap interval above zero, and Macro-F1 improvement on at least two folds.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `{summaries[0]['oof_assignment_sha256']}`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- All nine member summaries pass the preprocessing and sampler scope-provenance checks.
- Raw histories, sampling provenance, and predictions remain under `outputs/skeleton_c1_balanced_seed_ensemble_strict_oof`; no checkpoint or model weight was saved.
"""
    (args.report_dir / "skeleton_c1_balanced_seed_ensemble_strict_oof_report.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({
        "e1_accuracy": baseline["accuracy"],
        "e2_accuracy": candidate["accuracy"],
        "accuracy_delta": accuracy_delta,
        "accuracy_bootstrap_ci": accuracy_ci,
        "macro_f1_delta": macro_delta,
        "macro_f1_bootstrap_ci": macro_ci,
        "e1_only": e1_only,
        "e2_only": e2_only,
        "mcnemar_p": mcnemar_p,
        "folds_macro_improved": folds_macro_improved,
        "users_accuracy_improved": users_accuracy_improved,
        "classes_f1_improved": classes_f1_improved,
        "passes_replacement": passes,
    }, indent=2))


if __name__ == "__main__":
    main()
