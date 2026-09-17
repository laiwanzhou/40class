from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
RUNS = PROJECT_DIR / "runs"
BASELINE_HISTORY = RUNS / "p86_visual_pixel_fold0_hybrid_v2/inner_history.csv"
TRIAL_LOGS = {
    "direct_clip_feature_distillation_v3": RUNS / "p86_visual_pixel_fold0_feature_v3.out.log",
    "structured_six_token_fusion_v4": RUNS / "p86_visual_pixel_fold0_structured_v4.out.log",
    "gated_structured_residual_v5": RUNS / "p86_visual_pixel_fold0_gated_residual_v5.out.log",
    "unfreeze_layer2_v6": RUNS / "p86_visual_pixel_fold0_unfreeze_l2_v6.out.log",
    "resolution_160_v7": RUNS / "p86_visual_pixel_fold0_r160_v7.out.log",
    "twelve_frames_v8": RUNS / "p86_visual_pixel_fold0_t12_v8.out.log",
}
OUTPUT = RUNS / "p86_visual_iteration_audit_v1"


def score(row: dict[str, Any]) -> float:
    return (
        float(row["val_accuracy"])
        + 0.5 * float(row["val_macro_f1"])
        + 0.25 * float(row["val_worst_subject_accuracy"])
    )


def read_csv_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_inner_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_accuracy" in row:
            rows.append(row)
        elif rows:
            # Refit begins after the inner records and has no validation keys.
            break
    return rows


def summarize(name: str, rows: list[dict[str, Any]], status: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"no inner validation rows for {name}")
    best = max(rows, key=score)
    best_accuracy = max(rows, key=lambda row: float(row["val_accuracy"]))
    return {
        "name": name,
        "status": status,
        "inner_epochs": len(rows),
        "selected_epoch": int(best["epoch"]),
        "selected_accuracy": float(best["val_accuracy"]),
        "selected_macro_f1": float(best["val_macro_f1"]),
        "selected_worst_subject_accuracy": float(best["val_worst_subject_accuracy"]),
        "selection_score": score(best),
        "maximum_accuracy_epoch": int(best_accuracy["epoch"]),
        "maximum_accuracy": float(best_accuracy["val_accuracy"]),
    }


def main() -> None:
    baseline_rows = read_csv_history(BASELINE_HISTORY)
    baseline = summarize("gated_hybrid_v2", baseline_rows, "best_rollback")
    trials = [
        summarize(name, read_inner_log(path), "rejected_before_outer_held")
        for name, path in TRIAL_LOGS.items()
    ]
    for trial in trials:
        trial["selected_accuracy_delta_pp_vs_v2"] = 100.0 * (
            trial["selected_accuracy"] - baseline["selected_accuracy"]
        )
        trial["selection_score_delta_vs_v2"] = (
            trial["selection_score"] - baseline["selection_score"]
        )
    summary = {
        "protocol": "P86 visual-stage autonomous iteration audit",
        "gate": (
            "A candidate that does not exceed the rollback model on the inner subject-disjoint "
            "selection score is stopped before outer-held evaluation."
        ),
        "baseline": baseline,
        "rejected_trials": trials,
        "outer_held_preservation": (
            "Every listed trial was rejected from inner subject-disjoint evidence; no outer-held "
            "predictions were generated for any rejected candidate."
        ),
        "next_controlled_trial": (
            "Replace the framewise 2D+TSM encoder with a Kinetics-pretrained lightweight MC3-18 "
            "backbone under the same nested subject-disjoint split, six-clip inputs, and hybrid "
            "training targets. This tests whether genuine local 3D motion modeling is the missing "
            "mechanism after resolution, frame-count, fusion, feature-distillation, and unfreezing "
            "changes all failed the inner gate."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "inner_trial_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [baseline, *trials]
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
