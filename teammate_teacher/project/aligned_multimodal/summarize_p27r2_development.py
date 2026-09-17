from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
RESEARCH_DOCS = REPO_DIR / "docs" / "research"
AUDIT_DIR = PROJECT_DIR / "runs" / "p27_r2_event_audit"
PILOT_DIR = PROJECT_DIR / "runs" / "p27_r2_fold0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    summaries = {
        1: json.loads(
            (PILOT_DIR / "development_iteration1_summary.json").read_text(
                encoding="utf-8"
            )
        ),
        2: json.loads(
            (PILOT_DIR / "development_iteration2_summary.json").read_text(
                encoding="utf-8"
            )
        ),
        3: json.loads(
            (PILOT_DIR / "development_summary.json").read_text(encoding="utf-8")
        ),
    }
    rows = []
    for iteration, summary in summaries.items():
        for fold_name, fold in summary["folds"].items():
            row: dict[str, Any] = {
                "iteration": iteration,
                "inner_fold": int(fold_name),
                "base_overall": fold["base"]["overall"]["accuracy"],
                "ce_overall": fold["ce_only"]["overall"]["accuracy"],
                "event_overall": fold["event"]["overall"]["accuracy"],
                "base_small": fold["base"]["small"]["accuracy"],
                "ce_small": fold["ce_only"]["small"]["accuracy"],
                "event_small": fold["event"]["small"]["accuracy"],
                "base_hard": fold["base"]["hard"]["accuracy"],
                "ce_hard": fold["ce_only"]["hard"]["accuracy"],
                "event_hard": fold["event"]["hard"]["accuracy"],
                "event_minus_ce_overall_pp": 100
                * (
                    fold["event"]["overall"]["accuracy"]
                    - fold["ce_only"]["overall"]["accuracy"]
                ),
                "event_minus_ce_hard_pp": 100
                * (
                    fold["event"]["hard"]["accuracy"]
                    - fold["ce_only"]["hard"]["accuracy"]
                ),
                "initial_max_abs_logit_error": fold["event_model"][
                    "initial_max_abs_logit_error"
                ],
            }
            rows.append(row)
    write_csv(PILOT_DIR / "development_iterations.csv", rows)

    paths = [
        PROJECT_DIR / "p27r2_event_data.py",
        PROJECT_DIR / "audit_p27r2_event_labels.py",
        PROJECT_DIR / "p27r2_model.py",
        PROJECT_DIR / "train_p27r2_fold0.py",
        PROJECT_DIR / "summarize_p27r2_development.py",
        PROJECT_DIR / "configs" / "p27_r2_fold0.json",
        PROJECT_DIR / "data" / "p27_r2_outer_train_event_anchors.csv",
        RESEARCH_DOCS
        / "20_events_and_sequence"
        / "33_P27-R2困难动作事件重构与嵌套开发验证_未进入outer-held.md",
        AUDIT_DIR / "summary.json",
        AUDIT_DIR / "candidate_event_audit.csv",
        AUDIT_DIR / "inner_event_prediction.csv",
        AUDIT_DIR / "outer_train_anchor_alignment.csv",
        AUDIT_DIR / "outer_train_pair_distributions.csv",
        AUDIT_DIR / "outer_train_anchor_traces.png",
        AUDIT_DIR / "outer_train_subject_distributions.png",
        PILOT_DIR / "development_iteration1_summary.json",
        PILOT_DIR / "development_iteration1_ledger.csv",
        PILOT_DIR / "development_iteration2_summary.json",
        PILOT_DIR / "development_iteration2_ledger.csv",
        PILOT_DIR / "development_summary.json",
        PILOT_DIR / "development_ledger.csv",
        PILOT_DIR / "development_iterations.csv",
        PILOT_DIR / "sampling_coverage_audit.json",
        PILOT_DIR / "runtime_benchmark.json",
        PILOT_DIR / "frozen_final_config.json",
    ]
    local_only = [
        AUDIT_DIR / "event_cache_v2.npz",
        PILOT_DIR / "development_core_loso_fold_0.npz",
        PILOT_DIR / "development_core_loso_fold_1.npz",
        PILOT_DIR / "development_core_loso_fold_2.npz",
    ]
    manifest = []
    for path in paths + local_only:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest.append(
            {
                "path": str(path.resolve().relative_to(REPO_DIR.resolve())),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "git_policy": "local_only_cache" if path in local_only else "commit",
            }
        )
    write_csv(PILOT_DIR / "artifact_manifest.csv", manifest)
    (PILOT_DIR / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "protocol": "p27-r2-fold0-nested-subject-development-v1",
                "outer_held_predictions_generated": False,
                "artifacts": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
