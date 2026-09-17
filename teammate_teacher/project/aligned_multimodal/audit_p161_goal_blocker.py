"""Requirement-by-requirement completion and repeated-blocker audit for 91%."""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUTPUT = RUNS / "p161_goal_blocker_audit_v1"
CHAMPION = RUNS / "p150_repeat_branch_confidence_selector_v1/summary.json"
CEILING = RUNS / "p126_expanded_candidate_ceiling_v1/summary.json"
UPSTREAM = (
    RUNS / "p140_source_stable_expanded_sequence_v1/summary.json",
    RUNS / "p143_p142_source_user_safe_selector_v1/summary.json",
    RUNS / "p145_dual_token_agreement_selector_v1/summary.json",
    CHAMPION,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_violations(summary: dict) -> list[str]:
    protocol = summary.get("protocol", {})
    violations = []
    for key in ("test_labels_loaded", "test_rows_loaded"):
        if int(protocol.get(key, 0)) != 0:
            violations.append(f"{key}={protocol[key]}")
    if bool(protocol.get("submission_generated", False)):
        violations.append("submission_generated=true")
    if bool(protocol.get("user_id_used_as_feature", False)):
        violations.append("user_id_used_as_feature=true")
    if bool(protocol.get("held_labels_used_for_selection", False)):
        violations.append("held_labels_used_for_selection=true")
    return violations


def main() -> None:
    champion = load(CHAMPION)
    ceiling = load(CEILING)
    rows = int(champion["aggregate"]["rows"])
    correct = int(champion["aggregate"]["correct"])
    target = int(champion["aggregate"]["target_0.91_correct"])
    if target != 2248 or rows != 2470:
        raise RuntimeError(f"target universe changed rows={rows} target={target}")
    if int(ceiling["current_champion"]["correct"]) != correct:
        raise RuntimeError("champion and ceiling disagree")

    boundary = {}
    all_violations = []
    for path in UPSTREAM:
        summary = load(path)
        violations = protocol_violations(summary)
        boundary[path.parent.name] = {
            "summary": str(path),
            "violations": violations,
        }
        all_violations.extend(f"{path.parent.name}:{value}" for value in violations)
    if all_violations:
        raise RuntimeError(f"protocol boundary violated: {all_violations}")

    stage_pattern = re.compile(r"^p(13[0-9]|14[0-9]|15[0-9]|160)_")
    submission_files = [
        str(path)
        for directory in RUNS.iterdir()
        if directory.is_dir() and stage_pattern.match(directory.name)
        for path in directory.rglob("*")
        if path.is_file() and "submission" in path.name.lower()
    ]
    if submission_files:
        raise RuntimeError(f"research stage emitted submissions: {submission_files[:5]}")

    expanded = int(ceiling["expanded_bank_oracle"]["correct"])
    recoverable = int(
        ceiling["current_champion_errors_recoverable_by_expanded_bank"]
    )
    report = {
        "stage": "P161_goal_completion_and_blocker_audit_v1",
        "status": "blocked_after_repeated_cross_subject_arbitration_failure",
        "objective": {
            "required_accuracy": 0.91,
            "rows": rows,
            "required_correct": target,
            "current_correct": correct,
            "current_accuracy": correct / rows,
            "gap_correct": target - correct,
            "achieved": correct >= target,
        },
        "legal_boundary": {
            "all_checked_stages_clean": not all_violations,
            "checked": boundary,
            "submission_files_in_p130_p160": submission_files,
            "test_labels_loaded": 0,
            "identifier_mapping_used": False,
            "user_id_used_as_feature": False,
            "submission_generated": False,
        },
        "coverage": {
            "expanded_oracle_correct": expanded,
            "expanded_oracle_accuracy": expanded / rows,
            "oracle_headroom_over_target": expanded - target,
            "current_errors": rows - correct,
            "current_errors_recoverable": recoverable,
        },
        "repeated_blocker": {
            "condition": (
                "Candidate coverage exceeds 91%, but source-trained rescue/harm "
                "arbitration repeatedly reverses on held subjects."
            ),
            "observed_across": "P117-P160",
            "consecutive_goal_turns": ">=3",
            "recent_independent_evidence_attempts": [
                "EPIC SlowFast",
                "EgoVLP",
                "LaViLa Ego4D and EK100",
                "P96 token Transformer variants",
                "repeat/session joint decoding",
                "candidate Set-Transformer",
                "unlabeled domain weighting and kNN graphs",
            ],
            "meaningful_progress_without_external_change": False,
            "required_external_change": [
                "new independent labeled subject cohort for frozen confirmation",
                "genuinely new physical modality/evidence with transferable confidence",
                "or user-authorized scope change (Test labels/leakage remain prohibited)",
            ],
        },
        "decision": (
            "91% is not achieved. Do not claim completion. Mark the persistent goal "
            "blocked until new independent evidence or an external-state change exists."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
