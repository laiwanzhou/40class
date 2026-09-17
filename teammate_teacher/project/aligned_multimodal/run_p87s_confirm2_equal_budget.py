from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent
RUNS = PROJECT_DIR / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the formal H2 pipeline, then add the equal-budget structured "
            "control and build the four-branch causal comparison."
        )
    )
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def wait_read_only(pid: int, poll_seconds: float) -> None:
    if pid <= 0:
        return
    while psutil.pid_exists(pid):
        try:
            process = psutil.Process(pid)
            create_time = process.create_time()
            name = process.name()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
        print(
            json.dumps(
                {
                    "waiting_for_pid": pid,
                    "name": name,
                    "create_time": create_time,
                    "poll_seconds": poll_seconds,
                    "action": "read-only wait; no signal is sent",
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"The formal H2 pipeline ended without required artifact: {path}"
        )


def require_complete_adaptation(path: Path, expected_rows: int) -> bool:
    summary_path = path / "summary.json"
    audit_path = path / "decoder_audit.json"
    if not summary_path.is_file() or not audit_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return (
        summary.get("status") == "formal"
        and int(summary.get("pseudo_rows", -1)) == expected_rows
        and int(summary.get("adapted_metrics", {}).get("total", -1)) == expected_rows
        and int(audit.get("rows", -1)) == expected_rows
    )


def main() -> None:
    args = parse_args()
    wait_read_only(args.wait_pid, args.poll_seconds)
    python = sys.executable
    targets = RUNS / "p87s_confirm2_structured_targets_v1/structured_targets.npz"
    c0 = RUNS / "p87s_fusion_confirm2_c0_v1"
    c1 = RUNS / "p87s_fusion_confirm2_c1_emission_v1"
    equal = RUNS / "p87s_fusion_confirm2_c2_structured4_v1"
    best = RUNS / "p87s_fusion_confirm2_c2_structured12_v1"
    for path in (
        targets,
        targets.parent / "summary.json",
        c0 / "unified_student.pt",
        c1 / "decoder_audit.json",
        best / "decoder_audit.json",
    ):
        require(path)
    expected_rows = 834
    if not require_complete_adaptation(c1, expected_rows):
        raise RuntimeError("formal H2 C1 is incomplete or has the wrong row universe")
    if not require_complete_adaptation(best, expected_rows):
        raise RuntimeError("formal H2 structured-12 is incomplete or has the wrong row universe")
    if not require_complete_adaptation(equal, expected_rows):
        run(
            [
                python,
                str(PROJECT_DIR / "adapt_p87s_structured_student.py"),
                "--base-checkpoint",
                str(c0 / "unified_student.pt"),
                "--structured-targets",
                str(targets),
                "--target",
                "structured",
                "--epochs",
                "4",
                "--output-dir",
                str(equal),
            ]
        )
        run(
            [
                python,
                str(PROJECT_DIR / "audit_p87s_student_decoder.py"),
                "--run-dir",
                str(equal),
                "--structured-targets",
                str(targets),
                "--target-summary",
                str(targets.parent / "summary.json"),
            ]
        )
    comparison = RUNS / "p87s_confirm2_ablation_v1"
    run(
        [
            python,
            str(PROJECT_DIR / "compare_p87s_student_ablation.py"),
            "--c0",
            str(c0),
            "--c1",
            str(c1),
            "--c2",
            str(equal),
            "--c2-best",
            str(best),
            "--output-dir",
            str(comparison),
        ]
    )
    print(json.dumps({"status": "complete", "comparison": str(comparison)}), flush=True)


if __name__ == "__main__":
    main()
