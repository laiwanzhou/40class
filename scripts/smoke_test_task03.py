from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("imu", "skeleton", "radar", "ir", "thermal", "depth_color")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task03_smoke"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "task03_smoke_summary_latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current six-modality Task 03 smoke suite."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Temporary smoke output root (default: outputs/task03_smoke).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Latest smoke summary path (default: reports/task03_smoke_summary_latest.json).",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep successful smoke run directories instead of removing them.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def extract_result(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith("RESULT_JSON=")]
    if not lines:
        raise ValueError("Training command did not emit RESULT_JSON.")
    result = json.loads(lines[-1].split("=", 1)[1])
    if not isinstance(result, dict):
        raise ValueError("RESULT_JSON must contain an object.")
    return result


def safe_remove_run(run_dir: Path, output_root: Path, run_prefix: str) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_run.parent.parent != resolved_root:
        raise ValueError(f"Refusing to remove run outside smoke output root: {resolved_run}")
    if not resolved_run.name.startswith(run_prefix):
        raise ValueError(f"Refusing to remove run with unexpected name: {resolved_run}")
    if resolved_run.is_dir():
        shutil.rmtree(resolved_run)


def prune_new_empty_directories(
    output_root: Path,
    modality_existed: dict[str, bool],
    output_root_existed: bool,
) -> None:
    for modality, existed in modality_existed.items():
        modality_dir = output_root / modality
        if not existed and modality_dir.is_dir() and not any(modality_dir.iterdir()):
            modality_dir.rmdir()
    if not output_root_existed and output_root.is_dir() and not any(output_root.iterdir()):
        output_root.rmdir()


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_root = project_path(args.output_root)
    report_path = project_path(args.report_path)
    output_root_existed = output_root.exists()
    modality_existed = {
        modality: (output_root / modality).exists() for modality in MODALITIES
    }
    session_prefix = f"smoke_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"
    successful_runs: list[Path] = []
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "device": "cuda",
        "output_root": str(output_root),
        "keep_outputs": bool(args.keep_outputs),
        "modalities": {},
    }

    for modality in MODALITIES:
        config_path = PROJECT_ROOT / "configs" / "task03" / f"{modality}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_id = f"{session_prefix}_{modality}"
        expected_run = output_root / modality / run_id
        if expected_run.exists():
            raise FileExistsError(f"Unique smoke run already exists: {expected_run}")
        command = [
            sys.executable,
            "-m",
            "src.train_unimodal",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--smoke-test",
            "--max-epochs",
            "1",
            "--max-train-batches",
            "5",
            "--max-val-batches",
            "3",
            "--run-id",
            run_id,
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        print(completed.stdout)
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            summary["modalities"][modality] = {
                "status": "failed",
                "num_workers": int(config["num_workers"]),
                "batch_size": int(config["batch_size"]),
                "error": completed.stderr[-4000:],
                "output_dir": str(expected_run) if expected_run.exists() else None,
            }
            summary["status"] = "failed"
            summary["retained_output_dirs"] = (
                [str(expected_run)] if expected_run.exists() else []
            )
            for run_dir in successful_runs:
                safe_remove_run(run_dir, output_root, session_prefix)
            prune_new_empty_directories(
                output_root, modality_existed, output_root_existed
            )
            write_report(report_path, summary)
            raise SystemExit(completed.returncode)

        try:
            result = extract_result(completed.stdout)
            result_run = Path(str(result["output_dir"])).resolve()
            if result_run != expected_run.resolve() or not result_run.is_dir():
                raise ValueError(
                    f"Smoke result returned unexpected output directory: {result_run}"
                )
            if result.get("status") != "passed":
                raise ValueError(f"Smoke result status is not passed: {result.get('status')}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            summary["modalities"][modality] = {
                "status": "failed",
                "num_workers": int(config["num_workers"]),
                "batch_size": int(config["batch_size"]),
                "error": str(exc),
                "output_dir": str(expected_run) if expected_run.exists() else None,
            }
            summary["status"] = "failed"
            summary["retained_output_dirs"] = (
                [str(expected_run)] if expected_run.exists() else []
            )
            for run_dir in successful_runs:
                safe_remove_run(run_dir, output_root, session_prefix)
            prune_new_empty_directories(
                output_root, modality_existed, output_root_existed
            )
            write_report(report_path, summary)
            raise
        successful_runs.append(result_run)
        summary["modalities"][modality] = {
            key: result[key]
            for key in (
                "status",
                "train_samples",
                "val_samples",
                "num_workers",
                "batch_size",
                "input_shape",
                "logits_shape",
                "embedding_shape",
                "loss",
                "gpu_memory_peak_mb",
                "gpu_memory_peak_reserved_mb",
                "checkpoint_size_mb",
                "output_dir",
            )
        }

    summary["status"] = "passed"
    summary["outputs_retained"] = bool(args.keep_outputs)
    if not args.keep_outputs:
        for run_dir in successful_runs:
            safe_remove_run(run_dir, output_root, session_prefix)
        prune_new_empty_directories(output_root, modality_existed, output_root_existed)
    write_report(report_path, summary)
    print(f"All six smoke tests passed: {report_path}")
    print("Successful smoke outputs retained." if args.keep_outputs else "Successful smoke outputs removed.")


if __name__ == "__main__":
    main()
