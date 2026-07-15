from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("imu", "skeleton", "radar", "ir", "thermal", "depth_color")
REPORT = PROJECT_ROOT / "reports" / "task03_smoke_summary_optimized.json"


def extract_result(stdout: str) -> dict[str, object]:
    lines = [line for line in stdout.splitlines() if line.startswith("RESULT_JSON=")]
    if not lines:
        raise ValueError("Training command did not emit RESULT_JSON.")
    return json.loads(lines[-1].split("=", 1)[1])


def main() -> None:
    summary: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "device": "cuda",
        "modalities": {},
    }
    for modality in MODALITIES:
        config_path = PROJECT_ROOT / "configs" / "task03" / f"{modality}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_id = f"optimized_smoke_{datetime.now():%Y%m%d_%H%M%S}_{modality}"
        command = [
            sys.executable,
            "-m",
            "src.train_unimodal",
            "--config",
            str(config_path),
            "--output-root",
            str(PROJECT_ROOT / "outputs" / "task03_optimized_smoke"),
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
        completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, encoding="utf-8")
        print(completed.stdout)
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            summary["modalities"][modality] = {  # type: ignore[index]
                "status": "failed",
                "num_workers": int(config["num_workers"]),
                "batch_size": int(config["batch_size"]),
                "error": completed.stderr[-4000:],
            }
            REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(completed.returncode)
        result = extract_result(completed.stdout)
        summary["modalities"][modality] = {  # type: ignore[index]
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
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All six optimized smoke tests passed: {REPORT}")


if __name__ == "__main__":
    main()
