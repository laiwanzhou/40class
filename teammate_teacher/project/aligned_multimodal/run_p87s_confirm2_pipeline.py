from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent
RUNS = PROJECT_DIR / "runs"
HOLDOUT_USERS = ("user5", "user7", "user16", "user18", "user19")
TARGETS = RUNS / "p87s_confirm2_structured_targets_v1/structured_targets.npz"


def other_python_processes() -> list[dict[str, object]]:
    current = psutil.Process().pid
    result = []
    for process in psutil.process_iter(("pid", "name", "create_time")):
        try:
            name = str(process.info["name"] or "").lower()
            if process.info["pid"] != current and name in {"python.exe", "pythonw.exe"}:
                result.append(
                    {
                        "pid": int(process.info["pid"]),
                        "create_time": float(process.info["create_time"]),
                    }
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return result


def wait_for_python_idle(poll_seconds: float = 30.0) -> None:
    while running := other_python_processes():
        print(
            json.dumps(
                {
                    "waiting_for_other_python_processes": running,
                    "poll_seconds": poll_seconds,
                    "action": "read-only wait; no signals are sent",
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main() -> None:
    wait_for_python_idle()
    python = sys.executable
    visual = RUNS / "p87s_visual_confirm2_v1"
    # The earlier launch was stopped before any epoch/checkpoint. It produced only
    # empty console logs, so the standard trainer safely starts this formal run anew.
    run(
        [
            python,
            str(PROJECT_DIR / "train_p86_visual_pixel_oof.py"),
            "--mode",
            "hybrid",
            "--backbone",
            "mc3_18_temporal",
            "--freeze-through",
            "layer2",
            "--frames",
            "16",
            "--input-resolution",
            "160",
            "--pixel-cache",
            str(RUNS / "p86_visual_pixel_cache_t16_r160_v12"),
            "--head-learning-rate",
            "0.0002",
            "--backbone-learning-rate",
            "0.00001",
            "--minimum-learning-rate",
            "0.00001",
            "--weight-decay",
            "0.08",
            "--distillation-weight",
            "1.0",
            "--relation-weight",
            "0.2",
            "--feature-weight",
            "0.5",
            "--label-smoothing",
            "0.1",
            "--augmentation-mode",
            "subject_robust",
            "--batch-size",
            "4",
            "--gradient-accumulation",
            "4",
            "--subject-holdout-users",
            *HOLDOUT_USERS,
            "--subject-holdout-fixed-epochs",
            "16",
            "--output-dir",
            str(visual),
        ]
    )
    sequence = RUNS / "p87s_mc3_sequence_confirm2_v1"
    run(
        [
            python,
            str(PROJECT_DIR / "build_p86_mc3_sequence_cache.py"),
            "--checkpoint",
            str(visual / "visual_student.pt"),
            "--pixel-cache",
            str(RUNS / "p86_visual_pixel_cache_t16_r160_v12"),
            "--output-dir",
            str(sequence),
        ]
    )
    motion = RUNS / "p87s_mobind_confirm2_v1"
    run(
        [
            python,
            str(PROJECT_DIR / "train_p86_mobind_pretrain.py"),
            "--output-dir",
            str(motion),
            "--motion-cache",
            str(RUNS / "p86_motion_window_cache_t16_v1"),
            "--imu-teacher-logits",
            str(RUNS / "p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"),
            "--imu-teacher-weight",
            "1.0",
            "--subject-holdout-users",
            *HOLDOUT_USERS,
        ]
    )
    c0 = RUNS / "p87s_fusion_confirm2_c0_v1"
    run(
        [
            python,
            str(PROJECT_DIR / "train_p86_mobind_fusion_proxy.py"),
            "--visual-checkpoint",
            str(visual / "visual_student.pt"),
            "--sequence-cache",
            str(sequence),
            "--pretrain-checkpoint",
            str(motion / "mobind_lite.pt"),
            "--motion-cache",
            str(RUNS / "p86_motion_window_cache_t16_v1"),
            "--pixel-cache",
            str(RUNS / "p86_visual_pixel_cache_t16_r160_v12"),
            "--output-dir",
            str(c0),
            "--modality",
            "separate",
            "--stage-a-epochs",
            "4",
            "--stage-b-epochs",
            "20",
            "--joint-freeze-pretrained-encoders",
            "--distillation-weight",
            "1.0",
            "--selective-anchor-weight",
            "0.3",
            "--reliability-weight",
            "0.0",
            "--visual-corruption-probability",
            "0.75",
            "--visual-feature-dropout",
            "0.3",
            "--visual-view-dropout",
            "0.4",
            "--subject-holdout-users",
            *HOLDOUT_USERS,
        ]
    )
    run(
        [python, str(PROJECT_DIR / "audit_p87s_student_decoder.py"), "--run-dir", str(c0),
         "--structured-targets", str(TARGETS),
         "--target-summary", str(TARGETS.parent / "summary.json")]
    )
    branches = (
        ("emission", 4, RUNS / "p87s_fusion_confirm2_c1_emission_v1"),
        ("structured", 12, RUNS / "p87s_fusion_confirm2_c2_structured12_v1"),
    )
    for target, epochs, output in branches:
        run(
            [
                python,
                str(PROJECT_DIR / "adapt_p87s_structured_student.py"),
                "--base-checkpoint",
                str(c0 / "unified_student.pt"),
                "--structured-targets",
                str(TARGETS),
                "--target",
                target,
                "--epochs",
                str(epochs),
                "--output-dir",
                str(output),
            ]
        )
        run(
            [
                python,
                str(PROJECT_DIR / "audit_p87s_student_decoder.py"),
                "--run-dir",
                str(output),
                "--structured-targets",
                str(TARGETS),
                "--target-summary",
                str(TARGETS.parent / "summary.json"),
            ]
        )
    print(
        json.dumps(
            {"status": "complete", "c0": str(c0), "c1": str(branches[0][2]), "c2": str(branches[1][2])},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
