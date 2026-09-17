from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent
RUNS = PROJECT_DIR / "runs"
HOLDOUT_USERS = ("user6", "user8", "user17", "user23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe sequential P87-S pipeline after the fresh visual holdout "
            "checkpoint completes. Never runs two training stages concurrently."
        )
    )
    parser.add_argument(
        "--visual-dir", type=Path, default=RUNS / "p87s_visual_holdout1_v1"
    )
    parser.add_argument(
        "--wait-for-visual",
        action="store_true",
        help="Poll the existing visual run and begin only after its formal summary appears.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def is_formal(run_dir: Path, expected_stage: str) -> bool:
    summary = read_summary(run_dir / "summary.json")
    return bool(
        summary
        and summary.get("stage") == expected_stage
        and str(summary.get("status", "")).startswith("formal")
    )


def main() -> None:
    args = parse_args()
    visual_dir = args.visual_dir.resolve()
    visual_summary = read_summary(visual_dir / "summary.json")
    while args.wait_for_visual and (
        not visual_summary or visual_summary.get("status") != "formal"
    ):
        if args.poll_seconds < 5:
            raise ValueError("poll-seconds must be at least 5")
        print(
            json.dumps(
                {"waiting_for_visual": str(visual_dir), "poll_seconds": args.poll_seconds}
            ),
            flush=True,
        )
        time.sleep(args.poll_seconds)
        visual_summary = read_summary(visual_dir / "summary.json")
    if not visual_summary or visual_summary.get("status") != "formal":
        raise RuntimeError(
            f"Fresh visual subject-holdout run is not complete: {visual_dir}"
        )
    if visual_summary.get("counts") != {"train": 2251, "holdout": 663}:
        raise RuntimeError("Visual P87-S split counts differ from the frozen protocol")
    if sorted(visual_summary.get("holdout_subjects", [])) != sorted(HOLDOUT_USERS):
        raise RuntimeError("Visual P87-S holdout subjects differ from the frozen protocol")

    python = sys.executable
    sequence_dir = RUNS / "p87s_mc3_sequence_holdout1_v1"
    sequence_summary = read_summary(sequence_dir / "summary.json")
    if not sequence_summary or int(sequence_summary.get("completed", 0)) != 2914:
        run(
            [
                python,
                str(PROJECT_DIR / "build_p86_mc3_sequence_cache.py"),
                "--checkpoint",
                str(visual_dir / "visual_student.pt"),
                "--pixel-cache",
                str(RUNS / "p86_visual_pixel_cache_t16_r160_v12"),
                "--output-dir",
                str(sequence_dir),
            ]
        )

    motion_dir = RUNS / "p87s_mobind_holdout1_v1"
    if not is_formal(motion_dir, "P87S_mobind_subject_holdout"):
        run(
            [
                python,
                str(PROJECT_DIR / "train_p86_mobind_pretrain.py"),
                "--output-dir",
                str(motion_dir),
                "--motion-cache",
                str(RUNS / "p86_motion_window_cache_t16_v1"),
                "--imu-teacher-logits",
                str(
                    RUNS
                    / "p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"
                ),
                "--imu-teacher-weight",
                "1.0",
                "--subject-holdout-users",
                *HOLDOUT_USERS,
            ]
        )

    c0_dir = RUNS / "p87s_fusion_holdout1_c0_v1"
    if not is_formal(c0_dir, "P87S_mobind_fusion_subject_holdout"):
        run(
            [
                python,
                str(PROJECT_DIR / "train_p86_mobind_fusion_proxy.py"),
                "--visual-checkpoint",
                str(visual_dir / "visual_student.pt"),
                "--sequence-cache",
                str(sequence_dir),
                "--pretrain-checkpoint",
                str(motion_dir / "mobind_lite.pt"),
                "--motion-cache",
                str(RUNS / "p86_motion_window_cache_t16_v1"),
                "--pixel-cache",
                str(RUNS / "p86_visual_pixel_cache_t16_r160_v12"),
                "--output-dir",
                str(c0_dir),
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
        [
            python,
            str(PROJECT_DIR / "audit_p87s_student_decoder.py"),
            "--run-dir",
            str(c0_dir),
        ]
    )

    branches = (
        ("emission", RUNS / "p87s_fusion_holdout1_c1_emission_v1"),
        ("structured", RUNS / "p87s_fusion_holdout1_c2_structured_v1"),
    )
    for target, branch_dir in branches:
        if not is_formal(branch_dir, "P87S_label_free_adaptation"):
            run(
                [
                    python,
                    str(PROJECT_DIR / "adapt_p87s_structured_student.py"),
                    "--base-checkpoint",
                    str(c0_dir / "unified_student.pt"),
                    "--target",
                    target,
                    "--output-dir",
                    str(branch_dir),
                ]
            )
        run(
            [
                python,
                str(PROJECT_DIR / "audit_p87s_student_decoder.py"),
                "--run-dir",
                str(branch_dir),
            ]
        )

    comparison_dir = RUNS / "p87s_holdout1_ablation_v1"
    run(
        [
            python,
            str(PROJECT_DIR / "compare_p87s_student_ablation.py"),
            "--c0",
            str(c0_dir),
            "--c1",
            str(branches[0][1]),
            "--c2",
            str(branches[1][1]),
            "--output-dir",
            str(comparison_dir),
        ]
    )
    summary = {
        "status": "complete",
        "visual": str(visual_dir),
        "sequence": str(sequence_dir),
        "motion": str(motion_dir),
        "c0": str(c0_dir),
        "c1": str(branches[0][1]),
        "c2": str(branches[1][1]),
        "comparison": str(comparison_dir),
    }
    (comparison_dir / "pipeline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
