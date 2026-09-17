"""Reproducible P87-S H3 baseline/routed distillation comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RUNS = HERE / "runs"
HOLDOUT_USERS = ("user20", "user22", "user24", "user3", "user4", "user9")
BASE_TARGETS = RUNS / "p90_p87s_h3_structured_targets_v1/structured_targets.npz"
ROUTED_TARGETS = RUNS / "p90_p87s_routed_targets_h3_v1/structured_targets.npz"


def run(command: list[str], required_output: Path) -> None:
    if required_output.exists():
        print(json.dumps({"skip_existing": str(required_output)}), flush=True)
        return
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not required_output.exists():
        raise RuntimeError(f"Command completed without expected output {required_output}")


def audit(run_dir: Path, targets: Path) -> None:
    run(
        [
            sys.executable,
            str(HERE / "audit_p87s_student_decoder.py"),
            "--run-dir",
            str(run_dir),
            "--structured-targets",
            str(targets),
            "--target-summary",
            str(BASE_TARGETS.parent / "summary.json"),
        ],
        run_dir / "decoder_audit.json",
    )


def main() -> None:
    python = sys.executable
    visual = RUNS / "p90_p87s_visual_h3_v1"
    run(
        [
            python,
            str(HERE / "train_p86_visual_pixel_oof.py"),
            "--mode", "hybrid",
            "--backbone", "mc3_18_temporal",
            "--freeze-through", "layer2",
            "--frames", "16",
            "--input-resolution", "160",
            "--pixel-cache", str(RUNS / "p86_visual_pixel_cache_t16_r160_v12"),
            "--head-learning-rate", "0.0002",
            "--backbone-learning-rate", "0.00001",
            "--minimum-learning-rate", "0.00001",
            "--weight-decay", "0.08",
            "--distillation-weight", "1.0",
            "--relation-weight", "0.2",
            "--feature-weight", "0.5",
            "--label-smoothing", "0.1",
            "--augmentation-mode", "subject_robust",
            "--batch-size", "4",
            "--gradient-accumulation", "4",
            "--subject-holdout-users", *HOLDOUT_USERS,
            "--subject-holdout-fixed-epochs", "16",
            "--output-dir", str(visual),
        ],
        visual / "visual_student.pt",
    )

    sequence = RUNS / "p90_p87s_mc3_sequence_h3_v1"
    run(
        [
            python,
            str(HERE / "build_p86_mc3_sequence_cache.py"),
            "--checkpoint", str(visual / "visual_student.pt"),
            "--pixel-cache", str(RUNS / "p86_visual_pixel_cache_t16_r160_v12"),
            "--output-dir", str(sequence),
        ],
        sequence / "summary.json",
    )

    motion = RUNS / "p90_p87s_mobind_h3_v1"
    run(
        [
            python,
            str(HERE / "train_p86_mobind_pretrain.py"),
            "--output-dir", str(motion),
            "--motion-cache", str(RUNS / "p86_motion_window_cache_t16_v1"),
            "--imu-teacher-logits",
            str(RUNS / "p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"),
            "--imu-teacher-weight", "1.0",
            "--subject-holdout-users", *HOLDOUT_USERS,
        ],
        motion / "mobind_lite.pt",
    )

    c0 = RUNS / "p90_p87s_fusion_h3_c0_v1"
    run(
        [
            python,
            str(HERE / "train_p86_mobind_fusion_proxy.py"),
            "--visual-checkpoint", str(visual / "visual_student.pt"),
            "--sequence-cache", str(sequence),
            "--pretrain-checkpoint", str(motion / "mobind_lite.pt"),
            "--motion-cache", str(RUNS / "p86_motion_window_cache_t16_v1"),
            "--pixel-cache", str(RUNS / "p86_visual_pixel_cache_t16_r160_v12"),
            "--output-dir", str(c0),
            "--modality", "separate",
            "--stage-a-epochs", "4",
            "--stage-b-epochs", "20",
            "--joint-freeze-pretrained-encoders",
            "--distillation-weight", "1.0",
            "--selective-anchor-weight", "0.3",
            "--reliability-weight", "0.0",
            "--visual-corruption-probability", "0.75",
            "--visual-feature-dropout", "0.3",
            "--visual-view-dropout", "0.4",
            "--subject-holdout-users", *HOLDOUT_USERS,
        ],
        c0 / "unified_student.pt",
    )
    audit(c0, BASE_TARGETS)

    comparisons = (
        ("baseline", BASE_TARGETS, RUNS / "p90_p87s_h3_structured12_baseline_v1"),
        ("routed", ROUTED_TARGETS, RUNS / "p90_p87s_routed_h3_structured12_v1"),
    )
    for name, targets, output in comparisons:
        run(
            [
                python,
                str(HERE / "adapt_p87s_structured_student.py"),
                "--base-checkpoint", str(c0 / "unified_student.pt"),
                "--structured-targets", str(targets),
                "--target", "structured",
                "--epochs", "12",
                "--output-dir", str(output),
            ],
            output / "unified_student.pt",
        )
        audit(output, targets)
        print(json.dumps({"completed_branch": name, "output": str(output)}), flush=True)

    print(
        json.dumps(
            {
                "status": "complete",
                "c0": str(c0),
                "baseline": str(comparisons[0][2]),
                "routed": str(comparisons[1][2]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
