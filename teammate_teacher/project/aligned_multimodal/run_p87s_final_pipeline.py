from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent
RUNS = PROJECT_DIR / "runs"


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def stage_complete(run_dir: Path, stage: str, artifact: str) -> bool:
    summary = read_json(run_dir / "summary.json")
    return bool(
        summary
        and summary.get("stage") == stage
        and (run_dir / artifact).is_file()
    )


def cache_complete(
    run_dir: Path,
    stage: str,
    expected_rows: int,
    count_key: str = "completed",
) -> bool:
    summary = read_json(run_dir / "summary.json")
    return bool(
        summary
        and summary.get("stage") == stage
        and int(summary.get(count_key, -1)) == expected_rows
    )


def assert_frozen_evidence() -> None:
    h1 = read_json(RUNS / "p87s_fusion_holdout1_c7_structured12_v1/summary.json")
    h2 = read_json(RUNS / "p87s_fusion_confirm2_c2_structured12_v1/summary.json")
    comparison = read_json(RUNS / "p87s_confirm2_ablation_v1/summary.json")
    targets = read_json(RUNS / "p87s_test_structured_targets_v1/summary.json")
    if not all((h1, h2, comparison, targets)):
        raise RuntimeError("frozen H1/H2/Test-target evidence is incomplete")
    if int(h1["adapted_metrics"]["total"]) != 663:
        raise RuntimeError("H1 result universe changed")
    if int(h2["adapted_metrics"]["total"]) != 834:
        raise RuntimeError("H2 result universe changed")
    if float(h2["adapted_metrics"]["accuracy"]) < 0.80:
        raise RuntimeError("H2 did not clear the frozen finalization gate")
    branches = comparison["branches"]
    if (
        float(branches["C2_structured"]["raw_accuracy"])
        <= float(branches["C1_emission"]["raw_accuracy"])
    ):
        raise RuntimeError("equal-budget structured target did not beat emission")
    if int(targets["test_target_rows"]) != 401:
        raise RuntimeError("final structured Test target universe changed")
    print(
        json.dumps(
            {
                "frozen_recipe_gate": "passed",
                "h1_structured12_raw": h1["adapted_metrics"]["accuracy"],
                "h2_structured12_raw": h2["adapted_metrics"]["accuracy"],
                "h2_structured12_decoded": branches["C2_structured_best"][
                    "decoded_accuracy"
                ],
                "test_pseudo_rows": targets["test_target_rows"],
            }
        ),
        flush=True,
    )


def main() -> None:
    assert_frozen_evidence()
    python = sys.executable
    train_pixels = RUNS / "p86_visual_pixel_cache_t16_r160_v12"
    train_motion_window = RUNS / "p86_motion_window_cache_t16_v1"
    visual = RUNS / "p87s_visual_all2914_v1"
    train_sequence = RUNS / "p87s_mc3_sequence_all2914_v1"
    motion = RUNS / "p87s_mobind_all2914_v1"
    fusion = RUNS / "p87s_fusion_all2914_v1"
    test_pixels = RUNS / "p87s_test_pixel_cache_t16_r160_v1"
    test_motion_source = RUNS / "p87s_test_motion_source_v1"
    test_motion_window = RUNS / "p87s_test_motion_window_t16_v1"
    test_sequence = RUNS / "p87s_test_mc3_sequence_v1"
    adapted = RUNS / "p87s_test_adapt_structured12_v1"
    decoder = RUNS / "p87s_tiny_decoder_v1"
    predictions = RUNS / "p87s_final_test_predictions_v1"
    equivalence = RUNS / "p87s_deployment_equivalence_v1"
    package_audit = RUNS / "p87s_final_package_audit_v1"

    if not cache_complete(
        test_pixels, "P87S_label_free_test_pixel_cache", 405, count_key="trials"
    ):
        run(
            [
                python,
                str(PROJECT_DIR / "build_p87s_test_pixel_cache.py"),
                "--output-dir",
                str(test_pixels),
            ]
        )
    if not cache_complete(
        test_motion_source,
        "P87S_label_free_test_motion_source_cache",
        405,
        count_key="completed_trials",
    ):
        run(
            [
                python,
                str(PROJECT_DIR / "build_p87s_test_motion_source_cache.py"),
                "--output-dir",
                str(test_motion_source),
            ]
        )
    if not cache_complete(test_motion_window, "P86_exact_visual_grid_motion_cache", 405):
        run(
            [
                python,
                str(PROJECT_DIR / "build_p86_motion_window_cache.py"),
                "--pixel-cache",
                str(test_pixels),
                "--p31-run",
                str(test_motion_source),
                "--output-dir",
                str(test_motion_window),
            ]
        )

    if not stage_complete(visual, "P87S_visual_all2914_refit", "visual_student.pt"):
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
                str(train_pixels),
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
                "--all-label-fixed-epochs",
                "16",
                "--output-dir",
                str(visual),
            ]
        )
    if not cache_complete(
        train_sequence, "P86_MC3_layer4_sequence_cache", 2914
    ):
        run(
            [
                python,
                str(PROJECT_DIR / "build_p86_mc3_sequence_cache.py"),
                "--checkpoint",
                str(visual / "visual_student.pt"),
                "--pixel-cache",
                str(train_pixels),
                "--output-dir",
                str(train_sequence),
            ]
        )
    if not cache_complete(
        test_sequence,
        "P87S_label_free_test_MC3_sequence_cache",
        405,
    ):
        run(
            [
                python,
                str(PROJECT_DIR / "build_p87s_test_sequence_cache.py"),
                "--checkpoint",
                str(visual / "visual_student.pt"),
                "--pixel-cache",
                str(test_pixels),
                "--output-dir",
                str(test_sequence),
            ]
        )

    if not stage_complete(motion, "P87S_mobind_all2914_refit", "mobind_lite.pt"):
        run(
            [
                python,
                str(PROJECT_DIR / "train_p86_mobind_pretrain.py"),
                "--output-dir",
                str(motion),
                "--motion-cache",
                str(train_motion_window),
                "--imu-teacher-logits",
                str(RUNS / "p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz"),
                "--imu-teacher-weight",
                "1.0",
                "--epochs",
                "24",
                "--all-label-refit",
            ]
        )
    if not stage_complete(fusion, "P87S_mobind_fusion_all2914_refit", "unified_student.pt"):
        run(
            [
                python,
                str(PROJECT_DIR / "train_p86_mobind_fusion_proxy.py"),
                "--visual-checkpoint",
                str(visual / "visual_student.pt"),
                "--sequence-cache",
                str(train_sequence),
                "--pretrain-checkpoint",
                str(motion / "mobind_lite.pt"),
                "--motion-cache",
                str(train_motion_window),
                "--pixel-cache",
                str(train_pixels),
                "--output-dir",
                str(fusion),
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
                "--all-label-refit",
            ]
        )
    if not stage_complete(
        adapted, "P87S_label_free_test_adaptation", "unified_student.pt"
    ):
        run(
            [
                python,
                str(PROJECT_DIR / "adapt_p87s_test_student.py"),
                "--base-checkpoint",
                str(fusion / "unified_student.pt"),
                "--structured-targets",
                str(RUNS / "p87s_test_structured_targets_v1/structured_targets.npz"),
                "--sequence-cache",
                str(test_sequence),
                "--motion-cache",
                str(test_motion_window),
                "--pixel-cache",
                str(test_pixels),
                "--epochs",
                "12",
                "--output-dir",
                str(adapted),
            ]
        )
    if not stage_complete(decoder, "P87S_frozen_tiny_decoder", "tiny_decoder.npz"):
        run(
            [
                python,
                str(PROJECT_DIR / "freeze_p87s_tiny_decoder.py"),
                "--output-dir",
                str(decoder),
            ]
        )
    if not stage_complete(
        predictions,
        "P87S_final_student_test_inference",
        "submission_p87s_student_decoded.csv",
    ):
        run(
            [
                python,
                str(PROJECT_DIR / "predict_p87s_test_student.py"),
                "--checkpoint",
                str(adapted / "unified_student.pt"),
                "--sequence-cache",
                str(test_sequence),
                "--motion-cache",
                str(test_motion_window),
                "--pixel-cache",
                str(test_pixels),
                "--tiny-decoder",
                str(decoder / "tiny_decoder.npz"),
                "--structured-targets",
                str(
                    RUNS
                    / "p87s_test_structured_targets_v1/structured_targets.npz"
                ),
                "--output-dir",
                str(predictions),
            ]
        )
    if not stage_complete(
        equivalence,
        "P87S_raw_vs_cached_deployment_equivalence",
        "summary.json",
    ):
        run(
            [
                python,
                str(PROJECT_DIR / "verify_p87s_deployment_equivalence.py"),
                "--checkpoint",
                str(adapted / "unified_student.pt"),
                "--sequence-cache",
                str(test_sequence),
                "--motion-cache",
                str(test_motion_window),
                "--pixel-cache",
                str(test_pixels),
                "--output-dir",
                str(equivalence),
            ]
        )
    if not stage_complete(
        package_audit,
        "P87S_final_package_audit",
        "summary.json",
    ):
        run(
            [
                python,
                str(PROJECT_DIR / "audit_p87s_final_package.py"),
                "--checkpoint",
                str(adapted / "unified_student.pt"),
                "--tiny-decoder",
                str(decoder / "tiny_decoder.npz"),
                "--predictions-dir",
                str(predictions),
                "--equivalence-summary",
                str(equivalence / "summary.json"),
                "--output-dir",
                str(package_audit),
            ]
        )
    with np.load(
        RUNS / "p87s_test_structured_targets_v1/structured_targets.npz",
        allow_pickle=False,
    ) as targets:
        if int(targets["target_mask"].sum()) != 401:
            raise RuntimeError("final target mask changed after inference")
    final = read_json(predictions / "summary.json")
    if not final or int(final.get("test_rows", -1)) != 405:
        raise RuntimeError("final P87-S inference did not cover 405 Test rows")
    equivalent = read_json(equivalence / "summary.json")
    if not equivalent or equivalent.get("status") != "passed":
        raise RuntimeError("raw-input deployment equivalence was not established")
    package = read_json(package_audit / "summary.json")
    if not package or package.get("status") != "passed":
        raise RuntimeError("final <=100 MB package audit did not pass")
    print(
        json.dumps(
            {
                "status": "complete",
                "final": final,
                "equivalence": equivalent,
                "package": package,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
