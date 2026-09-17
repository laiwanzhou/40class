"""P389 directed branches after balancing nonvisual teachers into five families."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p386_visual_scope_nonvisual_competence_gate as p386
import p389_nonvisual_teacher_class_branches_oof as p389


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p390_nonvisual_family_class_branches_oof_v1"
ORIGINAL_TEACHERS = tuple(p386.NONVISUAL_TEACHERS)
ORIGINAL_LOAD = p386.load_data
FAMILIES = {
    "thermal_family": ("p12_thermal_candidate", "expanded_thermal"),
    "imu_family": ("p90_deep_imu", "expanded_p12_imu"),
    "skeleton_family": (
        "p90_motionbert_3view",
        "expanded_motionbert_front",
        "expanded_skeleton_invariant",
    ),
    "session_family": (
        "a18_best_session",
        "p87_sequence",
        "p88_repeat",
        "p88_latent_prefix",
    ),
    "multimodal_family": ("p128_hierarchical_multimodal",),
}


def load_family_data():
    p386.NONVISUAL_TEACHERS = ORIGINAL_TEACHERS
    parts = ORIGINAL_LOAD()
    lookup = {name: index for index, name in enumerate(ORIGINAL_TEACHERS)}
    for cohort in p389.COHORTS:
        original = parts[cohort]["nonvisual_probability"]
        parts[cohort]["nonvisual_probability"] = np.stack(
            [original[:, [lookup[name] for name in members], :].mean(axis=1) for members in FAMILIES.values()],
            axis=1,
        ).astype(np.float32)
    return parts


def main():
    print(
        "P390 learns directed family-by-class branches after averaging correlated "
        "Thermal/IMU/Skeleton/Session teachers within each modality family.",
        flush=True,
    )
    p389.OUT = OUT
    p389.NONVISUAL_TEACHERS = tuple(FAMILIES)
    p389.load_data = load_family_data
    p389.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P390_nonvisual_family_class_branches_OOF"
    report["protocol"]["nonvisual_families"] = {key: list(value) for key, value in FAMILIES.items()}
    report["protocol"]["rule_unit"] = "nonvisual modality family x target class"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: P389 directed branches after balancing correlated teachers into five families.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
