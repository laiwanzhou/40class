"""P386 with five balanced nonvisual modality families and Top-3 support."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p386_visual_scope_nonvisual_competence_gate as p386


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p388_visual_scope_nonvisual_family_top3_gate_v1"
ORIGINAL_LOAD = p386.load_data
ORIGINAL_TEACHERS = tuple(p386.NONVISUAL_TEACHERS)
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
    active_names = tuple(p386.NONVISUAL_TEACHERS)
    p386.NONVISUAL_TEACHERS = ORIGINAL_TEACHERS
    try:
        parts = ORIGINAL_LOAD()
    finally:
        p386.NONVISUAL_TEACHERS = active_names
    lookup = {name: index for index, name in enumerate(ORIGINAL_TEACHERS)}
    for cohort in p386.COHORTS:
        original = parts[cohort]["nonvisual_probability"]
        parts[cohort]["nonvisual_probability"] = np.stack(
            [original[:, [lookup[name] for name in members], :].mean(axis=1) for members in FAMILIES.values()],
            axis=1,
        ).astype(np.float32)
    return parts


def main():
    print(
        "P388 balances nonvisual evidence into Thermal/IMU/Skeleton/Session/Multimodal "
        "families before class-competence Top-3 arbitration.",
        flush=True,
    )
    p386.OUT = OUT
    p386.NONVISUAL_TEACHERS = tuple(FAMILIES)
    p386.NONVISUAL_SUPPORT_K = 3
    p386.VOTE_THRESHOLDS = (1, 2, 3, 4, 5)
    p386.load_data = load_family_data
    p386.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P388_visual_scope_nonvisual_family_Top3_gate"
    report["protocol"]["nonvisual_families"] = {key: list(value) for key, value in FAMILIES.items()}
    report["protocol"]["nonvisual_support_k"] = 3
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: balanced five-family nonvisual Top-3 competence gate behind visual scope.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
