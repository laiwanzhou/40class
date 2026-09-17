"""P410 with only the three severe nonvisual Test fallbacks replaced exactly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p399_candidate_conditioned_raw_detail_head_oof as p399
import p410_target_aware_weighted_raw_detail_oof as p410
from p412_matched_nonvisual_target_aware_oof import align, softmax


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p413_minimal_matched_replacement_target_aware_oof_v1"
ORIGINAL_LOAD = p399.load_data
REPLACEMENTS = {
    1: (HERE / "runs/p86_imu_composite_teacher_v1/composite_imu_teacher.npz", "imu_logits", True, "matched_composite_imu"),
    4: (HERE / "runs/p89_imu_sensor_attention_expert_v1/oof_probabilities.npz", "equal_probability", False, "matched_sensor_attention_imu"),
    6: (HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz", "skeleton_logits", True, "matched_skeleton_invariant"),
}


def minimally_matched_load():
    parts = ORIGINAL_LOAD()
    archives = {}
    for index, (path, key, is_logits, name) in REPLACEMENTS.items():
        archive = np.load(path)
        values = softmax(archive[key]) if is_logits else np.asarray(archive[key], dtype=np.float32)
        archives[index] = (archive["sample_ids"].astype(str), values)
    for cohort in p399.COHORTS:
        part = parts[cohort]
        probability = part["nonvisual_probability"].copy()
        for index, (ids, values) in archives.items():
            probability[:, index, :] = align(values, ids, part["ids"])
        part["nonvisual_probability"] = probability
    return parts


def main():
    print(
        "P413 reruns P410 after replacing only the three nonvisual teachers whose "
        "source/Test domain separability exceeded 87 percent.",
        flush=True,
    )
    p399.load_data = minimally_matched_load
    p410.OUT = OUT
    p410.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P413_minimal_matched_replacement_target_aware_OOF"
    report["protocol"]["replacements"] = {
        str(index): value[3] for index, value in REPLACEMENTS.items()
    }
    report["protocol"]["unchanged_nonvisual_teacher_count"] = 9
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: P410 with only severe fallback slots 1/4/6 replaced by exact matched experts.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
