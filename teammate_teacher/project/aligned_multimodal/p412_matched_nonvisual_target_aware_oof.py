"""P410 target-aware raw detail with only exact OOF/Test-matched nonvisual experts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p399_candidate_conditioned_raw_detail_head_oof as p399
import p410_target_aware_weighted_raw_detail_oof as p410


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "runs/p412_matched_nonvisual_target_aware_oof_v1"
ORIGINAL_LOAD = p399.load_data
ORIGINAL_NAMES = tuple(p399.NONVISUAL_TEACHERS)
MATCHED_NAMES = (
    "p12_thermal_candidate",
    "a18_best_session",
    "p87_sequence",
    "p88_repeat",
    "p128_hierarchical_multimodal",
    "motionbert_front",
    "imu_orientation",
    "imu_rocket",
    "imu_sensor_attention_equal",
    "imu_spectral",
    "composite_imu",
    "skeleton_invariant",
    "skeleton_imu_actor",
)
EXTERNAL = (
    (ROOT / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz", "probabilities", False),
    (HERE / "runs/p89_imu_orientation_expert_v1/oof_logits.npz", "imu_logits", True),
    (HERE / "runs/p89_imu_rocket_expert_v1/oof_logits.npz", "imu_logits", True),
    (HERE / "runs/p89_imu_sensor_attention_expert_v1/oof_probabilities.npz", "equal_probability", False),
    (HERE / "runs/p89_imu_spectral_forest_v1/oof_logits.npz", "imu_logits", True),
    (HERE / "runs/p86_imu_composite_teacher_v1/composite_imu_teacher.npz", "imu_logits", True),
    (HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz", "skeleton_logits", True),
    (HERE / "runs/p89_skeleton_imu_actor_matching_v1/oof_logits.npz", "skeleton_logits", True),
)


def softmax(values):
    value = np.asarray(values, dtype=np.float64)
    value -= value.max(axis=1, keepdims=True)
    probability = np.exp(value)
    return probability / probability.sum(axis=1, keepdims=True)


def align(values, source_ids, target_ids):
    values = np.asarray(values, dtype=np.float32)
    output = np.full((len(target_ids), 40), 1.0 / 40.0, dtype=np.float32)
    lookup = {sample_id: index for index, sample_id in enumerate(np.asarray(source_ids).astype(str))}
    for row, sample_id in enumerate(np.asarray(target_ids).astype(str)):
        if sample_id in lookup:
            output[row] = values[lookup[sample_id]]
    output /= output.sum(axis=1, keepdims=True)
    return output


def matched_load_data():
    p399.NONVISUAL_TEACHERS = ORIGINAL_NAMES
    parts = ORIGINAL_LOAD()
    original_lookup = {name: index for index, name in enumerate(ORIGINAL_NAMES)}
    archives = []
    for path, key, is_logits in EXTERNAL:
        archive = np.load(path)
        values = softmax(archive[key]) if is_logits else np.asarray(archive[key], dtype=np.float32)
        archives.append((archive["sample_ids"].astype(str), values))
    for cohort in p399.COHORTS:
        part = parts[cohort]
        original = part["nonvisual_probability"]
        blocks = [
            original[:, original_lookup["p12_thermal_candidate"], :],
            original[:, original_lookup["a18_best_session"], :],
            original[:, original_lookup["p87_sequence"], :],
            original[:, original_lookup["p88_repeat"], :],
            original[:, original_lookup["p128_hierarchical_multimodal"], :],
        ]
        blocks.extend(
            align(values, source_ids, part["ids"])
            for source_ids, values in archives
        )
        part["nonvisual_probability"] = np.stack(blocks, axis=1).astype(np.float32)
    p399.NONVISUAL_TEACHERS = MATCHED_NAMES
    return parts


def main():
    print(
        "P412 reruns P410 using only nonvisual teachers with exact paired OOF and Test "
        "artifacts; all safe-posterior proxy teachers are removed.",
        flush=True,
    )
    p399.NONVISUAL_TEACHERS = MATCHED_NAMES
    p399.load_data = matched_load_data
    p410.OUT = OUT
    p410.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P412_matched_nonvisual_target_aware_OOF"
    report["protocol"]["matched_nonvisual_teachers"] = list(MATCHED_NAMES)
    report["protocol"]["proxy_or_safe_fallback_teachers"] = 0
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: P410 target-aware raw-detail head with exact OOF/Test-matched nonvisual bank.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
