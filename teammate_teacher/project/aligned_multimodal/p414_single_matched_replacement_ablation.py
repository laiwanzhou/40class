"""Single-slot matched nonvisual replacement ablation for P410."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import p399_candidate_conditioned_raw_detail_head_oof as p399
import p410_target_aware_weighted_raw_detail_oof as p410
from p412_matched_nonvisual_target_aware_oof import align, softmax


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p414_single_matched_replacement_ablation_v1"
ORIGINAL_LOAD = p399.load_data
VARIANTS = {
    "replace_deep_imu": (1, HERE / "runs/p86_imu_composite_teacher_v1/composite_imu_teacher.npz", "imu_logits", True),
    "replace_p12_imu": (4, HERE / "runs/p89_imu_sensor_attention_expert_v1/oof_probabilities.npz", "equal_probability", False),
    "replace_skeleton_invariant": (6, HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz", "skeleton_logits", True),
}


def make_loader(index, path, key, is_logits):
    archive = np.load(path)
    values = softmax(archive[key]) if is_logits else np.asarray(archive[key], dtype=np.float32)
    source_ids = archive["sample_ids"].astype(str)

    def load():
        parts = ORIGINAL_LOAD()
        for cohort in p399.COHORTS:
            part = parts[cohort]
            probability = part["nonvisual_probability"].copy()
            probability[:, index, :] = align(values, source_ids, part["ids"])
            part["nonvisual_probability"] = probability
        return parts

    return load


def main():
    print(
        "P414 runs one complete P410 audit for each severe fallback slot replaced alone.",
        flush=True,
    )
    results = {}
    for name, (index, path, key, is_logits) in VARIANTS.items():
        run = OUT / name
        p399.load_data = make_loader(index, path, key, is_logits)
        p410.OUT = run
        p410.main()
        result = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        results[name] = result["aggregate"]
        print(json.dumps({"variant": name, **result["aggregate"]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "P414_single_matched_replacement_ablation",
        "status": "complete",
        "protocol": {
            "base": "P410",
            "one_replacement_per_run": True,
            "variants": list(VARIANTS),
            "test_rows_loaded": 0,
            "test_labels_read": False,
        },
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: single severe-fallback replacement ablation for P410.\n"
        + json.dumps(results, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
