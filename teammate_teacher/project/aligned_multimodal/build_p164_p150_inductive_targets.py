"""Build source-only P150 targets for honest inductive Student validation.

For each held H1/H2/H3 cohort, the adaptation target contains only the other
two cohorts.  The held sample IDs are written separately and are never present
in the pseudo-target training file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/p162_p150_student_targets_v1"
OUTPUT = HERE / "runs/p164_p150_inductive_targets_v1"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
KEYS = (
    "sample_ids",
    "target_mask",
    "emission_probability",
    "structured_distillation_probability",
    "structured_confidence",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def load_cohort(name: str) -> dict[str, np.ndarray]:
    path = SOURCE / name / "structured_targets.npz"
    with np.load(path, allow_pickle=False) as saved:
        values = {key: np.asarray(saved[key]) for key in KEYS}
    if not values["target_mask"].all():
        raise RuntimeError(f"{name}: expected every frozen P150 target row to be active")
    if len(set(values["sample_ids"].astype(str).tolist())) != len(values["sample_ids"]):
        raise RuntimeError(f"{name}: duplicate sample IDs")
    return values


def main() -> None:
    cohorts = {name: load_cohort(name) for name in COHORTS}
    id_sets = {
        name: set(values["sample_ids"].astype(str).tolist())
        for name, values in cohorts.items()
    }
    for left_index, left in enumerate(COHORTS):
        for right in COHORTS[left_index + 1 :]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                raise RuntimeError(f"cohorts overlap: {left}/{right}: {len(overlap)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "stage": "P164_P150_source_only_inductive_target_build",
        "status": "complete",
        "protocol": (
            "For each held cohort, pseudo-target adaptation sees only the other two "
            "cohorts. Held inputs and held P150 targets are excluded before training."
        ),
        "held_labels_or_targets_used_for_training": False,
        "cohorts": {},
    }
    cohort_report: dict[str, object] = {}
    for held in COHORTS:
        source_names = [name for name in COHORTS if name != held]
        source = {
            key: np.concatenate([cohorts[name][key] for name in source_names], axis=0)
            for key in KEYS
        }
        source_ids = source["sample_ids"].astype(str)
        held_ids = cohorts[held]["sample_ids"].astype(str)
        if set(source_ids.tolist()) & set(held_ids.tolist()):
            raise RuntimeError(f"{held}: held IDs leaked into source targets")
        held_dir = OUTPUT / held
        held_dir.mkdir(parents=True, exist_ok=True)
        target_path = held_dir / "source_structured_targets.npz"
        held_path = held_dir / "held_sample_ids.npy"
        np.savez_compressed(target_path, **source)
        np.save(held_path, held_ids)
        cohort_report[held] = {
            "source_cohorts": source_names,
            "source_rows": int(len(source_ids)),
            "held_rows": int(len(held_ids)),
            "source_held_overlap": 0,
            "target_path": str(target_path.resolve()),
            "target_sha256": digest(target_path),
            "held_ids_path": str(held_path.resolve()),
            "held_ids_sha256": digest(held_path),
        }
    report["cohorts"] = cohort_report
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
