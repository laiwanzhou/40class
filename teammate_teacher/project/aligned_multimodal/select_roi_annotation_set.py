from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OOF = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_ROWS = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof_rows.csv"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_PRIOR = (
    PROJECT_DIR
    / "data"
    / "local_action_audit_v1"
    / "annotations_temporal_pilot30.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "local_roi_annotation_v2"
SMALL_ACTION_IDS = (
    1, 2, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select 216 ROI annotations over all 40 classes: 60 strictly blind "
            "locator references and 156 correction labels (including Pilot30)."
        )
    )
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--oof-rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def greedy_representative(
    candidates: list[dict[str, object]],
    count: int,
    fold_counts: Counter[int],
    user_counts: Counter[str],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    pool = list(candidates)
    confidences = np.asarray(
        [float(row["final_confidence"]) for row in pool], dtype=np.float64
    )
    lengths = np.log1p(
        np.asarray([int(row["num_aligned_frames"]) for row in pool], dtype=np.float64)
    )
    median_confidence = float(np.median(confidences))
    median_length = float(np.median(lengths))
    while len(selected) < count:
        if not pool:
            raise RuntimeError("Not enough candidates for blind selection")
        scored = []
        for row in pool:
            score = (
                2.0 * abs(float(row["final_confidence"]) - median_confidence)
                + 0.15
                * abs(math.log1p(int(row["num_aligned_frames"])) - median_length)
                + 0.22 * fold_counts[int(row["fold"])]
                + 0.04 * user_counts[str(row["user_id"])]
            )
            scored.append((score, str(row["sample_id"]), row))
        _, _, chosen = min(scored)
        pool.remove(chosen)
        selected.append(chosen)
        fold_counts[int(chosen["fold"])] += 1
        user_counts[str(chosen["user_id"])] += 1
    return selected


def greedy_risk(
    candidates: list[dict[str, object]],
    count: int,
    fold_counts: Counter[int],
    user_counts: Counter[str],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    pool = list(candidates)
    while len(selected) < count:
        if not pool:
            raise RuntimeError("Not enough candidates for correction selection")
        scored = []
        for row in pool:
            disagreement = int(row["sd_prediction"]) != int(row["final_prediction"])
            risk = (
                2.0 * (1 - int(row["final_correct"]))
                + 1.2 * (1.0 - float(row["final_margin"]))
                + 0.35 * disagreement
                + 0.10 * int(row["route_to_thermal"])
                - 0.18 * fold_counts[int(row["fold"])]
                - 0.025 * user_counts[str(row["user_id"])]
            )
            scored.append((risk, str(row["sample_id"]), row))
        _, _, chosen = max(scored)
        pool.remove(chosen)
        selected.append(chosen)
        fold_counts[int(chosen["fold"])] += 1
        user_counts[str(chosen["user_id"])] += 1
    return selected


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {row["sample_id"]: row for row in read_csv(args.manifest.resolve())}
    oof_rows = {row["sample_id"]: row for row in read_csv(args.oof_rows.resolve())}
    with np.load(args.oof.resolve(), allow_pickle=False) as data:
        if len(data["sample_ids"]) != 2914:
            raise ValueError("ROI selection requires the complete 2914-row OOF")
    prior_rows = read_csv(args.prior.resolve())
    prior_by_id = {row["sample_id"]: row for row in prior_rows}
    if len(prior_by_id) != 30:
        raise ValueError(f"Expected 30 prior temporal annotations, got {len(prior_by_id)}")

    rows: list[dict[str, object]] = []
    for sample_id, manifest_row in manifest.items():
        oof = oof_rows[sample_id]
        rows.append(
            {
                **manifest_row,
                **{
                    key: oof[key]
                    for key in (
                        "fold",
                        "imu_present",
                        "thermal_present",
                        "sd_prediction",
                        "sd_imu_prediction",
                        "thermal_candidate_prediction",
                        "route_to_thermal",
                        "route_probability",
                        "final_prediction",
                        "final_correct",
                        "final_confidence",
                        "final_margin",
                    )
                },
            }
        )
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)

    class_accuracy = {
        class_id: float(
            np.mean([int(row["final_correct"]) for row in class_rows])
        )
        for class_id, class_rows in by_class.items()
    }
    non_small = sorted(set(range(40)) - set(SMALL_ACTION_IDS))
    extra_non_small = sorted(
        non_small, key=lambda class_id: (class_accuracy[class_id], class_id)
    )[:10]
    total_targets = {
        class_id: (
            8
            if class_id in SMALL_ACTION_IDS
            else 3
            if class_id in extra_non_small
            else 2
        )
        for class_id in range(40)
    }
    if sum(total_targets.values()) != 216:
        raise AssertionError("Class allocation must sum to 216")

    # Every class receives one blind sample. The 20 hardest of 21 small-action
    # classes receive a second blind sample, yielding exactly 60.
    blind_extra_small = sorted(
        SMALL_ACTION_IDS,
        key=lambda class_id: (class_accuracy[class_id], class_id),
    )[:20]
    blind_targets = {
        class_id: 1 + int(class_id in blind_extra_small) for class_id in range(40)
    }
    if sum(blind_targets.values()) != 60:
        raise AssertionError("Blind allocation must sum to 60")

    blind: list[dict[str, object]] = []
    blind_fold_counts: Counter[int] = Counter()
    blind_user_counts: Counter[str] = Counter()
    for class_id in range(40):
        candidates = [
            row
            for row in by_class[class_id]
            if row["sample_id"] not in prior_by_id
        ]
        blind.extend(
            greedy_representative(
                candidates,
                blind_targets[class_id],
                blind_fold_counts,
                blind_user_counts,
            )
        )
    blind_ids = {str(row["sample_id"]) for row in blind}
    if len(blind_ids) != 60:
        raise AssertionError("Blind set must contain 60 unique samples")

    correction: list[dict[str, object]] = []
    correction_fold_counts: Counter[int] = Counter()
    correction_user_counts: Counter[str] = Counter()
    for prior_id in prior_by_id:
        row = next(row for row in rows if row["sample_id"] == prior_id)
        correction.append(row)
        correction_fold_counts[int(row["fold"])] += 1
        correction_user_counts[str(row["user_id"])] += 1

    existing_by_class = Counter(int(row["class_id"]) for row in correction)
    for class_id in range(40):
        needed = (
            total_targets[class_id]
            - blind_targets[class_id]
            - existing_by_class[class_id]
        )
        if needed < 0:
            raise ValueError(f"Prior Pilot30 overfills class {class_id}")
        excluded = blind_ids | set(prior_by_id)
        candidates = [
            row for row in by_class[class_id] if row["sample_id"] not in excluded
        ]
        correction.extend(
            greedy_risk(
                candidates,
                needed,
                correction_fold_counts,
                correction_user_counts,
            )
        )
    if len(correction) != 156:
        raise AssertionError(f"Expected 156 correction rows, got {len(correction)}")

    selected: list[dict[str, object]] = []
    for annotation_mode, subset in (("blind", blind), ("correction", correction)):
        for row in subset:
            sample_id = str(row["sample_id"])
            prior = prior_by_id.get(sample_id)
            selected.append(
                {
                    "annotation_mode": annotation_mode,
                    "prior_pilot30": int(prior is not None),
                    "sample_id": sample_id,
                    "fold": int(row["fold"]),
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "is_small_action": int(int(row["class_id"]) in SMALL_ACTION_IDS),
                    "user_id": row["user_id"],
                    "trial_id": row["trial_id"],
                    "num_aligned_frames": int(row["num_aligned_frames"]),
                    "imu_present": int(row["imu_present"]),
                    "thermal_present": int(row["thermal_present"]),
                    "final_prediction": int(row["final_prediction"]),
                    "final_correct": int(row["final_correct"]),
                    "final_confidence": float(row["final_confidence"]),
                    "final_margin": float(row["final_margin"]),
                    "selection_reason": (
                        "blind_representative"
                        if annotation_mode == "blind"
                        else "prior_verified_pilot30"
                        if prior is not None
                        else "correction_error_or_low_margin"
                    ),
                    "prior_bbox_quality": prior["bbox_quality"] if prior else "",
                    "prior_has_manual_bbox": (
                        int(prior["has_manual_bbox"]) if prior else 0
                    ),
                    "prior_free_note": prior["free_note"] if prior else "",
                }
            )
    selected.sort(
        key=lambda row: (
            0 if row["annotation_mode"] == "blind" else 1,
            int(row["class_id"]),
            int(row["fold"]),
            str(row["sample_id"]),
        )
    )
    for index, row in enumerate(selected, 1):
        row["selection_index"] = index

    fieldnames = ["selection_index"] + [
        key for key in selected[0] if key != "selection_index"
    ]
    with (output_dir / "selection.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    selected_counts = Counter(int(row["class_id"]) for row in selected)
    summary = {
        "protocol": (
            "60 blind locator references (auto box, model prediction and true class "
            "must be hidden in the annotation UI) plus 156 correction labels. "
            "Pilot30 is retained inside correction. Blind rows are evaluation-only "
            "and must never train the locator."
        ),
        "counts": {
            "total": len(selected),
            "blind": len(blind),
            "correction": len(correction),
            "prior_pilot30": len(prior_by_id),
            "classes": len(selected_counts),
            "subjects": len(set(str(row["user_id"]) for row in selected)),
        },
        "fold_counts": dict(
            sorted(Counter(int(row["fold"]) for row in selected).items())
        ),
        "blind_fold_counts": dict(sorted(blind_fold_counts.items())),
        "correction_fold_counts": dict(sorted(correction_fold_counts.items())),
        "extra_non_small_classes": extra_non_small,
        "blind_extra_small_classes": blind_extra_small,
        "class_accuracy_used_for_allocation": class_accuracy,
        "class_targets": total_targets,
        "selected_class_counts": dict(sorted(selected_counts.items())),
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
