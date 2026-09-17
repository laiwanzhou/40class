from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROUTE = PROJECT_DIR / "data" / "hard_local_v1" / "oof_route_features.csv"
DEFAULT_PROTOCOL = (
    PROJECT_DIR / "data" / "hard_local_v1" / "hard_action_protocol.json"
)
DEFAULT_EXISTING = (
    PROJECT_DIR / "data" / "local_roi_annotation_v2" / "roi_annotations_final.csv"
)
DEFAULT_MOTION = PROJECT_DIR / "data" / "motion_crop_audit.csv"
DEFAULT_MOTION_BOXES = (
    PROJECT_DIR
    / "runs"
    / "p12_fold_pure_locator_predictions"
    / "motion_boxes_all2914.csv"
)
DEFAULT_SKELETON = PROJECT_DIR / "data" / "skeleton_quality.csv"
DEFAULT_UNION = (
    PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
)
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "hard_local_v1" / "annotation500"
SEED = 20260728


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select 350 hard + 100 boundary negative + 50 localization-risk ROIs"
    )
    parser.add_argument("--route-features", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--motion-audit", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-boxes", type=Path, default=DEFAULT_MOTION_BOXES)
    parser.add_argument("--skeleton-quality", type=Path, default=DEFAULT_SKELETON)
    parser.add_argument("--union-manifest", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def user_id(sample_id: str) -> str:
    return sample_id.split("__")[2]


def trial_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["class_name"], row["user_id"], row["trial_id"]


def choose_diverse(
    candidates: list[dict[str, object]],
    count: int,
    base_score_field: str,
) -> list[dict[str, object]]:
    remaining = list(candidates)
    selected: list[dict[str, object]] = []
    users: Counter[str] = Counter()
    folds: Counter[int] = Counter()
    classes: Counter[int] = Counter()
    while len(selected) < count:
        if not remaining:
            raise ValueError(f"Candidate pool exhausted at {len(selected)}/{count}")

        def adjusted(row: dict[str, object]) -> tuple[float, float, str]:
            score = float(row[base_score_field])
            score -= 0.20 * users[str(row["user_id"])]
            score -= 0.15 * folds[int(row["fold"])]
            score -= 0.08 * classes[int(row["true_class_id"])]
            return score, float(row["top1_probability"]), str(row["sample_id"])

        best = max(remaining, key=adjusted)
        remaining.remove(best)
        selected.append(best)
        users[str(best["user_id"])] += 1
        folds[int(best["fold"])] += 1
        classes[int(best["true_class_id"])] += 1
    return selected


def parse_motion_bbox(value: str) -> tuple[float, float, float, float]:
    decoded = ast.literal_eval(value)
    if not isinstance(decoded, list) or len(decoded) != 4:
        raise ValueError(f"Invalid motion bbox: {value}")
    x0, y0, x1, y1 = (float(item) for item in decoded)
    return x0 * 2.0, y0 * 2.0, (x1 + 1) * 2.0 - 1, (y1 + 1) * 2.0 - 1


def allocate_hard_quotas(
    hard_candidates: dict[int, list[dict[str, object]]],
    total: int,
) -> dict[int, int]:
    quotas = {
        class_id: min(10, len(rows))
        for class_id, rows in hard_candidates.items()
    }
    weights = {}
    for class_id, rows in hard_candidates.items():
        error_rate = np.mean(
            [int(row["final_correct"]) == 0 for row in rows]
        )
        weights[class_id] = math.sqrt(len(rows)) * (1.0 + float(error_rate))
    while sum(quotas.values()) < total:
        eligible = [
            class_id
            for class_id, rows in hard_candidates.items()
            if quotas[class_id] < len(rows)
        ]
        if not eligible:
            raise ValueError("Hard-class pool cannot fill requested quota")
        selected_class = max(
            eligible,
            key=lambda class_id: weights[class_id] / (quotas[class_id] + 1),
        )
        quotas[selected_class] += 1
    return quotas


def main() -> None:
    args = parse_args()
    route_rows = read_csv(args.route_features.resolve())
    route_by_id = {row["sample_id"]: row for row in route_rows}
    existing_ids = {
        row["sample_id"] for row in read_csv(args.existing.resolve())
    }
    motion_by_id = {
        row["sample_id"]: row for row in read_csv(args.motion_audit.resolve())
    }
    motion_box_by_id = {
        row["sample_id"]: row for row in read_csv(args.motion_boxes.resolve())
    }
    skeleton_by_id = {
        row["sample_id"]: row
        for row in read_csv(args.skeleton_quality.resolve())
    }
    union_by_key = {
        trial_key(row): row for row in read_csv(args.union_manifest.resolve())
    }
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    groups = {
        name: set(int(value) for value in data["class_ids"])
        for name, data in protocol["confusion_groups"].items()
    }

    enriched: list[dict[str, object]] = []
    for route in route_rows:
        sample_id = route["sample_id"]
        motion = motion_by_id.get(sample_id)
        motion_box = motion_box_by_id[sample_id]
        skeleton = skeleton_by_id[sample_id]
        union = union_by_key[
            (route["true_class_name"], user_id(sample_id), sample_id.split("__")[3])
        ]
        top1 = int(route["final_prediction"])
        true_class = int(route["true_class_id"])
        shared_true_prediction_groups = [
            name
            for name, members in groups.items()
            if true_class in members and top1 in members
        ]
        x0, y0, x1, y1 = (
            float(motion_box[field]) for field in ("x0", "y0", "x1", "y1")
        )
        area = (x1 - x0 + 1) * (y1 - y0 + 1) / (640 * 480)
        edge_touch = int(x0 <= 8 or y0 <= 8 or x1 >= 631 or y1 >= 471)
        abnormal_scale = int(area < 0.12 or area > 0.75)
        multi_ratio = float(skeleton["multi_ratio"])
        fallback = int(motion_box["fallback"])
        expanded_fraction = (
            float(motion["expanded_bbox_fraction"])
            if motion is not None
            else area
        )
        raw_fraction = (
            float(motion["raw_bbox_fraction"])
            if motion is not None
            else area
        )
        invalid_flip_fraction = (
            float(motion["invalid_flip_fraction"])
            if motion is not None
            else 0.0
        )
        active_pixel_fraction = (
            float(motion["active_pixel_fraction"])
            if motion is not None
            else 0.0
        )
        large_motion = int(
            expanded_fraction > 0.75
            or raw_fraction > 0.50
            or invalid_flip_fraction > 0.05
        )
        high_confidence_error = int(
            route["final_correct"] == "0"
            and float(route["top1_probability"]) >= 0.50
        )
        low_margin = int(float(route["top1_top2_margin"]) <= 0.15)
        internal_confusion = int(
            route["final_correct"] == "0"
            and bool(shared_true_prediction_groups)
        )
        hard_priority = (
            6.0 * int(route["final_correct"] == "0")
            + 3.0 * internal_confusion
            + 2.0 * high_confidence_error
            + 1.5 * low_margin
            + float(route["hard_probability_mass"])
            + 0.25 * fallback
        )
        boundary_priority = (
            3.0 * int(route["top1_is_hard"])
            + 2.0 * int(route["final_correct"] == "0")
            + 2.0 * high_confidence_error
            + float(route["hard_probability_mass"])
            + 0.75 * low_margin
            + 0.5 * int(route["top2_pair_in_frozen_group"])
        )
        risk_priority = (
            8.0 * fallback
            + 4.0 * int(multi_ratio > 0)
            + 2.0 * min(1.0, multi_ratio * 5)
            + 2.0 * edge_touch
            + 2.0 * abnormal_scale
            + 1.5 * large_motion
            + expanded_fraction
            + invalid_flip_fraction * 5
        )
        enriched.append(
            {
                **route,
                "fold": int(route["fold"]),
                "true_class_id": true_class,
                "user_id": user_id(sample_id),
                "trial_id": sample_id.split("__")[3],
                "motion_fallback": fallback,
                "motion_x0": x0,
                "motion_y0": y0,
                "motion_x1": x1,
                "motion_y1": y1,
                "motion_box_area_fraction": area,
                "motion_edge_touch": edge_touch,
                "motion_abnormal_scale": abnormal_scale,
                "motion_large_range": large_motion,
                "active_pixel_fraction": active_pixel_fraction,
                "expanded_bbox_fraction": expanded_fraction,
                "invalid_flip_fraction": invalid_flip_fraction,
                "full_motion_audit_available": int(motion is not None),
                "multi_ratio": multi_ratio,
                "max_people": int(skeleton["max_people"]),
                "tracking_changes_input": int(skeleton["tracking_changes_input"]),
                "thermal_present": int(union["thermal_present"]),
                "thermal_usable": int(union["thermal_usable"]),
                "thermal_path": union["thermal_path"],
                "shared_true_prediction_groups": "|".join(
                    shared_true_prediction_groups
                ),
                "internal_group_confusion": internal_confusion,
                "high_confidence_error": high_confidence_error,
                "low_margin_le_0_15": low_margin,
                "hard_selection_score": hard_priority,
                "boundary_selection_score": boundary_priority,
                "risk_selection_score": risk_priority,
            }
        )
    # This batch is explicitly for joint Depth/Thermal ROI auditing.
    available = [
        row
        for row in enriched
        if row["sample_id"] not in existing_ids
        and int(row["thermal_present"]) == 1
        and int(row["thermal_usable"]) == 1
    ]

    hard_by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in available:
        if int(row["true_is_hard"]) == 1:
            hard_by_class[int(row["true_class_id"])].append(row)
    quotas = allocate_hard_quotas(hard_by_class, 350)
    selected_a: list[dict[str, object]] = []
    for class_id in sorted(quotas):
        class_rows = hard_by_class[class_id]
        selected_a.extend(
            choose_diverse(class_rows, quotas[class_id], "hard_selection_score")
        )
    selected_ids = {str(row["sample_id"]) for row in selected_a}

    boundary_pool = [
        row
        for row in available
        if row["sample_id"] not in selected_ids
        and int(row["true_is_hard"]) == 0
        and int(row["candidate_local_trigger_v1"]) == 1
    ]
    boundary_correct = [
        row for row in boundary_pool if int(row["final_correct"]) == 1
    ]
    boundary_wrong = [
        row for row in boundary_pool if int(row["final_correct"]) == 0
    ]
    selected_b = [
        *choose_diverse(
            boundary_correct, 50, "boundary_selection_score"
        ),
        *choose_diverse(
            boundary_wrong, 50, "boundary_selection_score"
        ),
    ]
    selected_ids.update(str(row["sample_id"]) for row in selected_b)

    risk_pool = [
        row for row in available if row["sample_id"] not in selected_ids
    ]
    fallback_risk = [
        row for row in risk_pool if int(row["motion_fallback"]) == 1
    ]
    selected_c = choose_diverse(
        fallback_risk, min(20, len(fallback_risk)), "risk_selection_score"
    )
    selected_ids.update(str(row["sample_id"]) for row in selected_c)
    remaining_risk = [
        row for row in risk_pool if row["sample_id"] not in selected_ids
    ]
    selected_c.extend(
        choose_diverse(
            remaining_risk,
            50 - len(selected_c),
            "risk_selection_score",
        )
    )

    selected: list[dict[str, object]] = []
    for category, rows in (
        ("A_hard_true_class", selected_a),
        ("B_boundary_negative", selected_b),
        ("C_localization_risk", selected_c),
    ):
        for row in rows:
            copy = dict(row)
            copy["selection_category"] = category
            if category == "A_hard_true_class":
                reasons = [
                    name
                    for flag, name in (
                        (int(row["final_correct"]) == 0, "baseline_error"),
                        (int(row["internal_group_confusion"]) == 1, "within_group_confusion"),
                        (int(row["high_confidence_error"]) == 1, "high_confidence_error"),
                        (int(row["low_margin_le_0_15"]) == 1, "low_margin"),
                        (int(row["motion_fallback"]) == 1, "fallback"),
                    )
                    if flag
                ]
            elif category == "B_boundary_negative":
                reasons = [
                    "true_class_outside_hard_set",
                    "candidate_router_triggered",
                    (
                        "baseline_correct_destroy_risk"
                        if int(row["final_correct"]) == 1
                        else "baseline_error_boundary"
                    ),
                ]
            else:
                reasons = [
                    name
                    for flag, name in (
                        (int(row["motion_fallback"]) == 1, "fallback"),
                        (float(row["multi_ratio"]) > 0, "multi_person"),
                        (int(row["motion_edge_touch"]) == 1, "edge_touch"),
                        (int(row["motion_abnormal_scale"]) == 1, "abnormal_scale"),
                        (int(row["motion_large_range"]) == 1, "large_motion_range"),
                    )
                    if flag
                ]
            copy["selection_reasons"] = "|".join(reasons)
            selected.append(copy)
    if len(selected) != 500 or len({row["sample_id"] for row in selected}) != 500:
        raise AssertionError("Selection must contain 500 unique samples")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(selected))
    shuffled = [selected[index] for index in order]
    for index, row in enumerate(shuffled, start=1):
        row["annotation_index"] = index

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows = sorted(shuffled, key=lambda row: int(row["annotation_index"]))
    with (output_dir / "selection.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(write_rows[0]))
        writer.writeheader()
        writer.writerows(write_rows)

    def distribution(rows: list[dict[str, object]], field: str) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in sorted(
                Counter(str(row[field]) for row in rows).items()
            )
        }

    report = {
        "total": len(selected),
        "existing_roi_excluded": len(existing_ids),
        "thermal_requirement": "thermal_present=1 and thermal_usable=1",
        "categories": {
            "A_hard_true_class": len(selected_a),
            "B_boundary_negative": len(selected_b),
            "C_localization_risk": len(selected_c),
        },
        "hard_class_quotas": {str(key): value for key, value in sorted(quotas.items())},
        "folds": distribution(selected, "fold"),
        "subjects": distribution(selected, "user_id"),
        "classes": distribution(selected, "true_class_id"),
        "fallback": distribution(selected, "motion_fallback"),
        "multi_person_any": {
            "count": int(sum(float(row["multi_ratio"]) > 0 for row in selected))
        },
        "boundary_negative": {
            "baseline_correct": int(
                sum(int(row["final_correct"]) == 1 for row in selected_b)
            ),
            "baseline_wrong": int(
                sum(int(row["final_correct"]) == 0 for row in selected_b)
            ),
        },
        "selection_file": str(output_dir / "selection.csv"),
        "hard_protocol": str(args.protocol.resolve()),
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
