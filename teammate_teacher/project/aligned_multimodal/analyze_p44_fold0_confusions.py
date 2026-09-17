from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
FAMILIES: dict[str, set[int]] = {
    "face_grooming_health": {0, 1, 2, 4, 38, 39},
    "food_tableware": {6, 7, 8, 9, 10, 11, 14, 37},
    "document_device": {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
    "clothing_cleaning": {3, 5, 12, 13, 15, 16},
    "body_motion": {28, 29, 30, 31, 32, 33, 34, 35, 36},
}
LOCAL_DETAIL_PRIORITY_FAMILIES = {"food_tableware", "document_device"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate provisional fold0-only P44 confusion groups."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p44_p12_base_inner_oof"
        / "fold0_pilot"
        / "fold0_base_logits.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p44_p12_base_inner_oof"
        / "fold0_pilot"
        / "confusion_groups",
    )
    parser.add_argument("--min-errors", type=int, default=4)
    parser.add_argument("--min-subjects", type=int, default=2)
    parser.add_argument("--min-edge-weight", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def family_of(class_id: int) -> str:
    matches = [name for name, members in FAMILIES.items() if class_id in members]
    if len(matches) != 1:
        raise ValueError(f"Class {class_id} must belong to exactly one semantic family")
    return matches[0]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def components(self) -> list[list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for value in sorted(self.parent):
            groups[self.find(value)].append(value)
        return sorted(groups.values(), key=lambda values: (-len(values), values))


def plot_confusion(matrix: np.ndarray, path: Path) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )
    figure, axis = plt.subplots(figsize=(13, 11), dpi=160)
    image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_xlabel("Predicted class ID")
    axis.set_ylabel("True class ID")
    axis.set_title("P44 fold0 pilot: row-normalized Base confusion")
    axis.set_xticks(range(40))
    axis.set_yticks(range(40))
    axis.tick_params(labelsize=6)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.base.resolve(), allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        subjects = data["subjects"].astype(str)
        skeleton_logits = data["skeleton_logits"].astype(np.float32)
        depth_logits = data["depth_logits"].astype(np.float32)
        imu_logits = data["imu_logits"].astype(np.float32)
        imu_present = data["imu_present"].astype(bool)
        base_logits = data["base_logits"].astype(np.float32)
        outer_held_generated = bool(data["outer_held_predictions_generated"].item())
    if outer_held_generated:
        raise RuntimeError("This analysis refuses artifacts containing outer-held predictions")
    if len(labels) != 593 or len(set(sample_ids.tolist())) != len(sample_ids):
        raise RuntimeError("Expected 593 unique fold0-pilot validation samples")

    manifest_rows = read_csv(args.manifest.resolve())
    names = {int(row["class_id"]): row["class_name"] for row in manifest_rows}
    if set(names) != set(range(40)):
        raise RuntimeError("Expected exactly 40 class names")
    expected_ids = {row["sample_id"] for row in manifest_rows if row["split"] == "val"}
    if set(sample_ids.tolist()) != expected_ids:
        raise RuntimeError("Base rows do not match manifest validation rows")

    probabilities = softmax(base_logits)
    predictions = probabilities.argmax(1)
    top2 = np.sort(probabilities, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]
    entropies = -(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))
    ).sum(1)
    skeleton_predictions = skeleton_logits.argmax(1)
    depth_predictions = depth_logits.argmax(1)
    imu_predictions = imu_logits.argmax(1)
    disagreement = skeleton_predictions != depth_predictions
    disagreement |= imu_present & (
        (imu_predictions != skeleton_predictions) | (imu_predictions != depth_predictions)
    )

    confusion = np.zeros((40, 40), dtype=np.int64)
    for true_label, prediction in zip(labels, predictions):
        confusion[int(true_label), int(prediction)] += 1
    write_csv(
        output / "confusion_matrix_counts.csv",
        [
            {"true_class": row, **{f"pred_{column}": int(confusion[row, column]) for column in range(40)}}
            for row in range(40)
        ],
    )
    plot_confusion(confusion, output / "confusion_heatmap.png")

    class_rows: list[dict[str, Any]] = []
    class_stats: dict[int, dict[str, float]] = {}
    for class_id in range(40):
        selected = labels == class_id
        support = int(selected.sum())
        recall = float(np.mean(predictions[selected] == class_id)) if support else float("nan")
        stats = {
            "support": float(support),
            "recall": recall,
            "mean_confidence": float(probabilities[selected].max(1).mean()) if support else float("nan"),
            "mean_margin": float(margins[selected].mean()) if support else float("nan"),
            "mean_entropy": float(entropies[selected].mean()) if support else float("nan"),
            "expert_disagreement_rate": float(disagreement[selected].mean()) if support else float("nan"),
        }
        class_stats[class_id] = stats
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "semantic_family": family_of(class_id),
                "support": support,
                "correct": int(confusion[class_id, class_id]),
                "recall": recall,
                "mean_confidence": stats["mean_confidence"],
                "mean_margin": stats["mean_margin"],
                "mean_entropy": stats["mean_entropy"],
                "expert_disagreement_rate": stats["expert_disagreement_rate"],
                "pilot_unobserved": int(support == 0),
            }
        )
    write_csv(output / "per_class_metrics.csv", class_rows)

    edge_rows: list[dict[str, Any]] = []
    retained_rows: list[dict[str, Any]] = []
    for left in range(40):
        for right in range(left + 1, 40):
            left_to_right = int(confusion[left, right])
            right_to_left = int(confusion[right, left])
            error_count = left_to_right + right_to_left
            if error_count == 0:
                continue
            left_support = int(confusion[left].sum())
            right_support = int(confusion[right].sum())
            edge_weight = 0.5 * (
                left_to_right / max(left_support, 1)
                + right_to_left / max(right_support, 1)
            )
            selected = ((labels == left) & (predictions == right)) | (
                (labels == right) & (predictions == left)
            )
            error_subjects = sorted(set(subjects[selected].tolist()))
            left_stats = class_stats[left]
            right_stats = class_stats[right]
            hard_endpoint = (
                (np.isfinite(left_stats["recall"]) and left_stats["recall"] < 0.60)
                or (np.isfinite(right_stats["recall"]) and right_stats["recall"] < 0.60)
                or (np.isfinite(left_stats["mean_margin"]) and left_stats["mean_margin"] < 0.20)
                or (np.isfinite(right_stats["mean_margin"]) and right_stats["mean_margin"] < 0.20)
                or (
                    np.isfinite(left_stats["expert_disagreement_rate"])
                    and left_stats["expert_disagreement_rate"] > 0.45
                )
                or (
                    np.isfinite(right_stats["expert_disagreement_rate"])
                    and right_stats["expert_disagreement_rate"] > 0.45
                )
            )
            statistical_retained = (
                error_count >= int(args.min_errors)
                and len(error_subjects) >= int(args.min_subjects)
                and edge_weight >= float(args.min_edge_weight)
                and hard_endpoint
            )
            left_family = family_of(left)
            right_family = family_of(right)
            same_family = left_family == right_family
            row = {
                "class_a": left,
                "name_a": names[left],
                "class_b": right,
                "name_b": names[right],
                "a_to_b_errors": left_to_right,
                "b_to_a_errors": right_to_left,
                "bidirectional_errors": error_count,
                "edge_weight": edge_weight,
                "error_subject_count": len(error_subjects),
                "error_subjects": json.dumps(error_subjects),
                "family_a": left_family,
                "family_b": right_family,
                "same_semantic_family": int(same_family),
                "hard_endpoint": int(hard_endpoint),
                "statistical_retained": int(statistical_retained),
                "local_detail_edge": int(
                    statistical_retained
                    and same_family
                    and left_family in LOCAL_DETAIL_PRIORITY_FAMILIES
                ),
            }
            edge_rows.append(row)
            if statistical_retained:
                retained_rows.append(row)
    edge_rows.sort(
        key=lambda row: (-float(row["edge_weight"]), -int(row["bidirectional_errors"]))
    )
    retained_rows.sort(
        key=lambda row: (-float(row["edge_weight"]), -int(row["bidirectional_errors"]))
    )
    write_csv(output / "confusion_edges_all.csv", edge_rows)
    write_csv(output / "confusion_edges_retained.csv", retained_rows)

    union_by_family: dict[str, UnionFind] = defaultdict(UnionFind)
    for row in retained_rows:
        if not int(row["same_semantic_family"]):
            continue
        family = str(row["family_a"])
        union_by_family[family].union(int(row["class_a"]), int(row["class_b"]))
    groups: list[dict[str, Any]] = []
    for family, union in sorted(union_by_family.items()):
        for component in union.components():
            if len(component) < 2:
                continue
            component_set = set(component)
            component_edges = [
                row
                for row in retained_rows
                if int(row["class_a"]) in component_set
                and int(row["class_b"]) in component_set
                and str(row["family_a"]) == family
                and str(row["family_b"]) == family
            ]
            evidence_subjects = sorted(
                {
                    subject
                    for row in component_edges
                    for subject in json.loads(str(row["error_subjects"]))
                }
            )
            group_type = (
                "local_detail_priority"
                if family in LOCAL_DETAIL_PRIORITY_FAMILIES
                else "base_or_future_specialist_review"
            )
            groups.append(
                {
                    "group_id": "pending_rank_assignment",
                    "family": family,
                    "group_type": group_type,
                    "class_ids": component,
                    "class_names": [names[class_id] for class_id in component],
                    "validation_support": int(
                        sum(class_stats[class_id]["support"] for class_id in component)
                    ),
                    "retained_edges": [
                        {
                            "pair": [int(row["class_a"]), int(row["class_b"])],
                            "errors": int(row["bidirectional_errors"]),
                            "edge_weight": float(row["edge_weight"]),
                            "subjects": json.loads(str(row["error_subjects"])),
                        }
                        for row in component_edges
                    ],
                    "evidence_subjects": evidence_subjects,
                    "warning": "fold0-only provisional group; do not freeze before fold1/2 confirmation",
                }
            )
    groups.sort(
        key=lambda group: (
            group["group_type"] != "local_detail_priority",
            -sum(edge["errors"] for edge in group["retained_edges"]),
        )
    )
    for index, group in enumerate(groups, start=1):
        group["rank"] = index
        group["group_id"] = f"pilot_group_{index:02d}_{group['family']}"

    unobserved = [row for row in class_rows if int(row["pilot_unobserved"])]
    result = {
        "protocol": "p44-fold0-pilot-confusion-groups-v1",
        "status": "provisional_fold0_only_not_frozen",
        "scope": {
            "samples": int(len(labels)),
            "subjects": sorted(set(subjects.tolist())),
            "outer_held_predictions_generated": False,
        },
        "predeclared_edge_rule": {
            "edge_weight": "0.5 * [P(pred=b|y=a) + P(pred=a|y=b)]",
            "min_bidirectional_errors": int(args.min_errors),
            "min_error_subjects": int(args.min_subjects),
            "min_edge_weight": float(args.min_edge_weight),
            "hard_endpoint": "recall<0.60 or mean_margin<0.20 or expert_disagreement_rate>0.45",
            "semantic_rule": "graph proposes edges; connected components are only formed within label-defined semantic families",
        },
        "counts": {
            "nonzero_edges": int(len(edge_rows)),
            "retained_statistical_edges": int(len(retained_rows)),
            "candidate_groups": int(len(groups)),
            "local_detail_priority_groups": int(
                sum(group["group_type"] == "local_detail_priority" for group in groups)
            ),
        },
        "pilot_unobserved_classes": [
            {"class_id": int(row["class_id"]), "class_name": row["class_name"]}
            for row in unobserved
        ],
        "candidate_groups": groups,
        "stop_condition": "Candidate groups generated; Detail/Router training not started.",
    }
    (output / "candidate_groups.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# P44 fold0 临时候选困难/混淆组",
        "",
        "> 仅使用 inner-fold0 的 593 条、4 个未见 subject。这里的组只能用于便宜的 Detail 可行性试验，不能正式冻结。",
        "",
        f"统计保留边：`{len(retained_rows)}`；候选组：`{len(groups)}`；优先局部 Detail 组：`{sum(group['group_type'] == 'local_detail_priority' for group in groups)}`。",
        "",
    ]
    for group in groups:
        lines.extend(
            [
                f"## Rank {group['rank']}：{group['group_id']}",
                "",
                f"- 类型：`{group['group_type']}`",
                f"- 类别：{', '.join(f'`{class_id} {names[class_id]}`' for class_id in group['class_ids'])}",
                f"- fold0 验证支持量：{group['validation_support']}",
                f"- 错误涉及 subject：{', '.join(group['evidence_subjects'])}",
                "- 保留混淆边："
                + "; ".join(
                    f"{edge['pair'][0]}↔{edge['pair'][1]} errors={edge['errors']}, weight={edge['edge_weight']:.3f}"
                    for edge in group["retained_edges"]
                ),
                "",
            ]
        )
    if unobserved:
        lines.extend(
            [
                "## fold0 无法判断的类别",
                "",
                ", ".join(
                    f"`{row['class_id']} {row['class_name']}`" for row in unobserved
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 本轮停止点",
            "",
            "已生成候选组；尚未训练 Detail、Router，也未读取 fold0 outer-held 标签。",
        ]
    )
    (output / "candidate_groups.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
