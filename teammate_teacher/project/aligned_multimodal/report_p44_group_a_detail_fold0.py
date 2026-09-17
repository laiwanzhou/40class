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
DEFAULT_RUN = PROJECT_DIR / "runs" / "p44_group_a_detail_fold0_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report P44 Group-A fold0 Detail pilot")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    comparisons = read_csv(run / "per_sample_comparison.csv")
    manifest = read_csv(PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv")
    names = {int(row["class_id"]): row["class_name"] for row in manifest}

    base = {
        int(row["class_id"]): row
        for row in summary["base_comparison"]["metrics_restricted_oracle_group"]["per_class"]
    }
    detail = {int(row["class_id"]): row for row in summary["detail_metrics"]["per_class"]}
    shuffled = {
        int(row["class_id"]): row
        for row in summary["roi_information_audit"]["local_roi_shuffled_across_trials"]["per_class"]
    }
    per_class = []
    for class_id in sorted(base):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "support": int(detail[class_id]["support"]),
                "base_recall": float(base[class_id]["recall"]),
                "detail_recall": float(detail[class_id]["recall"]),
                "recall_delta_pp": 100
                * (float(detail[class_id]["recall"]) - float(base[class_id]["recall"])),
                "base_f1": float(base[class_id]["f1"]),
                "detail_f1": float(detail[class_id]["f1"]),
                "f1_delta_pp": 100
                * (float(detail[class_id]["f1"]) - float(base[class_id]["f1"])),
                "shuffled_local_roi_f1": float(shuffled[class_id]["f1"]),
                "detail_minus_shuffle_f1_pp": 100
                * (float(detail[class_id]["f1"]) - float(shuffled[class_id]["f1"])),
            }
        )
    write_csv(run / "per_class_analysis.csv", per_class)

    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in comparisons:
        by_user[row["user_id"]].append(row)
    per_user = []
    for user, rows in sorted(by_user.items()):
        base_correct = sum(int(row["base_restricted_correct"]) for row in rows)
        detail_correct = sum(int(row["detail_correct"]) for row in rows)
        rescue = sum(row["outcome"] == "rescue" for row in rows)
        new_error = sum(row["outcome"] == "new_error" for row in rows)
        per_user.append(
            {
                "user_id": user,
                "samples": len(rows),
                "base_correct": base_correct,
                "detail_correct": detail_correct,
                "accuracy_delta_pp": 100 * (detail_correct - base_correct) / len(rows),
                "rescue": rescue,
                "new_error": new_error,
                "net_rescue": rescue - new_error,
            }
        )
    write_csv(run / "per_subject_analysis.csv", per_user)

    base_metrics = summary["base_comparison"]["metrics_restricted_oracle_group"]
    detail_metrics = summary["detail_metrics"]
    shuffle_metrics = summary["roi_information_audit"]["local_roi_shuffled_across_trials"]
    zero_metrics = summary["roi_information_audit"]["local_roi_zeroed"]
    methods = ["Base\n8-class", "Detail", "Detail\nROI shuffle", "Detail\nROI zero"]
    accuracy = [
        base_metrics["accuracy"],
        detail_metrics["accuracy"],
        shuffle_metrics["accuracy"],
        zero_metrics["accuracy"],
    ]
    macro_f1 = [
        base_metrics["macro_f1"],
        detail_metrics["macro_f1"],
        shuffle_metrics["macro_f1"],
        zero_metrics["macro_f1"],
    ]
    class_ids = [int(row["class_id"]) for row in per_class]
    base_class_f1 = [float(row["base_f1"]) for row in per_class]
    detail_class_f1 = [float(row["detail_f1"]) for row in per_class]

    figure, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=160)
    positions = np.arange(len(methods))
    width = 0.36
    axes[0].bar(positions - width / 2, np.asarray(accuracy) * 100, width, label="Accuracy")
    axes[0].bar(positions + width / 2, np.asarray(macro_f1) * 100, width, label="Macro-F1")
    axes[0].set_xticks(positions, methods)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Group-A aggregate result")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    positions = np.arange(len(class_ids))
    axes[1].bar(positions - width / 2, np.asarray(base_class_f1) * 100, width, label="Base")
    axes[1].bar(positions + width / 2, np.asarray(detail_class_f1) * 100, width, label="Detail")
    axes[1].set_xticks(positions, [str(value) for value in class_ids])
    axes[1].set_xlabel("Class ID")
    axes[1].set_ylabel("F1 (%)")
    axes[1].set_title("Per-class F1")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    outcomes = [
        int(summary["base_comparison"]["rescue_base_wrong_detail_right"]),
        int(summary["base_comparison"]["new_error_base_right_detail_wrong"]),
        int(summary["base_comparison"]["both_correct"]),
        int(summary["base_comparison"]["both_wrong"]),
    ]
    outcome_names = ["Rescue", "New error", "Both right", "Both wrong"]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f"]
    axes[2].bar(outcome_names, outcomes, color=colors)
    axes[2].set_ylabel("Samples")
    axes[2].set_title("Detail vs Base outcomes")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.25)
    for index, value in enumerate(outcomes):
        axes[2].text(index, value + 1, str(value), ha="center", fontsize=9)
    figure.suptitle("P44 Group-A fold0 pilot (358 train / 201 unseen-subject validation)")
    figure.tight_layout()
    figure.savefig(run / "result_overview.png", bbox_inches="tight")
    plt.close(figure)

    checks = {
        "detail_macro_f1_above_base": detail_metrics["macro_f1"] > base_metrics["macro_f1"],
        "rescue_exceeds_new_error": summary["base_comparison"]["rescue_base_wrong_detail_right"]
        > summary["base_comparison"]["new_error_base_right_detail_wrong"],
        "local_roi_shuffle_hurts_macro_f1": detail_metrics["macro_f1"]
        > shuffle_metrics["macro_f1"],
        "majority_inner_folds_positive": None,
        "missing_modality_stability_checked": False,
    }
    decision = {
        "local_information_learned": bool(
            summary["roi_information_audit"]["normal_minus_shuffle_macro_f1_pp"] > 5.0
        ),
        "ready_for_router": False,
        "reason": (
            "Detail improves macro-F1 and strongly depends on local ROI, but it produces "
            "28 new errors versus 26 rescues and has only fold0 exploratory evidence."
        ),
        "recommended_next_step": (
            "Run the same frozen Group-A protocol on inner fold1; do not train Router yet."
        ),
        "entry_checks": checks,
    }
    (run / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# P44 Group-A fold0 局部专家结果",
        "",
        "> 这是fold0探索性验证。Group-A来自同一fold0的混淆分析，因此不能视为无偏最终结论。",
        "",
        "## 核心结果",
        "",
        "| 方法 | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| Base限制在8类 | {100*base_metrics['accuracy']:.2f}% | {100*base_metrics['macro_f1']:.2f}% |",
        f"| Detail正常输入 | {100*detail_metrics['accuracy']:.2f}% | {100*detail_metrics['macro_f1']:.2f}% |",
        f"| Detail局部ROI跨样本打乱 | {100*shuffle_metrics['accuracy']:.2f}% | {100*shuffle_metrics['macro_f1']:.2f}% |",
        f"| Detail局部ROI置零 | {100*zero_metrics['accuracy']:.2f}% | {100*zero_metrics['macro_f1']:.2f}% |",
        "",
        f"Detail相对Base：Accuracy `{100*(detail_metrics['accuracy']-base_metrics['accuracy']):+.2f}` pp，Macro-F1 `{100*(detail_metrics['macro_f1']-base_metrics['macro_f1']):+.2f}` pp。",
        f"救回 `{summary['base_comparison']['rescue_base_wrong_detail_right']}` 条，新增错误 `{summary['base_comparison']['new_error_base_right_detail_wrong']}` 条，净救回 `{summary['base_comparison']['net_rescue']}`。",
        "",
        "## 逐类变化",
        "",
        "| ID | 类别 | 支持量 | Base F1 | Detail F1 | ΔF1 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in per_class:
        lines.append(
            f"| {row['class_id']} | {row['class_name']} | {row['support']} | "
            f"{100*row['base_f1']:.2f}% | {100*row['detail_f1']:.2f}% | {row['f1_delta_pp']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- 局部ROI确实提供了可学习信息：通过。",
            "- 组内Macro-F1超过Base：通过。",
            "- rescue数量超过new-error：未通过（26 vs 28）。",
            "- 多数inner folds方向一致：尚未验证。",
            "- 当前不进入Router；建议下一步只补相同协议的inner fold1确认。",
        ]
    )
    (run / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
