from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare C0/C1/C2 P87-S raw and frozen-decoder predictions."
    )
    parser.add_argument(
        "--c0", type=Path, default=PROJECT_DIR / "runs/p87s_fusion_holdout1_c0_v1"
    )
    parser.add_argument(
        "--c1", type=Path, default=PROJECT_DIR / "runs/p87s_fusion_holdout1_c1_emission_v1"
    )
    parser.add_argument(
        "--c2", type=Path, default=PROJECT_DIR / "runs/p87s_fusion_holdout1_c2_structured_v1"
    )
    parser.add_argument(
        "--c2-best",
        type=Path,
        help=(
            "Optional longer-converged structured branch. When supplied, C2 remains "
            "the equal-budget causal control and this branch measures extra convergence."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "runs/p87s_holdout1_ablation_v1"
    )
    return parser.parse_args()


def read_predictions(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "decoded_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def paired_change(
    labels: np.ndarray, first: np.ndarray, second: np.ndarray
) -> dict[str, int | float]:
    first_correct = first == labels
    second_correct = second == labels
    return {
        "delta_correct": int(second_correct.sum() - first_correct.sum()),
        "delta_accuracy_pp": float(
            100.0 * (second_correct.mean() - first_correct.mean())
        ),
        "rescue": int(np.sum(~first_correct & second_correct)),
        "harm": int(np.sum(first_correct & ~second_correct)),
        "both_wrong_same_prediction": int(
            np.sum(~first_correct & ~second_correct & (first == second))
        ),
        "both_wrong_changed_prediction": int(
            np.sum(~first_correct & ~second_correct & (first != second))
        ),
    }


def metric_row(audit: dict[str, Any]) -> dict[str, Any]:
    comparison = audit["raw_vs_decoder"]
    raw = comparison["raw"]
    decoded = comparison["decoded"]
    return {
        "raw_accuracy": raw["accuracy"],
        "raw_macro_f1": raw["macro_f1"],
        "decoded_accuracy": decoded["accuracy"],
        "decoded_macro_f1": decoded["macro_f1"],
        "decoder_delta_correct": comparison["delta_correct"],
        "decoder_delta_accuracy_pp": comparison["delta_accuracy_pp"],
        "decoder_rescue": comparison["rescue"],
        "decoder_harm": comparison["harm"],
        "raw_teacher_agreement": audit["raw_vs_structured_teacher"]["agreement"],
        "raw_teacher_errors_copied": audit["raw_vs_structured_teacher"][
            "teacher_errors_copied"
        ],
        "raw_student_correct_when_teacher_wrong": audit[
            "raw_vs_structured_teacher"
        ]["student_correct_when_teacher_wrong"],
    }


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    args = parse_args()
    run_dirs = {
        "C0_labeled_only": args.c0.resolve(),
        "C1_emission": args.c1.resolve(),
        "C2_structured": args.c2.resolve(),
    }
    if args.c2_best is not None:
        run_dirs["C2_structured_best"] = args.c2_best.resolve()
    audits = {
        name: json.loads((path / "decoder_audit.json").read_text(encoding="utf-8"))
        for name, path in run_dirs.items()
    }
    rows = {name: read_predictions(path) for name, path in run_dirs.items()}
    canonical_ids = [row["sample_id"] for row in rows["C0_labeled_only"]]
    for name, values in rows.items():
        if [row["sample_id"] for row in values] != canonical_ids:
            raise ValueError(f"{name} row order differs from C0")
    labels = np.asarray(
        [int(row["label"]) for row in rows["C0_labeled_only"]], dtype=np.int64
    )
    users = np.asarray(
        [row["user_id"] for row in rows["C0_labeled_only"]]
    ).astype(str)

    raw = {
        name: np.asarray([int(row["raw_prediction"]) for row in values], dtype=np.int64)
        for name, values in rows.items()
    }
    decoded = {
        name: np.asarray(
            [int(row["decoded_prediction"]) for row in values], dtype=np.int64
        )
        for name, values in rows.items()
    }
    result: dict[str, Any] = {
        "protocol": (
            "C0/C1/C2 share one subject-disjoint P86 initialization. C1 and C2 both "
            "use the same label-free adaptation budget; their only target difference "
            "is P85 emission versus entropy-backed-off P87 structured posterior."
        ),
        "branches": {name: metric_row(audit) for name, audit in audits.items()},
        "paired": {
            "C1_minus_C0_raw": paired_change(
                labels, raw["C0_labeled_only"], raw["C1_emission"]
            ),
            "C2_minus_C1_raw": paired_change(
                labels, raw["C1_emission"], raw["C2_structured"]
            ),
            "C2_minus_C0_raw": paired_change(
                labels, raw["C0_labeled_only"], raw["C2_structured"]
            ),
            "C1_minus_C0_decoded": paired_change(
                labels, decoded["C0_labeled_only"], decoded["C1_emission"]
            ),
            "C2_minus_C1_decoded": paired_change(
                labels, decoded["C1_emission"], decoded["C2_structured"]
            ),
            "C2_minus_C0_decoded": paired_change(
                labels, decoded["C0_labeled_only"], decoded["C2_structured"]
            ),
        },
        "by_subject": {},
        "by_class": {},
    }
    if "C2_structured_best" in raw:
        result["paired"].update(
            {
                "C2best_minus_C2equal_raw": paired_change(
                    labels, raw["C2_structured"], raw["C2_structured_best"]
                ),
                "C2best_minus_C0_raw": paired_change(
                    labels, raw["C0_labeled_only"], raw["C2_structured_best"]
                ),
                "C2best_minus_C2equal_decoded": paired_change(
                    labels,
                    decoded["C2_structured"],
                    decoded["C2_structured_best"],
                ),
                "C2best_minus_C0_decoded": paired_change(
                    labels,
                    decoded["C0_labeled_only"],
                    decoded["C2_structured_best"],
                ),
            }
        )
    by_subject: dict[str, Any] = {}
    for user in sorted(set(users.tolist())):
        selected = users == user
        by_subject[user] = {
            "rows": int(selected.sum()),
            "C1_minus_C0_raw": paired_change(
                labels[selected],
                raw["C0_labeled_only"][selected],
                raw["C1_emission"][selected],
            ),
            "C2_minus_C1_raw": paired_change(
                labels[selected],
                raw["C1_emission"][selected],
                raw["C2_structured"][selected],
            ),
            "C2_minus_C0_decoded": paired_change(
                labels[selected],
                decoded["C0_labeled_only"][selected],
                decoded["C2_structured"][selected],
            ),
        }
        if "C2_structured_best" in raw:
            by_subject[user]["C2best_minus_C0_raw"] = paired_change(
                labels[selected],
                raw["C0_labeled_only"][selected],
                raw["C2_structured_best"][selected],
            )
            by_subject[user]["C2best_minus_C0_decoded"] = paired_change(
                labels[selected],
                decoded["C0_labeled_only"][selected],
                decoded["C2_structured_best"][selected],
            )
    result["by_subject"] = by_subject
    by_class: dict[str, Any] = {}
    for class_id in sorted(map(int, np.unique(labels))):
        selected = labels == class_id
        by_class[str(class_id)] = {
            "support": int(selected.sum()),
            "C1_minus_C0_raw": paired_change(
                labels[selected],
                raw["C0_labeled_only"][selected],
                raw["C1_emission"][selected],
            ),
            "C2_minus_C1_raw": paired_change(
                labels[selected],
                raw["C1_emission"][selected],
                raw["C2_structured"][selected],
            ),
            "C2_minus_C0_decoded": paired_change(
                labels[selected],
                decoded["C0_labeled_only"][selected],
                decoded["C2_structured"][selected],
            ),
        }
        if "C2_structured_best" in raw:
            by_class[str(class_id)]["C2best_minus_C0_raw"] = paired_change(
                labels[selected],
                raw["C0_labeled_only"][selected],
                raw["C2_structured_best"][selected],
            )
            by_class[str(class_id)]["C2best_minus_C0_decoded"] = paired_change(
                labels[selected],
                decoded["C0_labeled_only"][selected],
                decoded["C2_structured_best"][selected],
            )
    result["by_class"] = by_class

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# P87-S Student ablation",
        "",
        "| Branch | Raw Acc | Raw Macro-F1 | +Decoder Acc | +Decoder Macro-F1 | Decoder rescue/harm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in result["branches"].items():
        markdown.append(
            f"| {name} | {format_percent(values['raw_accuracy'])} | "
            f"{format_percent(values['raw_macro_f1'])} | "
            f"{format_percent(values['decoded_accuracy'])} | "
            f"{format_percent(values['decoded_macro_f1'])} | "
            f"{values['decoder_rescue']}/{values['decoder_harm']} |"
        )
    markdown.extend(
        [
            "",
            "## Paired causal contrasts",
            "",
            "- C1-C0 isolates unlabeled P85 emission self-training.",
            "- C2-C1 isolates P87 structured knowledge under the same adaptation budget.",
            "- C2best-C2equal isolates the effect of longer structured convergence when supplied.",
            "- Decoder deltas are measured separately for each Student probability shape.",
            "",
            "```json",
            json.dumps(result["paired"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
