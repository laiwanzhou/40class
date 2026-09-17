from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import p89_validate_final_nometa_pipeline as validation
from audit_p87_sequence_decoder import classification_metrics
from p46_protocol import HARD_CLASS_IDS
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
P87_DIR = PROJECT_DIR / "runs/p87s_final_test_predictions_v1"
P89_DIR = PROJECT_DIR / "runs/p89_final_test_predictions_nometa_v2"
OUTPUT = PROJECT_DIR / "runs/p89_family_gated_repair_v1"
HARD_CLASSES = np.asarray(HARD_CLASS_IDS, dtype=np.int64)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def predictions(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)


def write_submission(
    path: Path, source_rows: list[dict[str, str]], prediction: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for row, value in zip(source_rows, prediction, strict=True):
            writer.writerow({"path": row["path"], "prediction": int(value)})


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def apply_family_gate(base: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gate = np.isin(base, HARD_CLASSES) & (candidate != base)
    output = base.copy()
    output[gate] = candidate[gate]
    return output, gate


def validation_metrics(
    split: str,
    run: str,
    users: list[str],
    values: np.lib.npyio.NpzFile,
) -> dict[str, dict[str, object]]:
    protocol = validation.protocol(run, users)
    labels = protocol[1]
    p87 = protocol[3]
    detail, _ = validation.decode(values[f"{split}_probability"], protocol)
    candidates = {
        "detail": detail,
        "template": values[f"{split}_template_prediction"],
    }
    output = {}
    for name, candidate in candidates.items():
        gated, gate = apply_family_gate(p87, candidate)
        output[name] = {
            "metrics": classification_metrics(labels, gated),
            "rescue_harm_vs_p87": rescue_harm(labels, p87, gated),
            "changes": int(gate.sum()),
        }
    return output


def main() -> None:
    p87_rows = read_rows(P87_DIR / "submission_p87s_student_decoded.csv")
    p87 = predictions(p87_rows)
    source_candidates = {
        "detail": predictions(read_rows(P89_DIR / "submission_p89_detail_global.csv")),
        "template": predictions(read_rows(P89_DIR / "submission_p89_template.csv")),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    test_report = {}
    test_predictions = {}
    for name, candidate in source_candidates.items():
        gated, gate = apply_family_gate(p87, candidate)
        path = OUTPUT / f"submission_p89_family_gated_{name}.csv"
        write_submission(path, p87_rows, gated)
        invalid_changes = (candidate != p87) & ~np.isin(p87, HARD_CLASSES)
        test_report[name] = {
            "original_changes_vs_p87": int(np.sum(candidate != p87)),
            "removed_outside_family_changes": int(invalid_changes.sum()),
            "gated_changes_vs_p87": int(gate.sum()),
            "path": str(path.resolve()),
            "sha256": digest(path),
        }
        test_predictions[name] = gated

    values = np.load(PROJECT_DIR / "runs/p89_final_nometa_validation_v1/predictions.npz")
    report = {
        "stage": "P89_repair_missing_Detail21_deployment_gate_v1",
        "cause": (
            "Validation applied the specialist only to rows whose true label was "
            "in Detail21, while Test applied it to every readable row. This repair "
            "uses the deployable P87 top-1 family decision and only allows within-"
            "Detail21 reordering."
        ),
        "H1": validation_metrics(
            "h1",
            "p87s_fusion_holdout1_c7_structured12_v1",
            validation.H1_USERS,
            values,
        ),
        "H2": validation_metrics(
            "h2",
            "p87s_fusion_confirm2_c2_structured12_v1",
            validation.H2_USERS,
            values,
        ),
        "Test": test_report,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "prediction_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = (
            "row",
            "sample_id",
            "p87",
            "p89_detail",
            "p89_template",
            "family_gated_detail",
            "family_gated_template",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(p87)):
            writer.writerow(
                {
                    "row": index,
                    "sample_id": f"SM_test_{index + 1:04d}",
                    "p87": int(p87[index]),
                    "p89_detail": int(source_candidates["detail"][index]),
                    "p89_template": int(source_candidates["template"][index]),
                    "family_gated_detail": int(test_predictions["detail"][index]),
                    "family_gated_template": int(test_predictions["template"][index]),
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
