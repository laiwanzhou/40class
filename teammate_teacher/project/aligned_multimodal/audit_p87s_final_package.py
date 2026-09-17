from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p87s_deploy_model import load_p87s_deploy_checkpoint
from predict_p87s_test_student import load_decoder


PROJECT_DIR = Path(__file__).resolve().parent
RUNS = PROJECT_DIR / "runs"
DEFAULT_CHECKPOINT = RUNS / "p87s_test_adapt_structured12_v1/unified_student.pt"
DEFAULT_DECODER = RUNS / "p87s_tiny_decoder_v1/tiny_decoder.npz"
DEFAULT_PREDICTIONS = RUNS / "p87s_final_test_predictions_v1"
DEFAULT_EQUIVALENCE = RUNS / "p87s_deployment_equivalence_v1/summary.json"
DEFAULT_OUTPUT = RUNS / "p87s_final_package_audit_v1"
DEFAULT_STRUCTURED_TARGETS = (
    RUNS / "p87s_test_structured_targets_v1/structured_targets.npz"
)
DEFAULT_TEST_CSV = PROJECT_DIR.parent / "Testing/test.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the final <=100 MB P87-S Student plus tiny decoder package."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tiny-decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--equivalence-summary", type=Path, default=DEFAULT_EQUIVALENCE)
    parser.add_argument(
        "--structured-targets", type=Path, default=DEFAULT_STRUCTURED_TARGETS
    )
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-package-bytes", type=int, default=100_000_000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_id(path_value: str) -> str:
    clean = path_value.replace("\\", "/").rstrip("/")
    return clean.rsplit("/", 1)[-1]


def audit_submission(
    name: str,
    path: Path,
    official_paths: list[str],
) -> dict[str, Any]:
    columns, rows = read_csv(path)
    paths = [row.get("path", "") for row in rows]
    predictions: list[int] = []
    valid = True
    for row in rows:
        value = row.get("prediction", "")
        try:
            parsed = int(value)
        except ValueError:
            valid = False
            continue
        valid &= str(parsed) == value.strip() and 0 <= parsed < 40
        predictions.append(parsed)
    checks = {
        "columns_exact": columns == ["path", "prediction"],
        "rows_exact": len(rows) == 405,
        "official_path_order_exact": paths == official_paths,
        "paths_unique": len(set(paths)) == len(paths),
        "integer_prediction_0_39": valid and len(predictions) == 405,
    }
    counts = Counter(predictions)
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "predicted_classes": len(counts),
        "missing_predicted_classes": sorted(set(range(40)) - set(counts)),
        "class_histogram": {
            str(class_id): int(counts[class_id])
            for class_id in range(40)
            if counts[class_id]
        },
    }


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    decoder_path = args.tiny_decoder.resolve()
    predictions_dir = args.predictions_dir.resolve()
    adapted_dir = checkpoint_path.parent
    fusion_dir = Path(
        str(read_json(adapted_dir / "summary.json")["base_checkpoint"])
    ).resolve().parent
    test_sequence_dir = RUNS / "p87s_test_mc3_sequence_v1"
    visual_dir = RUNS / "p87s_visual_all2914_v1"
    structured_path = args.structured_targets.resolve()

    model, checkpoint = load_p87s_deploy_checkpoint(checkpoint_path)
    transition, decoder_config = load_decoder(decoder_path)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if transition.bigram_log_probability.shape != (40, 40):
        raise RuntimeError("tiny decoder did not load with the 40-class contract")
    checkpoint_bytes = checkpoint_path.stat().st_size
    decoder_bytes = decoder_path.stat().st_size
    package_bytes = checkpoint_bytes + decoder_bytes

    _, official_rows = read_csv(args.test_csv)
    official_paths = [row["path"] for row in official_rows]
    official_ids = [official_id(value) for value in official_paths]
    submissions = [
        audit_submission(
            "raw", predictions_dir / "submission_p87s_student_raw.csv", official_paths
        ),
        audit_submission(
            "decoded",
            predictions_dir / "submission_p87s_student_decoded.csv",
            official_paths,
        ),
    ]

    visual_summary = read_json(visual_dir / "summary.json")
    sequence_summary = read_json(test_sequence_dir / "summary.json")
    fusion_summary = read_json(fusion_dir / "summary.json")
    adapted_summary = read_json(adapted_dir / "summary.json")
    decoder_summary = read_json(decoder_path.parent / "summary.json")
    prediction_summary = read_json(predictions_dir / "summary.json")
    equivalence = read_json(args.equivalence_summary)
    with np.load(structured_path, allow_pickle=False) as targets:
        target_ids = np.asarray(targets["sample_ids"]).astype(str)
        target_mask_count = int(np.asarray(targets["target_mask"], dtype=bool).sum())

    provenance_checks = [
        check(
            "visual_terminal_refit",
            visual_summary.get("stage") == "P87S_visual_all2914_refit"
            and visual_summary.get("counts") == {"train": 2914, "validation": 0},
            visual_summary.get("counts"),
        ),
        check(
            "test_sequence_bound_to_visual",
            sequence_summary.get("checkpoint_sha256")
            == sha256(visual_dir / "visual_student.pt"),
            sequence_summary.get("checkpoint_sha256"),
        ),
        check(
            "fusion_terminal_refit",
            fusion_summary.get("stage") == "P87S_mobind_fusion_all2914_refit",
            fusion_summary.get("stage"),
        ),
        check(
            "adaptation_bound_to_base",
            adapted_summary.get("base_checkpoint_sha256")
            == sha256(fusion_dir / "unified_student.pt"),
            adapted_summary.get("base_checkpoint_sha256"),
        ),
        check(
            "adaptation_bound_to_structured_targets",
            adapted_summary.get("structured_targets_sha256") == sha256(structured_path)
            and adapted_summary.get("pseudo_rows") == 401,
            adapted_summary.get("pseudo_rows"),
        ),
        check(
            "structured_target_universe",
            len(target_ids) == 401
            and target_mask_count == 401
            and set(target_ids).issubset(set(official_ids)),
            {"rows": len(target_ids), "mask": target_mask_count},
        ),
        check(
            "prediction_bound_to_student",
            prediction_summary.get("student_checkpoint_sha256")
            == sha256(checkpoint_path),
            prediction_summary.get("student_checkpoint_sha256"),
        ),
        check(
            "prediction_bound_to_decoder",
            prediction_summary.get("tiny_decoder_sha256") == sha256(decoder_path),
            prediction_summary.get("tiny_decoder_sha256"),
        ),
        check(
            "decoder_frozen_from_train_only",
            decoder_summary.get("train_rows") == 2914
            and decoder_summary.get("large_model_required_at_inference") is False
            and "Test labels/emissions are never read"
            in str(decoder_summary.get("protocol", "")),
            {
                "train_rows": decoder_summary.get("train_rows"),
                "protocol": decoder_summary.get("protocol"),
            },
        ),
        check(
            "raw_input_equivalence",
            equivalence.get("status") == "passed"
            and equivalence.get("prediction_agreement") == 1.0,
            equivalence.get("prediction_agreement"),
        ),
        check(
            "all_405_without_large_fallback",
            prediction_summary.get("test_rows") == 405
            and prediction_summary.get("large_model_required_at_inference") is False
            and prediction_summary.get("large_or_legacy_fallback_rows") == 0,
            {
                "rows": prediction_summary.get("test_rows"),
                "fallback": prediction_summary.get("large_or_legacy_fallback_rows"),
            },
        ),
    ]
    package_checks = [
        check(
            "adapted_checkpoint_stage",
            checkpoint.get("stage") == "P87S_label_free_test_adaptation",
            checkpoint.get("stage"),
        ),
        check(
            "self_contained_model_config",
            isinstance(checkpoint.get("deployment_model_config"), dict),
            sorted(checkpoint.get("deployment_model_config", {})),
        ),
        check(
            "student_parameter_contract",
            parameters == 23_560_564,
            parameters,
        ),
        check(
            "strict_decimal_100MB_package",
            package_bytes <= args.max_package_bytes,
            {"bytes": package_bytes, "limit": args.max_package_bytes},
        ),
        check(
            "decoder_config_frozen",
            decoder_config.gap_seconds == 30.0
            and decoder_config.transition_weight == 0.25
            and decoder_config.trigram_backoff == 5.0
            and decoder_config.beam_width == 50,
            {
                "gap_seconds": decoder_config.gap_seconds,
                "transition_weight": decoder_config.transition_weight,
                "trigram_backoff": decoder_config.trigram_backoff,
                "beam_width": decoder_config.beam_width,
            },
        ),
    ]
    all_passed = (
        all(item["passed"] for item in provenance_checks)
        and all(item["passed"] for item in package_checks)
        and all(item["all_checks_passed"] for item in submissions)
    )
    summary = {
        "stage": "P87S_final_package_audit",
        "status": "passed" if all_passed else "failed",
        "all_checks_passed": all_passed,
        "official_test_rows": len(official_rows),
        "student": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_bytes,
            "sha256": sha256(checkpoint_path),
            "parameters": parameters,
            "fp32_parameter_mib": parameters * 4 / 1024**2,
        },
        "tiny_decoder": {
            "path": str(decoder_path),
            "bytes": decoder_bytes,
            "sha256": sha256(decoder_path),
        },
        "package": {
            "bytes": package_bytes,
            "decimal_mb": package_bytes / 1_000_000,
            "mib": package_bytes / 1024**2,
            "max_bytes": args.max_package_bytes,
        },
        "submissions": submissions,
        "package_checks": package_checks,
        "provenance_checks": provenance_checks,
        "runtime_cache_required_for_deployment": False,
        "test_labels_read": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
