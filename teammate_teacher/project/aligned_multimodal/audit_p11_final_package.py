from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit P11 local candidates and the self-contained model package."
    )
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--fixed-csv", type=Path, required=True)
    parser.add_argument("--routed-csv", type=Path, required=True)
    parser.add_argument(
        "--model-file", type=Path, action="append", required=True
    )
    parser.add_argument("--max-model-mib", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_candidate(
    name: str,
    path: Path,
    official_paths: list[str],
) -> dict[str, object]:
    columns, rows = read_csv(path)
    paths = [row.get("path", "") for row in rows]
    prediction_text = [row.get("prediction", "") for row in rows]
    valid_integer = True
    predictions: list[int] = []
    for value in prediction_text:
        try:
            parsed = int(value)
        except ValueError:
            valid_integer = False
            continue
        valid_integer &= 0 <= parsed < 40 and str(parsed) == value.strip()
        predictions.append(parsed)
    checks = {
        "columns_exact": columns == ["path", "prediction"],
        "row_count_exact": len(rows) == len(official_paths),
        "path_order_exact": paths == official_paths,
        "paths_unique": len(set(paths)) == len(paths),
        "predictions_integer_0_39": valid_integer
        and len(predictions) == len(rows),
    }
    return {
        "name": name,
        "path": str(path.resolve()),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "rows": len(rows),
        "predicted_classes": len(set(predictions)),
        "missing_predicted_classes": sorted(set(range(40)) - set(predictions)),
        "class_counts": {
            str(class_id): int(Counter(predictions).get(class_id, 0))
            for class_id in range(40)
        },
    }


def audit_model(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    suffixes = "".join(resolved.suffixes).lower()
    if resolved.suffix.lower() == ".pt":
        value = torch.load(resolved, map_location="cpu", weights_only=False)
        load_check = (
            isinstance(value, dict)
            and isinstance(value.get("model_state_dict"), dict)
        )
        metadata = {
            "storage_dtype": value.get("storage_dtype", "original"),
            "epoch": int(value.get("epoch", -1)),
        }
    elif suffixes.endswith(".joblib"):
        value = joblib.load(resolved)
        load_check = hasattr(value, "predict") or hasattr(value, "predict_proba")
        metadata = {"estimator_type": type(value).__name__}
    else:
        raise ValueError(f"Unsupported model file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "size_mib": resolved.stat().st_size / 2**20,
        "sha256": sha256(resolved),
        "load_check_passed": bool(load_check),
        **metadata,
    }


def main() -> None:
    args = parse_args()
    columns, official_rows = read_csv(args.official_test)
    if columns != ["path", "prediction"]:
        raise ValueError("Unexpected official test columns.")
    official_paths = [row["path"] for row in official_rows]
    candidates = [
        audit_candidate("fixed_thermal", args.fixed_csv, official_paths),
        audit_candidate("routed_thermal", args.routed_csv, official_paths),
    ]
    models = [audit_model(path) for path in args.model_file]
    total_bytes = sum(int(model["bytes"]) for model in models)
    result = {
        "all_engineering_checks_passed": (
            all(candidate["all_checks_passed"] for candidate in candidates)
            and all(model["load_check_passed"] for model in models)
            and total_bytes / 2**20 <= float(args.max_model_mib)
        ),
        "official_test_rows": len(official_rows),
        "candidates": candidates,
        "model_package": {
            "files": models,
            "file_count": len(models),
            "total_bytes": total_bytes,
            "total_mib": total_bytes / 2**20,
            "total_decimal_mb": total_bytes / 1_000_000,
            "max_model_mib": float(args.max_model_mib),
            "within_limit": total_bytes / 2**20
            <= float(args.max_model_mib),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_engineering_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
