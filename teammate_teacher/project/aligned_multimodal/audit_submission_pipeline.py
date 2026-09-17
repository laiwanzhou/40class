from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "residual_skeleton_depth_imagenet_w040"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="核验当前 Test 提交的 ID、标签、CSV 与 checkpoint 链路")
    parser.add_argument("--official-test", type=Path, default=REPO_DIR / "Testing" / "test.csv")
    parser.add_argument("--class-mapping", type=Path, default=REPO_DIR / "class_mapping.csv")
    parser.add_argument("--training-manifest", type=Path, default=PROJECT_DIR / "data" / "manifest.csv")
    parser.add_argument("--submission", type=Path, default=DEFAULT_RUN_DIR / "test_predictions.csv")
    parser.add_argument("--detailed", type=Path, default=DEFAULT_RUN_DIR / "test_predictions_detailed.csv")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_RUN_DIR / "test_manifest.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_RUN_DIR / "test_predictions_summary.json")
    parser.add_argument("--skeleton-checkpoint", type=Path, default=PROJECT_DIR / "runs" / "skeleton_only" / "best_accuracy.pt")
    parser.add_argument("--depth-checkpoint", type=Path, default=PROJECT_DIR / "runs" / "depth_imagenet" / "best_accuracy.pt")
    parser.add_argument("--reproduced-submission", type=Path, default=None)
    parser.add_argument("--expected-depth-weight", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "runs" / "p0_six_modality_audit" / "submission_pipeline_audit.json")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_info(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "modalities": list(config.get("modalities", [])),
        "imagenet_pretrained": config.get("imagenet_pretrained"),
        "num_frames": config.get("num_frames"),
        "image_height": config.get("image_height"),
        "image_width": config.get("image_width"),
        "depth_representation": config.get("depth_representation"),
        "visual_normalization": config.get("visual_normalization", "legacy"),
        "seed": config.get("seed"),
        "final_refit": bool(checkpoint.get("final_refit", False)),
        "refit_training_samples": checkpoint.get("refit_training_samples"),
        "refit_training_subjects": checkpoint.get("refit_training_subjects"),
        "selection_protocol": checkpoint.get("selection_protocol"),
    }


def main() -> None:
    args = parse_args()
    official_columns, official = read_csv(args.official_test.resolve())
    submission_columns, submission = read_csv(args.submission.resolve())
    detailed_columns, detailed = read_csv(args.detailed.resolve())
    manifest_columns, manifest = read_csv(args.test_manifest.resolve())
    mapping_columns, mapping = read_csv(args.class_mapping.resolve())
    _, training_manifest = read_csv(args.training_manifest.resolve())
    summary = json.loads(args.summary.resolve().read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []
    checks.append(check("official_columns", official_columns == ["path", "prediction"], official_columns))
    checks.append(check("submission_columns", submission_columns == ["path", "prediction"], submission_columns))
    checks.append(check("official_row_count", len(official) == 405, len(official)))
    checks.append(check("submission_row_count", len(submission) == len(official), len(submission)))

    official_paths = [row["path"] for row in official]
    submission_paths = [row["path"] for row in submission]
    checks.append(check("official_paths_unique", len(set(official_paths)) == len(official_paths), len(set(official_paths))))
    checks.append(check("submission_path_order_exact", submission_paths == official_paths, "exact row-by-row comparison"))
    checks.append(check("submission_path_set_exact", set(submission_paths) == set(official_paths), "set comparison"))

    prediction_parse_errors: list[dict[str, str]] = []
    predictions: list[int] = []
    for index, row in enumerate(submission):
        raw = row.get("prediction", "")
        try:
            value = int(raw)
            if str(value) != raw.strip() or not 0 <= value < 40:
                raise ValueError(raw)
            predictions.append(value)
        except ValueError:
            prediction_parse_errors.append({"row": str(index + 2), "value": raw})
    checks.append(check("predictions_are_integer_0_39", not prediction_parse_errors, prediction_parse_errors[:10]))

    mapping_ids = [int(row["action_id"]) for row in mapping]
    mapping_names = [row["action_name"] for row in mapping]
    checks.append(check("class_mapping_columns", mapping_columns == ["action_id", "action_name"], mapping_columns))
    checks.append(check("class_mapping_contiguous_0_39", mapping_ids == list(range(40)), mapping_ids))
    checks.append(check("class_mapping_unique_names", len(set(mapping_names)) == 40, len(set(mapping_names))))

    expected_sample_ids = [Path(path.rstrip("/\\")).name for path in official_paths]
    detailed_sample_ids = [row["sample_id"] for row in detailed]
    manifest_sample_ids = [row["sample_id"] for row in manifest]
    detailed_predictions = [int(row["prediction"]) for row in detailed]
    checks.append(check("detailed_columns", detailed_columns == ["sample_id", "prediction", "confidence", "entropy", "top3"], detailed_columns))
    checks.append(check("detailed_sample_order_exact", detailed_sample_ids == expected_sample_ids, "exact row-by-row comparison"))
    checks.append(check("manifest_sample_order_exact", manifest_sample_ids == expected_sample_ids, "exact row-by-row comparison"))
    checks.append(check("detailed_predictions_match_submission", detailed_predictions == predictions, "exact row-by-row comparison"))
    checks.append(check("manifest_has_required_columns", all(name in manifest_columns for name in ["sample_id", "depth_dir", "skeleton_dir", "num_aligned_frames"]), manifest_columns))
    checks.append(check("manifest_all_have_aligned_frames", all(int(row["num_aligned_frames"]) > 0 for row in manifest), min(int(row["num_aligned_frames"]) for row in manifest)))

    actual_counts = {str(class_id): Counter(predictions)[class_id] for class_id in range(40)}
    checks.append(check("summary_num_samples", int(summary["num_samples"]) == len(predictions), summary["num_samples"]))
    checks.append(check("summary_depth_weight", abs(float(summary["depth_weight"]) - args.expected_depth_weight) < 1e-12, summary["depth_weight"]))
    checks.append(check("summary_class_counts", summary["class_counts"] == actual_counts, {"summary": summary["class_counts"], "actual": actual_counts}))

    skeleton = checkpoint_info(args.skeleton_checkpoint.resolve())
    depth = checkpoint_info(args.depth_checkpoint.resolve())
    checks.append(check("skeleton_checkpoint_modality", skeleton["modalities"] == ["skeleton"], skeleton["modalities"]))
    checks.append(check("depth_checkpoint_modality", depth["modalities"] == ["depth"], depth["modalities"]))
    checks.append(check("depth_checkpoint_is_imagenet", depth["imagenet_pretrained"] is True, depth["imagenet_pretrained"]))
    checks.append(check("expert_input_protocol_match", all(skeleton[key] == depth[key] for key in ["num_frames", "image_height", "image_width"]), {key: [skeleton[key], depth[key]] for key in ["num_frames", "image_height", "image_width"]}))
    checks.append(check("summary_skeleton_checkpoint_path", Path(summary["skeleton_checkpoint"]).resolve() == args.skeleton_checkpoint.resolve(), summary["skeleton_checkpoint"]))
    checks.append(check("summary_depth_checkpoint_path", Path(summary["depth_checkpoint"]).resolve() == args.depth_checkpoint.resolve(), summary["depth_checkpoint"]))

    reproduction: dict[str, Any] | None = None
    if args.reproduced_submission is not None:
        reproduced_columns, reproduced = read_csv(args.reproduced_submission.resolve())
        reproduced_predictions = [int(row["prediction"]) for row in reproduced]
        reproduction = {
            "path": str(args.reproduced_submission.resolve()),
            "columns": reproduced_columns,
            "rows": len(reproduced),
            "paths_exact": [row["path"] for row in reproduced] == official_paths,
            "predictions_exact": reproduced_predictions == predictions,
            "prediction_differences": int(sum(a != b for a, b in zip(reproduced_predictions, predictions))),
        }
        checks.append(check("reproduced_submission_exact", reproduction["paths_exact"] and reproduction["predictions_exact"], reproduction))

    split_counts = Counter(row["split"] for row in training_manifest)
    split_subjects: dict[str, list[str]] = {}
    for split in sorted(split_counts):
        split_subjects[split] = sorted({row["user_id"] for row in training_manifest if row["split"] == split})
    both_final_refit = bool(skeleton["final_refit"] and depth["final_refit"])
    if both_final_refit:
        refit_subjects = sorted(
            set(skeleton["refit_training_subjects"] or [])
            | set(depth["refit_training_subjects"] or [])
        )
        current_training_samples = min(
            int(skeleton["refit_training_samples"] or 0),
            int(depth["refit_training_samples"] or 0),
        )
        current_training_subjects = len(refit_subjects)
        scope_note = (
            "Both experts are fixed-epoch final refits on all labelled subjects; no validation split was used "
            "to select a checkpoint during final refit."
        )
    else:
        refit_subjects = []
        current_training_samples = split_counts.get("train", 0)
        current_training_subjects = len(split_subjects.get("train", []))
        scope_note = (
            "This is not a CSV/checkpoint mismatch, but the public submission experts are fixed-validation "
            "checkpoints, not final models refit on all labelled subjects."
        )
    training_scope = {
        "manifest": str(args.training_manifest.resolve()),
        "split_counts": dict(split_counts),
        "split_subjects": split_subjects,
        "checkpoint_training_behavior": "train.py trains on split=train and selects checkpoint on split=val",
        "both_experts_final_refit": both_final_refit,
        "current_submission_expert_training_samples": current_training_samples,
        "current_submission_expert_training_subjects": current_training_subjects,
        "current_submission_refit_subjects": refit_subjects,
        "held_out_validation_samples_not_used_for_weight_updates": 0 if both_final_refit else split_counts.get("val", 0),
        "held_out_validation_subjects_not_used_for_weight_updates": [] if both_final_refit else split_subjects.get("val", []),
        "note": scope_note,
    }

    result = {
        "all_engineering_checks_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "submission": {
            "path": str(args.submission.resolve()),
            "rows": len(predictions),
            "predicted_classes": len(set(predictions)),
            "missing_predicted_classes": [class_id for class_id in range(40) if class_id not in set(predictions)],
            "class_counts": actual_counts,
        },
        "checkpoints": {"skeleton": skeleton, "depth": depth},
        "reproduction": reproduction,
        "training_scope": training_scope,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_engineering_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
