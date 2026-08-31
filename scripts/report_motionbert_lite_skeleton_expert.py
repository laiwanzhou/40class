from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_ir_depth_videomaev2_p2a import _metrics
from scripts.fetch_motionbert_lite_checkpoint import file_sha256
from src.experiments.motionbert_p6b_config import (
    load_motionbert_p6b_config,
    project_path,
)


METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "nll",
    "worst_user_accuracy",
    "top3_accuracy",
    "top5_accuracy",
)


def _assert_metrics(stored: dict[str, Any], current: dict[str, Any]) -> None:
    for key in METRIC_KEYS:
        if key not in stored or not np.isclose(
            float(stored[key]), float(current[key]), atol=1e-12, rtol=0.0
        ):
            raise RuntimeError(f"MotionBERT B1 metric mismatch: {key}")


def build_motionbert_p6b_report(config_path: Path) -> dict[str, Any]:
    config = load_motionbert_p6b_config(config_path)
    smoke_path = project_path(str(config["outputs"]["smoke_report"]))
    b1_path = project_path(str(config["outputs"]["b1_report"]))
    if not smoke_path.is_file() or not b1_path.is_file():
        raise FileNotFoundError("MotionBERT P6-B smoke or B1 report is missing")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    b1 = json.loads(b1_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "smoke_passed" or smoke.get(
        "pretrained_element_coverage"
    ) != 1.0:
        raise RuntimeError("MotionBERT smoke evidence changed")
    if b1.get("epochs_completed") != 20 or b1.get("validation_evaluation_count") != 1:
        raise RuntimeError("MotionBERT B1 execution contract changed")
    train_path = Path(str(b1["train_predictions"]))
    validation_path = Path(str(b1["validation_predictions"]))
    with np.load(train_path, allow_pickle=False) as train:
        train_values = {name: train[name].copy() for name in train.files}
    with np.load(validation_path, allow_pickle=False) as validation:
        validation_values = {
            name: validation[name].copy() for name in validation.files
        }
    if len(train_values["sample_ids"]) != 2039 or len(
        np.unique(train_values["sample_ids"].astype(str))
    ) != 2039:
        raise RuntimeError("MotionBERT train prediction population changed")
    if len(validation_values["sample_ids"]) != 388 or len(
        np.unique(validation_values["sample_ids"].astype(str))
    ) != 388:
        raise RuntimeError("MotionBERT validation prediction population changed")
    if set(validation_values["labels"].astype(int).tolist()) != set(range(40)):
        raise RuntimeError("MotionBERT validation class coverage changed")
    if not np.isfinite(train_values["logits"]).all() or not np.isfinite(
        validation_values["logits"]
    ).all():
        raise RuntimeError("MotionBERT predictions contain non-finite logits")
    train_metrics = _metrics(
        train_values["labels"], train_values["logits"], train_values["user_ids"]
    )
    validation_metrics = _metrics(
        validation_values["labels"],
        validation_values["logits"],
        validation_values["user_ids"],
    )
    _assert_metrics(b1["train_metrics"], train_metrics)
    _assert_metrics(b1["validation_metrics"], validation_metrics)

    visual_path = project_path(str(config["data"]["visual_validation_predictions"]))
    with np.load(visual_path, allow_pickle=False) as visual:
        if not np.array_equal(
            visual["sample_ids"].astype(str),
            validation_values["sample_ids"].astype(str),
        ):
            raise RuntimeError("MotionBERT/visual sample order changed")
        visual_logits = visual["logits"].astype(np.float32)
    labels = validation_values["labels"].astype(np.int64)
    users = validation_values["user_ids"].astype(str)
    visual_correct = visual_logits.argmax(1) == labels
    motion_correct = validation_values["logits"].argmax(1) == labels
    rescued = (~visual_correct) & motion_correct
    harmed = visual_correct & (~motion_correct)
    oracle = visual_correct | motion_correct
    rescues_by_user = {
        user: int((rescued & (users == user)).sum()) for user in sorted(set(users))
    }
    visual_comparison = {
        "unique_rescues": int(rescued.sum()),
        "harms": int(harmed.sum()),
        "net": int(rescued.sum() - harmed.sum()),
        "oracle_accuracy": float(oracle.mean()),
        "rescues_by_user": rescues_by_user,
    }
    gates_config = config["b1"]["gates"]
    gates = {
        "accuracy": float(validation_metrics["accuracy"])
        >= float(gates_config["accuracy"]),
        "macro_f1": float(validation_metrics["macro_f1"])
        >= float(gates_config["macro_f1"]),
        "unique_rescues": int(rescued.sum())
        >= int(gates_config["unique_rescues"]),
        "visual_oracle_accuracy": float(oracle.mean())
        >= float(gates_config["visual_oracle_accuracy"]),
        "unique_rescue_each_user": all(
            value >= int(gates_config["unique_rescue_each_user"])
            for value in rescues_by_user.values()
        ),
    }
    if gates != b1["gates"] or all(gates.values()) != b1["b1_passed"]:
        raise RuntimeError("MotionBERT B1 gate recomputation changed")
    b2_path = project_path(str(config["outputs"]["b2_report"]))
    if not b1["b1_passed"] and b2_path.exists():
        raise RuntimeError("B2 evidence exists after failed B1 gate")

    cache_path = Path(str(b1["cache"]["path"]))
    if file_sha256(cache_path) != b1["cache"]["sha256"]:
        raise RuntimeError("MotionBERT embedding cache hash changed")
    with np.load(cache_path, allow_pickle=False) as cache:
        embeddings = cache["embeddings"].astype(np.float32)
        available = cache["available"].astype(bool)
        partition = cache["partition"].astype(str)
    supported_train = available & (partition == "train")
    embedding_std = embeddings[supported_train].std(axis=0)
    report = {
        "stage": "P6-B",
        "status": "stopped_after_b1_rejection",
        "candidate": "motionbert_lite",
        "development_validation": True,
        "independent_final_test": False,
        "metrics_recomputed_from_archives": True,
        "smoke": smoke,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "visual_comparison": visual_comparison,
        "b1_gates": gates,
        "b1_passed": bool(b1["b1_passed"]),
        "b2_executed": b2_path.exists(),
        "fusion_qualified": False,
        "embedding_diagnostics": {
            "cache_sha256": b1["cache"]["sha256"],
            "supported_train_rows": int(supported_train.sum()),
            "mean_dimension_std": float(embedding_std.mean()),
            "zero_variance_dimensions": int((embedding_std <= 1e-8).sum()),
        },
        "artifacts": {
            "b1_report": str(b1_path),
            "train_predictions": str(train_path),
            "train_predictions_sha256": file_sha256(train_path),
            "validation_predictions": str(validation_path),
            "validation_predictions_sha256": file_sha256(validation_path),
        },
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    validation = report["validation_metrics"]
    comparison = report["visual_comparison"]
    lines = [
        "# MotionBERT-Lite Skeleton Expert P6-B Result",
        "",
        "- Status: `stopped_after_b1_rejection`",
        "- B1 passed: `False`",
        "- B2 executed: `False`",
        "- Fusion qualified: `False`",
        "",
        "## B1 validation",
        "",
        f"- Accuracy: `{validation['accuracy']:.6f}`",
        f"- Macro-F1: `{validation['macro_f1']:.6f}`",
        f"- Worst-user Accuracy: `{validation['worst_user_accuracy']:.6f}`",
        f"- Predicted classes: `{validation['predicted_class_count']}/40`",
        f"- Zero-recall classes: `{validation['zero_recall_classes']}/40`",
        "",
        "## Visual complementarity",
        "",
        f"- Unique rescues: `{comparison['unique_rescues']}`",
        f"- Harms: `{comparison['harms']}`",
        f"- Net: `{comparison['net']}`",
        f"- Visual + MotionBERT oracle Accuracy: `{comparison['oracle_accuracy']:.6f}`",
        "",
        "All fixed B1 gates except unique rescue coverage across both users failed. "
        "The frozen pretrained representation is not qualified for partial fine-tuning "
        "or multimodal fusion under this experiment.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report MotionBERT-Lite P6-B")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml",
    )
    args = parser.parse_args()
    report = build_motionbert_p6b_report(args.config.resolve())
    output = project_path(
        str(load_motionbert_p6b_config(args.config.resolve())["outputs"]["final_report"])
    )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(_markdown(report), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "b1_passed": report["b1_passed"],
                "fusion_qualified": report["fusion_qualified"],
            }
        )
    )


if __name__ == "__main__":
    main()
