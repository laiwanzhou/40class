"""Build, run and summarize the fixed-budget P99 H1 Student transfer probe.

The generated target files deliberately contain no ground-truth label array.
All branches start from the same P87-S C0 checkpoint and differ only in the
teacher probability target.  H2 and H3 have no path in this runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/p99_transfer_probe_h1.json"
NUM_CLASSES = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P99 fixed-budget Student transfer probe")
    parser.add_argument("--mode", choices=("build", "run", "summarize", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != NUM_CLASSES:
        raise ValueError(f"expected [N,{NUM_CLASSES}] probability, got {probability.shape}")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("target probability is invalid")
    probability = np.clip(probability, 1e-8, None)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability.astype(np.float32)


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    values = normalize_probability(probability).astype(np.float64)
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum(axis=1)
    return (entropy / np.log(NUM_CLASSES)).astype(np.float32)


def write_target(
    path: Path,
    sample_ids: np.ndarray,
    users: np.ndarray,
    probability: np.ndarray,
    source_name: str,
) -> dict[str, Any]:
    probability = normalize_probability(probability)
    if len(sample_ids) != len(users) or len(sample_ids) != len(probability):
        raise ValueError("target ids/users/probabilities differ in length")
    prediction = probability.argmax(axis=1).astype(np.int64)
    confidence = (1.0 - normalized_entropy(probability)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.asarray(sample_ids).astype(str),
        users=np.asarray(users).astype(str),
        target_mask=np.ones(len(sample_ids), dtype=bool),
        decoded_mask=np.zeros(len(sample_ids), dtype=bool),
        emission_probability=probability,
        structured_probability=probability,
        structured_distillation_probability=probability,
        structured_distillation_weight=np.ones(len(sample_ids), dtype=np.float32),
        emission_prediction=prediction,
        structured_map_prediction=prediction,
        structured_marginal_prediction=prediction,
        structured_distillation_prediction=prediction,
        structured_confidence=confidence,
        sequence_session_id=np.full(len(sample_ids), -1, dtype=np.int64),
        sequence_changed=np.zeros(len(sample_ids), dtype=bool),
        teacher_source=np.asarray(source_name),
    )
    # Ground-truth labels must not enter the adaptation artifact.
    with np.load(path, allow_pickle=False) as written:
        if "labels" in written.files or "label" in written.files:
            raise RuntimeError("ground truth leaked into a P99 transfer target")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(sample_ids)),
        "hard_prediction_histogram": np.bincount(prediction, minlength=NUM_CLASSES).tolist(),
        "mean_confidence": float(confidence.mean()),
    }


def load_prediction_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def build_targets(config: dict[str, Any], output: Path) -> dict[str, Any]:
    d0_path = resolve(config["d0_predictions"])
    t1_path = resolve(config["t1_predictions"])
    d0 = load_prediction_artifact(d0_path)
    t1 = load_prediction_artifact(t1_path)
    for key in ("sample_ids", "labels", "users"):
        if not np.array_equal(d0[key], t1[key]):
            raise RuntimeError(f"D0/T1 {key} alignment differs")
    sample_ids = t1["sample_ids"].astype(str)
    users = t1["users"].astype(str)
    target_dir = output / "targets"
    targets = {
        "anchor": write_target(
            target_dir / "anchor_targets.npz", sample_ids, users,
            t1["anchor_probability"], "P91_source_safe_anchor_fixed_0.94",
        ),
        "depth_d0": write_target(
            target_dir / "depth_d0_targets.npz", sample_ids, users,
            d0["direct_probability"], "P99_D0_roi_all_source_OOF",
        ),
        "anchor_depth_t1": write_target(
            target_dir / "anchor_depth_t1_targets.npz", sample_ids, users,
            t1["direct_probability"], "P99_T1_anchor_depth_inner_OOF_pool",
        ),
    }
    manifest = {
        "stage": "P99_H1_transfer_target_build",
        "status": "complete",
        "labels_written_to_targets": False,
        "rows": int(len(sample_ids)),
        "users": sorted(set(users.tolist())),
        "sources": {"d0": str(d0_path), "t1": str(t1_path)},
        "targets": targets,
    }
    (output / "target_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def branch_command(
    config: dict[str, Any], target: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        str(HERE / "adapt_p87s_structured_student.py"),
        "--base-checkpoint", str(resolve(config["base_checkpoint"])),
        "--structured-targets", str(target),
        "--target", "emission",
        "--epochs", str(int(config["epochs"])),
        "--batch-size", str(int(config["batch_size"])),
        "--workers", str(int(config["workers"])),
        "--fusion-learning-rate", str(float(config["fusion_learning_rate"])),
        "--visual-head-learning-rate", str(float(config["visual_head_learning_rate"])),
        "--minimum-learning-rate", str(float(config["minimum_learning_rate"])),
        "--weight-decay", str(float(config["weight_decay"])),
        "--temperature", str(float(config["temperature"])),
        "--seed", str(int(config["seed"])),
        "--output-dir", str(output),
    ]


def run_branches(config: dict[str, Any], output: Path) -> None:
    manifest_path = output / "target_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("build transfer targets before running branches")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in config["targets"]:
        target = Path(manifest["targets"][name]["path"])
        branch_output = output / f"student_{name}"
        summary_path = branch_output / "summary.json"
        if summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "formal"
                and Path(previous["structured_targets"]).resolve() == target.resolve()
                and int(previous["config"]["epochs"]) == int(config["epochs"])
            ):
                print(f"reuse complete branch {name}: {branch_output}", flush=True)
                continue
            raise RuntimeError(f"incompatible existing branch output: {branch_output}")
        command = branch_command(config, target, branch_output)
        print(json.dumps({"branch": name, "command": command}, ensure_ascii=False), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)


def read_base_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([row["sample_id"] for row in rows]),
        np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        np.asarray([row["user_id"] for row in rows]),
    )


def probability_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probability = normalize_probability(probability)
    prediction = probability.argmax(axis=1)
    top5 = np.argpartition(-probability, kth=4, axis=1)[:, :5]
    return {
        "correct": int(np.sum(prediction == labels)),
        "total": int(len(labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "top5": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=np.arange(NUM_CLASSES)
        ).astype(int).tolist(),
    }


def rescue_harm(
    labels: np.ndarray, base_prediction: np.ndarray, candidate_prediction: np.ndarray
) -> dict[str, int]:
    return {
        "rescue": int(np.sum((base_prediction != labels) & (candidate_prediction == labels))),
        "harm": int(np.sum((base_prediction == labels) & (candidate_prediction != labels))),
        "net": int(
            np.sum(candidate_prediction == labels) - np.sum(base_prediction == labels)
        ),
        "changed": int(np.sum(base_prediction != candidate_prediction)),
    }


def target_probability(path: Path, ids: np.ndarray, key: str = "emission_probability") -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        source_ids = source["sample_ids"].astype(str)
        mask = source["target_mask"].astype(bool)
        probability = source[key].astype(np.float32)
    lookup = {value: index for index, value in enumerate(source_ids[mask])}
    missing = [value for value in ids if value not in lookup]
    if missing:
        raise KeyError(f"target misses {len(missing)} base rows")
    return probability[mask][[lookup[value] for value in ids]]


def summarize_branch(
    name: str,
    run_dir: Path,
    target_path: Path,
    ids: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    anchor_student_prediction: np.ndarray | None,
    target_key: str = "emission_probability",
) -> tuple[dict[str, Any], np.ndarray]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    logits = np.asarray(np.load(run_dir / "subject_holdout_logits.npy"), dtype=np.float64)
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = probability.argmax(axis=1)
    teacher_probability = target_probability(target_path, ids, key=target_key)
    teacher_prediction = teacher_probability.argmax(axis=1)
    student_metrics = probability_metrics(probability, labels)
    teacher_metrics = probability_metrics(teacher_probability, labels)
    per_user: dict[str, Any] = {}
    for user in sorted(set(users.tolist())):
        selected = users == user
        per_user[user] = {
            "rows": int(selected.sum()),
            "teacher_correct": int(np.sum(teacher_prediction[selected] == labels[selected])),
            "student_correct": int(np.sum(prediction[selected] == labels[selected])),
            "student_vs_anchor_control": (
                None
                if anchor_student_prediction is None
                else int(
                    np.sum(prediction[selected] == labels[selected])
                    - np.sum(anchor_student_prediction[selected] == labels[selected])
                )
            ),
        }
    checkpoint = run_dir / "unified_student.pt"
    result = {
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
        "teacher_to_student_correct_gap": int(
            teacher_metrics["correct"] - student_metrics["correct"]
        ),
        "student_target_agreement": float(np.mean(prediction == teacher_prediction)),
        "student_correct_when_teacher_wrong": int(
            np.sum((teacher_prediction != labels) & (prediction == labels))
        ),
        "teacher_correct_not_absorbed": int(
            np.sum((teacher_prediction == labels) & (prediction != labels))
        ),
        "checkpoint_bytes": int(checkpoint.stat().st_size),
        "checkpoint_under_100MB": bool(checkpoint.stat().st_size < 100_000_000),
        "per_user": per_user,
        "training_ground_truth_fields_seen": int(
            summary["ground_truth_fields_seen_during_training"]
        ),
    }
    if anchor_student_prediction is not None:
        result["student_vs_anchor_control"] = rescue_harm(
            labels, anchor_student_prediction, prediction
        )
    return result, prediction


def summarize(config: dict[str, Any], output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "target_manifest.json").read_text(encoding="utf-8"))
    base_dir = resolve(config["base_checkpoint"]).parent
    ids, labels, users = read_base_rows(base_dir / "subject_holdout_predictions.csv")
    branch_targets = {
        name: Path(manifest["targets"][name]["path"]) for name in config["targets"]
    }
    anchor_result, anchor_prediction = summarize_branch(
        "anchor", output / "student_anchor", branch_targets["anchor"],
        ids, labels, users, None,
    )
    branches: dict[str, Any] = {"anchor": anchor_result}
    for name in config["targets"]:
        if name == "anchor":
            continue
        result, _ = summarize_branch(
            name, output / f"student_{name}", branch_targets[name],
            ids, labels, users, anchor_prediction,
        )
        branches[name] = result

    reference_dir = resolve(config["reference_structured_run"])
    reference_target = resolve(config["reference_structured_targets"])
    if reference_dir.joinpath("summary.json").exists():
        reference, reference_prediction = summarize_branch(
            "p87_structured_4epoch", reference_dir, reference_target,
            ids, labels, users, anchor_prediction,
            target_key="structured_distillation_probability",
        )
        reference["student_vs_anchor_control"] = rescue_harm(
            labels, anchor_prediction, reference_prediction
        )
        branches["p87_structured_4epoch_reference"] = reference

    base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    report = {
        "stage": "P99_H1_fixed_budget_transfer_probe",
        "status": "complete",
        "protocol": (
            "All new branches start from the identical P87-S H1 C0 checkpoint, train "
            "four epochs with identical optimizer scope, and differ only by label-free "
            "teacher probability. H1 labels are read only after adaptation for evaluation."
        ),
        "base_student": base_summary["subject_holdout_metrics"],
        "branches": branches,
        "selection_rule": (
            "Prefer final Student Top-1 and worst-user stability; use Teacher score and "
            "Teacher-to-Student gap as explanatory axes, not as an automatic ranking."
        ),
        "h2_h3_accessed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if bool(config.get("h2_h3_access")):
        raise ValueError("P99 H1 transfer probe cannot access H2/H3")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"build", "all"}:
        print(json.dumps(build_targets(config, output), ensure_ascii=False, indent=2), flush=True)
    if args.mode in {"run", "all"}:
        run_branches(config, output)
    if args.mode in {"summarize", "all"}:
        report = summarize(config, output)
        compact = {
            name: {
                "teacher_correct": value["teacher_metrics"]["correct"],
                "student_correct": value["student_metrics"]["correct"],
                "student_top5": value["student_metrics"]["top5"],
                "gap": value["teacher_to_student_correct_gap"],
                "vs_anchor": value.get("student_vs_anchor_control"),
                "bytes": value["checkpoint_bytes"],
            }
            for name, value in report["branches"].items()
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
