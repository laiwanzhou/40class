from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from p46_protocol import HARD_CLASS_IDS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "p46_single_split.csv"
DEFAULT_REFERENCE_OOF = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46_v3_train_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-final P46-v3 logits for every one of the 14 P46 training "
            "users using three subject-disjoint outer folds."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-oof", type=Path, default=DEFAULT_REFERENCE_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage-a-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--folds", nargs="*", type=int, default=(0, 1, 2), choices=(0, 1, 2)
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row["detail_selected"] == "1" and row["p46_split"] == "train"
    ]
    if len(rows) != 1094 or len({row["user_id"] for row in rows}) != 14:
        raise RuntimeError("Frozen P46 training universe changed")
    return rows


def reference_user_folds(path: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    user_by_sample = {row["sample_id"]: row["user_id"] for row in rows}
    with np.load(path, allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        folds = np.asarray(data["folds"], dtype=np.int64)
    by_user: dict[str, set[int]] = {}
    for sample_id, fold in zip(sample_ids, folds):
        user = user_by_sample.get(sample_id)
        if user is not None:
            by_user.setdefault(user, set()).add(int(fold))
    expected = {row["user_id"] for row in rows}
    if set(by_user) != expected:
        raise RuntimeError("Reference OOF does not cover the P46 training users")
    if any(len(values) != 1 for values in by_user.values()):
        raise RuntimeError("Reference OOF splits a user across folds")
    result = {user: next(iter(values)) for user, values in by_user.items()}
    if set(result.values()) != {0, 1, 2}:
        raise RuntimeError("Reference OOF must contain three user folds")
    return result


def expected_fold_output(
    path: Path,
    expected_users: list[str],
    expected_epoch: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            users = sorted(set(np.asarray(data["users"]).astype(str)))
            epoch = int(np.asarray(data["epoch"]).item())
            logits = np.asarray(data["logits"])
            source_ids = np.asarray(data["source_ids"])
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return (
        users == expected_users
        and epoch == expected_epoch
        and logits.shape == (len(source_ids), len(HARD_CLASS_IDS))
        and np.isfinite(logits).all()
    )


def run_fold(
    fold: int,
    train_users: list[str],
    val_users: list[str],
    args: argparse.Namespace,
) -> Path:
    fold_dir = args.output_dir.resolve() / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / "fixed_final_oof_logits.npz"
    if expected_fold_output(result_path, val_users, args.epochs):
        print(
            json.dumps(
                {"stage": "fold_skip_complete", "fold": fold, "path": str(result_path)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return result_path
    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "train_p46_unified_repair_v3.py"),
        "--output-dir",
        str(fold_dir),
        "--train-users",
        *train_users,
        "--val-users",
        *val_users,
        "--stage-a-epochs",
        str(args.stage_a_epochs),
        "--epochs",
        str(args.epochs),
        "--minimum-epochs",
        str(args.epochs),
        "--patience",
        "0",
        "--offset-eval-every",
        "0",
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed + fold * 1000),
        "--log-every",
        "20",
    ]
    print(
        json.dumps(
            {
                "stage": "fold_start",
                "fold": fold,
                "train_users": train_users,
                "val_users": val_users,
                "command": command,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    log_path = fold_dir / "orchestrator_train.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[fold {fold}] {line}", end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"P46-v3 OOF fold {fold} failed with exit code {return_code}")
    if not expected_fold_output(result_path, val_users, args.epochs):
        raise RuntimeError(f"P46-v3 OOF fold {fold} did not produce valid fixed-final logits")
    return result_path


def merge(
    paths: dict[int, Path],
    rows: list[dict[str, str]],
    user_folds: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row_by_source = {row["source_id"]: row for row in rows}
    merged: dict[str, tuple[int, np.ndarray, str, int]] = {}
    fold_metrics: dict[str, Any] = {}
    for fold, path in sorted(paths.items()):
        with np.load(path, allow_pickle=False) as data:
            source_ids = np.asarray(data["source_ids"]).astype(str)
            users = np.asarray(data["users"]).astype(str)
            detail_labels = np.asarray(data["labels"], dtype=np.int64)
            logits = np.asarray(data["logits"], dtype=np.float32)
        predictions = logits.argmax(axis=1)
        fold_metrics[str(fold)] = {
            "samples": int(len(labels := detail_labels)),
            "correct": int((predictions == labels).sum()),
            "accuracy": float((predictions == labels).mean()),
            "users": sorted(set(users)),
        }
        for source_id, user, detail_label, logit in zip(
            source_ids, users, detail_labels, logits
        ):
            if source_id in merged:
                raise RuntimeError(f"duplicate P46 OOF source: {source_id}")
            row = row_by_source.get(source_id)
            if row is None:
                raise RuntimeError(f"P46 OOF source is outside frozen train universe: {source_id}")
            if row["user_id"] != user or user_folds[user] != fold:
                raise RuntimeError(f"P46 OOF provenance mismatch: {source_id}")
            if int(row["detail_index"]) != int(detail_label):
                raise RuntimeError(f"P46 OOF label mismatch: {source_id}")
            merged[source_id] = (int(row["class_id"]), logit, user, fold)
    if set(merged) != set(row_by_source):
        missing = sorted(set(row_by_source) - set(merged))
        raise RuntimeError(f"P46 OOF merge incomplete: missing {missing[:10]}")
    ordered_rows = sorted(rows, key=lambda row: row["sample_id"])
    sample_ids = np.asarray([row["sample_id"] for row in ordered_rows])
    source_ids = np.asarray([row["source_id"] for row in ordered_rows])
    labels = np.asarray([merged[row["source_id"]][0] for row in ordered_rows], dtype=np.int64)
    logits = np.stack([merged[row["source_id"]][1] for row in ordered_rows]).astype(np.float32)
    users = np.asarray([merged[row["source_id"]][2] for row in ordered_rows])
    folds = np.asarray([merged[row["source_id"]][3] for row in ordered_rows], dtype=np.int64)
    prediction = logits.argmax(axis=1)
    detail_labels = np.asarray(
        [HARD_CLASS_IDS.index(int(value)) for value in labels], dtype=np.int64
    )
    output_path = args.output_dir.resolve() / "complete_oof.npz"
    temporary = output_path.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=sample_ids,
            source_ids=source_ids,
            labels=labels,
            users=users,
            folds=folds,
            logits=logits,
            predictions=np.asarray(HARD_CLASS_IDS, dtype=np.int64)[prediction],
        )
    temporary.replace(output_path)
    return {
        "protocol": "P46-v3 fixed-final three-fold held-user OOF over P46-train14 only",
        "target_validation_users_excluded": ["user1", "user2", "user8", "user9"],
        "fixed_training_schedule": {
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.epochs,
            "early_stopping": False,
            "epoch_selection_from_held_fold": False,
        },
        "samples": int(len(labels)),
        "correct": int((prediction == detail_labels).sum()),
        "accuracy": float((prediction == detail_labels).mean()),
        "fold_metrics": fold_metrics,
        "user_folds": dict(sorted(user_folds.items())),
        "output": str(output_path),
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest.resolve())
    user_folds = reference_user_folds(args.reference_oof.resolve(), rows)
    selected_folds = sorted(set(args.folds))
    paths: dict[int, Path] = {}
    for fold in selected_folds:
        val_users = sorted(user for user, value in user_folds.items() if value == fold)
        train_users = sorted(user for user, value in user_folds.items() if value != fold)
        paths[fold] = run_fold(fold, train_users, val_users, args)
    if selected_folds == [0, 1, 2]:
        summary = merge(paths, rows, user_folds, args)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"stage": "complete", **summary}, ensure_ascii=False, indent=2), flush=True)
    else:
        print(
            json.dumps(
                {"stage": "partial_complete", "folds": selected_folds}, ensure_ascii=False
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
