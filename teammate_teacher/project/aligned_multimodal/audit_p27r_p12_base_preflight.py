from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_P12 = PROJECT_DIR / "runs" / "p12_complete_oof" / "complete_oof.npz"
DEFAULT_FOLDS = PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_r_fold0_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight strict outer-fold purity of P12 base logits for P27-R"
    )
    parser.add_argument("--outer-fold", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--p12", type=Path, default=DEFAULT_P12)
    parser.add_argument("--fold-summary", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def accuracy(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(np.mean(logits.argmax(axis=1) == labels))


def main() -> None:
    args = parse_args()
    outer_fold = int(args.outer_fold)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fold_summary = json.loads(args.fold_summary.resolve().read_text(encoding="utf-8"))
    fold_info = {int(row["fold"]): row for row in fold_summary["folds"]}
    outer_held_users = set(fold_info[outer_fold]["val_users"])
    outer_train_users = set(fold_info[outer_fold]["train_users"])
    with np.load(args.p12.resolve(), allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        final_logits = data["final_logits"].astype(np.float32)
    subjects = np.asarray([sample_id.split("__")[2] for sample_id in sample_ids])
    expected_held = np.isin(subjects, sorted(outer_held_users))
    if not np.array_equal(expected_held, folds == outer_fold):
        raise RuntimeError("P12 folds disagree with subject fold summary")

    train_source_rows = []
    contaminated_train_rows = 0
    for source_fold in sorted(set(folds[folds != outer_fold].tolist())):
        source_mask = (folds != outer_fold) & (folds == source_fold)
        source_train_users = set(fold_info[int(source_fold)]["train_users"])
        held_overlap = sorted(
            source_train_users & outer_held_users,
            key=lambda value: int(value[4:]),
        )
        contaminated = bool(held_overlap)
        contaminated_train_rows += int(source_mask.sum()) if contaminated else 0
        train_source_rows.append(
            {
                "p12_oof_source_fold": int(source_fold),
                "outer_train_rows_using_source": int(source_mask.sum()),
                "source_model_train_users": sorted(
                    source_train_users, key=lambda value: int(value[4:])
                ),
                "outer_fold_held_user_overlap": held_overlap,
                "strict_outer_pure": not contaminated,
            }
        )

    # The P12 fold-0 meta protocol was fitted on the other two OOF folds. Those
    # OOF predictions came from experts whose training sets include fold-0
    # held users, so it is cross-fitted but not a nested outer-fold protocol.
    p12_summary_path = args.p12.resolve().with_name("summary.json")
    p12_summary = json.loads(p12_summary_path.read_text(encoding="utf-8"))
    thermal_protocol = next(
        row
        for row in p12_summary["thermal_protocols"]
        if int(row["held_fold"]) == outer_fold
    )
    router_protocol = next(
        row
        for row in p12_summary["router_protocols"]
        if int(row["held_fold"]) == outer_fold
    )

    fold0_assets = [
        PROJECT_DIR
        / "runs"
        / "p11_fp16_oof"
        / f"fold_{outer_fold}"
        / "skeleton_best_accuracy_fp16.pt",
        PROJECT_DIR
        / "runs"
        / "p11_fp16_oof"
        / f"fold_{outer_fold}"
        / "depth_best_accuracy_fp16.pt",
        REPO_DIR
        / "thermal_baseline"
        / "runs"
        / "p11_thermal_imagenet_fp16"
        / f"fold_{outer_fold}"
        / "best_accuracy_fp16.pt",
    ]
    full18_assets = [
        PROJECT_DIR / "runs" / "p11_final_package" / "skeleton_final_fp16.pt",
        PROJECT_DIR / "runs" / "p11_final_package" / "depth_final_fp16.pt",
        PROJECT_DIR / "runs" / "p11_final_package" / "thermal_final_fp16.pt",
        PROJECT_DIR / "runs" / "p11_final_package" / "imu_random_forest.joblib",
    ]
    persisted_router_candidates = list(
        (PROJECT_DIR / "runs" / "p11_final_package").glob("*router*.joblib")
    )
    persisted_fold_rf_candidates = list(
        (PROJECT_DIR / "runs").glob(f"**/fold_{outer_fold}/*random_forest*.joblib")
    )
    held = folds == outer_fold
    train = ~held
    result = {
        "protocol": "p27-r-p12-base-preflight-v1",
        "outer_fold": outer_fold,
        "status": "blocked",
        "blocking_condition": (
            "No one base-logit source is simultaneously (a) the frozen P12 "
            "decision function, (b) available for both outer-train and held "
            "rows, and (c) nested outer-fold-pure."
        ),
        "outer_split": {
            "train_rows": int(train.sum()),
            "held_rows": int(held.sum()),
            "train_users": sorted(outer_train_users, key=lambda value: int(value[4:])),
            "held_users": sorted(outer_held_users, key=lambda value: int(value[4:])),
        },
        "p12_read_only_reference": {
            "artifact": artifact(args.p12.resolve()),
            "held_rows": int(held.sum()),
            "held_accuracy": accuracy(labels[held], final_logits[held]),
            "outer_train_rows": int(train.sum()),
            "outer_train_oof_accuracy": accuracy(labels[train], final_logits[train]),
            "allowed_use": "held read-only comparator only",
        },
        "nested_purity_audit": {
            "outer_train_rows_with_held-user-contaminated_source_experts": contaminated_train_rows,
            "outer_train_rows_total": int(train.sum()),
            "sources": train_source_rows,
            "fold0_thermal_protocol": thermal_protocol,
            "fold0_router_protocol": router_protocol,
            "meta_protocol_nested_outer_pure": False,
        },
        "available_outer_fold_experts": [
            artifact(path) for path in fold0_assets if path.is_file()
        ],
        "missing_callable_fold_components": {
            "persisted_fold_specific_rf_models": [
                str(path) for path in persisted_fold_rf_candidates
            ],
            "persisted_router_models": [
                str(path) for path in persisted_router_candidates
            ],
            "note": (
                "The RF can be deterministically refit on outer-train, and S/D/T "
                "experts can be replayed. That yields in-sample train logits, not "
                "nested OOF train logits, and therefore is not accepted for this "
                "residual-learning pilot."
            ),
        },
        "rejected_shortcuts": [
            {
                "method": "use P12 OOF logits for outer-train rows",
                "reason": (
                    f"{contaminated_train_rows}/{int(train.sum())} rows come from "
                    "source experts trained on outer-fold held users"
                ),
            },
            {
                "method": "replay fold-0 experts on outer-train rows",
                "reason": (
                    "outer-pure but in-sample; base error and confidence distribution "
                    "is not comparable to held OOF, invalidating residual learning"
                ),
            },
            {
                "method": "use full18 P12 package",
                "reason": "contains all subjects and directly violates the user protocol",
                "artifacts": [
                    artifact(path) for path in full18_assets if path.is_file()
                ],
            },
        ],
        "safe_unexecuted_solution": {
            "method": (
                "Create an inner subject-disjoint OOF P12 replica on the 12 outer-train "
                "subjects, including inner-only RF/Thermal calibration and routing; "
                "then train the residual on those inner OOF logits and evaluate once "
                "on the frozen fold-0 P12 held logits."
            ),
            "why_not_automatic": (
                "This requires training new P12 expert replicas and changes the meaning "
                "of 'freeze the original P12 experts'. It is a material protocol "
                "expansion, not a safe implementation detail."
            ),
        },
        "initial_equivalence_gate": {
            "evaluated": False,
            "reason": "No admissible train+held base-logit source; residual model not instantiated.",
        },
        "training_started": False,
    }
    path = output / "p12_base_preflight.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
