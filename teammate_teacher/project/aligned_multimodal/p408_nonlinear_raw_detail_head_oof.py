"""Regularized shallow LightGBM candidate scorer on the P399 raw-detail features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

import p399_candidate_conditioned_raw_detail_head_oof as p399


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p408_nonlinear_raw_detail_head_oof_v1"


def fit_model(train, k, _c_value):
    prototype = p399.prototypes(train)
    matrix, sample_rows, candidate_ids, _ = p399.pair_features(train, prototype, k)
    target = (candidate_ids == train["labels"][sample_rows]).astype(int)
    if int(target.sum()) < 20 or int((1 - target).sum()) < 20:
        return None
    model = LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=10.0,
        class_weight="balanced",
        random_state=20260903,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(matrix, target)
    return model, prototype


def main():
    print(
        "P408 keeps the complete P399 protocol but replaces the linear L1 scorer with a "
        "strongly regularized depth-3 LightGBM to model sparse teacher/sensor interactions.",
        flush=True,
    )
    p399.OUT = OUT
    p399.CS = (0.03,)
    p399.fit_model = fit_model
    p399.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P408_nonlinear_raw_detail_head_OOF"
    report["protocol"]["model"] = {
        "type": "LightGBM binary candidate scorer",
        "n_estimators": 120,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 30,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: shallow nonlinear candidate scorer on the frozen P399 feature/protocol stack.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
