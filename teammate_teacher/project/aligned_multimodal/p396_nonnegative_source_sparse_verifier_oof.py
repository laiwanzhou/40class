"""P395 sparse verifier with nonnegative rather than strictly positive source folds."""
from __future__ import annotations

import json
from pathlib import Path

import p394_sparse_nonvisual_class_verifier_oof as sparse
import p395_loso_sparse_nonvisual_class_verifier_oof as loso


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p396_nonnegative_source_sparse_verifier_oof_v1"


def main():
    print(
        "P396 permits a sparse class rule when both source cohorts are nonnegative, total "
        "rescue is at least three, harm is zero, and no source user regresses.",
        flush=True,
    )
    sparse.MINIMUM_RESCUE = 3
    sparse.MINIMUM_COHORT_GAIN = 0
    loso.OUT = OUT
    loso.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P396_nonnegative_source_sparse_verifier_OOF"
    report["protocol"]["minimum_total_source_rescue"] = 3
    report["protocol"]["minimum_source_cohort_gain"] = 0
    report["protocol"]["source_harm"] = 0
    report["protocol"]["minimum_source_user_gain"] = 0
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: LOSO sparse verifier allowing one zero-gain source cohort but no negative source evidence.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
