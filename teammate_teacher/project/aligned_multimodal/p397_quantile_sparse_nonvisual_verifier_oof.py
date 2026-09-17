"""P396 sparse rules with source route-rate quantile transfer to held cohorts."""
from __future__ import annotations

import json
from pathlib import Path

import p394_sparse_nonvisual_class_verifier_oof as sparse
import p395_loso_sparse_nonvisual_class_verifier_oof as loso


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p397_quantile_sparse_nonvisual_verifier_oof_v1"


def main():
    print(
        "P397 transfers each zero-harm sparse source rule by selected candidate-pool "
        "quantile instead of absolute logistic probability.",
        flush=True,
    )
    sparse.MINIMUM_RESCUE = 3
    sparse.MINIMUM_COHORT_GAIN = 0
    loso.THRESHOLD_TRANSFER = "quantile"
    loso.OUT = OUT
    loso.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P397_quantile_sparse_nonvisual_verifier_OOF"
    report["protocol"]["minimum_total_source_rescue"] = 3
    report["protocol"]["minimum_source_cohort_gain"] = 0
    report["protocol"]["source_harm"] = 0
    report["protocol"]["threshold_transfer"] = "source selected fraction -> held score quantile"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: source-safe sparse class rules with unlabeled held quantile calibration.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
