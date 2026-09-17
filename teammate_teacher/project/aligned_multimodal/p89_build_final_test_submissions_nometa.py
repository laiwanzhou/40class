from __future__ import annotations

from pathlib import Path

import p89_build_final_test_submissions as pipeline


# H1 and H2 ablation predictions are exactly unchanged after removing these two
# distilled meta-experts.  Keep only the 33 experts with direct Test deployment.
pipeline.P46_EXPERT_RUNS = tuple(
    name
    for name in pipeline.P46_EXPERT_RUNS
    if name
    not in {
        "p46_70_subject_calibrated_v2",
        "p46_validation70_final_v1",
    }
)
pipeline.OUTPUT = (
    Path(__file__).resolve().parent / "runs/p89_final_test_predictions_nometa_v2"
)


if __name__ == "__main__":
    pipeline.main()
