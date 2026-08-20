# X3D-S Ordinal Depth Motion Adapter Result

> Development-only evidence on user6/user7. No folds, extra seeds, heldout evaluation, or stability work are authorized.

Decision: `non_winning_ablation`

| Metric | Candidate | Delta vs absolute-Depth anchor | Delta vs IR |
|---|---:|---:|---:|
| Accuracy | 0.548052 | +0.000000 | +0.010390 |
| Macro-F1 | 0.462041 | +0.028964 | +0.034527 |
| Worst-user Accuracy | 0.542289 | -0.001190 | +0.024876 |

Selected epoch: `9` of `17`.

Human review is mandatory below Accuracy `0.528052`; artifacts remain preserved.

A separately approved stability review requires Accuracy >= `0.63`.

Depth is represented only by fixed-context aligned ordinal relative displacement, source-time-normalized velocity, and motion magnitude. The unchanged repeated-IR path remains exact at initialization and Depth enters through the zero-initialized nine-parameter residual adapter.
