# X3D-S Single13 Fixed-Context Result

> Development-only evidence on user6/user7. Canonical Phase 4/5 evidence remains unchanged.

Decision: `preferred_spatial_candidate`

| Metric | Fixed context | Delta vs moving-context Single13 |
|---|---:|---:|
| Accuracy | 0.537662 | +0.015584 |
| Macro-F1 | 0.427514 | +0.016069 |
| Worst-user Accuracy | 0.517413 | +0.014925 |

Selected epoch: `14` of `17`. 
Train Accuracy at selection was `0.956589`; the train-minus-validation gap was `0.418927`.

Peak CUDA memory: `197556736` bytes. Checkpoint: `14388671` bytes.

One trial-level fixed person-context box improves Accuracy, Macro-F1, and worst-user Accuracy over the matched moving-context Single13 route. The combined fixed-context Single13 route is retained for the approved four-channel IR+Depth experiment.
