# X3D-S Single13-Global Result

> Development-only evidence on user6/user7. Canonical Phase 4/5 evidence remains unchanged.

Decision: `non_winning_ablation`

| Metric | Single13 | Delta vs adaptive Partial2 |
|---|---:|---:|
| Accuracy | 0.522078 | -0.010390 |
| Macro-F1 | 0.411445 | -0.010129 |
| Worst-user Accuracy | 0.502488 | -0.029851 |

Selected epoch: `7` of `15`. 
Train Accuracy at selection was `0.833075`; the train-minus-validation gap was `0.310997`.

Peak CUDA memory: `197556736` bytes. Checkpoint: `14388671` bytes.

Replacing adaptive Kx13 local windows with one globally stratified 13-frame clip reduces compute and the selected-epoch train/validation gap, but does not improve the matched user6/user7 development metrics.
