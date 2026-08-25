# VideoMAE wrist-person residual result

- Selected by train-user grouped CV: `person_fixed10`
- Decision: `reject_cached_fusion`
- Validation users entered training/CV selection: `False`

## Grouped-CV candidate selection

| Candidate | Grouped Acc | Grouped Macro-F1 |
|---|---:|---:|
| person_fixed10 | 0.935401 | 0.933109 |
| person_no_margin | 0.928165 | 0.924358 |
| person_margin | 0.926615 | 0.926387 |
| person_margin_no_aux | 0.929199 | 0.926308 |

## Selected-only final development evaluation

| Candidate | Val Acc | Macro-F1 | Worst-user | Person-only | Gate | Rescue/Harm |
|---|---:|---:|---:|---:|---:|---:|
| person_fixed10 | 0.722078 | 0.636735 | 0.701493 | 0.667532 | 0.100000 | 3/3 |
