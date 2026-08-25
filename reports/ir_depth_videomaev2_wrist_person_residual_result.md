# VideoMAE wrist-person residual result

- Selected by train-user grouped CV: `person_fixed10`
- Decision: `reject_cached_fusion`
- Validation users entered training/CV selection: `False`

| Candidate | Grouped Acc | Val Acc | Macro-F1 | Worst-user | Person-only | Gate | Rescue/Harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| person_fixed10 | 0.939018 | 0.722078 | 0.635584 | 0.701493 | 0.706494 | 0.100000 | 4/4 |
| person_no_margin | 0.920413 | 0.719481 | 0.631011 | 0.686567 | 0.706494 | 1.000000 | 15/16 |
| person_margin | 0.929199 | 0.729870 | 0.637191 | 0.686567 | 0.703896 | 0.760602 | 16/13 |
| person_margin_no_aux | 0.933333 | 0.701299 | 0.625225 | 0.651741 | 0.028571 | 0.752017 | 10/18 |
