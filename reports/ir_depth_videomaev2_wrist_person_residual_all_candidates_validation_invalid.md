# INVALID: all-candidates wrist-person validation result

This report is retained only as audit evidence. It evaluated every grouped-CV
candidate on user6/user7 rather than evaluating only the preselected candidate.
Use the selected-only result instead.

- Selected by train-user grouped CV: `person_fixed10`
- Decision: `reject_cached_fusion`
- Validation users entered training/CV selection: `False`

| Candidate | Grouped Acc | Val Acc | Macro-F1 | Worst-user | Person-only | Gate | Rescue/Harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| person_fixed10 | 0.935401 | 0.722078 | 0.636735 | 0.701493 | 0.667532 | 0.100000 | 3/3 |
| person_no_margin | 0.928165 | 0.722078 | 0.627381 | 0.681592 | 0.693506 | 1.000000 | 17/17 |
| person_margin | 0.926615 | 0.716883 | 0.631598 | 0.671642 | 0.688312 | 0.719881 | 15/17 |
| person_margin_no_aux | 0.929199 | 0.693506 | 0.605455 | 0.671642 | 0.049351 | 0.745756 | 11/22 |
