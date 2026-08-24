# IR + Depth VideoMAE V2 P2-A diagnostics

- Full Accuracy: `0.714286`
- Full Macro-F1: `0.630941`
- Eight-stream prediction oracle: `0.841558`
- Gate entropy: `2.079359` / uniform `2.079442`
- Cached logits/embeddings finite: `True`
- Reproduced selected predictions exactly: `True`

Current fusion is static class-conditioned late-logit fusion. It is not sample-conditioned
and does not perform IR/Depth feature interaction before the classifier.

> These user6/user7 ablations are exploratory diagnostics. They must not be used
> to fit or select a gate/P2-B recipe without a new user-grouped evaluation boundary.

## Modality only

| Experiment | Accuracy | Delta | Macro-F1 | Worst-user | Zero recall |
|---|---:|---:|---:|---:|---:|
| ir | 0.716883 | +0.002597 | 0.633415 | 0.681592 | 4 |
| depth | 0.709091 | -0.005195 | 0.624854 | 0.676617 | 4 |

## Modality drop

| Experiment | Accuracy | Delta | Macro-F1 | Worst-user | Zero recall |
|---|---:|---:|---:|---:|---:|
| depth | 0.716883 | +0.002597 | 0.633415 | 0.681592 | 4 |
| ir | 0.709091 | -0.005195 | 0.624854 | 0.676617 | 4 |

## View type only

| Experiment | Accuracy | Delta | Macro-F1 | Worst-user | Zero recall |
|---|---:|---:|---:|---:|---:|
| left_hand_object | 0.729870 | +0.015584 | 0.656239 | 0.681592 | 4 |
| right_hand_object | 0.722078 | +0.007792 | 0.639080 | 0.696517 | 4 |
| person_context | 0.690909 | -0.023377 | 0.610607 | 0.646766 | 4 |
| global | 0.672727 | -0.041558 | 0.591337 | 0.641791 | 5 |

## View type drop

| Experiment | Accuracy | Delta | Macro-F1 | Worst-user | Zero recall |
|---|---:|---:|---:|---:|---:|
| global | 0.719481 | +0.005195 | 0.636856 | 0.676617 | 4 |
| person_context | 0.719481 | +0.005195 | 0.637819 | 0.676617 | 4 |
| right_hand_object | 0.706494 | -0.007792 | 0.626564 | 0.656716 | 4 |
| left_hand_object | 0.701299 | -0.012987 | 0.613872 | 0.656716 | 4 |

## Individual stream only

| Experiment | Accuracy | Delta | Macro-F1 | Worst-user | Zero recall |
|---|---:|---:|---:|---:|---:|
| ir:right_hand_object | 0.703896 | -0.010390 | 0.617295 | 0.686567 | 4 |
| ir:left_hand_object | 0.696104 | -0.018182 | 0.602514 | 0.666667 | 4 |
| ir:person_context | 0.685714 | -0.028571 | 0.590688 | 0.641791 | 4 |
| depth:left_hand_object | 0.680519 | -0.033766 | 0.620220 | 0.626866 | 3 |
| depth:right_hand_object | 0.677922 | -0.036364 | 0.595405 | 0.651741 | 4 |
| depth:person_context | 0.659740 | -0.054545 | 0.569319 | 0.631841 | 4 |
| ir:global | 0.649351 | -0.064935 | 0.551026 | 0.597015 | 5 |
| depth:global | 0.636364 | -0.077922 | 0.557409 | 0.630435 | 5 |

## Individual stream drop

| Experiment | Accuracy | Delta | Macro-F1 | Worst-user | Zero recall |
|---|---:|---:|---:|---:|---:|
| ir:global | 0.727273 | +0.012987 | 0.642272 | 0.686567 | 4 |
| ir:person_context | 0.719481 | +0.005195 | 0.641696 | 0.671642 | 4 |
| depth:person_context | 0.719481 | +0.005195 | 0.632888 | 0.671642 | 4 |
| depth:global | 0.716883 | +0.002597 | 0.631738 | 0.671642 | 4 |
| ir:right_hand_object | 0.714286 | +0.000000 | 0.633106 | 0.666667 | 4 |
| depth:left_hand_object | 0.711688 | -0.002597 | 0.627709 | 0.676617 | 4 |
| depth:right_hand_object | 0.711688 | -0.002597 | 0.630968 | 0.666667 | 4 |
| ir:left_hand_object | 0.706494 | -0.007792 | 0.625027 | 0.661692 | 4 |

## Dataset coverage deferred

Competition-provided classes with five or fewer train12 users are recorded as
`deferred_dataset_coverage`; this diagnostic does not attempt to repair them.

## P2-B boundary

P2-B was not started. The only recorded candidate is fixed 32 frames for long trials;
motion-peak sampling is explicitly disabled pending this report's review.

Deferred classes: `3 Take_off_clothes` (5 users), `16 Fold_clothes` (3 users), `24 Use_a_mobile_phone` (5 users), `25 Watch_TV` (1 users), `26 Play_games` (3 users), `33 Lie_down` (4 users)
