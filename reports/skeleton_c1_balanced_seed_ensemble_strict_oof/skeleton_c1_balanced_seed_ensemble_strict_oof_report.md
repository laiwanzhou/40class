# E2-v1 Balanced E1 Strict Cross-User OOF Development Evidence

## Frozen Contract

- E1 architecture, C1 preprocessing, T=64, optimizer, scheduler, CE loss, batch size, nested OOF protocol, and all three seeds are unchanged.
- The only experimental variable is train-scope sampling weight `1 / sqrt(n_c)` with replacement and exactly one train-scope-size draw per epoch.
- Class counts are fitted independently on inner-fit and formal outer-train. Validation labels are never used to fit sampling or preprocessing.
- Fixed equal mean of three member softmax probabilities; no member selection, weighting, or outer-fold gate.
- E2 config SHA256: `c2aa6c0da5e0a652cf136320af6bfdf2b31d3e6e0318250977bc5ed0e5487c42`.

## Member Results

| model            |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:-----------------|----------:|-----------:|-------------------:|--------------:|
| E2-seed-20260812 |      2341 |   0.362238 |           0.261731 |      0.340775 |
| E2-seed-20260912 |      2341 |   0.367364 |           0.278776 |      0.353536 |
| E2-seed-20261012 |      2341 |   0.397693 |           0.324793 |      0.392672 |

## Fold Results

|     fold |    samples |   e1_accuracy |   e2_accuracy |   e1_macro_f1_40class |   e2_macro_f1_40class |   e1_only |   e2_only |   e2_minus_e1_accuracy |   e2_minus_e1_macro_f1 |
|---------:|-----------:|--------------:|--------------:|----------------------:|----------------------:|----------:|----------:|-----------------------:|-----------------------:|
| 0.000000 | 812.000000 |      0.390394 |      0.370690 |              0.272696 |              0.279475 | 35.000000 | 19.000000 |              -0.019704 |               0.006780 |
| 1.000000 | 667.000000 |      0.451274 |      0.434783 |              0.349947 |              0.346373 | 47.000000 | 36.000000 |              -0.016492 |              -0.003574 |
| 2.000000 | 862.000000 |      0.433875 |      0.422274 |              0.328724 |              0.340423 | 51.000000 | 41.000000 |              -0.011601 |               0.011699 |

## Combined OOF

| model                         |   samples |   accuracy |   macro_f1_40class |   weighted_f1 |
|:------------------------------|----------:|-----------:|-------------------:|--------------:|
| E1-C1-3seed-ensemble          |      2341 |   0.423751 |           0.315606 |      0.397567 |
| E2-balanced-E1-3seed-ensemble |      2341 |   0.407945 |           0.323639 |      0.398976 |

- E2 - E1 Accuracy: **-0.015805**; paired bootstrap 95% CI `[-0.028193, -0.003417]`.
- E2 - E1 Macro-F1: **+0.008034**; paired bootstrap 95% CI `[-0.005637, +0.022019]`.
- Paired discordance: E1-only correct=133, E2-only correct=96; exact McNemar p=0.0171715.
- E2 improves Macro-F1 for 2/3 folds, Accuracy for 3/14 users, and F1 for 18/40 classes.

## Class-Support Diagnosis

| support_bucket   |   classes |   samples |   e1_macro_f1 |   e2_macro_f1 |   e2_minus_e1_macro_f1 |
|:-----------------|----------:|----------:|--------------:|--------------:|-----------------------:|
| support_le_15    |         2 |        21 |      0.000000 |      0.000000 |               0.000000 |
| support_16_31    |         8 |       215 |      0.256178 |      0.281211 |               0.025032 |
| support_32_63    |        19 |       891 |      0.314971 |      0.328243 |               0.013273 |
| support_ge_64    |        11 |      1214 |      0.417306 |      0.405388 |              -0.011918 |

This support analysis is diagnostic and was not used to set the sampling exponent or select a model.

## Largest Class Changes

Largest F1 gains:

|   class_id | action_name           |   support |    f1_e1 |    f1_e2 |   e2_minus_e1_f1 |
|-----------:|:----------------------|----------:|---------:|---------:|-----------------:|
|          3 | Take_off_clothes      |        30 | 0.057143 | 0.301887 |         0.244744 |
|         38 | Massage_oneself       |        50 | 0.067797 | 0.243243 |         0.175447 |
|          1 | Brush_teeth           |        36 | 0.204082 | 0.338028 |         0.133947 |
|         32 | Stand_up              |        60 | 0.672131 | 0.750000 |         0.077869 |
|         39 | Take_body_temperature |        40 | 0.130435 | 0.203390 |         0.072955 |

Largest F1 declines:

|   class_id | action_name   |   support |    f1_e1 |    f1_e2 |   e2_minus_e1_f1 |
|-----------:|:--------------|----------:|---------:|---------:|-----------------:|
|         22 | Turn_pages    |        50 | 0.266667 | 0.173913 |        -0.092754 |
|          4 | Wipe_hands    |        55 | 0.234043 | 0.165138 |        -0.068905 |
|         14 | Wipe_bowls    |        29 | 0.117647 | 0.048780 |        -0.068867 |
|         28 | Jog_in_place  |        26 | 0.679245 | 0.625000 |        -0.054245 |
|         13 | Mop_the_floor |        50 | 0.408163 | 0.357895 |        -0.050269 |

## Pre-Registered Decision

**E2-v1 does not pass the pre-registered replacement criterion; E1 remains the preferred Skeleton expert.** Replacement requires non-decreasing combined Accuracy, increasing Macro-F1, a Macro-F1 paired-bootstrap interval above zero, and Macro-F1 improvement on at least two folds.

## Provenance

- Evidence type: strict cross-user OOF development evidence, not a one-time untouched final test.
- Frozen split SHA256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`.
- Sample space: 2,341 exactly paired trials, 14 users, 40 classes.
- All nine member summaries pass the preprocessing and sampler scope-provenance checks.
- Raw histories, sampling provenance, and predictions remain under `outputs/skeleton_c1_balanced_seed_ensemble_strict_oof`; no checkpoint or model weight was saved.
