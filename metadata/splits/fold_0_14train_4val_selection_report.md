# fold_0 14-train / 4-validation user selection

## Scope

The manifest contains 18 users: user1, user2, user3, user4, user5, user6, user7, user8, user9, user16, user17, user18, user19, user20, user21, user22, user23, user24.
All C(18,4) = 3060 validation-user combinations were evaluated; 1033 passed the hard constraints.
The selection uses only metadata/manifest.csv structure. It does not use model predictions, Accuracy, loss, checkpoints, confusion matrices, or any test-set information.

## Hard constraints

- Exactly 14 train users and 4 validation users, with no overlap and complete 18-user coverage.
- Union train and validation splits must each cover all 40 classes.
- Every modality must have non-empty train and validation samples.
- Every modality train split must cover all 40 classes.
- Missing validation classes in an individual modality are heavily penalized; full 40-class modality validation coverage is preferred.
- Modality presence is defined exactly as series.fillna("").astype(str).str.strip().ne("").

## Score

Class-distribution error is mean absolute difference: MAD(p,q) = (1/40) * sum_c |p_c - q_c|, comparing normalized candidate-validation counts with normalized full-manifest counts.
User-count imbalance is the coefficient of variation of the four users' union sample counts. Missing-rate error is the mean absolute difference between candidate-validation and full-manifest modality missing rates.
The minimum-class penalty is 1/(1+union minimum) plus the mean of 1/(1+modality minimum), so larger minimum class counts are preferred.

total_score = coverage_penalty + 8*union_ratio_error + 8*mean_modality_ratio_error + 4*union_class_distribution_error + 4*mean_modality_class_distribution_error + 1*user_count_imbalance + 4*missing_rate_error + 0.5*min_class_penalty.
coverage_penalty = 100 times the number of missing validation classes summed across the six modalities. Lower score is better; exact ties use the lexicographically sorted val_users tuple.

| Weight | Value |
| --- | ---: |
| coverage_penalty_per_missing_val_modality_class | 100 |
| union_ratio_error | 8 |
| mean_modality_ratio_error | 8 |
| union_class_distribution_error | 4 |
| mean_modality_class_distribution_error | 4 |
| user_count_imbalance | 1 |
| modality_missing_rate_error | 4 |
| min_class_penalty | 0.5 |

## Selected split

- Validation users: user4, user17, user23, user24
- Train users: user1, user2, user3, user5, user6, user7, user8, user9, user16, user18, user19, user20, user21, user22
- Union train/validation: 2427/609 (validation ratio 0.200593)
- Selection score: 0.311704142086

| Modality | Train | Validation | Validation ratio | Train classes | Validation classes | Minimum validation class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Depth_Color | 2341 | 590 | 0.201296 | 40 | 40 | 5 |
| IMU | 2319 | 584 | 0.201171 | 40 | 40 | 5 |
| IR | 2342 | 591 | 0.201500 | 40 | 40 | 5 |
| Radar | 2329 | 585 | 0.200755 | 40 | 40 | 5 |
| Skeleton | 2341 | 590 | 0.201296 | 40 | 40 | 5 |
| Thermal | 2299 | 592 | 0.204773 | 40 | 40 | 5 |

## Validation class-count summary

| Population | Minimum | Maximum | Mean | Class coverage |
| --- | ---: | ---: | ---: | ---: |
| Union | 6 | 71 | 15.225 | 40 |
| Depth_Color | 5 | 71 | 14.750 | 40 |
| IMU | 5 | 68 | 14.600 | 40 |
| IR | 5 | 71 | 14.775 | 40 |
| Radar | 5 | 71 | 14.625 | 40 |
| Skeleton | 5 | 71 | 14.750 | 40 |
| Thermal | 5 | 63 | 14.800 | 40 |

## Top 10 candidates

| Rank | Validation users | Score | Union ratio | Mean modality ratio error | Class distribution error | Missing-rate error | Union minimum class |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | user4, user17, user23, user24 | 0.311704142086 | 0.200593 | 0.001799 | 0.011345 | 0.005753 | 6 |
| 2 | user1, user17, user23, user24 | 0.367155036877 | 0.203887 | 0.006137 | 0.010604 | 0.010601 | 6 |
| 3 | user3, user4, user23, user24 | 0.389905025442 | 0.199605 | 0.000746 | 0.013369 | 0.003433 | 3 |
| 4 | user4, user20, user23, user24 | 0.418466371690 | 0.198287 | 0.001226 | 0.012830 | 0.009241 | 3 |
| 5 | user1, user4, user17, user23 | 0.428501355849 | 0.198617 | 0.002872 | 0.010289 | 0.009393 | 3 |
| 6 | user1, user4, user23, user24 | 0.432932633335 | 0.196311 | 0.003776 | 0.012052 | 0.006421 | 3 |
| 7 | user4, user8, user23, user24 | 0.435723550908 | 0.201581 | 0.002030 | 0.015297 | 0.004998 | 3 |
| 8 | user1, user3, user4, user23 | 0.435753602138 | 0.197628 | 0.003320 | 0.011985 | 0.008445 | 3 |
| 9 | user17, user20, user23, user24 | 0.447142828770 | 0.205863 | 0.009908 | 0.010348 | 0.018875 | 6 |
| 10 | user1, user3, user23, user24 | 0.448228883222 | 0.202899 | 0.004651 | 0.012676 | 0.008297 | 3 |

## Selection rationale

Candidate 1 has the lowest deterministic structural score (0.311704142086), ahead of candidate 2 (0.367155036877) by 0.055450894790. The ranking jointly balances sample ratios, class distributions, per-user sample imbalance, modality missing rates, class coverage, and minimum class support; it is not a model-performance ranking.

## Provenance and archive

The old 12-train/6-validation fold was archived byte-for-byte at metadata/splits/fold_0_12train_6val_20260715.json (SHA-256 7FEE7733B9AC55F61A0BD4BBF4D68FDF56CF3C7DF76107F670BE28DA28AA85C1).
No test directory was read. No model result or prediction artifact was read. The old formal Baseline reports and outputs remain unchanged.
