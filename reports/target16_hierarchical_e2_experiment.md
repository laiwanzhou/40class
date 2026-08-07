# Target16 conditional E2 hierarchical fusion

## Integrity

- B2 checkpoint: Epoch 25.
- E2 training/selection: 880 Target16 train samples and 222 Target16 validation samples.
- Fusion evaluation: the same 590 validation samples, with no test access.
- Gate audit: 56 correct target, 123 target-to-target errors, 43 target-to-external errors, 43 external-to-target errors.
- Samples outside the B2 Target16 Top-1 gate are unchanged for every alpha.
- All 24 non-target class F1 scores are numerically invariant across the alpha sweep.

## E2 closed-set training

- Best checkpoint: Epoch 11, Target16 validation Accuracy 0.360360, Macro-F1 0.332899.
- B2 Target16 closed-set baseline on all 222 target samples: Accuracy 0.324324, Macro-F1 0.315527.
- E2 change over that baseline: Accuracy +0.036036, Macro-F1 +0.017371.
- At the selected epoch, train Accuracy was 0.962500; the train-validation gap was 0.602140.
- By Epoch 30, validation Accuracy/Macro-F1 had fallen to 0.315315/0.280678, while validation loss reached 5.333203. This is clear late-stage overfitting.

## Fixed alpha sweep

| alpha | accuracy | macro_f1 | weighted_f1 | target16_macro_f1 | gated_target_accuracy | gated_target_macro_f1 | rescued | harmed | net_rescue | correct_count | gated_samples | gated_true_target_samples | gated_external_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 0.462712 | 0.392827 | 0.463045 | 0.257814 | 0.312849 | 0.321230 | 0 | 0 | 0 | 273 | 222 | 179 | 43 |
| 0.250000 | 0.466102 | 0.398725 | 0.468312 | 0.272560 | 0.324022 | 0.347821 | 7 | 5 | 2 | 275 | 222 | 179 | 43 |
| 0.500000 | 0.471186 | 0.393401 | 0.470915 | 0.259249 | 0.340782 | 0.323107 | 15 | 10 | 5 | 278 | 222 | 179 | 43 |
| 0.750000 | 0.476271 | 0.395909 | 0.476809 | 0.265520 | 0.357542 | 0.328308 | 21 | 13 | 8 | 281 | 222 | 179 | 43 |
| 1.000000 | 0.481356 | 0.401706 | 0.482088 | 0.280012 | 0.374302 | 0.342805 | 25 | 14 | 11 | 284 | 222 | 179 | 43 |

## Diagnostic result

The strongest validation Macro-F1 in the fixed sweep is alpha=1.00: Accuracy 0.481356, Macro-F1 0.401706, rescued 25, harmed 14, net 11.
Relative to B2, this is Accuracy +0.018644, Macro-F1 +0.008879, and Target16 Macro-F1 +0.022198.
This is a validation diagnostic, not an independently tested deployment threshold.

## Largest Target16 gains at alpha=1.0

- Stir_drinks: 0.200 -> 0.462 (+0.262).
- Play_games: 0.190 -> 0.364 (+0.173).
- Read_documents: 0.200 -> 0.364 (+0.164).
- Eat_food: 0.279 -> 0.391 (+0.112).
- Peel_fruits: 0.222 -> 0.333 (+0.111).
- Wipe_bowls: 0.222 -> 0.296 (+0.074).

## Largest Target16 harms at alpha=1.0

- Watch_TV: 0.333 -> 0.000 (-0.333).
- Write: 0.200 -> 0.091 (-0.109).
- Turn_pages: 0.303 -> 0.200 (-0.103).
- Drink_water: 0.593 -> 0.500 (-0.093).
- Make_a_phone_call: 0.353 -> 0.300 (-0.053).
- Take_medicine: 0.160 -> 0.143 (-0.017).

## Validation-user behavior

- user17: accuracy 0.482 -> 0.530, correct-count delta +8.
- user23: accuracy 0.389 -> 0.382, correct-count delta -1.
- user24: accuracy 0.509 -> 0.541, correct-count delta +5.
- user4: accuracy 0.455 -> 0.448, correct-count delta -1.

## Assessment

The hard hierarchy succeeds on this validation fold: both overall Accuracy and Macro-F1 improve, and the non-target 24-class decision surface is preserved. The improvement is not uniform across target actions or users, and the E2 training curve still shows severe user-level overfitting. The result supports the conditional-expert mechanism, but alpha=1.0 remains a validation-selected diagnostic until confirmed on an independent fold or held-out calibration protocol.
