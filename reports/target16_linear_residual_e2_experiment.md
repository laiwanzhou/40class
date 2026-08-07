# Target16 linear residual E2 hierarchical fusion

## Integrity

- B2 checkpoint: Epoch 25.
- E2 training/selection: 880 Target16 train samples and 222 Target16 validation samples.
- Fusion evaluation: the same 590 validation samples, with no test access.
- Gate audit: 56 correct target, 123 target-to-target errors, 43 target-to-external errors, 43 external-to-target errors.
- Samples outside the B2 Target16 Top-1 gate are unchanged for every alpha.
- All 24 non-target class F1 scores are numerically invariant across the alpha sweep.

## E2 closed-set training

- Best checkpoint: Epoch 19, Target16 validation Accuracy 0.328829, Macro-F1 0.321381.
- B2 Target16 closed-set baseline on all 222 target samples: Accuracy 0.324324, Macro-F1 0.315527.
- E2 change over that baseline: Accuracy +0.004505, Macro-F1 +0.005854.
- At the selected epoch, train Accuracy was 0.972727; the train-validation gap was 0.643898.
- By Epoch 34, validation Accuracy/Macro-F1 were 0.324324/0.318805, with validation loss 3.821269. Only the 3,088-parameter residual head was trained; the large absolute train-validation gap is inherited from the frozen B2 representation rather than created by backbone fine-tuning.

## Fixed alpha sweep

| alpha | accuracy | macro_f1 | weighted_f1 | target16_macro_f1 | gated_target_accuracy | gated_target_macro_f1 | rescued | harmed | net_rescue | correct_count | gated_samples | gated_true_target_samples | gated_external_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 0.462712 | 0.392827 | 0.463045 | 0.257814 | 0.312849 | 0.321230 | 0 | 0 | 0 | 273 | 222 | 179 | 43 |
| 0.250000 | 0.462712 | 0.392795 | 0.463217 | 0.257733 | 0.312849 | 0.321230 | 0 | 0 | 0 | 273 | 222 | 179 | 43 |
| 0.500000 | 0.462712 | 0.392390 | 0.463101 | 0.256723 | 0.312849 | 0.319343 | 0 | 0 | 0 | 273 | 222 | 179 | 43 |
| 0.750000 | 0.462712 | 0.392390 | 0.463101 | 0.256723 | 0.312849 | 0.319343 | 0 | 0 | 0 | 273 | 222 | 179 | 43 |
| 1.000000 | 0.462712 | 0.391956 | 0.463890 | 0.255637 | 0.312849 | 0.316964 | 1 | 1 | 0 | 273 | 222 | 179 | 43 |

## Diagnostic result

The strongest validation Macro-F1 in the fixed sweep is alpha=0.00: Accuracy 0.462712, Macro-F1 0.392827, rescued 0, harmed 0, net 0.
Relative to B2, this is Accuracy +0.000000, Macro-F1 +0.000000, and Target16 Macro-F1 +0.000000.
This is a validation diagnostic, not an independently tested deployment threshold.

## Largest Target16 gains at alpha=1.0

- Eat_food: 0.279 -> 0.326 (+0.047).
- Turn_pages: 0.303 -> 0.323 (+0.020).
- Use_a_mobile_phone: 0.200 -> 0.205 (+0.005).
- Drink_water: 0.593 -> 0.593 (+0.000).
- Make_a_phone_call: 0.353 -> 0.353 (+0.000).
- Write: 0.200 -> 0.200 (+0.000).

## Largest Target16 harms at alpha=1.0

- Watch_TV: 0.333 -> 0.286 (-0.048).
- Peel_fruits: 0.222 -> 0.186 (-0.036).
- Take_and_use_tableware: 0.222 -> 0.200 (-0.022).
- Drink_water: 0.593 -> 0.593 (+0.000).
- Wipe_bowls: 0.222 -> 0.222 (+0.000).
- Stir_drinks: 0.200 -> 0.200 (+0.000).

## Validation-user behavior

- user17: accuracy 0.482 -> 0.482, correct-count delta +0.
- user23: accuracy 0.389 -> 0.389, correct-count delta +0.
- user24: accuracy 0.509 -> 0.509, correct-count delta +0.
- user4: accuracy 0.455 -> 0.455, correct-count delta +0.

## Assessment

The linear residual experiment does not improve the hard hierarchy on this validation fold. The strongest alpha is 0.0, meaning the unmodified B2 output remains preferable. Fully freezing B2 prevents additional representation overfitting, but its fixed 192-dimensional embedding is not linearly sufficient to repair the 123 target-group errors.

## Comparison with partially fine-tuned E2

- Previous E2 best: alpha=1.00, Accuracy 0.481356, Macro-F1 0.401706, net rescue 11.
- Linear residual best: alpha=0.00, Accuracy 0.462712, Macro-F1 0.392827, net rescue 0.
- The capacity reduction removes the previous net gain; a useful next capacity point must lie between a 3,088-parameter head and the 1,124,977-parameter partial fine-tune.
