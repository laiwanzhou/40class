# X3D Direct-Head versus A4-T Diagnosis

Date: 2026-08-20

This is a read-only diagnosis. It does not change either experiment, promote a candidate, access heldout4/test, or authorize another IR run.

## Headline

The `0.615000` A4-T Accuracy and `0.516883` Direct-Head Accuracy are not matched measurements. Their validation users are disjoint and each validation population was included in the other experiment's training population. The observed `0.098117` gap therefore cannot be interpreted as a 9.81-point architecture regression.

Under the canonical strict train-14 OOF evidence, which evaluates every user as unseen under one frozen protocol, the A4-T validation users score `0.571250` while user6/user7 score `0.493506`. This `0.077744` population gap explains about 79% of the raw A4-T versus Direct-Head Accuracy gap before comparing model interventions.

## Population Audit

| Experiment | Train users | Validation users | Validation trials | Accuracy |
|---|---:|---|---:|---:|
| A4-T | 9 | user18, user20, user21, user3, user9 | 800 | 0.615000 |
| Direct-Head | 12 | user6, user7 | 385 | 0.516883 |

- All five A4-T validation users are Direct-Head training users.
- Both Direct-Head validation users are A4-T training users.
- No sample-level matched comparison exists between these two headline scores.
- A4-T validation observes 39 classes; user6/user7 validation observes all 40, with minimum class support only 2.

## Same-Protocol User-Difficulty Control

The canonical `ir_x3d_s_k400_pure` OOF archive supplies predictions for both groups under the same strict cross-fitted protocol.

| Canonical OOF population | Trials | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| A4-T validation users | 800 | 0.571250 | 0.486918 |
| user6/user7 | 385 | 0.493506 | 0.361817 |
| Difference | | +0.077744 | +0.125101 |

The user-difficulty result is not only a class-frequency artifact. Mean observed-class recall is `0.516509` for the A4-T users and `0.446180` for user6/user7, a `0.070330` difference after giving classes equal weight.

Canonical per-user Accuracy further confirms the shift:

| User | Accuracy |
|---|---:|
| user18 | 0.539326 |
| user20 | 0.572327 |
| user21 | 0.533835 |
| user3 | 0.550336 |
| user9 | 0.646409 |
| user6 | 0.482587 |
| user7 | 0.505435 |

## Matched Intervention Effects

The interventions can only be interpreted inside their own frozen population.

### Temporal clip dropout on the A4-T population

| Candidate | Accuracy | Macro-F1 | Worst-user Accuracy |
|---|---:|---:|---:|
| A2, full temporal coverage | 0.607500 | 0.526830 | 0.539326 |
| A4-T, train clip keep 0.5 | 0.615000 | 0.525929 | 0.563910 |
| A4-T minus A2 | +0.007500 | -0.000901 | +0.024584 |

Clip dropout accounts for only 0.75 Accuracy point on that population. It improves short durations but regresses longer trials: versus A2, duration Accuracy changes by `+0.022222`, `+0.019830`, `-0.021739`, and `-0.052632` for `<=13`, `14-32`, `33-64`, and `>64`.

### Direct classifier on user6/user7

| Candidate | Accuracy | Macro-F1 | Worst-user Accuracy |
|---|---:|---:|---:|
| Partial2 projected head | 0.532468 | 0.421575 | 0.532338 |
| Direct-Head | 0.516883 | 0.423792 | 0.472637 |
| Direct minus Partial2 | -0.015584 | +0.002217 | -0.059701 |

The Direct head explains a matched 1.56-point Accuracy regression, not the full 9.81-point headline difference. It reduces the train-to-validation Accuracy gap from `0.419987` to `0.273298`, but it removes useful transferable capacity, especially for user6.

### Residual decomposition

The raw A4-T versus Direct gap is `0.098117`.

- Same-protocol population difficulty: `0.077744`.
- A4-T temporal intervention over A2: `+0.007500`.
- Direct head versus projected Partial2: `-0.015584` for Direct, equivalently `+0.015584` in favor of the projected head.

The last two matched effects sum to `0.023084`, close to the `0.020373` residual after removing the canonical population gap. Small remaining disagreement is expected because the development training populations, checkpoint trajectories, and full versus partial backbone policies are not jointly matched.

## Training Dynamics

| Measurement | A4-T | Direct-Head |
|---|---:|---:|
| Selected epoch | 18 | 7 |
| Train Accuracy at selection | 0.957237 | 0.790181 |
| Validation Accuracy | 0.615000 | 0.516883 |
| Train-validation gap | 0.342237 | 0.273298 |
| Validation NLL | 1.472460 | 2.122651 |
| Wrong-prediction confidence | 0.589645 | 0.565309 |
| Trainable backbone parameters | 2,974,674 | 2,315,984 |
| Custom head parameters | 535,336 | 81,960 |

Direct-Head is not failing because it cannot fit at all: train Accuracy reaches `0.790181` at its selected epoch and exceeds `0.92` later. Its validation peak occurs early, then training continues to improve while validation does not. The smaller head reduces memorization and confidence, but also has worse NLL and loses user6-specific transferable decisions. This is a representation/generalization tradeoff, not a simple insufficient-epochs problem.

## Current-Split Error Concentration

On user6/user7, projected Partial2 is already weak on object-interaction and low-support classes:

- `Take_and_use_tableware`: 1/27 correct.
- `Use_a_mobile_phone`: 1/9.
- `Eat_food`: 8/25.
- `Take_body_temperature`, `Stand_up`, `Write`, `Make_a_phone_call`, `Watch_TV`, `Massage_oneself`, and `Play_games`: 0 correct.

Direct-Head changes which samples are correct but does not add a new visual signal. The Partial2 plus Direct oracle is only `0.600000`, and the best fixed probability mixture is `0.540260`. Therefore head selection or simple ensembling cannot reach `0.63` on this population.

## Conclusion

Ranked causes of the apparent regression:

1. **Validation-user difficulty and population swap (dominant).** Same-protocol OOF attributes about 7.77 of the 9.81 Accuracy points to the user group.
2. **Direct-head capacity removal.** The matched penalty is 1.56 Accuracy points and 5.97 worst-user points, despite a smaller train-validation gap.
3. **A4-T clip dropout.** It contributes only 0.75 Accuracy point on its own population and trades away long-duration performance.
4. **Full versus partial backbone and different development trajectories.** These remain confounded; current evidence does not support assigning the headline gap to them.

The correct development baseline for user6/user7 is projected Partial2 at `0.532468`, not A4-T at `0.615000`. Reaching `0.63` on user6/user7 requires new evidence, such as interaction/relation ROI, pose-motion, or another modality, rather than another classifier-head variation.

## Evidence

- `reports/x3d_s_fold0_a2_report.json`
- `reports/x3d_s_fold0_a4_t_report.json`
- `reports/x3d_s_train12_val2_user6_user7_partial2_report.json`
- `reports/x3d_s_train12_val2_user6_user7_direct_head1_report.json`
- `outputs/x3d_s_ir_evidence/ir_x3d_s_k400_pure/oof_evidence.npz`
- A4-T and Direct-Head `development_provenance.json`, `resolved_config.yaml`, `run_summary.json`, `history.csv`, and best-Accuracy prediction archives
