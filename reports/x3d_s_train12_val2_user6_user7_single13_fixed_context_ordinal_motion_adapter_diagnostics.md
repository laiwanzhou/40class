# Ordinal Depth Motion Adapter Diagnostics

This is a post-result diagnostic on the frozen user6/user7 development population. It does not alter the preregistered decision.

## Result

The selected checkpoint is epoch 9 of 17. Accuracy is `0.548052`, Macro-F1 is `0.462041`, and worst-user Accuracy is `0.542289`. Accuracy ties the absolute-Depth anchor, Macro-F1 improves by `0.028964`, and worst-user Accuracy changes by `-0.001190`.

At epoch 9, train Accuracy is `0.897674`; the train-to-validation gap remains `0.349622`. The representation changes which unseen-user trials are solved, but it does not remove X3D cross-subject overfitting.

## Complementarity

| Reference | Agreement | Motion-only correct | Reference-only correct | Pair oracle Accuracy |
|---|---:|---:|---:|---:|
| Absolute Depth anchor | 0.781818 | 16 | 16 | 0.589610 |
| Fixed-context IR | 0.807792 | 18 | 14 | 0.584416 |
| Expanded raw 4-channel stem | 0.535065 | 65 | 30 | 0.625974 |

The largest reliable gain is `Drink_water` (`+0.1333`, support 15). Larger fractional gains occur for several low-support classes and should not be over-interpreted. The largest losses are `Tap_the_keyboard` (`-0.2857`, support 7), `Stir_drinks` (`-0.2222`, support 18), and `Read_documents` (`-0.2222`, support 9).

## Frozen Interpretation

The exact route tested here was useful but not decisive: fixed-box spatial alignment followed by dense ordinal motion maps and X3D convolution preserves the matched Accuracy while improving class balance. It remains below the `0.63` stability-review gate and does not authorize another seed, fold, heldout evaluation, or automatic follow-up experiment.
