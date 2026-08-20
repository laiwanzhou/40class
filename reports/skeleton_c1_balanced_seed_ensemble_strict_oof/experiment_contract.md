# E2-v1 Balanced E1 Frozen Experiment Contract

## Scientific Question

Does moderate class-frequency rebalancing improve the frozen E1 Skeleton expert under the same strict cross-user OOF protocol?

## Single Experimental Variable

- E1 uses ordinary shuffled training batches.
- E2-v1 assigns each training sample from class `c` the sampler weight `1 / sqrt(n_c)`.
- `n_c` is computed independently from the current training scope only: inner-fit for epoch selection and outer-train for formal refit.
- Sampling uses replacement and draws exactly the training-scope sample count per epoch.

## Frozen Components

- Master Clean Skeleton v1 and all six strict clean views.
- Frozen `metadata/splits/train14_oof_3fold.json` user ownership.
- C1 per-frame H36M bone-length scale.
- Joint `xyz + velocity` 102D input, gap-aware T=64 representation, and train-scope normalization.
- Original `TemporalClassifier` with channels `[64, 128]`, 128D embedding, and 40-class logits.
- Optimizer, scheduler, CE loss, batch size, epoch selection, and formal refit protocol.
- Seeds `20260812`, `20260912`, and `20261012` and fixed equal mean of softmax probabilities.
- No augmentation, weighted loss, logit adjustment, member selection, or ensemble-weight fitting.

## Pre-Registered Decision

E2-v1 replaces E1 only if all conditions hold:

1. Combined OOF Accuracy does not decrease.
2. Combined OOF 40-class Macro-F1 increases.
3. The paired bootstrap 95% interval for the Macro-F1 delta is above zero.
4. Macro-F1 improves in at least two of three outer folds.

Per-user, per-class, class-support-bucket, Accuracy bootstrap, and McNemar results are diagnostics. No outer-fold result gates execution of later folds.

