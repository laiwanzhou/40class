# X3D Single13-Global user6/user7 experiment

**Goal:** Determine whether the adaptive `K x 13` temporal decomposition is contributing to cross-subject overfitting by replacing it with one 13-frame clip sampled across the complete trial.

**Scientific role:** Development-only, single-variable temporal ablation. This experiment is an explicitly approved exception to the frozen pure-IR route. It cannot replace canonical Phase 4/5 evidence without a later, separately approved train-14 OOF experiment.

## Frozen population

- Split artifact: `metadata/splits/train12_val2_user6_user7_development.json`.
- Development train: `user1,user2,user3,user5,user8,user9,user16,user18,user19,user20,user21,user22`.
- Development validation: `user6,user7`.
- Sealed heldout: `user4,user17,user23,user24`.
- Usable IR: 1,935 train trials and 385 validation trials.
- Both sides cover all 40 classes; validation minimum class support is 2.
- Heldout-4 and competition test access remain forbidden.

## Matched reference

The sole matched reference is `x3d_s_ir_context_train12_val2_user6_user7_partial2_seed20260715`:

- Accuracy: `0.5324675325`.
- Macro-F1: `0.4215746252` over labels `0..39`.
- Worst-user Accuracy: `0.5323383085`.
- Selected epoch: 14.

Historical fold0, user21/user22, canonical OOF, notebook public-LB, and Direct-Head results are descriptive context only and are not matched references.

## Sole intervention

Reference:

```text
ordered trial
  -> K = min(8, max(1, ceil(T/32))) contiguous local windows
  -> 13 stratified frames per window
  -> K X3D clips
  -> mean probability
```

Candidate:

```text
ordered trial
  -> one window [0,T)
  -> 13 equal temporal bins over the complete trial
  -> one sampled frame per bin
  -> one X3D clip
```

Training selects one random frame inside each bin using the existing epoch-deterministic generator. Validation selects each bin midpoint deterministically. Motion-peak selection, clip dropout, extra temporal views, and test-time augmentation are forbidden.

All other fields are byte-for-byte inherited from the user6/user7 Partial2 config: ROI assets, X3D-S projected head, K400 initialization, last-two-block unfreeze, learning rates, optimizer, equal-trial accumulation, augmentation, BatchNorm policy, loss, seed, epoch budget, scheduler, early stopping, and checkpoint rule.

## Metrics and decision

Report Accuracy, fixed-40-class Macro-F1, per-user Accuracy, worst-user Accuracy, duration buckets, selected epoch, train-to-validation gap, and matched prediction disagreement.

- `preferred_temporal_candidate`: Accuracy improves over the matched reference, Macro-F1 does not fall by more than 0.01, and worst-user Accuracy does not fall by more than 0.02.
- `non_winning_ablation`: the candidate does not satisfy the preferred rule but Accuracy remains above `0.5124675325`.
- `human_review_regression`: Accuracy is below `0.5124675325`. Preserve artifacts and stop; never delete automatically.

No threshold in this development experiment changes canonical IR evidence automatically.

## Execution tasks

- [x] Task 0: Verify user6/user7 IR and strict IR/Depth_Color pairing coverage.
- [x] Task 1: Add explicit `global_single_clip` dataset/config mode with regression tests.
- [x] Task 2: Freeze hashes and pre-result experiment registration.
- [x] Task 3: Run CPU contract tests and protected CUDA smoke.
- [ ] Task 4: Run one formal seed `20260715` development training.
- [ ] Task 5: Generate matched report, verify canonical hashes, and freeze the result.

## Next boundary

The later IR+Depth_Color co-expert is a separate experiment generation. This Single13 result may select its temporal prior, but Depth_Color must not be introduced into the present run.
