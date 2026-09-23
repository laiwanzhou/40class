# Fixed-Split Visual-Motion No-Vote Ablation Design

## Status

**BLOCKED BEFORE IMPLEMENTATION.**

The fixed-split and label-isolation protocol is now specified, but exact reproduction of the teammate P86/MoBind fusion cannot proceed with the current fixed visual teacher. The visual representation contracts are incompatible, and the existing visual checkpoint is historically user6/user7-selected. No training or candidate generation may start until the user chooses one of the alternatives in §10.

## 1. Objective

Condition on the existing correct four-view IR+Depth VideoMAEv2 teacher, then measure the descriptive accuracy change from recreating the teammate's Skeleton, IMU, fusion, repeat/session, and unlabeled-adaptation processing without the 30-teacher bank.

This is a fixed-split development experiment, not an OOF experiment and not an independent unseen-user estimate.

## 2. Fixed Population Split

### Final training population

The following 12 users are the complete supervised training population:

`user1, user2, user3, user5, user8, user9, user16, user18, user19, user20, user21, user22`

### Fixed internal development split

Only one internal split is used for epoch and hyperparameter selection:

- internal fit users: `user1, user2, user3, user9, user16, user18, user19, user20, user21, user22`;
- internal development users: `user5, user8`.

The internal fit side contains all 40 classes. The internal development side contains 33 classes; missing development classes are allowed because it is used only for model/epoch comparison. All heads retain 40 outputs and all metrics use fixed class IDs 0–39.

After the training recipe and epoch budget are frozen, each Skeleton, IMU, and fusion model is refit once on all 12 users. There is no three-fold OOF generation and no OOF claim.

### Final development test

`user6` and `user7`, 388 rows, are used only by the final evaluator.

No user6/user7 label may be read by cache generation, training, checkpoint selection, calibration, fusion fitting, repeat/session fitting, pseudo-target generation, or target adaptation.

## 3. Consequence for the Class-25 Audit Finding

The earlier class-25 blocker was caused by three-fold user OOF: all class-25 training rows belong to user1, so the fold holding out user1 has no class-25 fit examples.

That blocker no longer applies under this fixed split because user1 remains on the internal fit side and in the final 12-user refit. The protocol must assert that the internal fit side contains all 40 classes before training.

This does not make three-fold OOF valid; it removes OOF from the experiment entirely.

## 4. Correct Fixed Visual Teacher

The visual teacher is read-only:

`outputs/ir_depth_videomaev2_teacher/ir_depth_videomaev2_vit_b_train12_val2_seed20260715/selected_checkpoint.pt`

- SHA256: `4b3e89542abd33cb306814f277e3bb40bb7143833a3c0ac9ab23b2d38271429c`
- selected epoch: 6
- architecture: OpenGVLab VideoMAEv2 ViT-B
- modalities: IR and Depth
- views: global, person context, left-hand object, right-hand object
- cached representation: `[N, 2 modalities, 4 views, 768]`
- historical usable-target result: 275/385 = 71.43%

The visual checkpoint is not retrained, fine-tuned, or replaced.

### Historical contamination limitation

This checkpoint was selected using user6/user7 validation metrics. Therefore user6/user7 cannot be described as an untouched test set for the complete pipeline, even if all new Skeleton/IMU/fusion code is label-isolated.

The revised experiment can prevent **new** target-label leakage, but it cannot undo this historical checkpoint-selection contamination. Final results are retrospective descriptive deltas conditional on this checkpoint.

If a genuinely leakage-free user6/user7 test is mandatory, the visual teacher must be retrained with epoch/hyperparameters selected exclusively inside the 12-user population, or a new untouched target population must be used. The current constraint forbids that retraining, so independent-test claims are unavailable.

## 5. Target-Label Isolation

The historical `p2a_view_cache.npz` contains a `labels` array and is forbidden as a generation input.

S0 must be rebuilt from raw user6/user7 visual inputs using the fixed checkpoint. The generated target cache uses an allowlist containing only:

```text
sample_ids
view_logits
view_embeddings
full_logits
availability
quality_features
```

The generation process cannot open historical user6/user7 prediction/report files containing labels, correctness, confusion matrices, selected metrics, or per-class outcomes.

All S0–S6 predictions are generated and hashed before a separate evaluator receives the label path. Predictions must be identical when an inaccessible label file is deleted or randomly permuted.

## 6. Included Processing if the Interface Blocker Is Resolved

The intended non-voting path remains:

| Stage | Processing |
|---|---|
| S0 | fixed visual teacher |
| S1 | teammate-style compact Skeleton branch |
| S2 | teammate-style compact IMU branch plus statistical RF teacher |
| S3 | fixed simple Visual/Skeleton/IMU fusion controls |
| S4 | teammate-style learned motion residual fusion |
| S5 | repeat/session processing on the single fused probability |
| S6 | unlabeled user6/user7 adaptation without P310 targets |

Excluded throughout: Thermal, Radar, MotionBERT, HD-GCN, LaViLa, V-JEPA, InternVideo2, P128/P158/P231/P238/P253/P306, P310 targets, and every expert-bank probability.

## 7. No OOF and No Learned Meta-Stacker

Because this experiment uses a fixed split:

- do not generate or report three-fold OOF metrics;
- do not train a stacker on in-sample base predictions and call it OOF;
- do not learn fusion weights from user6/user7;
- do not select session/repeat thresholds from user6/user7;
- do not use historical target metrics to choose among candidates.

Allowed fusion controls are predeclared equal-probability means and a learned fusion model selected on the fixed internal user5/user8 development split, followed by a full-12-user refit.

## 8. Primary Result

If the interface blocker is resolved, the unique primary artifact is predeclared as:

```text
S4 full-12-user fused model
→ fixed S5 repeat/session operator selected on user5/user8
→ pseudo-target generation on unlabeled user6/user7
→ two independently initialized 12-epoch S6 adaptations
→ arithmetic mean of their logits
→ reapply the same frozen S5 operator
```

The primary contrast is `S6-primary − S0` on all 388 rows.

The 40-epoch S6 endpoint is exploratory and cannot replace the primary result. Individual seeds are sensitivity analyses and cannot be selected by target accuracy.

S5 and S6 are transductive because predictions depend on the complete target batch.

## 9. Fixed-Split Stage Selection

Skeleton, IMU, fusion epochs, and all S5 thresholds are selected only on user5/user8 after fitting on the ten internal-fit users. After selection:

1. freeze architecture, losses, thresholds, epoch budgets, and seeds;
2. refit on all 12 users without early stopping;
3. generate unlabeled user6/user7 outputs;
4. reveal labels once for descriptive evaluation;
5. make no post-reveal changes.

Any later change is a new exploratory experiment and cannot reuse the same result as confirmation.

## 10. Blocking Visual/P86 Interface Mismatch

The fixed visual teacher and teammate P86 fusion do not have matching representations.

### Fixed teacher output

```text
view_embeddings: [B, 2 modalities, 4 views, 768]
view_logits:     [B, 2 modalities, 4 views, 40]
full_logits:     [B, 40]
```

These are pooled clip-level representations. They have no early/late-window dimension and no per-time token dimension.

### Original P86 requirement

```text
visual_sequence: [B, 2 windows, 3 views, T, 512]
visual_width: 512
teacher_features expected by P86 loaders: 1024-dimensional P85 features
```

P86 performs local part/time attention between motion tokens and visual time tokens. A linear `768 → 512` layer only matches the final numeric width; it cannot recreate the missing windows, view semantics, or time-token structure. Repeating a pooled vector across time would fabricate evidence and is prohibited.

Therefore exact teammate MoBind reproduction is impossible without changing one of the approved constraints.

### Available alternatives

#### Alternative A: retrain a compatible visual student

Train the teammate MC3 visual Student from the fixed VideoMAEv2 teacher, then use the original P86 interface exactly. This does not retrain the large visual teacher, but it does create a new visual Student and changes the experiment from “fixed current visual model” to “fixed teacher plus compatible distilled visual Student.”

#### Alternative B: pooled P86-inspired fusion

Keep the current visual model and design a new 768-dimensional clip-level Skeleton/IMU residual adapter. This is feasible and can measure motion-modality value, but it is not an exact reproduction of the teammate P86/MoBind local temporal fusion.

#### Alternative C: probability-level teacher fusion

Keep the current visual teacher and combine independently trained Skeleton/IMU probabilities. This is the smallest and cleanest fixed-split ablation, but it measures teacher-probability fusion rather than the teammate compact Student processing.

The user instructed execution to stop if simple matching is impossible. Accordingly, no implementation plan or training may proceed until one alternative is explicitly approved.

## 11. Engineering Availability

Present locally:

- fixed visual checkpoint and K710 initialization;
- 1,935 train and 385 target historical visual caches, usable only as reference because target cache contains labels;
- raw IR, Depth, Skeleton, and IMU inputs;
- P28/P31/P86 source modules;
- CUDA environment and RTX 5060 Laptop GPU.

Missing and requiring regeneration after an alternative is selected:

- label-free target visual cache;
- P28 pose cache;
- P31 Skeleton/IMU cache;
- P86 motion-window cache;
- fixed-split RF IMU teacher artifacts;
- fixed-split Skeleton/IMU/fusion checkpoints.

The shipped final P87-S checkpoint cannot substitute for these artifacts because it contains an adapted MC3 visual path and motion residual trained with a different visual contract.

## 12. Interpretation

Under the fixed split, the class-25 OOF issue disappears and new target-label leakage can be prevented. Two limitations remain:

1. the fixed visual teacher was historically selected on user6/user7, so the final result is not an independent test estimate;
2. the visual/P86 representation mismatch prevents exact MoBind reproduction without a new compatible Student or a newly designed adapter.

The specification is intentionally blocked at this decision point rather than silently substituting a different method.
