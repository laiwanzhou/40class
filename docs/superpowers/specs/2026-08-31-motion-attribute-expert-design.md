# Motion Attribute Expert Design

Approved direction: 2026-08-31 (Asia/Shanghai)

## 1. Objective

Train one native H36M-17 Motion Attribute Expert that preserves geometric and
temporal evidence useful for correcting a frozen IR+Depth visual anchor. The
expert is not expected to identify held objects or serve as a primary 40-class
classifier.

The first experiment ends after expert qualification. A bounded residual gate
may run only if every expert gate passes. No visual backbone is fine-tuned.

## 2. Evidence Boundary

Repository evidence establishes:

- E1 three-seed C1-TCN strict OOF: Accuracy `0.423751`, Macro-F1 `0.315606`;
- T=96 TCN: Accuracy `0.422469`, Macro-F1 `0.321171`, but unstable by user/fold;
- lightweight fixed ST-GCN: Accuracy `0.235797`, severe underfitting;
- MotionBERT-Lite frozen probe: Accuracy `0.118557`, domain mismatch and class
  collapse;
- current visual-only: Accuracy `0.706186`;
- visual+Skeleton: Accuracy `0.742268`, with concentrated harm on `Eat_food`;
- Skeleton body auxiliary: Accuracy `0.314433`.

The new model therefore uses known-compatible native `xyz+velocity`, a modest
TCN, deterministic geometric targets, and explicit motion-family supervision.

## 3. Population and Leakage Policy

- Canonical train12: 2,039 rows; Skeleton-supported: 1,956.
- Canonical validation: 388 user6/user7 rows; Skeleton-supported: 385.
- Class order: `0..39`; all canonical rows remain in prediction archives.
- Fixed split: train12 -> user6/user7 development validation.
- user6/user7 may enter final expert and conditional residual metrics only.
- user6/user7 may not enter normalization, attribute scaling, gradients,
  sampling, checkpoint selection, or class-prior fitting.
- No grouped CV, multiple seed, early stopping, competition-test access, or
  post-hoc epoch selection.

## 4. Input Contract

Source:

```text
outputs/midfusion_skeleton_clean_views/selected_final/clean_view.csv
```

Use the frozen selected candidate identity and train12-fit projection ownership,
but Motion Attribute input remains native H36M-17 3D coordinates.

For each trial:

1. retain `use_for_frame_training=True` rows;
2. sort by frame ID and preserve retained-segment IDs;
3. normalize each frame by median H36M bone length;
4. compute velocity only between adjacent rows in the same retained segment;
5. resample to `T=96` on the complete normalized trial axis;
6. interpolate only inside one retained segment;
7. leave cross-gap target positions masked and zero;
8. output `features [96,17,6]`, `mask [96]`, `segment_ids [96]`;
9. unsupported canonical rows return zero tensors and `available=False`.

## 5. Six Motion Families

The auxiliary target is multi-label. Class membership is frozen before results:

```text
locomotion:
  28 Jog_in_place, 36 Walk

posture_transition:
  32 Stand_up, 33 Lie_down, 34 Sit_down

exercise:
  29 Do_squats, 30 Do_jumping_jacks,
  31 Do_stretching_exercises, 35 Do_lunges

whole_body_motion:
  3 Take_off_clothes, 5 Put_on_clothes,
  12 Sweep_the_floor, 13 Mop_the_floor,
  15 Wipe_windows_and_tables, 16 Fold_clothes,
  plus all locomotion/posture/exercise classes

upper_body_dominant:
  0,1,2,3,4,5,6,7,8,9,10,11,14,17,18,19,
  20,21,22,23,24,25,26,27,37,38,39

mostly_static_fine:
  17 Tap_the_keyboard, 18 Write, 20 Check_the_time,
  21 Read_documents, 22 Turn_pages,
  23 Listen_to_music_with_headphones,
  25 Watch_TV, 26 Play_games
```

Classes may belong to more than one family.

## 6. Sixteen Deterministic Geometric Attributes

Attributes are computed before model training from normalized valid frames:

1. root horizontal displacement;
2. root vertical displacement;
3. mean root speed;
4. maximum root speed;
5. mean all-joint speed;
6. maximum all-joint speed;
7. upper-body motion energy;
8. lower-body motion energy;
9. upper/lower motion-energy log ratio;
10. bilateral motion asymmetry;
11. mean knee flexion;
12. maximum knee flexion;
13. mean hand-to-head distance;
14. mean hand-to-torso distance;
15. static-frame fraction;
16. dominant normalized motion frequency.

H36M root is the midpoint of hips 1/4; hands are joints 13/16; head is joint
10; torso is joint 8; knees are joints 2/5 with hip-knee-ankle chains
`1-2-3` and `4-5-6`. Upper-body joints are `8..16`; lower-body joints are
`1..6`.

Attribute mean/std are fit on Skeleton-supported train12 rows only. Regression
targets are z-scored with those statistics. Validation attributes never affect
scaling.

## 7. Model

Input is flattened to `[B,96,102]`. Three residual multi-scale temporal blocks
use output channels `[128,192,256]`. Each block combines temporal convolutions
with kernels `3,5,9`, dilation `1,2,4`, masked residuals, LayerNorm, GELU, and
dropout `0.2`. Masked temporal pooling produces one 256D motion embedding.

Heads:

```text
family_head:    256 -> 6
attribute_head: 256 -> 16
action_head:    256 -> 40
```

Total parameters must remain below 3,000,000. The model is an expert/training
artifact and is not yet a student-package component.

## 8. Loss and Training

```text
L = 1.00 * BCEWithLogits(motion families)
  + 0.50 * SmoothL1(normalized attributes)
  + 0.25 * CE(40-class auxiliary head)
```

- seed `20260715`;
- fixed 15 epochs;
- batch size 32;
- uniform trial shuffle over Skeleton-supported train rows;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- CUDA AMP;
- gradient clip `1.0`;
- no validation evaluation until epoch 15;
- checkpoint every epoch with optimizer and RNG state;
- `num_workers=0` for the first run because inputs are cached tensors.

## 9. Smoke Gate

Two supported train rows and two supported validation rows must prove:

- exact input/output shapes;
- finite family, attribute, action losses and gradients;
- encoder and all three heads change after one train optimizer step;
- validation rows never enter backward;
- unsupported-row fallback is finite;
- checkpoint reload reproduces logits;
- peak CUDA memory is below 8,151 MiB;
- formal smoke output path did not previously exist.

## 10. Expert Qualification Gate

Primary expert gates on canonical user6/user7:

- family macro-F1 at least `0.65`;
- locomotion recall at least `0.65`;
- posture-transition recall at least `0.65`;
- exercise recall at least `0.65`;
- normalized attribute MAE at most `0.35`;
- at least 15 unique correct rows relative to visual-only;
- visual-only plus action-head oracle Accuracy at least `0.77`;
- at least one unique rescue for user6 and user7;
- finite predictions and exact canonical sample ownership.

40-class auxiliary Accuracy/Macro-F1 are reported but are not promotion gates.

If any expert gate fails, stop without residual training.

## 11. Conditional Frozen-Visual Residual

This stage is conditional on a complete expert pass. It consumes frozen
visual-only logits/action evidence and the 256D motion embedding. The visual
model is never updated.

Residual budgets are fixed by class family:

```text
mostly-static/object classes: 0.05
whole-body mixed classes:     0.20
locomotion/posture/exercise:   0.50
```

The residual gate is a small MLP trained on train12 cached evidence. It qualifies
only if:

- canonical Accuracy exceeds `0.742268`;
- rescue > harm;
- user6 and user7 do not regress;
- `Eat_food` loses at most one correct row;
- zero-recall class count does not increase.

## 12. Outputs

```text
outputs/motion_attribute_expert/
reports/motion_attribute_expert_smoke.json
reports/motion_attribute_expert_result.json
reports/motion_attribute_expert_result.md
reports/motion_attribute_residual_result.json   # conditional
```

All reports include canonical and supported metrics, family metrics, attribute
errors, per-user/per-class results, visual rescue/harm/oracle, resource use,
train-validation gap, and source/config/cache/checkpoint/prediction hashes.

## 13. Non-Goals

- No MotionBERT continuation.
- No CTR-GCN, PoseC3D, ST-GCN search, or ensemble.
- No IR/Depth backbone fine-tuning.
- No IMU, Thermal, Radar, grouped CV, multi-seed, or competition test.
- No student distillation in this experiment.
