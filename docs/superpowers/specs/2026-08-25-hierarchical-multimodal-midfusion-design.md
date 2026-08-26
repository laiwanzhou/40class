# Hierarchical Multimodal Mid-Fusion Design

Approved direction: 2026-08-25 (Asia/Shanghai)

Protocol amendment approved: 2026-08-25 (Asia/Shanghai). The default experiment
boundary is the fixed train12 to user6/user7 development split. Grouped
three-fold evaluation is disabled unless every fold has all 40 classes in both
fit and validation scopes and the user explicitly authorizes that run.

## 1. Objective

Build a competition-compliant multimodal student whose first retained modality
set is:

```text
IR + Depth_Color + Skeleton + raw IMU
```

The model must fuse aligned evidence before the final classifier. It must not
average independently trained modality logits as its primary fusion mechanism.
The immediate research objective is to determine whether body-motion evidence
can raise the current IR+Depth development result materially enough to justify
a full multimodal teacher and distillation run. The final aspiration remains:

- multimodal teacher Accuracy at least `0.90` on the frozen user6/user7
  canonical union;
- compliant student as close as possible to the teacher while remaining below
  the exact serialized inference-package limit.

This design does not claim that the first four-modality experiment will reach
`0.90`. It defines evidence gates that prevent weak or non-complementary
modalities from entering the deployable model merely because they exist.

## 2. Evidence That Determines the Architecture

### 2.1 Strong visual anchor and its failure mode

The selected IR+Depth VideoMAE V2 checkpoint reaches `0.722078` Accuracy on
385 user6/user7 trials with both modalities. Its Top-3 and Top-5 Accuracy are
`0.906494` and `0.950649`. The current hard Top-2 view router always selects
both wrists; global and person receive zero final routing weight.

Depth is net-positive: disabling it changes 34 predictions, rescues 10, harms
17, and lowers Accuracy to `0.703896`. Depth therefore remains in the visual
anchor, but its residual must be quality- and sample-conditioned rather than
unconditionally saturated.

### 2.2 Early aligned visual fusion is supported

The matched Depth+IR dual-stem pose-ROI experiment improved over Depth-only by:

- Accuracy `+0.027027`;
- Macro-F1 `+0.103658`;
- weighted F1 `+0.056432`.

Masking or sample-shuffling IR sharply reduced performance. IR and Depth must
therefore interact at an aligned spatial feature seam, not only as final
probabilities.

### 2.3 Skeleton is complementary but insufficient as a final classifier

The preferred Skeleton E1 ensemble has strict train14 OOF Accuracy `0.423751`.
On the exact 385 user6/user7 VideoMAE population, its Accuracy is `0.400000`.
It uniquely corrects 19 VideoMAE errors, including:

- 4 `Stand_up` samples;
- 3 `Read_documents` samples;
- 1 `Do_stretching_exercises` sample.

The VideoMAE/Skeleton final-prediction oracle is only `0.771429`. Skeleton must
therefore affect intermediate action representations; perfect late routing
between the current classifiers cannot reach the target.

The image-pose/Skeleton audit supports sequence-level alignment but rejects
frame-level joint fusion. Normalized-time segment tokens are mandatory.

### 2.4 IMU contains motion evidence but the RF interface is too compressed

The finalized compact IMU RF reaches fold0 Accuracy `0.424084` and Macro-F1
`0.344430`. It is strongest on walking, posture changes, jumping, squatting,
and large-body dynamics. Its 2,310 summary features and final probabilities do
not preserve an explicit temporal phase sequence. Mid-fusion therefore uses
raw role-aware IMU segment tokens; RF probabilities remain a diagnostic
reference, not a fusion input.

### 2.5 Thermal and Radar are deferred from Stage 1

Thermal A-direct reaches `0.273210` on user6/user7; the corrected Thermal
R(2+1)D-18 teacher peaks at `0.050398` under severe cross-user overfitting.
Thermal has useful motion, pose, and quality fields, but it has not earned a
Stage-1 model slot.

The existing Radar encoder reduces every frame to 21 statistics and reaches
`0.134538`. This rejects that representation, not raw Radar. Radar cannot enter
mid-fusion before a raw point-set frame encoder produces controlled evidence.

## 3. Frozen Development Population

The authoritative development split remains:

```text
metadata/splits/train12_val2_user6_user7_development.json
```

The canonical population is always the union in `metadata/manifest.csv`.

### Train12

- 2,039 canonical trials;
- 40 classes;
- 1,934 trials with all four Stage-1 modalities;
- 22 trials with IR, Depth, and Skeleton but no IMU;
- 1 trial with IR but no Depth, Skeleton, or IMU;
- 82 Thermal-only trials with none of the Stage-1 modalities.

### User6/user7 development boundary

- 388 canonical trials;
- 40 classes;
- 385 trials with all four Stage-1 modalities;
- 3 Thermal-only trials with none of the Stage-1 modalities.

The primary reported Accuracy is over all 388 rows. The report must separately
show:

- 385 Stage-1-supported rows;
- 3 unsupported Thermal-only rows;
- user6 and user7;
- all 40 classes.

During Stage 1, a row with no usable Stage-1 modality receives a deterministic
train12-only class-prior prediction and `core_available=False`. Its label may
not influence model or fallback selection. This preserves the canonical union
without pretending that a modality the model does not consume is available.

### 3.1 Three-fold coverage audit and protocol decision

The previously frozen train12 grouped split does not satisfy full class
coverage:

| Fold | Fit rows/classes | Validation rows/classes | Missing from fit | Missing from validation |
|---|---:|---:|---|---|
| 0 | 1,414 / 39 | 625 / 39 | class 25 | class 26 |
| 1 | 1,302 / 40 | 737 / 39 | none | class 25 |
| 2 | 1,362 / 40 | 677 / 39 | none | class 25 |

Class 25 has only three train12 samples and all belong to `user1`. Class 26
appears only for `user16`, `user19`, and `user9`. Consequently no strict
user-held-out assignment that validates `user1` can retain class 25 in that
fold's fit scope. Pooling the three validation folds covers 40 classes, but the
fold-0 model has never learned class 25 and its pooled fixed-40 Macro-F1 is
structurally distorted.

The grouped split remains an audit artifact only. It is not an executable
default. A future three-fold run requires both:

1. every fold's fit and validation scopes contain class IDs `0..39`;
2. explicit user authorization after the qualifying split is shown.

Until both conditions hold, all Stage-1 candidate training uses the 2,039-row
train12 scope and all candidate evaluation uses the fixed 388-row user6/user7
development validation scope.

## 4. Normalized Segment Contract

Every usable modality is represented on `K=8` normalized trial segments. Raw
frame-to-frame correspondence across sensor clocks is not required.

```python
@dataclass(frozen=True)
class SegmentBatch:
    tokens: torch.Tensor       # [B, 8, S, D]
    token_mask: torch.Tensor   # bool [B, 8, S]
    quality: torch.Tensor      # [B, 8, S, Q]
    quality_mask: torch.Tensor # bool [B, 8, S, Q]
```

`S` is the number of semantic streams within one modality group. Empty source
intervals remain masked; they are never filled with an apparently valid zero
token. All normalization statistics are fit only on the current training-user
scope.

## 5. Stage-1 Architecture

### 5.1 Visual group

The visual group consumes IR and Depth_Color for four views:

```text
global, person_context, left_hand_object, right_hand_object
```

For every segment and view:

1. modality-specific stems encode aligned IR and Depth crops;
2. a zero-initialized Depth residual is added to the IR anchor at the spatial
   feature map level;
3. global/person are softly fused inside the `context` subgroup;
4. left/right hand are softly fused inside the `wrist` subgroup;
5. both subgroup tokens are always returned when available.

The module output is:

```text
visual_tokens [B,8,2,D]  # subgroup order: context, wrist
visual_mask   [B,8,2]
```

There is no hard Top-k across context and wrist. A subgroup may receive a small
weight at inference, but it must remain in the training graph.

### 5.2 Body-motion group

Skeleton preprocessing uses the accepted H36M-17 representation:

- root/scale-normalized `xyz`;
- segment-local velocity;
- gap-aware masks;
- no interpolation across disconnected retained segments.

Multi-person Skeleton candidate identity and image-to-Skeleton projection use
the existing `selected_final` clean view. Its projection is fit on all train12
users and applied to train12 plus user6/user7. The projection never fits a
user6/user7 row. Previously generated fold-specific clean views remain audit
artifacts and are not consumed by the fixed-validation experiment.

Raw IMU preserves five device-role slots and 16 channels per role. Missing
roles have explicit role masks. It is resampled into the same eight normalized
segments without converting the complete trial to summary statistics.

Skeleton and IMU are encoded separately, then interact through one bidirectional
masked cross-attention block:

```text
skeleton segment token <-> IMU segment token
```

The output is one body-motion token per segment:

```text
body_tokens [B,8,1,D]
body_mask   [B,8,1]
```

If only one body modality is usable, the group returns that modality's token.
If neither is usable, the body token is masked.

### 5.3 Cross-group action-query fusion

Forty learned action queries attend to the concatenated segment tokens:

```text
[context, wrist, body_motion] x 8 segments
```

The fusion block has two masked cross-attention layers with residual MLPs. It
returns:

```text
logits          [B,40]
action_features [B,40,D]
group_attention [B,40,3]
segment_attention [B,40,8]
```

The classifier is applied to each action-specific feature using one class row,
not to a single globally pooled vector.

### 5.4 Auxiliary heads and anti-collapse rules

Training-only heads predict the 40 actions from:

- context tokens;
- wrist tokens;
- body-motion tokens;
- the fused action-query representation.

Required anti-collapse mechanisms:

- independent group auxiliary CE;
- visual subgroup dropout: context 10%, wrist 10%, never both simultaneously;
- body-group dropout: 15% on rows where the visual group is usable;
- visual-group dropout: 10% on rows where the body group is usable;
- group-attention entropy floor during the first two epochs only;
- no positive wrist routing bias;
- availability and quality masks applied before every softmax.

The entropy floor is a warm-start constraint, not a permanent equal-weight
penalty. After epoch 2, the model may ignore a group for an individual sample.

## 6. Teacher and Student

### 6.1 Teacher

The teacher uses:

- VideoMAE V2 ViT-B visual encoder initialized from the approved K710
  checkpoint;
- a 128- or 256-channel Skeleton temporal encoder initialized from accepted
  Skeleton evidence when shape-compatible, otherwise trained with its own
  auxiliary head;
- a role-aware IMU temporal encoder trained from raw Stage-1 sequences;
- action-query fusion dimension `D=256`.

The large teacher is training-only and is excluded from the inference package.

### 6.2 Student

The student consumes the same four Stage-1 modalities and uses:

- a lightweight 3D CNN or MobileNet-style visual encoder;
- compact depthwise temporal Skeleton and IMU encoders;
- action-query fusion dimension `D=128`;
- the same masks, subgroup semantics, and output contract as the teacher.

Teacher and student need not share embedding coordinates. Distillation aligns
observable relations rather than raw hidden vectors.

### 6.3 Distillation losses

```text
L = 1.0 * CE(student, label)
  + 1.0 * KL(student_logits / T, teacher_logits / T) * T^2
  + 0.25 * relation_loss
  + 0.10 * group_attention_loss
  + 0.20 * auxiliary_group_CE
```

with `T=4` for the first registered run.

`relation_loss` compares the cosine-similarity matrices of eight segment-level
action features, not their coordinates. `group_attention_loss` compares
teacher and student group distributions only where both models mark the group
usable.

## 7. Training and Leakage Policy

The architecture, loss weights, fixed 15-epoch schedule, three-candidate set,
metric order, masks, and fallback behavior are frozen before user6/user7 model
evaluation. For each candidate:

1. fit Skeleton and IMU normalization on train12 only;
2. train on eligible train12 rows only;
3. evaluate train12 once after epoch 15 with dropout disabled;
4. evaluate the canonical 388-row user6/user7 development validation once;
5. store logits, masks, quality, action/group attention, and provenance.

Diagnostic tensors produced under BF16 autocast are converted to FP32 only at
the NumPy archive boundary. A completed-training checkpoint may cross a source
hash change only through an explicitly registered evaluation-only recovery
whose candidate, prior config hash, and completed epoch all match; incomplete
training checkpoints remain strict-hash only.

User6/user7 rows may enter metric computation and the pre-registered three-way
candidate selection. They may not enter gradients, normalization, the sampler,
class-prior fitting, epoch selection, loss design, threshold fitting, or
fallback fitting. Repeated candidate comparison makes this a development-set
selection result, not an untouched independent final-test estimate.

The candidate order is fixed as `visual_only`, `visual_skeleton`, and
`visual_skeleton_imu`. The standalone `visual_imu` candidate is omitted by
explicit user decision because existing IMU-only evidence is weak. IMU's
incremental value is still measured conditionally by comparing
`visual_skeleton_imu` against `visual_skeleton`. Selection uses Accuracy, Macro-F1, worst-user
Accuracy, negative NLL, then fixed candidate order. Because every candidate is
already trained on all train12 users, no second selected-candidate retraining or
second user6/user7 evaluation event is performed.

Teacher-to-student distillation remains outside this Stage-1 execution. Its
later plan must define train12 target ownership without assuming that the
disabled grouped split is available.

## 8. Modality Admission and Removal

A modality is not retained because its standalone Accuracy is high. In this
protocol, admission requires pre-registered fixed user6/user7 development
evidence that it contributes complementary information.

### Skeleton and IMU

They are jointly admitted to the Stage-1 hypothesis because their physical
signals and observed class strengths target the current visual failure modes.
The experiment must still report each modality's removal ablation:

```text
visual only
visual + Skeleton
visual + Skeleton + IMU
```

All three pre-registered candidates receive one user6/user7 metric evaluation.
Only the metric-selected candidate is eligible for later teacher promotion.

### Thermal admission gate

Thermal remains excluded until an authorized experiment using native
thermal motion/pose/quality tokens satisfies all of:

- at least `+0.010` Accuracy over the frozen four-modality model;
- positive net rescue;
- no more than `-0.005` worst-user Accuracy;
- usable coverage and fallback behavior reported on the canonical union.

### Radar admission gate

Radar remains excluded until an authorized raw point-set frame experiment
satisfies all of:

- standalone Accuracy at least `0.25` on its frozen evaluation boundary;
- at least `+0.010` Accuracy over the frozen retained model;
- positive net rescue;
- finite masked behavior for empty Radar frames.

## 9. Missing-Modality Behavior

Every forward call accepts an availability mask. Required exact behaviors:

- with visual only, return the visual action-query model;
- when and only when YOLO produces no valid person pose at every fixed trial
  probe, retain the global IR/Depth view and mark person plus both wrist views
  unavailable; do not apply this fallback to cache, pairing, shape, or geometry
  errors;
- with Skeleton or IMU missing, use the remaining body stream;
- with the complete body group missing, mask all body attention;
- with visual missing but body available, return a body-only prediction;
- with no Stage-1 modality usable, return the train12 class-prior fallback and
  `core_available=False`;
- no unavailable token may receive nonzero attention;
- no missing modality may be represented by an unmasked learned token.

Natural missing patterns are primary training evidence. Synthetic group dropout
is regularization and receives lower loss weight than natural examples.

## 10. Evaluation and Gates

Every result reports:

- canonical-union Accuracy and fixed-40 Macro-F1;
- supported-row Accuracy;
- user6 and user7 Accuracy;
- worst-user Accuracy;
- Top-3 and Top-5 Accuracy;
- per-class recall;
- predicted-class coverage and zero-recall classes;
- group removal ablations;
- rescue/harm against the visual anchor;
- group and segment attention statistics;
- natural missing-pattern metrics;
- train/validation generalization gap;
- serialized model and preprocessing bytes;
- exact cache, split, checkpoint, config, and code hashes.

The report must label user6/user7 as `development_validation`, record that all
three candidates were compared on it, and set `independent_final_test=False`.
It must report final dropout-disabled train12 and validation metrics for every
candidate so the train-to-validation generalization gap is directly auditable.

### Stage-1 research gates

The first four-modality result is categorized as:

- `reject`: canonical-union Accuracy below `0.76` or net rescue non-positive;
- `promising`: Accuracy `0.76–0.799999`, positive net rescue, and worst-user
  no worse than the visual anchor by more than `0.01`;
- `full_teacher_worthy`: Accuracy at least `0.80`, positive net rescue, and
  worst-user at least `0.78`;
- `teacher_target_reached`: Accuracy at least `0.90`, Macro-F1 at least `0.86`,
  and worst-user at least `0.88`.

The eventual `0.90` target requires at least 350 correct rows out of 388. If all
three Thermal-only rows remain incorrect, the supported 385 rows must reach at
least `350/385 = 0.909091`.

### Student promotion gates

The student is retained only when:

- its canonical-union Accuracy is no more than `0.03` below the selected
  teacher;
- it improves over the same architecture trained without distillation;
- worst-user regression versus the nondistilled student is at most `0.01`;
- the exact complete inference package is below `95,000,000` bytes.

## 11. Size Budget

The Stage-1 student target budget is:

| Artifact | Target bytes |
|---|---:|
| YOLO11n-pose | 6,255,593 |
| visual encoder and classifier | <= 30,000,000 |
| Skeleton encoder | <= 3,000,000 |
| IMU encoder | <= 3,000,000 |
| hierarchical fusion and auxiliary-free inference head | <= 5,000,000 |
| normalization/configuration/quality metadata | <= 1,000,000 |
| reserve for Thermal/Radar or packaging variance | >= 46,000,000 |

Training-only auxiliary heads may be removed from the inference state only when
the exported inference model is reloaded and shown to reproduce final logits.

## 12. Non-Goals

- No late weighted average of the current six classifier logits as the primary
  model.
- No frame-level YOLO/Skeleton hard joint correspondence.
- No current Thermal classifier or Radar 21-stat logits in Stage 1.
- No exhaustive artificial generation of all modality subsets.
- No user6/user7 rows in gradients, normalization, sampling, class-prior
  fitting, epoch selection, loss design, threshold fitting, or fallback
  fitting. Their labels are allowed only for the pre-registered three-candidate
  development comparison and reporting.
- No three-fold execution without both complete `0..39` class coverage in each
  fit and validation scope and explicit user authorization.
- No competition-test access.

## 13. First Implementation Boundary

The first implementation ends after all three pre-registered candidates are
trained on train12, evaluated once on fixed user6/user7 development validation,
and compared in one atomic report. There is no second selected-only retraining
or evaluation event.
Student distillation starts only if the candidate reaches
`full_teacher_worthy`. Thermal and Radar are separate approved designs after
their admission evidence exists.
