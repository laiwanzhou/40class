# Visual-Motion Processing Without Teacher Voting Design

## 1. Objective

Measure how much accuracy is added by the teammate's non-voting training and processing pipeline when applied to the owner's correct visual teacher.

The experiment keeps the current four-view IR+Depth VideoMAEv2 teacher fixed, reconstructs only the Skeleton and IMU branches actually used by the teammate's compact P87-S/P315 Student, then measures fusion training, repeat/session processing, and unlabeled target adaptation as separate stages.

The experiment explicitly excludes the 30-teacher bank and its voting/router outputs. It does not attempt to reproduce the final 0.91542 submission. Its result is a stage-by-stage development ablation on user6/user7.

## 2. Success Criteria

The primary success criterion is a trustworthy attribution of change from S0 through S6. A practically meaningful improvement is at least four additional correct rows on the full 388-row user6/user7 holdout, approximately 1.03 percentage points, without lowering either user's accuracy by more than two percentage points.

Reaching 0.91 is not a completion requirement. A negative result is valid when the protocol and label isolation pass.

## 3. Branch and Isolation

Implementation work belongs to branch `experiment/single-visual-processing-replication`, based on commit `9e77f66a5102c5ca0bb282c0c1e9189e41497ec6`, in an isolated worktree.

The following existing locations are read-only:

- `40class-x3d-adaptive-multiclip/` and all of its dirty working-tree files;
- `40class/` and its dirty working tree;
- both teammate submission packages;
- all existing checkpoints, caches, reports, and predictions.

New generated artifacts live only under:

`outputs/visual_motion_no_vote_ablation/`

## 4. Fixed Visual Teacher

The sole visual teacher is:

`outputs/ir_depth_videomaev2_teacher/ir_depth_videomaev2_vit_b_train12_val2_seed20260715/selected_checkpoint.pt`

- SHA256: `4b3e89542abd33cb306814f277e3bb40bb7143833a3c0ac9ab23b2d38271429c`
- selected epoch: 6
- architecture: OpenGVLab VideoMAEv2 ViT-B
- initialization: Kinetics-710, distilled from VideoMAEv2 ViT-Giant
- modalities: IR and Depth
- views: global, person context, left-hand object, right-hand object
- frames: 16
- historical result: 275/385 = 71.43% on the usable user6/user7 cache

The visual teacher weights, view-fusion mechanism, and input geometry remain unchanged throughout S0-S6. The experiment may cache its embeddings and logits but may not fine-tune the visual teacher.

Explicitly excluded visual alternatives:

- P3-R1 `full_hard2` and other hard view routes;
- wrist-person residual models;
- `visual_skeleton`;
- MCG-NJU VideoMAE-Large Ridge reconstruction;
- any second pretrained visual teacher.

## 5. Included and Excluded Modalities

### Included

- Visual: fixed VideoMAEv2 IR+Depth teacher.
- Skeleton: teammate-style P86 MoBind Skeleton branch.
- IMU: teammate-style P86 MoBind IMU branch plus its statistical Random Forest teacher.

### Excluded from the first ablation

- Thermal;
- Radar;
- LaViLa;
- V-JEPA;
- InternVideo2;
- MotionBERT;
- HD-GCN;
- P128/P158/P231/P238/P253/P306 experts;
- P310 structured targets derived from the 30-teacher bank.

Thermal and Radar are not part of the deployed compact P87-S Student. Adding them in this experiment would reintroduce the historical expert-bank factor that the ablation is intended to exclude.

## 6. Population and Fold Contract

The frozen target holdout is all 388 user6/user7 rows.

Training uses the same 12 users as the correct visual teacher:

`user1, user2, user3, user5, user8, user9, user16, user18, user19, user20, user21, user22`

All model or threshold selection inside the training population uses the existing three grouped folds:

- fold 0 held users: `user1, user18, user21, user5`;
- fold 1 held users: `user16, user19, user22, user8`;
- fold 2 held users: `user2, user20, user3, user9`.

Each train row receives exactly one OOF prediction per stage. After OOF evaluation freezes architecture and epoch budgets, the branch is refit on all 12 training users and predicts unlabeled user6/user7 inputs.

Every outer source-fit partition must contain all 40 classes; otherwise that fold stops rather than silently training a reduced head. Held partitions may lack a class naturally. All heads retain 40 outputs, and macro-F1 is calculated over the fixed class IDs 0–39 with zero contribution for absent/zero-recall classes.

The primary denominator is all 388 rows. Results on the historical 385-row visual-cache subset are secondary. Three rows missing from the historical visual cache remain in the primary denominator and use only actually available modalities.

## 7. Common Teacher Artifact Contract

Each stage writes a self-describing NPZ plus JSON provenance. Required arrays are:

```text
sample_ids: [N] string
user_ids: [N] string for training artifacts only
fold_id: [N] int for OOF artifacts only
logits: [N, 40] float32
probabilities: [N, 40] float32
availability: [N] bool
quality_features: [N, Q] float32
embedding: optional [N, D] float16/float32
```

Target artifacts must not contain labels or correctness fields. Training OOF artifacts may contain labels in a separate evaluation payload, but label arrays cannot be passed into target-generation datasets.

Each JSON record contains source hashes, configuration hash, exact user ownership, checkpoint hash, row count, class count, missing-row policy, and software versions.

## 8. Label Isolation and Reveal Boundary

Candidate generation and evaluation are separate processes.

Before the reveal, generation may read:

- labels for the 12 training users;
- raw and cached modalities for user6/user7;
- user6/user7 sample IDs, timestamps, duration, modality availability, and device metadata.

Before the reveal, generation may not read:

- user6/user7 labels;
- saved user6/user7 correctness indicators;
- label-derived user6/user7 confusion tables;
- historical reports when they expose candidate performance on user6/user7 for a candidate being selected.

All S0-S6 predictions and hashes are written to `generation_complete.json` before the evaluator receives the label path. After reveal, no configuration, threshold, candidate set, confidence cutoff, epoch count, or seed may change.

## 9. Stage S0: Fixed Visual Baseline

S0 reproduces the original 385-row predictions from `p2a_view_cache.npz`:

- exact sample IDs and order;
- identical argmax prediction;
- maximum logit difference no greater than `1e-5`;
- 275/385 historical result after reveal.

For the three excluded cache rows, perform a fresh visual forward pass using available IR and/or Depth with the original availability mask. If neither modality is usable, S0 falls back to the 12-user training class prior. These three rows are reported individually.

S0 is the reference for every rescue/harm comparison.

## 10. Stage S1: Teammate-Style Skeleton Branch

S1 reconstructs the compact P86 MoBind Skeleton branch used by the deployed Student rather than selecting a new standalone Skeleton architecture.

### Input

- H36M-17 joints from official Skeleton predictions;
- body-relative normalization matching the P31/P86 preprocessing;
- 16 aligned visual-time bins;
- part-aware joint groups;
- position and local motion features;
- explicit availability masks and quality summaries.

### Model and supervision

- P86 MoBind Skeleton encoder and Skeleton classification head;
- supervised 40-class cross-entropy;
- visual-teacher logit distillation;
- visual-teacher feature cosine alignment;
- local part/time token alignment where the fixed visual teacher exposes compatible embeddings;
- reconstruction and contrastive terms from the teammate P86 pretraining recipe;
- no MotionBERT, HD-GCN, C1-TCN, CTR-GCN, or Skeleton ensemble.

For each outer fold, visual teacher targets for the held training users are generated by the frozen visual checkpoint. Skeleton normalization statistics and trainable parameters use source users only.

S1 produces Skeleton-only OOF/target probabilities and a Visual+Skeleton fusion candidate.

## 11. Stage S2: Teammate-Style IMU Branch

S2 reconstructs the P86 MoBind IMU branch and the selected statistical IMU teacher.

### Input

- five device roles;
- raw accelerometer and gyroscope channels;
- device-relative compensated channels;
- relative quaternion features;
- exact 16-bin alignment to the visual time grid;
- role and missing-device masks;
- train-only normalization statistics.

### Teacher and supervision

- reproduce `stat_random_forest_device_dropout_aligned_oof` inside each outer training fold;
- IMU teacher distillation weight `1.0`;
- supervised IMU classification loss;
- visual feature/logit alignment following P86 pretraining;
- asymmetric IMU-to-Skeleton token, semantic, and logit alignment only in the combined S3 candidate;
- no separate deep IMU ensemble.

S2 produces IMU-only OOF/target probabilities and a Visual+IMU fusion candidate.

## 12. Stage S3: Simple Three-Modality Controls

Before reproducing the teammate fusion trainer, S3 establishes low-capacity controls:

- Visual only;
- Visual + Skeleton calibrated probability mean;
- Visual + IMU calibrated probability mean;
- Visual + Skeleton + IMU calibrated probability mean;
- confidence-weighted mean with weights learned only from training OOF;
- a frozen linear stacker trained on OOF logits.

All probability calibration is fitted on source folds only. These controls determine whether the other modalities contain usable complementary evidence before attributing gains to MoBind fusion.

## 13. Stage S4: Teammate MoBind Fusion Training

S4 adapts the teammate P86 fusion structure to the fixed VideoMAEv2 visual embedding dimension without changing the visual teacher.

The frozen recipe is:

- separate Skeleton and IMU encoders;
- additive motion residual into visual representation;
- stage A: 4 epochs;
- stage B: 20 epochs;
- pretrained motion encoders frozen during fusion training;
- fusion learning rate: `4e-4`;
- encoder learning rate: `1e-4` when an encoder is permitted to update in its own stage;
- visual teacher remains frozen;
- weight decay: `0.05`;
- class-weight power: `0.35`;
- label smoothing: `0.08`;
- distillation temperature: `2.0`;
- distillation weight: `1.0`;
- relation weight: `0.1`;
- motion auxiliary weight: `0.35`;
- selective anchor weight: `0.3`;
- visual corruption probability: `0.75`;
- visual feature dropout: `0.3`;
- visual view dropout: `0.4`;
- seed: `20260811`.

S4 generates strict training OOF predictions, then refits the frozen recipe on all 12 users and predicts user6/user7.

## 14. Stage S5: Repeat and Session Processing Without Voting

S5 applies the teammate's cross-sample processing to the single S4 fused probability only.

- build label-free recording metadata from timestamps, duration, modality availability, and device signatures;
- learn class transition counts from chronological sessions of training users;
- use a 30-second maximum session gap and Laplace-smoothed transitions;
- construct repeat candidates using time, duration, availability, and S4 probability similarity;
- propagate only when at least two peers agree and confidence passes a frozen training-only threshold;
- compare transition-only, repeat-only, and combined variants;
- do not import P89/P128/P137/P165/P173/P307/P309 expert banks or their probabilities.

All S5 strengths and thresholds are selected on training OOF only. User identity is not a target feature.

## 15. Stage S6: Unlabeled Target Adaptation Without P310 Targets

S6 uses all 388 user6/user7 inputs without labels and adapts the S4 compact multimodal model.

Pseudo targets come only from the frozen best training-selected S5 emission. P310 and all 30-teacher structured targets are forbidden.

Adaptation follows the teammate mechanism where compatible:

- freeze the visual VideoMAEv2 teacher;
- update fusion head plus compact motion encoder;
- confidence threshold: `0.90`;
- class-balanced pseudo-label cap: median non-empty predicted class count;
- strong/weak augmentation consistency;
- KL anchor to the pre-adaptation S4 model;
- epochs: 12 for direct comparison with the original P87-S adaptation, plus a predeclared 40-epoch endpoint to measure the later P310 recipe;
- learning rates: fusion `1e-4`, motion encoder `1e-4`, visual teacher `0`;
- minimum learning rate: `5e-6`;
- weight decay: `0.02`;
- seeds: `20260811`, `20260826`;
- no early stopping or target-label checkpoint selection.

Both 12-epoch and 40-epoch outputs are frozen before reveal. This stage is explicitly reported as transductive.

## 16. Ablation Matrix

The evaluator reports the following fixed comparisons:

| ID | Visual | Skeleton | IMU | MoBind | Session/repeat | Target adaptation |
|---|---|---|---|---|---|---|
| S0 | yes | no | no | no | no | no |
| S1a | yes | yes | no | simple mean | no | no |
| S2a | yes | no | yes | simple mean | no | no |
| S3a | yes | yes | yes | simple mean | no | no |
| S3b | yes | yes | yes | linear OOF stacker | no | no |
| S4 | yes | yes | yes | teammate-style | no | no |
| S5 | yes | yes | yes | teammate-style | yes | no |
| S6-12 | yes | yes | yes | teammate-style | yes | 12 epochs |
| S6-40 | yes | yes | yes | teammate-style | yes | 40 epochs |

This matrix attributes gains without introducing multiple independent teachers per modality.

## 17. Metrics

Every stage reports:

- correct/388 and accuracy;
- correct/385 and accuracy on the historical visual subset;
- 40-class macro-F1;
- user6 and user7 accuracy;
- per-class recall;
- rescue, harm, net, and disagreement versus S0 and versus the immediately previous stage;
- missing-modality subgroup metrics;
- prediction entropy and calibration error;
- OOF metrics for the 12-user training population;
- Student/teacher agreement for S6.

Oracle selection is permitted only as a clearly labeled upper-bound diagnostic and never as an inference result.

## 18. Artifacts

Required outputs under `outputs/visual_motion_no_vote_ablation/`:

- `protocol.json`;
- `fold_contract.json`;
- `visual_baseline/`;
- `skeleton_teacher/`;
- `imu_teacher/`;
- `simple_fusion/`;
- `mobind_fusion/`;
- `session_repeat/`;
- `adaptation/seed_<seed>/`;
- `candidate_registry.json`;
- `predictions_unlabeled.npz` without labels;
- `generation_complete.json` with artifact hashes;
- `evaluation.json`, `evaluation.md`, per-user CSV, and per-class CSV;
- immutable stdout/stderr logs.

Completed fold artifacts are resumable only when protocol, source, checkpoint, input, and configuration hashes match exactly.

## 19. Tests and Guards

The implementation must test:

- exact 385-row S0 reproduction;
- complete, unique, ordered ownership of all 388 target rows;
- no target label reaches generation code or artifacts;
- source/held user disjointness for every fold;
- fold-scoped normalization and Random Forest fitting;
- all OOF rows predicted exactly once;
- finite, normalized 40-class probabilities;
- modality availability masks prevent missing tensors from contributing;
- Skeleton and IMU-only branches can be disabled without changing S0;
- simple fusion uses OOF predictions rather than in-sample training predictions;
- MoBind updates only permitted parameters;
- S5 is invariant to input row ordering after restoring sample IDs;
- S5 disabled gates reproduce S4 exactly;
- S6 cannot load P310 or any expert-bank artifact;
- S6 cannot access target labels or select checkpoints from target metrics;
- generated predictions and registry hashes are frozen before evaluation.

## 20. Failure Handling

- Stop on any source, checkpoint, split, cache, raw-input, or configuration hash mismatch.
- Stop if S0 does not reproduce the original 385-row prediction within tolerance.
- Stop if a training fold lacks a class required by a loss without a declared missing-class policy.
- Stop if any stage drops or duplicates target rows.
- Stop if target labels, correctness arrays, or historical label-derived target reports enter generation.
- Preserve completed immutable fold outputs on interruption; never silently overwrite them.
- Do not substitute MotionBERT, HD-GCN, C1-TCN, CTR-GCN, Thermal, Radar, or a teacher-bank probability when a required P86 component is unavailable.

## 21. Interpretation Boundary

This is development evidence on user6/user7. It does not estimate the official anonymous-test score independently because user6/user7 has already informed prior project decisions.

The experiment measures the teammate's compact Visual+Skeleton+IMU processing path without the 30-teacher bank. It does not measure the contribution of Thermal, Radar, large external Skeleton teachers, or multi-teacher voting. Those require separate later specifications if this no-vote pipeline shows useful gains.
