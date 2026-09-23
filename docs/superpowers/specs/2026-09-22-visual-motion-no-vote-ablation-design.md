# Fixed-Split Teammate-Style Single-Teacher Pipeline Design

## 1. Objective

Rebuild one complete teammate-style teacher-to-Student pipeline from raw competition training data, using the owner's fixed user split and excluding the historical 30-teacher bank.

The experiment measures the accuracy contribution of the teammate's non-voting operations:

- automatic pose/ROI preprocessing;
- one large pretrained visual teacher and its trained classification head;
- visual Student distillation;
- compact Skeleton and IMU branches;
- MoBind fusion and visual-corruption training;
- repeat/session processing;
- unlabeled target-domain adaptation.

No historical teammate checkpoint, generated feature cache, OOF tensor, test prediction, P310 target, or expert-bank probability is used as a training input.

“Train from scratch” means regenerate all task-specific caches and train all task-specific heads, students, motion branches, and fusion modules. Public pretrained visual/pose initialization is retained because that is part of the teammate method; the large VideoMAE backbone is not randomly initialized.

## 2. Claim Boundary

This is a fixed-split development ablation on user6/user7. It is not a three-fold OOF experiment and does not estimate the anonymous official-test score.

The primary question is:

> On the fixed user6/user7 development test, how much accuracy is added by each teammate-style processing stage when the 30-teacher voting system is absent?

A negative result is valid. Reaching 0.91 is not a completion criterion.

## 3. User Split

### Internal fit users

`user1, user2, user3, user9, user16, user18, user19, user20, user21, user22`

This 10-user fit population contains all 40 classes, including class 25 from user1.

### Internal development users

`user5, user8`

This population is used only for selecting epoch budgets, visual-head configuration, calibration, fusion thresholds, and repeat/session strengths. It naturally contains 33 classes; all models retain 40 outputs and all metrics use fixed IDs 0–39.

### Final refit users

All 12 users from the fit and internal development populations.

After a stage's recipe is frozen on the fixed fit/development split, it is retrained once on all 12 users for the selected fixed epoch budget. There is no early stopping during final refit.

### Final development test

`user6, user7`, 388 rows.

Their labels are unavailable to every generation/training process and are read once by a separate evaluator after all candidate predictions are frozen and hashed.

## 4. Consequence of Using a Fixed Split

The earlier class-25 OOF blocker no longer applies because user1 always remains in the fit side. The implementation must assert all 40 classes are present before every fit and final-refit run.

No stage may use the terms “strict OOF,” “three-fold OOF,” or “outer-fold estimate.” Training reports use:

- internal-fit metrics;
- internal-development metrics;
- full-12 refit diagnostics without accuracy-based selection;
- final user6/user7 descriptive metrics after reveal.

No learned meta-stacker is trained from in-sample component predictions and reported as OOF.

## 5. Label Isolation

All user6/user7 inputs are rebuilt from raw modalities. Historical target caches are forbidden because several contain labels.

The target-generation process may read only:

- sample IDs;
- modality paths and availability;
- raw IR, Depth, Skeleton, and IMU data;
- timestamps, duration, and device metadata;
- generated logits, probabilities, embeddings, and quality fields.

It may not read:

- user6/user7 labels;
- correctness or confusion arrays;
- historical user6/user7 predictions or reports;
- Kaggle predictions or scores;
- P310 targets;
- any expert-bank artifact.

All target candidate predictions are written to `predictions_unlabeled.npz`. A generation-complete manifest hashes every candidate before the evaluator receives the label path. Deleting or randomly permuting the inaccessible label file must not change generated output bytes.

## 6. Included and Excluded Modalities

### Included deployment modalities

- IR/Depth visual input used to train one visual teacher and one compact visual Student;
- Skeleton;
- IMU.

### Excluded first-round teacher-bank modalities and models

- Thermal and Radar;
- InternVideo2, V-JEPA, LaViLa, DINOv2;
- MotionBERT and HD-GCN;
- P128/P158/P231/P238/P253/P306;
- all P89/P137/P165/P173/P307/P309 expert banks and routers;
- P310 structured targets.

Thermal and Radar are excluded because the deployed compact P87-S Student does not consume them. Including them would reintroduce the historical teacher-bank factor.

## 7. Stage A0: Raw Manifest and Automatic Geometry

Build a new manifest from raw data for the 10-user fit, two-user internal development, 12-user refit, and unlabeled user6/user7 target partitions.

Recreate teammate preprocessing:

- YOLO11n-pose initialization with recorded SHA256;
- two-pass IR pose detection and temporal tracking;
- exact synchronized transfer to Depth;
- scene, person, hand/forearm, and hand-workspace geometry;
- timestamp-preserving Skeleton alignment;
- explicit missing-modality masks and quality summaries.

Outputs are split-neutral raw/geometry caches. They must not contain labels for user6/user7.

## 8. Stage A1: Single Large Visual Teacher

Use exactly one large visual teacher family:

- `MCG-NJU/videomae-large-finetuned-kinetics`;
- revision `0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4`;
- frozen pretrained backbone;
- early and late temporal windows;
- scene, person, and workspace views;
- 16 frames per window;
- 1024-dimensional features and 400 Kinetics logits.

Generate features independently for fit, internal development, full-12 refit, and unlabeled target partitions. The backbone remains frozen, matching the teammate P85 teacher extraction.

Train the P85 40-class Ridge head family on internal-fit users. Select the feature family, class-weight power, and alpha on user5/user8 using the teammate candidate grid:

- feature families: early, late, window mean, early+late, temporal delta, Kinetics logits;
- class-weight powers: `0.0, 0.5, 0.75`;
- Ridge alpha: `300, 1000, 3000, 10000`.

Tie-breaking order is accuracy, macro-F1, worst-user accuracy, then lower-dimensional feature family, then larger alpha.

Refit the selected head on all 12 users and produce target probabilities. This is the single privileged visual teacher for every later distillation stage.

## 9. Stage A2: Compact MC3 Visual Student

Train the teammate P86 MC3-18 temporal visual Student from newly generated raw pixels and A1 teacher targets.

The fixed recipe is transferred from the teammate visual-Student path:

- torchvision MC3-18 Kinetics-400 initialization;
- early and late windows;
- scene, person, workspace views;
- 16 frames at 160×160;
- subject-robust synchronized augmentation;
- classifier cross-entropy;
- teacher probability distillation, temperature `2.0`, weight `1.0`;
- relation loss weight `0.2`;
- feature alignment weight `0.5`;
- label smoothing `0.1`;
- head learning rate `2e-4`;
- backbone learning rate `1e-5`;
- weight decay `0.08`;
- batch size `4`, gradient accumulation `4`;
- seed `20260811`.

Select the epoch budget on user5/user8 after fitting on the ten internal-fit users. Refit for the frozen budget on all 12 users without early stopping.

The resulting Student exposes the original P86-compatible seam:

`[B, 2 windows, 3 views, T, 512]`.

No dimension adapter or fabricated repeated temporal token is used.

## 10. Stage A3: Skeleton/IMU Motion Cache

Rebuild the teammate P31/P86 motion inputs from raw Skeleton and IMU:

- Skeleton body-relative H36M-17 coordinates;
- exact 16-bin alignment to visual time;
- part-aware joint groups and local motion features;
- five IMU device roles;
- raw accelerometer and gyroscope channels;
- device-relative compensated channels;
- relative quaternion features;
- exact missing-role masks;
- normalization statistics fitted on internal-fit users only during selection and on all 12 users during final refit.

The internal-development and target datasets reuse fitted statistics but never contribute to them.

## 11. Stage A4: Statistical IMU Teacher

Reproduce one teammate-selected IMU teacher:

`stat_random_forest_device_dropout_aligned`

Train on internal-fit users, select only its fixed epoch-independent preprocessing and Random Forest hyperparameters on user5/user8, then refit on all 12 users. The teacher produces 40-class logits/probabilities and an availability mask.

No additional deep IMU teacher or ensemble is allowed.

## 12. Stage A5: P86 MoBind Skeleton/IMU Pretraining

Train the compact P86 MoBind motion model using the newly generated A1 teacher and A4 IMU teacher.

Required losses and components:

- Skeleton supervised cross-entropy;
- IMU supervised cross-entropy;
- Skeleton and IMU reconstruction;
- visual-teacher probability distillation;
- visual-teacher feature alignment;
- local part/time contrastive alignment;
- Skeleton/IMU global semantic alignment;
- IMU teacher distillation weight `1.0`;
- width `96`, alignment width `64`;
- 24 epochs;
- missing-modality and role masks.

Select the fixed training budget and any non-transferred coefficient only on user5/user8. Refit on all 12 users.

The ablation records Skeleton-only, IMU-only, and combined motion probabilities before visual fusion.

## 13. Stage A6: Simple Fusion Controls

Before training MoBind fusion, produce fixed controls:

- MC3 visual Student only;
- equal calibrated mean of visual and Skeleton probabilities;
- equal calibrated mean of visual and IMU probabilities;
- equal calibrated mean of visual, Skeleton, and IMU probabilities.

Temperatures are fitted on internal-fit users and selected on user5/user8. No target label or historical target metric is used.

These controls establish whether motion modalities contain useful evidence before adding fusion capacity.

## 14. Stage A7: Teammate MoBind Fusion

Use the original P86/P87 compatible MC3 seam and compact motion encoders. No visual interface adaptation is required.

Frozen transferred recipe:

- separate Skeleton and IMU encoders;
- additive motion residual;
- stage A `4` epochs;
- stage B `20` epochs;
- pretrained motion encoders frozen during fusion training;
- fusion learning rate `4e-4`;
- visual learning rate `2e-5` for the compact MC3 Student where permitted;
- weight decay `0.05`;
- class-weight power `0.35`;
- label smoothing `0.08`;
- distillation temperature `2.0`;
- distillation weight `1.0`;
- relation weight `0.1`;
- motion auxiliary weight `0.35`;
- selective anchor weight `0.3`;
- visual corruption probability `0.75`;
- visual feature dropout `0.3`;
- visual view dropout `0.4`;
- additive global fusion;
- seed `20260811`.

Train on internal-fit users and select only the transferred-stage endpoint on user5/user8. Refit the frozen recipe on all 12 users.

Matched controls use the same architecture, optimizer, seed, and losses with:

- both motion modalities masked;
- Skeleton shuffled across samples within a batch;
- IMU shuffled across samples within a batch;
- inference-time Skeleton zeroed;
- inference-time IMU zeroed.

These controls distinguish motion content from added capacity and regularization.

## 15. Stage A8: Repeat and Session Processing Without Voting

Apply cross-sample processing only to the single A7 fused probability:

- label-free timestamp/duration/device metadata;
- 30-second maximum session gap;
- class-transition counts learned from internal-fit users;
- repeat groups based on time, duration, availability, and A7 probability similarity;
- at least two agreeing peers;
- confidence-gated fallback to the unchanged A7 emission.

Compare transition-only, repeat-only, and combined candidates on user5/user8. Freeze one primary operator by accuracy, macro-F1, then fewer changed rows. Refit transition counts/metadata statistics on all 12 users without changing thresholds.

A8 is transductive because target rows can influence other rows in the same target batch.

## 16. Stage A9: Unlabeled Target Adaptation Without P310

Generate pseudo targets only from the frozen A8 output. P310 and all expert-bank artifacts are forbidden.

Primary adaptation reproduces the teammate 12-epoch mechanism:

- start from the all-12-user A7 model;
- update fusion heads and the compact motion encoder;
- keep the large A1 visual teacher frozen and absent from target inference;
- 12 epochs;
- fixed target probabilities for all available rows;
- no target-label filtering or checkpoint selection;
- no early stopping;
- seed `20260812`;
- use the original P87 optimizer/scope where compatible with the rebuilt model.

The later 40-epoch configuration is exploratory:

- 40 epochs;
- fusion learning rate `1e-4`;
- motion encoder learning rate `1e-4`;
- minimum learning rate `5e-6`;
- weight decay `0.02`;
- seed `20260826`.

Both outputs are generated and frozen before reveal. A9 is transductive and batch-specific.

## 17. Fixed Ablation Matrix

| ID | Output |
|---|---|
| A1 | single VideoMAE-Large teacher |
| A2 | compact MC3 visual Student |
| A5-S | Skeleton branch only |
| A5-I | IMU branch only |
| A6-VS | visual + Skeleton simple fusion |
| A6-VI | visual + IMU simple fusion |
| A6-VSI | visual + Skeleton + IMU simple fusion |
| A7 | teammate MoBind fusion |
| A7-mask | matched visual-only capacity control |
| A7-shuffle-S | shuffled Skeleton control |
| A7-shuffle-I | shuffled IMU control |
| A8 | A7 + fixed repeat/session operator |
| A9-12 | A8 pseudo targets + 12-epoch adaptation, primary |
| A9-40 | A8 pseudo targets + 40-epoch adaptation, exploratory |

Primary contrast: `A9-12 − A1` on all 388 rows.

Stage contributions are also reported as:

- distillation/compression: `A2 − A1`;
- Skeleton evidence: `A6-VS − A2`;
- IMU evidence: `A6-VI − A2`;
- three-modality simple fusion: `A6-VSI − A2`;
- learned fusion bundle: `A7 − A6-VSI`;
- session/repeat processing: `A8 − A7`;
- target adaptation: `A9-12 − A8`.

## 18. Metrics and Reveal

Before reveal, write every candidate probability and artifact hash to an immutable generation manifest.

The evaluator reports:

- correct/388 and accuracy;
- metrics on the 385 rows with visual input;
- fixed-40-class macro-F1;
- user6 and user7 separately;
- per-class recall;
- rescue, harm, net, and disagreement against A1 and previous nested stage;
- missing-modality subgroups;
- calibration NLL and Brier score;
- A9 Student/A8 pseudo-teacher agreement.

After reveal, no model, threshold, stage, seed, epoch, or fusion rule may change.

## 19. Artifact Contract

All new artifacts live under:

`outputs/teammate_single_teacher_fixed_split/`

Required directories:

- `protocol/`;
- `pose_roi/`;
- `visual_teacher/`;
- `visual_student/`;
- `motion_cache/`;
- `imu_rf_teacher/`;
- `mobind_pretrain/`;
- `simple_fusion/`;
- `mobind_fusion/`;
- `session_repeat/`;
- `adaptation_12/`;
- `adaptation_40/`;
- `evaluation/`.

Every model/cache includes source hashes, input hashes, split identity, user ownership, configuration hash, row count, availability policy, software versions, and parent artifact hashes.

## 20. Guards

The implementation stops when:

- any source, pretrained weight, manifest, split, raw input, or parent artifact hash changes;
- internal fit lacks any of the 40 classes;
- user6/user7 labels or historical label-derived reports enter generation;
- any target cache contains labels or correctness fields;
- any row is duplicated, lost, reordered without a recorded mapping, or produces non-finite probabilities;
- any training statistic includes internal-development or target rows outside its declared stage;
- A1 feature dimensions differ from the recorded P85 contract;
- A2 fails to expose the exact P86 visual sequence contract;
- A9 attempts to load P310 or an expert-bank artifact.

Interrupted stages resume only when protocol and artifact hashes match. Existing artifacts are never silently overwritten.

## 21. Compute and Storage Preconditions

The local RTX 5060 Laptop GPU with 8 GiB VRAM is sufficient using batch-size-one teacher extraction, cached features, gradient accumulation, and the teammate's sequential multiview strategy.

Expected wall-clock cost after implementation debugging is approximately 8–16 GPU-hours for teacher extraction, internal fit/development runs, 12-user refits, MoBind fusion, and both adaptation endpoints.

At least 20 GiB free disk space is required before execution. Current free space is below this threshold, so execution must not start until output storage is expanded or redirected to a location with sufficient capacity.

## 22. Interpretation Boundary

This experiment eliminates new user6/user7 label access and removes the 30-teacher voting system. It measures one rebuilt teammate-style privileged teacher → compact multimodal Student path under the owner's fixed split.

It does not prove independent unseen-user generalization because architecture and recipe choices are informed by prior project history. Confirmation requires new subjects or the anonymous official test.
