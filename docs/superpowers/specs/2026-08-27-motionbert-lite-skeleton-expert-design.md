# MotionBERT-Lite Skeleton Expert Qualification Design

Approved direction: 2026-08-27 (Asia/Shanghai)

## 1. Objective

Qualify exactly one pretrained large Skeleton expert, MotionBERT-Lite, as a
future specialist for the IR-anchored multimodal teacher. This experiment asks
whether pretrained motion representations add enough unique correct
user6/user7 predictions to justify a later bounded residual fusion design.

This experiment does not implement multimodal fusion or student distillation.
It does not run CTR-GCN, PoseC3D, multiple seeds, or grouped CV.

## 2. Prior Repository Evidence

The repository contains no true CTR-GCN experiment. Branch `skeleton_raw_data`
commit `72c0c68` implemented a 60,920-parameter fixed-adjacency lightweight
ST-GCN. Its frozen report at
`reports/skeleton_lightweight_stgcn_strict_oof/skeleton_lightweight_stgcn_strict_oof_report.md`
records strict OOF Accuracy `0.235797` versus `0.409227` for C1-TCN. It had no
adaptive adjacency, topology refinement, attention, extra streams, or
augmentation and underfit every fold.

The strongest frozen Skeleton evidence remains the E1 three-seed C1-TCN
ensemble at strict OOF Accuracy `0.423751` and Macro-F1 `0.315606`. T=96 and
joint/bone TCN variants did not satisfy their stability/replacement gates.

## 3. Fixed Population and Leakage Boundary

- Training population: canonical 2,039 train12 trials.
- Development validation: canonical 388 user6/user7 trials.
- Skeleton-supported rows: 1,956 train and 385 validation.
- Unsupported rows remain in canonical metrics and use a train12-only class
  prior with `skeleton_available=False`.
- Class order is exactly `0..39`; both canonical populations contain all 40
  classes.
- user6/user7 may enter fixed development metrics and B1/B2 promotion gates.
  They may not enter gradients, projection fitting, normalization, sampling,
  class-prior fitting, or checkpoint initialization.
- Grouped CV is forbidden unless separately authorized under the existing
  complete-class rule. It is not part of P6-B.

## 4. External Source and Weight Contract

- Upstream repository: `https://github.com/Walter0807/MotionBERT`.
- Frozen upstream commit: `705d3a95354db8bdb696b3492e47a3b5537174ff`.
- License: Apache-2.0; vendored source must retain the license and attribution.
- Vendored implementation is limited to the DSTformer and DropPath dependencies
  needed by MotionBERT-Lite. No training framework is vendored.
- Weight repository: `walterzhu/MotionBERT` revision
  `370a9196aa3c89198b134c82476143b01c0fb32c`.
- Weight path: `checkpoint/pretrain/MB_lite/latest_epoch.bin`.
- Expected bytes: `64,099,897`.
- Expected SHA-256:
  `6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461`.

The SHA is the Hugging Face LFS object OID. The Xet transport hash
`d8fea532c99311114000f08ce0fa037170cffa9f19310f329c9b2826daba28be`
is recorded separately and must not be substituted for the file SHA-256.

Any source, license, weight size, or SHA mismatch blocks smoke and training.

## 5. Input Contract

The authoritative Skeleton identity and projection artifact is:

```text
outputs/midfusion_skeleton_clean_views/selected_final/clean_view.csv
```

Its projection was fit on train12 only and applied to train12 plus user6/user7.
P6-B never regenerates it from validation labels.

For each supported trial:

1. retain only rows with `use_for_frame_training=True` and the frozen selected
   candidate identity;
2. choose the longest retained contiguous segment by frame count, breaking ties
   by the smallest retained-segment index;
3. load H36M-17 coordinates and apply the accepted root/scale normalization;
4. apply the train12-fit 3D-to-2D projection from `selected_final` provenance;
5. resample only inside that retained segment to exactly `T=96`;
6. form MotionBERT input `[96,17,3]` as normalized projected `x`, normalized
   projected `y`, and confidence/validity;
7. never interpolate across retained-segment gaps;
8. record retained frames, discarded-segment frames, projection quality, and
   availability.

The longest-segment policy is label-free. It avoids cross-gap attention while
keeping the first experiment to one MotionBERT sequence per trial. No `z` or
velocity side adapter is introduced in P6-B.

## 6. Model Contract

MotionBERT-Lite uses the frozen upstream architecture:

```text
dim_in=3, dim_feat=256, dim_rep=512, depth=5,
num_heads=8, num_joints=17, maxlen=243, att_fuse=True
```

P6-B uses `T=96`, one person, and the pretrained motion representation. The
classification head is deliberately small:

```text
masked mean over time and joints -> LayerNorm(512) -> Dropout(0.5) -> Linear(512,40)
```

The backbone returns both per-joint temporal representation and pooled 512D
embedding so a future residual fusion plan can cache segment/motion evidence.

## 7. Stage B0: Resource and Compatibility Smoke

B0 uses two supported train12 rows and two supported user6/user7 rows. It must:

- verify source, license, checkpoint bytes, and SHA;
- instantiate MotionBERT-Lite under PyTorch 2.7;
- load all shape-compatible pretrained backbone tensors and report exact
  missing/unexpected keys;
- require at least 99% of backbone parameter elements to load from the frozen
  checkpoint;
- prove pretrained and deterministic random-initialized embeddings differ;
- run frozen-backbone head forward/backward with finite loss, logits, and head
  gradients;
- run one optimizer step and prove only the head changes;
- use CUDA AMP and record peak memory and wall time;
- refuse an existing smoke output directory.

Any failure blocks B1.

## 8. Stage B1: Frozen-Backbone Qualification

- Cache one 512D pretrained embedding for every canonical train and validation
  row; unsupported rows receive a zero embedding and `available=False`.
- Fit the class prior from all train12 labels only.
- Freeze the MotionBERT-Lite backbone completely.
- Train only the classification head for exactly 20 epochs.
- Use uniform trial sampling over Skeleton-supported train rows, batch size 64
  on cached embeddings, AdamW, learning rate `1e-3`, weight decay `1e-2`, and
  seed `20260715`.
- Do not inspect validation metrics per epoch. Evaluate train and validation
  once after epoch 20.

B1 may enter B2 only if all gates hold on canonical user6/user7:

- Accuracy at least `0.40`;
- fixed-40 Macro-F1 at least `0.30`;
- at least 15 unique correct rows relative to the frozen visual-only archive;
- visual-only plus MotionBERT oracle Accuracy at least `0.75`;
- at least one unique rescue for both user6 and user7;
- finite predictions and all 40 labels retained in the archive.

## 9. Stage B2: Partial Fine-Tuning

B2 is conditional and starts automatically only after a recorded B1 pass.

- Initialize from the B1 head and the same pretrained backbone.
- Freeze input embedding and the first three DSTformer block pairs.
- Unfreeze only the final two spatial/temporal block pairs, final norm,
  pre-logits layer, and classification head.
- Train exactly 10 epochs on raw T=96 supported train rows.
- Use batch size 4, gradient accumulation 8, CUDA AMP, AdamW,
  backbone learning rate `2e-5`, head learning rate `4e-4`, weight decay
  `1e-2`, gradient clip `1.0`, and seed `20260715`.
- Validation is evaluated once after epoch 10; no early stopping or epoch
  selection is allowed.

B2 qualifies for a future fusion design only if all gates hold:

- canonical Accuracy at least `0.48`;
- fixed-40 Macro-F1 at least `0.38`;
- at least 30 unique correct rows relative to visual-only;
- visual-only plus MotionBERT oracle Accuracy at least `0.82`;
- both user6 and user7 contribute at least one unique rescue;
- motion-family recall is reported;
- checkpoint and 512D embeddings reload exactly with recorded hashes.

Failure stops P6-B without fusion or distillation.

## 10. Evidence and Outputs

All outputs use a new root and refuse silent overwrite:

```text
outputs/motionbert_lite_skeleton_expert_p6b/
reports/motionbert_lite_skeleton_expert_p6b_smoke.json
reports/motionbert_lite_skeleton_expert_p6b_b1.json
reports/motionbert_lite_skeleton_expert_p6b_b2.json   # only if B1 passes
reports/motionbert_lite_skeleton_expert_p6b.md
```

Reports include canonical/supported metrics, per-user/per-class metrics,
visual paired rescue/harm, oracle, zero-recall classes, train-validation gap,
resource use, source/config/data/checkpoint/prediction hashes, and the exact
promotion decision. user6/user7 is labeled development validation, not an
independent final test.

## 11. Non-Goals

- No P6-A same-checkpoint counterfactual.
- No CTR-GCN, PoseC3D, SkateFormer, full MotionBERT, ensemble, or multiple seed.
- No three-fold or OOF execution.
- No Skeleton/visual fusion in this experiment.
- No IMU, Thermal, Radar, or competition-test access.
- No student distillation unless a later fusion teacher separately passes its
  teacher gate.
