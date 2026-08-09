# Plan: IR-primary + ordinal Depth full-sequence experiment

This document is the versioned experiment plan and execution record. Generated datasets, checkpoints, caches, and preview images are intentionally excluded from Git.

## Objective

Implement and run a controlled input-representation pilot for the latest visual design:

```text
All original paired frames (no 24/96 preprocessing sampling)

IR: context + left directional + right directional + adaptive relation
Depth_Color: inverse JET at native resolution -> ordinal depth + pixel-valid mask
Depth: context + adaptive relation

independent spatial encoders
-> shallow gated IR/Depth fusion
-> full-sequence multi-scale TCN
-> 40-class main head + small-action gate head
-> NO conditional fine-action head
```

Compare exactly three Depth loader representations (`raw`, `relative`, `raw+relative`) under a fixed pilot protocol. Select one, then run only one formal full-sequence experiment. Do not start unrelated experiments.

## Execution status (updated 2026-08-09)

- [x] **Stage 1 complete: freeze the route and preserve the current assets.** The fixed `imgsz=640` pose cache, full-frame ROI export, original frame ordering, approved masks/reliability rules, fixed small-action 24 set, and experiment boundaries were inventoried and frozen. No `imgsz=1280` comparison was introduced.
- [x] **Stage 2 complete: temporal/identity audit and hard input acceptance gate.** The fixed 640 export passed every predeclared gate. The audit scanned all `84,906` source Depth_Color frames, evaluated effective content masks, three-local-view redundancy, person-normalized temporal jumps, transient left/right identity swaps, and long invalid runs, and generated short-sequence P1 review artifacts.
- [x] **Stage 3 complete: inverse-JET ordinal codec and mask-aware resize.** Added an exact OpenCV JET-to-ordinal decoder, explicit black/unknown-color pixel masks, strict unknown-color rejection, and mask-aware value/mask resizing. Eight focused unit tests pass. A real-data verification checked one source frame from every one of the `2,910` train/val trials: `723,756,503` valid pixels round-tripped with zero mismatches, `170,195,497` black pixels remained invalid, and resizing to `256x256` produced zero nonzero pixels behind the output invalid mask.
- [x] **Stage 4 complete: Depth-only ordinal/mask exporter and combined manifest implementation.** The exporter writes only two ordinal Depth value views and their separate binary masks, while referencing the existing four IR files without copying them. The combined manifest preserves source paths, frame order, timestamps/frame ids/deltas, pose/content/effective masks, pixel coverage, and deterministic reliability. Eleven focused tests pass, and a read-only join over all `84,906` train/val frames found `0` missing IR references or pairing/key mismatches.
- [x] **Stage 5 complete: real-data export smoke test and pixel audit.** A fixed 82-trial subset covers all 40 classes, one train and one validation trial per class, plus the two known Stage 2 Depth content-invalid trials. It exported all `3,562` selected original frames into `14,248` Depth PNGs (two ordinal views plus two independent masks), while reusing rather than copying the four IR views. Every stored value/mask was independently recomputed from native Depth_Color and ROI coordinates: value mismatches `0`, mask mismatches `0`, and nonzero values behind invalid masks `0`. Twelve neighboring-frame contact sheets were generated and inspected.
- [x] **Stage 6 complete: full Depth-only ordinal export and integrity audit.** All `2,910` train/val trials and `84,906` original paired frames were exported without temporal sampling. The full asset contains `169,812` ordinal value PNGs and `169,812` independent pixel masks (`339,624` data PNGs, about `3.507 GiB`) while referencing the existing four IR views. A full native-source recomputation found `0` ordinal pixel mismatches, `0` mask mismatches, and `0` nonzero values behind invalid masks. All manifest, frame-order, timestamp/delta, shape/dtype, binary-mask, IR-reference, and retained-content checks passed.
- [x] **Stage 7 complete: variable-length loader/model/trainer refactor plus expert-interface contract.** The obsolete Stage A/B/cache training path has been removed from the active trainer. Complete-trial loading, frame-budget batching, in-memory temporal padding, padding-safe multi-scale TCN processing, end-to-end single-stage training, and the reserved visual expert contract are implemented. Real-manifest read-only acceptance, `42` repository tests, implementation reporting, and independent review all passed.

Completed Stage 7 implementation:

- `src/data/ir_primary_full_sequence_dataset.py`: complete variable-length trials, fixed `[raw, relative, pixel_valid]` Depth interface, clip-level relative statistics/fallback, deterministic quality schema, collate-time padding, and `FrameBudgetBatchSampler`.
- `src/models/expert_contract.py`: `ExpertOutput`, `ExpertBatchResult`, strict `sample_id` alignment, class-map checks, and the predeclared calibrated probability-mixture primitive.
- `src/models/full_sequence_multiscale_tcn.py`: temporal masks are reapplied inside residual blocks and normalization is per time step so padded positions do not affect valid-frame statistics.
- `src/models/ir_primary_depth_residual_tcn.py`: valid-view-only spatial encoding, mask-aware Depth pooling, six-view reliability injection, full-sequence TCN, 40-class main output, and visual-only small-action gate.
- `src/train_ir_primary_depth_residual_fullseq.py`: one end-to-end variable-length training stage; no spatial cache, Stage A, Stage B, 24-frame sampling, or 96-frame sampling.
- New tests: `tests/test_ir_primary_variable_sequence_dataset.py`, `tests/test_expert_contract.py`, and `tests/test_variable_sequence_trainer_contract.py`, plus the updated model tests.

Stage 7 boundaries remain intact: Skeleton/IMU/Radar were not read or connected; competition test was not read; no training ran; Stage 8 has not started.

Stage 7 outputs:

- `reports/depth_ordinal_stage7_variable_sequence_implementation.md`
- `reports/depth_ordinal_stage7_independent_review.md`

Stage 7 result: **passed with no P0/P1 findings**. The real manifest contains `84,906` frames and `2,320/590` train/validation trials with disjoint 14/4 users, 40 classes, and complete sequence lengths from 1 to 236 frames. Frame-budget batching had zero omissions, duplicates, or batches above the 256 padded-frame budget. The independent follow-up also confirmed checkpoint-on/off equality for all 43 BatchNorm states and both spatial-encoder gradients.

Stage 6 outputs:

- Full asset root: `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256`
- `reports/depth_ordinal_stage6_audit.json`
- `reports/depth_ordinal_stage6_audit.md`
- `reports/depth_ordinal_stage6_independent_review.md`

Stage 6 result: **passed**. Train/validation frame counts are `67,216` / `17,690`. Mean pixel coverage is `0.8035` for Depth context and `0.8182` for Depth relation. The 3,078 raw low-information flags comprise 3,076 expected pose-invalid placeholders plus the two pose-valid Stage 2 findings, both retained. Competition test read `false`; training run `false`.

Independent Stage 6 review: **passed with no P0/P1 findings**. A new reviewer independently recomputed both ordinal views and masks for all `84,906` frames / `339,624` data PNGs, again finding zero ordinal mismatches, zero mask mismatches, and zero nonzero values behind invalid masks. Stage 7 was not started.

Stage 5 outputs:

- Smoke asset root: `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256_smoke_stage5`
- `reports/depth_ordinal_stage5_selection.csv`
- `reports/depth_ordinal_stage5_audit.json`
- `reports/depth_ordinal_stage5_audit.md`
- `scripts/audit_depth_ordinal_stage5_smoke.py`

Stage 5 result: **passed**. The subset contains `1,877` train frames and `1,685` validation frames. Mean valid-pixel coverage is `0.8132` for Depth context and `0.8065` for Depth relation. Of 121 raw scalar low-information flags, 119 are expected pose-invalid zero placeholders; the only two pose-valid content-invalid views are the two Stage 2 findings, both retained. Competition test read `false`; training run `false`; full export run `false`.

Stage 3 outputs:

- `src/data/ordinal_depth.py`
- `tests/test_ordinal_depth.py`
- `scripts/verify_depth_ordinal_stage3.py`
- `reports/depth_ordinal_stage3_verification.json`
- `reports/depth_ordinal_stage3_verification.md`
- `reports/depth_ordinal_stage3_independent_review.md`

Stage 3 boundaries: competition test read `false`; Depth assets exported `false`; training run `false`; YOLO 1280 tested `false`.

Independent Stage 3 review: **passed with no P0/P1 findings**. The reviewer reran all eight focused tests, independently checked 60 frames from 20 train/held-out-validation-user trials, and reran the complete 2,910-trial verification. The full-run counts exactly match the formal Stage 3 JSON.

Stage 4 outputs:

- `scripts/export_depth_ordinal_assets.py`
- `tests/test_depth_ordinal_exporter.py`
- `reports/depth_ordinal_stage4_implementation.md`
- `reports/depth_ordinal_stage4_independent_review.md`

Stage 4 boundaries: competition test read `false`; real Depth assets exported `false`; Stage 5 smoke export run `false`; training run `false`.

Independent Stage 4 review: **passed with no P0/P1 findings and no blocker for Stage 5**. The reviewer independently verified all `2,910` trials / `84,906` frames, with zero Depth/IR pairing gaps, duplicate keys, frame-order defects, six-view omissions, or missing IR references. Stage 5 was not run.

Stage 2 acceptance summary:

```text
status: passed
known P0 still effective-valid: 0
source Depth_Color frames scanned: 84,906
unreadable source Depth frames: 0
unexpected non-black/non-JET pixels: 0
all-three-local-IR-invalid: global 3.6228%, small24 2.1372%
worst small-action all-local-invalid: Take_a_selfie 9.6573%
high-confidence left/right swap upper bound: global 0.0444%, worst class 0.4420%
high-risk normalized jump upper bound: global 0.2109%, worst class 2.3613%
test read: false
training run: false
pose image size: 640
pose 1280 tested: false
```

Stage 2 outputs:

- `reports/roi640_stage2_temporal_identity_audit.md`
- `reports/roi640_stage2_temporal_identity/stage2_acceptance_gate.json`
- `reports/roi640_stage2_temporal_identity/normalized_temporal_transitions.csv`
- `reports/roi640_stage2_temporal_identity/normalized_jump_candidates.csv`
- `reports/roi640_stage2_temporal_identity/left_right_swap_candidates.csv`
- `reports/roi640_stage2_temporal_identity/long_invalid_runs.csv`
- `reports/roi640_stage2_temporal_identity/stage2_review_artifacts.csv`
- Review images: `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_stage2_temporal_review` (`161` contact sheets: `28` swap, `60` jump, `73` long-invalid; all referenced files exist).

## Workspace state

- Project worktree: `D:\work\2026.7.14_kaggle\40class-ir-primary-interaction-wt`
- Branch: `ir-primary-interaction-hierarchical`
- HEAD before current uncommitted work: `d8ebfa1a761e1669b8a42df82e37b7c0a18c6c35`
- Python: `D:\Anaconda\envs\pyTorch2.7\python.exe`
- No training job is running. The two visible Python processes are unrelated local Codex/plugin processes and must not be terminated.
- Worktree contains multiple untracked implementation/audit files. Inspect `git status --short`; do not discard them.
- Do not read competition test. Do not copy checkpoints, data, caches, or generated images into Git.
- User previously requested not to use the Superpowers skill family.

## Latest confirmed design

### Spatial inputs

- Keep expanded person context.
- Keep left and right directional interaction ROI.
- Keep adaptive relation/joint interaction ROI.
- Remove original global frame.
- Remove independent upper-body ROI.
- Final crop size remains `256x256`; YOLO pose `imgsz=640` is only the upstream pose inference size.

### Modalities

- IR is the high-capacity appearance source for phones, watches, pages, medicine, cups, tableware, keyboards, etc.
- Depth is a lightweight ordinal/relative geometry residual, not raw metric millimetre depth.
- Spatially encode modalities independently, fuse with a shallow reliability-aware gate before the full-sequence TCN.
- `High-capacity IR` is a role description, not a newly approved backbone change. The current prototype uses ImageNet-pretrained MobileNetV3-Small. Before the pilot, verify that this remains the intended IR backbone; whichever backbone is chosen must be fixed identically across all three Depth variants. Do not combine the Depth representation comparison with an IR backbone replacement.

### Supervision

- 40-class main head.
- Binary small-action gate head.
- No conditional small-action classifier. A third-party review accidentally reintroduced it; explicitly reject that change for this experiment.
- Keep the current gate target set fixed as the 24 class IDs already declared in `configs/experiments/ir_primary_depth_residual_fullseq.yaml`: `[1, 2, 4, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 37, 39]`. Do not redefine this set while comparing Depth representations; that would introduce another variable.

## Existing exported dataset

Current baseline export:

- Root: `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_full_inputs_256`
- `2910` train/val samples, `84906` original paired frames, `509436` PNGs, about `17.972 GiB`.
- Every original paired frame is present; `temporal_sampling_applied=false`.
- `all_frame_inputs.csv` preserves per-sample frame order via `source_frame_index`.
- Four IR views and two current RGB Depth_Color views are present.
- Do not duplicate the four IR views in the new export. Create a Depth-only ordinal/mask asset directory plus a combined manifest referencing existing IR files and new Depth files.
- The current two Depth crops were resized in RGB JET space with Lanczos. They are unsuitable for exact inverse-JET recovery and must be regenerated from the original `Depth_Color` files.

Integrity/audit references:

- `reports/roi640_full_inputs_quality_audit.md`
- `reports/roi640_quality_audit/roi640_manual_review_candidates.csv`
- `reports/roi640_quality_audit/roi640_quality_audit_metrics.json`
- Audit script: `scripts/audit_exported_roi_dataset.py`
- Full export script: `scripts/export_ir_primary_model_inputs.py`
- ROI preview/audit: dataset `yulan2` and `reports/roi640_quality_audit/`

## Confirmed Depth_Color facts

- Original files are `640x480` RGB PNGs under `train/Depth_Color`.
- A stratified sample of 50 raw frames showed that 100% of non-black pixels exactly match OpenCV `COLORMAP_JET`.
- The JET LUT has 256 unique colors and contains no pure black, so native RGB can be inverted exactly to an 8-bit LUT index; pure black is an unambiguous invalid-depth pixel.
- This recovers ordinal `0..255`, not physical millimetres.
- The organizer has not documented whether the pre-JET scalar uses a fixed global range or per-frame min/max. Millimetre recovery is not a blocker.
- The model must not be described as using raw metric depth. Use `ordinal/relative-depth geometry branch`.

## New offline Depth asset

At native source resolution, before any crop/resize:

1. Assert every non-black RGB pixel belongs to the OpenCV JET LUT; record unexpected colors and fail loudly above a tiny explicit tolerance.
2. Invert JET to a single-channel `uint8` ordinal index.
3. Produce a separate per-pixel valid mask; black RGB is invalid.
4. Apply the existing context and adaptive-relation ROI coordinates.
5. Resize depth values with mask-aware interpolation:

   ```text
   resize(depth * valid) / max(resize(valid), eps)
   ```

   Use bilinear (or area for downsampling) for values and nearest-neighbour for the final binary mask. Never interpolate in JET RGB space.
6. Save raw ordinal depth and pixel-valid mask without per-frame, per-trial, or dataset normalization.
7. Preserve all original frames and source order. Do not pad or sample offline.
8. Add original timestamp, frame id, source IR path, source Depth path, and inter-frame delta to the combined manifest if available.

Keep `pixel_valid` separate from view validity. A view may be valid while containing invalid depth holes.

## Content/view/temporal masks

Use at least:

```text
view_valid_mask:     [B,T,6]
view_reliability:    [B,T,6]
depth_pixel_valid:   [B,T,2,1,H,W]
temporal_valid_mask: [B,T]
```

- Preserve pose validity separately from pixel/content validity.
- The user approved this content rule: a pose-valid view becomes content-invalid when at least two of these are true: `dynamic_range <= 8`, `std <= 2`, `entropy_32 <= 1`.
- On the existing RGB-JET export this identifies 18 views across 9 samples: 16 IR views and 2 old Depth views. Keep those findings as baseline invalid evidence. The 16 IR flags can be reused directly. After rebuilding ordinal Depth, recompute the same content rule on the new scalar crops because scalar interpolation/statistics differ from RGB JET; retain the already-known invalid Depth findings and add any newly detected ordinal-Depth failures. Therefore the final post-export count is not guaranteed to remain exactly 18.
- Keep every image/frame/sample; only change effective masks.
- This route does not introduce or tune new data augmentation. If the unchanged loader applies any existing transform, multiply invalid input views by the mask again after that transform and normalization so placeholders cannot become non-zero.
- Mask invalid views during spatial attention/fusion. Do not invalidate an entire time step when one local view fails; context remains available.
- A pixel mask used as an input channel is only an additional hint; it never replaces explicit masking. Zero raw/relative values with `depth_pixel_valid`, concatenate the mask if required by the fixed encoder interface, apply the mask again after transforms/normalization, use mask-aware Depth pooling, and still apply hard view masks during fusion.
- Do not add a learnable reliability head. Use this fixed deterministic score in every pilot:

  ```text
  effective_valid = pose_valid AND content_valid

  pose_score:
    context = 1
    directional = min(elbow_confidence, wrist_confidence)
    two-hand relation = min(left_score, right_score)
    single-hand relation = max(left_score, right_score)

  normalized_pose = clip((pose_score - 0.25) / 0.75, 0, 1)
  pixel_coverage: IR = 1; Depth = mean(depth_pixel_valid)

  view_reliability = effective_valid * normalized_pose * pixel_coverage
  ```

- Hard masks remain constraints; reliability only weights evidence that passed the hard masks. Concatenate the six deterministic reliability values with each time step's fused feature through a fixed projection before the TCN. Monitor user/class-specific missingness shortcuts.

## Depth loader variants

The offline asset is always raw ordinal + mask. Implement loader-selectable representations:

### `raw`

```text
d_raw = depth_index / 255
```

### `relative`

- Compute one median/IQR from valid Depth person-context pixels across the entire trial.
- Use the same statistics for every frame and for both context/relation views.
- Never normalize each frame or each ROI independently.

```text
d_rel = (depth_index - clip_median) / (clip_IQR + eps)
```

Use this fixed fallback and stability rule in every variant:

```text
required_valid_pixels = max(1024, ceil(0.01 * T * H * W))

if valid_context_pixel_count < required_valid_pixels:
    relative = 2 * raw - 1
    relative_stats_valid = 0
else:
    median = median(valid trial-context ordinal values)
    scale = max(IQR(valid trial-context ordinal values), 8)
    relative = clip((depth_index - median) / scale, -4, 4)
    relative_stats_valid = 1
```

- Compute statistics only from valid Depth context pixels across the entire trial.
- Use the same median/scale for context and relation and for every frame.
- Never compute separate per-frame or per-view statistics.
- Record every fallback and report counts by action, split, and user.

### `raw+relative`

Concatenate both channels. Keep the pixel mask explicit; invalid zeros must not be confused with a legitimate median relative depth.

Do not assume `raw+relative` wins. The raw channel can preserve user/camera-distance shortcuts. Select using held-out-user validation behavior, not training fit alone.

For a strict comparison, keep one fixed Depth encoder input interface for every variant, for example `[raw, relative, pixel_valid]`: zero the inactive representation channel in the `raw` and `relative` variants, while `raw+relative` activates both. This prevents the first convolution and parameter count from changing between pilots. Continue using the pixel mask explicitly in computation; concatenating it as an input channel does not replace mask-aware gating.

## Full-sequence loading/training

- No Stage A 24-frame / Stage B 96-frame pipeline.
- Current sequence statistics: mean `29.18`, median `24`, p95 `69`, p99 `106.91`, maximum `236` frames.
- Use length-bucketed batches and batch padding with `temporal_valid_mask`.
- Control memory via a maximum total frames per batch, small physical batch, gradient accumulation, AMP, and activation checkpointing. Do not solve memory by permanently sampling the offline dataset.
- Evaluation must use each complete original sequence.
- The spatial encoder and TCN should train end-to-end for the selected formal configuration.

## Reserved multimodal expert contract (Stage 7 interface only)

Skeleton and other sensor experts are intentionally deferred until after the three Depth representation pilots and one frozen formal visual baseline. Stage 7 only reserves an interface; it does not add a modality or a fusion loss.

Keep model tensors separate from data/evaluation metadata:

```python
@dataclass
class ExpertOutput:
    main_logits: Tensor
    embedding: Tensor
    quality: Tensor
    quality_mask: Tensor
    availability: Tensor

@dataclass
class ExpertBatchResult:
    sample_ids: list[str]
    output: ExpertOutput
    small_gate_logits: Tensor | None = None
    sequence_features: Tensor | None = None
    temporal_mask: Tensor | None = None
    timestamps: Tensor | None = None
```

Stage 7 requirements:

- `main_logits`, `embedding`, fixed-schema `quality`, `quality_mask`, and `availability` are the stable model-output contract.
- `sample_ids` belong to the batch/evaluation wrapper, not `forward()`.
- Diagnostic sequence tensors and timestamps are optional wrapper fields and are not inputs to the first fusion experiment.
- Preserve a stable `class_map_version/hash`.
- Future expert outputs must be fused by an explicit one-to-one `sample_id` join with duplicate/missing/set-equality checks. Never rely on dataloader or array order.
- Preserve `small_gate_logits` as a visual-only auxiliary output. It is not a conditional classifier and does not control the first fusion experiment.
- A no-op fusion path (`alpha=0`) must reproduce the visual main prediction exactly.

## Deferred Skeleton/fusion plan (after the formal visual baseline)

Do not execute this section during the current pure-visual route. The predeclared order is:

1. Finish the variable-length visual pipeline, all three Depth pilots, representation selection, and one formal visual run.
2. Freeze the formal visual checkpoint.
3. Train and evaluate an independent full-sequence Skeleton expert using the official complete 17-joint representation.
4. Generate user-cross-fitted OOF logits for both experts using only the 14 training users.
5. Fit temperatures and fusion parameters on those OOF predictions.
6. Retrain/finalize the two experts on all 14 training users.
7. Evaluate the four held-out users once; never fit temperatures, thresholds, `alpha`, or gates on them.

The primary first fusion is a calibrated probability mixture:

```python
pv = softmax(visual_logits / Tv, dim=-1)
ps = softmax(skeleton_logits / Ts, dim=-1)
alpha = alpha0 * skeleton_available * deterministic_skeleton_quality
p_fused = (1 - alpha) * pv + alpha * ps
fused_log_prob = log(p_fused.clamp_min(1e-8))
```

- `alpha=0` strictly recovers visual probabilities; `alpha=1` strictly recovers Skeleton probabilities.
- The log-space convex combination is a product/log-opinion pool, not a probability mixture. It may be predeclared only as a secondary control because a weak Skeleton expert can veto visual small-object evidence.
- Do not choose between mixture and log-opinion pooling on the four held-out users.
- Predeclare temperature objective, `alpha0` grid, selection metric, tie-break rule, and whether any small/non-small grouping is permitted before viewing fusion results.
- The first fusion round stops at deterministic quality weighting. No embedding-aware gate, cross-modal matching score, or `p_small`-dependent alpha.
- Skeleton quality must be label-free and inference-time computable, based on sequence completeness/effective-frame rate, bone-length anomalies, velocity spikes, and availability. Do not use an always-one official confidence or tune thresholds on held-out results.
- Skeleton first participates only in the 40-class main prediction. The visual `small_gate_logits` remain unchanged.

Only after the probability-mixture baseline shows stable cross-user gains may a quality-only learned gate be tested, followed later by projected embedding residual fusion. Skeleton must never be inserted into IR/Depth spatial gating, treated as a seventh ROI, or concatenated per frame in the current visual TCN.

## Controlled representation pilot

Run exactly three predeclared variants with all other settings identical:

1. inverse-JET `raw`
2. inverse-JET `relative`
3. inverse-JET `raw+relative`

Use the same split, seed, ROI, IR files, batch frame budget, optimizer, learning rate, epochs/patience, model capacity, unchanged augmentation setting, and checkpoint rule. Do not add or tune augmentation in this route. Pilot budget must be fixed before viewing results. Do not tune each variant separately.

Report at minimum:

- train/val Accuracy and Macro-F1 plus generalization gap
- weighted F1, Top-3/Top-5, validation loss
- 40-class and small-action per-class F1
- zero-F1 classes and predicted-class coverage
- performance for each of the four held-out validation users
- high-confidence errors
- view/pixel mask rates by action and user
- if practical, user-ID linear probe on clip embeddings

Selection priority:

1. held-out-user validation Macro-F1 and small-action Macro-F1
2. no meaningful overall Accuracy loss
3. smaller train/val gap and lower user-ID predictability
4. stability across validation users/classes

After selecting one representation, run one formal experiment only. Do not automatically add new heads, BN changes, loss weighting, Router, Skeleton, IMU, Radar, Thermal, or other experiments.

## Remaining input-quality work before formal training

The current audit is conditionally passing, not final.

- Add direct redundancy statistics to the report. Already calculated from current pose masks:
  - all classes: `P(relation valid | right invalid)=72.20%`, `P(relation valid | left invalid)=17.56%`, `P(relation valid | both invalid)=0%`;
  - small-object 24: `82.38%`, `6.92%`, `0%`;
  - all-frame local IR valid-count distribution: 0 views `3.62%`, 2 views `10.18%`, 3 views `86.20%`;
  - small-object distribution: 0 views `2.14%`, 2 views `10.15%`, 3 views `87.71%`.
- Implement a left/right identity-swap audit. High left/right box IoU does not prove identity continuity. Combine wrist/elbow trajectories, direct-vs-cross temporal assignment cost, adjacent ROI IoU, person-normalized coordinates, and transient swap-back checks.
- Replace the old `center displacement / 800 >= 20%` conclusion with adjacent ROI IoU, center displacement normalized by person bbox diagonal, ROI area ratio, and wrist displacement normalized by person size. The existing 454 events are candidates, not confirmed defects.
- Improve P1 review artifacts:
  - boundary/low-information: still image;
  - jump/swap: `t-3..t+3` or `t-5..t+5` contact sheet;
  - long invalid run: full mask timeline plus start/middle/recovery windows;
  - small-object semantics: sparse full-sequence summary.
- Clarify report count wording: 72 is the full count under the old broad OR low-information rule; 39 is only the manual list sampling cap of up to 3 per `(action, view)`, not deduplication with jump/boundary findings. Under the newly approved two-of-three rule, only 18 views are invalidated.
- Two known exact-black, pose-valid crops are examples of why modality-specific content masks are required:
  - `Walk / train__c36__user8__2-2-1 / source frame 0 / depth_relation`
  - `Write / train__c18__user3__3-2-1 / source frame 28 / ir_right`

### Hard input acceptance gate

The audit being complete does not imply automatic acceptance. Fix these thresholds before viewing the new audit results. Stop before full Depth export/training if any gate fails:

```text
missing / unreadable / wrong-size exported files = 0
known P0 views still marked effective-valid = 0
unexplained non-JET source colors = 0
all-three-local-IR-invalid rate, global <= 5%
all-three-local-IR-invalid rate, small-action 24 aggregate <= 3%
all-three-local-IR-invalid rate, every small-action class <= 10%
confirmed left/right identity-swap rate, global <= 1%
confirmed left/right identity-swap rate, every class <= 5%
confirmed abnormal normalized-ROI-jump rate, global <= 1%
confirmed abnormal normalized-ROI-jump rate, every class <= 5%
```

The completed Stage 2 audit measures `3.62%` global all-local-invalid, `2.14%` for the small-action 24 aggregate, and a worst small-action class of `Take_a_selfie = 9.66%`. The conservative high-confidence swap upper bound is `0.0444%` globally and `0.4420%` for the worst class. The high-risk normalized-jump upper bound is `0.2109%` globally and `2.3613%` for the worst class. Every predeclared coverage, swap, and jump gate passed. These are conservative automatic-candidate upper bounds rather than manually labelled defect rates; the generated short-sequence review artifacts remain available for targeted human inspection.

For this experiment, keep the existing `imgsz=640` pose cache fixed. Do not test or regenerate an `imgsz=1280` pose cache: the user has decided not to introduce that variable in the current route. The normalized temporal/swap audit is for validating masks and identifying unreliable views under the fixed 640 input, not for opening a 640/1280 comparison.

## Current code status

The active dataset, model, trainer, configuration, expert contract, and Stage 7 tests have been refactored to the variable-length design described above. The obsolete Stage A/Stage B cache path is no longer present in the active trainer. Formal training remains prohibited until the Stage 7 independent review passes and Stages 8-9 complete the preregistered representation smoke tests and pilots.

## Suggested implementation sequence

1. Re-read this handoff plus the referenced audit report and inspect the dirty worktree.
2. Add tests for exact JET inversion, unknown-color rejection, black-invalid behavior, mask-aware resize, two-of-three content validity, and complete frame ordering.
3. Build the Depth-only ordinal/mask exporter and a combined manifest that reuses existing IR images.
4. Run a small export smoke test; verify scalar/mask alignment against native pixels and neighboring frames.
5. Complete normalized temporal and left/right swap audits; generate improved P1 review artifacts.
6. Run the full Depth-only export and integrity audit without reading test.
7. Refactor loader/model/trainer to true variable-length end-to-end full sequence, remove Stage A/B behavior, and expose the reserved visual `ExpertOutput`/`ExpertBatchResult` contract without reading or connecting Skeleton.
8. Smoke test all three loader representations.
9. Run the fixed three-variant pilot, write comparison reports, select one representation.
10. Run one formal experiment with the selected representation, report, commit code/config/reports only, and stop.

Explicitly out of scope for this route: YOLO pose `imgsz=1280`, pose-cache resolution comparisons, full pose-cache regeneration, new data augmentation, conditional fine-action heads, Router, expert routing, and additional sensor modalities.

## Suggested skills

- Do not invoke the Superpowers skill family because the user explicitly opted out earlier in this project.
- Use `diagnosing-bugs` only if the exporter, mask alignment, or training pipeline produces an actual failure.
- Use `code-review` after the new loader/model/trainer is implemented and before formal training.
- Use `handoff` again if the next session must stop before the pilot or formal training completes.
