# Thermal Single-Modal 0.60 Feasibility Design

**Status:** Approved for implementation planning only. No training is authorized by this document.

**Scientific baseline:** `c42bb43091c79903e5fde5655c2846c87305895a`

**Program branch at design freeze:** `experiment/thermal-iformer-t-t1b` at `c3be5496d1d13a8d98289ffefbb4dc74c1572b18`

## Objective

Determine whether a competition-compliant Thermal-only model can reach at least `0.60` Accuracy on the fixed user6+user7 development validation set before spending compute on train-14 OOF. The experiment must distinguish gains from a true 3D raster model, Thermal-native context streams, and training-only teacher distillation.

The program ends after the fixed development comparison. A separate human approval is required for OOF, heldout evaluation, competition-test inference, or packaging.

## Inherited authority

This design inherits, in descending order of specificity:

1. `docs/superpowers/plans/2026-08-11-six-modal-program-charter.md`
2. `docs/superpowers/specs/2026-08-11-six-modal-sparse-evidence-fusion-design.md`
3. `docs/superpowers/plans/2026-08-20-thermal-native-expert.md`
4. `reports/thermal_stage0_data_alignment_audit.{md,json}`
5. `reports/thermal_stage0_5_localization_route_audit.{md,json}`
6. `reports/thermal_iformer_t_tsm_train12_val2_interrupted_audit.{md,json}`
7. `reports/thermal_t1b1_bn_diagnostic.{md,json}`
8. `reports/thermal_t1b2_activation_trace.{md,json}`
9. `reports/thermal_t1b3_block_attribution.{md,json}`
10. `reports/thermal_t1b4_head_only_probe.{md,json}`
11. `reports/thermal_mobilenetv3_tsm_epoch14_stopped_audit.{md,json}`

If this design conflicts with the program charter or evidence-isolation rules, the stricter rule wins.

## Frozen evidence boundary

- Training and inference inputs are Thermal frames only.
- Use `metadata/splits/train12_val2_user6_user7_development.json` without rewriting membership.
- Fit model parameters, normalization statistics, crop policy constants, and teacher logits from train12 only.
- Validation labels may be read only by the evaluator for checkpoint ranking and final development metrics.
- Do not read heldout-4 labels, competition test, quarantined evidence, or isolated ExpertEvidence.
- Do not import IR/Depth frames, indices, boxes, logits, embeddings, motion peaks, or timestamps.
- The frozen IR/X3D expert, checkpoints, evidence, and runtime files are read-only and outside this program.
- No train-14 OOF is authorized unless a candidate passes the feasibility gate and a human approves the next stage.

## Prior-generation decision

The first Thermal generation is closed:

| Candidate | Accuracy | Macro-F1 | Worst-user Accuracy | Decision |
|---|---:|---:|---:|---|
| iFormer-T+TSM epoch 16 | 0.29443 | 0.19583 | 0.21026 | Failed: weak validation and finetuning-induced activation tails |
| frozen iFormer-T head-only | 0.16180 | 0.07120 | 0.15897 | Failed: stable but not linearly separable enough |
| MobileNetV3-Small+TSM epoch 12 | 0.28647 | 0.21370 | 0.25128 | Frozen weak baseline: stable but below feasibility |

Neither model may be resumed, retuned, or promoted by the new program. Their predictions remain comparison artifacts only.

## Fixed experiment matrix

### Route B: compact true-3D baseline

`B-X3D-XS` is a Thermal-only raster baseline:

- Architecture: PyTorchVideo X3D-XS topology constructed with `pretrained=False`.
- Initialization: random; no Kinetics or other external weight tensor may be loaded.
- Input: full-frame Thermal RGB only.
- Temporal contract: three normalized windows, 16 frames per window.
- Window aggregation: mean of the three clip embeddings, then a linear 40-class head.
- Purpose: measure the value of native spatiotemporal convolutions and longer trial coverage without localization, motion, pose, or distillation.

This is a new Thermal model. It must not import or modify the frozen IR X3D implementation or checkpoint.

### Route A: compact Thermal multi-stream student

`A-Thermal-MultiStream` has one fixed architecture used by both A experiments:

- Full-frame appearance: shared randomly initialized X3D-XS raster encoder.
- Thermal YOLO context appearance: the same encoder weights applied to a fixed trial-level crop.
- Motion context: a small 3D CNN over signed consecutive grayscale differences from the full-frame clips.
- Pose context: a small temporal convolution network over Thermal YOLO11n-pose keypoints.
- Fusion: projected stream embeddings, explicit availability and quality values, LayerNorm, and a linear 40-class head.
- No class-conditional routing, motion-peak sampling, IR boxes, or per-frame moving crops.

The full-frame and crop views are encoded sequentially to bound VRAM. Crop features are masked to zero when localization is unavailable; a failed crop never causes a canonical trial to be dropped.

Two training variants must proceed:

1. `A-direct`: random initialization and hard-label supervised training only.
2. `A-KD`: the exact same architecture, random initialization seed, data order, augmentations, optimizer, and schedule as `A-direct`, adding only fixed train12 teacher-logit loss.

Changing the student architecture between A-direct and A-KD invalidates the comparison.

### Route C: training-only teacher ceiling

Route C produces teacher validation metrics and train12 logits; it is not deployable evidence.

1. `C1-R2Plus1D18`: torchvision R(2+1)D-18 with official Kinetics-400 weights, full-frame plus fixed Thermal context crop, and the common three-window contract.
2. `C2-VideoMAE-S`: official VideoMAE-S weights, eligible for separate approval only if C1 fails at least one teacher gate. It uses the same Thermal views and trial identities. If C1 passes all teacher gates, skip C2 for this generation.

Large pretrained teacher backbones are permitted only during training under the stated competition interpretation. Teacher weights, optimizer states, features, code-only caches, and checkpoints are excluded from the final inference package. Only 40-class train12 logits and their provenance may cross into A-KD training.

A-KD is authorized to train only when one teacher reaches all teacher gates:

- Accuracy at least `0.60`.
- Macro-F1 at least `0.45`.
- Worst-user Accuracy at least `0.50`.
- Finite logits for every train12 sample.

If no teacher passes, Route C stops and A-KD is recorded as blocked by teacher quality, not as a student failure.

Generation 2 does not ensemble teachers. If C1 fails and C2 passes, use C2 alone.

## Thermal-only input contract v2

### Canonical identity and availability

- Preserve every canonical sample ID from the fixed split.
- `directory_present`, `decodable`, and `usable` remain distinct.
- A non-usable trial emits availability false, finite zero tensors, and quality metadata; it is excluded from supervised loss and metric denominators by the existing explicit mask rule.
- Single-frame and short trials are retained. Repeated nearest indices are exposed through a uniqueness mask.

### Normalized temporal windows

For each trial with `N >= 1` decoded frames, generate three windows on the Thermal-native normalized axis:

1. early: `[0.00, 0.50]`
2. middle: `[0.25, 0.75]`
3. late: `[0.50, 1.00]`

Each window contains 16 equally spaced normalized targets. Map each target to the nearest Thermal frame with half-up rounding and clamp to `[0, N-1]`. Do not select frames from motion peaks. Do not copy indices from another modality. For `N=1`, all targets map to frame zero and the uniqueness mask marks only the first occurrence unique.

### Raster preprocessing

- Decode the existing Thermal RGB pseudocolor without attempting absolute-temperature recovery.
- Resize each view to a 176-pixel short side and use a 160x160 crop.
- Validation uses a center crop and no random flip.
- Training uses one clip-consistent random resized crop with scale `[0.80, 1.00]`, ratio `[0.90, 1.10]`, and horizontal flip probability `0.5`.
- Spatial parameters are identical across all frames and windows of one view in one trial.
- Do not use color jitter in generation 2 because the pseudocolor-to-temperature relationship is not proven invariant.
- Normalize RGB with train12-only channel mean and standard deviation written to a hash-gated JSON artifact.

### Fixed Thermal YOLO context crop

Localization is computed from Thermal frames only:

1. Probe eight uniformly spaced normalized positions over the complete trial.
2. On each distinct probe frame, run the existing YOLO11n-pose and retain only the highest-confidence person with confidence at least `0.25`.
3. Require one hit for a one-frame trial and at least two hits otherwise.
4. Take the union of accepted boxes.
5. Expand the union by `1.4x` around its center.
6. Enforce a square side of at least `0.35 * max(frame_width, frame_height)`.
7. Shift and clamp the square inside the Thermal frame without resizing an IR box.
8. Use this one fixed box for every frame and window of the trial.
9. On failure, mark crop availability false and retain the full-frame path.

The crop artifact records probe indices, detections, confidence statistics, final box, coverage, fallback reason, source frame dimensions, YOLO weight SHA256, and policy version. A visual montage is required before any training authorization.

### Motion stream

- Convert normalized full-frame RGB to grayscale with fixed coefficients `(0.299, 0.587, 0.114)`.
- Compute signed differences `gray[t] - gray[t-1]`; prepend an all-zero first difference.
- Preserve the common window and uniqueness masks.
- Do not use motion magnitude to select frames or crops.

### Pose stream

- Run YOLO11n-pose on the already selected Thermal frames.
- Use 17 COCO keypoints `(x, y, confidence)` normalized by Thermal width and height.
- Append normalized person-box center, width, height, and detector confidence, yielding 56 values per time step.
- Missing detections emit zeros and a false pose mask.
- Pose detections are input context and quality evidence, not a reason to remove a trial.

## Exact student architecture

### Shared raster encoder

- X3D-XS topology from PyTorchVideo, `pretrained=False`.
- Remove the Kinetics projection and retain the pooled feature vector.
- Project each aggregated full/crop feature to 256 dimensions with `Linear -> LayerNorm -> SiLU`.
- The same X3D parameters and projection are used for both views.

### Motion encoder

- `Conv3d(1,16,kernel=(3,5,5),stride=(1,2,2),padding=(1,2,2))`
- `BatchNorm3d -> SiLU`
- `Conv3d(16,32,3,stride=2,padding=1) -> BatchNorm3d -> SiLU`
- `Conv3d(32,64,3,stride=2,padding=1) -> BatchNorm3d -> SiLU`
- Adaptive global average pooling and `Linear(64,128) -> LayerNorm -> SiLU`.

### Pose encoder

- Input projection `Conv1d(56,128,kernel_size=1)`.
- Two residual depthwise-separable temporal blocks with kernel size 3 and dilations 1 and 2.
- Masked temporal mean, then `Linear(128,128) -> LayerNorm -> SiLU`.

### Fusion and output

- Concatenate full appearance 256, crop appearance 256, motion 128, pose 128, four stream-availability values, and eight versioned quality values.
- Apply `LayerNorm(780)`, `Dropout(0.2)`, and `Linear(780,40)`.
- Return 40 finite logits plus availability and quality tensors compatible with `ExpertEvidence`.
- B-X3D-XS uses only the shared raster feature followed by `Dropout(0.2)` and `Linear(feature_dim,40)`.

## Optimization contracts

### B and A-direct

- Seed: `20260715`.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `0.05`.
- Schedule: 3 warmup epochs followed by cosine decay.
- Maximum epochs: 50; no automatic extension or resume.
- Label smoothing: `0.1`.
- Gradient clipping: global norm `1.0`.
- Mixed precision: bfloat16 on CUDA, FP32 fallback.
- Effective batch: 8 trials through gradient accumulation.
- Checkpoint rank: fixed-label Macro-F1, then Accuracy, then worst-user Accuracy, then lower epoch.

### Teachers

- Seed: `20260715`.
- Training uses inverse-frequency weighted random sampling with replacement over usable train12 class IDs, with one epoch containing the same number of draws as usable train12 trials. This gives every class equal sampling probability mass. Validation preserves its natural distribution and evaluates every usable validation trial exactly once.
- Pretrained backbone learning rate `1e-5`; new classifier learning rate `1e-4`.
- AdamW, weight decay `0.05`, 3 warmup epochs, cosine decay, 30 epochs maximum.
- Cross-entropy with label smoothing `0.1`.
- Effective batch: 8 trials; sequential view/window execution is mandatory.
- Checkpoint rank matches the student rule.

The natural-shuffle C1 run started on 2026-08-21 was stopped during epoch 6 and is invalid for teacher-family conclusions: usable train12 class counts ranged from 3 to 225 and the run collapsed toward the majority class. It must never be resumed; the corrected run restarts from the official pretrained weights.

### A-KD

- Clone the A-direct recipe and random initialization seed.
- Teacher targets are deterministic float32 40-class logits for train12 sample IDs only.
- Temperature `T=4.0`.
- Loss: `0.5 * CE(hard_label) + 0.5 * T^2 * KL(student/T, teacher/T)`.
- No feature matching, attention transfer, validation-logit training, or teacher parameter loading in the student process.

## Hardware and runtime gates

The reference workstation is an NVIDIA GeForce RTX 5060 Laptop GPU with 8151 MiB VRAM, a Ryzen 9 8945HX, and approximately 31 GiB RAM.

Before every full training run:

- Run a real-data forward/backward smoke test with the final view/window count.
- Require peak CUDA allocated memory below `7,300 MiB`.
- Start with physical batch 2 for B/A and batch 1 for teachers.
- If the memory gate fails, reduce physical batch to 1 and increase accumulation to preserve effective batch 8. Do not silently reduce frames, windows, resolution, streams, or precision.
- Require finite loss, logits, gradients, and a one-optimizer-step state change.
- Record median and p95 trial latency, peak allocated/reserved memory, parameter count, checkpoint bytes, and serialized deployment bytes.

## Development decision gates

Every trained candidate reports combined user6+user7 and per-user results for Accuracy, fixed-label Macro-F1, per-class recall, zero-recall classes, NLL, and confusion pairs.

A deployable student passes the feasibility gate only if its selected checkpoint has:

- combined Accuracy at least `0.60`;
- combined Macro-F1 at least `0.45`;
- worst-user Accuracy at least `0.50`;
- finite outputs for every usable validation trial;
- no more than 10 zero-recall classes;
- complete deployment package strictly below `95,000,000` bytes.

The 0.60 threshold is a feasibility decision, not a claim of generalization. Repeated user6+user7 experiments are capped by the fixed matrix; no extra architecture or hyperparameter search may be introduced from validation errors without a new preregistration.

## Ordered execution and stopping

1. Implement and audit the shared input/runtime contracts without training.
2. Train B-X3D-XS.
3. Train A-direct.
4. Train C1; conditionally train C2 under its declared rule.
5. Produce hash-gated train12 teacher logits only if the teacher gate passes.
6. Train A-KD.
7. Compare B, A-direct, and A-KD with paired predictions.
8. Stop and request human review.

If B and A-direct both remain below `0.40` Accuracy, teacher work may still proceed because it tests a distinct training hypothesis. If all eligible students remain below `0.60`, the program stops without OOF. If one or more students pass, the best candidate is not automatically promoted; a separate plan must authorize train-14 OOF and final evidence production.

## Explicit exclusions

- No iFormer continuation or iFormer-S upgrade.
- No MobileNet continuation.
- No ResNet18 rerun.
- No VideoMamba, DART, LLM API, IR+Thermal fusion, or Depth input.
- No motion-peak frame selection.
- No IR-bbox reuse or resolution-scaled IR localization.
- No early fusion with another modality.
- No heldout or competition-test evaluation.
- No OOF in this plan.
- No automatic model promotion.

## Source provenance

- Torchvision R(2+1)D-18 official model and weights: <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html>
- Official VideoMAE model zoo: <https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md>
- PyTorchVideo X3D model zoo: <https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html>

All source revisions, licenses, checkpoint URLs, byte counts, and SHA256 values must be captured by the no-training environment probe before a corresponding route is authorized.
