# Thermal Route A Operational Workflow

**Status:** A1 complete; A2 runtime probe not started. This runbook does not authorize training.

**Branch:** `experiment/thermal-route-a-workflow`

**Machine state:** `configs/experiments/thermal_route_a_workflow.yaml`

**Validator:** `scripts/validate_thermal_route_a_workflow.py`

## Purpose

Run both variants of the fixed Thermal multi-stream student without mixing architecture changes with distillation effects:

1. `A-direct` trains the student from random initialization with hard labels.
2. `A-KD` trains the same student from the same random seed and recipe, adding only fixed train12 teacher-logit loss.

Route A owns the deployable student, its Thermal-only inputs, runtime qualification, direct training, KD consumption, and matched comparison. Route C owns all teacher construction and training. Route A accepts only an audited 40-class train12 logits artifact and never loads a teacher model.

## Authority and boundaries

Read these before changing workflow state:

1. `docs/superpowers/specs/2026-08-20-thermal-single-modal-60-design.md`
2. `docs/superpowers/plans/2026-08-20-thermal-single-modal-60.md`
3. `configs/experiments/thermal_route_a_workflow.yaml`

The fixed development split is `metadata/splits/train12_val2_user6_user7_development.json`. Route A inherits only its user membership and fixed-label metric policy; the file's historical `ir_audit` statistics and Direct-Head boundary text are provenance, not Thermal measurements or Route A permissions. Heldout-4 labels, competition test, quarantined evidence, IR/Depth inputs, IR indices, IR boxes, and frozen IR/X3D evidence remain inaccessible. Train-14 OOF is outside this workflow.

The complete deployment package must remain strictly below `95,000,000` bytes. Teacher assets never count as deployable Thermal assets because they are forbidden from the package entirely.

## State-control rule

Before and after every workflow change, run:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.validate_thermal_route_a_workflow --workflow configs/experiments/thermal_route_a_workflow.yaml
```

The command is read-only. It prints JSON containing `valid`, `status`, `next_action`, and `errors`. A valid initial workflow reports `next_action: a0_input_audit`.

Only the current stage may move to `in_progress`. A stage moves to `completed` only after every listed completion artifact exists, its focused tests pass, and any required manual review is recorded. Then set `current_stage` to the first incomplete stage. Never mark several stages complete in one retrospective edit.

Training stages additionally require the named authorization boolean to be set true with the exact human approval recorded in the corresponding experiment config and report. Approval of this workflow or the parent plan is not training approval.

## Frozen student identity

Both variants use `thermal_multistream_x3d_xs_v1`:

- randomly initialized PyTorchVideo X3D-XS topology;
- shared weights for Thermal full-frame and fixed trial-context views;
- small 3D CNN over signed grayscale differences;
- TCN over Thermal YOLO11n-pose coordinates;
- explicit availability and eight quality values;
- 40-class trial logits;
- three normalized Thermal windows with 16 frames each and 160x160 crops.

No pretrained student tensor is allowed. A-direct and A-KD must match in architecture, initialization seed, sample order, augmentations, optimizer, learning-rate schedule, batch semantics, checkpoint ranking, and epoch cap. The only permitted difference is the teacher-logit KL term declared in the YAML.

## A0: Input audit

**Completed checkpoint (2026-08-20):** The zero-training machine audit and six montage pages passed structured visual review. The user's conditional instruction authorized A1 after that pass; the report records `training_input_approved: true`. This approval does not authorize A2 or training.

**Current artifacts:**

- `reports/thermal_v2_input_audit.md`
- `reports/thermal_v2_input_audit.json`
- `reports/thermal_v2_input_montages/`
- `metadata/thermal/thermal_v2_train12_normalization.json`
- `metadata/thermal/thermal_v2_trial_context.jsonl`
- `metadata/thermal/thermal_v2_trial_context_summary.json`

**Purpose:** Freeze the actual Thermal tensors before model work.

**Actions:**

1. Implement the normalized-window, trial-context, motion, pose, availability, and quality contracts from parent-plan Tasks 2-5.
2. Fit RGB normalization on train12 only.
3. Build the fixed trial-level Thermal YOLO context artifact for every canonical development trial.
4. Generate stratified montages across user, class, duration, confidence, and fallback type.
5. Obtain manual montage approval; scripts cannot approve their own inputs.

**Completion evidence:** `reports/thermal_v2_input_audit.{md,json}` and the montage directory.

**Stop:** Any sample-ID loss, forbidden data access, unreviewed montage, or unstable crop contract.

## A1: Student implementation

**Completed checkpoint (2026-08-20):** The random X3D-XS baseline, fixed multi-stream student, objective-strategy trainer core, locked CLI, configs, and focused tests are implemented. See `reports/thermal_a1_student_implementation.{md,json}`. The full pose cache is absent and remains mandatory before formal training.

**Purpose:** Implement one student architecture before either objective is enabled.

**Actions:**

1. Implement the from-scratch X3D-XS raster module without importing the frozen IR X3D trainer or checkpoint.
2. Implement the shared full/crop encoder, motion CNN, pose TCN, and 780-value fusion input.
3. Test unavailable crop/pose paths, singleton trials, finite logits, shared parameter identity, and state-dict size.
4. Add one strict trainer entry that selects A-direct or A-KD by objective strategy, not by model class.

**Completion evidence:** focused model and trainer tests plus the model source paths named in the parent plan.

**Stop:** Any hidden pretrained student load, architecture divergence between variants, or non-finite unavailable path.

## A2: Runtime probe

**Purpose:** Establish whether the fixed student fits the reference RTX 5060 Laptop GPU before training.

**Actions:**

1. Run real-data FP32 inference and a throwaway bfloat16 forward/backward/optimizer smoke step.
2. Start with physical batch 2 and effective batch 8.
3. Require peak CUDA allocated memory below 7300 MiB.
4. If necessary, reduce physical batch to 1 and raise accumulation to preserve effective batch 8.
5. Do not reduce frames, windows, resolution, streams, or precision to pass the gate.
6. Record parameters, checkpoint proxy bytes, complete-package bytes, latency, and projected training time.

**Completion evidence:** `reports/thermal_generation2_environment_probe.json`.

**Stop:** VRAM or deployment failure. Do not start formal training.

## A3: A-direct

**Purpose:** Measure the fixed multi-stream student without teacher information.

**Prerequisites:**

- A0-A2 completed.
- Route B report verified as required by the parent plan.
- `authorization.a_direct_training: true` after a new explicit human approval.

**Frozen run:** seed `20260715`, AdamW, LR `3e-4`, weight decay `0.05`, three warmup epochs, cosine schedule, 50-epoch hard cap, no resume or extension, and checkpoint selection by fixed-label Macro-F1 then Accuracy, worst-user Accuracy, and lower epoch.

**Required report:** combined and per-user metrics, all class recalls, zero-recall classes, NLL, route availability, stream norms, bytes, latency, memory, and proof that no teacher logits loaded.

**Stop:** Commit the report, not checkpoints. Do not tune from user6/user7 errors.

## A4: Teacher-logits handoff

**Purpose:** Admit one Route C artifact without importing teacher runtime state.

The manifest must prove exactly one teacher passed Accuracy `>=0.60`, Macro-F1 `>=0.45`, and worst-user Accuracy `>=0.50`. It must bind every train12 sample exactly once to finite float32 `[40]` logits and include sample, logits, teacher checkpoint/config, split, class-map, and input-contract hashes.

Reject validation logits, missing/extra/duplicate sample IDs, features, teacher modules, mismatched hashes, TTA-dependent exports, and teacher ensembles. If no teacher passes, mark A-KD blocked by teacher quality and stop without treating it as a student failure.

**Completion evidence:** `reports/thermal_teacher_logits_manifest.json` with `manifest_status: verified` and `passing_teacher_count: 1` reflected in the workflow YAML.

## A5: A-KD

**Purpose:** Isolate the effect of teacher logits on the exact A-direct student.

**Prerequisites:**

- A4 completed with a verified manifest.
- `authorization.a_kd_training: true` after a new explicit human approval.
- A-direct/A-KD config-diff test proves only objective and teacher-manifest fields differ.

Use `T=4.0` and `0.5 * CE + 0.5 * T^2 * KL`. The student process loads detached train12 logits only. It does not load a teacher model or use validation logits, feature matching, attention transfer, a second seed, alpha/temperature search, resume, or extension.

**Required report:** all A-direct metrics plus paired changed predictions, recall deltas, calibration, activation stability, and proof of the matched recipe.

**Stop:** Commit the report, not the student checkpoint or logits archive.

## A6: Paired decision

Join A-direct and A-KD predictions by canonical sample ID. Report paired correctness, Accuracy, fixed-label Macro-F1, worst-user and per-user Accuracy, zero-recall classes, NLL, checkpoint/package bytes, median/p95 latency, and peak memory.

A student passes only when Accuracy is at least `0.60`, Macro-F1 at least `0.45`, worst-user Accuracy at least `0.50`, no more than 10 classes have zero recall, every usable validation logit is finite, and the deployment package is below `95,000,000` bytes.

Output one decision:

- `feasibility_pass_human_review_required`
- `feasibility_failed_stop_without_oof`
- `incomplete_authorization_or_teacher_block`

No result automatically authorizes OOF, heldout evaluation, competition-test inference, or IR fusion. Set A6 completed, validate the workflow, confirm no training process remains, and stop for human review.
