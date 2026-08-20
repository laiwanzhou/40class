# Thermal Single-Modal 0.60 Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** Test three attributable Thermal-only routes on the fixed train12/user6+user7 development split and determine whether a competition-compliant student can reach Accuracy `>=0.60` without starting OOF.

**Architecture:** Route B is a randomly initialized X3D-XS full-frame baseline. Route A is one fixed multi-stream student with shared full/crop X3D-XS appearance, a small motion CNN, a pose TCN, and explicit availability/quality fusion; A-direct and A-KD differ only by the distillation loss. Route C first uses a training-only pretrained R(2+1)D-18 teacher; VideoMAE-S is eligible only if that teacher fails its gates. Teacher weights never enter deployment.

**Tech Stack:** Python 3.11, PyTorch 2.7, torchvision 0.22, PyTorchVideo 0.1.5, Ultralytics YOLO11n-pose, NumPy, Pillow/OpenCV, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-thermal-single-modal-60-design.md`

## Execution rules

- Create a new isolated worktree and branch `experiment/thermal-single-modal-60` from the `experiment/thermal-iformer-t-t1b` commit that contains this plan. Before Task 1, its only changes relative to implementation base `c3be5496d1d13a8d98289ffefbb4dc74c1572b18` must be this plan and its design spec.
- Record scientific baseline `c42bb43091c79903e5fde5655c2846c87305895a` separately from the implementation base.
- Do not alter frozen IR/X3D code, checkpoints, evidence, or reports. Do not resume iFormer or MobileNet.
- Tasks 1-9 are implementation and no-training audits. Tasks 10-14 require their own explicit human training approvals; approval of this plan is insufficient.
- Stop after Task 15. OOF, heldout, competition test, and cross-modal fusion remain unauthorized.

## Task 1: Isolate the work and freeze generation-1 decisions

**Files:**
- Create: `reports/thermal_generation2_decision_freeze.md`
- Create: `reports/thermal_generation2_decision_freeze.json`
- Create: `tests/test_thermal_generation2_decision_freeze.py`

**Step 1: Verify SHAs and create the worktree**

Run `git status --short`, `git rev-parse HEAD`, `git rev-parse c42bb43091c79903e5fde5655c2846c87305895a`, and `git diff --name-only c3be5496d1d13a8d98289ffefbb4dc74c1572b18..HEAD`. Require the final command to list only this plan and its design spec, then use the Superpowers worktree safety procedure to create the new branch from that plan-bearing commit.

**Step 2: Write the failing contract test**

Assert that the JSON contains both SHAs; the frozen iFormer epoch-16, head-only, and MobileNet epoch-12 metrics; decisions `failed`, `failed`, and `weak_stable_baseline`; `resume_authorized: false`; the exact generation-2 experiment matrix; and all sealed/test/OOF permissions false.

**Step 3: Verify RED**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_thermal_generation2_decision_freeze.py -q
```

Expected: FAIL because the artifacts are absent.

**Step 4: Create artifacts from committed JSON reports**

Do not recompute old metrics from ignored checkpoints. Include report paths and commits, selected epochs, metrics, checkpoint hashes/bytes, the new feasibility gates, and the four ordered experiments `B-X3D-XS`, `A-direct`, `C-teacher`, `A-KD`.

**Step 5: Verify GREEN and commit**

Run Step 3, then commit the three files with message `docs: freeze thermal generation two decision`.

## Task 2: Implement Thermal v2 normalized windows

**Files:**
- Create: `src/data/thermal_v2_sampling.py`
- Create: `tests/test_thermal_v2_sampling.py`
- Modify: `src/data/thermal_native_dataset.py`
- Modify: `tests/test_thermal_native_dataset.py`

**Step 1: Write failing tests for the pure API**

```python
def normalized_window_indices(
    frame_count: int,
    *,
    windows: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0)),
    frames_per_window: int = 16,
) -> tuple[tuple[int, ...], ...]: ...

def uniqueness_mask(indices: Sequence[int]) -> torch.Tensor: ...
```

Cover `N=1,2,8,16,101`, half-up rounding, bounds, monotonicity, three 16-frame windows, early index zero, late index `N-1`, singleton repeats, and absence of IR/motion arguments.

**Step 2: Add a failing dataset test**

`contract_version="thermal_v2"` must produce `[3,16,3,160,160]`, source indices, and uniqueness masks. V1 must retain its frozen 16x224 behavior.

**Step 3: Verify RED**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_thermal_v2_sampling.py tests/test_thermal_native_dataset.py -q
```

**Step 4: Implement the pure sampler and explicit v2 branch**

Use only Thermal frame count. V2 resizes the short side to 176, crops 160, shares spatial parameters across a view/trial, preserves unavailable trials as finite zeros with false availability, and exposes source/unique masks. Do not relax V1 checks.

**Step 5: Verify GREEN and regression tests**

Run Step 3, then `pytest tests/test_thermal_stage0_audit.py tests/test_thermal_stage0_5_audit.py -q`. Commit with `feat: add thermal v2 normalized windows`.

## Task 3: Implement fixed trial-level Thermal context

**Files:**
- Create: `src/roi/thermal_trial_context.py`
- Create: `tests/test_thermal_trial_context.py`
- Create: `scripts/build_thermal_trial_context.py`
- Create: `tests/test_build_thermal_trial_context.py`

**Step 1: Write failing geometry tests**

```python
@dataclass(frozen=True)
class ThermalTrialContext:
    sample_id: str
    available: bool
    probe_indices: tuple[int, ...]
    accepted_detections: tuple[tuple[float, float, float, float, float], ...]
    bbox_xyxy: tuple[int, int, int, int] | None
    fallback_reason: str | None

def build_trial_context(
    *,
    sample_id: str,
    frame_size: tuple[int, int],
    frame_count: int,
    detections_by_index: Mapping[int, Sequence[Sequence[float]]],
    confidence_threshold: float = 0.25,
    expansion: float = 1.4,
    minimum_side_ratio: float = 0.35,
) -> ThermalTrialContext: ...
```

Test eight normalized probes, short-trial probe deduplication, highest-confidence person selection, singleton/normal hit counts, union, expansion, square enforcement, shift/clamp, invalid detections, and explicit fallback reasons.

**Step 2: Verify RED, then implement pure geometry**

Run `pytest tests/test_thermal_trial_context.py tests/test_build_thermal_trial_context.py -q`. Keep the geometry independent of Ultralytics and accept original Thermal pixel coordinates only.

**Step 3: Implement the offline builder**

The script verifies the YOLO11n-pose SHA256, runs only deduplicated Thermal probe frames, retains the highest-confidence person, writes every canonical trial to JSONL, refuses frozen IR output paths, and supports `--dry-run` without model loading.

**Step 4: Verify GREEN and commit**

Run the Step 2 tests. Commit with `feat: add fixed thermal trial context`.

## Task 4: Build normalization, motion, pose, and quality tensors

**Files:**
- Create: `src/data/thermal_v2_features.py`
- Create: `tests/test_thermal_v2_features.py`
- Modify: `src/data/thermal_native_dataset.py`
- Modify: `tests/test_thermal_native_dataset.py`
- Create: `scripts/fit_thermal_v2_normalization.py`
- Create: `tests/test_fit_thermal_v2_normalization.py`

**Step 1: Write failing feature tests**

Test `signed_grayscale_differences`, `encode_pose_step`, and `masked_stream_mean`. Require signed differences with a zero first step, 56 pose values, finite zeros for missing pose, and mask-respecting aggregation.

**Step 2: Write leakage tests**

The normalization fitter must reject every sample outside train12. Its JSON records count, RGB mean/std, split hash, sampling policy, code revision, and artifact hash.

**Step 3: Verify RED**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_thermal_v2_features.py tests/test_fit_thermal_v2_normalization.py tests/test_thermal_native_dataset.py -q
```

**Step 4: Implement v2 items and collation**

Return `full_rgb`, `crop_rgb`, `motion`, `pose`, `pose_mask`, window/source/unique masks, four availability values (`full,crop,motion,pose`), and eight named quality values. The qualities are decodable ratio, unique sampled ratio, crop hit ratio, median detector confidence, crop area ratio, pose valid-step ratio, duplicate-frame ratio, and capped short-trial ratio. Loss eligibility remains separate.

**Step 5: Verify GREEN and commit**

Run Step 3. Commit with `feat: build thermal v2 feature streams`.

## Task 5: Run the no-training input audit

**Files:**
- Create: `scripts/audit_thermal_v2_inputs.py`
- Create: `tests/test_audit_thermal_v2_inputs.py`
- Generate: `metadata/thermal/thermal_v2_train12_normalization.json`
- Generate: `metadata/thermal/thermal_v2_trial_context.jsonl`
- Generate: `metadata/thermal/thermal_v2_trial_context_summary.json`
- Generate: `reports/thermal_v2_input_audit.md`
- Generate: `reports/thermal_v2_input_audit.json`
- Generate: `reports/thermal_v2_input_montages/thermal_v2_context_01.jpg` and subsequent pages

**Step 1: Write the failing audit test**

Require split/artifact hashes; present/decodable/usable counts; crop detection/fallback by user/class/duration; confidence/area/continuity; singleton/short behavior; every canonical ID once; tensor shapes/finite checks; montage manifest; and explicit zero access to forbidden data.

**Step 2: Verify RED, then generate artifacts**

Run `pytest tests/test_audit_thermal_v2_inputs.py -q`, fit train12 normalization, build contexts, and generate stratified montages showing full frame, box overlay, fixed crop, sample ID, user/class, confidence, and fallback.

**Step 3: Require manual montage approval**

The report contains a human-written `training_input_approved` field; scripts cannot set it. Stop if false or absent.

**Step 4: Verify GREEN and commit**

Run the test and commit code, tests, metadata, reports, and montages with `audit: freeze thermal v2 model inputs`.

## Task 6: Implement B-X3D-XS from scratch

**Files:**
- Create: `src/models/thermal_x3d_xs.py`
- Create: `tests/test_thermal_x3d_xs.py`

**Step 1: Write failing model tests**

```python
def build_thermal_x3d_xs_backbone(*, pretrained: bool = False) -> nn.Module: ...

class ThermalX3DXSBaseline(nn.Module):
    def forward(self, full_rgb, *, window_mask, availability, quality): ...
```

Require `[B,3,3,16,160,160]`, sequential window encoding, masked mean, 40 finite logits, and preserved availability/quality. `pretrained=True` must raise a policy error; a monkeypatched downloader must never be called.

**Step 2: Verify RED, implement, verify GREEN**

Run `pytest tests/test_thermal_x3d_xs.py -q`. Construct PyTorchVideo X3D-XS directly with `pretrained=False`, remove only the Kinetics projection, and declare random initialization provenance. Do not import or edit frozen IR X3D modules. Rerun tests.

**Step 3: Commit**

Commit with `feat: add from-scratch thermal x3d-xs`.

## Task 7: Implement the fixed A student

**Files:**
- Create: `src/models/thermal_multistream.py`
- Create: `tests/test_thermal_multistream.py`

**Step 1: Write failing architecture tests**

Test the exact motion CNN, pose TCN, shared raster encoder object identity, 780-value fusion input, and 40 logits. Crop unavailability must zero only its projected feature; missing auxiliary streams must still produce finite full-frame logits. Assert fewer than 10 million parameters and a state dict below 45 million bytes.

**Step 2: Implement the public model**

```python
class ThermalMultiStreamStudent(nn.Module):
    def forward(
        self, *, full_rgb, crop_rgb, motion, pose, window_mask,
        pose_mask, availability, quality
    ) -> dict[str, torch.Tensor]: ...
```

Return `logits`, `embedding`, `availability`, and `quality`. Encode views/windows sequentially. Use exactly the architecture in the spec.

**Step 3: Verify and commit**

Run `pytest tests/test_thermal_multistream.py -q`. Commit with `feat: add thermal multistream student`.

## Task 8: Add a strict generation-2 trainer

**Files:**
- Create: `src/train_thermal_generation2.py`
- Create: `tests/test_train_thermal_generation2.py`
- Create: `scripts/run_thermal_generation2.py`
- Create: `tests/test_run_thermal_generation2.py`
- Create: `configs/experiments/thermal_b_x3d_xs_train12_val2.yaml`
- Create: `configs/experiments/thermal_a_multistream_direct_train12_val2.yaml`

**Step 1: Write failing config/trainer tests**

Require the fixed split/class map; route enum; `training_authorized: false`; matching CLI authorization token; v2 shapes; random B/A initialization; no resume; 50-epoch hard stop; masked loss; fixed-label metrics; rank `(macro_f1,accuracy,worst_user_accuracy,-epoch)`; selected-checkpoint prediction archives; deterministic seed/data order; and no import from `src.train_x3d_s_visual_expert`.

**Step 2: Verify RED, then implement**

Run `pytest tests/test_train_thermal_generation2.py tests/test_run_thermal_generation2.py -q`. Reuse only stable metrics and `ExpertEvidence` contracts. Save config/code revision, histories, checkpoint SHA/bytes, sample IDs, labels/logits/predictions/users, memory, elapsed time, and initialization provenance.

**Step 3: Freeze configs and verify GREEN**

Encode all optimization values from the spec. Run focused tests and:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests -q -k "thermal and not x3d_s_online_roi_parity"
```

Commit with `feat: add thermal generation two trainer`.

## Task 9: Probe hardware and deployment without training

**Files:**
- Create: `scripts/probe_thermal_generation2.py`
- Create: `tests/test_probe_thermal_generation2.py`
- Generate: `reports/thermal_generation2_environment_probe.json`

**Step 1: Write failing probe tests**

Require a real-data forward/backward/optimizer step for B and A, FP32 evaluation, bfloat16 training, finite checks, CUDA peak allocated/reserved, physical/effective batch, parameters, state-dict bytes, median/p95 latency, and output shape. Require peak allocated below 7300 MiB and provisional complete package below 95,000,000 bytes.

Add a provenance scanner that rejects pretrained B/A tensors and any teacher file in their deployment manifests.

**Step 2: Verify RED, implement, and run**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.probe_thermal_generation2 --config configs/experiments/thermal_b_x3d_xs_train12_val2.yaml --config configs/experiments/thermal_a_multistream_direct_train12_val2.yaml --output reports/thermal_generation2_environment_probe.json
```

Start B/A at physical batch 2. If the gate fails, use batch 1 plus accumulation 8. Do not change frames, windows, resolution, streams, or precision.

**Step 3: Verify, estimate compute, and stop**

Run `pytest tests/test_probe_thermal_generation2.py -q`. Report measured seconds/step and projected time for B, A-direct, C1, conditional C2, and A-KD. Confirm no training process. Commit with `audit: qualify thermal generation two runtime`.

## Task 10: Train and report Route B

**Authorization gate:** A new human message must explicitly authorize `B-X3D-XS` after Task 9.

**Files:**
- Generate ignored: `outputs/thermal_b_x3d_xs_train12_val2/`
- Create: `scripts/report_thermal_b_x3d_xs.py`
- Create: `tests/test_report_thermal_b_x3d_xs.py`
- Generate: `reports/thermal_b_x3d_xs_train12_val2.md`
- Generate: `reports/thermal_b_x3d_xs_train12_val2.json`

**Step 1: Write the failing report test**

Require selected epoch/history; combined and per-user metrics; 40 recalls; zero-recall classes; confusion pairs; NLL; checkpoint hash/bytes; latency/memory; and all policy assertions.

**Step 2: Obtain approval and commit authorization**

The test may fail only because run artifacts are absent. Record the exact approval text/date in config and commit before launching.

**Step 3: Run exactly once**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.run_thermal_generation2 --config configs/experiments/thermal_b_x3d_xs_train12_val2.yaml --authorize-training thermal-b-x3d-xs
```

No resume, extension, alternate seed, or validation-driven tuning.

**Step 4: Report, test, commit, and stop**

Generate the report, run its test, and commit config/report/script/test with `report: record thermal x3d-xs baseline`. Never commit checkpoints.

## Task 11: Train and report A-direct

**Authorization gate:** A new human message must explicitly authorize `A-direct` after Route B review. B need not pass 0.60.

**Files:**
- Generate ignored: `outputs/thermal_a_multistream_direct_train12_val2/`
- Create: `scripts/report_thermal_a_direct.py`
- Create: `tests/test_report_thermal_a_direct.py`
- Generate: `reports/thermal_a_multistream_direct_train12_val2.md`
- Generate: `reports/thermal_a_multistream_direct_train12_val2.json`

**Step 1: Write the failing report test**

Require common metrics, route availability, stream norms, parameters/bytes, and proof that no teacher logits loaded.

**Step 2: Approve, commit authorization, and run once**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.run_thermal_generation2 --config configs/experiments/thermal_a_multistream_direct_train12_val2.yaml --authorize-training thermal-a-direct
```

**Step 3: Produce paired B comparison**

Join canonical sample IDs and report paired correctness plus Accuracy/Macro-F1/worst-user/compute/byte deltas. Do not tune from individual validation errors.

**Step 4: Test, commit, and stop**

Run `pytest tests/test_report_thermal_a_direct.py -q`; commit with `report: record direct thermal multistream student`.

## Task 12: Implement, probe, and conditionally train Route C

**Authorization gate:** Teacher construction/probing is no-training work. C1 and C2 finetuning each require separate explicit approval.

**Files:**
- Create: `src/models/thermal_teachers.py`
- Create: `tests/test_thermal_teachers.py`
- Create: `configs/experiments/thermal_c1_r2plus1d18_train12_val2.yaml`
- Create: `configs/experiments/thermal_c2_videomae_s_train12_val2.yaml`
- Create: `scripts/probe_thermal_teachers.py`
- Create: `tests/test_probe_thermal_teachers.py`
- Generate: `reports/thermal_teacher_environment_probe.json`
- Generate after training: `reports/thermal_c1_r2plus1d18_train12_val2.{md,json}`
- Conditionally generate: `reports/thermal_c2_videomae_s_train12_val2.{md,json}`

**Step 1: Write failing provenance/forward tests**

Require official source revision, license scope, checkpoint URL/SHA256, strict missing/unexpected keys, preprocessing identity, and finite logits. Deployment entry points must not import teacher modules.

**Step 2: Implement adapters and probe**

C1/C2 consume full/crop and three windows sequentially, average clip logits, and expose only trial logits. Probe physical batch 1; VideoMAE uses gradient checkpointing. A teacher exceeding 7300 MiB is ineligible; do not shrink the input contract.

**Step 3: Verify and commit no-training qualification**

Run `pytest tests/test_thermal_teachers.py tests/test_probe_thermal_teachers.py -q`; commit with `audit: qualify thermal training-only teachers`.

**Step 4: Train C1 only after approval**

Record approval in config, commit, run C1 once, and report common metrics. Leave checkpoints ignored.

**Step 5: Apply the fixed C2 rule**

- If C1 passes all three teacher gates, skip C2 and use C1 alone.
- If C1 fails any teacher gate, C2 may be proposed for separate approval.
- Record `skipped_by_rule`, `blocked_by_hardware`, or `completed`; never substitute a new family.

**Step 6: Commit reports, not weights**

Commit configs/report code/tests/reports after each authorized run and stop.

## Task 13: Export and audit train12 teacher logits

**Authorization gate:** At least one teacher must satisfy Accuracy `>=0.60`, Macro-F1 `>=0.45`, and worst-user `>=0.50`.

**Files:**
- Create: `src/distillation/thermal_teacher_logits.py`
- Create: `tests/test_thermal_teacher_logits.py`
- Create: `scripts/export_thermal_teacher_logits.py`
- Create: `tests/test_export_thermal_teacher_logits.py`
- Generate ignored: `outputs/thermal_teacher_logits/train12_logits.npz`
- Generate: `reports/thermal_teacher_logits_manifest.json`

**Step 1: Write failing schema/leakage tests**

Require every train12 ID exactly once, finite float32 `[N,40]`, teacher/checkpoint/config/code hashes, split/class/input hashes, deterministic no-TTA inference, and no validation/heldout/test/quarantined IDs. Generation 2 accepts exactly one passing teacher and forbids teacher ensembling.

**Step 2: Implement a fail-closed lookup/exporter**

`ThermalTeacherLogits` rejects missing, duplicate, or extra IDs and returns detached logits only. No feature tensors or teacher modules cross into student training.

**Step 3: Verify deterministic exports**

Export twice and require identical arrays and SHA256. Commit code/tests/manifest with `feat: freeze thermal train12 teacher logits`; keep logits ignored while recording exact path/bytes/hash.

## Task 14: Train and report A-KD

**Authorization gate:** A new human message must explicitly authorize `A-KD` after teacher/logit review.

**Files:**
- Create: `configs/experiments/thermal_a_multistream_kd_train12_val2.yaml`
- Modify: `src/train_thermal_generation2.py`
- Modify: `tests/test_train_thermal_generation2.py`
- Create: `scripts/report_thermal_a_kd.py`
- Create: `tests/test_report_thermal_a_kd.py`
- Generate ignored: `outputs/thermal_a_multistream_kd_train12_val2/`
- Generate: `reports/thermal_a_multistream_kd_train12_val2.md`
- Generate: `reports/thermal_a_multistream_kd_train12_val2.json`

**Step 1: Write failing matched-loss tests**

Test exact `T=4`, `0.5*CE + 0.5*T^2*KL`; detached teacher tensors; train12-only lookup; hard-label-only validation evaluation; and equality of A-direct/A-KD configs except route, teacher manifest, and loss.

**Step 2: Implement KD without a teacher model**

Add an isolated route strategy. Refuse mismatched logits hashes and any missing ID. Do not load teacher weights in the student process.

**Step 3: Verify implementation before approval**

Run `pytest tests/test_train_thermal_generation2.py tests/test_thermal_teacher_logits.py tests/test_report_thermal_a_kd.py -q`; only the absent-run report test may fail.

**Step 4: Approve, commit authorization, and run once**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.run_thermal_generation2 --config configs/experiments/thermal_a_multistream_kd_train12_val2.yaml --authorize-training thermal-a-kd
```

No alpha/temperature search, teacher swap, second seed, resume, or extension.

**Step 5: Report the matched A comparison**

Report paired correctness, changed predictions, recall deltas, calibration, latency/bytes, activation stability, and proof of matched initialization/data order/augmentation/optimizer.

**Step 6: Test, commit, and stop**

Run the report test and commit with `report: record distilled thermal multistream student`.

## Task 15: Decide feasibility and stop

**Files:**
- Create: `scripts/report_thermal_generation2_decision.py`
- Create: `tests/test_report_thermal_generation2_decision.py`
- Generate: `reports/thermal_generation2_feasibility_decision.md`
- Generate: `reports/thermal_generation2_feasibility_decision.json`

**Step 1: Write the failing decision test**

Require rows for MobileNet, iFormer, B, A-direct, completed teachers, and A-KD. Deployable rows report Accuracy, Macro-F1, worst-user, per-user Accuracy, zero-recall count, NLL, checkpoint/package bytes, median/p95 latency, peak memory, and policy status. Require paired B/A-direct, A-direct/A-KD, best/old-baseline, and teacher/A-KD comparisons.

**Step 2: Implement mechanical gates**

Join predictions by canonical sample ID and apply the spec without narrative overrides. Output one of `feasibility_pass_human_review_required`, `feasibility_failed_stop_without_oof`, or `incomplete_authorization_or_teacher_block`.

**Step 3: Audit deployment provenance**

Enumerate all inference assets recursively and count duplicate SHA256 once. Require `<95,000,000` bytes and reject teacher, iFormer, or frozen IR/X3D weights from the Thermal manifest.

**Step 4: Run final verification**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_report_thermal_generation2_decision.py -q
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests -q -k "not x3d_s_online_roi_parity"
D:\Anaconda\envs\pyTorch2.7\python.exe -m compileall src scripts tests
git diff --check
git status --short
```

If the frozen X3D smoke checkpoint remains absent, report exactly one deselected test; do not create or replace it.

**Step 5: Commit, push, and stop**

Commit the decision artifacts with `report: decide thermal single-modal feasibility`, push the generation-2 branch, verify local/remote SHA equality and no running trainer, and report checkpoint paths/hashes. Do not start OOF or packaging.

## Conditional next-stage handoff

- If every eligible student fails, close the program without OOF and retain the negative result.
- If a student passes, draft a separate train-14 OOF plan using `metadata/splits/train14_oof_3fold.json` while freezing architecture, inputs, objective, and tuning.
- Future formal evidence must include 40-class logits, availability, quality, `ExpertEvidence`, Accuracy, Macro-F1, worst-user, IR unique-correct/oracle-pair, bytes, and latency.
- This plan never authorizes that future work.
