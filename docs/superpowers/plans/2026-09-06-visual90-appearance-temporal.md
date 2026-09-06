# Visual90 Appearance Temporal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, as requested by the user. Steps use checkbox syntax. Do not spawn implementation agents by default.

**Goal:** Qualify inputs and resources, then run the two frozen-backbone visual teacher candidates on the fixed development split.

**Architecture:** A common four-clip/four-view evidence builder precedes independent frozen VideoMAE and DINO encoders. Trainable, masked regional/temporal fusion compares VideoMAE IR/Depth alone against an added appearance branch. Invalid input isolation happens before encoding, not after caching.

**Tech Stack:** Windows PowerShell; existing Python `D:/Anaconda/envs/PyTorch2.7/python.exe`; PyTorch, NumPy, pandas, Pillow, pytest; local VideoMAE implementation and revision-pinned official DINOv2.

**Spec:** `docs/superpowers/specs/2026-09-05-visual90-appearance-temporal-design.md`, revision v2, commit `a2bf8e5`.

## Global Constraints

- train2039 / validation388, 40 classes; only user6/user7 validation. No threefold, extra seeds, heldout4 or test.
- Only `temporal_visual` and `appearance_temporal`; no extra modalities or automatic finetuning/distillation.
- Four clips of16 frames; DINO uses positions0/5/10/15 within those clips; four independent views;224 pixels.
- New output root `outputs/visual90_appearance_temporal`; reports prefix `reports/visual90_appearance_temporal`.
- Fixed30 epochs, batch32, seed20260715; shared sample order and common initial parameters; no validation-selected epoch.
- Existing dirty Motion Attribute files are out of scope. Stage only explicit new-task paths and check every native exit code before committing.
- Geometry/continuity unverified stops formal caching. Encoder AND full-batch backward smoke precede full extraction. Do not silently relax input or resource gates.

## Task 1: Input provenance and continuous clips

**Files:** create `src/data/visual90_dataset.py`, `scripts/audit_visual90_inputs.py`, `tests/test_visual90_dataset.py`; reuse canonical index and pose cache readers without modifying legacy callers.

**Interfaces:** `select_continuous_clips(timestamps_ms, *, frame_ids=None, verified_counter_step=None)` returns indices[4,16], selected segment bounds and validity. `prepare_roi_track(boxes,width,height)` returns filled/smoothed boxes and a whole-track eligibility flag. `require_geometry(report)` rejects unverified reports. Audit CLI reads fixed config paths, outputs a source inventory and training-only geometry evidence with status unverified until inspected.

- [x] Write red tests for the initial contract slice: no source gap crossing, earliest tie, empty bins, one-frame repeats, unknown continuity rejected, internal1/3-frame interpolation,4-frame/endpoint rejection, geometry gate rejects unknown state. Pixel-encoder invariance tests remain in Task2.
  ```python
  selection = select_continuous_clips(np.array([0., 10., 100., 110., 120., 130., 140., 150.]))
  assert selection.indices.shape == (4, 16)
  assert set(selection.indices[0]) == {0, 1}
  with pytest.raises(ValueError):
      require_geometry({'status': 'geometry_unverified'})
  ```
- [x] Run `python -m pytest -q tests/test_visual90_dataset.py`; confirmed missing module failed, then11 tests passed after implementation. Audit seam added4 more red-to-green tests.
- [ ] Implement bin boundaries `floor(k*N/4)`, source breaks before sampling, longest/earliest segment and nearest-integer linspace. Fill only bounded interior gaps<=3. Smooth centers with radius2 median within the selected segment; choose one width/height large enough to cover all original valid boxes at their smoothed centers, clamp within the source image. Reject malformed/outside boxes rather than converting invalid evidence to full-frame.
- [ ] Inventory only canonical users; compute prescribed8 training examples by length quartiles; record sample/source/pose hashes, original dimensions and parsing exceptions. Inspect existing export metadata and training-only overlays. Do not approve geometry from timestamps alone.
- [ ] Run red tests green plus related legacy tests, examine real inventory. If an input hard gate fails, write a failure report, stop expensive work and request only the necessary protocol decision.
- [ ] Commit explicit task paths and progress/evidence, after `git diff --cached --check` succeeds.

## Task 2: Frozen encoders and resumable cache

**Files:** create `src/experiments/visual90_config.py`, `configs/experiments/visual90_appearance_temporal.yaml`, `src/models/visual90_encoders.py`, `scripts/cache_visual90_features.py`, `tests/test_visual90_encoders.py`, `tests/test_visual90_cache.py`.

**Interfaces:** `encode_video(clips)` -> [B,8,4,768], `encode_appearance(images)` -> [B,17,1024]; `cache_trial(evidence, encoders, provenance)` skips entire invalid clip-views. Config/provenance validation precedes load/resume. Dataset cache exposes features, per-token metadata/masks, labels and IDs separately.

- [ ] Red tests use a tiny real encoder fixture for spatial pooling order, no invalid-clip invocation, and gap-side pixel perturbation invariance; metadata changes must invalidate cache resume.
  ```python
  assert video_features.shape[1:] == (8, 4, 768)
  assert image_features.shape[1:] == (17, 1024)
  assert not any(p.requires_grad for p in encoder.parameters())
  ```
- [ ] Run new tests red. Lock official source revision/license and download hash before real extraction; validate complete tensor coverage, not permissive partial loading. Do not execute a mutable torch-hub main branch.
- [ ] Implement video pre-pool features by calling patch_embed, positional addition, all blocks, reshape8x14x14 and adaptive2x2 spatial pooling followed by fc_norm. DINO normalized patch grid16x16 pools to4x4, with CLS kept separately. Per-clip/view forward is isolated.
- [ ] Store FP16 shards with atomic completion markers and identity-bound provenance; include source-file integrity, exact sample order, times/ROI coordinates. Reject incomplete final population, duplicate IDs, or altered config/weights during resume.
- [ ] Run unit tests green; after Task1 geometry success, run prescribed8-trial real encoder smoke sequentially on GPU. Measure synchronized timings and storage estimates; stop if resource gates fail.
- [ ] Commit implementation/tests/provenance reports; do not commit weights or bulk cache.

## Task 3: Matched masked A/B fusion

**Files:** create `src/models/visual90_fusion.py`, `tests/test_visual90_fusion.py`.

**Interface:** `Visual90Fusion(appearance: bool).forward(features, masks, coordinates)` returns logits[B,40], embedding[B,256], support and residual diagnostics. No labels/users are accepted by forward.

- [ ] Red tests cover initial A/B equality, disabled Depth fallback, all-empty finite output, view/time identity, sample-varying attention, and nonzero new-branch gradient after optimizer steps.
  ```python
  torch.testing.assert_close(a(batch)['logits'], b(batch)['logits'])
  assert torch.isfinite(b(empty_batch)['logits']).all()
  ```
- [ ] Implement spec topology:128 visual tokens/clip,32x32 Depth attention per clip/view,128x272 appearance attention per clip, one local transformer, four independent view pool queries,16 view/clip tokens plus trial query through one temporal transformer. Zero-init output projections; residual bounded with tanh, scale0.1 initially fixed for both auxiliary paths. Explicit validity masking precedes every attention/pool; empty groups bypass softmax.
- [ ] Preserve common-module initialization by constructing the same shared trunk before an independently seeded B-only branch. Use shared input token masks; attention/internal dropout disabled in both branches for the first protocol, retaining the specified input token dropout0.10. Mixed-precision forward uses BF16 on the qualified device, float32 loss/normalization.
- [ ] Run tests red-to-green. Test perturbing coordinates and time identity affects actual model outputs, not only tensor shapes. Commit only new model/test files.

## Task 4: Sampling, loss, deterministic recovery and decisions

**Files:** create `src/training/visual90_training.py`, `scripts/run_visual90_experiment.py`, `tests/test_visual90_training.py`.

**Interfaces:** `paired_epoch_indices(labels,users,seed,epoch)` -> deterministic index vector; `cross_user_supcon(z,labels,users,ids)` -> scalar; `decide_results(a,b)` -> absolute/incremental discussion flags; runner handles train, resume, eval and reports.

- [ ] Red tests verify exact SupCon hand examples from spec6.1, duplicate-ID invariance, no-positive differentiable zero; validation users rejected before sampler; resume yields identical next-batch order and final tiny-run parameters.
  ```python
  assert decide_results({'correct':350}, {'correct':354})['target_reached']
  loss = cross_user_supcon(z, labels, users, ids)
  loss.backward()
  assert torch.isfinite(z.grad).all()
  ```
- [ ] Implement fixed class/user pair sampling, first-ID-only contrastive sets, CE across physical examples, loss weights and stable logsumexp. AdamW3e-4/0.05,2-epoch warmup then cosine through epoch30, gradient clip5. Save model/optimizer/scheduler and Python/NumPy/Torch/CUDA RNG, exact sampler state, hashes and progress atomically.
- [ ] Keep train-only prior for unsupported canonical rows. Final unaugmented eval includes all2039/388, fixed40-class macro metrics, per-user and paired rescue/harm. Keep all checkpoint selection rules independent from validation until both candidates finish.
- [ ] Run tiny real training/recovery tests green, corruption tests, changed-population/config rejection. Commit explicit new files.

## Task 5: Training resource qualification and batch audit

**Files:** extend runner's `--smoke` mode; create `tests/test_visual90_smoke.py`; generated report `reports/visual90_appearance_temporal_smoke.json`.

- [ ] Red test ensures formal mode rejects a missing/failed input/encoder/training gate report and never silently lowers batch or token count.
- [ ] Implement gate check before launch:
  ```python
  if not all(gates[name]['passed'] for name in ('inputs','geometry','encoders','training')):
      raise RuntimeError('visual90 qualification incomplete')
  ```
- [ ] Run full-layout batch32 A/B pressure smoke (warmup plus3 measured forward/backward/AdamW steps), then real small-cache gradient smoke. Record CUDA allocated/reserved/free, finite losses/gradients, state changes and timing. Discard smoke state.
- [ ] Audit all task changes against spec and tests, run `python -m pytest -q`, compileall new modules and `git diff --check`. Report unrelated baseline failures honestly; do not patch old experiments incidentally.
- [ ] Proceed only if allocated<7300MiB, no OOM, cache+temporary disk estimate<70% available, extraction estimate<=12h, all data/source gates passed. Commit qualification evidence and reviewed implementation.

## Task 6: Background cache, fixed A/B training and final report

- [ ] Launch cache process with hidden window, exact interpreter, source/config hashes and isolated stdout/stderr/state files. Then launch runner only on complete verified cache. Persist PID/start-time/run identity and do not treat a stale PID as proof of liveness.
- [ ] Runner executes A then B without changing B after seeing A validation. Resume through recorded state only. Nonfinite tensors or exceptions create failure status and stop; no automated recipe changes.
- [ ] Completion outputs: cache provenance, latest A/B checkpoints, full train/validation predictions, paired per-user/per-class metrics and decision report. Verify IDs, hashes and counts by independent recomputation.
- [ ] Update plan checkboxes, commit source and compact reports only, provide result and temporary handoff. Do not push, run a third candidate, or launch distillation without user direction.

## Progress log

- 2026-09-06: plan created from approved v2; existing worktree/branch preserved. Task1 starts with data preflight. No formal training started.
- 2026-09-06: source inventory completed over2039/388;22 training trials have numeric-only legacy frame names without verified continuity evidence. Spec3.1 hard gate blocked; Task1 remains partial, geometry/resource qualification and Tasks2–6 not started. See `reports/visual90_appearance_temporal_input_preflight.md`; input-support amendment requires user direction before continuing.
