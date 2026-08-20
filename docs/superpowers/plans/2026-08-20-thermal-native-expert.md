# Thermal-Native Expert Implementation Plan

> **For implementation:** STOP after Stage T0. Begin later tasks only after explicit human approval. When approved, use `superpowers:executing-plans` task-by-task in this worktree; do not dispatch agents unless the user separately authorizes delegation.

**Goal:** Build one Thermal-native 40-class expert that preserves independent Thermal timing and heat appearance, produces leakage-free shared-fold `ExpertEvidence`, complements the frozen IR/X3D expert, and keeps the exact complete inference package below 95,000,000 serialized bytes.

**Architecture:** Uniform normalized-time sampling is defined solely on each Thermal trial. The primary candidate is pretrained iFormer-T with temporal shift (TSM); pretrained MobileNetV3+TSM is the matched control, and iFormer-S+TSM is a conditional capacity upgrade. Localization remains Thermal-native with a full-frame path and label-free quality; IR frame indices and IR boxes never enter the Thermal preprocessing graph.

**Tech Stack:** Python 3.12, PyTorch 2.7, torchvision 0.22, OpenCV 4.11, NumPy, pandas, scikit-learn metrics, Ultralytics YOLO11n-pose for the audited optional locator, pytest, CUDA AMP.

---

## Frozen authority and boundary

- Baseline commit: `c42bb43091c79903e5fde5655c2846c87305895a`.
- Branch/worktree: `experiment/thermal-native-expert` at `D:\work\2026.7.14_kaggle\40class-thermal-native-expert`.
- Stage T0 authority: `reports/thermal_stage0_data_alignment_audit.md` and `.json`.
- Development split: `metadata/splits/train12_val2_user6_user7_development.json` only.
- Formal evidence split: immutable `metadata/splits/train14_oof_3fold.json` only.
- Sealed users `user4,user17,user23,user24`, their labels, competition test, and quarantined heldout evidence remain inaccessible until the program's Phase 10 gate.
- Frozen IR/X3D source, checkpoints, configs, and `ExpertEvidence` are read-only. Thermal work may compare against registered train-14 IR OOF predictions only after a Thermal recipe is frozen; it may not train, tune, regenerate, or overwrite IR/X3D.
- The complete deployment bundle, including every retained expert, shared YOLO, calibration, preprocessing weights, and fusion weights counted once, must be `<95,000,000` bytes.

## Stage T0 decision record

Stage T0 audited all 2,427 canonical train-14 trials without enumerating sealed users. Thermal has 2,299 present/decodable/usable directories and 160,125/160,125 decodable JPEGs, with zero corrupt JPEGs, five exact duplicate frames, 23 singleton trials, 34 trials of at most four frames, and 122 trials below 13 frames. All 2,299 present trials are 320x240 stable rendered pseudocolor; grayscale channel replication is rejected, while automatic temperature-range scaling cannot be established from rendered JPEGs.

IR/Thermal has 2,214 common train-14 trials and a median Thermal/IR frame-count ratio of 2.419. Thermal has independent frame indices and no shared wall-clock timestamps. On 56 trials covering 14 users, 40 classes, and five duration buckets, only 22.7% of valid motion fits simultaneously reached correlation >=0.5, absolute offset <=0.10, and scale within 0.90-1.10. Trial-level fusion is supported; frame-level temporal or spatial registration and IR-box transfer are not.

YOLO11n-pose detected a person at threshold 0.25 on 85.4% of 268 representative Thermal frames. Median confidence was 0.590 and median detected-to-detected bbox IoU was 0.826, but 126 boxes touched image edges, 39 frames were below threshold, 29 had multiple candidates, and montage review found a small-object/background false detection when the person left frame. YOLO is therefore a conditional quality-bearing locator, not a mandatory single path. Full-frame fallback is required; heat/motion context remains a label-free optional route and no detector is trained.

## Frozen candidate order

1. **iFormer-T + TSM** is the primary budget-friendly candidate.
2. **iFormer-S + TSM** may run only after official pretrained-loading provenance and provisional complete-package byte gates pass.
3. **Pretrained MobileNetV3-Small + TSM** is the matched control.
4. **VideoMamba, DART, and IR+Thermal early fusion are deferred.** They are not fallback experiments in this generation.

## Frozen Thermal sampling contract

- Independently natural-sort decodable Thermal frames.
- Assign `t_i=i/(N-1)`; a singleton trial uses `t_0=0`.
- Use 16 uniform targets `linspace(0,1,16)` and nearest Thermal source indices. This is not IR's 13-frame contract.
- Repeated indices are explicit for `N<16`; emit `unique_sampled_source_ratio` and the 16-position source-uniqueness mask. Never delete or relabel the canonical row.
- Do not select motion peaks. Motion energy is permitted only in offline alignment/localization diagnostics.
- Required label-free quality includes directory presence, decoded fraction, distinct-frame ratio, duration bucket, unique sampled-source ratio, localization route/confidence, bbox continuity, and fallback reason.

## Task 0: Human Stage T0 approval

**Files:**
- Review: `reports/thermal_stage0_data_alignment_audit.md`
- Review: `reports/thermal_stage0_data_alignment_audit.json`
- Review: `reports/thermal_stage0_montages/thermal_yolo_pose_montage_01.jpg` through `_06.jpg`

- [ ] Confirm the montage supports the conditional-locator/full-frame-fallback conclusion.
- [ ] Confirm the primary/control/conditional-upgrade order above.
- [ ] Confirm no additional Stage T0 audit is required.
- [ ] Record approval in a dated, committed amendment before any optimizer step.

Expected now: unchecked. Stage T0 ends here.

## Stage T0.5 amendment: localization-route and anomaly audit

Stage T0.5 is recorded in `reports/thermal_stage0_5_localization_route_audit.md` and its machine-readable companion. It reviewed all six T0 YOLO montage pages (56 trials / 268 uniformly sampled frames), all 122 Thermal trials below 13 frames, all four duplicate-frame trials, and all 327 IR/Thermal frame-count Tukey outliers. No sealed user, competition test, quarantined evidence, IR/X3D artifact, optimizer, or learned weight was opened or modified.

The candidate conditional chain routed 228/268 frames through Thermal YOLO context, 19 through heat/motion context, and 21 to full frame. Human review still found context clipping, low-confidence gaps, furniture/hot-object candidates, and partial-body crops. The audited heat/motion proposal was valid on 53.6% of representative trials, but only 7.1% of trials produced a selective expanded bbox below 95% of the image; 92.9% expanded to at least 95% and was effectively full-frame. It is therefore retained only as an offline label-free quality diagnostic, not an online fallback route.

The first iFormer-T+TSM and matched MobileNetV3+TSM development experiments are frozen to the simple `full_frame` view for every decodable Thermal trial. A later matched localization ablation may compare `thermal_yolo_context -> full_frame` against that baseline, but may not replace it without controlled train12/user6-user7 evidence. Current heat/motion localization must be redesigned and re-audited before any online use.

All singleton, short, duplicate, and count-asymmetric trials remain canonical. Present and decodable short trials keep `availability=True`; frame scarcity, source uniqueness, duplicate ratio, subject-visibility risk, localization confidence, route, and fallback reason are quality. Seven very low Thermal/IR ratios are partial Thermal capture/export candidates; six reverse cases are partial IR/Depth capture/export candidates. These diagnoses are not promoted to proven sensor faults without acquisition metadata.

Rendered background and endpoint diagnostics found weak auto-scale indications on 46 representative trials, moderate indications on eight, and insufficient temporal evidence on two singleton trials. They remain rendered-RGB indications only and do not establish absolute temperature-scale stability. Color jitter remains prohibited.

- [x] T0.5 montage and anomaly audit completed without training.
- [x] First development preprocessing frozen to `full_frame` with the independent 16-target Thermal sampler.
- [x] Heat/motion removed from the default online route; YOLO retained only for quality and a later matched ablation.
- [x] Human authorization to run no-training T1-A backbone qualification.
- [ ] Human authorization to enter any Stage T1 optimizer step.

## Task 1: Freeze model loading and byte feasibility

**Files:**
- Create: `scripts/probe_thermal_backbones.py`
- Create: `configs/experiments/thermal_iformer_t_tsm_train12_val2.yaml`
- Create: `configs/experiments/thermal_mobilenetv3_tsm_train12_val2.yaml`
- Create conditionally: `configs/experiments/thermal_iformer_s_tsm_train12_val2.yaml`
- Create: `reports/thermal_backbone_environment_probe.json`
- Test: `tests/test_thermal_backbone_environment.py`

- [x] Write tests that reject random or partially loaded pretrained state dictionaries, non-40-class heads, non-finite `[2,40]` forward output, unrecorded source/license/hash, and any provisional complete-package estimate `>=95,000,000` bytes.
- [x] Run the tests and verify RED because the probe and loaders do not exist.
- [x] Record the superseded source-identity error: revision `725d8e7f455b5e17be20788b9bcd6c6c505c4be0` belongs to Sail-SG's homonymous *Inception Transformer*. Its S/B/L-only finding is valid for that repository but irrelevant to this plan's candidate.
- [x] Audit the intended ICLR 2025 mobile iFormer from `ChuanyangZheng/iFormer` at revision `2a87540fcb345afe9d950a58d0eb3873b938c3dc` under the MIT source license. Pin official symbol `iFormer_t`, config `configs/iFormer_t.yaml`, release checkpoint SHA256 `7cbd778e3604694eb1a0becbf2e6a22798586f6bb46610a5c22b39880efb967e`, and prohibit random-initialization fallback. The official constructor does not load weights even when passed `pretrained=True`, so every runtime must perform the explicit hash-gated strict load.
- [x] Implement the torchvision pretrained MobileNetV3-Small control loader with source, revision, code-license scope, weight-terms caveat, URL, hash, strict-loading, and byte provenance.
- [x] Insert TSM before selected MobileNet spatial stages and after each iFormer downsample/before its native stage with `num_segments=16` and `fold_div=8`; verify shifts cross segment boundaries and never cross trial boundaries.
- [x] Inventory frozen IR/X3D, shared YOLO, current retained scope, calibration/fusion upper bound, and each Thermal candidate. Count deployable files once.
- [x] Strict-load the official `iFormer_t` checkpoint before replacing the ImageNet head; require zero missing/unexpected keys and finite `[2,16,3,224,224] -> [2,40]`. The 40-class state-dict proxy is `10,840,569` bytes and its current provisional package is `34,484,769` bytes.
- [x] Re-apply the iFormer-S conditional gate to the correct mobile family. Strict loading and budget gates pass; its state-dict proxy is `25,400,973` bytes and provisional package is `49,045,173` bytes. Its config exists only as `conditional_not_authorized`, and it cannot precede the primary/control comparison.
- [x] Run `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_thermal_backbone_environment.py -q` and require PASS.
- [x] Run `D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.probe_thermal_backbones`; iFormer-T and MobileNet produce finite `[2,40]`, while iFormer-S passes load/byte qualification only as a conditional upgrade. Stop without training.

## Task 2: Implement Thermal input and localization contracts

**Files:**
- Create: `src/data/thermal_native_dataset.py`
- Create: `src/roi/thermal_context_locator.py`
- Create: `scripts/audit_thermal_context_routes.py`
- Create: `reports/thermal_context_route_audit.md`
- Test: `tests/test_thermal_native_dataset.py`
- Test: `tests/test_thermal_context_locator.py`

- [ ] Write RED tests for 1, 2, 8, 16, 17, and 595-frame trials. Assert 16 normalized targets, Thermal-only indices, explicit repeats, deterministic validation, shared spatial augmentation across time, and unchanged canonical `sample_id`.
- [ ] Write RED tests that reject sealed users, IR frame-index fields, IR bbox fields, motion-peak sampling, frame-level pairing by list position, and silent removal of missing/singleton/short trials.
- [ ] Implement full-frame preprocessing as resize-short-side 256 then 224 crop; training uses one shared random crop and horizontal flip across all 16 frames, while validation uses a deterministic center crop. Do not use color jitter because pseudocolor-to-temperature calibration is unresolved.
- [ ] Implement audited YOLO context as an optional route only when confidence is `>=0.25`, the bbox area ratio is `>=0.05`, and the expanded box remains geometrically valid. Expand by 25% on every side and clip to Thermal bounds; never scale an IR box.
- [ ] Keep the audited heat/motion proposal as an offline quality diagnostic only. Do not place it in the dataset's default preprocessing route unless a redesigned proposal passes a separate selectivity and montage audit.
- [ ] The first primary and control datasets always use `full_frame`. For the later matched localization ablation only, route priority is `thermal_yolo_context -> full_frame`; every fallback reason is stored and full frame remains valid for every decodable trial.
- [ ] Reuse the completed T0.5 route coverage, bbox continuity, area, and montage audit. Do not train or select a detector.
- [ ] Run both new test files and require PASS. Human authorization is still required before Task 3.

## Task 3: Train the primary iFormer-T + TSM development candidate

**T1-B split freeze (2026-08-20):** Keep the original train12/user6-user7 split and its source file unchanged. The Thermal-only sidecar is `reports/thermal_train12_val2_split_sidecar_audit.json`: among all 91 two-user pairs, only user6+user7 gives 40-class Thermal-usable coverage on both validation and remaining-train12 sides. Checkpoint selection uses combined user6+user7 Macro-F1 with labels fixed to `0..39`, then Accuracy, then worst-user Accuracy, then lower epoch. Report user6/user7 Accuracy and fixed-label Macro-F1 separately without treating a user's absent classes as model zero capability. Ten validation classes have at most three usable trials; do not retune architecture from one or two such errors. Final retention remains controlled by shared train-14 OOF. The intended iFormer-T identity was resolved by corrective T1-A. Human authorization started Task 3 on 2026-08-20, then human instruction stopped it after 17 completed epochs because validation remained weak. The interrupted snapshot is recorded in `reports/thermal_iformer_t_tsm_train12_val2_interrupted_audit.{md,json}`; no automatic resume is authorized.

**T1-B.1 zero-training diagnosis (2026-08-20):** Epoch 16 was replayed without optimizer, backward, checkpoint mutation, or any new epoch. Eval-mode analytic BN replay and per-frame consensus exactly matched the existing trial-mean path. Although physical batch 4 makes simulated BN-EMA order-sensitive, the checkpoint BN matches the robust train12 trial population (median variance ratio 0.834; normalized mean error RMS 0.182), and robust population stats do not sufficiently reduce logit scale. The more important unresolved failure is upstream embedding spikes: robust flags cover 77 train12 and 23 validation trials, the largest validation trial norm is 22,947 under bfloat16 and 32,076 under FP32, and the most severe validation spikes cluster in class 36/user7. Therefore classifier BN is not confirmed as the root cause, a BN-free short run is neither recommended nor authorized, and epoch 18 remains prohibited. The next eligible work is a separate zero-training layerwise activation trace on paired normal/spike frames.

**T1-B.2 zero-training activation trace (2026-08-20):** Three fixed user7/class36 spikes were paired with same-user/class stable controls at 0, 2, and 1 frame-count difference. Full-frame 16-segment normalized-time inputs were traced in FP32 and bfloat16. Both group-median traces first crossed the preregistered 10x RMS threshold at `spatial.backbone.stages.2.5`: FP32 rose from 6.43x at block 4 to 17.04x at block 5; bfloat16 rose from 4.75x to 12.31x. Individual crossings varied from stage 2 block 4 to later blocks with spike severity, so block 5 is the first representative observed amplification boundary, not proof of a defective operator. Hooked/unhooked logits and embeddings were bitwise identical; every parameter and BN buffer, the state digest, and the checkpoint SHA remained unchanged. Stop pending human decision; epoch 18, BN-free training, head changes, and crop changes remain unauthorized.

**Files:**
- Create: `src/models/thermal_iformer_tsm.py`
- Create: `src/train_thermal_native_expert.py`
- Create: `scripts/run_thermal_train12_val2.py`
- Create: `scripts/report_thermal_train12_val2.py`
- Test: `tests/test_thermal_iformer_tsm.py`
- Test: `tests/test_train_thermal_native_expert.py`

- [ ] Write RED tests for `[B,16,3,224,224] -> main_logits[B,40]`, availability, quality/quality-mask, finite loss, trial-grouped batching, and exact train12/user6-user7 membership.
- [ ] Implement the common Thermal expert surface. TSM operates across the 16 Thermal segments; logits are trial-level and no per-frame label or loss is introduced.
- [ ] Freeze the first recipe: pretrained backbone; 40-class head; cross entropy with label smoothing 0.1; AdamW; backbone LR `3e-5`; head LR `3e-4`; weight decay `0.05`; two warmup epochs; cosine decay; 30 epochs; batch size 4; gradient accumulation 4; clipping 1.0; CUDA AMP; seed `20260715`; `num_workers=0` for the first Windows run.
- [ ] Use train12 only for weight fitting. Select the checkpoint by highest user6-user7 Macro-F1, then Accuracy, then worst-user Accuracy, then lower epoch. Fix metric labels to `0..39`.
- [ ] Report Accuracy, Macro-F1, per-user Accuracy, worst-user Accuracy, per-class recall/F1, zero-recall classes, route/quality strata, exact checkpoint bytes, GPU memory, preprocessing latency, model latency, and end-to-end trial latency.
- [ ] Run the focused tests and a one-batch CUDA smoke before the formal 30-epoch development run.

## Task 4: Run the matched MobileNetV3-Small + TSM control

**Files:**
- Modify: `src/models/thermal_iformer_tsm.py` or create a focused `src/models/thermal_mobilenet_tsm.py` if the shared class becomes unclear
- Reuse: `src/train_thermal_native_expert.py`
- Create: `reports/thermal_mobilenetv3_tsm_train12_val2.md`
- Test: `tests/test_thermal_mobilenet_tsm.py`

- [ ] Keep the split, 16-frame sampler, route policy, augmentations, optimizer groups, schedule, seed, selection rule, quality fields, and report metrics identical to Task 3.
- [ ] Change only the pretrained spatial backbone to torchvision MobileNetV3-Small and insert TSM with the same 16 segments/fold divisor.
- [ ] Report paired sample-level predictions for iFormer-T and the control, plus Accuracy/Macro-F1/worst-user deltas and exact byte/latency deltas.
- [ ] Do not promote the control merely for lower bytes; promotion requires non-inferior Macro-F1 (within 0.005), non-inferior worst-user Accuracy (within 0.01), and a passing complete-package ledger.

## Task 5: Apply the conditional iFormer-S upgrade gate

**Files:**
- Reuse conditionally: `configs/experiments/thermal_iformer_s_tsm_train12_val2.yaml`
- Create conditionally: `reports/thermal_iformer_s_tsm_train12_val2.md`
- Create: `reports/thermal_development_candidate_decision.md`

- [ ] Skip iFormer-S when Task 1 marks pretrained loading, licensing, exact bytes, or headroom ineligible. Record the skip as the completed result; do not substitute another family.
- [ ] If eligible, run iFormer-S with the exact Task 3 sampler, routes, training recipe, seed, and checkpoint rule.
- [ ] Promote iFormer-S only if it improves user6-user7 Macro-F1 by at least 0.020 or Accuracy by at least 0.020 over iFormer-T, regresses neither worst-user Accuracy nor Macro-F1 by more than 0.005 on the other measure, and leaves the provisional complete package `<95,000,000` bytes with 2,000,000 bytes headroom.
- [ ] Otherwise freeze iFormer-T, unless the matched MobileNet control passed its explicit non-inferiority rule.
- [ ] Commit one candidate-decision report before opening any formal OOF result.

## Task 6: Generate shared-fold train-14 OOF Thermal evidence

**Files:**
- Create: `configs/experiments/thermal_native_oof.yaml`
- Create: `scripts/run_thermal_native_oof.py`
- Create: `scripts/build_thermal_expert_evidence.py`
- Create: `scripts/report_thermal_oof.py`
- Test: `tests/test_thermal_oof_contract.py`
- Test: `tests/test_thermal_expert_evidence.py`

- [ ] Freeze architecture, preprocessing, localization routes, sampling, optimizer, and seed from Task 5. No formal OOF result may reopen development choices.
- [ ] Reuse exact outer user ownership and recorded inner epoch-selection users from `train14_oof_3fold.json`. Fit weights and any learned preprocessing only on each outer-fit population; the outer-validation user influences nothing upstream of its prediction.
- [ ] Select an epoch on the persisted inner users, refit the frozen recipe on all outer-train users for exactly that epoch, and predict only the outer-validation users. Concatenate exactly-once OOF ownership over all train-14 usable Thermal rows.
- [ ] Emit 40-class logits, availability, native quality, quality mask, scalar fusion-quality score, sample/user IDs, labels for `role=oof_train14`, class-map hash, model/config hashes, deployed bytes, preprocessing dependencies, and fold lineage through the existing `ExpertEvidence` contract.
- [ ] Validate that missing/unusable Thermal rows remain canonical with `availability=False`; no prediction is fabricated to fill them.
- [ ] Read the registered frozen IR train-14 OOF archive read-only and compute matched Accuracy, Macro-F1, worst-user Accuracy, error agreement, Thermal unique-correct, IR unique-correct, oracle-pair Accuracy, class-wise rescues, exact bytes, and latency.
- [ ] Do not open, regenerate, or overwrite IR heldout evidence, Thermal heldout evidence, or any competition-test path.

## Task 7: Final Stage T1 decision and stop

**Files:**
- Create: `reports/thermal_native_expert_final_decision.md`
- Update: package byte ledger used by the six-modal program

- [ ] Recompute the exact complete inference package with every retained artifact counted once. Require `<95,000,000` bytes; report both bytes and MiB.
- [ ] Record standalone and IR-complementarity evidence, worst-user behavior, per-class failures, quality/fallback strata, and latency.
- [ ] Retain Thermal only if its reproducible OOF standalone or complementary evidence justifies its bytes and the complete package passes.
- [ ] Stop before any sealed heldout label access or competition-test inference. A later Phase 10 instruction must separately authorize final train-14 fitting and structurally label-free heldout evidence generation.

## Verification commands after approval

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests\test_thermal_stage0_audit.py tests\test_thermal_backbone_environment.py tests\test_thermal_native_dataset.py tests\test_thermal_context_locator.py tests\test_thermal_iformer_tsm.py tests\test_train_thermal_native_expert.py tests\test_thermal_mobilenet_tsm.py tests\test_thermal_oof_contract.py tests\test_thermal_expert_evidence.py -q
D:\Anaconda\envs\pyTorch2.7\python.exe -m compileall -q scripts src tests
git diff --check
```

Expected: all focused tests pass, compileall exits 0, `git diff --check` exits 0, no heldout/test/evidence-quarantine path appears in runtime access logs, and the byte ledger is strictly below 95,000,000 bytes.
