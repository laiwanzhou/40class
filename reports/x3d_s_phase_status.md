# X3D-S Execution Phase Status

Updated: 2026-08-13 (Asia/Shanghai)

| Phase | Status | Git SHA | Evidence | Risks | Next decision |
|---|---|---|---|---|---|
| Phase 0: Compliance and runtime | Completed | `c088db0` | Official rules rechecked; official X3D forward, provisional IR-route subtotal, dependency check and 15 focused/regression tests passed | X3D-S has no model-specific organizer approval; final trained checkpoint must be remeasured | Phase 1 may begin |
| Phase 1: Temporal data contract | Completed | `481ccb4` | 23 dataset tests, 30 focused/regression tests, and all 74 repository tests passed; real shortest/longest trial probe passed | Exported quality fields are constant in this manifest, so they are contract metadata rather than discriminative evidence in the first run | Phase 2 may begin after review |
| Phase 2: Expert and trainer | Completed | `1601a31` | Trial-level trainer, dual checkpoint archives, fixed BN policy, train-14 finalization, resource audit and held-out rejection implemented; 24 focused and 94 full-suite tests passed | Real CUDA/data smoke is intentionally deferred to Phase 3 | Begin Phase 3 Task 5 |
| Phase 3: End-to-end verification | Completed | `dc13b96` | Revised six-trial parity, fixed-checkpoint sensitivity, CUDA/overfit/size audits, 65 focused tests, and 107 full-suite tests pass | Latency is representative smoke evidence, not a production benchmark | Begin Phase 4 Task 6 |
| Phase 4: Train-14 OOF scientific evaluation | Completed under compute amendment | `a1a2fa2` | Three X3D seeds, nine-checkpoint exact regeneration, canonical diagnostics, and fixed fold-0 x 10-epoch matched sanity are complete; X3D is competition-retained | Seed17 fold1 used the approved 28/30-epoch recovery; original full matched three-fold primary rule and paired CI were not evaluated; `>64` bucket remains weak | Begin Phase 5 evidence registration |
| Phase 5: Register IR sparse evidence | Completed; IR/X3D frozen | `139fc3c`; freeze diagnostic `de76bd5` | Canonical evidence is unchanged; final pure-inference diagnostic measured deterministic train_eval 0.965733/0.964861 versus formal OOF 0.565517/0.480786 | Held-out archive remains evaluation-only and quarantined until Phase 10; cross-subject gaps are 0.400216 Accuracy and 0.484075 Macro-F1 | No new IR single-modality training or tuning without explicit approval |
| Phase 6: Freeze expert portfolio | Pending | - | Every retained expert must emit global OOF plus label-free held-out evidence | Historical metrics use mixed folds; outer four users are unavailable for selection | Execute later on a dedicated branch after Phase 5 review |
| Phase 7: Build sparse evidence registry | Pending | - | Global registries plus nested outer-fold fusion evidence | Base-level global OOF alone is insufficient for unbiased stacker CV | Wait for Phase 6 |
| Phase 8: Fit safe anchor | Pending | - | Program-level roadmap only | Calibration must use train-14 OOF evidence only | Wait for Phase 7 |
| Phase 9: Test residual correction | Pending | - | Program-level roadmap only | Rare modality combinations must not dominate training | Wait for Phase 8 |
| Phase 10: Assemble final inference | Pending | - | Program-level roadmap only | The four held-out users may be evaluated only once after freeze | Wait for Phase 9 |

## Frozen Program Context

- Approved design: `docs/superpowers/specs/2026-08-11-six-modal-sparse-evidence-fusion-design.md` (`0af881e`).
- Authoritative execution plan: `docs/superpowers/plans/2026-08-10-x3d-s-adaptive-multiclip.md`.
- X3D-S is the IR specialist, not the complete six-modal model.
- Preserve route diversity: this branch is fixed around YOLO-guided person context, adaptive local temporal decomposition, and X3D 3D spatiotemporal encoding. Teammate-specific architectures or distillation recipes are not branch inputs.
- Final architecture: heterogeneous experts -> canonical sparse OOF registry -> calibrated available-expert probability anchor -> optional tiny zero-initialized residual set mixer -> trial probability.
- The canonical population is the 3,036-trial union. `present`, `usable`, and label-free `quality` are separate states; fusion uses `usable`.
- The anchor remains a complete deployable fallback. One usable expert returns that expert's calibrated probability exactly; zero usable experts is an explicit routed failure.
- YOLO11n-pose is verified as the IR locator. IR/Depth are timestamp-aligned in the audited export, while Thermal has independent frame numbering and no demonstrated common timestamps.
- Skeleton/YOLO diagnostics support sequence-level evidence fusion, not frame-level or joint-level hard fusion.
- Historical expert scores use mixed folds and are context only until regenerated on the canonical split and common train-14 OOF assignment.
- Teacher-assisted X3D is an optional separately pre-registered candidate, never an automatic Phase 4B. It must use distinct evidence identity and may not overwrite the pure-X3D archive.
- `train14_oof_3fold.json` is generated and hashed once in Phase 4. Phases 5-9 may verify and reuse it but may never regenerate it.
- Held-out/test `ExpertEvidence` and registries are structurally label-free. Phase 10 joins predictions to a separate sealed held-out label source.
- Phase 8 reports cross-fitted A; Phase 9 cross-fits A and D on identical outer user folds with residual selection inside outer-train users only.
- Every expert exposes native quality plus a pre-registered label-free scalar `fusion_quality_score` in `[0,1]`; fusion never directly averages heterogeneous native quality coordinates.
- Pure X3D final train-14 duration is the median of the nine Phase 4 selected best-Accuracy epochs; its finalization seed is `20260715`.
- Unbiased Phase 8/9 evaluation requires outer-fold-local base evidence: inner OOF on outer-train users and base experts finalized on outer-train only for outer-validation prediction.
- Missing-pattern support for `g(A,Q)` is computed from each outer-train package; complete train-14 support is used only for final refit.
- Phase 4 complementary retention freezes worst-user Accuracy delta at `>= -0.02` relative to the matched baseline. Any seed or aggregate below `-0.02` triggers a mandatory human-review stop with all checkpoints, predictions, logs, histories, manifests, and audit artifacts preserved; no automatic deletion, rejection, registration, or Phase 5 continuation is allowed.
- Phase 4 formal evidence uses strict checkpoint-selection OOF: epoch selection occurs only inside outer-train, a fresh model refits all outer-train users for the selected epoch, and untouched outer-validation is evaluated once. CLI `--seed` must propagate into resolved config, summaries, manifests, and hashes. Seed `20260715` is the immutable Phase 5 canonical OOF archive; `20260716/17` are stability-only.
- Formal inner epoch selection runs all 30 epochs without early stopping. Formal refit stops at the selected epoch but retains the identical prefix of the fixed 30-epoch cosine schedule through the separate `scheduler_horizon_epochs=30` field.
- Final route-freeze diagnostic: `reports/x3d_s_ir_train_vs_oof_generalization.md`. It performs deterministic eval-mode inference on each canonical formal checkpoint's own outer-train users and compares it with saved outer-val OOF. It does not access heldout4/test or alter evidence. The IR/X3D single-modality route is frozen after this diagnostic; further training, tuning, matched baselines, or ablations require explicit approval.
- Per-fold validation 40-class coverage is impossible because train-14 class 25 occurs only for `user1` and `user7`. The frozen fold gate therefore requires every outer-train partition and the concatenated OOF population to cover all 40 classes, reports each fold's missing classes, and always computes metrics with the fixed 40-class label set.

## Phase 0 Evidence Log

### Commands

- `D:\Anaconda\envs\pyTorch2.7\python.exe -c "import sys, torch, torchvision; ..."`
- `Get-FileHash -Algorithm SHA256 D:\work\2026.7.14_kaggle\40class\yolo11n-pose.pt`
- `D:\Anaconda\envs\pyTorch2.7\python.exe -m pip install -r requirements-x3d.txt`
- `D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.probe_x3d_s_environment --output reports/x3d_s_environment_probe.json --pip-log <log> --pip-exit-code 1`
- `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_environment_contract.py -v`
- `D:\Anaconda\envs\pyTorch2.7\python.exe -m pip check`
- `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_environment_contract.py tests/test_expert_contract.py tests/test_ir_primary_input_builder.py tests/test_ir_primary_variable_sequence_dataset.py -v`
- `git diff --check`

### Artifacts

- `docs/x3d_s_rule_compliance.md`
- `requirements-x3d.txt`
- `scripts/probe_x3d_s_environment.py`
- `tests/test_x3d_s_environment_contract.py`
- `reports/x3d_s_environment_probe.json`

### Current Findings

- Python 3.12.9, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128.
- CUDA is available on NVIDIA GeForce RTX 5060 Laptop GPU.
- PyTorchVideo 0.1.5 installed successfully and imports under PyTorch 2.7.0.
- The pip PowerShell pipeline returned 1 because warnings were emitted to stderr; the log ends with successful installation and the installed version is independently verified as 0.1.5.
- Official X3D-S produced finite `[1,400]` output for `[1,3,13,182,182]` on CUDA.
- X3D-S source checkpoint: 3,794,274 parameters, 30,779,313 bytes, SHA-256 `26b95f1605d49650b54049db40ba3a56e023b86b58c3b3e0e10e0992a9c8682f`.
- YOLO11n-pose: 2,874,462 parameters, 6,255,593 bytes, SHA-256 `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`.
- Estimated custom head: 535,336 parameters and 2,144,149 serialized bytes.
- Provisional IR-route deployment subtotal: 39,179,055 bytes of the 95,000,000-byte internal limit; route gate passed. This is not the complete six-modal package.
- Peak CUDA memory during the one-sample X3D probe: 80,395,264 bytes.

### Exit Gate

Passed on 2026-08-10. PyTorchVideo imports cleanly, `pip check` reports no broken requirements, the official CUDA forward is finite with the fixed input shape, the conservative aggregate is below the internal limit, all 15 focused/regression tests pass, and implementation evidence is committed as `c088db0`.

## Phase 1 Evidence Log

### Read-Only Manifest Audit

- Source: `D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\roi640_depth_ordinal_256\combined_frame_manifest.csv`
- 84,906 frame rows, 2,910 unique trials, and class IDs 0 through 39.
- Train: 67,216 frames, 2,320 trials, 14 users. Validation: 17,690 frames, 590 trials, 4 users.
- Train and validation have no user or `sample_id` overlap.
- Required columns contain no nulls; `(sample_id, source_frame_index)` is unique.
- Every trial begins at source index zero and has a complete contiguous index range.
- Selected `ir_context_path` values are absolute and exist.
- `temporal_valid`, `ir_context_effective_valid`, and `ir_context_reliability` are all 1 for every row.

### Implementation and Verification

- RED: `tests/test_x3d_clip_dataset.py` failed during collection because `src.data.x3d_clip_dataset` did not exist.
- GREEN: all 23 adaptive dataset tests passed after implementing strict indexing, adaptive windowing, stratified sampling, temporally consistent transforms, variable-clip collation, and fusion metadata.
- Real manifest construction produced 2,320 train trials and 590 validation trials, preserving 14/4 users.
- The one-frame trial emitted `[1,1,3,13,182,182]`; the 236-frame trial emitted `[8,1,3,13,182,182]` with contiguous bounds `[[0,30], ..., [207,236]]`.
- Repeated access to the longest validation trial produced identical clips and source indices.
- `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_clip_dataset.py tests/test_expert_contract.py tests/test_ir_primary_variable_sequence_dataset.py -v`: 30 passed.
- `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest -q`: 74 passed.
- `python -m compileall` and `git diff --check` passed.

### Artifacts

- `src/data/x3d_clip_dataset.py`
- `tests/test_x3d_clip_dataset.py`
- Implementation commit: `481ccb4 Add trial-safe adaptive X3D clip dataset`

### Exit Gate

Passed on 2026-08-10. Boundary lengths 1, 13, 32, 33, 64, 65, and 236 are covered; window union, deterministic validation, epoch-seeded training views, consistent spatial transforms, split leakage rejection, variable-`K` padding, metadata, and real shortest/longest trial behavior are verified. Stage 2 was not started.

## Phase 2 Evidence Log

### Task 3: X3D-S Visual Expert Wrapper

- RED confirmed: `tests/test_x3d_s_visual_expert.py` failed because `src.models.x3d_s_visual_expert` did not exist.
- Implemented official PyTorchVideo X3D-S feature construction by removing the Kinetics projection while retaining the 2,048-dimensional pooled feature path.
- Implemented valid-clip-only forwarding, masked mean probability aggregation, log trial probabilities, masked mean plus L2-normalized embeddings, and the existing `ExpertOutput` contract.
- Padded clip values are never forwarded and cannot alter outputs; trials with zero valid clips fail explicitly.
- Implemented backbone/head learning-rate groups with zero decay for bias and normalization parameters.
- First-run BN policy is explicit: K400 running statistics remain frozen after `model.train()` and backbone unfreeze, while BN affine parameters become trainable with the backbone.
- Verification: 6 wrapper tests passed; wrapper plus expert-contract tests passed 10/10; full repository suite passed 80/80; compileall and `git diff --check` passed.
- Implementation commit: `cc5bc26 Add fusion-compatible X3D-S visual expert`.

### Task 4: Training and Evaluation Entry Point

- RED confirmed for the absent trainer module, batch epoch runner, partition trainer, and fixed-epoch train-14 finalizer before each implementation layer.
- Implemented deterministic clip-budget batching with at most two trials/eight valid clips, trial-level NLL, BF16 autocast, accumulation four, clipping, warmup/cosine scheduling, and head-only epochs 1-2.
- Validation emits exactly one row per trial and saves independent best-Accuracy and best-Macro-F1 checkpoints, archives, and per-class reports; primary OOF selection is fixed to best Accuracy.
- CLI rejects held-out users and output overwrite, consumes frozen three-fold assignments, and exposes a separate `finalize_train14` path requiring explicit epochs, all train-14 users, and no validation labels.
- Run summaries record scientific metrics, throughput, length-bucket latency, peak CUDA allocation, parameter/state bytes and hashes, source provenance, YOLO/X3D weight evidence, and the provisional IR-route size gate.
- Verification: 14 trainer tests and 24 trainer/wrapper/expert-contract tests passed; full repository suite passed 94/94; both YAML configurations validated; CLI help, compileall, and `git diff --check` passed.
- Implementation commit: `1601a31 Add adaptive multi-clip X3D-S training pipeline`.

### Exit Gate

Passed on 2026-08-11. Task 3 and Task 4 jointly satisfy the Phase 2 masked aggregation, gradient, archive, partition, BN, resource-recording, and focused-test contracts. Real CUDA/data behavior remains the explicit Phase 3 gate.

## Phase 3 Evidence Log

- Real-manifest read-only contract passed on 84,906 frames: 40 classes, disjoint users/sample IDs, contiguous ordering, no competition-test paths, exact adaptive window coverage, deterministic validation, and boundary shapes for 1/236 frames.
- Three-epoch CUDA smoke used two train/two validation trials. Epoch 3 had nonzero learning rates after correcting a cosine boundary bug; backbone and head both received finite nonzero gradients. Checkpoints reloaded and both prediction archives passed shape/uniqueness/finite checks.
- Sixteen-trial implementation overfit covered eight classes plus one-/multi-clip trials with augmentation disabled. Loss fell from 3.6859 to 0.00413 and final training Accuracy was 1.0; this is explicitly not scientific evidence.
- Provisional route audit passed: trained X3D+head checkpoint 14,388,607 bytes plus YOLO 6,255,593 bytes = 20,644,200 / 95,000,000 bytes. The head is embedded and not double-counted; class-map, archive completeness, shuffled alignment and alpha-zero recovery passed.
- Online inference now reuses `UltralyticsPoseLocator`, the historically frozen `IRPrimaryInputROIBuilder` parameters, adaptive windows and X3D normalization. It forbids silent full-frame fallback and handles low-confidence recovery at 0.01.
- The original exact-box/max-pixel parity definition was rejected as statistically inappropriate after correcting ROI configuration drift. Four shorter representatives produced byte-identical crops and normalized clips, while fresh-GPU-YOLO subpixel drift crossed PIL crop resampling boundaries on both available train eight-clip trials. The 231-frame trial had max box drift 0.048 px, crop MAE 0.0316/255, P99 1/255, PSNR 52.58 dB, and max pixel error 168; the 236-frame trial had max box drift 0.920 px, crop MAE 0.5757/255, P99 7/255, PSNR 43.09 dB, and max pixel error 148. The large maxima were edge-localized resampling effects rather than semantic ROI displacement.
- The approved frozen replacement keeps frame order, clip counts, windows, sampled indices, routing/recovery, and normalization exact; forbids silent full-frame fallback; and gates spatial parity on box drift <=1 px, crop MAE <=1/255, P99 <=8/255, PSNR >=40 dB, and worst-frame crop MAE <=2/255. A fixed-checkpoint embedding/probability/top-1 sensitivity audit is diagnostic and carries no post-hoc pass threshold. Step 6 remains in progress until all six representatives are rerun under this contract.
- Step 6 passed on all six representatives with clip counts `[1,1,2,4,8,8]`. Aggregate worst cases were bbox drift 0.9204 px, crop MAE 0.5757/255, crop P99 7/255, minimum PSNR 43.09 dB, and worst-frame MAE 1.6682/255. Frame order, windows, sampled indices, recovery path, shared normalization, and no-fallback gates all passed.
- Fixed checkpoint SHA-256 `77ed27ddb9a09edf2c675cdcfbe4e052bacb11a0e6b9f1ab1582108609404039` was used for model sensitivity. The largest observed change was on the 236-frame trial: embedding cosine 0.998655, probability L1 0.014518, JS divergence 0.00004112, and max class-probability delta 0.001600. Top-1 agreed on all six trials; no sensitivity thresholds were applied.
- Step 7 focused verification passed 65/65 tests. The first full-suite run exposed Ultralytics import-time replacement of global OpenCV I/O functions; lazy YOLO loading plus restoration of `cv2.imread/imwrite/imshow` and defensive singleton-channel grayscale reads fixed the process-wide side effect. The targeted regression passed 7/7 tests and the complete repository suite then passed 107/107 tests. Compileall, CLI help, and `git diff --check` also passed.

### Exit Gate

Passed on 2026-08-11. All Task 5 steps are complete and the end-to-end implementation/evidence is committed as `dc13b96`. Phase 4 may begin under the frozen train-14 OOF contract without held-out access.

## Phase 4 Evidence Log

- Review fixes were pre-registered before formal results: runtime `--seed` propagation, strict checkpoint-selection OOF, and canonical Phase 5 evidence seed `20260715`; seeds `20260716/17` are stability-only.
- Formal outer fold assignment was generated once from the 2,427-trial canonical train-14 union with `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20260715)`. The usable IR population contains 2,320 trials.
- Assignment SHA-256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`. Every user owns exactly one outer-validation fold, every outer-train and inner-fit partition covers all 40 classes, and concatenated OOF covers all 40 classes.
- Outer-validation class counts are `[39,40,40]`; fold 0 lacks class 25 because that class occurs for only two train-14 users. Metrics retain the fixed 40-class label set.
- Phase 4 experiment manifest records code SHA `a1663bb60bebdd44047db2810e066cddbbf7a805`, actual seed-specific resolved config hashes, config/data/class-map/weights/rules/environment hashes, no held-out archive access, and a provisional IR route of 20,644,200 / 95,000,000 bytes.
- Write-once retry correctly fails with `FileExistsError`.
- The original seed-20260715 run was interrupted during fold-0 inner epoch selection after epoch 4 when review found that formal refit shortened the cosine horizon to the selected epoch. No `formal_outer_refit.pt` or formal prediction exists and outer-validation was not accessed. Its files are preserved with `ABORTED_PRE_REFIT.json`; the run cannot be resumed or overwritten.
- The corrected protocol separates `training.epochs` from `training.scheduler_horizon_epochs`, keeps the latter fixed at 30 during refit, disables early stopping for the full 30-epoch inner search, emits the Plan duration buckets `<=13`, `14-32`, `33-64`, and `>64`, and records the scheduler horizon in summaries and formal checkpoint provenance.
- Frozen usable-IR inner coverage audit SHA-256: `3cfbc0717948d914e0c5f98682d3fee362bc3458f7d7bcb2f568387f41788105`, bound to assignment SHA-256 `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`. Inner-fit counts are 1,043 / 1,212 / 937 trials and all three cover 40 classes; the immutable assignment therefore remains unchanged.
- TDD verification for the amendment passed 34 focused tests; the complete repository suite passed 123 tests in 58.01 seconds. Compileall and `git diff --check` passed.
- The replacement protocol is frozen in `reports/x3d_s_phase4_protocol_amendment.json`; it binds commit `335f5e2`, all three resolved-config hashes, the unchanged assignment, the IR audit, and the preserved aborted run before the replacement run starts.
- The strict-v2 launch stopped before epoch 1 because PowerShell promoted a CuBLAS warning to `NativeCommandError`; its empty run shell is preserved. The launch addendum fixes `CUBLAS_WORKSPACE_CONFIG=:4096:8` and records that no scientific protocol or result changed.
- The strict-v2b launch then reached fold-0 inner epoch 6 before review identified variable-microbatch weighting: accumulating four microbatch-mean losses weights a one-trial batch twice as much per trial as a two-trial batch. The process was interrupted immediately; there is no formal refit or formal prediction and outer-validation was not accessed. History SHA-256 is `58cb064dbca9dc674bcf4662e2b215bc50b7a6ee363eb72b2c1f3a650a1740b5`.
- The corrected equal-trial objective backpropagates summed trial NLL per microbatch, divides accumulated gradients by the optimizer window's actual trial count, then clips and steps. The same summed trial NLL supplies the reported mean loss, so logged and optimized objectives agree. The `1+2+2+2` versus seven-trial full-batch regression test passes; 35 focused tests and the complete 124-test suite pass, along with compileall and `git diff --check`.
- The correction is frozen before restart in `reports/x3d_s_phase4_trial_weighting_addendum.json`, bound to implementation commit `b95e18f` and strict-v3 run ID.
- Canonical seed `20260715` completed 2,320/2,320 usable-IR trials at Accuracy `0.565517`, Macro-F1 `0.480786`, and worst-user Accuracy `0.420690` (`user1`). Stability seed `20260716` achieved `0.569397 / 0.489552 / 0.448276`. Recovery-qualified seed `20260717` achieved `0.550000 / 0.477880 / 0.400000`; its fold1 epoch 10 was human-approved after the inner search stopped at 28/30 epochs.
- Across all three seeds, mean Accuracy is `0.561638` (sample SD `0.010264`) and mean Macro-F1 is `0.482739` (sample SD `0.006077`). Seed `20260715` remains the canonical Phase 5 archive regardless of the higher seed16 result.
- `reports/x3d_s_phase4_checkpoint_reverification.json` records exact regeneration for all nine formal checkpoints: every saved array and metric reproduced with maximum delta `0.0` under the frozen assignment SHA `2a0dde67...`.
- The initially preregistered complete MobileNet/TCN matched run was stopped after fold0 inner epoch 4, before formal refit or outer-validation prediction. The user-approved competition-compute amendment is recorded in `reports/mobilenet_tcn_ir_context_sanity_amendment.json`; the abandoned run is not treated as completed evidence.
- The replacement fresh fold0 fixed-10-epoch sanity check produced Accuracy `0.283750`, Macro-F1 `0.173536`, and worst-user Accuracy `0.237569` on the same 800 outer-validation trials. Canonical X3D fold0 produced `0.571250 / 0.486918 / 0.533835`, leading by `0.287500 / 0.313382 / 0.296266`. The baseline is far below the frozen `0.53` anomaly threshold, so no fold1/2 baseline or paired bootstrap was run.
- Phase 4 decision: `competition-retained; full matched primary rule not evaluated`. This is an explicit compute-budget competition decision, not a claim that the original three-fold paired-bootstrap primary criterion passed. No held-out-4 labels or predictions were accessed.
- Canonical duration Accuracy is `0.540541` (`<=13`), `0.587364` (`14-32`), `0.575503` (`33-64`), and `0.474359` (`>64`). The weak longest bucket remains a deferred temporal-follow-up risk but does not block Phase 5 registration.

## Post-Phase-5 Shared Development Split

- New tuning no longer uses the former X3D fold0 population. The shared development split is frozen in `metadata/splits/train12_val2_development.json`: 1,996 usable-IR trials from 12 train users and 324 usable-IR trials from validation users `user21,user22`.
- Validation observes 36 classes and misses class IDs `25,26,33,35`; every report still computes Macro-F1 over fixed labels `0..39`.
- This 12/2 split is the common iterative development and hyperparameter-selection population for later modality experts. It does not replace strict train-14 OOF evidence, nested fusion evaluation, or the sealed heldout-4 gate.
- The first full-temporal-coverage partial-backbone X3D-S candidate trained only the final two backbone blocks after warmup. Its best-Accuracy checkpoint was epoch 10 at Accuracy `0.552469`, fixed-40 Macro-F1 `0.418651`, and worst-user Accuracy `0.458647`; train Accuracy was `0.889780`, leaving a `0.337310` train-to-validation gap.
- The result is frozen as standalone development evidence. Because there is no matched full-backbone run on the same 12/2 split, it does not causally establish that partial unfreezing is better or worse than full unfreezing. Canonical Phase 4/5 evidence remains unchanged.
- A matched partial1 capacity ablation then changed only `unfrozen_backbone_blocks: 2 -> 1`, retaining full temporal coverage and every other partial2 setting. CUDA smoke verified exactly 968,544 trainable backbone parameters, finite block5/head gradients, and unchanged canonical hashes.
- Partial1 completed 20 epochs and selected epoch 13 at Accuracy `0.503086`, fixed-40 Macro-F1 `0.399940`, and worst-user Accuracy `0.451128`. Relative to partial2 this is `-0.049383 / -0.018711 / -0.007519`; it crosses the pre-registered greater-than-two-point Accuracy regression floor and is frozen as `human_review_regression` with all artifacts preserved.
- Partial1 regressed in both validation users and all four duration buckets. It disagreed with partial2 on 94/324 trials, with 14 partial1-only correct versus 30 partial2-only correct; validation NLL worsened from `1.655127` to `1.954141`. Freezing block4 therefore removes useful transferable adaptation without preventing block5/head from fitting train users. No subsequent experiment is authorized automatically.
- After explicit user approval, the local partial1 formal and smoke output directories were deleted (`58,300,105` bytes total). The committed config, preregistration, JSON/Markdown reports, metrics, checkpoint/prediction hashes, and deletion audit remain versioned in the repository; canonical Phase 4/5 and partial2 artifacts were not deleted.
- The explicitly approved layer-wise LR1 experiment retained both block4 and block5, all temporal windows, and the partial2 data/augmentation/loss/BN/scheduler contract. Its only optimization intervention was block4 LR `3e-6`, block5 LR `1e-5`, and custom-head LR `3e-4`. Pre-result implementation and preregistration were pushed as `25c007b`; active-scope LR logging was corrected and pushed as `81953b5` before formal training.
- CUDA smoke verified the intended active learning rates, 2,315,984 trainable backbone parameters, finite block4/block5/head gradients, full temporal keep fraction, and unchanged canonical hashes. Formal run `x3d_s_ir_context_train12_val2_layerwise_lr1_seed20260715` completed all 20 epochs in `2273.12` seconds and selected epoch 17.
- Independent matched metrics are Accuracy `0.521604938`, fixed-40 Macro-F1 `0.409073639`, and worst-user Accuracy `0.451127820` (`user21`). Relative to partial2, the deltas are `-0.030864198 / -0.009577860 / -0.007518797`; train Accuracy at the selected epoch is `0.937875752`, leaving a `0.416270813` train-to-validation gap. The pre-registered greater-than-two-point Accuracy regression floor is crossed, so the frozen decision is `human_review_regression`.
- User21 changes by `-0.007519` and user22 by `-0.047120`. Duration Accuracy deltas are `-0.067797` (`<=13`), `-0.027586` (`14-32`), `+0.063830` (`33-64`), and `-0.071429` (`>64`). The candidate and partial2 disagree on 89/324 trials, with 15 layerwise-only correct versus 25 partial2-only correct; NLL worsens from `1.655127` to `1.802232`.
- Relative L2 drift from the same seeded K400 initialization confirms that the intervention operated as intended: block4 drift falls from `0.003632` to `0.000618` (ratio `0.170`) and block5 from `0.008501` to `0.004070` (ratio `0.479`). However, embedding-head drift rises from `0.471703` to `0.522514` and classifier drift from `1.027802` to `1.166021`. Backbone drift alone is therefore not the dominant overfitting mechanism; reducing only backbone LR shifts or leaves substantial memorization in the custom head.
- Preserve the complete layerwise_lr1 smoke and formal artifacts for human review. Do not automatically launch backbone-only L2-SP or any further IR experiment. Canonical Phase 4/5 evidence and sealed heldout4/test remain untouched. Authoritative result: `reports/x3d_s_train12_val2_layerwise_lr1_report.{json,md}`.
- Fresh result verification: report regeneration was byte-deterministic, report/preregistration/deletion-audit JSON parsed, partial1 output directories remained absent, no layerwise training process remained, compileall passed, `git diff --check` passed, and the complete repository suite passed 189 tests in 77.22 seconds.

## User6/User7 Matched Reference Generation

- Generation R is preregistered for the explicitly approved `12 train / user6-user7 val / 4 heldout` development split. The 1,935 usable-IR train trials and 385 validation trials each cover all 40 classes; heldout users `user4,user17,user23,user24` remain sealed.
- The run will repeat the unchanged partial2 architecture and recipe at seed `20260715`. Its best-Accuracy checkpoint and deterministic validation prediction will become the sole matched reference for Direct-Head.
- Direct-Head implementation/result access, heldout4/test access, recipe changes, extra seeds, canonical Phase 4/5 mutation, and automatic artifact deletion are forbidden during Generation R.
- Pre-result run ID: `x3d_s_ir_context_train12_val2_user6_user7_partial2_seed20260715`. Status: preregistered, no result.
- Generation R completed all 20 frozen epochs in `2360.57` seconds. The best-Accuracy checkpoint is epoch 14 at Accuracy `0.532467532`, fixed-40 Macro-F1 `0.421574625`, and worst-user Accuracy `0.532338308` (`user6`). User7 Accuracy is `0.532608696`.
- Training-log Accuracy at epoch 14 is `0.952454780`, leaving a train-to-validation Accuracy gap of `0.419987248`. The result confirms severe cross-user overfitting but is retained unchanged as the required matched reference.
- The frozen Direct-Head human-review Accuracy floor is `0.512467532` (`reference Accuracy - 0.02`). Generation D must bind this numeric value before its formal result exists.
- Best-Accuracy checkpoint SHA-256: `b78b8de08811a3cd00e08be1a79e659864e0d1ddd6f9b44a486c5af930ea928a`. Prediction archive SHA-256: `4994d51fd828e70516cde7ca3b5b5b3b269a4261ad74cbb6e548d67c029876da`.
- Reference report: `reports/x3d_s_train12_val2_user6_user7_partial2_report.{json,md}`. Decision: `freeze_matched_reference_for_direct_head`. Canonical Phase 4/5 artifacts remain unchanged; heldout4/test were not accessed.
- Non-decision context: canonical train-14 OOF Accuracy for the same validation users was `0.482587` on user6 and `0.505435` on user7, or `0.493506` combined. The new partial2 reference is therefore `+0.038961` above that same-user historical OOF subset. The approximately `0.61` A2/A4-T scores used a different five-user fold0 development population and must not be interpreted as a matched regression.

## Direct-Head Generation D Preregistration

- Generation D is preregistered on the frozen `12 train / user6-user7 val / 4 heldout` development split at seed `20260715`; no Direct-Head formal or smoke result was accessed before this freeze.
- The sole architecture intervention is a composite-head replacement: the learned `2048 -> 256` projector plus classifier becomes dropout plus a direct `2048 -> 40` classifier. The expected custom-head parameter count is `81,960`; all temporal, augmentation, optimizer, BN, scheduler, checkpoint-selection, and split fields remain matched to Generation R.
- The sole matched reference is `reports/x3d_s_train12_val2_user6_user7_partial2_report.json` at Accuracy `0.532467532`, Macro-F1 `0.421574625`, and worst-user Accuracy `0.532338308`. Historical `user21,user22` evidence is explicitly ineligible as the matched reference.
- The frozen human-review Accuracy floor is `0.512467532`. The three decisions are `preferred`, `human_review_regression`, and `non_winning_ablation`; diagnostics cannot promote or reject the candidate.
- Candidate config, new split, matched reference report/checkpoint/prediction, canonical Phase 4/5 artifacts, approved design spec, and all three review records are SHA-bound in `reports/x3d_s_train12_val2_user6_user7_direct_head1_preregistration.json`.
- Pre-result run ID: `x3d_s_ir_context_train12_val2_user6_user7_direct_head1_seed20260715`. Status: preregistered, no result. Heldout4/test remain sealed, and no follow-up IR experiment or artifact deletion is automatic.
- Generation D CUDA smoke `x3d_s_ir_context_train12_val2_user6_user7_direct_head1_smoke_20260816` passed the protected gate. The runner completed its two-epoch smoke override with one train/validation microbatch per epoch; no formal result was produced.
- The smoke confirmed `head_type=direct`, embedding dimension `2048`, `81,960` custom-head parameters, `2,315,984` trainable backbone parameters after unfreeze, and finite nonzero gradients in block4, block5, and classifier. Strict checkpoint reload succeeded.
- Both prediction archives contain finite `[2,40]` logits and `[2,2048]` embeddings. Peak CUDA allocation was `260,064,768` bytes; the provisional IR route is `18,829,336 / 95,000,000` bytes.
- Train/validation clip keep fractions remain `1.0`; the frozen `1935/385` trial ownership and all 40 validation classes remain intact. Canonical Phase 4/5 hashes are unchanged and heldout4/test were not accessed.
- PyTorch emitted the existing warn-only notice that `avg_pool3d_backward_cuda` lacks a deterministic implementation. It did not affect the smoke's finite-gradient, strict-load, shape, resource, or isolation gates. Authoritative audit: `reports/x3d_s_train12_val2_user6_user7_direct_head1_smoke_audit.json`.

## Direct-Head Generation D Formal Result

- Formal run `x3d_s_ir_context_train12_val2_user6_user7_direct_head1_seed20260715` completed under the frozen contract. Macro-F1 patience stopped training after epoch 17; runtime was `2002.47` seconds. The best-Accuracy checkpoint was epoch 7.
- Independent matched metrics are Accuracy `0.516883117`, fixed-40 Macro-F1 `0.423791836`, and worst-user Accuracy `0.472636816` (`user6`). Relative to the frozen partial2 reference, deltas are `-0.015584416 / +0.002217211 / -0.059701493`.
- The candidate remains above the preregistered human-review floor `0.512467532`, but does not satisfy the no-worse primary metrics rule. The sole frozen decision is `non_winning_ablation`; preserve all artifacts and do not automatically start another IR experiment.
- At selected epoch 7, train Accuracy/Macro-F1 are `0.790180879 / 0.787937667`; train-to-validation gaps are `0.273297762 / 0.364145831`, respectively `0.146689486 / 0.169894553` smaller than the partial2 gaps. The Direct head reduces memorization but also removes transferable performance, especially for user6.
- Per-user Accuracy/Macro-F1 deltas are user6 `-0.059701493 / -0.021055875` and user7 `+0.032608696 / +0.023394597`. Duration deltas are `<=13 -0.023529412 / -0.024343715`, `14-32 +0.017857143 / +0.013585989`, `33-64 -0.057142857 / -0.043706936`, and `>64 -0.037037037 / -0.004166667`.
- Direct-Head and partial2 disagree on `131/385` trials. Direct-only correct is `26`, partial2-only correct `32`, and both wrong `154`. Validation NLL worsens from `2.000032349` to `2.122650600`; wrong-prediction confidence changes from `0.576646077` to `0.565309047`.
- Direct classifier relative L2 drift from its own seeded initialization is `1.915404005`. The model has `3,056,634` total and `2,397,944` trainable parameters, including the frozen `81,960`-parameter direct classifier. Peak CUDA allocation is `450,658,304` bytes; checkpoint/archive are `12,573,743 / 1,464,668` bytes and the IR route is `18,829,336 / 95,000,000` bytes.
- Best-Accuracy checkpoint SHA-256 is `f813e238e55c3030f46d3ee05c36176f666916a526bff4d339f9b1f507dcca3c`; prediction SHA-256 is `63967f043e162719b47645de3710c42f75d6b80d5738c16323852233ef8124ac`. JSON/Markdown report SHA-256 are `591635758489106898ef4a1daf730da7832f031df228d10f8a0648f7b2238499` and `a59f335bdbb92980b6a39cbdfd50f25927f86b550a16bb53ff54dd88899b85d1`.
- Fresh verification: report regeneration was byte-identical, complete suite `217 passed`, compileall and `git diff --check` passed, no formal process remained, canonical and partial2 hashes were unchanged, and heldout4/test were not accessed. Authoritative report: `reports/x3d_s_train12_val2_user6_user7_direct_head1_report.{json,md}`.

## Post-freeze approved Single13-Global exception (2026-08-20)

- The user explicitly reopened the pure-IR route for one development-only temporal ablation motivated by external IR+Depth_Color R(2+1)D evidence and the hypothesis that excessive temporal coverage increases cross-subject overfitting.
- The experiment uses the frozen `12 train / user6-user7 val / 4 heldout` population. Both usable IR and strictly paired IR/Depth_Color populations are `1935/385` trials and cover all 40 classes; the present experiment remains IR-only.
- The sole intervention is `adaptive K x 13 local clips -> one global 13-frame clip`. It retains the user6/user7 Partial2 architecture, optimization, augmentation, seed, and loss. Canonical Phase 4/5 evidence remains frozen, and heldout4/test access remains forbidden.
- Authoritative experiment plan: `docs/superpowers/plans/2026-08-20-x3d-single13-global-user6-user7.md`.
- Single13-Global CUDA smoke `x3d_s_ir_context_train12_val2_user6_user7_single13_global_smoke_20260820` passed. It verified one clip per trial, finite nonzero gradients in X3D blocks 4/5 and the projected head, a `186,125,824`-byte CUDA peak, protected size compliance, and unchanged canonical hashes. Formal development training had not started when this evidence was recorded.
- Generation D closes without promotion. Retain the projected partial2 model as the matched development winner; no new IR single-modality training or tuning is authorized automatically.

## Approved IR+Depth Single13 Fixed-Context Experiments (2026-08-21)

- The user explicitly authorized development-only paired IR+Depth work on the unchanged `12 train / user6-user7 val / 4 heldout` split after Single13 and one fixed trial person-context box improved training efficiency and fixed-context IR performance. Heldout4 and competition test remained sealed.
- The absolute Depth RGB residual anchor completed at Accuracy `0.548052`, Macro-F1 `0.433077`, and worst-user Accuracy `0.543478`. It retained the immutable repeated-IR K400 path and introduced Depth only through a zero-initialized nine-parameter residual adapter.
- The ordinal-motion experiment changed only the Depth representation: exact inverse OpenCV-JET ordinal values were cropped with the same fixed IR box and converted to dense per-pixel median-relative displacement, source-frame-time-normalized signed velocity, and absolute velocity before the same zero-initialized adapter and X3D convolution.
- Formal run `x3d_s_ir_ordinal_motion_adapter_train12_val2_user6_user7_single13_fixed_context_workers4_seed20260715` stopped by frozen patience after 17 epochs in `5736.26` seconds and selected epoch 9. Metrics are Accuracy `0.548052`, fixed-40 Macro-F1 `0.462041`, and worst-user Accuracy `0.542289` (`user6`).
- Relative to the matched absolute-Depth anchor, deltas are `+0.000000 / +0.028964 / -0.001190`. Relative to fixed-context IR, deltas are `+0.010390 / +0.034527 / +0.024876`. The preregistered strict-Accuracy decision is `non_winning_ablation`; it does not reach the `0.63` stability-review gate.
- Selected-epoch train Accuracy is `0.897674`, leaving a `0.349622` train-to-validation gap. The explicit motion representation improves class balance but does not solve cross-subject overfitting. Against the absolute-Depth anchor, each model uniquely gets 16 trials correct and their pair oracle Accuracy is `0.589610`.
- Preserve all formal artifacts. No extra seed, fold, heldout4 evaluation, stability run, automatic artifact deletion, or canonical Phase 4/5 mutation is authorized. Authoritative result and diagnostics: `reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_ordinal_motion_adapter_{report,diagnostics}.{json,md}`.
