# X3D-S Execution Phase Status

Updated: 2026-08-11 (Asia/Shanghai)

| Phase | Status | Git SHA | Evidence | Risks | Next decision |
|---|---|---|---|---|---|
| Phase 0: Compliance and runtime | Completed | `c088db0` | Official rules rechecked; official X3D forward, provisional IR-route subtotal, dependency check and 15 focused/regression tests passed | X3D-S has no model-specific organizer approval; final trained checkpoint must be remeasured | Phase 1 may begin |
| Phase 1: Temporal data contract | Completed | `481ccb4` | 23 dataset tests, 30 focused/regression tests, and all 74 repository tests passed; real shortest/longest trial probe passed | Exported quality fields are constant in this manifest, so they are contract metadata rather than discriminative evidence in the first run | Phase 2 may begin after review |
| Phase 2: Expert and trainer | Completed | `1601a31` | Trial-level trainer, dual checkpoint archives, fixed BN policy, train-14 finalization, resource audit and held-out rejection implemented; 24 focused and 94 full-suite tests passed | Real CUDA/data smoke is intentionally deferred to Phase 3 | Begin Phase 3 Task 5 |
| Phase 3: End-to-end verification | Completed | `dc13b96` | Revised six-trial parity, fixed-checkpoint sensitivity, CUDA/overfit/size audits, 65 focused tests, and 107 full-suite tests pass | Latency is representative smoke evidence, not a production benchmark | Begin Phase 4 Task 6 |
| Phase 4: Train-14 OOF scientific evaluation | Pending | - | Pure X3D has primary and complementary retention paths | Held-out labels must remain sealed | Wait for Phase 3 exit gate |
| Phase 5: Register IR sparse evidence | Pending | - | Pure candidate registers as `ir_x3d_s_k400_pure` | Held-out archive is evaluation-only until Phase 10 | Wait for Phase 4 primary/complementary retain decision |
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
