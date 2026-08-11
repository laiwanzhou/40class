# X3D-S IR Expert and Six-Modal Sparse-Evidence Fusion Implementation Plan

> **For implementation:** Use `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rules-audited X3D-S IR expert, register leakage-free sparse evidence for it, and integrate it with heterogeneous Depth_Color, Thermal, IMU, Skeleton, and Radar experts into one missing-modality-tolerant inference model below the Small Model Track size limit.

**Architecture:** Reuse the verified IR YOLO pose-guided person-context assets and represent each variable-duration IR trial with adaptive local 13-frame X3D-S clips. X3D-S is one heterogeneous IR expert, not the complete model. Every expert later registers sparse user-grouped OOF evidence into the canonical 3,036-trial union; a calibrated available-expert probability mixture is the permanent safe anchor, and a tiny zero-initialized residual set mixer may only correct that anchor on supported modality patterns. VideoMAE remains excluded from submitted inference unless the organizer explicitly approves the exact checkpoint.

**Tech Stack:** Python 3, PyTorch 2.7, torchvision 0.22, PyTorchVideo 0.1.5, pandas, OpenCV/Pillow, NumPy, scikit-learn, joblib, pytest, CUDA BF16 AMP.

## Global Constraints

- Work only in `D:\work\2026.7.14_kaggle\40class-x3d-adaptive-multiclip` on branch `x3d-s-adaptive-multiclip`.
- Phases 0-5 belong to this IR-expert branch. Phases 6-10 are the approved program-level roadmap and must execute later on dedicated branches/worktrees after Phase 5 closes.
- Do not read competition test data and do not generate a submission.
- Treat the official challenge page and the competition host's Kaggle clarification as the compliance basis: lightweight pretrained CNNs and knowledge distillation are allowed, while large pretrained foundation backbones are prohibited.
- Record the official rule page, the host clarification URL, access date, and verbatim model-size statement in every compliance report.
- Use the existing `combined_frame_manifest.csv`; do not regenerate the full ROI dataset in the first experiment.
- The first scientific run uses only `ir_context_path`; `ir_relation`, left/right ROI, Depth, Skeleton, and IMU are excluded.
- Treat X3D-S as the IR specialist. Do not turn this branch into a joint six-modality trainer.
- Preserve methodological diversity from teammate visual pipelines. Teammate results may motivate transferable principles, but their student architecture, clip construction, teacher protocol, distillation hyperparameters, and training recipe are not inputs to this branch unless a separately pre-registered replication experiment is explicitly approved.
- Freeze this branch's visual inductive bias as pose-guided person context, length-adaptive local temporal decomposition, and X3D spatiotemporal encoding. Do not silently replace any of these with a teammate-specific pipeline after seeing a higher external score.
- The canonical future fusion population is the 3,036-row union in `metadata/manifest.csv`, not the 2,748-row all-six intersection and not the 2,910-row IR ROI export.
- Preserve the fixed 14-train-user/4-held-out-user split in `metadata/splits/fold_0.json`. The four held-out users may never fit temperatures, expert weights, support thresholds, residual parameters, or architecture choices.
- Distinguish raw-modality `present`, expert `usable`, and label-free `quality`. Only `usable` controls fusion availability.
- Keep `strict_alignment` for matched equal-sample experiments and add a separate canonical-union `outer_evidence_alignment` for sparse multimodal fusion.
- The calibrated available-expert probability mixture is a complete deployable model and permanent fallback. The residual mixer is optional and may never be the only classification path.
- If exactly one expert is usable, return its calibrated probability exactly. If no supplied modality produces usable evidence, fail explicitly with per-modality reasons.
- Train the residual mixer primarily on all-six and sufficiently supported natural missingness patterns. Rare combinations use anchor-only inference and must not dominate training.
- Keep the existing neural `ExpertOutput` unchanged. Add a serialized `ExpertEvidence` fusion contract that permits optional embeddings or engineered summaries.
- Pose-guided visual localization is reusable infrastructure. YOLO11n-pose is currently verified for IR; reuse its coordinates for Depth or Thermal only after registration/parity evidence.
- Preserve the existing 14-train-user/4-held-out-user and `sample_id` membership exactly. Within train-14, use persisted user-grouped folds for all epoch, checkpoint, architecture, and hyperparameter decisions; the official held-out labels remain sealed until Phase 10.
- Never split or shuffle individual frames across train and validation.
- Each local X3D input is `[3,13,182,182]`; grayscale IR is repeated to three channels.
- Use `K = min(8, max(1, ceil(num_frames / 32)))` local clips per trial. Split the complete ordered trial into `K` contiguous, near-equal windows before sampling 13 frames inside each window.
- Treat 13 as the local pretrained clip length, never as a forced representation of the entire trial.
- Apply identical spatial augmentation to every frame in a clip.
- Train with one stochastic temporal view per local window; validate the first experiment with one deterministic midpoint view per local window. Aggregate clips before computing trial-level metrics.
- Keep three-view temporal test-time augmentation out of the first experiment so adaptive coverage and view augmentation remain separate variables.
- Store trial-level logits, embeddings, users, qualities, `sample_ids`, and `class_map_hash`. Store labels only for explicitly labeled training/OOF roles; held-out and competition-test evidence files must not contain labels.
- Do not choose expert checkpoints, temporal settings, architectures, fusion temperatures, weights, support rules, or residual behavior on the four held-out users.
- Count every inference-time weight file in one aggregate: X3D-S, classification head, YOLO pose, sensor experts, fusion modules, and any learned preprocessing model.
- Use `95,000,000` serialized bytes as the internal acceptance ceiling, leaving headroom below the published `100 MB` limit and avoiding MB/MiB ambiguity.
- Do not rely on ZIP compression, optimizer removal, or FP16 conversion as the sole reason a model passes the size gate; record serialized bytes and FP32 parameter bytes.
- Treat X3D-S Kinetics pretraining as provisionally admissible under the lightweight-pretrained-CNN clarification, subject to the aggregate size gate and reproducibility requirements. This is a documented rule interpretation, not a model-specific organizer approval.
- VideoMAE weights may not enter the final inference stack without written organizer confirmation naming the exact variant and pretraining source.
- A larger VideoMAE teacher may be used only for training-time distillation; teacher code and provenance must be disclosed, and no teacher weights may be required by `inference.sh`.
- Initial Windows DataLoader configuration is `num_workers: 0`; worker optimization is outside the first experiment.
- The existing Stage 8/9 process must not be interrupted or have its outputs overwritten.

## Rule Compliance Record

| Question | Plan decision |
|---|---|
| Are CNN/RNN/Transformer architectures allowed? | Yes, stated on the official challenge page. |
| Are all pretrained models prohibited? | No. The competition host explicitly allowed lightweight ImageNet-pretrained ResNet18-style backbones. |
| Is knowledge distillation allowed? | Yes. The competition host explicitly allowed it when the final submitted model remains lightweight. |
| Is X3D-S with Kinetics pretraining acceptable? | Provisionally yes under the lightweight pretrained CNN clarification; document the inference stack and retain a model-specific organizer question before final submission. |
| Is pretrained VideoMAE-S acceptable? | Unresolved. Do not place it in the submitted stack without written model-specific confirmation. |
| What counts toward 100 MB? | Conservatively count every inference-time learned weight file together. |
| Internal size ceiling | `95,000,000` serialized bytes for the complete inference stack. |

Official references:

- Challenge rules: `https://openaiotlab.github.io/CUHK-X-Challenge/`
- Host clarification on lightweight pretrained backbones and distillation: `https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/711665`
- Kaggle Small Track Rules: `https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/rules`

The Kaggle Small Track Rules page currently contains copied Large Track wording and does not define the 100 MB measurement method. Preserve this discrepancy in the compliance report and use the stricter interpretation above until the organizer publishes a correction.

## Duration Evidence and Design Rationale

The current `combined_frame_manifest.csv` contains 2,910 trials with frame-count minimum `1`, median `24`, 75th percentile `37`, 95th percentile `69`, and maximum `236`. A total of 2,250 trials (`77.3%`) exceed 13 frames, and class-level median duration ranges from 10 to 56 frames. Therefore, globally reducing every complete trial to 13 samples is retained only as a possible ablation, not the primary design.

With `target_window_frames=32` and `max_clips=8`, the observed data produces a mean of approximately `1.414` local clips per trial: 1,972 trials use one clip, 743 use two, 146 use three, 37 use four, 6 use five, 4 use six, and 2 use eight. This preserves local temporal density for the 236-frame extreme while keeping average X3D work close to the single-clip baseline.

A second motivation for the X3D route is to preserve local spatial structure while temporal features are learned, rather than independently pooling each frame into a compact 2D/frame vector before temporal modeling. This principle is already realized by applying X3D directly to each `[3,13,182,182]` person-context clip; it does not require copying another visual architecture or training recipe.

## Recovered Project Evidence and Its Planning Consequences

The following evidence was recovered from the three referenced project histories and is part of this plan's context. Historical metrics were produced by several different user folds and sample populations; unless a row explicitly says canonical OOF, its score is context only and must not be compared numerically with the new fixed 14/4 protocol.

| Recovered evidence | Planning consequence |
|---|---|
| Canonical raw-trial union: 3,036 trials, 40 classes, 18 users. Modality directory counts are Depth 2,931, IMU 2,903, IR 2,933, Radar 2,914, Skeleton 2,931, Thermal 2,891; only 2,748 rows have all six directories. | Train experts on every usable modality row and outer-align sparse evidence. Never shrink the system to the all-six intersection, which discards about 9.5% of the union. |
| The current ROI export contains 2,910 trials and 84,906 temporally aligned IR/Depth frame rows. IR and Depth share exact frame timestamps in this export. | Reuse the verified IR person-context route for X3D. Preserve IR/Depth pixel correspondence for a later controlled co-expert, but do not add Depth to the first IR experiment. |
| Thermal has independent frame numbering, roughly 2.41 times as many frames at the median, and no common timestamps with IR/Depth in the audited export. | Do not copy IR frame indices, ROI coordinates, or frame-level fusion assumptions into Thermal. Require spatial and temporal registration evidence first; otherwise use a Thermal-native locator/context and trial-level fusion only. |
| The skeleton-to-depth projection search paired 42/42 trials temporally, but all 96 coordinate-mapping candidates were visually unreliable because Skeleton coordinates are normalized/relative. | Do not use direct Skeleton-to-image projection as a production ROI source. Keep Skeleton and visual experts independent until trial-level evidence fusion. |
| The 12-common-joint YOLO/Skeleton diagnostic achieved 2D retrieval Top-1 0.593, Top-5 0.832 and AUC 0.841, while 2.5D pseudodepth was worse. Frame motion correlation was weak. | Use pose correspondence as a quality/identity diagnostic, not a frame-level hard-fusion contract. The Skeleton expert keeps all native joints and full sequence dynamics. |
| Historical aligned Depth+IR B2 reached about 0.4627 Accuracy / 0.3928 Macro-F1, while an IR-primary plus weak ordinal-Depth-vector full-sequence model fell to about 0.2780 / 0.1585. | Early aligned visual interaction can matter. Do not compress Depth to a weak late residual and call that a replacement; defer an aligned IR+Depth co-expert until independent OOF evidence and byte headroom justify it. |
| Historical B2 training Accuracy reached about 0.9763 while held-user Accuracy remained about 0.4627; cross-user SupCon did not reliably improve both Accuracy and Macro-F1. | Cross-user generalization, not raw capacity, is the governing risk. Require user-grouped OOF for every selection and keep the residual mixer tiny. |
| Historical MobileNet and ResNet18 object-interaction experts were similar; the specific ResNet18 inference bundle was 68,994,661 bytes (65.80 MiB) for little Macro-F1 gain. | Treat this as evidence against that specific expert's benefit per byte, not a universal rejection of ResNet18. Do not replicate large visual backbones across modalities without measured complementarity. |
| Historical IMU compact RF was about 0.4188 Accuracy / 0.3352 Macro-F1 at about 5.919 MiB, while Skeleton was a strong anchor candidate. Radar's 21-stat TCN underfit badly. | Audit and canonically rerun IMU RF and full-sequence Skeleton first. Treat raw-point PointNet-frame plus masked TCN as a Radar candidate, not an established result. |
| Tight person crops can remove phones, cups, documents, tableware, medicine and other class-defining objects. Existing ROI audits support extended person context and aligned local interaction regions. | Keep pose-guided person-context as the default IR view. Preserve relation/local ROI generators as reusable infrastructure, but add them only through separately registered ablations. |

Historical results are evidence about architecture and failure modes, not reusable fusion predictions. Any expert retained for the final system must regenerate predictions on the canonical split and the shared user-grouped OOF assignment.

## Fixed First-Run Design

| Item | Value |
|---|---|
| Input view | `ir_context_path` |
| Local frames per clip | 13 |
| Adaptive clip count | `K = min(8, max(1, ceil(num_frames / 32)))` |
| Trial windowing | `K` contiguous, near-equal windows covering the full trial |
| Spatial size | 182 x 182 |
| Training temporal views | 1 stochastic stratified view per window |
| Validation temporal views | 1 deterministic midpoint view per window |
| Trial aggregation | masked mean of clip probabilities; masked mean plus L2 normalization for embeddings |
| Pretraining | Official Kinetics-400 X3D-S weights |
| Embedding dimension | 256 |
| Classes | 40 |
| Optimizer | AdamW |
| Backbone LR | `3e-5` |
| Head LR | `3e-4` |
| Weight decay | `0.05` |
| Warmup | 2 epochs, linear |
| Scheduler | cosine over the fixed full 30-epoch horizon; stopping epoch and scheduler horizon are separate fields |
| Epochs / patience | Formal Phase 4 inner search always runs all 30 epochs with no early stopping; patience 8 remains development-only |
| Batch / accumulation | at most 2 trials and 8 valid clips per microbatch / accumulate 4 microbatches |
| Accumulation objective | equal-weight trial-level NLL: backpropagate summed trial losses, divide accumulated gradients by the actual trial count in each optimizer window, then clip and step |
| AMP | BF16 on CUDA |
| Gradient clipping | 1.0 |
| Checkpoints | best Accuracy and best Macro-F1 |
| Primary report metrics | Accuracy, Macro-F1, worst-user Accuracy, per-class recall, class coverage |

---

## Execution Phases

Execute phases in order. A phase may start only after the previous phase's exit gate passes; update the status table and attach the named evidence before proceeding.

| Phase | Scope | Entry condition | Exit gate | Status |
|---|---|---|---|---|
| Phase 0: Compliance and runtime | Task 1 | New branch/worktree ready | Rule record complete; official pretrained X3D forward passes; provisional IR-route deployment subtotal `<95,000,000` bytes | Completed (`c088db0`) |
| Phase 1: Temporal data contract | Task 2 | Phase 0 passes | Adaptive window boundary tests, real duration audit, determinism, padding and leakage tests pass | Completed (`481ccb4`) |
| Phase 2: Expert and trainer | Tasks 3-4 | Phase 1 passes | Trial-level masked aggregation, gradients, archive schema and focused tests pass | Completed (Task 3 `cc5bc26`; Task 4 `1601a31`) |
| Phase 3: End-to-end verification | Task 5 | Phase 2 passes | Online/offline ROI parity, shortest/longest trial smoke, overfit test, size audit and full tests pass | Completed (`dc13b96`; closure `03eed56`) |
| Phase 4: Train-14 OOF scientific evaluation | Task 6 | Phase 3 passes | Pre-registered grouped-OOF comparison, duration/user/class reports, and frozen X3D decision are complete without held-out access | In progress (Step 1 `74c679a`; strict-v2 amendment `335f5e2`) |
| Phase 5: Register IR sparse evidence | Task 7 | Phase 4 retains pure X3D as a primary or complementary IR expert | Verified train-user OOF archive plus one quarantined held-out IR archive, provenance, evidence contract and size record pass | Pending |
| Phase 6: Freeze expert portfolio | Tasks 8-9 | Phase 5 passes | Six retained experts use only train-14 OOF for selection and each has OOF plus structurally label-free held-out evidence | Pending |
| Phase 7: Build sparse evidence registry | Task 10 | Phase 6 passes | Global registries plus outer-fold nested fusion evidence packages pass lineage and missingness audits | Pending |
| Phase 8: Fit safe anchor | Task 11 | Phase 7 passes | Calibrated masked probability mixture passes singleton, missing-pattern, user and budget gates | Pending |
| Phase 9: Test residual correction | Task 12 | Phase 8 passes | D-versus-A user-grouped comparison either retains the mixer or freezes anchor-only | Pending |
| Phase 10: Assemble final inference | Task 13 | Phase 9 decision freezes architecture | Raw-trial routing, held-out evaluation, all-18-user production refit, exact package size and latency gates pass | Pending |

At each phase boundary, record the Git SHA, changed files, executed commands, pass/fail evidence, unresolved risks, and the next phase decision in `reports/x3d_s_phase_status.md`. Phases 6-10 may use a successor program-status report but must link back to this plan and the approved design spec. Do not silently continue after a failed exit gate.

## Phase 0: Compliance and Runtime Gate

### Task 1: Freeze the Rule Interpretation and Verify the Official X3D-S Runtime

**Files:**
- Create: `requirements-x3d.txt`
- Create: `docs/x3d_s_rule_compliance.md`
- Create: `reports/x3d_s_phase_status.md`
- Create: `scripts/probe_x3d_s_environment.py`
- Create: `tests/test_x3d_s_environment_contract.py`
- Create at runtime: `reports/x3d_s_environment_probe.json`

**Interfaces:**
- Consumes: the existing `pyTorch2.7` environment and CUDA device.
- Produces: a dated compliance record, `load_pretrained_x3d_s() -> torch.nn.Module` import path decision, and a machine-readable environment probe.

- [x] **Step 1: Write the rule compliance record**

Create `docs/x3d_s_rule_compliance.md` with:

1. The three official URLs listed above and access date.
2. The official `100 MB` and no-large-pretrained-backbone statements.
3. The host clarification allowing lightweight pretrained CNNs.
4. The host clarification allowing knowledge distillation.
5. The copied-Large-Track inconsistency on the Kaggle Small Track Rules page.
6. The internal `95,000,000`-byte aggregate inference ceiling.
7. A component table with `component`, `required_at_inference`, `license`, `pretraining_data`, `parameter_count`, `fp32_parameter_bytes`, `serialized_bytes`, and `sha256`.

The initial components are `x3d_s`, `x3d_custom_head`, and `yolo11n_pose`. Mark VideoMAE as `teacher_only_pending_written_confirmation`.

Also include, but do not post automatically, a concise organizer clarification draft naming `X3D-S/Kinetics-400`, `VideoMAE-S` and its exact pretraining source, whether the `100 MB` cap applies to the sum of all inference weights, and whether FP32 serialized state-dict bytes are the intended measurement. Lack of a reply does not block the X3D experiment, but it keeps model-specific approval marked provisional; lack of a VideoMAE reply blocks VideoMAE from final inference.

At Phase 0 this initialized `reports/x3d_s_phase_status.md` with the then-current Phases 0-5. The approved 2026-08-11 six-modal design later expanded the roadmap through Phase 10; keep the status report synchronized with the current phase table, Git SHA, evidence commands, artifacts, risks, and next decision.

- [x] **Step 2: Write the failing environment contract test**

```python
def test_x3d_s_environment_probe_declares_fixed_input_and_source() -> None:
    probe = build_probe_payload(run_forward=False)
    assert probe["input_shape"] == [1, 3, 13, 182, 182]
    assert probe["model_name"] == "x3d_s"
    assert probe["pretraining"] == "kinetics_400"
    assert probe["source_revision"]
```

- [x] **Step 3: Run the test and verify that the probe module is absent**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_environment_contract.py -v`

Expected: FAIL because `scripts.probe_x3d_s_environment` does not exist.

- [x] **Step 4: Pin the primary dependency**

Write `requirements-x3d.txt` with:

```text
pytorchvideo==0.1.5
```

Do not install into a different Python environment. Attempt installation only in `D:\Anaconda\envs\pyTorch2.7` and preserve the full pip output in the probe report.

- [x] **Step 5: Implement the compatibility probe**

The probe must:

1. Record Python, PyTorch, torchvision, CUDA, GPU, and PyTorchVideo versions.
2. Load official `x3d_s(pretrained=True)` weights.
3. Run inference on a finite `[1,3,13,182,182]` tensor.
4. Record output shape, parameter count, FP32 parameter bytes, serialized state-dict bytes, peak allocated CUDA memory, source version/revision, license, pretraining dataset, and SHA-256.
5. Inventory `yolo11n-pose.pt` with the same byte and hash fields.
6. Compute `aggregate_inference_serialized_bytes` for X3D-S, the custom head estimate, and YOLO, counting each deployable byte exactly once.
7. Exit nonzero if weights are random, output is non-finite, the model cannot complete forward inference, or the aggregate is at least `95,000,000` bytes.

If PyTorchVideo 0.1.5 is incompatible with PyTorch 2.7, use the official `facebookresearch/pytorchvideo` source revision that passes this same probe, record the exact commit SHA in `requirements-x3d.txt`, and do not copy an unverified third-party X3D implementation.

- [x] **Step 6: Run the environment probe**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.probe_x3d_s_environment --output reports/x3d_s_environment_probe.json`

Expected: exit 0, finite output, `x3d_s` source recorded, every inference component hashed, and aggregate serialized size below `95,000,000` bytes.

- [x] **Step 7: Run the contract test**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_environment_contract.py -v`

Expected: PASS.

- [x] **Step 8: Commit the compliance and dependency probe**

```bash
git add requirements-x3d.txt docs/x3d_s_rule_compliance.md reports/x3d_s_phase_status.md scripts/probe_x3d_s_environment.py tests/test_x3d_s_environment_contract.py reports/x3d_s_environment_probe.json
git commit -m "Add X3D-S compliance and runtime gate"
```

## Phase 1: Temporal Data Contract

### Task 2: Build a Trial-Safe Adaptive Multi-Clip Dataset

**Files:**
- Create: `src/data/x3d_clip_dataset.py`
- Create: `tests/test_x3d_clip_dataset.py`

**Interfaces:**
- Consumes: `combined_frame_manifest.csv` columns `split`, `class_id`, `action_name`, `sample_id`, `user_id`, `source_frame_index`, `temporal_valid`, `ir_context_path`, `ir_context_effective_valid`, and `ir_context_reliability`.
- Produces: `X3DClipDataset`, `X3DClipSample`, `adaptive_clip_count()`, `partition_trial_windows()`, `stratified_temporal_indices()`, and `collate_x3d_clips()`.

- [x] **Step 1: Write failing tests for grouping and leakage protection**

```python
def test_dataset_groups_complete_trials_and_preserves_frame_order(tmp_path: Path) -> None:
    dataset = X3DClipDataset(fixture_manifest(tmp_path), split="train", training=True)
    assert len(dataset) == 40
    item = dataset[0]
    assert item["clips"].shape == (1, 1, 3, 13, 182, 182)
    assert item["sample_id"] == "train__c00__u__trial"
    assert item["source_indices"].shape == (1, 1, 13)
    assert item["clip_mask"].tolist() == [True]

def test_dataset_rejects_sample_on_both_split_sides(tmp_path: Path) -> None:
    frame = fixture_manifest(tmp_path)
    frame.loc[len(frame)] = {**frame.iloc[0].to_dict(), "split": "val"}
    with pytest.raises(ValueError, match="both train and val"):
        X3DClipDataset(frame, split="train", training=True)
```

- [x] **Step 2: Write failing tests for deterministic single-view validation**

```python
def test_validation_emits_one_deterministic_midpoint_view(tmp_path: Path) -> None:
    dataset = X3DClipDataset(fixture_manifest(tmp_path), split="val", training=False)
    first = dataset[0]
    second = dataset[0]
    assert first["clips"].shape == (1, 1, 3, 13, 182, 182)
    torch.testing.assert_close(first["clips"], second["clips"])
    assert first["source_indices"].tolist() == second["source_indices"].tolist()

def test_236_frame_trial_is_partitioned_into_eight_local_clips(tmp_path: Path) -> None:
    dataset = X3DClipDataset(long_fixture_manifest(tmp_path, frames=236), split="val", training=False)
    item = dataset[0]
    assert item["clips"].shape == (8, 1, 3, 13, 182, 182)
    assert item["clip_mask"].sum().item() == 8
    assert item["window_bounds"][0][0] == 0
    assert item["window_bounds"][-1][1] == 236
    assert all(
        start <= index < end
        for (start, end), indices in zip(item["window_bounds"], item["source_indices"][:, 0])
        for index in indices.tolist()
    )
```

- [x] **Step 3: Run the dataset tests and verify failure**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_clip_dataset.py -v`

Expected: FAIL because `X3DClipDataset` is absent.

- [x] **Step 4: Implement strict manifest indexing**

`X3DClipDataset` must validate all required columns, validate the global 40-class map, reject duplicated `(sample_id, source_frame_index)` rows, reject non-contiguous frame order, reject a `sample_id` appearing on both split sides, and verify every selected image exists.

Build one sample per `sample_id`. Never treat individual frames or temporal views as independent labels.

- [x] **Step 5: Implement length-adaptive trial partitioning**

For a trial with `L` ordered frames, compute:

```python
K = min(8, max(1, math.ceil(L / 32)))
```

Partition all `L` frames into `K` contiguous, non-empty, near-equal windows whose union covers the complete trial exactly and whose order is preserved. Do not derive `K` from the action label. The observed 236-frame maximum therefore emits eight windows of roughly 29-30 frames instead of one globally subsampled clip.

Return `window_bounds`, `num_frames`, `num_clips`, and `temporal_coverage_fraction`. Unit-test boundary lengths `1`, `13`, `32`, `33`, `64`, `65`, and `236`.

- [x] **Step 6: Implement temporal sampling inside each local window**

Use 13 temporal bins inside each local window:

- Training: sample one index uniformly inside each bin using a generator derived from `seed + epoch + dataset_index + window_index`.
- Validation: choose each bin's midpoint and emit exactly one deterministic view per window in the first-run configuration.
- Windows shorter than 13 frames: allow repeated source indices; report per-clip and trial-level `unique_frame_fraction` in quality.
- Never interpolate or synthesize pixels between frames.

Expose `dataset.set_epoch(epoch)` so training views change deterministically by epoch.

- [x] **Step 7: Implement temporally consistent transforms**

Load `ir_context_path` as grayscale, repeat it to three channels, and apply one spatial parameter set to all 13 frames:

- Training: random resized crop to `182 x 182`, scale `[0.8, 1.0]`, ratio `[0.9, 1.1]`, horizontal flip probability `0.5`.
- Validation: resize shorter side to 200 and center crop `182 x 182`.
- Normalize with the exact mean/std attached to the selected official X3D-S weights.

Return clips as `[K, views, 3, 13, 182, 182]`. Spatial randomness may differ between local windows but must be identical across all 13 frames of one clip.

- [x] **Step 8: Pad variable clip counts without changing trial semantics**

`collate_x3d_clips()` must pad only the `K` dimension and return `[B,K_max,views,3,13,182,182]`, `clip_mask=[B,K_max]`, and padded `source_indices=[B,K_max,views,13]`. Padded clips must never enter probability or embedding aggregation and should be skipped before X3D execution when practical.

- [x] **Step 9: Emit fusion-ready metadata**

Each item must include:

```python
{
    "clips": Tensor,
    "clip_mask": BoolTensor,
    "num_frames": int,
    "num_clips": int,
    "label": int,
    "sample_id": str,
    "user_id": str,
    "class_map_hash": str,
    "quality": Tensor,       # temporal_valid_fraction, context_effective_rate,
                             # context_reliability_mean, unique_frame_fraction,
                             # temporal_coverage_fraction, normalized_trial_length
    "quality_mask": BoolTensor,
    "availability": BoolTensor,
    "source_indices": LongTensor,
}
```

- [x] **Step 10: Run dataset and existing contract tests**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_clip_dataset.py tests/test_expert_contract.py -v`

Expected: PASS.

- [x] **Step 11: Commit the dataset**

```bash
git add src/data/x3d_clip_dataset.py tests/test_x3d_clip_dataset.py
git commit -m "Add trial-safe adaptive X3D clip dataset"
```

## Phase 2: IR Expert and Trainer

### Task 3: Wrap X3D-S as a Fusion-Compatible IR Expert

**Files:**
- Create: `src/models/x3d_s_visual_expert.py`
- Create: `tests/test_x3d_s_visual_expert.py`
- Modify: `src/models/__init__.py`

**Interfaces:**
- Consumes: `[B,K,3,13,182,182]` local clips, `clip_mask=[B,K]`, and fixed trial-quality tensors.
- Produces: `ExpertOutput(main_logits=[B,40], embedding=[B,256], quality, quality_mask, availability)`.

- [x] **Step 1: Write the failing model contract test**

```python
def test_x3d_visual_expert_emits_standard_expert_output() -> None:
    model = X3DSVisualExpert(backbone=tiny_test_backbone(), num_classes=40, embedding_dim=256)
    output = model(
        torch.randn(2, 3, 3, 13, 182, 182),
        clip_mask=torch.tensor([[True, True, True], [True, False, False]]),
        quality=torch.ones(2, 6),
        quality_mask=torch.ones(2, 6, dtype=torch.bool),
        availability=torch.ones(2, 1, dtype=torch.bool),
    )
    assert output.main_logits.shape == (2, 40)
    assert output.embedding.shape == (2, 256)
    assert torch.isfinite(output.main_logits).all()
```

- [x] **Step 2: Run the test and verify failure**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_visual_expert.py -v`

Expected: FAIL because the wrapper is absent.

- [x] **Step 3: Implement the wrapper without forking the expert contract**

Load the verified official X3D-S backbone, remove only its Kinetics classification projection, preserve pretrained spatiotemporal blocks, globally pool each valid local clip to one feature vector, and add:

```python
self.embedding_head = nn.Sequential(
    nn.Linear(backbone_dim, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.25),
)
self.classifier = nn.Linear(256, 40)
```

Expose clip-level logits and embeddings internally, then aggregate them with `clip_mask`:

```python
clip_probabilities = softmax(clip_logits, dim=-1)
trial_probabilities = masked_mean(clip_probabilities, clip_mask, dim=1)
trial_logits = log(trial_probabilities.clamp_min(1e-8))
trial_embedding = normalize(masked_mean(clip_embeddings, clip_mask, dim=1))
```

Return the existing `src.models.expert_contract.ExpertOutput`; do not create an X3D-specific output type. Reject trials with zero valid clips and prove by test that changing padded clip values cannot change the output.

- [x] **Step 4: Implement optimizer parameter groups**

Expose:

```python
def parameter_groups(self, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict[str, object]]
```

The pretrained backbone uses `3e-5`; embedding and classifier heads use `3e-4`. Bias and normalization parameters use zero weight decay; other parameters use `0.05`.

- [x] **Step 5: Add a frozen-backbone warmup switch**

Expose `set_backbone_trainable(enabled: bool)`. Epochs 1-2 train only the new embedding and classifier heads; epoch 3 onward trains the full model. BatchNorm layers in a frozen backbone must remain in evaluation mode.

Expose an explicit `update_backbone_bn_running_stats` policy, fixed to `False` for the first run. Calling `model.train()` must keep every backbone BatchNorm module in evaluation mode so Kinetics-400 running mean/variance remain frozen, including after epoch 3. After backbone unfreeze, BatchNorm affine parameters (`weight`, `bias`) remain trainable with the other backbone parameters. A later running-stat adaptation experiment is a separate pre-registered ablation.

- [x] **Step 6: Run model and expert-contract tests**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_visual_expert.py tests/test_expert_contract.py -v`

Expected: PASS.

- [x] **Step 7: Commit the model wrapper**

```bash
git add src/models/x3d_s_visual_expert.py src/models/__init__.py tests/test_x3d_s_visual_expert.py
git commit -m "Add fusion-compatible X3D-S visual expert"
```

### Task 4: Add the X3D-S Training and Evaluation Entry Point

**Files:**
- Create: `src/train_x3d_s_visual_expert.py`
- Create: `configs/experiments/x3d_s_ir_context_fold0.yaml`
- Create: `configs/experiments/x3d_s_ir_context_oof.yaml`
- Create: `tests/test_x3d_s_trainer_contract.py`

**Interfaces:**
- Consumes: `X3DClipDataset`, `X3DSVisualExpert`, the fixed YAML configuration, and an explicit train-14 user partition or persisted OOF-fold assignment.
- Produces: checkpoints, `history.csv`, per-class metrics, trial-level prediction archives, `run_summary.json`, and fusion-ready `ExpertBatchResult` data.

- [x] **Step 1: Write failing trainer contract tests**

```python
def test_validation_aggregates_valid_clips_with_one_view() -> None:
    # [B, K, V=1, classes], with the second clip padded.
    clip_view_logits = torch.tensor([[[[4.0, 0.0]], [[0.0, 9.0]]]])
    probabilities = aggregate_clip_predictions(
        clip_view_logits,
        clip_mask=torch.tensor([[True, False]]),
    )
    assert probabilities.shape == (1, 2)
    assert probabilities.argmax(dim=1).item() == 0

def test_prediction_archive_contains_fusion_contract_fields(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    save_prediction_archive(path, fixture_batch_result())
    with np.load(path) as data:
        assert {"sample_ids", "user_ids", "labels", "logits", "embeddings", "quality", "quality_mask", "availability", "class_map_hash", "num_frames", "num_clips"} <= set(data.files)
```

- [x] **Step 2: Run the trainer tests and verify failure**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_trainer_contract.py -v`

Expected: FAIL because the trainer helpers are absent.

- [x] **Step 3: Implement configuration and CLI validation**

Support:

```text
--config
--smoke-test
--run-id
--epochs
--max-train-batches
--max-val-batches
--train-user-ids
--validation-user-ids
--oof-fold-assignment
--oof-role
```

Reject any config with a non-IR-context first-run view, a class count other than 40, local clip length other than 13, `target_window_frames` other than 32, `max_clips` other than 8, `val_views_per_window` other than 1, `backbone_bn.update_running_stats` other than `false`, `backbone_bn.train_affine_after_unfreeze` other than `true`, nonpositive temperatures or learning rates, or an output directory that would overwrite an existing run. The persisted OOF assignment must contain only the 14 training users and disjoint fit/validation user groups. Training mode must reject the four official held-out user IDs in either partition unless an explicit Phase 5 `finalize_train14` mode is active; that mode accepts all train-14 users for fitting and no labeled validation set.

Freeze the temporal section as:

```yaml
temporal:
  local_frames: 13
  target_window_frames: 32
  max_clips: 8
  train_views_per_window: 1
  val_views_per_window: 1
  aggregation: mean_probability
loader:
  max_trials_per_batch: 2
  max_valid_clips_per_batch: 8
backbone_bn:
  update_running_stats: false
  train_affine_after_unfreeze: true
```

- [x] **Step 4: Implement training**

Use AdamW parameter groups from the model, two-epoch linear warmup, cosine decay over 30 epochs, BF16 autocast, gradient accumulation 4, gradient clipping 1.0, and deterministic seeding. Train the head only for epochs 1-2 and unfreeze the backbone at epoch 3.

Call `train_dataset.set_epoch(epoch)` before every training epoch.

Flatten only valid `[B,K]` clips through X3D, restore the trial structure, aggregate clip probabilities, and compute NLL from the aggregated trial log-probabilities. Do not treat clips as independent labeled samples and do not add a per-clip auxiliary loss in the first run. Use a deterministic batch sampler that admits at most two trials and eight valid clips per microbatch; an eight-clip trial forms a batch by itself, and no window or trial is dropped.

- [x] **Step 5: Implement trial-level validation**

For the first run, `V` is fixed to 1. Flatten only valid entries from `[B,K,1,C,T,H,W]` to `[num_valid_clips,C,T,H,W]`, run X3D, and restore clip logits to `[B,K,40]`. Convert each valid clip to probability, masked-average over `K`, and take `log(clamp_min(1e-8))` as the stored trial logits. Average embeddings over valid clips and L2-normalize the final 256-dimensional trial embedding. Do not generate or average additional validation views in this phase.

Compute metrics only after this aggregation. Assert that every `sample_id` contributes exactly one trial prediction regardless of `K`.

- [x] **Step 6: Save both checkpoint objectives**

Save:

- `best_accuracy.pt`
- `best_macro_f1.pt`
- `val_predictions_best_accuracy.npz`
- `val_predictions_best_macro_f1.npz`

Each NPZ must contain exactly one row per validation `sample_id`. Validate uniqueness, finite values, `[N,40]` logits, `[N,256]` embeddings, the same class-map hash as the dataset, and diagnostic `num_frames=[N]` and `num_clips=[N]` arrays. These diagnostic arrays must not be treated as action evidence by later fusion.

For every Phase 4 fold, `best_accuracy.pt` is the sole checkpoint used for the primary scientific OOF prediction. Break equal-Accuracy ties by higher Macro-F1, then earlier epoch. `best_macro_f1.pt` and its predictions are diagnostic only and may not be substituted after results are visible or mixed across folds.

- [x] **Step 7: Record scientific and resource metrics**

`history.csv` and `run_summary.json` must include train/val Accuracy, Macro-F1, weighted F1, loss, class coverage, zero-recall classes, worst-user Accuracy, learning rates, epoch duration, peak CUDA memory, parameter count, state-dict bytes, checkpoint file sizes, mean/max clips per trial, processed clips per second, and trial-level latency by length bucket.

`run_summary.json` must also embed the compliance manifest: X3D weight hash and bytes, YOLO weight hash and bytes, custom-head bytes, IR-route serialized weight subtotal, the `95,000,000`-byte internal limit, pretrained source, and an explicit `ir_route_provisional_size_gate_passed` boolean. This flag covers only the raw-IR-to-X3D route, not the six-modal submission package. A run that fails this provisional route gate may be used for diagnosis but must not be marked deployment-eligible.

- [x] **Step 8: Run trainer contract tests**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_trainer_contract.py -v`

Expected: PASS.

- [x] **Step 9: Commit the trainer**

```bash
git add src/train_x3d_s_visual_expert.py configs/experiments/x3d_s_ir_context_fold0.yaml configs/experiments/x3d_s_ir_context_oof.yaml tests/test_x3d_s_trainer_contract.py
git commit -m "Add adaptive multi-clip X3D-S training pipeline"
```

## Phase 3: End-to-End Verification

### Task 5: Prove the Pipeline Before Full Training

**Files:**
- Create: `scripts/audit_x3d_s_run.py`
- Create: `src/inference/x3d_s_ir_context_pipeline.py`
- Create: `tests/test_x3d_s_real_manifest_contract.py`
- Create: `tests/test_x3d_s_online_roi_parity.py`
- Create at runtime: `reports/x3d_s_smoke_report.md`

**Interfaces:**
- Consumes: the real combined manifest and a smoke-run directory.
- Produces: a pass/fail audit covering split membership, shapes, pretrained loading, gradients, provisional IR-route deployment subtotal, online ROI parity, latency, and fusion contract.

- [x] **Step 1: Write the real-manifest read-only contract test**

The test must assert:

- exactly 40 classes;
- no train/val user overlap;
- no train/val `sample_id` overlap;
- every selected trial has ordered frames;
- the shortest trial emits `[1,1,3,13,182,182]` and the 236-frame trial emits `[8,1,3,13,182,182]`;
- all window bounds are ordered, non-overlapping, non-empty, and cover every source frame exactly once before within-window sampling;
- validation access is deterministic;
- no path contains a competition-test directory.

- [x] **Step 2: Run the real-manifest test**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_real_manifest_contract.py -v`

Expected: PASS without writing new data assets.

- [x] **Step 3: Run a two-sample forward/backward smoke test**

Run:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m src.train_x3d_s_visual_expert `
  --config configs/experiments/x3d_s_ir_context_fold0.yaml `
  --smoke-test `
  --run-id x3d_s_ir_context_adaptive_smoke
```

Expected: train and validation complete, both backbone and head receive finite gradients after unfreezing, checkpoints reload, and prediction archives pass validation.

- [x] **Step 4: Run a small-subset overfit test**

Use 16 training samples covering at least 8 classes and containing both one-clip and multi-clip trials for 20 epochs with augmentation disabled. Require aggregated trial loss to fall and training Accuracy to exceed 80%. This is an implementation test only; do not report its validation metrics as scientific evidence.

- [x] **Step 5: Audit model size and fusion contract**

`scripts/audit_x3d_s_run.py` must verify:

- X3D plus custom head parameter bytes;
- whether YOLO weights are required at inference;
- combined size if YOLO is included;
- unique and complete validation sample set;
- exact `class_map_hash` agreement with the existing visual expert;
- successful `align_expert_batch()` against a fixture with shuffled sample order;
- `calibrated_probability_mixture(alpha=0)` exactly recovers X3D probabilities.

The audit must sum the serialized weights of every component needed by the provisional IR route. This includes YOLO11n-pose, X3D-S, the embedding/classification head, and any learned IR preprocessing or calibration module. Count deployable files, not conceptual submodules: if the custom head is already inside the final X3D checkpoint, do not add it twice. Require this IR-route subtotal to remain below the conservative internal limit of `95,000,000` bytes; do not call it the complete submission package or report only X3D's parameter count.

- [x] **Step 6: Implement and verify raw-trial online inference parity**

Implement a single inference entry point that accepts an official raw IR trial and performs:

```text
raw IR frames -> existing UltralyticsPoseLocator -> IRPrimaryInputROIBuilder
              -> adaptive contiguous windows -> 13-frame local clips
              -> shared X3D-S -> masked trial aggregation -> trial logits
```

Reuse the existing YOLO pose locator and ROI builder rather than duplicating box-selection logic. On a deterministic representative sample set covering one-, two-, four-, and eight-clip trials, low/high pose reliability, recovered context cases, and both available eight-clip trials, compare online boxes and crops with the existing exported `ir_context` assets. Require identical trial frame ordering, clip counts, window bounds, sampled frame indices, person-selection/recovery path, and X3D normalization. Fresh YOLO detections need not reproduce historical floating-point boxes exactly. Require maximum absolute ROI-box drift <= 1 pixel, crop MAE <= 1/255, P99 absolute pixel error <= 8/255, PSNR >= 40 dB, and worst-frame crop MAE <= 2/255. Differences caused by subpixel detector drift interacting with floating-point crop boundaries and resampling are admissible when all geometric and image-level gates pass. Silent full-frame fallback remains forbidden. Fail on missing weights, ordering changes, clip-count drift, person-selection/recovery drift, or normalization drift.

Using one fixed hashed X3D checkpoint, compare offline-export and online-generated inputs at the model level by trial-embedding cosine similarity, probability L1 distance, Jensen-Shannon divergence, maximum class-probability delta, and top-1 agreement. These are sensitivity diagnostics supporting the input-parity audit; top-1 disagreement alone does not invalidate otherwise passing preprocessing parity. Record the checkpoint SHA-256 and report the observed values without selecting post-hoc sensitivity thresholds from these representative trials.

The parity test uses training data only. Record end-to-end latency separately for YOLO/ROI preprocessing, each X3D clip, and complete trial inference in the `<=13`, `14-32`, `33-64`, and `>64` frame buckets so the report reflects the actual submission path.

- [x] **Step 7: Run the focused and full test suites**

Run:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_environment_contract.py tests/test_x3d_clip_dataset.py tests/test_x3d_s_visual_expert.py tests/test_x3d_s_trainer_contract.py tests/test_x3d_s_real_manifest_contract.py tests/test_x3d_s_online_roi_parity.py tests/test_expert_contract.py -v
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest -q
```

Expected: all tests pass.

- [x] **Step 8: Commit smoke evidence**

```bash
git add scripts/audit_x3d_s_run.py src/inference/x3d_s_ir_context_pipeline.py tests/test_x3d_s_real_manifest_contract.py tests/test_x3d_s_online_roi_parity.py reports/x3d_s_smoke_report.md
git commit -m "Verify compliant X3D-S inference end to end"
```

## Phase 4: Train-14 OOF Scientific Evaluation

### Task 6: Run the Pre-Registered Train-14 OOF Adaptive Multi-Clip IR-Context Experiment

**Files:**
- Create at runtime: `outputs/x3d_s_ir_context_oof/<run-id>/...`
- Create: `scripts/summarize_x3d_s_experiment.py`
- Create at runtime: `metadata/splits/train14_oof_3fold.json`
- Create at runtime: `metadata/splits/train14_oof_3fold_ir_coverage_audit.json`
- Create at runtime: `reports/x3d_s_ir_context_oof_report.md`
- Create at runtime: `reports/x3d_s_ir_context_oof_per_class.csv`
- Create at runtime: `reports/x3d_s_ir_context_oof_per_user.csv`

**Interfaces:**
- Consumes: the verified configuration, fixed outer 14/4 split, and the persisted three-fold user assignment inside train-14.
- Produces: one cross-fitted adaptive multi-clip X3D-S comparison over train-14 and a frozen decision on whether X3D remains an IR expert candidate; it does not read held-out labels or predictions.

- [x] **Step 1: Freeze and hash the experiment inputs**

Record the configuration hash, manifest hash, class-map hash, X3D and YOLO pretrained-weight hashes, compliance-document hash, rule-source URLs and access date, Git SHA, environment probe, provisional IR-route deployment subtotal, outer train/held-out users, formal outer OOF fold users, outer-train-only epoch-selection users, trial counts, and all three actual runtime seeds before looking at OOF results. Freeze `20260715` as the canonical Phase 5 OOF evidence seed; `20260716` and `20260717` are training-stability confirmation only and may not replace it based on results. The experiment may start only if `ir_route_provisional_size_gate_passed=true` and the command cannot load the held-out archive. Bind a write-once usable-IR inner-coverage audit to the assignment SHA-256 and require every inner-fit partition actually consumed by X3D to cover all 40 classes. The canonical assignment remains immutable; if this gate fails, create a separately versioned superseding assignment before any formal result rather than rewriting it.

- [ ] **Step 2: Run the fixed three-fold user-grouped OOF experiment**

Run:

Generate `metadata/splits/train14_oof_3fold.json` exactly once from canonical train-14 labels/users using `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20260715)`. Validate disjoint users, complete 40-class coverage in every outer-train partition, complete 40-class coverage after concatenating the three outer-validation partitions, and exactly-once outer-validation ownership for every train-14 user. Report per-fold outer-validation class coverage rather than requiring 40 classes in every individual fold: class 25 occurs for only `user1` and `user7` in train-14, so three validation folds with class 25 are mathematically impossible. Keep the metric label set fixed at all 40 classes. Record the assignment SHA-256 in the Phase 4 experiment manifest and make the file immutable for later phases.

For each outer fold, also freeze one epoch-selection split entirely inside the outer-train users before model results. Generate three candidate inner splits with `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20260715 + outer_fold)`, discard any candidate whose canonical inner-fit side lacks any of the 40 classes, then choose the remaining candidate with maximum canonical inner-validation class coverage and lower candidate index as the tie break. Before training, separately audit the selected candidate on the usable-IR population and require its inner-fit side to cover all 40 classes; record usable-IR fit/validation trial counts, class counts, and missing class IDs. If that usable-IR gate fails, preserve the original assignment and create a separately versioned successor that discards candidates lacking usable-IR 40-class fit coverage. All three training seeds reuse the same accepted outer and inner user assignments. The already frozen assignment passes the usable-IR gate, so preserve it and attach the separate hashed coverage audit instead of mutating it.

The formal three-seed confirmation uses actual runtime seeds `20260715`, `20260716`, and `20260717`. The CLI must expose `--seed`; it overrides the YAML seed and the resolved config, run summary, Phase 4 experiment manifest, and config/provenance hash must all record the actual seed. Merely changing `run-id` is not a seed change. Run each seed with:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m src.train_x3d_s_visual_expert `
  --config configs/experiments/x3d_s_ir_context_oof.yaml `
  --oof-fold-assignment metadata/splits/train14_oof_3fold.json `
  --oof-role train14 `
  --seed 20260715 `
  --run-id x3d_s_ir_context_adaptive_oof_strict_v3_seed20260715
```

For each outer fold and seed, first train only on the frozen inner-fit users for all 30 epochs with no early stopping and select the epoch using Accuracy, then Macro-F1, then earlier epoch on the frozen inner-validation users. Keep `best_macro_f1.pt` diagnostic only. Then discard that selection model for formal prediction, initialize a fresh model from the same pretrained source, train on every outer-train user for exactly the selected epoch with no labeled validation or early stopping, freeze `formal_outer_refit.pt`, and evaluate the untouched outer-validation users exactly once. Both stages use the identical first `selected_epoch` learning rates from the frozen 30-epoch cosine trajectory: `training.epochs` controls when the loop stops, while `training.scheduler_horizon_epochs=30` remains unchanged. Outer-validation labels may compute final metrics only after the formal checkpoint is frozen; they may never select an epoch, stop training, or choose a checkpoint. Emit exactly one formal cross-fitted prediction per usable IR trial in the outer fold. Do not change ROI views, local frame count, `target_window_frames=32`, `max_clips=8`, crop size, aggregation method, loss, or `val_views_per_window=1` during this run.

The accidentally started run `x3d_s_ir_context_adaptive_oof_seed20260715` was interrupted during fold-0 inner epoch selection before any formal refit or outer-validation access after the scheduler-horizon defect was identified. Preserve it unchanged with its abort sidecar; it is invalid scientific evidence and may not be resumed, overwritten, or merged into the replacement run.

Two later launch directories are also excluded and preserved. `x3d_s_ir_context_adaptive_oof_strict_v2_seed20260715` stopped before epoch 1 because PowerShell promoted a CUDA warning to a native-command error. `x3d_s_ir_context_adaptive_oof_strict_v2b_seed20260715` stopped after fold-0 inner epoch 6, before refit or outer-validation access, when review found that averaging microbatch-mean losses over a variable-size accumulation window gave singleton trial batches twice the per-trial weight of two-trial batches. Formal Phase 4 training must use summed trial NLL per microbatch, accumulate raw gradients, divide gradients by the actual number of trials in the optimizer window, and only then apply gradient clipping and the optimizer step. The reported loss must be the same total trial NLL divided by total trials. A regression test must require a `1+2+2+2` microbatch update to match the same seven trials evaluated as one full batch.

- [ ] **Step 3: Re-evaluate the selected fold checkpoints**

Reload every `formal_outer_refit.pt`, regenerate that outer fold's predictions, and require exact prediction and metric agreement with the saved formal values within `1e-6`. Verify the checkpoint records the actual runtime seed, frozen inner epoch-selection users, selected epoch, full outer-train refit users, outer-validation users, resolved-config hash, and assignment hash. Concatenate only after verifying that each train-14 usable IR `sample_id` occurs exactly once and its labels participated in neither weight training nor epoch/checkpoint selection.

- [ ] **Step 4: Generate the report**

Compare X3D-S with:

- the teammate VideoMAE result as descriptive context only after confirming identical outer split and trial-level metric semantics; its held-out score cannot select this X3D configuration. No teammate-specific architecture, clip schedule, teacher hyperparameters, or distillation recipe is required for this comparison; record only coarse comparability metadata such as evaluation population and metric definition;
- a matched MobileNet/TCN IR-context baseline using the exact same train-14 OOF user folds, trial `sample_id` set, class map, ROI assets, and trial-level metric implementation; rerun it if the existing result does not satisfy every matching condition;
- a canonically rerun Skeleton TCN train-14 OOF result when available; historical held-out scores remain context only.

Report Accuracy, Macro-F1, worst-user Accuracy, per-user metrics, per-class recall, zero-recall classes, training/generalization gap, provisional IR-route deployment subtotal, individual component sizes, checkpoint size, GPU memory, X3D-only latency, YOLO/ROI latency, and end-to-end trial latency.

Also report Accuracy, Macro-F1, sample count, mean clip count, and mean latency separately for trial-length buckets `<=13`, `14-32`, `33-64`, and `>64`. These exact bucket labels and boundaries must be emitted by the trainer itself; downstream reports may not silently consume the historical `1-32`, `65-128`, or `129+` buckets. Compare short-action and long-action class recall so a global score cannot hide duration-dependent failure.

For the matched IR-context comparison, align cross-fitted predictions by `sample_id` and run 10,000 paired bootstrap replicates stratified by train-14 OOF user. Every replicate must retain every OOF user, independently resample paired trials with replacement within each user, concatenate the resampled users, and recompute metrics. Keep the class label set fixed at all 40 classes in every replicate and report point deltas plus 95% confidence intervals for Accuracy and Macro-F1. Do not bootstrap only the user clusters. This bootstrap quantifies cross-fitted sample uncertainty only; three-seed confirmation is still required before claiming training stability.

- [ ] **Step 5: Apply the pre-registered decision rule**

- **Primary IR expert:** Train-14 cross-fitted pure X3D-S has positive Accuracy and Macro-F1 deltas over the matched IR-context baseline, the paired 95% confidence interval for the Accuracy delta excludes zero, Macro-F1 improves, and worst-user Accuracy does not regress. Run three OOF seeds before registration and report mean, standard deviation, and per-seed deltas. `20260715` remains the canonical Phase 5 evidence seed even when another seed has higher metrics.
- **Complementary IR expert:** Standalone Accuracy/Macro-F1 does not satisfy the primary rule, but its worst-user Accuracy delta relative to the matched IR-context baseline is at least `-0.02` (no more than two absolute percentage points of regression) and X3D supplies reproducible additional evidence. This `-0.02` threshold is frozen before Phase 4 OOF results are inspected. Require unique-correct counts in both directions, oracle-pair Accuracy and its paired confidence interval, class-wise rescues, and error disagreement; retention requires a positive oracle-pair gain whose 95% interval excludes zero plus class-wise rescue that persists across the three OOF seeds. Register it as complementary rather than primary.
- **Promising but unconfirmed:** point metrics or complementarity are positive but their paired intervals include zero, or results vary materially by user, seed, class, or duration bucket. Preserve the predictions for diagnosis, but do not register the candidate until the pre-registered three-seed confirmation passes one of the two retention paths.
- **Temporal follow-up only:** the train-14 OOF `>64` bucket trails the matched baseline while shorter buckets improve. Keep the visual input fixed and test only `target_window_frames=24` versus 32 using the same inner folds; do not mix this with three-view TTA or inspect held-out results.
- **Mandatory manual-review stop:** If any completed seed or the three-seed aggregate has a worst-user Accuracy delta below `-0.02`, interrupt the automatic workflow and report the result for human review. Preserve every trained checkpoint, prediction archive, run summary, history, log, hash manifest, and audit artifact exactly as generated. Do not delete, prune, overwrite, replace, or automatically reject/register the candidate, and do not begin Phase 5 until the user records a manual decision.
- **Stop and audit:** X3D satisfies neither the primary nor complementary path. Audit normalization, pretrained loading, adaptive window coverage, padded-clip masking, aggregation, and split comparability before architectural expansion. This path also preserves all generated artifacts and requires a reported human decision before any cleanup or continuation.
- Do not add Depth or additional ROI views merely because the first score is low.

- [ ] **Step 6: Commit only code and small reports**

Do not commit weights, NPZ files, or the output directory.

```bash
git add scripts/summarize_x3d_s_experiment.py metadata/splits/train14_oof_3fold.json reports/x3d_s_ir_context_oof_report.md reports/x3d_s_ir_context_oof_per_class.csv reports/x3d_s_ir_context_oof_per_user.csv
git commit -m "Report X3D-S IR-context baseline"
```

## Phase 5: Register the IR Expert into the Sparse Evidence System

### Task 7: Generate Leakage-Free IR ExpertEvidence

**Files:**
- Create: `src/fusion/expert_evidence.py`
- Create: `scripts/build_x3d_s_ir_evidence.py`
- Create: `docs/x3d_s_ir_evidence_contract.md`
- Create: `tests/test_expert_evidence_contract.py`
- Create at runtime: `outputs/x3d_s_ir_evidence/<run-id>/...`
- Modify: `reports/x3d_s_phase_status.md`

**Interfaces:**
- Consumes: the retained Phase 4 X3D configuration, canonical union manifest, fixed 14/4 user split, and exact IR ROI manifest.
- Produces: a verified sparse pure-X3D IR OOF archive and one quarantined held-out `ExpertEvidence` archive plus immutable provenance under `expert_id=ir_x3d_s_k400_pure`; it does not fit a multimodal fusion model.

- [ ] **Step 1: Write RED tests for the serialized evidence contract**

```python
def test_evidence_allows_optional_embedding_but_requires_provenance() -> None:
    evidence = fixture_evidence(embeddings=None, engineered_summary=np.ones((3, 5)))
    evidence.validate()
    assert evidence.logits.shape == (3, 40)
    assert evidence.model_sha256
    assert evidence.config_sha256

def test_evidence_rejects_duplicate_or_unknown_samples() -> None:
    with pytest.raises(ValueError, match="duplicate sample"):
        fixture_evidence(sample_ids=("a", "a")).validate()

def test_heldout_evidence_forbids_labels() -> None:
    with pytest.raises(ValueError, match="labels forbidden"):
        fixture_evidence(role="heldout", labels=np.array([0, 1])).validate()
```

- [ ] **Step 2: Run the evidence tests and observe RED**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_expert_evidence_contract.py -v`

Expected: FAIL because `src.fusion.expert_evidence` does not exist.

- [ ] **Step 3: Implement `ExpertEvidence` without changing `ExpertOutput`**

Required for every role are `role`, `expert_id`, `sample_ids`, `user_ids`, `logits`, `availability`, modality-native `quality`, `quality_mask`, scalar `fusion_quality_score=[N,1]`, `class_map_hash`, `model_sha256`, `config_sha256`, `deployed_weight_bytes`, and `preprocessing_dependencies`. `labels` is optional at the type level but required exactly for `role=oof_train14` and forbidden for `role=heldout` or `role=competition_test`. Optional fields are `embeddings`, `engineered_summary`, and diagnostic arrays. Require finite logits, unique IDs, 40 classes, matching row counts, a non-empty class hash, SHA-256 values, and fusion quality in `[0,1]`.

Each expert must pre-register a deterministic label-free mapping from its native quality vector/mask to `fusion_quality_score`. The mapping and its hash are provenance. It may use sensor validity, detection confidence, point coverage, or similar input diagnostics, but never class labels, correctness, held-out performance, or fitted action metrics. If no defensible scaling exists, pre-register the constant score `1.0` and let availability/support control the first gate rather than inventing a post-hoc scale. For IR, preserve native diagnostics such as pose-detection rate, keypoint confidence, context-effective rate, recovered-ROI fraction, and local-view reliability when available.

This Phase 5 registration is immutable. A future teacher-assisted candidate must use a distinct identity such as `ir_x3d_s_teacher_assisted`, its own config/model/provenance hashes, and independently generated OOF and held-out evidence. It may later replace the pure candidate through the Phase 6 portfolio gate, but it may not overwrite `ir_x3d_s_k400_pure` artifacts or be silently treated as the same experiment.

- [ ] **Step 4: Load and verify the frozen Phase 4 OOF assignment**

Load the exact `metadata/splits/train14_oof_3fold.json` created in Phase 4. Require disjoint train/OOF users, 40-class coverage in every outer-train partition and in the concatenated outer-validation population, exactly-once validation ownership, frozen inner epoch-selection users contained entirely inside outer-train, and an SHA-256 equal to the Phase 4 experiment manifest. Individual outer-validation folds may have fewer than 40 classes because class 25 exists for only two train-14 users; preserve a fixed 40-class metric label set and report missing classes. Fail if the file is missing or differs; Phase 5 must never regenerate or repair it. All later modalities and fusion folds must reuse this same assignment.

- [ ] **Step 5: Generate sparse IR OOF evidence**

Reuse only the canonical-seed `20260715` complete Phase 4 OOF archive when its config, manifest, fold assignment, model, actual seed, strict outer-refit protocol, and code hashes match; otherwise regenerate that seed deterministically under the same strict protocol. Seeds `20260716` and `20260717` remain stability evidence and cannot replace the canonical archive post hoc. For each OOF fold, select the epoch only inside outer-train, refit the complete IR preprocessing and X3D expert on all outer-train users for that fixed epoch, then predict only usable IR rows belonging to the untouched OOF users. Concatenate folds and require exactly one prediction for every usable IR trial in the 14-user union, no prediction for unavailable IR, and no user whose label influenced weight training, epoch selection, or checkpoint selection for its prediction.

Save `oof_evidence.npz`, `oof_provenance.json`, fold checkpoints, per-fold metrics, hashes, frame/clip diagnostics, and deployed-byte totals. Do not copy neutral rows into the sparse expert archive; outer alignment belongs to Phase 7.

- [ ] **Step 6: Generate quarantined held-out evidence**

Freeze the architecture, preprocessing, and checkpoint rule from Phase 4. Compute `finalize_epochs` as the median of the nine selected `best_accuracy.pt` epochs from the three fixed seeds times three OOF folds; with nine values this is an observed integer epoch and requires no rounding rule. Set the final train-all-14 seed to `20260715`. Record the nine source epochs, median, seed, and hashes before finalization, train the retained X3D expert on all 14 training users for exactly `finalize_epochs` without labeled validation, and predict usable IR rows from the four held-out users exactly once. Save `heldout_evidence.npz` and provenance separately with `role=heldout`, `labels=None`, and no label array in the serialized file. Mark the archive `evaluation_only=true`; no fitting, selection, reporting, or diagnostic command before Phase 10 may accept or inspect that path.

- [ ] **Step 7: Report expert complementarity inputs**

Using train-14 OOF evidence only, report standalone Accuracy, Macro-F1, worst-user Accuracy, per-class recall, duration buckets, unique-correct samples relative to the matched IR baseline, oracle-pair Accuracy, error agreement, latency, YOLO/X3D bytes, and the exact evidence population. For held-out evidence, report only row counts, hashes, schema validity, routing availability, and quarantine status until Phase 10. These metrics assess whether X3D is useful as an IR expert, not whether it is the whole model.

- [ ] **Step 8: Verify and commit Phase 5**

Run:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_expert_evidence_contract.py tests/test_x3d_s_trainer_contract.py tests/test_expert_contract.py -v
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.build_x3d_s_ir_evidence --config configs/experiments/x3d_s_ir_context_oof.yaml --audit-only
git diff --check
```

Expected: contracts pass, the audit proves user-held-out OOF generation and held-out quarantine, and the exact IR deployment dependencies remain recorded below the program budget.

```bash
git add src/fusion/expert_evidence.py scripts/build_x3d_s_ir_evidence.py docs/x3d_s_ir_evidence_contract.md tests/test_expert_evidence_contract.py reports/x3d_s_phase_status.md
git commit -m "Register X3D-S IR sparse evidence"
```

Stop this branch after Phase 5 review. Phases 6-10 execute in later dedicated worktrees.

## Phase 6: Freeze the Six-Expert Portfolio

### Task 8: Audit and Reproduce Existing Strong Experts on the Canonical Split

**Files:**
- Create: `scripts/audit_six_modal_expert_portfolio.py`
- Create: `reports/six_modal_expert_portfolio.md`
- Create: `tests/test_six_modal_expert_portfolio.py`
- Reuse: IMU compact Random Forest and Skeleton sequence-expert source from their accepted branches.

**Interfaces:**
- Consumes: canonical manifest/split, historical checkpoints/reports, expert configs, and the 95 MB ledger.
- Produces: reproducible candidate records for IMU and Skeleton and an explicit rerun decision for every historical expert.

- [ ] **Step 1: Write RED provenance and split-compatibility tests**

```python
def test_portfolio_rejects_metric_without_reproducible_artifact() -> None:
    with pytest.raises(ValueError, match="reproducible artifact"):
        validate_candidate(candidate_with_metrics_only())

def test_candidate_must_match_canonical_users_and_class_map() -> None:
    with pytest.raises(ValueError, match="canonical split"):
        validate_candidate(candidate_with_old_fold())
```

- [ ] **Step 2: Audit IMU RF and Skeleton evidence**

Verify source commit, feature schema, train-only preprocessing, class map, sample population, weights, imputer, hashes, inference entry point, serialized bytes, and license. Historical scores from a different user split remain context only. Rerun each accepted architecture on the canonical train-14 population using the exact Phase 4 `train14_oof_3fold.json` and its registered SHA-256. Select candidates only from cross-fitted predictions; the outer four users and their labels remain unavailable throughout Phase 6.

For Skeleton, retain all native joints and sequence features such as root-centered coordinates, scale-normalized geometry, bones, and velocity. The 12-joint YOLO correspondence is an audit/quality signal only; do not reduce the expert to those joints or impose frame-level visual alignment.

- [ ] **Step 3: Add complementarity reporting**

For each candidate, compute standalone Accuracy/Macro-F1, worst-user Accuracy, pairwise error agreement with registered experts, unique-correct counts in both directions, oracle-pair Accuracy, and class-wise rescues. Do not eliminate a smaller expert solely because its standalone Accuracy is lower.

- [ ] **Step 4: Run and commit the audit**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_six_modal_expert_portfolio.py -v`

Expected: PASS with every retained historical result linked to reproducible artifacts and canonical-rerun status.

```bash
git add scripts/audit_six_modal_expert_portfolio.py reports/six_modal_expert_portfolio.md tests/test_six_modal_expert_portfolio.py
git commit -m "Audit six-modal expert portfolio"
```

### Task 9: Select Compact Depth, Thermal, and Radar Experts

**Files:**
- Create: `configs/experiments/depth_compact_expert.yaml`
- Create: `configs/experiments/thermal_compact_expert.yaml`
- Create: `configs/experiments/radar_point_tcn_expert.yaml`
- Create: `tests/test_candidate_expert_contracts.py`
- Create at runtime: candidate reports and sparse OOF-ready archives.

**Interfaces:**
- Consumes: modality-native raw training data, canonical split, common expert/evidence contracts, and remaining byte budget after IR/YOLO/IMU/Skeleton.
- Produces: one retained or explicitly rejected candidate for Depth, Thermal, and Radar.

- [ ] **Step 1: Audit visual registration before model selection**

Verify the known exact IR/Depth timestamp pairing and prove spatial ROI parity before reusing IR coordinates for Depth. Thermal's audited export has independent frame numbering and no common timestamps with IR/Depth, so default to Thermal-native temporal sampling and trial-level fusion. Reuse IR coordinates for Thermal only after new field-of-view, resolution, camera-geometry, and temporal-registration evidence passes. Count YOLO once when preprocessing is genuinely shared.

- [ ] **Step 2: Write RED modality-contract tests**

Require each candidate to emit 40-class logits, label-free quality, usability, provenance, latency, and exact deployed bytes. Radar tests must cover an empty point-set trial without deleting the canonical row. Visual tests must cover locator failure and documented context fallback.

Every Depth, Thermal, and Radar candidate must use the exact frozen Phase 4 `train14_oof_3fold.json`. Fit preprocessing, checkpoints, and candidate decisions inside those train-14 folds only. The outer four users and their labels are unavailable for candidate metrics, early stopping, architecture choice, or portfolio selection.

- [ ] **Step 3: Run controlled candidate experiments**

Depth starts with a compact spatial-temporal/geometric expert that preserves native geometry. Thermal starts with a compact visual-temporal expert. Radar compares the failed 21-stat TCN reference with raw variable point sets encoded by a shared point MLP, mean/max frame pooling, and a small masked TCN. PointNet+TCN is retained only if it improves relevant metrics or complementarity per deployed byte.

- [ ] **Step 4: Apply the portfolio gate**

Retain a candidate only when it has reproducible canonical-split outputs, valid missingness behavior, useful standalone or complementary evidence, acceptable latency, and a provisional six-expert package below 95,000,000 bytes. Record this estimate separately as `provisional_complete_package_size_gate_passed`; it includes declared upper-bound estimates for the anchor, adapters, and optional residual mixer and is not the final Phase 10 `complete_submission_size_gate_passed`. Do not clone X3D-sized capacity into Depth or Thermal without a measured benefit-per-byte case.

- [ ] **Step 5: Freeze all five non-IR experts and generate quarantined evidence**

After IMU, Skeleton, Depth, Thermal, and Radar candidate identities are frozen from train-14 OOF evidence, require each retained expert to have a complete OOF `ExpertEvidence` archive on the shared Phase 4 fold assignment. For iterative experts, set `finalize_epochs` to the median selected epoch across their pre-registered multi-seed OOF runs and use their pre-registered primary seed; for non-iterative experts such as the IMU RF, use the frozen deterministic fit seed and full train-14 fitting policy. Record the policy and source values before finalization.

Finalize each retained expert on all usable train-14 rows, predict only its usable outer-four rows, and serialize a separate `role=heldout` archive with no labels key. Quarantine all five archives exactly like the Phase 5 IR archive: before Phase 10 only hashes, row counts, schema, finite outputs, availability, and routing may be inspected. The Phase 6 exit gate fails unless all six retained experts now have both leakage-free OOF evidence and structurally label-free held-out evidence.

- [ ] **Step 6: Commit candidate code/config/reports only**

Do not commit weights or prediction arrays. Commit source, fixed configs, tests, and small reports with one commit per accepted modality candidate.

## Phase 7: Build the Canonical Sparse Evidence Registry

### Task 10: Outer-Align Six Leakage-Free OOF Archives

**Files:**
- Create: `src/fusion/sparse_evidence_registry.py`
- Create: `scripts/build_sparse_evidence_registry.py`
- Create: `scripts/build_nested_fusion_evidence.py`
- Create: `tests/test_sparse_evidence_registry.py`
- Create: `tests/test_nested_fusion_evidence.py`
- Create at runtime: `outputs/sparse_evidence_registry/<run-id>/registry.npz`
- Create: `reports/sparse_evidence_registry_audit.md`

**Interfaces:**
- Consumes: canonical 3,036-trial manifest, fixed OOF fold assignment, and sparse OOF/held-out evidence from six experts.
- Produces: a labeled global `oof_train14` registry, a structurally label-free `heldout` registry, and three outer-fold nested fusion evidence packages. It never invents predictions for missing evidence. The global OOF registry supports complementarity analysis, contracts, and final train-14 fusion refit; by itself it is not sufficient for unbiased stacker-level validation.

- [ ] **Step 1: Preserve strict alignment and write RED outer-alignment tests**

```python
def test_outer_alignment_fills_neutral_values_and_false_availability() -> None:
    registry = outer_align_evidence(canonical_ids=("a", "b"), evidence=only_a())
    assert registry.logits.shape == (2, 1, 40)
    assert registry.availability[:, 0].tolist() == [True, False]
    assert not registry.logits[1, 0].any()

def test_outer_alignment_rejects_unknown_ids_and_label_mismatch() -> None:
    with pytest.raises(ValueError, match="unknown sample"):
        outer_align_evidence(canonical_ids=("a",), evidence=only_b())

def test_serialized_heldout_registry_has_no_labels_key(tmp_path: Path) -> None:
    path = save_registry(tmp_path, role="heldout", canonical=label_free_canonical())
    with np.load(path) as data:
        assert "labels" not in data.files
```

- [ ] **Step 2: Implement canonical registry state**

Use fixed expert order `IR, Depth_Color, Thermal, IMU, Skeleton, Radar`. Both registry roles store canonical IDs/users, `logits=[N,6,40]`, `availability=[N,6]`, modality-native quality payloads/masks, `fusion_quality_score=[N,6]`, optional embedding/summary references, fold lineage, expert hashes, and evidence hashes. Only `registry_role=oof_train14` stores canonical labels and OOF fold IDs. Before building `registry_role=heldout`, project the canonical index to label-free ID/user/missingness fields so the builder never receives `class_id`; the serialized registry must not contain a labels key or accept expert evidence carrying labels. Neutral values are storage placeholders and may never be treated as available evidence.

- [ ] **Step 3: Audit present versus usable**

Compare raw manifest directory presence with evidence usability for every expert. Report present-but-unusable rows, reasons, and modality-pattern counts for train-14 and held-out-4 separately. Require every usable train row to have exactly one user-held-out OOF prediction.

- [ ] **Step 4: Quarantine held-out registry**

Write training OOF and held-out registries to separate paths and metadata domains. Fusion fitting APIs accept only `registry_role=oof_train14`; held-out loading is evaluation-only and raises if passed to calibration or training functions. Tests must inspect the serialized held-out registry and prove that no `labels` field exists.

- [ ] **Step 5: Build nested base evidence for unbiased fusion evaluation**

Use the Phase 4 three-fold user assignment as the outer fusion folds. For each outer fold:

1. Remove the outer-validation users from every expert's training universe before any preprocessing or base-model fit.
2. Within the remaining outer-train users, create a deterministic three-fold inner user assignment using seed `20260715 + outer_fold_index`; persist and hash it.
3. For each expert, generate fusion-training evidence by inner user-OOF: fit only on the other outer-train users and predict the inner-validation users. Concatenate these predictions into a sparse nested training registry whose base features depend only on outer-train data.
4. Finalize each expert on all outer-train users using its frozen finalization policy, then predict the untouched outer-validation users into a label-free nested validation registry.
5. Record model, preprocessing, inner-fold, sample, user, class-map, and evidence lineage. Assert that no base model used to create either nested training or outer-validation features saw an outer-validation user's labels or raw training rows.

The output for each outer fold is `(nested_train_evidence, outer_validation_evidence)`. Phase 8/9 must consume these packages for reported A/D evaluation. They may not substitute the cheaper global train-14 OOF registry.

- [ ] **Step 6: Verify and commit**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_expert_evidence_contract.py tests/test_sparse_evidence_registry.py tests/test_nested_fusion_evidence.py tests/test_expert_contract.py -v`

Expected: PASS with canonical row membership, sparse masks, provenance, fold lineage, and held-out quarantine audited.

```bash
git add src/fusion/sparse_evidence_registry.py scripts/build_sparse_evidence_registry.py scripts/build_nested_fusion_evidence.py tests/test_sparse_evidence_registry.py tests/test_nested_fusion_evidence.py reports/sparse_evidence_registry_audit.md
git commit -m "Build canonical sparse expert registry"
```

## Phase 8: Fit the Safe Available-Expert Anchor

### Task 11: Calibrate and Mix Every Usable Expert Subset

**Files:**
- Create: `src/fusion/calibrated_anchor.py`
- Create: `scripts/fit_calibrated_anchor.py`
- Create: `tests/test_calibrated_anchor.py`
- Create at runtime: per-fold calibration parameters, cross-fitted OOF anchor predictions, and final train-14 refit parameters.
- Create: `reports/six_modal_anchor_report.md`

**Interfaces:**
- Consumes: the three Phase 7 nested `(outer_train_inner_oof, outer_validation)` evidence packages for unbiased reporting, plus the global train-14 OOF registry only for the final frozen refit.
- Produces: cross-fitted anchor A predictions for unbiased train-14 reporting, final train-14-refit scalar temperatures/weights for later frozen evaluation, and anchor probabilities for any non-empty usable expert subset.

- [ ] **Step 1: Write RED anchor invariance tests**

```python
def test_single_available_expert_is_recovered_exactly() -> None:
    result = anchor(logits, availability=only_expert_two(), temperatures, weights)
    expected = softmax(logits[:, 2] / temperatures[2], dim=-1)
    torch.testing.assert_close(result, expected, atol=0.0, rtol=0.0)

def test_no_usable_expert_fails_explicitly() -> None:
    with pytest.raises(NoUsableExpertError):
        anchor(logits, availability=torch.zeros(1, 6, dtype=torch.bool), ...)

def test_crossfit_calibration_excludes_validation_users() -> None:
    result = crossfit_anchor(registry_fixture(), frozen_user_folds())
    assert all(set(f.fit_users).isdisjoint(f.validation_users) for f in result.fold_lineage)
```

- [ ] **Step 2: Implement masked calibrated mixture**

Fit one positive scalar temperature per expert from its OOF rows. Convert IMU RF probabilities to clipped log-probability inputs before the same scalar calibration. Parameterize six global weights with softmax for non-negativity and identifiability, then renormalize weights over the usable subset per trial.

- [ ] **Step 3: Fit without held-out leakage**

For each outer fusion fold, fit all six temperatures and global expert weights only on that fold's nested outer-train inner-OOF evidence, freeze them, and predict the corresponding outer-validation evidence produced by base experts trained only on outer-train users. Every reported validation prediction must therefore use both base features and anchor parameters created without that outer-validation user's labels or training rows. Concatenate the three outer-validation partitions into `anchor_oof_crossfit.npz` and compute all Phase 8 labeled metrics only from these predictions.

Predeclare the anchor form, optimization, and metric tie-breaking as Macro-F1, worst-user Accuracy, then smaller calibration complexity before nested cross-fitting. Only after those choices are frozen may the same form be refit on the global train-14 OOF registry to produce the final anchor parameters consumed by Phase 10. Never substitute global-OOF or all-train-refit predictions into the nested Phase 8 report.

- [ ] **Step 4: Stress every train-14 observed pattern and every singleton**

Evaluate labeled behavior on train-14 cross-fitted presence patterns and add synthetic singleton fixtures for all six experts. Report OOF Accuracy, Macro-F1, worst-user, calibration error, pattern counts, failures, latency, and anchor-versus-best-single-expert deltas. Before Phase 10, the structurally label-free held-out registry may be checked only for schema, finite outputs, routing integrity, and unlabeled pattern counts. Rare train patterns remain diagnostic and do not drive selection.

- [ ] **Step 5: Freeze A as the permanent baseline**

Save temperatures, weights, class hash, expert hashes, registry hash, code SHA, and exact serialized bytes. The anchor must remain independently runnable even if Phase 9 is rejected.

- [ ] **Step 6: Verify and commit**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_calibrated_anchor.py tests/test_sparse_evidence_registry.py -v`

```bash
git add src/fusion/calibrated_anchor.py scripts/fit_calibrated_anchor.py tests/test_calibrated_anchor.py reports/six_modal_anchor_report.md
git commit -m "Add safe six-modal probability anchor"
```

## Phase 9: Test the Optional Residual Set Mixer

### Task 12: Compare D Against the Frozen A Anchor

**Files:**
- Create: `src/fusion/residual_set_mixer.py`
- Create: `scripts/train_residual_set_mixer.py`
- Create: `configs/experiments/six_modal_residual_mixer.yaml`
- Create: `tests/test_residual_set_mixer.py`
- Create: `reports/six_modal_residual_mixer_report.md`

**Interfaces:**
- Consumes: the frozen anchor form, exact Phase 7 nested fusion evidence packages and outer folds, the global train-14 OOF registry only for final refit, modality-specific token adapters, availability, native label-free quality, and normalized `fusion_quality_score`. The global-OOF final anchor fit is deployment-only and may not be used to create Phase 9 validation inputs.
- Produces: an optional `delta_logits=[B,40]`, deterministic `g(A,Q)`, and a retain/reject decision against anchor A.

- [ ] **Step 1: Write RED structural-invariant tests**

```python
def test_zero_initialized_mixer_exactly_recovers_anchor() -> None:
    mixer = tiny_mixer(zero_initialize_output=True)
    final = mixer.apply(anchor_probabilities, evidence_tokens, availability, fusion_quality_score)
    torch.testing.assert_close(final, anchor_probabilities, atol=0.0, rtol=0.0)

def test_singleton_and_unsupported_pattern_disable_residual() -> None:
    assert deterministic_support_gate(singleton_availability(), fusion_quality()).item() == 0
    assert deterministic_support_gate(rare_pair_availability(), fusion_quality()).item() == 0

def test_a_and_d_share_outer_folds_without_validation_label_access() -> None:
    result = crossfit_anchor_and_residual(registry_fixture(), frozen_user_folds())
    assert result.anchor_validation_ids == result.residual_validation_ids
    assert all(set(f.fit_users).isdisjoint(f.validation_users) for f in result.fold_lineage)
```

- [ ] **Step 2: Implement modality-specific token adapters**

Neural experts may use logits, embeddings, native quality, and `fusion_quality_score`. IMU RF may use logits, engineered summaries, native quality, and the same normalized scalar. Project each available expert to a fixed 64-dimensional token with a modality ID. Padded/missing tokens are masked and may not affect normalization or attention.

- [ ] **Step 3: Implement the tiny correction model**

Use either DeepSets/gated MLP or at most two masked self-attention layers, chosen before viewing held-out results. Feed anchor log-probability as context and emit 40 residual logits through a zero-initialized final layer. Enforce:

```text
z_final = log(clamp(p_anchor)) + g(A,Q) * lambda * delta_logits
```

Keep `lambda` separate from `g`. The first `g` is deterministic: zero for fewer than two usable experts, zero for modality patterns whose support is below 16 in the current outer fusion-train nested evidence, otherwise multiplied by the mean available `fusion_quality_score` in `[0,1]`. Never compute pattern support from complete train-14 before applying the gate to an outer-validation fold. Never average or directly compare modality-native quality-vector coordinates. The normalized scalar must come from each expert's pre-registered label-free mapping and may not be refit inside fusion CV.

- [ ] **Step 4: Build the restricted training distribution**

Within each outer fold, use natural patterns from that fold's nested outer-train evidence with at least two usable experts and outer-train-local support at least 16. From its all-six rows, drop exactly one randomly selected modality with probability 0.15; weight this synthetic-dropout loss by 0.25 relative to natural-pattern loss. Do not synthesize arbitrary pairs/singletons. Seed sampling by `20260715 + outer_fold_index + epoch + sample_index`. For the final global refit only, recompute support from the complete global train-14 OOF registry.

- [ ] **Step 5: Select residual strength without held-out users**

Cross-fit A and D on exactly the same outer user folds and Phase 7 nested evidence. For each outer fusion fold: fit A temperatures/weights on the nested outer-train inner-OOF evidence; choose residual early stopping and `lambda` from `[0.0, 0.25, 0.5, 1.0]` through a user-grouped split contained entirely inside that nested outer-train evidence; refit the selected residual on all nested outer-train evidence with that fold-specific A; freeze both; then predict the untouched outer-validation evidence produced by base experts finalized only on outer-train users. No base-model training row, temperature, weight, adapter, residual parameter, pattern-support count, early-stopping decision, or `lambda` may use an outer-validation user's label or training row.

Concatenate the outer-fold A and D predictions and compute D-minus-A only from these paired nested cross-fitted rows. After the architecture, stopping rule, and `lambda` are frozen, refit final A and then final D on the global train-14 OOF registry for deployment. Never use the global-OOF final A as input to the reported residual evaluation, and never use the four held-out users for fitting or selection.

- [ ] **Step 6: Apply the D-versus-A retention rule**

Retain D only when cross-validated Accuracy and Macro-F1 point deltas are positive, worst-user Accuracy does not materially regress, all-six/common-pattern performance improves or holds, and singleton/rare patterns exactly recover A. Otherwise discard the mixer checkpoint and freeze A as the final fusion architecture.

- [ ] **Step 7: Verify and commit**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_residual_set_mixer.py tests/test_calibrated_anchor.py tests/test_sparse_evidence_registry.py -v`

```bash
git add src/fusion/residual_set_mixer.py scripts/train_residual_set_mixer.py configs/experiments/six_modal_residual_mixer.yaml tests/test_residual_set_mixer.py reports/six_modal_residual_mixer_report.md
git commit -m "Evaluate residual multimodal correction"
```

## Phase 10: Assemble and Audit the Final Inference Model

### Task 13: Build Raw-Trial Routing, Held-Out Evaluation, and Production Refit

**Files:**
- Create: `src/inference/six_modal_pipeline.py`
- Create: `scripts/audit_six_modal_inference.py`
- Create: `tests/test_six_modal_inference.py`
- Create: `docs/six_modal_inference_contract.md`
- Create at runtime: `reports/six_modal_heldout_evaluation.md`
- Create at runtime: `reports/six_modal_deployment_audit.md`
- Create at runtime: exact deployment manifest.

**Interfaces:**
- Consumes: raw non-empty modality subsets, six finalized experts, safe anchor, optional retained mixer, and exact preprocessing artifacts.
- Produces: one audited trial probability or an explicit no-usable-expert failure, followed by a production package after architecture freeze.

- [ ] **Step 1: Write RED end-to-end routing tests**

Cover all-six input, each six singleton input, observed missing patterns, present-but-unusable Radar, IR locator failure, sensor-only input, and empty input. If at least one expert becomes usable, output must be finite `[1,40]`; if none does, return `NoUsableExpertError` with per-modality reasons.

- [ ] **Step 2: Implement modality-isolated preprocessing and routing**

Run each supplied modality independently. Execute YOLO on IR only when the verified IR route is used; reuse coordinates for Depth only under the registered parity contract. Record raw presence, usability, quality, hashes, preprocessing latency, expert latency, anchor probability, residual gate, and final probability.

- [ ] **Step 3: Evaluate the four held-out users once**

Load only frozen 14-user final experts, OOF-fitted calibration/weights, the retained-or-rejected Phase 9 decision, and structurally label-free held-out predictions. Load held-out labels from a separate sealed evaluation source only inside this Phase 10 command, join by `sample_id`, and never write those labels back into `ExpertEvidence` or the held-out registry. Report overall, per-user, per-class, and per-missing-pattern Accuracy/Macro-F1, calibration, common-path latency, extreme-fallback failures, and D-versus-A deltas. Do not modify the system after viewing this result except to fix a demonstrated implementation defect with a new audit trail.

- [ ] **Step 4: Audit the exact complete inference package**

Sum unique serialized bytes for YOLO, X3D/head, Depth, Thermal, Skeleton, Radar, IMU RF, IMU imputer, calibration, adapters, optional residual mixer, and learned preprocessing. Require `<95,000,000` bytes, record `complete_submission_size_gate_passed`, SHA-256, and license/provenance for every artifact, and fail on duplicate or undeclared weights. This is the first final-package use of that flag. Measure CPU/GPU memory and latency by modality pattern and trial duration.

- [ ] **Step 5: Freeze the evaluation architecture before production refit**

After the held-out report and retain/reject decisions are immutable, generate a new deterministic three-fold user-grouped OOF registry across all 18 training users using the frozen expert architectures. Refit only the already-approved calibration form and residual form, then train each expert on all usable data from all 18 users. Architecture, features, hyperparameters, thresholds, and expert membership may not change.

- [ ] **Step 6: Audit the all-18-user production package**

Reload every artifact in a clean process, repeat missing-subset contract tests, verify class-map/sample schema, run raw-training fixtures only, and remeasure exact bytes and latency. Competition test data may be read only after this production gate passes and all experiment choices are frozen.

- [ ] **Step 7: Commit code and small audit reports**

Do not commit weights, test predictions, NPZ registries, or submissions.

```bash
git add src/inference/six_modal_pipeline.py scripts/audit_six_modal_inference.py tests/test_six_modal_inference.py docs/six_modal_inference_contract.md reports/six_modal_heldout_evaluation.md reports/six_modal_deployment_audit.md
git commit -m "Assemble audited six-modal inference pipeline"
```

## Deliberately Deferred Work

- A globally sampled single 13-frame trial ablation.
- Three deterministic temporal views per local window as a standalone test-time-augmentation ablation.
- Learned clip-level attention, Top-k, or LogSumExp trial aggregation; the first run uses masked mean probabilities.
- `ir_context + ir_relation` shared-backbone fusion.
- Direct YOLO keypoint-vector classification.
- YOLO keypoint motion-peak clip sampling, because the original pose-track NPZ is not currently present.
- Depth ordinal input.
- An aligned IR+Depth co-expert, pending OOF complementarity and byte-budget evidence.
- Reusing IR ROI coordinates for Thermal without a passed registration audit.
- VideoMAE as a final inference expert, pending written organizer approval for the exact checkpoint and pretraining source.
- High-priority optional VideoMAE-to-X3D teacher-assisted ablation only after standalone pure X3D-S completes Phase 4 and its failure modes are understood. It is not an automatic next phase, must not replace the independently evaluated pure-X3D route by default, and requires a separate pre-registration. The host clarification permits distillation, but only the compliant student may ship; register it as `ir_x3d_s_teacher_assisted` without overwriting `ir_x3d_s_k400_pure`.
- A learned per-sample residual-support gate beyond the deterministic first-run `g(A,Q)`.
- Competition-test inference or submission generation before the Phase 10 production gate.
- DataLoader worker and batch-size optimization.
- Quantization and pruning.

These items require separate pre-registered experiments after the relevant IR-expert or six-modal program gate passes.
