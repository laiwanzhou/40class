# X3D-S Adaptive Multi-Clip Visual Expert Implementation Plan

> **For implementation:** Use `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rules-audited, Kinetics-pretrained X3D-S IR-context video expert that represents variable-duration trials with length-adaptive local clips, keeps its complete reproducible inference stack conservatively below the Small Model Track size limit, and emits fusion-ready trial-level outputs.

**Architecture:** Reuse the existing YOLO-derived `ir_context` image sequences and quality metadata for training, while retaining an auditable raw-trial-to-YOLO-to-ROI inference path for verification and test data. Replace the MobileNet-plus-TCN path with a variable number of local 13-frame X3D-S clips: split each complete trial into length-adaptive contiguous windows, encode every valid clip with one shared X3D-S, and aggregate clip probabilities and embeddings into one trial output. Wrap the result in the existing `ExpertOutput` and `ExpertBatchResult` contracts. VideoMAE is excluded from the submitted inference stack until the organizer explicitly confirms it; it may be used later as a distillation teacher because the competition host has explicitly allowed knowledge distillation.

**Tech Stack:** Python 3, PyTorch 2.7, torchvision 0.22, PyTorchVideo 0.1.5 or a compatibility-verified official PyTorchVideo revision, pandas, OpenCV/Pillow, NumPy, scikit-learn, pytest, CUDA BF16 AMP.

## Global Constraints

- Work only in `D:\work\2026.7.14_kaggle\40class-ir-primary-interaction-wt`.
- Do not read competition test data and do not generate a submission.
- Treat the official challenge page and the competition host's Kaggle clarification as the compliance basis: lightweight pretrained CNNs and knowledge distillation are allowed, while large pretrained foundation backbones are prohibited.
- Record the official rule page, the host clarification URL, access date, and verbatim model-size statement in every compliance report.
- Use the existing `combined_frame_manifest.csv`; do not regenerate the full ROI dataset in the first experiment.
- The first scientific run uses only `ir_context_path`; `ir_relation`, left/right ROI, Depth, Skeleton, and IMU are excluded.
- Preserve the existing train/validation users and `sample_id` membership exactly.
- Never split or shuffle individual frames across train and validation.
- Each local X3D input is `[3,13,182,182]`; grayscale IR is repeated to three channels.
- Use `K = min(8, max(1, ceil(num_frames / 32)))` local clips per trial. Split the complete ordered trial into `K` contiguous, near-equal windows before sampling 13 frames inside each window.
- Treat 13 as the local pretrained clip length, never as a forced representation of the entire trial.
- Apply identical spatial augmentation to every frame in a clip.
- Train with one stochastic temporal view per local window; validate the first experiment with one deterministic midpoint view per local window. Aggregate clips before computing trial-level metrics.
- Keep three-view temporal test-time augmentation out of the first experiment so adaptive coverage and view augmentation remain separate variables.
- Store trial-level logits, embeddings, labels, users, qualities, `sample_ids`, and `class_map_hash`.
- Do not choose future fusion temperatures or weights on the four held-out validation users.
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
| Scheduler | cosine over the full 30-epoch horizon |
| Epochs / patience | 30 / 8 |
| Batch / accumulation | at most 2 trials and 8 valid clips per microbatch / accumulate 4 microbatches |
| AMP | BF16 on CUDA |
| Gradient clipping | 1.0 |
| Checkpoints | best Accuracy and best Macro-F1 |
| Primary report metrics | Accuracy, Macro-F1, worst-user Accuracy, per-class recall, class coverage |

---

## Execution Phases

Execute phases in order. A phase may start only after the previous phase's exit gate passes; update the status table and attach the named evidence before proceeding.

| Phase | Scope | Entry condition | Exit gate | Status |
|---|---|---|---|---|
| Phase 0: Compliance and runtime | Task 1 | New branch/worktree ready | Rule record complete; official pretrained X3D forward passes; complete inference stack `<95,000,000` bytes | Completed (`c088db0`) |
| Phase 1: Temporal data contract | Task 2 | Phase 0 passes | Adaptive window boundary tests, real duration audit, determinism, padding and leakage tests pass | Pending |
| Phase 2: Expert and trainer | Tasks 3-4 | Phase 1 passes | Trial-level masked aggregation, gradients, archive schema and focused tests pass | Pending |
| Phase 3: End-to-end verification | Task 5 | Phase 2 passes | Online/offline ROI parity, shortest/longest trial smoke, overfit test, size audit and full tests pass | Pending |
| Phase 4: Matched scientific evaluation | Task 6 | Phase 3 passes | Pre-registered run completes; paired comparison and duration/user/class reports generated | Pending |
| Phase 5: Fusion handoff | Task 7 | Phase 4 decision retains X3D | Fusion contract, alignment tests and aggregate submission-budget gate pass | Pending |

At each phase boundary, record the Git SHA, changed files, executed commands, pass/fail evidence, unresolved risks, and the next phase decision in `reports/x3d_s_phase_status.md`. Do not silently continue after a failed exit gate.

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

Initialize `reports/x3d_s_phase_status.md` with the six phases above, all statuses set to `Pending`, plus empty fields for Git SHA, evidence commands, artifacts, risks, and next decision. Set Phase 0 to `In progress` only when implementation begins.

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

- [ ] **Step 1: Write failing tests for grouping and leakage protection**

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

- [ ] **Step 2: Write failing tests for deterministic single-view validation**

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
    assert item["source_indices"].min().item() == 0
    assert item["source_indices"].max().item() == 235
```

- [ ] **Step 3: Run the dataset tests and verify failure**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_clip_dataset.py -v`

Expected: FAIL because `X3DClipDataset` is absent.

- [ ] **Step 4: Implement strict manifest indexing**

`X3DClipDataset` must validate all required columns, validate the global 40-class map, reject duplicated `(sample_id, source_frame_index)` rows, reject non-contiguous frame order, reject a `sample_id` appearing on both split sides, and verify every selected image exists.

Build one sample per `sample_id`. Never treat individual frames or temporal views as independent labels.

- [ ] **Step 5: Implement length-adaptive trial partitioning**

For a trial with `L` ordered frames, compute:

```python
K = min(8, max(1, math.ceil(L / 32)))
```

Partition all `L` frames into `K` contiguous, non-empty, near-equal windows whose union covers the complete trial exactly and whose order is preserved. Do not derive `K` from the action label. The observed 236-frame maximum therefore emits eight windows of roughly 29-30 frames instead of one globally subsampled clip.

Return `window_bounds`, `num_frames`, `num_clips`, and `temporal_coverage_fraction`. Unit-test boundary lengths `1`, `13`, `32`, `33`, `64`, `65`, and `236`.

- [ ] **Step 6: Implement temporal sampling inside each local window**

Use 13 temporal bins inside each local window:

- Training: sample one index uniformly inside each bin using a generator derived from `seed + epoch + dataset_index + window_index`.
- Validation: choose each bin's midpoint and emit exactly one deterministic view per window in the first-run configuration.
- Windows shorter than 13 frames: allow repeated source indices; report per-clip and trial-level `unique_frame_fraction` in quality.
- Never interpolate or synthesize pixels between frames.

Expose `dataset.set_epoch(epoch)` so training views change deterministically by epoch.

- [ ] **Step 7: Implement temporally consistent transforms**

Load `ir_context_path` as grayscale, repeat it to three channels, and apply one spatial parameter set to all 13 frames:

- Training: random resized crop to `182 x 182`, scale `[0.8, 1.0]`, ratio `[0.9, 1.1]`, horizontal flip probability `0.5`.
- Validation: resize shorter side to 200 and center crop `182 x 182`.
- Normalize with the exact mean/std attached to the selected official X3D-S weights.

Return clips as `[K, views, 3, 13, 182, 182]`. Spatial randomness may differ between local windows but must be identical across all 13 frames of one clip.

- [ ] **Step 8: Pad variable clip counts without changing trial semantics**

`collate_x3d_clips()` must pad only the `K` dimension and return `[B,K_max,views,3,13,182,182]`, `clip_mask=[B,K_max]`, and padded `source_indices=[B,K_max,views,13]`. Padded clips must never enter probability or embedding aggregation and should be skipped before X3D execution when practical.

- [ ] **Step 9: Emit fusion-ready metadata**

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

- [ ] **Step 10: Run dataset and existing contract tests**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_clip_dataset.py tests/test_expert_contract.py -v`

Expected: PASS.

- [ ] **Step 11: Commit the dataset**

```bash
git add src/data/x3d_clip_dataset.py tests/test_x3d_clip_dataset.py
git commit -m "Add trial-safe adaptive X3D clip dataset"
```

## Phase 2: Expert and Trainer

### Task 3: Wrap X3D-S as a Fusion-Compatible Visual Expert

**Files:**
- Create: `src/models/x3d_s_visual_expert.py`
- Create: `tests/test_x3d_s_visual_expert.py`
- Modify: `src/models/__init__.py`

**Interfaces:**
- Consumes: `[B,K,3,13,182,182]` local clips, `clip_mask=[B,K]`, and fixed trial-quality tensors.
- Produces: `ExpertOutput(main_logits=[B,40], embedding=[B,256], quality, quality_mask, availability)`.

- [ ] **Step 1: Write the failing model contract test**

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

- [ ] **Step 2: Run the test and verify failure**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_visual_expert.py -v`

Expected: FAIL because the wrapper is absent.

- [ ] **Step 3: Implement the wrapper without forking the expert contract**

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

- [ ] **Step 4: Implement optimizer parameter groups**

Expose:

```python
def parameter_groups(self, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict[str, object]]
```

The pretrained backbone uses `3e-5`; embedding and classifier heads use `3e-4`. Bias and normalization parameters use zero weight decay; other parameters use `0.05`.

- [ ] **Step 5: Add a frozen-backbone warmup switch**

Expose `set_backbone_trainable(enabled: bool)`. Epochs 1-2 train only the new embedding and classifier heads; epoch 3 onward trains the full model. BatchNorm layers in a frozen backbone must remain in evaluation mode.

- [ ] **Step 6: Run model and expert-contract tests**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_visual_expert.py tests/test_expert_contract.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the model wrapper**

```bash
git add src/models/x3d_s_visual_expert.py src/models/__init__.py tests/test_x3d_s_visual_expert.py
git commit -m "Add fusion-compatible X3D-S visual expert"
```

### Task 4: Add the X3D-S Training and Evaluation Entry Point

**Files:**
- Create: `src/train_x3d_s_visual_expert.py`
- Create: `configs/experiments/x3d_s_ir_context_fold0.yaml`
- Create: `tests/test_x3d_s_trainer_contract.py`

**Interfaces:**
- Consumes: `X3DClipDataset`, `X3DSVisualExpert`, and the fixed YAML configuration.
- Produces: checkpoints, `history.csv`, per-class metrics, trial-level prediction archives, `run_summary.json`, and fusion-ready `ExpertBatchResult` data.

- [ ] **Step 1: Write failing trainer contract tests**

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

- [ ] **Step 2: Run the trainer tests and verify failure**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_trainer_contract.py -v`

Expected: FAIL because the trainer helpers are absent.

- [ ] **Step 3: Implement configuration and CLI validation**

Support:

```text
--config
--smoke-test
--run-id
--epochs
--max-train-batches
--max-val-batches
```

Reject any config with a non-IR-context first-run view, a class count other than 40, local clip length other than 13, `target_window_frames` other than 32, `max_clips` other than 8, `val_views_per_window` other than 1, nonpositive temperatures or learning rates, or an output directory that would overwrite an existing run.

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
```

- [ ] **Step 4: Implement training**

Use AdamW parameter groups from the model, two-epoch linear warmup, cosine decay over 30 epochs, BF16 autocast, gradient accumulation 4, gradient clipping 1.0, and deterministic seeding. Train the head only for epochs 1-2 and unfreeze the backbone at epoch 3.

Call `train_dataset.set_epoch(epoch)` before every training epoch.

Flatten only valid `[B,K]` clips through X3D, restore the trial structure, aggregate clip probabilities, and compute NLL from the aggregated trial log-probabilities. Do not treat clips as independent labeled samples and do not add a per-clip auxiliary loss in the first run. Use a deterministic batch sampler that admits at most two trials and eight valid clips per microbatch; an eight-clip trial forms a batch by itself, and no window or trial is dropped.

- [ ] **Step 5: Implement trial-level validation**

For the first run, `V` is fixed to 1. Flatten only valid entries from `[B,K,1,C,T,H,W]` to `[num_valid_clips,C,T,H,W]`, run X3D, and restore clip logits to `[B,K,40]`. Convert each valid clip to probability, masked-average over `K`, and take `log(clamp_min(1e-8))` as the stored trial logits. Average embeddings over valid clips and L2-normalize the final 256-dimensional trial embedding. Do not generate or average additional validation views in this phase.

Compute metrics only after this aggregation. Assert that every `sample_id` contributes exactly one trial prediction regardless of `K`.

- [ ] **Step 6: Save both checkpoint objectives**

Save:

- `best_accuracy.pt`
- `best_macro_f1.pt`
- `val_predictions_best_accuracy.npz`
- `val_predictions_best_macro_f1.npz`

Each NPZ must contain exactly one row per validation `sample_id`. Validate uniqueness, finite values, `[N,40]` logits, `[N,256]` embeddings, the same class-map hash as the dataset, and diagnostic `num_frames=[N]` and `num_clips=[N]` arrays. These diagnostic arrays must not be treated as action evidence by later fusion.

- [ ] **Step 7: Record scientific and resource metrics**

`history.csv` and `run_summary.json` must include train/val Accuracy, Macro-F1, weighted F1, loss, class coverage, zero-recall classes, worst-user Accuracy, learning rates, epoch duration, peak CUDA memory, parameter count, state-dict bytes, checkpoint file sizes, mean/max clips per trial, processed clips per second, and trial-level latency by length bucket.

`run_summary.json` must also embed the compliance manifest: X3D weight hash and bytes, YOLO weight hash and bytes, custom-head bytes, combined serialized inference-weight bytes, the `95,000,000`-byte internal limit, pretrained source, and an explicit `submission_size_gate_passed` boolean. A run that fails the size gate may be used for diagnosis but must not be marked submission-eligible.

- [ ] **Step 8: Run trainer contract tests**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_trainer_contract.py -v`

Expected: PASS.

- [ ] **Step 9: Commit the trainer**

```bash
git add src/train_x3d_s_visual_expert.py configs/experiments/x3d_s_ir_context_fold0.yaml tests/test_x3d_s_trainer_contract.py
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
- Produces: a pass/fail audit covering split membership, shapes, pretrained loading, gradients, complete inference-stack size, online ROI parity, latency, and fusion contract.

- [ ] **Step 1: Write the real-manifest read-only contract test**

The test must assert:

- exactly 40 classes;
- no train/val user overlap;
- no train/val `sample_id` overlap;
- every selected trial has ordered frames;
- the shortest trial emits `[1,1,3,13,182,182]` and the 236-frame trial emits `[8,1,3,13,182,182]`;
- all window bounds are ordered, non-overlapping, non-empty, and cover every source frame exactly once before within-window sampling;
- validation access is deterministic;
- no path contains a competition-test directory.

- [ ] **Step 2: Run the real-manifest test**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_real_manifest_contract.py -v`

Expected: PASS without writing new data assets.

- [ ] **Step 3: Run a two-sample forward/backward smoke test**

Run:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m src.train_x3d_s_visual_expert `
  --config configs/experiments/x3d_s_ir_context_fold0.yaml `
  --smoke-test `
  --run-id x3d_s_ir_context_adaptive_smoke
```

Expected: train and validation complete, both backbone and head receive finite gradients after unfreezing, checkpoints reload, and prediction archives pass validation.

- [ ] **Step 4: Run a small-subset overfit test**

Use 16 training samples covering at least 8 classes and containing both one-clip and multi-clip trials for 20 epochs with augmentation disabled. Require aggregated trial loss to fall and training Accuracy to exceed 80%. This is an implementation test only; do not report its validation metrics as scientific evidence.

- [ ] **Step 5: Audit model size and fusion contract**

`scripts/audit_x3d_s_run.py` must verify:

- X3D plus custom head parameter bytes;
- whether YOLO weights are required at inference;
- combined size if YOLO is included;
- unique and complete validation sample set;
- exact `class_map_hash` agreement with the existing visual expert;
- successful `align_expert_batch()` against a fixture with shuffled sample order;
- `calibrated_probability_mixture(alpha=0)` exactly recovers X3D probabilities.

The audit must sum the serialized weights of every component needed to transform an official raw trial into final logits. For the initial route this includes YOLO11n-pose, X3D-S, the embedding/classification head, and any learned preprocessing or calibration module. Count deployable files, not conceptual submodules: if the custom head is already inside the final X3D checkpoint, do not add it twice. Require the aggregate to remain below the conservative internal limit of `95,000,000` bytes; do not report only X3D's parameter count.

- [ ] **Step 6: Implement and verify raw-trial online inference parity**

Implement a single inference entry point that accepts an official raw IR trial and performs:

```text
raw IR frames -> existing UltralyticsPoseLocator -> IRPrimaryInputROIBuilder
              -> adaptive contiguous windows -> 13-frame local clips
              -> shared X3D-S -> masked trial aggregation -> trial logits
```

Reuse the existing YOLO pose locator and ROI builder rather than duplicating box-selection logic. On a deterministic representative sample set covering one-, two-, four-, and eight-clip trials, low/high pose reliability, and recovered context cases, compare online boxes and crops with the existing exported `ir_context` assets. Require identical window bounds, sampled frame indices, box coordinates, and either byte-identical crops or a documented pixel tolerance caused only by the image codec. Fail on missing weights, silent full-frame fallback, ordering changes, clip-count drift, or normalization drift.

The parity test uses training data only. Record end-to-end latency separately for YOLO/ROI preprocessing, each X3D clip, and complete trial inference in the `<=13`, `14-32`, `33-64`, and `>64` frame buckets so the report reflects the actual submission path.

- [ ] **Step 7: Run the focused and full test suites**

Run:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_environment_contract.py tests/test_x3d_clip_dataset.py tests/test_x3d_s_visual_expert.py tests/test_x3d_s_trainer_contract.py tests/test_x3d_s_real_manifest_contract.py tests/test_x3d_s_online_roi_parity.py tests/test_expert_contract.py -v
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit smoke evidence**

```bash
git add scripts/audit_x3d_s_run.py src/inference/x3d_s_ir_context_pipeline.py tests/test_x3d_s_real_manifest_contract.py tests/test_x3d_s_online_roi_parity.py reports/x3d_s_smoke_report.md
git commit -m "Verify compliant X3D-S inference end to end"
```

## Phase 4: Matched Scientific Evaluation

### Task 6: Run the Pre-Registered Adaptive Multi-Clip IR-Context Experiment

**Files:**
- Create at runtime: `outputs/x3d_s_ir_context_fold0/<run-id>/...`
- Create: `scripts/summarize_x3d_s_experiment.py`
- Create at runtime: `reports/x3d_s_ir_context_fold0_report.md`
- Create at runtime: `reports/x3d_s_ir_context_fold0_per_class.csv`
- Create at runtime: `reports/x3d_s_ir_context_fold0_per_user.csv`

**Interfaces:**
- Consumes: the verified configuration and real train/validation split.
- Produces: one reproducible adaptive multi-clip X3D-S visual baseline and a decision on whether to continue X3D experiments.

- [ ] **Step 1: Freeze and hash the experiment inputs**

Record the configuration hash, manifest hash, class-map hash, X3D and YOLO pretrained-weight hashes, compliance-document hash, rule-source URLs and access date, Git SHA, environment probe, complete inference-weight byte total, train/val users, trial counts, and seed before looking at validation results. The experiment may start only if the submission-size gate passes.

- [ ] **Step 2: Run the single full experiment**

Run:

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m src.train_x3d_s_visual_expert `
  --config configs/experiments/x3d_s_ir_context_fold0.yaml `
  --run-id x3d_s_ir_context_adaptive_k400_seed20260715
```

Do not change ROI views, local frame count, `target_window_frames=32`, `max_clips=8`, crop size, aggregation method, loss, or `val_views_per_window=1` during this run.

- [ ] **Step 3: Re-evaluate both saved checkpoints**

Reload `best_accuracy.pt` and `best_macro_f1.pt`, regenerate validation outputs, and require exact metric agreement with the values saved during training within `1e-6`.

- [ ] **Step 4: Generate the report**

Compare X3D-S with:

- the teammate VideoMAE result only after confirming identical split and trial-level metric semantics;
- a matched MobileNet/TCN IR-context baseline using the exact same train/validation users, trial `sample_id` set, class map, ROI assets, and trial-level metric implementation; rerun it if the existing result does not satisfy all matching conditions;
- the existing Skeleton TCN result.

Report Accuracy, Macro-F1, worst-user Accuracy, per-user metrics, per-class recall, zero-recall classes, training/generalization gap, complete inference-stack bytes, individual component sizes, checkpoint size, GPU memory, X3D-only latency, YOLO/ROI latency, and end-to-end trial latency.

Also report Accuracy, Macro-F1, sample count, mean clip count, and mean latency separately for trial-length buckets `<=13`, `14-32`, `33-64`, and `>64`. Compare short-action and long-action class recall so a global score cannot hide duration-dependent failure.

For the matched IR-context comparison, align predictions by `sample_id` and run 10,000 paired bootstrap replicates stratified by held-out user. Keep the class label set fixed at all 40 classes in every replicate and report point deltas plus 95% confidence intervals for Accuracy and Macro-F1. This bootstrap quantifies validation-sample uncertainty only; three-seed confirmation is still required before claiming training stability.

- [ ] **Step 5: Apply the pre-registered decision rule**

- **Primary expert candidate:** X3D-S has positive Accuracy and Macro-F1 deltas over the matched IR-context baseline, the paired 95% confidence interval for the Accuracy delta excludes zero, Macro-F1 improves, and worst-user Accuracy does not regress. Run three seeds before fusion and report mean, standard deviation, and per-seed deltas.
- **Promising but unconfirmed:** both point deltas are positive but the Accuracy interval includes zero, or results vary materially by user or duration bucket. Retain the predictions for complementarity analysis and run the three-seed confirmation before expanding inputs.
- **Temporal follow-up only:** the `>64` bucket trails the matched baseline while shorter buckets improve. Keep the visual input fixed and test only `target_window_frames=24` versus 32; do not mix this with three-view TTA.
- **Stop and audit:** Accuracy or Macro-F1 fails to improve against the matched baseline without a clear complementary per-class benefit. Audit normalization, pretrained loading, adaptive window coverage, padded-clip masking, aggregation, and split comparability before architectural expansion.
- Do not add Depth or additional ROI views merely because the first score is low.

- [ ] **Step 6: Commit only code and small reports**

Do not commit weights, NPZ files, or the output directory.

```bash
git add scripts/summarize_x3d_s_experiment.py reports/x3d_s_ir_context_fold0_report.md reports/x3d_s_ir_context_fold0_per_class.csv reports/x3d_s_ir_context_fold0_per_user.csv
git commit -m "Report X3D-S IR-context baseline"
```

## Phase 5: Fusion Handoff

### Task 7: Preserve the Future Fusion Boundary

**Files:**
- Create: `docs/x3d_s_fusion_contract.md`
- Modify only if required by a failing contract test: `src/models/expert_contract.py`
- Modify only if required by a failing contract test: `tests/test_expert_contract.py`

**Interfaces:**
- Consumes: X3D-S and any later rule-approved compact expert's trial-level prediction archives. VideoMAE predictions may be used as teacher targets during training, but VideoMAE weights are excluded from the submitted inference graph unless the organizer gives written approval.
- Produces: one shared contract for later OOF calibration and sequence-level probability fusion.

- [ ] **Step 1: Document the immutable archive schema**

Require every expert to provide:

```text
sample_ids: [N]
user_ids: [N]
labels: [N]
logits: [N,40]
embeddings: [N,D]
quality: [N,Q]
quality_mask: [N,Q]
availability: [N,M]
class_map_hash: scalar string
```

Embedding dimensions may differ between experts. Logits, labels, sample sets, and class-map hashes may not differ.

Experts may append diagnostic arrays such as `num_frames` and `num_clips`; fusion loaders must ignore unknown diagnostic fields after validating the required schema. Such fields cannot be used as action evidence in the first fusion experiment.

- [ ] **Step 2: Verify strict sample alignment**

Use `align_expert_batch()` for all in-memory fusion. For NPZ fusion, build the same explicit `sample_id` lookup, reject duplicates, reject missing/extra samples, and reorder only after validation.

- [ ] **Step 3: Freeze the first fusion method**

The first later fusion experiment is calibrated probability mixture:

```python
px = softmax(x3d_logits / Tx)
ps = softmax(skeleton_logits / Ts)
p = (1.0 - alpha) * px + alpha * ps
```

Learn `Tx`, `Ts`, and `alpha` only from user-grouped cross-fitted OOF predictions generated inside the 14 training users. The four held-out validation users are final evaluation only.

- [ ] **Step 4: Keep YOLO quality separate from action evidence**

X3D quality fields may later gate `ir_relation` or expert availability, but the first fusion gate must not consume embeddings or class logits. Start with a global alpha, then a deterministic quality-only alpha, and only then consider a learned quality gate.

- [ ] **Step 5: Enforce the aggregate submission budget for every fusion candidate**

Before evaluating a fused submission candidate, rebuild the compliance manifest from the exact deployed files and sum all serialized inference-time weights: YOLO, X3D, Skeleton/IMU/other experts, calibration parameters, and learned fusion gates. Require the aggregate to remain below `95,000,000` bytes. Teacher-only models are recorded for provenance but excluded from the total only when their weights and code path are provably absent from inference.

Do not add VideoMAE weights to the inference bundle until the organizer has answered in writing that the specific checkpoint, pretraining source, and parameter scale are allowed in the Small Model Track. That approval is a separate gate from the byte limit.

- [ ] **Step 6: Run contract tests and commit documentation**

Run: `D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_expert_contract.py tests/test_x3d_s_trainer_contract.py -v`

Expected: PASS.

```bash
git add docs/x3d_s_fusion_contract.md src/models/expert_contract.py tests/test_expert_contract.py
git commit -m "Document X3D-S expert fusion contract"
```

## Deliberately Deferred Work

- A globally sampled single 13-frame trial ablation.
- Three deterministic temporal views per local window as a standalone test-time-augmentation ablation.
- Learned clip-level attention, Top-k, or LogSumExp trial aggregation; the first run uses masked mean probabilities.
- `ir_context + ir_relation` shared-backbone fusion.
- Direct YOLO keypoint-vector classification.
- YOLO keypoint motion-peak clip sampling, because the original pose-track NPZ is not currently present.
- Depth ordinal input.
- Skeleton and IMU training or fusion.
- VideoMAE as a final inference expert, pending written organizer approval for the exact checkpoint and pretraining source.
- VideoMAE-to-X3D knowledge distillation as a separate pre-registered teacher/student experiment; the host clarification permits distillation, but only the compliant student may ship.
- Learned per-sample fusion gates.
- Multi-fold or test-set inference.
- DataLoader worker and batch-size optimization.
- Quantization and pruning.

These items require separate pre-registered experiments after the IR-context baseline passes its acceptance gate.
