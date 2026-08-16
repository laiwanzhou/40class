# X3D Train12/Val2 Direct-Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish an unchanged partial2 reference and then implement, pre-register, smoke-test, and run one matched Direct-Head X3D-S experiment on the frozen `12 train / user6-user7 val / 4 heldout` split.

**Architecture:** First rerun the unchanged partial2 recipe on the new full-40-class validation split and freeze that result as the sole matched reference. Preserve the legacy projected head as the default module/state-dict layout, then add a `direct` head that classifies unchanged 2048D X3D outputs through `Dropout(0.25) -> Linear(2048,40)` and aggregates the pre-custom-dropout features into the required 2048D trial embedding. Extend trainer/archive/resource metadata without changing canonical configs or evidence.

**Tech Stack:** Python 3.12, PyTorch 2.7, PyTorchVideo X3D-S, NumPy, pandas, scikit-learn, PyYAML, pytest, CUDA BF16.

## Global Constraints

- Authority: `docs/superpowers/specs/2026-08-16-x3d-train12-val2-direct-head-design.md`; the head design has final review decision `APPROVED`, and the `user6,user7` split amendment is explicitly user-approved.
- Branch: `test/x3d-fold0-generalization`; do not create or switch branches.
- Split: `metadata/splits/train12_val2_user6_user7_development.json`; seed `20260715`; no additional seed.
- Train12: `user1,user2,user3,user5,user8,user9,user16,user18,user19,user20,user21,user22`; val2: `user6,user7`; sealed heldout4: `user4,user17,user23,user24`.
- The split contains 1935 usable-IR train trials and 385 usable-IR validation trials; both populations cover all 40 classes.
- Generation R must rerun the unchanged partial2 recipe on this split and freeze its report before Generation D implements or preregisters Direct-Head.
- Official X3D feature backbone, including internal `Dropout(p=0.5)`, remains unchanged.
- Direct path: `unchanged 2048D X3D output -> custom Dropout(0.25) -> Linear(2048,40)`.
- Direct runtime embedding: normalized trial mean of valid unchanged-X3D outputs before custom dropout; shape `[N,2048]`.
- Restore partial2 shared backbone LR `3e-5`; head LR `3e-4`; weight decay `0.05`; no block LR or L2-SP.
- Two head-only warmup epochs, then block4/block5; identical cosine schedule with `scheduler_horizon_epochs: 20`.
- Full temporal coverage; 13 local frames; target window 32; at most eight clips; no clip dropout or motion peaks.
- Photometric `[0.9,1.1]`; no noise, blur, label smoothing, EMA, or user-adversarial loss.
- `best_accuracy.pt`: Accuracy, fixed-40 Macro-F1, earlier epoch. Early stop: existing `best_macro_f1`, Accuracy tie-break, patience 8.
- Legacy exact-output compatibility is evaluated under `model.eval()` with identical inputs and masks.
- Human-review floor: Direct-Head Accuracy `< new-split partial2 reference Accuracy - 0.02`; bind the resulting numeric threshold in Generation D preregistration, preserve all artifacts, and stop. Never auto-delete.
- Do not access heldout4/test or mutate canonical Phase 4/5 artifacts.

---

### Task 0: Freeze The New Split And Generate Its Partial2 Reference

**Files:**
- Create: `metadata/splits/train12_val2_user6_user7_development.json`
- Create: `configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_partial2.yaml`
- Create: `reports/x3d_s_train12_val2_user6_user7_partial2_preregistration.json`
- Create: `reports/x3d_s_train12_val2_user6_user7_partial2_report.json`
- Create: `reports/x3d_s_train12_val2_user6_user7_partial2_report.md`
- Modify: `scripts/run_x3d_s_train12_val2_dev.py`
- Modify: `scripts/report_x3d_s_train12_val2_dev.py`
- Modify: `tests/test_x3d_s_train12_val2_dev.py`
- Modify: `reports/x3d_s_phase_status.md`

**Interfaces:**
- Consumes: frozen official train14 manifest, unchanged partial2 config, protected development runner, and standalone reporter.
- Produces: a named split profile, an unchanged partial2 run on that profile, and a write-once matched-reference report used by all later tasks.

- [x] **Step 1: Write failing named-split contract tests**

Add fixtures for both historical `train12_val2_development.json` and new `train12_val2_user6_user7_development.json`. Require the runner to accept only an explicitly supported profile whose exact train/validation/heldout users match the file; reject swapped users, overlaps, non-train14 development users, missing heldout users, and mismatched audit counts. Require the reporter to derive its expected validation users from frozen run provenance instead of importing the historical `user21,user22` constant.

- [x] **Step 2: Run RED**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_train12_val2_dev.py -q
```

Expected: fail because the runner and reporter currently hard-code the historical split.

- [x] **Step 3: Implement protected named split profiles**

In `scripts/run_x3d_s_train12_val2_dev.py`, replace the single expected-user constants with immutable profiles keyed by split `name`. Validate exact user tuples, disjoint roles, `development_only: true`, 40-class metric policy, and the new profile's audit values `1935/385/40/40`. Keep the historical profile valid for existing artifacts. In `scripts/report_x3d_s_train12_val2_dev.py`, read the split path/hash and validation users from run provenance and require exact agreement with the supplied reference/candidate artifacts.

- [x] **Step 4: Run GREEN and independently audit the split**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_train12_val2_dev.py -q
D:\Anaconda\envs\pyTorch2.7\python.exe -c "import json; p=json.load(open('metadata/splits/train12_val2_user6_user7_development.json')); assert len(p['train_user_ids'])==12 and p['validation_user_ids']==['user6','user7'] and len(p['heldout_user_ids'])==4 and not (set(p['train_user_ids']) & set(p['validation_user_ids']) or set(p['train_user_ids']) & set(p['heldout_user_ids']) or set(p['validation_user_ids']) & set(p['heldout_user_ids'])); assert p['ir_audit']['train_class_count']==p['ir_audit']['validation_class_count']==40; print('PASS')"
```

- [x] **Step 5: Freeze the unchanged partial2 config and preregistration**

Copy the existing partial2 config without changing model, optimizer, augmentation, temporal sampling, warmup, scheduler, checkpoint selection, or seed. Change only the split identity/output naming needed for `train12_val2_user6_user7`. Create a preregistration that binds SHA-256 for the new split, config, parent partial2 config, canonical artifacts, and sealed heldout boundary; state that no Direct-Head result or implementation is used in Generation R.

- [x] **Step 6: Commit and push Generation R before training**

```powershell
git add metadata/splits/train12_val2_user6_user7_development.json configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_partial2.yaml reports/x3d_s_train12_val2_user6_user7_partial2_preregistration.json scripts/run_x3d_s_train12_val2_dev.py scripts/report_x3d_s_train12_val2_dev.py tests/test_x3d_s_train12_val2_dev.py reports/x3d_s_phase_status.md docs/superpowers/specs/2026-08-16-x3d-train12-val2-direct-head-design.md docs/superpowers/plans/2026-08-16-x3d-train12-val2-direct-head.md
git commit -m "Preregister user6-user7 partial2 reference"
git push origin test/x3d-fold0-generalization
```

- [x] **Step 7: Run the unchanged partial2 reference exactly once**

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.run_x3d_s_train12_val2_dev `
  --config configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_partial2.yaml `
  --development-split metadata/splits/train12_val2_user6_user7_development.json `
  --run-id x3d_s_ir_context_train12_val2_user6_user7_partial2_seed20260715 `
  --seed 20260715
```

Do not inspect heldout4/test, change the recipe, or launch Direct-Head.

- [x] **Step 8: Generate and freeze the matched reference**

Generate `reports/x3d_s_train12_val2_user6_user7_partial2_report.{json,md}` from the best-Accuracy checkpoint and deterministic validation prediction. Record Accuracy, fixed-40 Macro-F1, `min(user6 Accuracy,user7 Accuracy)`, selected epoch, train metrics, resources, config/split/checkpoint/prediction hashes, and the numeric Direct-Head human-review floor `reference_accuracy - 0.02`.

- [x] **Step 9: Verify, commit, and push the reference result**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_train12_val2_dev.py -q
D:\Anaconda\envs\pyTorch2.7\python.exe -m compileall -q scripts tests
git diff --check
git add reports/x3d_s_train12_val2_user6_user7_partial2_report.json reports/x3d_s_train12_val2_user6_user7_partial2_report.md reports/x3d_s_phase_status.md
git commit -m "Freeze user6-user7 partial2 reference"
git push origin test/x3d-fold0-generalization
```

Verify local/origin/remote SHA equality and preserve all reference outputs. Only after this commit may Task 1 begin.

---

### Task 1: Direct-Head Model With Legacy Compatibility

**Files:**
- Modify: `src/models/x3d_s_visual_expert.py`
- Modify: `tests/test_x3d_s_visual_expert.py`

**Interfaces:**
- Consumes: existing `X3DSVisualExpert(...)` and `ExpertOutput`.
- Produces: `X3DSVisualExpert(..., head_type: str = "projected")`, `.head_type`, and `.output_embedding_dim`.

- [x] **Step 1: Write failing construction and shape tests**

```python
direct = X3DSVisualExpert(
    backbone=TinyBackbone(), num_classes=40, embedding_dim=8,
    dropout=0.25, head_type="direct",
)
output = direct.eval()(**fixture_inputs())
assert direct.head_type == "direct"
assert isinstance(direct.embedding_head, nn.Identity)
assert direct.classifier.in_features == 8
assert output.main_logits.shape == (2, 40)
assert output.embedding.shape == (2, 8)
```

Reject `direct` when `embedding_dim != backbone.output_dim`; reject unknown head types.

- [x] **Step 2: Write the failing embedding-boundary and padding tests**

Replace only the custom dropout with a deterministic transform:

```python
class AddOne(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1.0

baseline = direct.eval()(**inputs)
direct.direct_classifier_dropout = AddOne()
changed = direct(**inputs)
torch.testing.assert_close(changed.embedding, baseline.embedding)
assert not torch.equal(changed.main_logits, baseline.main_logits)
```

Parametrize the existing padded-clip invariance test over `projected` and `direct`.

- [x] **Step 3: Write the failing legacy strict-load test**

Create an implicit legacy model and explicit projected model under the same seed:

```python
explicit.load_state_dict(legacy.state_dict(), strict=True)
legacy.eval(); explicit.eval()
torch.testing.assert_close(
    legacy(**inputs).main_logits, explicit(**inputs).main_logits, atol=0, rtol=0
)
torch.testing.assert_close(
    legacy(**inputs).embedding, explicit(**inputs).embedding, atol=0, rtol=0
)
```

- [x] **Step 4: Run RED**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_visual_expert.py -q
```

Expected: fail because `head_type` and `direct_classifier_dropout` do not exist.

- [x] **Step 5: Implement the dual-head model**

Keep projected modules unchanged and add:

```python
self.head_type = str(head_type)
if self.head_type == "projected":
    self.embedding_head = nn.Sequential(
        nn.Linear(resolved_backbone_dim, embedding_dim),
        nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(dropout),
    )
    self.direct_classifier_dropout = nn.Identity()
    classifier_dim = embedding_dim
elif self.head_type == "direct":
    if embedding_dim != resolved_backbone_dim:
        raise ValueError("direct head requires embedding_dim == backbone output_dim")
    self.embedding_head = nn.Identity()
    self.direct_classifier_dropout = nn.Dropout(dropout)
    classifier_dim = resolved_backbone_dim
else:
    raise ValueError("head_type must be projected or direct")
self.output_embedding_dim = classifier_dim
self.classifier = nn.Linear(classifier_dim, num_classes)
```

After the unchanged backbone call:

```python
clip_embeddings = self.embedding_head(clip_features)
classifier_input = (
    self.direct_classifier_dropout(clip_features)
    if self.head_type == "direct" else clip_embeddings
)
clip_logits = self.classifier(classifier_input)
```

Continue mean aggregation and L2 normalization of `clip_embeddings`. Do not modify `build_x3d_s_feature_backbone`.

- [x] **Step 6: Run GREEN and commit**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_visual_expert.py tests/test_expert_contract.py -q
git add src/models/x3d_s_visual_expert.py tests/test_x3d_s_visual_expert.py
git commit -m "Add compatible X3D Direct-Head mode"
```

---

### Task 2: Trainer Provenance, Resources, And Gradient Audit

**Files:**
- Modify: `src/train_x3d_s_visual_expert.py`
- Modify: `tests/test_x3d_s_trainer_contract.py`

**Interfaces:**
- Consumes: model `.head_type` and `.output_embedding_dim`.
- Produces: `_resolved_head_type(config) -> str`, checkpoint/summary/archive metadata, resource counts, archive byte counts, and scoped gradient evidence.

- [x] **Step 1: Write failing config/build tests**

```python
legacy = minimal_config()
validate_config(legacy)
assert _resolved_head_type(legacy) == "projected"

direct = minimal_config()
direct["head_type"] = "direct"; direct["embedding_dim"] = 2048
validate_config(direct)
assert _build_model(direct).head_type == "direct"
```

Reject Direct-Head for non-X3D models, `embedding_dim != 2048`, and unknown strings.

- [x] **Step 2: Write failing provenance/resource tests**

Require checkpoint, archive, and summary evidence:

```python
assert checkpoint["head_type"] == "direct"
assert checkpoint["embedding_dim"] == 2048
assert archive["head_type"].item() == "direct"
assert int(archive["embedding_dim"].item()) == 2048
assert summary["custom_head_parameter_count"] == 2048 * 40 + 40
assert summary["prediction_archive_bytes"]["best_accuracy"] > 0
```

The projected fixture without `head_type` must remain valid.

- [x] **Step 3: Write the failing scoped-gradient test**

Run one optimizer step with final two `BlockBackbone` blocks trainable:

```python
scopes = outcome.metrics["gradient_scopes_with_finite_nonzero"]
assert scopes["backbone_block_2"] is True
assert scopes["backbone_block_3"] is True
assert scopes["classifier"] is True
```

- [x] **Step 4: Run RED**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_trainer_contract.py -q
```

- [x] **Step 5: Implement head resolution and metadata**

```python
def _resolved_head_type(config: Mapping[str, Any]) -> str:
    value = str(config.get("head_type", "projected"))
    if value not in {"projected", "direct"}:
        raise ValueError("head_type must be projected or direct")
    return value
```

Validate Direct-Head only for X3D and embedding 2048. Pass it from `_build_model`. Add it to `_save_checkpoint`, run summary, and `_resource_manifest`.

- [x] **Step 6: Implement archive/resource evidence**

Change the compatible API:

```python
def save_prediction_archive(
    path: Path, result: TrialPredictionResult, *, head_type: str = "projected"
) -> None:
```

Save scalar `head_type` and `embedding_dim`. Pass the resolved type at trainer call sites. Record both archive file sizes after creation. Extend resources with:

```python
"head_type": getattr(model, "head_type", None),
"embedding_dim": int(model.output_embedding_dim),
"custom_head_parameter_count": sum(
    p.numel() for n, p in model.named_parameters() if not n.startswith("backbone.")
),
```

Archive bytes do not enter the serialized route subtotal.

- [x] **Step 7: Implement scoped finite nonzero gradients**

Before gradient clipping, OR finite nonzero evidence for `backbone.blocks.<index>.` and `classifier.` parameters into `gradient_scopes_with_finite_nonzero`. Return it in epoch metrics and write sorted JSON in `_history_row`; retain aggregate flags.

- [x] **Step 8: Run GREEN and commit**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_trainer_contract.py tests/test_x3d_s_visual_expert.py tests/test_x3d_s_train12_val2_dev.py -q
git add src/train_x3d_s_visual_expert.py tests/test_x3d_s_trainer_contract.py
git commit -m "Audit Direct-Head training evidence"
```

---

### Task 3: Frozen Config, Preregistration, And Reporter

**Files:**
- Create: `configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_direct_head1.yaml`
- Create: `reports/x3d_s_train12_val2_user6_user7_direct_head1_preregistration.json`
- Create: `scripts/report_x3d_s_train12_val2_user6_user7_direct_head1.py`
- Create: `tests/test_x3d_s_direct_head1_experiment.py`
- Modify: `reports/x3d_s_phase_status.md`

**Interfaces:**
- Consumes: the frozen new-split partial2 config/report/run from Task 0 and the protected named-split runner.
- Produces: frozen candidate, immutable manifest, and `build_report(...) -> dict[str, Any]`.

- [x] **Step 1: Write failing candidate-isolation tests**

```python
assert candidate["head_type"] == "direct"
assert candidate["embedding_dim"] == 2048
candidate.pop("head_type")
candidate["embedding_dim"] = 256
assert candidate == partial2
```

Also assert no block LR, backbone/head LR `3e-5/3e-4`, warmup 2, horizon 20, full coverage, and protected output root.

- [x] **Step 2: Write failing decision/report tests**

```python
assert evaluate_direct_head_candidate(
    {**ref, "accuracy": ref["accuracy"] - 0.020001}, ref
) == "human_review_regression"
assert evaluate_direct_head_candidate({**ref, "accuracy": 0.56}, ref) == "preferred"
assert evaluate_direct_head_candidate(ref, ref) == "non_winning_ablation"
```

Synthetic aligned artifacts must yield train Accuracy/Macro-F1 gaps, per-user and duration Accuracy/Macro-F1 deltas, NLL/disagreement, direct classifier drift, head parameters, checkpoint/archive/route/CUDA bytes, `head_type=direct`, and embedding 2048.

- [x] **Step 3: Run RED**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_direct_head1_experiment.py -q
```

- [x] **Step 4: Create the exact candidate config**

Copy `x3d_s_ir_context_train12_val2_user6_user7_partial2.yaml` and change only:

```yaml
head_type: direct
embedding_dim: 2048
```

- [x] **Step 5: Create immutable preregistration**

Record candidate/parent/run ID/seed, expected classifier parameters `81960`, the frozen numeric human-review floor from Task 0, three-way decision, non-decision diagnostics, and forbidden actions. Bind SHA-256 for config, new split, new-split partial2 report/checkpoint/prediction, canonical artifacts, approved spec, and all review records. Reject the historical `user21,user22` partial2 report as a matched reference.

- [x] **Step 6: Implement the matched reporter**

Reuse `build_standalone_report` and `_matched_prediction_diagnostics`. Rebuild the seeded Direct model and calculate relative L2 drift for its classifier versus its own initialization; do not compare differently shaped partial2 classifier tensors. Emit JSON/Markdown and state that this is a composite head replacement.

- [x] **Step 7: Run GREEN and validate config**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest tests/test_x3d_s_direct_head1_experiment.py tests/test_x3d_s_train12_val2_dev.py -q
D:\Anaconda\envs\pyTorch2.7\python.exe -c "import yaml; from src.train_x3d_s_visual_expert import validate_config; c=yaml.safe_load(open('configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_direct_head1.yaml', encoding='utf-8')); validate_config(c); print('PASS')"
```

- [x] **Step 8: Record pre-result state and verify**

Append preregistered/no-result/sealed-evidence status. Run full pytest, `git diff --check`, JSON parsing, and config hash checks.

- [x] **Step 9: Commit and push before any result**

```powershell
git add configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_direct_head1.yaml reports/x3d_s_train12_val2_user6_user7_direct_head1_preregistration.json scripts/report_x3d_s_train12_val2_user6_user7_direct_head1.py tests/test_x3d_s_direct_head1_experiment.py reports/x3d_s_phase_status.md docs/superpowers/plans/2026-08-16-x3d-train12-val2-direct-head.md docs/superpowers/specs/2026-08-16-x3d-train12-val2-direct-head-design.md docs/superpowers/reviews/2026-08-16-x3d-direct-head-spec-final-review.md
git commit -m "Preregister Direct-Head X3D experiment"
git push origin test/x3d-fold0-generalization
```

Verify local/origin/remote SHA equality before smoke with `git rev-parse HEAD`, `git rev-parse origin/test/x3d-fold0-generalization`, and `git ls-remote origin refs/heads/test/x3d-fold0-generalization`.

---

### Task 4: CUDA Smoke And Contract Audit

**Files:**
- Create: `reports/x3d_s_train12_val2_user6_user7_direct_head1_smoke_audit.json`
- Modify: `reports/x3d_s_phase_status.md`

- [x] **Step 1: Launch protected smoke**

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.run_x3d_s_train12_val2_dev `
  --config configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_direct_head1.yaml `
  --development-split metadata/splits/train12_val2_user6_user7_development.json `
  --run-id x3d_s_ir_context_train12_val2_user6_user7_direct_head1_smoke_20260816 `
  --seed 20260715 --smoke-test
```

- [x] **Step 2: Audit smoke artifacts**

Require Direct-Head, embedding 2048, head parameters 81960, trainable backbone 2315984, block4/block5/classifier finite gradients, clip keep 1.0, `[N,2048]` archive, positive resource bytes, route `<95000000`, and unchanged canonical hashes.

- [x] **Step 3: Verify smoke and handle the gate**

Run focused tests, JSON parse, strict checkpoint reload through `_build_model(resolved_config)`, finite archive checks, compileall, and `git diff --check`. On any failure preserve smoke and stop; on pass record exact hashes/resources.

- [x] **Step 4: Commit and push smoke evidence**

```powershell
git add reports/x3d_s_train12_val2_user6_user7_direct_head1_smoke_audit.json reports/x3d_s_phase_status.md
git commit -m "Verify Direct-Head CUDA smoke"
git push origin test/x3d-fold0-generalization
```

Do not track smoke outputs.

---

### Task 5: Formal Training, Matched Decision, And Freeze

**Files:**
- Create: `reports/x3d_s_train12_val2_user6_user7_direct_head1_report.json`
- Create: `reports/x3d_s_train12_val2_user6_user7_direct_head1_report.md`
- Modify: `reports/x3d_s_phase_status.md`
- Modify: `docs/superpowers/plans/2026-08-16-x3d-train12-val2-direct-head.md`
- Update external: `C:\Users\LaiWanzhou\AppData\Local\Temp\x3d_adaptive_multiclip_handoff.md`

- [ ] **Step 1: Launch formal candidate exactly once**

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.run_x3d_s_train12_val2_dev `
  --config configs/experiments/x3d_s_ir_context_train12_val2_user6_user7_direct_head1.yaml `
  --development-split metadata/splits/train12_val2_user6_user7_development.json `
  --run-id x3d_s_ir_context_train12_val2_user6_user7_direct_head1_seed20260715 `
  --seed 20260715
```

Do not change seed, LR, epochs, patience, augmentation, or run ID after launch.

- [ ] **Step 2: Preserve artifacts under the frozen stop rule**

Allow the existing checkpoint/early-stop contract to finish. If completed Accuracy is below the numeric `new_partial2_reference_accuracy - 0.02` threshold frozen in Task 3 preregistration, classify `human_review_regression`, preserve everything, and stop after reporting. Never delete or launch follow-up work.

- [ ] **Step 3: Generate independent matched report**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m scripts.report_x3d_s_train12_val2_user6_user7_direct_head1 `
  --run-directory outputs/x3d_s_ir_context_train12_val2_dev/x3d_s_ir_context_train12_val2_user6_user7_direct_head1_seed20260715 `
  --reference-report reports/x3d_s_train12_val2_user6_user7_partial2_report.json `
  --reference-run-directory outputs/x3d_s_ir_context_train12_val2_dev/x3d_s_ir_context_train12_val2_user6_user7_partial2_seed20260715
```

Require exact sample/label/user/duration alignment before deltas.

- [ ] **Step 4: Apply exactly one decision**

Record only `preferred`, `human_review_regression`, or `non_winning_ablation`. Diagnostics and the aspirational `0.58-0.60` range cannot change it.

- [ ] **Step 5: Run fresh final verification**

```powershell
D:\Anaconda\envs\pyTorch2.7\python.exe -m pytest -q
D:\Anaconda\envs\pyTorch2.7\python.exe -m compileall -q src scripts tests
git diff --check
```

Regenerate report twice with identical SHA; parse new JSON; verify no training process; verify no heldout/test access; verify canonical and partial2 hashes unchanged.

- [ ] **Step 6: Update status, plan, and handoff**

Record selected epoch, primary metrics, train gaps, user/duration deltas, disagreement/NLL, classifier drift, parameters, resource bytes, hashes, decision, and exact next action.

- [ ] **Step 7: Commit, push, and verify remote**

```powershell
git add reports/x3d_s_train12_val2_user6_user7_direct_head1_report.json reports/x3d_s_train12_val2_user6_user7_direct_head1_report.md reports/x3d_s_phase_status.md docs/superpowers/plans/2026-08-16-x3d-train12-val2-direct-head.md
git commit -m "Record Direct-Head X3D result"
git push origin test/x3d-fold0-generalization
```

Verify local/origin/remote SHA equality with `git rev-parse` and `git ls-remote`, plus a clean worktree. Never track checkpoint, NPZ, history, outputs, or cache.
