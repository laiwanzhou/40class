# X3D Train12/Val2 Direct-Head Design

**Status:** User-approved direction on 2026-08-16; implementation and training remain gated on review of this written contract.

## Question

Does the randomly initialized nonlinear `2048 -> 256 -> 40` custom head contribute materially to cross-user overfitting, while the K400 X3D block4/block5 adaptation remains useful?

## Evidence

The matched partial1 experiment froze block4 and reduced Accuracy by `0.049383` relative to partial2. The matched layerwise-LR1 experiment retained block4/block5 but reduced their K400-relative parameter drift by about 83% and 52%; Accuracy still fell by `0.030864`, while embedding-head and classifier drift increased by about 11% and 13%. Layerwise-LR1 reached train Accuracy `0.937876` with validation Accuracy `0.521605`.

These results reject further backbone-only restriction as the first response. The next isolated mechanism is custom-head capacity.

## Single Model Intervention

Replace the current trainable head:

```text
2048D pooled X3D feature
-> Linear(2048, 256)
-> LayerNorm(256)
-> GELU
-> Dropout(0.25)
-> Linear(256, 40)
```

with:

```text
2048D pooled X3D feature
-> Dropout(0.25)
-> Linear(2048, 40)
```

The direct classifier has about 81,960 trainable parameters instead of about 535,336 custom-head parameters. The classifier remains randomly initialized under seed `20260715`.

For `ExpertOutput`, the candidate emits the L2-normalized 2048D pooled trial feature as its optional embedding. This does not enter the first-generation six-modal anchor or residual mixer, whose frozen boundary remains logits/probabilities plus availability and quality. The embedding dimension is candidate metadata, not a requirement that heterogeneous experts share a coordinate system.

## Fixed Matched Recipe

Relative to `x3d_s_ir_context_train12_val2_partial2_seed20260715`, freeze all non-head behavior:

- development split: `metadata/splits/train12_val2_development.json`;
- train users: the frozen 12-user partition; validation users: `user21,user22`;
- seed: `20260715`;
- K400 pretrained X3D-S and blocks0-3 frozen;
- block4 and block5 trainable after the same two head-only warmup epochs;
- restore partial2 shared backbone LR `3e-5`; do not use layerwise block LRs;
- head LR `3e-4`, weight decay `0.05`, gradient accumulation and clipping unchanged;
- BN running statistics frozen and trainable BN affine behavior unchanged;
- all adaptive temporal windows retained; no clip dropout or motion-peak sampling;
- 13 frames per local window, target window 32, maximum eight clips;
- mean-probability trial aggregation and equal-trial NLL unchanged;
- brightness/contrast/gamma `[0.9,1.1]`, no noise, no blur, no label smoothing;
- 20-epoch horizon, scheduler, patience, checkpoint selection, and deterministic validation unchanged.

No L2-SP, EMA, user-adversarial loss, stronger augmentation, temporal change, additional seed, or other regularizer belongs to this candidate.

## Implementation Boundary

Add an explicit configurable head type while preserving the existing projected head as the default. Existing configs, canonical Phase 4/5 checkpoints, loaders, archives, and inference behavior must remain backward compatible. For `head_type: direct`, `embedding_dim` is fixed to the backbone output dimension `2048`; the candidate config must state `embedding_dim: 2048`, and construction must fail closed for any other value or unsupported head type.

Tests must prove:

- default projected-head construction and output shapes remain unchanged;
- direct-head logits are `[N,40]` and optional embeddings are `[N,2048]`;
- padded clips still cannot affect outputs;
- optimizer grouping assigns the direct classifier to the custom-head LR/decay policy;
- resolved config, run summary, checkpoint provenance, and report identify the head type;
- a CUDA smoke produces finite block4/block5/direct-classifier gradients and preserves full temporal coverage;
- canonical Phase 4/5 hashes remain unchanged.

## Execution And Isolation

Use a new run ID and output directory, provisionally:

```text
x3d_s_ir_context_train12_val2_direct_head1_seed20260715
```

Pre-register and push the config, implementation, tests, and experiment manifest before the formal result exists. Run one CUDA smoke, then the frozen 20-epoch train12/val2 experiment. Do not access heldout4 or competition test and do not mutate canonical OOF/Phase 5 evidence.

## Decision Rule

Matched reference is partial2:

- Accuracy `0.5524691358`;
- Macro-F1 `0.4186514986`;
- worst-user Accuracy `0.4586466165`;
- mandatory human-review Accuracy floor `0.5324691358`.

Classify the result as:

- **preferred** only when Accuracy, Macro-F1, and worst-user Accuracy are all no worse than partial2 and at least one strictly improves;
- **human_review_regression** when Accuracy is below `0.5324691358`; preserve every checkpoint, prediction, log, history, manifest, and report, then stop;
- **non_winning_ablation** otherwise; preserve the result and do not automatically start another IR experiment.

The aspirational range `0.58-0.60` is descriptive, not a pass threshold. This development result cannot replace strict train-14 OOF evidence regardless of score.

## Non-Goals

- No claim that direct-head embeddings are aligned with other modalities.
- No automatic train-14 OOF, heldout evaluation, ensemble, L2-SP, or follow-up bottleneck experiment.
- No deletion of partial2, layerwise-LR1, canonical Phase 4/5, or Direct-Head artifacts.
