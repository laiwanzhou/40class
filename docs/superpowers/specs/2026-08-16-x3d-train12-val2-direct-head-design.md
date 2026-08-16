# X3D Train12/Val2 Direct-Head Design

**Status:** Approved for implementation planning on 2026-08-16 after three independent reviews; the final review decision is `APPROVED`.

## Question

Does replacing the complete randomly initialized nonlinear `2048 -> 256 -> 40` projected head with a direct K400-feature classifier improve cross-user generalization while retaining useful block4/block5 adaptation?

## Evidence

The matched partial1 experiment froze block4 and reduced Accuracy by `0.049383` relative to partial2. The matched layerwise-LR1 experiment retained block4/block5 but reduced their K400-relative parameter drift by about 83% and 52%; Accuracy still fell by `0.030864`, while embedding-head and classifier drift increased by about 11% and 13%. Layerwise-LR1 reached train Accuracy `0.937876` with validation Accuracy `0.521605`.

These results reject further backbone-only restriction as the first response. The next isolated component is the complete custom-head architecture.

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

This is one clean composite head replacement, not a parameter-count-only ablation: it simultaneously removes the learned projection, LayerNorm, GELU, and 256D classifier input. Any result may be attributed only to the projected nonlinear head as a whole.

For `ExpertOutput`, define the direct embedding exactly as:

```text
L2Normalize(
    mean(
        valid per-clip outputs of the unchanged X3D feature backbone,
        before the new custom-head Dropout(0.25)
    )
)
```

The official X3D feature backbone's internal `Dropout(p=0.5)` remains unchanged from partial2; the implementation must not remove, bypass, or relocate it. Only the new custom-head `Dropout(0.25)` is excluded from the embedding path. Prediction/archive inference always uses `model.eval()`, so saved embeddings are deterministic. `ExpertOutput.embedding` remains a required runtime tensor; it is optional only in the sense that first-generation fusion/evidence consumers may ignore it. The 2048D embedding does not enter the six-modal anchor or residual mixer, whose frozen boundary remains logits/probabilities plus availability and quality. Its dimension is candidate metadata, not a claim that heterogeneous experts share a coordinate system.

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
- use the identical partial2 two-epoch warmup plus cosine schedule over `scheduler_horizon_epochs: 20` for every optimizer group;
- 20-epoch maximum and deterministic validation unchanged;
- early stopping reuses the existing `best_macro_f1` comparator and patience 8: higher fixed-40 Macro-F1 wins, with Accuracy breaking a Macro-F1 tie and resetting patience;
- formal comparison uses `best_accuracy.pt`, selected by Accuracy, then fixed-40 Macro-F1, then earlier epoch;
- worst-user Accuracy is `min(user21 Accuracy, user22 Accuracy)`.

No L2-SP, EMA, user-adversarial loss, stronger augmentation, temporal change, additional seed, or other regularizer belongs to this candidate.

## Implementation Boundary

Add an explicit configurable head type while preserving the existing projected head as the default. Existing configs, canonical Phase 4/5 checkpoints, loaders, archives, and inference behavior must remain backward compatible. For `head_type: direct`, `embedding_dim` is fixed to the backbone output dimension `2048`; the candidate config must state `embedding_dim: 2048`, and construction must fail closed for any other value or unsupported head type.

Tests must prove:

- default projected-head construction and output shapes remain unchanged;
- direct-head logits are `[N,40]` and required runtime embeddings are `[N,2048]`;
- direct embedding equals the normalized mean of valid outputs from the unchanged X3D feature backbone before the new custom-head dropout; the official backbone-internal dropout remains present;
- with fixed backbone outputs, changing only the custom-head dropout state cannot alter the direct embedding;
- padded clips still cannot affect outputs;
- optimizer grouping assigns the direct classifier to the custom-head LR/decay policy;
- a legacy config with no `head_type` constructs the projected head, loads a projected checkpoint with `strict=True`, and reproduces its output exactly under `model.eval()` with identical inputs and masks;
- resolved config, run summary, checkpoint provenance, prediction archive, and report identify the head type and embedding dimension;
- a CUDA smoke produces finite block4/block5/direct-classifier gradients and preserves full temporal coverage;
- smoke records the `[N,2048]` archive shape, peak CUDA allocation, checkpoint bytes, archive bytes, and provisional route bytes;
- canonical Phase 4/5 hashes remain unchanged.

## Execution And Isolation

Use a new run ID and output directory, provisionally:

```text
x3d_s_ir_context_train12_val2_direct_head1_seed20260715
```

Pre-register and push the config, implementation, tests, and experiment manifest before the formal result exists. Run one CUDA smoke, then the frozen 20-epoch train12/val2 experiment. Do not access heldout4 or competition test and do not mutate canonical OOF/Phase 5 evidence.

The preregistration and result report must record the actual direct-head parameter count, checkpoint bytes, prediction/archive bytes, peak CUDA allocation, and provisional route size. The larger 2048D evidence archive is an expected resource change; it is not a model-size regression and does not alter the `<95 MB` deployment gate calculation.

## Diagnostic Reporting

In addition to the frozen validation decision metrics, report these mechanism diagnostics without using them for promotion or checkpoint selection:

- training-log Accuracy and Macro-F1 at the selected epoch;
- train-to-validation Accuracy and Macro-F1 gaps;
- actual custom-head and total trainable parameter counts;
- per-user Accuracy for user21 and user22 and matched deltas versus partial2;
- duration-bucket Accuracy/Macro-F1 and matched deltas for `<=13`, `14-32`, `33-64`, and `>64`;
- validation NLL, wrong-prediction confidence, prediction disagreement, candidate-only correct, and partial2-only correct counts;
- direct classifier parameter drift from its seeded initialization, reported descriptively because it has no K400 semantic anchor.

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

The aspirational range `0.58-0.60` is descriptive, not a pass threshold. Diagnostic train-gap, resource, disagreement, duration, and drift measurements cannot promote or reject the candidate. This development result cannot replace strict train-14 OOF evidence regardless of score.

## Non-Goals

- No claim that direct-head embeddings are aligned with other modalities.
- No automatic train-14 OOF, heldout evaluation, ensemble, L2-SP, or follow-up bottleneck experiment.
- No deletion of partial2, layerwise-LR1, canonical Phase 4/5, or Direct-Head artifacts.
