# X3D Train12/Val2 Layer-wise LR1 Design

**Status:** Approved direction from the user on 2026-08-14.

## Question

Can X3D preserve the transferable block4 adaptation lost by partial1 while reducing the user-specific drift observed in partial2?

## Intervention

Return to the partial2 architecture: train block4, block5, and the custom head after two head-only warmup epochs. Replace the shared backbone LR with auditable block-specific rates:

```yaml
optimizer:
  backbone_lr: 3.0e-5
  backbone_block_lrs:
    4: 3.0e-6
    5: 1.0e-5
  head_lr: 3.0e-4
training:
  unfrozen_backbone_blocks: 2
```

Relative to partial2, block4 updates at one tenth and block5 at one third of their previous rates. Blocks0-3 remain frozen; BN running statistics remain frozen. Split, seed, full temporal coverage, augmentation, loss, weight decay, scheduler, and checkpoint rule are unchanged.

## Why Layer-wise LR Before L2-SP

Layer-wise LR directly tests whether update magnitude, rather than trainable capacity itself, caused partial2 overfitting. It introduces no new loss coefficient and is easier to audit. K400-anchored L2-SP remains the next independent strategy only if this lower-LR experiment fails to balance adaptation and generalization.

## Decision Boundary

Use partial2 as the matched reference: `0.552469` Accuracy, `0.418651` Macro-F1, and `0.458647` worst-user Accuracy. A greater-than-two-point Accuracy regression triggers human review with all artifacts preserved. Prefer layerwise_lr1 only when all three metrics are no worse and at least one is strictly better.

## Evidence Isolation

This is development-only matched evidence on the frozen 12/2 split. It does not replace Phase 4/5 OOF evidence. Heldout4 and competition test remain inaccessible. Partial1 versioned records remain; its local smoke and formal output directories were explicitly deleted at the user's request after their hashes and metrics were frozen.
