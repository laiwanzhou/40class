# X3D Train12/Val2 Partial1 Design

**Status:** Approved by the user on 2026-08-14.

## Question

Does freezing X3D block4 while adapting only block5 reduce cross-user overfitting without losing the domain adaptation observed in the train12/val2 partial2 experiment?

## Intervention

Use the exact frozen train12/val2 development population and partial2 recipe. Change only:

```yaml
training:
  unfrozen_backbone_blocks: 1
```

Epochs 1-2 remain head-only. From epoch 3 onward, block5 and the custom embedding/classification head train while blocks0-4 and all backbone BN running statistics remain frozen. Full adaptive temporal coverage is retained; clip dropout and motion-peak sampling remain forbidden.

## Evidence Boundary

This is matched development evidence against partial2 because split, seed, data, augmentation, optimizer, loss, scheduler, and checkpoint rule are identical. It is not unbiased OOF and cannot replace canonical Phase 4/5 evidence. Heldout4 and competition test remain inaccessible.

## Decision Contract

- `partial2` reference: Accuracy `0.5524691358`, Macro-F1 `0.4186514986`, worst-user Accuracy `0.4586466165`.
- A greater-than-two-point Accuracy regression (`<0.5324691358`) requires human review with all artifacts preserved.
- Partial1 is preferred only if Accuracy, Macro-F1, and worst-user Accuracy are all no worse than partial2 and at least one is strictly better.
- Otherwise retain the result as a non-winning capacity ablation. Do not compensate post hoc with additional regularizers or changed epoch selection.

## Verification

The config-diff test must prove the only behavioral change from partial2 is `training.unfrozen_backbone_blocks: 2 -> 1`. CUDA smoke must verify exactly `968,544` trainable backbone parameters after warmup, finite block5/head gradients, full temporal coverage, and unchanged canonical artifacts.
