# Thermal T1-B.2 layerwise activation trace

- Status: **first_abnormal_amplification_localized_training_still_stopped**
- Checkpoint SHA256: `ca9c11c0f4d50f67c89da52284f726085dfeb3d1578621438872b9255a05d827`
- Execution: eval/inference only; zero training, backward, parameter update, or BN-buffer update.
- Input: full-frame Thermal, 16 uniform normalized-time frames, 224x224.

## Matched trials

| Spike | Control | Frames | FP32 embedding L2 spike/control |
|---|---|---:|---:|
| train__c36__user7__1-2-2 | train__c36__user7__6-1-1 | 17/17 | 32057.371/64.286 |
| train__c36__user7__1-2-1 | train__c36__user7__4-1-1 | 30/32 | 8981.925/64.944 |
| train__c36__user7__1-2-3 | train__c36__user7__5-2-2 | 27/26 | 2244.906/64.857 |

## First observed amplification

- `fp32`: `spatial.backbone.stages.2.5` at RMS ratio `17.043` (previous `spatial.backbone.stages.2.4`: `6.428`).
- `bfloat16`: `spatial.backbone.stages.2.5` at RMS ratio `12.310` (previous `spatial.backbone.stages.2.4`: `4.753`).
- Agreed module: `BasicBlock` wrapping `ConvBlock` with `Residual` token/channel mixer.

The location is the first observed 10x spike/control RMS crossing, not proof that the module itself is defective. Full per-layer statistics and per-pair crossings are in the JSON report.

## Integrity gates

- Hooks preserve logits exactly: `True`
- Hooks preserve embeddings exactly: `True`
- State dict digest unchanged: `True`
- Every state tensor unchanged: `True`
- BN-free run authorized: `False`
- Epoch 18 resume authorized: `False`

## Decision

T1-B.2 localizes the group-median first 10x crossing to iFormer stage 2 block 5 in both precisions. Individual crossings vary with spike severity, so this is an observed amplification boundary rather than proof of a defective block. No training or architecture change is authorized; human review is required.

## Limitations

- The first 10x spike/control crossing localizes where abnormal magnitude first becomes visible; it does not alone prove the module is defective.
- Controls match user, class, preprocessing route, and nearest available frame count, but source scene and performer pose can still differ.
- Only the three strongest T1-B.1 validation spikes are traced; all canonical trials remain retained and unchanged.
- Individual first crossings range from stage 2 block 4 to later blocks; the group-median stage 2 block 5 location is representative, not universal.
- No heldout, competition-test, quarantined evidence, or frozen IR/X3D asset was accessed.
