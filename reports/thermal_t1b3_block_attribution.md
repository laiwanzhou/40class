# Thermal T1-B.3 block-level attribution

- Status: **block_level_attribution_complete_training_still_stopped**
- Epoch-16 checkpoint: `ca9c11c0f4d50f67c89da52284f726085dfeb3d1578621438872b9255a05d827`
- Reconstructed spike cohort: `23` trials; class 36 `12`, other classes `11`.
- Execution: official ImageNet pretrained versus epoch16, FP32 and bfloat16, identical full-frame 16-segment inputs.

## Attribution conclusion

- Primary mechanism: **finetuning_induced_convolution_branch_amplification_with_extreme_tail**
- Pretrained comparison: **absent_in_pretrained_emerges_after_finetuning**
- Segment source: **mostly_distributed_with_single_or_few_segment_tail**
- Class-36 specificity: **class36_enriched_and_stronger_but_not_exclusive**
- Parameter/BN drift: **small_conv_weight_drift_and_modest_bn_gain_shift_can_compound_upstream**

The official pretrained model shows no spike/control separation at blocks 4 or 5. After finetuning, project/residual branches separate more strongly than skip inputs, and every 10x branch event occurs before residual add; residual alignment is not observed. Most cohort members remain distributed over 16 segments, with a single/few-segment tail. No intervention is authorized by this diagnostic.

Epoch16 FP32 block5 all-spike branch/project ratio is `2.518x` versus pretrained `1.005x`; skip is `1.289x` and residual-add output `1.790x`. Epoch16 block5 mechanism counts: `{'below_10x_at_this_block': 21, 'convolution_branch_explosion_before_add': 1, 'inherited_from_skip_input': 1}`.

Epoch16 block5 segment sources: `{'distributed_sequence': 17, 'single_segment': 2, 'few_segments': 4}`. The three largest spikes are `{'train__c36__user7__1-2-2': {'fp32': 'distributed_sequence', 'bfloat16': 'distributed_sequence'}, 'train__c36__user7__1-2-1': {'fp32': 'few_segments', 'bfloat16': 'few_segments'}, 'train__c36__user7__1-2-3': {'fp32': 'few_segments', 'bfloat16': 'single_segment'}}`.

## Cohort summary

| Model | Precision | Block | Group | Project RMS ratio | Branch RMS ratio | Skip RMS ratio | Add RMS ratio |
|---|---|---:|---|---:|---:|---:|---:|
| official_imagenet_pretrained | fp32 | 4 | class36 | 0.989 | 0.989 | 0.978 | 0.992 |
| official_imagenet_pretrained | fp32 | 4 | other_classes | 0.996 | 0.996 | 1.001 | 0.998 |
| official_imagenet_pretrained | fp32 | 4 | all_spikes | 0.995 | 0.995 | 0.994 | 0.996 |
| official_imagenet_pretrained | fp32 | 5 | class36 | 1.009 | 1.009 | 0.992 | 1.000 |
| official_imagenet_pretrained | fp32 | 5 | other_classes | 1.003 | 1.003 | 0.998 | 1.003 |
| official_imagenet_pretrained | fp32 | 5 | all_spikes | 1.005 | 1.005 | 0.996 | 1.001 |
| official_imagenet_pretrained | bfloat16 | 4 | class36 | 0.989 | 0.989 | 0.978 | 0.992 |
| official_imagenet_pretrained | bfloat16 | 4 | other_classes | 0.997 | 0.997 | 1.001 | 0.998 |
| official_imagenet_pretrained | bfloat16 | 4 | all_spikes | 0.995 | 0.995 | 0.994 | 0.996 |
| official_imagenet_pretrained | bfloat16 | 5 | class36 | 1.009 | 1.009 | 0.992 | 0.999 |
| official_imagenet_pretrained | bfloat16 | 5 | other_classes | 1.002 | 1.002 | 0.998 | 1.003 |
| official_imagenet_pretrained | bfloat16 | 5 | all_spikes | 1.005 | 1.005 | 0.996 | 1.001 |
| epoch16_finetuned | fp32 | 4 | class36 | 1.925 | 1.925 | 1.349 | 1.625 |
| epoch16_finetuned | fp32 | 4 | other_classes | 1.559 | 1.559 | 1.077 | 1.268 |
| epoch16_finetuned | fp32 | 4 | all_spikes | 1.716 | 1.716 | 1.081 | 1.289 |
| epoch16_finetuned | fp32 | 5 | class36 | 2.723 | 2.723 | 1.625 | 2.184 |
| epoch16_finetuned | fp32 | 5 | other_classes | 2.265 | 2.265 | 1.268 | 1.636 |
| epoch16_finetuned | fp32 | 5 | all_spikes | 2.518 | 2.518 | 1.289 | 1.790 |
| epoch16_finetuned | bfloat16 | 4 | class36 | 1.725 | 1.725 | 1.283 | 1.531 |
| epoch16_finetuned | bfloat16 | 4 | other_classes | 1.638 | 1.638 | 1.057 | 1.302 |
| epoch16_finetuned | bfloat16 | 4 | all_spikes | 1.638 | 1.638 | 1.091 | 1.313 |
| epoch16_finetuned | bfloat16 | 5 | class36 | 2.301 | 2.301 | 1.531 | 1.792 |
| epoch16_finetuned | bfloat16 | 5 | other_classes | 2.422 | 2.422 | 1.302 | 1.714 |
| epoch16_finetuned | bfloat16 | 5 | all_spikes | 2.414 | 2.413 | 1.313 | 1.733 |

## Integrity

- Hooked logits exact: `True`
- Hooked embeddings exact: `True`
- Pretrained state unchanged: `True`
- Epoch16 state unchanged: `True`
- Checkpoint file unchanged: `True`
- No training, backward, head/crop change, or epoch18 resume was performed or authorized.

## Limitations

- The spike cohort is reconstructed from the frozen T1-B.1 numeric threshold on user6/user7 validation; no canonical trial is removed.
- Matched controls minimize frame-count difference within user/class when possible, but scene and pose remain potential confounders.
- A 10x cohort threshold emphasizes severe magnitude pathology and does not imply smaller but systematic gains are harmless.
- Residual attribution identifies where magnitude arises, not why a learned convolution or BN responds to specific Thermal content.
