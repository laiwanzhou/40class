# ResNet18 object interaction expert experiment

## Integrity

- ImageNet weights: ResNet18_Weights.IMAGENET1K_V1; loaded: True.
- Depth stem: native RGB 3-channel conv1/bn1; IR stem: native 1-channel conv1 initialized by RGB-channel mean with independent bn1.
- Shared ResNet body count: 1; layer4 BN running statistics: frozen.
- Batch/accumulation/chunk: 1/4/32.
- Epoch 8 strict reproduction: True; test read: no.

- ROI train/validation audit tables match MobileNet exactly; Target16 class weights match exactly.

## B0 / M1 / R1

| model | checkpoint_epoch | accuracy | macro_f1 | weighted_f1 | target16_macro_f1 | hand_head7_macro_f1 | table7_macro_f1 | screen2_macro_f1 | zero_f1_count | target16_zero_f1_count | rescued | harmed | net_rescue | val_loss | ece | validation_seconds | peak_allocated_mb | model_parameter_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_epoch8 | 8 | 0.444068 | 0.339342 | 0.427365 | 0.188270 | 0.209546 | 0.183518 | 0.130435 | 5 | 4 | 0 | 0 | 0 | 2.205282 | 0.167296 | 61.507342 | 0.000000 | 5125116 |
| M1_mobilenet | 11 | 0.462712 | 0.349791 | 0.442325 | 0.225633 | 0.214335 | 0.265683 | 0.125000 | 6 | 5 | 22 | 11 | 11 | 2.519730 | 0.187504 | 253.187919 | 3408.244629 | 18606788 |
| R1_resnet18 | 8 | 0.461017 | 0.356157 | 0.443848 | 0.230479 | 0.214611 | 0.240771 | 0.250000 | 4 | 3 | 18 | 8 | 10 | 2.190053 | 0.159974 | 209.280484 | 3475.345215 | 59557956 |

## Target16

| action_name | base_f1 | mobilenet_f1 | resnet18_f1 | delta_resnet_vs_mobilenet |
| --- | --- | --- | --- | --- |
| Drink_water | 0.411765 | 0.444444 | 0.358974 | -0.085470 |
| Eat_food | 0.415094 | 0.387097 | 0.409091 | 0.021994 |
| Take_and_use_tableware | 0.000000 | 0.342857 | 0.272727 | -0.070130 |
| Stir_drinks | 0.242424 | 0.413793 | 0.173913 | -0.239880 |
| Peel_fruits | 0.226415 | 0.318182 | 0.259259 | -0.058923 |
| Wipe_bowls | 0.166667 | 0.000000 | 0.285714 | 0.285714 |
| Write | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Make_a_phone_call | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Read_documents | 0.333333 | 0.451613 | 0.421053 | -0.030560 |
| Turn_pages | 0.315789 | 0.333333 | 0.272727 | -0.060606 |
| Use_a_mobile_phone | 0.083333 | 0.250000 | 0.181818 | -0.068182 |
| Watch_TV | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Play_games | 0.260870 | 0.250000 | 0.500000 | 0.250000 |
| Take_a_selfie | 0.294118 | 0.307692 | 0.322581 | 0.014888 |
| Take_medicine | 0.068966 | 0.000000 | 0.142857 | 0.142857 |
| Take_body_temperature | 0.193548 | 0.111111 | 0.086957 | -0.024155 |

## Sample comparison

- {"resnet_only_correct": 18, "mobilenet_only_correct": 19, "both_correct": 254, "both_wrong": 299, "different_but_both_wrong": 63}
- R1 rescued/harmed/net: 18/8/10.

- Retention signals passed: 7/8; {"target16_macro_f1_higher": true, "overall_macro_f1_not_lower": true, "net_rescue_higher": false, "harmed_not_higher": true, "target16_zero_f1_not_higher": true, "damaged_targets_recovered": true, "under_100_mib": true, "stable_easy_not_materially_lower": true}.

## Resource budget

- Peak allocated/reserved VRAM: 3475.35/3682.00 MB.
- Training seconds: 13110.17.
- Actual inference bundle: 68994661 bytes (65.80 MiB); under 100 MiB: True.

## Decision

- Retain ResNet18 expert: yes.
- Capacity is not the sole bottleneck: the stronger backbone improves aggregate metrics and recovers Wipe_bowls/Take_medicine, but Write, Make_a_phone_call, and Watch_TV remain zero-F1 and several interaction classes trade places.
- Evidence points more strongly to small-object visibility/ROI evidence, limited examples, and class-boundary ambiguity than to backbone capacity alone. Do not replace the frozen 40-class base model in this experiment.
