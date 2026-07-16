# Task 03 formal fold_0 unimodal baseline report

This report covers the formal fold_0 unimodal baselines. It is not complete three-fold OOF, not a final model trained on all 18 users, and not a Kaggle submission model.
No test data, fold_1/fold_2, pretrained weights, preprocessing cache, architecture change, augmentation change, or parameter ablation was used.
The best checkpoint remains selected by validation Accuracy.

## Frozen training configuration

All runs use CrossEntropyLoss, AdamW (learning rate 3e-4, weight decay 1e-4), CosineAnnealingLR, dropout 0.2, gradient clipping at 1.0, CUDA AMP, seed 20260715, num_workers=4, embedding_dim=128, and 40 output classes. No pretrained weights are used.

| Modalities | Model and aggregation | Input | Batch | Epochs/patience |
| --- | --- | --- | ---: | ---: |
| Depth_Color / IR / Thermal | MobileNetV3-Small (`weights=None`) with temporal mean pooling | 12 frames, 3x192x192 | 4 | 30/6 |
| IMU | residual TCN, channels [64,128] | 256x80 sequence | 16 | 40/8 |
| Skeleton | root-centered + velocity residual TCN, channels [64,128] | 64x102 sequence | 16 | 40/8 |
| Radar | 21 per-frame statistics + residual TCN, channels [64,128] | 64x21 sequence | 16 | 40/8 |

## Formal configurations and results

| Modality | Train/val | Requested/completed | Early stop | Best epoch | Best Acc | Best Macro-F1 | Best val loss | Final Acc | Mean epoch | Total | GPU alloc/reserved MB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMU | 1916/987 | 40/31 | true | 23 | 0.255319 | 0.185963 | 3.703923 | 0.238095 | 12.10s | 0h 06m 15.0s | 37.40/58.00 |
| Skeleton | 1931/1000 | 40/32 | true | 24 | 0.475000 | 0.381188 | 2.113968 | 0.467000 | 10.60s | 0h 05m 39.3s | 23.77/30.00 |
| Radar | 1918/996 | 40/10 | true | 2 | 0.134538 | 0.025662 | 3.437751 | 0.121486 | 13.35s | 0h 02m 13.5s | 22.41/28.00 |
| IR | 1933/1000 | 30/23 | true | 17 | 0.201000 | 0.114874 | 4.155961 | 0.182000 | 33.62s | 0h 12m 53.2s | 357.34/504.00 |
| Thermal | 1845/1046 | 30/12 | true | 6 | 0.210325 | 0.083627 | 3.444660 | 0.195029 | 22.28s | 0h 04m 27.4s | 357.32/504.00 |
| Depth_Color | 1931/1000 | 30/24 | true | 18 | 0.222000 | 0.146910 | 4.722853 | 0.212000 | 50.54s | 0h 20m 12.9s | 357.34/484.00 |

Aggregate epoch time across the six serial runs: 0h 51m 41.2s. This excludes startup, data-loader spawn, artifact validation, and report-generation overhead.

## Accuracy and Macro-F1 checkpoint analysis

- **IMU**: Accuracy checkpoint epoch 23 has Accuracy 0.255319 and Macro-F1 0.185963; highest historical Macro-F1 0.191610 occurs at epoch 29; same epoch: false.
- **Skeleton**: Accuracy checkpoint epoch 24 has Accuracy 0.475000 and Macro-F1 0.381188; highest historical Macro-F1 0.385004 occurs at epoch 19; same epoch: false.
- **Radar**: Accuracy checkpoint epoch 2 has Accuracy 0.134538 and Macro-F1 0.025662; highest historical Macro-F1 0.034496 occurs at epoch 7; same epoch: false.
- **IR**: Accuracy checkpoint epoch 17 has Accuracy 0.201000 and Macro-F1 0.114874; highest historical Macro-F1 0.122429 occurs at epoch 20; same epoch: false.
- **Thermal**: Accuracy checkpoint epoch 6 has Accuracy 0.210325 and Macro-F1 0.083627; highest historical Macro-F1 0.127960 occurs at epoch 10; same epoch: false.
- **Depth_Color**: Accuracy checkpoint epoch 18 has Accuracy 0.222000 and Macro-F1 0.146910; highest historical Macro-F1 0.146910 occurs at epoch 18; same epoch: true.

## Training curves and fit assessment

- **IMU**: train_loss 3.4133->0.7651; train_acc 0.1357->0.7620; val_acc 0.1722->0.2381; min_val_loss_epoch=4; final_gap=0.5239; overfitting signal
- **Skeleton**: train_loss 3.2403->0.6207; train_acc 0.1916->0.7985; val_acc 0.2260->0.4670; min_val_loss_epoch=10; final_gap=0.3315; overfitting signal
- **Radar**: train_loss 3.5712->3.0019; train_acc 0.0699->0.1752; val_acc 0.0713->0.1215; min_val_loss_epoch=6; final_gap=0.0537; underfitting signal
- **IR**: train_loss 3.3860->1.1808; train_acc 0.1345->0.5892; val_acc 0.0520->0.1820; min_val_loss_epoch=9; final_gap=0.4072; overfitting signal
- **Thermal**: train_loss 3.4356->1.6027; train_acc 0.1301->0.4824; val_acc 0.1176->0.1950; min_val_loss_epoch=2; final_gap=0.2874; overfitting signal
- **Depth_Color**: train_loss 3.3966->0.8236; train_acc 0.1238->0.7069; val_acc 0.0520->0.2120; min_val_loss_epoch=9; final_gap=0.4949; overfitting signal

## Per-class recall and major confusions

- **IMU**: minimum-recall classes [1,2,13,14,16,18,21,24,25,26,27,35]; largest off-diagonal confusions: 34->36 (17); 11->7 (15); 6->7 (13); 10->9 (11); 32->36 (11).
- **Skeleton**: minimum-recall classes [16,24,26,38]; largest off-diagonal confusions: 13->12 (17); 10->9 (14); 9->8 (13); 21->22 (9); 11->9 (8).
- **Radar**: minimum-recall classes [0,1,2,3,5,6,8,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,37,38,39]; largest off-diagonal confusions: 9->36 (33); 34->36 (31); 10->36 (28); 20->36 (27); 29->36 (25).
- **IR**: minimum-recall classes [3,4,6,12,16,18,19,23,24,25,26,27,28,35,37,38,39]; largest off-diagonal confusions: 9->10 (21); 7->10 (14); 11->10 (13); 6->10 (12); 8->10 (12).
- **Thermal**: minimum-recall classes [1,3,8,11,12,13,14,15,16,18,19,22,24,25,26,27,28,30,32,33,35,37,38,39]; largest off-diagonal confusions: 9->10 (24); 33->36 (19); 29->36 (16); 8->10 (15); 7->6 (15).
- **Depth_Color**: minimum-recall classes [3,14,16,17,18,19,21,22,25,26,27,28,35,37,39]; largest off-diagonal confusions: 20->34 (16); 7->10 (14); 8->10 (13); 11->10 (13); 11->7 (11).

## Comparison with optimized two-epoch verification

| Modality | 2-epoch Acc | Formal Acc | Delta | 2-epoch Macro-F1 | Formal Macro-F1 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IMU | 0.193516 | 0.255319 | +0.061803 | 0.051064 | 0.185963 | +0.134899 |
| Skeleton | 0.301000 | 0.475000 | +0.174000 | 0.133185 | 0.381188 | +0.248003 |
| Radar | 0.087349 | 0.134538 | +0.047189 | 0.024499 | 0.025662 | +0.001163 |
| IR | 0.153000 | 0.201000 | +0.048000 | 0.026103 | 0.114874 | +0.088771 |
| Thermal | 0.186424 | 0.210325 | +0.023901 | 0.047492 | 0.083627 | +0.036135 |
| Depth_Color | 0.142000 | 0.222000 | +0.080000 | 0.033530 | 0.146910 | +0.113380 |

## Conclusions and next controlled ablations

- Radar should first test richer per-frame statistics and then a PointNet-style frame encoder; more epochs alone cannot recover point-level structure discarded by the current 21 statistics.
- Visual modalities should next compare 192x192 with aspect-ratio-preserving 224x224, then 12 versus 16 frames and weak spatial/temporal augmentation, one variable at a time.
- IMU should next compare sequence length 256/384 and controlled sensor noise or amplitude scaling; Skeleton should compare 64/96 steps and stronger joint-time structure.
- Learning rate, weight decay, dropout, alternative user ratios, fold_1/fold_2, candidate-B models, and multimodal fusion remain outside this run.

## Input-pipeline and hardware observations

The earlier workers=0 pipeline saturated CPU-side loading and left substantial GPU idle time. The selected workers=4 pipeline materially improved throughput, but image decoding remains online, so the visual modalities are still partly CPU/input limited and GPU utilization remains bursty rather than continuously saturated.

During this formal run, manual nvidia-smi snapshots for the visual modalities were approximately 1.8-2.0 GiB total device memory with utilization fluctuating around 5%-47%. These snapshots include CUDA context and non-tensor overhead. The reproducible PyTorch peaks were IR 357.34/504.00 MB, Thermal 357.32/504.00 MB, and Depth_Color 357.34/484.00 MB allocated/reserved. Thus training definitely ran on CUDA; low instantaneous utilization reflects input/validation waits, not CPU-only training.

## Integrity and runtime checks

All six runs used CUDA AMP, num_workers=4, the frozen fold_0 split, unchanged model parameter counts, and the expected modality sample counts. Required checkpoints, histories, metrics, confusion matrices, and validation prediction archives were reopened and validated. No worker deadlock, CUDA OOM, NaN/Inf, test-set access, preprocessing cache, or output-format change occurred.
