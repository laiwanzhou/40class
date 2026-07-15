# Task 03 optimized two-epoch report

This is a two-epoch pipeline retest after input-pipeline optimization, not the formal 30-40 epoch training.
No test data, pretrained weights, preprocessing cache, fold_1/fold_2, or multimodal fusion was used.

## Old and optimized runs

| Modality | Old workers | New workers | Old batch | New batch | Old mean epoch s | New mean epoch s | Speedup | Old accuracy | New accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMU | 0 | 4 | 16 | 16 | 43.76 | 19.95 | 2.19x | 0.182371 | 0.193516 |
| Skeleton | 0 | 4 | 16 | 16 | 43.98 | 20.85 | 2.11x | 0.277000 | 0.301000 |
| Radar | 0 | 4 | 16 | 16 | 42.87 | 22.69 | 1.89x | 0.089357 | 0.087349 |
| IR | 0 | 4 | 4 | 4 | 160.57 | 48.38 | 3.32x | 0.132000 | 0.153000 |
| Thermal | 0 | 4 | 4 | 4 | 124.74 | 34.43 | 3.62x | 0.165392 | 0.186424 |
| Depth_Color | 0 | 4 | 4 | 4 | 246.14 | 66.78 | 3.69x | 0.137000 | 0.142000 |

## Optimized results

| Modality | Train/val | Accuracy | Macro-F1 | Train samples/s | GPU allocated/reserved MB | Checkpoint MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IMU | 1916/987 | 0.193516 | 0.051064 | 245.50 | 37.40/58.00 | 0.642 |
| Skeleton | 1931/1000 | 0.301000 | 0.133185 | 255.46 | 23.77/30.00 | 0.674 |
| Radar | 1918/996 | 0.087349 | 0.024499 | 215.90 | 22.41/28.00 | 0.556 |
| IR | 1933/1000 | 0.153000 | 0.026103 | 76.35 | 357.34/504.00 | 3.970 |
| Thermal | 1845/1046 | 0.186424 | 0.047492 | 105.77 | 357.32/504.00 | 3.970 |
| Depth_Color | 1931/1000 | 0.142000 | 0.033530 | 50.78 | 357.34/484.00 | 3.970 |

## Pipeline findings

Depth_Color steady-state training throughput increased from 11.928 to 52.878 samples/s (4.43x).
All tested worker configurations (0, 2, 4) exited normally, and all selected configurations completed without worker deadlock or CUDA OOM.
Batch 12 was the provisional throughput choice, but its first two-epoch retest materially reduced visual validation quality because fixed epochs meant about one third as many optimizer updates. The final visual batch is 4 with workers 4; all temporal modalities retain batch 16.
CPU use can remain high because image/CSV/JSON decoding is still performed online, but higher samples/s means GPU input waiting is substantially reduced.
During a 10-sample IR visual-training observation with the provisional batch 12 configuration, total CPU was approximately 29.7%-40.4%, GPU utilization 5%-51%, and nvidia-smi memory 3183-3207 MiB. A final batch 4 visual snapshot used about 2308 MiB process memory. These are approximate manual snapshots; the reproducible PyTorch peak metrics above are authoritative for tensor allocations.
