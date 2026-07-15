# Task 03 two-epoch pipeline verification report

## Scope and status

All six unimodal baselines passed the real-data smoke test and then completed a
full fold_0 run for two epochs. This run verifies data loading, CUDA training,
validation, checkpointing, metrics, and prediction export. It is not the later
30-40 epoch baseline training and should not be treated as a converged model
comparison.

No test-set path, external data, pretrained weights, fold_1, fold_2, complete
OOF, multimodal model, or Kaggle submission was used or generated.

## Hardware and observed bottleneck

- Python: `D:\Anaconda\envs\pyTorch2.7\python.exe`
- PyTorch: 2.7.0+cu128; torchvision: 0.22.0+cu128
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB (7.96 GiB)
- Training device recorded by every run: `cuda`
- CUDA AMP: enabled
- Windows DataLoader workers: `0` (enforced by code)
- Observed during visual training: CPU 100%, GPU usually about 5%-12%
- Observed NVIDIA process memory: about 1.9-2.1 GiB during visual training

The workload is GPU-trained, but it is input-pipeline bound. With
`num_workers=0`, one CPU process opens and decodes 12 images per trial, resizes
them to 192x192, converts them to tensors, and then supplies the GPU. The GPU
finishes each relatively small MobileNet batch quickly and waits for the next
CPU-prepared batch. The low GPU utilization is therefore CPU/image-decoding
back-pressure, not CPU-only model training.

PyTorch's peak allocated tensor memory is lower than the total process memory
shown by `nvidia-smi`, because CUDA context and library allocations are not all
included in `torch.cuda.max_memory_allocated`.

## Common training protocol

| Setting | Actual value |
| --- | --- |
| Classes | 40 |
| Embedding dimension | 128 |
| Epochs | 2 |
| Seed | 20260715 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Initial learning rate | 3e-4 |
| Epoch-2 learning rate | 1.5e-4 |
| Schedule | CosineAnnealingLR |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| AMP | CUDA float16 autocast + GradScaler |
| Model selection | Highest fold_0 validation accuracy |
| DataLoader | `num_workers=0`, `pin_memory=true` |
| Validation output | sample_ids, labels, logits, embeddings, class_order 0-39 |

All non-image normalization statistics were computed only from fold_0 training
users and saved in the corresponding run directory. Validation users did not
contribute to those statistics.

## Models and algorithms

### IMU

- Input: both `up(LA+RA+C).csv` and `down(LL+RL).csv` as one trial.
- Stable device roles: WTRA, WTLA, WTC, WTRL, WTLL.
- Features: 16 acceleration, angular velocity, angle, magnetic field, and
  quaternion channels per device; 80 channels total.
- Time handling: sort each device by timestamp and interpolate to 256 steps.
- Model: residual 1D TCN, channels 64/128, dilations 1/2, kernels 5 and 3,
  masked mean pooling, 128-D projection, 40-class linear head.
- Batch size: 16; dropout: 0.2.

### Skeleton

- Input: naturally sorted `predictions/*.json`, one JSON per frame.
- Main person: highest mean keypoint confidence when multiple people exist.
- Features: 17x3 keypoints, hip-midpoint root centering, per-frame RMS scale,
  first-order velocity; 102 features per step.
- Time handling: interpolate to 64 steps.
- Model: residual 1D TCN, channels 64/128, dilations 1/2, masked mean pooling,
  128-D projection, 40-class head.
- Batch size: 16; dropout: 0.2.

### Radar

- Input: CSV fields `frame,x,y,z,v,snr,noise`; no intensity field.
- Features: 21 statistics per frame: point count; mean/std/min/max for x, y,
  z, and v; mean/std for snr and noise.
- Missing frames: deterministic zeros plus temporal mask. A trial with no valid
  frame rows remains a valid all-zero masked sample and is not deleted.
- Time handling: interpolate to 64 steps.
- Model: residual 1D TCN, channels 64/128, dilations 1/2, masked mean pooling,
  128-D projection, 40-class head.
- Batch size: 16; dropout: 0.2.

### IR, Thermal, and Depth_Color

- Backbone: MobileNetV3-Small with `weights=None` (no download/pretraining).
- Input: 12 naturally sorted, uniformly sampled frames per trial, resized to
  192x192 and normalized from [0,1] to [-1,1].
- IR: read as grayscale and repeat to three channels.
- Thermal and Depth_Color: retain observed RGB channels.
- Time handling: encode each frame independently, mean pool over 12 frame
  features, project to 128-D, then use a 40-class linear head.
- Batch size: 4; dropout: 0.2.

## Two-epoch results

These values are pipeline-verification measurements after only two epochs.
Different modalities have different available trial counts, so their absolute
accuracies should not be used to claim modality superiority.

| Modality | Train/val | Best epoch | Accuracy | Macro-F1 | Parameters | Checkpoint MB | Inference ms/sample | PyTorch GPU peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMU | 1916/987 | 2 | 0.182371 | 0.043026 | 164,328 | 0.642 | 4.783 | 37.29 |
| Skeleton | 1931/1000 | 2 | 0.277000 | 0.118469 | 172,776 | 0.674 | 4.804 | 23.65 |
| Radar | 1918/996 | 1 | 0.089357 | 0.023688 | 141,672 | 0.556 | 3.980 | 22.29 |
| IR | 1933/1000 | 2 | 0.132000 | 0.029542 | 1,006,024 | 3.970 | 7.419 | 356.87 |
| Thermal | 1845/1046 | 2 | 0.165392 | 0.034676 | 1,006,024 | 3.970 | 6.841 | 356.87 |
| Depth_Color | 1931/1000 | 2 | 0.137000 | 0.035302 | 1,006,024 | 3.970 | 6.542 | 356.87 |

### Visual-modality memory observation

| Visual modality | Batch | Frames/resolution | PyTorch peak MB | Observed process memory |
| --- | ---: | --- | ---: | --- |
| IR | 4 | 12 x 192x192 | 356.87 | roughly 1.9-2.1 GiB |
| Thermal | 4 | 12 x 192x192 | 356.87 | roughly 1.9-2.1 GiB |
| Depth_Color | 4 | 12 x 192x192 | 356.87 | roughly 1.9-2.1 GiB |

The 8GB GPU has substantial memory headroom at these settings. This report does
not change batch size because the goal was first to prove correctness with a
stable Windows configuration. A later performance-tuning pass can test larger
visual batches while retaining `num_workers=0`, then separately investigate a
Windows-safe caching or prefetch strategy.

## Output contract verified

Each formal run contains `config.yaml`, `best_model.pt`, `last_model.pt`,
`history.csv`, `metrics.json`, `fold_0_val_predictions.npz`, and
`confusion_matrix.png`. Sequence runs also contain training-only
`normalization_stats.json`. Every checkpoint is below the 95 MB safety gate,
and every saved prediction archive was reopened and shape-checked.

Model weights, prediction NPZ files, and all `outputs/` content remain ignored
by Git. Only source, configuration, schema/smoke reports, and small summary
files are eligible for commit.
