# Stage 9 selected-relative formal experiment: early-stop diagnosis

## Stop integrity

The user-requested `STOP_AFTER_EPOCH` sentinel was observed at the next epoch boundary. Epoch 16 completed training and validation, then history and checkpoints were written before the process exited normally. Both best checkpoints were subsequently reloaded and evaluated. The pipeline status is `complete`; competition test and sensor modalities were not read.

## Result

| checkpoint / point | epoch | train Accuracy | val Accuracy | val Macro-F1 | val weighted F1 | val loss | predicted classes | zero-F1 classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best Accuracy | 11 | 0.4112 | **0.2780** | 0.1349 | 0.2306 | 3.1498 | 31 | 21 |
| best Macro-F1 | 14 | 0.5272 | 0.2559 | **0.1585** | 0.2545 | 3.2683 | 37 | 16 |
| stopped point | 16 | 0.5940 | 0.2186 | 0.1272 | 0.2144 | 3.6172 | 38 | 20 |
| B2-256 reference | 25 | 0.9763 | **0.4627** | **0.3928** | 0.4630 | 3.4091 | 39 | 2 |

The formal schedule disproved the idea that the low eight-epoch pilot score was only caused by its compressed cosine schedule. With the 30-epoch schedule, validation Accuracy plateaued in the 22%-28% range after Epoch 6 while training Accuracy continued rising. The best-Accuracy deficit relative to B2 is `-0.1847`; the best-Macro-F1 deficit is `-0.2343`.

## Why this structure underperformed B2

### 1. The experiment removed B2's strongest modality interaction

B2 uses a pretrained Depth_Color stem and a separate IR stem, fuses their spatial feature maps immediately after the stem, and sends the fused map through the full shared MobileNet body. Every one of its four views has both modalities.

The new structure instead treats IR as the high-capacity primary path. Ordinal Depth is processed by a small randomly initialized encoder, pooled to a vector before fusion, and introduced as a gated residual initialized to only `0.10`. Depth is available only for context and relation views. This loses pixel-level Depth/IR correspondence and sharply reduces Depth capacity. Although Depth_Color is pseudocolor, B2 could still exploit its dense monotonic depth boundaries and spatial alignment; inverse-JET plus lightweight vector fusion is not behaviorally equivalent.

### 2. Full-sequence input removed a useful temporal prior and augmentation

B2 trains on 24-frame windows. Long trials supply different windows across epochs, which acts as temporal augmentation and limits idle/noisy context. The new loader presents the same complete trial on every epoch. Its TCN and attention pooling must discover the informative interval while also processing long idle periods, tracking failures, and redundant adjacent frames. More frames therefore increased optimization burden and identity/background exposure instead of automatically adding action evidence.

### 3. Model capacity increased while user diversity did not

The new model has about `2.245M` trainable parameters versus B2's `1.269M` (1.77x), but still only 14 training users. Expanded context plus four overlapping IR views repeatedly expose clothing, body shape, scene, and user-specific motion style. The widening train/validation gap after Epoch 6 shows that this capacity increasingly fits training-user evidence rather than transferable action evidence.

### 4. Spatial BatchNorm still sees correlated frame/view chunks

Activation checkpointing now preserves BatchNorm running-state correctness, but statistical quality is a separate issue. Spatial encoding chunks contain adjacent frames and overlapping views from a small number of trials. Gradient accumulation does not enlarge a BatchNorm batch. Length-bucketed sampling further groups trials with similar temporal structure. These conditions can make ImageNet BatchNorm adaptation noisy and user/action-specific, although this experiment does not isolate its exact contribution.

### 5. One learning rate serves incompatible components

The pretrained IR backbone and randomly initialized Depth encoder, fusion layers, TCN, attention pooling, and heads all use the same `2e-4` learning rate. Early training underfits the new components; later training memorizes the train users while validation stalls. The curve therefore transitions from optimization difficulty to overfitting without reaching B2-level transfer.

### 6. Too many changes were bundled to assign the loss to TCN alone

This run simultaneously changed temporal coverage, temporal encoder, ROI definitions, Depth representation, modality capacity, fusion location, and auxiliary supervision. It proves that the complete package is inferior, but it does not prove that full-sequence TCN itself is harmful. Earlier controlled evidence that only changed the temporal module should retain higher weight when deciding whether full-sequence processing is worth preserving.

## Decision supported by the data

The current package should not replace B2 and is not a credible path to 60%-70% Accuracy through additional epochs. The formal run was worth doing because it separates the compressed-pilot concern from the structural result: the longer schedule improved training fit and class coverage, but did not restore validation Accuracy.

Some ideas remain worth isolated testing, especially full-sequence temporal modeling and improved interaction ROIs. They should be reintroduced one at a time on top of B2's proven high-capacity, pixel-aligned Depth/IR spatial fusion. The specific combination `IR-primary + lightweight ordinal Depth vector residual + all-frame six-view TCN` should be treated as rejected unless a future ablation identifies and repairs one of its individual bottlenecks.
