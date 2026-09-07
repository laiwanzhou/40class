# Visual90 resource smoke — stop before formal training

Measured 2026-09-06; closeout 2026-09-07. No30-epoch training, full-population extraction, validation fitting or retained smoke optimizer state. This report qualifies measured resource settings, not teacher accuracy or readiness of the entire formal training pipeline.

## Recommended settings

For frozen-feature GPU fusion: **physical/effective batch32, workers4, prefetch_factor1, persistent_workers=true, pin_memory=true**. Main PyTorch threads8, each worker1. Batch64 passed capacity tests but is not adopted because changing batch changes the preregistered optimization/SupCon protocol.

For extraction: VideoMAE clip batch1; DINO image batch16 is the best measured of1/4/8/16. The current smoke extractor used4 DINO frames per call; batching across records to16 needs a metadata-preserving implementation before adopting the recommendation. Raw decoder workers4 is supported by the measured test.

CPU can run batch32 but is not recommended for formal work on this machine. Batch4 is a convenient CPU debug setting, not an equivalent replacement for batch32 SupCon training.

## Hardware and scope

RTX5060 Laptop GPU,8151MiB; Ryzen9 8945HX,16 physical/32 logical cores; approximately32GB RAM. Initial GPU free memory was about5GB because desktop programs were active; none were closed. PyTorch CUDA allocation below excludes other processes and is not total device use.

The prescribed8 training trials were used for real feature extraction. Available clip-views: global31, person31, left31, right28; one clip was empty. Source hashes and masks are recorded in the local smoke evidence. Geometry inspection supports corresponding local regions in those examples, not perfect pixel registration or all acquisition configurations.

## GPU fused-model backward smoke

Every case includes warmup and3 measured forward/backward/AdamW steps with CE and the implemented cross-user SupCon. Model/optimizer state is discarded.18 synthetic-layout CPU/GPU cases passed, with changed parameters and finite losses/gradients.

| Candidate / input | Batch | Median compute step | Peak CUDA allocated |
|---|---:|---:|---:|
| A, real cached features | 32 | 0.05527s | 637.1MiB |
| B, real cached features | 32 | 0.08260s | 906.0MiB |
| A, full-layout synthetic pressure | 64 | 0.10249s | 1227.3MiB |
| B, full-layout synthetic pressure | 64 | 0.14163s | 1755.2MiB |

At CPU batch32/8 threads, the synthetic-layout step was1.228s for A and2.069s for B. CPU batch1/4/8/16/32 all passed. These short, sequential cases are not confidence intervals or controlled thermal steady-state benchmarks.

## Integrated loader + GPU backward, B batch32

| Workers | Warm median step | Warm samples/s | First step | Observed process-tree RSS |
|---|---:|---:|---:|---:|
| 0 | 0.18060s | 177.2 | 0.898s | 2314MiB |
| 2 | 0.10516s | 304.3 | 4.724s | 3902MiB |
| 4 | 0.09005s | 355.4 | 5.234s | 5220MiB |

Each case ran12 steps, reporting the median after the first2; all completed with finite losses and gradients. Four workers roughly doubled the observed warm throughput relative to zero, with additional RAM/startup cost. The shorter12-step wall time still favors zero workers because startup dominates; persistent workers are important for longer runs.

Separate two-epoch loader tests delivered512 examples for each worker setting, with identical first-batch video checksum17618.162109375. This validates the reported first batch and successful loading, not byte equality of every tensor in every batch. The backing mmap cache has only8 examples and was repeatedly accessed; these are hot-cache observations, not full-corpus cold I/O guarantees.

Raw16-frame clip decoding, batch1:11.92 clips/s with workers0,25.37 with2,36.16 with4. Startup increased from0.062s to7.846s/15.313s. Full source corpus decode cost can differ.

## Encoder inference batches

| Encoder | Batch | Items/s, forward only | Peak CUDA allocated |
|---|---:|---:|---:|
| VideoMAE | 1 | 14.73 clips/s | 641.7MiB |
| VideoMAE | 2 | 14.57 clips/s | 947.9MiB |
| VideoMAE | 4 | 13.52 clips/s | 1555.8MiB |
| VideoMAE | 8 | 13.30 clips/s | 2773.1MiB |
| DINOv2-L | 1 | 43.49 images/s | 1184.9MiB |
| DINOv2-L | 4 | 73.36 images/s | 1202.9MiB |
| DINOv2-L | 8 | 86.82 images/s | 1230.5MiB |
| DINOv2-L | 16 | 90.81 images/s | 1292.8MiB |

Actual small-cache extraction:242 video calls in33.57s,121 DINO calls (4 images each) in12.83s, excluding model initialization outside the timed loops. These durations are not a measured full-corpus ETA. Official checkpoints loaded strictly; DINO source revision/weights are locked in `configs/experiments/visual90_encoder_lock.json`. xFormers was unavailable; upstream PyTorch attention fallback was used, not a substituted backbone.

## Verification and retained evidence

On2026-09-07, the first closeout run of the complete test suite passed **441 tests in83.70s**. Cache array SHA256 values and the evidence SHA256 were recomputed and matched the completion manifest. All four raw resource reports explicitly state `formal_training_started=false`.

Independent closeout review found no issue invalidating the resource measurements. It identified two pre-formal safeguards, now fixed with observed failing-then-passing regressions: model initialization preserves the caller's CUDA RNG (CPU-only initialization generator inside fork_rng), and every feature-cache constructor verifies all array hashes before workers start, including integrated mode. Original9/6 timings precede this RNG fix; they are retained as measurements of that revision, not claimed as exact post-fix reruns. Hash checking happens outside timed warm steps. CPU initial model weights, tensor layouts and attention topology are unchanged. Historical report gaps (source/config snapshot, free GPU memory and encoder reserved peak) are not retroactively fabricated; add these to future full-run reporting.

After these fixes, the fresh full suite passed **443 tests in61.03s**. The reviewer independently confirmed both fixes and reported no new blocking issue for this smoke closeout.

Raw reports and cache remain at `outputs/visual90_appearance_temporal/smoke/`: `encoder_report.json`, `encoder_batch_report.json`, `resource_report.json`, `integrated_report.json`, `features/complete.json`. The four reports are archived together in `reports/visual90_appearance_temporal_smoke_measurements.json`. Runtime recommendations are saved separately and do not enable formal training.

Pending beyond this closeout: full-population geometry applicability, complete source/config-locked resumable cache workflow, formal sampler/runner/checkpoint recovery and final388-row evaluation. Do not mark all plan tasks complete or report this smoke as val_acc evidence.
