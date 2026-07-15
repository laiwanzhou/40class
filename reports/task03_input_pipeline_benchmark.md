# Task 03 input pipeline benchmark

The original configuration used `num_workers=0` for every modality, visual
`batch_size=4`, and temporal `batch_size=16`. All tested combinations passed;
there were no worker failures, deadlocks, or CUDA OOM cases.

Each passed combination used 10 warmup batches and up to 100 measured training batches in an isolated Python process.
Measurements include real data loading and decoding, H2D transfer, CUDA AMP forward/backward, gradient clipping, and AdamW step.

| Modality | Workers | Batch | Measured batches | Samples/s | Batch s | Peak allocated MB | Peak reserved MB | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Depth_Color | 0 | 4 | 100 | 11.928 | 0.3353 | 337.64 | 408.00 | passed |
| Depth_Color | 2 | 4 | 100 | 32.174 | 0.1243 | 337.64 | 408.00 | passed |
| Depth_Color | 4 | 4 | 100 | 52.878 | 0.0756 | 337.64 | 408.00 | passed |
| Depth_Color | 4 | 8 | 100 | 48.933 | 0.1635 | 631.05 | 864.00 | passed |
| Depth_Color | 4 | 12 | 100 | 57.986 | 0.2069 | 927.36 | 1268.00 | passed |
| Depth_Color | 4 | 16 | 100 | 58.171 | 0.2751 | 1224.34 | 1728.00 | passed |
| IMU | 0 | 16 | 100 | 65.379 | 0.2447 | 36.03 | 58.00 | passed |
| IMU | 2 | 16 | 100 | 187.025 | 0.0856 | 36.03 | 58.00 | passed |
| IMU | 4 | 16 | 100 | 225.225 | 0.0710 | 36.03 | 58.00 | passed |
| Skeleton | 0 | 16 | 100 | 72.278 | 0.2214 | 23.25 | 28.00 | passed |
| Skeleton | 2 | 16 | 100 | 222.298 | 0.0720 | 23.25 | 28.00 | passed |
| Skeleton | 4 | 16 | 100 | 260.136 | 0.0615 | 23.25 | 28.00 | passed |
| Radar | 0 | 16 | 100 | 65.943 | 0.2426 | 22.20 | 28.00 | passed |
| Radar | 2 | 16 | 100 | 180.286 | 0.0887 | 22.20 | 28.00 | passed |
| Radar | 4 | 16 | 100 | 265.055 | 0.0604 | 22.20 | 28.00 | passed |
| IR | 4 | 12 | 100 | 65.160 | 0.1842 | 927.36 | 1268.00 | passed |
| Thermal | 4 | 12 | 100 | 113.843 | 0.1054 | 927.36 | 1268.00 | passed |

## Selected configurations

- `depth_color`: `num_workers=4`, `batch_size=12`
- `imu`: `num_workers=4`, `batch_size=16`
- `skeleton`: `num_workers=4`, `batch_size=16`
- `radar`: `num_workers=4`, `batch_size=16`
- `ir`: `num_workers=4`, `batch_size=12`
- `thermal`: `num_workers=4`, `batch_size=12`

Selection requires a passed run with no CUDA OOM or worker failure. Throughput is primary; when configurations are within 5% of the best rate, the smaller worker or batch value is selected.
CPU utilization may remain high; success is judged by stable completion and measured throughput rather than a single utilization snapshot.

Depth_Color improved from 11.928 samples/s at workers 0 and batch 4 to 57.986
samples/s at workers 4 and batch 12, a 4.86x increase. Batch 16 reached 58.171
samples/s, only 0.32% faster, while peak reserved memory rose from 1268 MB to
1728 MB; batch 12 was therefore selected. IMU, Skeleton, and Radar improved by
3.45x, 3.60x, and 4.02x respectively when moving from workers 0 to workers 4.
IR and Thermal both passed their 100-batch confirmation at workers 4 and batch
12. The pipeline still performs online decode and may remain CPU-limited, but
the measured throughput increase demonstrates substantially less GPU waiting.
