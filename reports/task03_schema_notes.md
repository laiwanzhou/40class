# Task 03 training schema notes

Only two training trials per modality were inspected. No test-set path was read.

| Modality | Observed trial contents | Reader decision |
| --- | --- | --- |
| Depth_Color | RGB PNG, 640x480, naturally sortable timestamp/frame names | Uniformly sample 12 frames, resize to 192x192, keep RGB |
| IR | Grayscale PNG (`L`), 640x480 | Uniformly sample 12 frames, convert grayscale to three equal channels |
| Thermal | RGB JPEG, 320x240 | Uniformly sample 12 frames, resize to 192x192, keep RGB |
| IMU | `up(LA+RA+C).csv` and `down(LL+RL).csv`; UTF-8 Chinese headers | Sort each device by `时间`; use WTRA/WTLA/WTC/WTRL/WTLL and 16 motion/orientation channels; resample each role to 256 |
| Skeleton | `predictions/*.json`; each frame is a list of people with 17x3 `keypoints` and `keypoint_scores` | Select the person with highest mean score, center on hip midpoint, RMS-scale, append velocity, resample to 64 |
| Radar | One CSV with `timestamp, frame, DetObj#, x, y, z, v, snr, noise` | Build 21 statistics per frame, represent missing frame IDs with zero plus temporal mask, resample to 64 |

Fixed fold modality counts (train/validation): Depth_Color 1931/1000, IR
1933/1000, Thermal 1845/1046, IMU 1916/987, Skeleton 1931/1000, and
Radar 1918/996.

Task 03 smoke and first-round settings are deliberately conservative for
Windows and an 8GB GPU: `num_workers=0`, visual batch size 4, temporal batch
size 16, CUDA AMP enabled, and two epochs. The two-epoch run is a pipeline
verification run, not the later 30-40 epoch baseline training.
