# Thermal T1-A Candidate Qualification

## Scope

Environment, source/license/hash, strict pretrained loading, TSM forward, and deployment-byte audit only. No training, labels, sealed data, competition test, or ExpertEvidence were read.

## Decision

- **iFormer-T + TSM: ineligible / blocked.** Official Sail-SG iFormer revision `725d8e7f455b5e17be20788b9bcd6c6c505c4be0` publishes S/B/L only. No official iFormer-T architecture or pretrained weight was found, so no loader, config, or forward was fabricated.
- **pretrained MobileNetV3-Small + TSM: technically eligible matched control, with a weight-terms caveat.** Strict pretrained load completed with zero missing/unexpected keys; `[2,16,3,224,224] -> [2,40]` is finite. Torchvision source is BSD-3-Clause, while its official model documentation says pretrained-weight permission remains the user's responsibility because training-data terms may apply.
- **iFormer-S + TSM: ineligible on budget.** Official pretrained loading is complete, but the 40-class state-dict proxy is 78,132,757 bytes and the provisional package is 101,776,957 bytes. No config, TSM runtime, forward, or training was created.

## Resources

| Candidate | Parameters | FP32 parameter bytes | Serialized bytes | Peak CUDA allocated | Median trial latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3-Small + TSM | 1,558,856 | 6,235,424 | 6,365,399 | 57,529,344 | 6.573 ms |
| iFormer-S 40-class budget proxy | 19,496,744 | 77,986,976 | 78,132,757 | not run | not run |

## Deployment Ledger

Current retained learned assets are frozen IR/X3D-S and shared YOLO11n-pose: 20,644,200 bytes after deduplicating the repeated YOLO reference. A 3,000,000-byte calibration/fusion upper-bound reserve is included in candidate projections. Skeleton, IMU, Radar, Depth, Thermal, and fusion assets are not yet retained, so a complete six-modal package pass is not claimed.

T1-A stops here pending a human decision on the undefined iFormer-T candidate identity.
