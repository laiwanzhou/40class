# Thermal T1-A Candidate Qualification

## Scope

Corrective source identity, environment, source/license/hash, strict pretrained loading, TSM forward, and deployment-byte audit only. No training, labels, sealed data, competition test, or ExpertEvidence were read.

Scientific baseline remains `c42bb43091c79903e5fde5655c2846c87305895a`.

## Source identity correction

The earlier T1-A audit inspected Sail-SG's homonymous *Inception Transformer*. That finding was valid for that repository but irrelevant to the intended candidate. The authoritative source is Chuanyang Zheng's *iFormer: Integrating ConvNet and Transformer for Mobile Application* at revision `2a87540fcb345afe9d950a58d0eb3873b938c3dc`, where the official symbol is `iFormer_t` and the source-code license is MIT.

## Decision

- **iFormer-T + TSM: technically eligible primary candidate, pending human training approval.** The official `iFormer_t(pretrained=True)` constructor does not download or load weights. The probe therefore explicitly downloads the checkpoint, verifies its fixed SHA, extracts it from the legacy training bundle, and strictly loads it with zero missing/unexpected keys before replacing the ImageNet classifier. The TSM wrapper produced finite `[2,16,3,224,224] -> [2,40]` output. Random-initialization fallback is prohibited.
- **pretrained MobileNetV3-Small + TSM: technically eligible matched control, with a weight-terms caveat.** Strict pretrained load completed with zero missing/unexpected keys; `[2,16,3,224,224] -> [2,40]` is finite. Torchvision source is BSD-3-Clause, while its official model documentation says pretrained-weight permission remains the user's responsibility because training-data terms may apply.
- **iFormer-S + TSM: budget/load eligible only as the frozen conditional upgrade.** Its correct-family checkpoint also loads strictly. The 40-class TSM state-dict proxy is 25,400,973 bytes and the provisional package is 49,045,173 bytes, leaving 45,954,827 bytes. It may not train before the primary and matched-control comparison authorizes the upgrade gate.

## Resources

| Candidate | Parameters | FP32 parameter bytes | Serialized bytes | Peak CUDA allocated | Median trial latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| iFormer-T + TSM | 2,639,736 | 10,558,944 | 10,840,569 | 87,148,544 | 5.923 ms |
| MobileNetV3-Small + TSM | 1,558,856 | 6,235,424 | 6,365,399 | 57,529,344 | 4.620 ms |
| iFormer-S + TSM budget proxy | 6,255,208 | 25,020,832 | 25,400,973 | not run | not run |

## Deployment Ledger

Current retained learned assets are frozen IR/X3D-S and shared YOLO11n-pose: 20,644,200 bytes after deduplicating the repeated YOLO reference. A 3,000,000-byte calibration/fusion upper-bound reserve is included in candidate projections. Skeleton, IMU, Radar, Depth, Thermal, and fusion assets are not yet retained, so a complete six-modal package pass is not claimed.

Corrective T1-A stops here. No optimizer, backward pass, epoch loop, or learned Thermal weight was created. T1-B training remains unauthorized until explicit human approval.
