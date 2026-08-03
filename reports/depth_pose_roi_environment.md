# Depth pose ROI environment check

Checked on 2026-08-03 with the required interpreter
`D:\Anaconda\envs\pyTorch2.7\python.exe`. No package was installed, upgraded,
removed, or otherwise modified during this check.

## Available runtimes

| Component | Available | Version | Notes |
| --- | --- | --- | --- |
| Python | yes | 3.12.9 | Anaconda environment `pyTorch2.7` |
| PyTorch | yes | 2.7.0+cu128 | CUDA available on NVIDIA GeForce RTX 5060 Laptop GPU |
| torchvision | yes | 0.22.0+cu128 | Includes MobileNetV3-Small |
| ultralytics | yes | 8.3.165 | Selected for the feasibility probe |
| mediapipe | yes | 0.10.21 | Available fallback; not selected for the first probe |
| OpenCV | yes | 4.11.0 | Used for image loading and overlays |
| ONNX Runtime | yes | 1.23.2 CPU | Not required by the selected PyTorch model |
| mmpose / rtmlib / openpifpaf | no | N/A | No installation attempted |

## Selected pose locator

The probe uses the official Ultralytics `yolo11n-pose.pt` nano model. It returns
a person bounding box and the 17 COCO pose keypoints, including nose, shoulders,
elbows, wrists, and hips. The official weight was fetched by Ultralytics after
confirming that the package was already installed.

- Weight file: `yolo11n-pose.pt`
- Weight size: 6,255,593 bytes (5.97 MiB)
- Detection threshold for the formal probe: 0.25
- Keypoint confidence threshold: 0.25
- Inference image size: 640
- Device: CUDA device 0

The weight is ignored by Git through the existing `*.pt` rule. Its 5.97 MiB
size leaves substantial room under the 100 MB combined inference-weight limit.
The final combined size can only be reported after an expert classifier exists.

## Initial paired-frame check

One paired `Take_medicine` frame was tested before the full probe. At confidence
0.15, the Depth_Color frame produced one person with confidence 0.172, while the
aligned IR frame produced one person with confidence 0.538. This is only a
runtime check, not feasibility evidence; the full probe evaluates both sources
at the fixed 0.25 threshold.
