# Thermal Route A A0 Input Audit

**Status:** `pending_human_montage_approval`

No model was trained. Heldout labels, competition test, quarantined evidence, IR/Depth inputs, and frozen IR/X3D evidence were not read or modified.

## Canonical population

- Canonical: **2427**; usable: **2299**; unavailable retained: **128**.
- Train12 normalization: **1922 trials / 5712 frames**.
- RGB mean `[0.6677977762218014, 0.2642016758744894, 0.3548692698442481]`; std `[0.24659209711299557, 0.23712366371061488, 0.21979727234388996]`.

## Fixed Thermal context

- Available: **2207/2299 (96.0%)**.
- Fallbacks: `{"insufficient_detection_hits": 53, "no_detection": 39, "thermal_unavailable": 128}`.
- Probe detection: **81.9%** over 18137 frames.
- Median confidence `0.601`; crop area `0.750`; adjacent IoU `0.887`.

One square union box is fixed for each Thermal trial. No IR bbox or motion-peak frame selection is used; every failure remains a full-frame fallback.

## Representative tensors

- **56 trials / 40 classes / 2229 unique pose frames**.
- Pose valid-step rate: **61.2%**.
- Shapes: `{"availability": [4], "crop_rgb": [3, 16, 3, 160, 160], "full_rgb": [3, 16, 3, 160, 160], "motion": [3, 16, 1, 160, 160], "pose": [3, 16, 56], "pose_mask": [3, 16], "quality": [8]}`.
- Finite and shape-consistent: `True`.

Montage pages:

- `reports/thermal_v2_input_montages/thermal_v2_context_01.jpg`
- `reports/thermal_v2_input_montages/thermal_v2_context_02.jpg`
- `reports/thermal_v2_input_montages/thermal_v2_context_03.jpg`
- `reports/thermal_v2_input_montages/thermal_v2_context_04.jpg`
- `reports/thermal_v2_input_montages/thermal_v2_context_05.jpg`
- `reports/thermal_v2_input_montages/thermal_v2_context_06.jpg`

## Stop

Machine audit is complete, but A0 is not approved. A human must inspect every montage page before the workflow may advance to A1.
