# Stage 5: ordinal Depth smoke export audit

- Status: **passed**
- Samples / frames / classes: 82 / 3,562 / 40
- Train / val frames: 1,877 / 1,685
- Depth PNGs: 14,248 (expected 14,248)
- Recomputed ordinal value mismatches: 0 pixels
- Recomputed pixel-mask mismatches: 0 pixels
- Nonzero values behind invalid masks: 0 pixels
- Raw scalar low-information flags: 121
- Pose-valid scalar content-invalid views: 2
- Pose-invalid zero placeholders among those flags: 119
- Final content-invalid views after retaining Stage 2 evidence: 2
- Mean pixel coverage, context / relation: 0.8132 / 0.8065
- Neighboring-frame contact sheets: 12

## Checks

- PASS: `success_marker`
- PASS: `sample_count_82`
- PASS: `all_40_classes`
- PASS: `both_splits`
- PASS: `metadata_frame_count`
- PASS: `not_full_dataset_export`
- PASS: `duplicate_frame_keys_zero`
- PASS: `frame_order_failures_zero`
- PASS: `depth_png_count_exact`
- PASS: `missing_files_zero`
- PASS: `wrong_shape_zero`
- PASS: `wrong_dtype_zero`
- PASS: `mask_nonbinary_zero`
- PASS: `invalid_nonzero_zero`
- PASS: `value_mismatches_zero`
- PASS: `mask_mismatches_zero`
- PASS: `content_mismatches_zero`
- PASS: `timestamps_and_deltas_match`
- PASS: `ir_missing_zero`
- PASS: `ir_not_duplicated`
- PASS: `stage2_depth_invalid_retained`

Competition test was not read, no model was trained, and the full export was not started.
