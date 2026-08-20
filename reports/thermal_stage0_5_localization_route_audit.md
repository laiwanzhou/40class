# Thermal Stage T0.5 Localization Route Audit

## Scope and boundaries

- Baseline commit: `c42bb43091c79903e5fde5655c2846c87305895a`; branch: `experiment/thermal-native-expert`.
- Population is the official train-14 users only. Sealed heldout users, heldout labels, competition test, and quarantined evidence were not enumerated or opened.
- IR/X3D source, weights, configs, and ExpertEvidence remained read-only. No model, detector, or classifier was trained or tuned.
- T0.5 reused the T0 YOLO outputs at the same Thermal-native uniform normalized-time points. It did not import IR indices or IR bboxes and did not use motion peaks for frame selection.

## Executive findings

- Codex structured visual review covered **56 trials / 268 frames** across all six T0 YOLO montage pages. Verdict: `yolo_is_quality_bearing_conditional_locator_not_a_mandatory_crop`. Independent human sign-off remains pending.
- Candidate-chain route coverage on the same frames: `{"full_frame": 21, "thermal_heat_motion_context": 19, "thermal_yolo_context": 228}`; fallback from YOLO was **14.9%**.
- Heat/motion proposal valid-trial rate was **53.6%**. Against accepted YOLO boxes, its diagnostic coverage proxy was `{"min": 0.4061003242801815, "p25": 1.0, "median": 1.0, "p75": 1.0, "p95": 1.0, "max": 1.0, "mean": 0.9917515315819248}`; YOLO is not treated as ground truth.
- Heat/motion was selective (valid expanded bbox below 95% of frame) on only **7.1%** of trials; **92.9%** expanded to at least 95% of the frame. It is not promoted into the frozen online route.
- T0 anomalies were retained: 23 singleton, 34 at most four frames, 122 below 13 frames, and 5 exact duplicate frames in 4 trials.
- Rendered auto-scale indication counts were `{"moderate": 8, "unavailable": 2, "weak": 46}`. These are RGB drift indicators only; absolute temperature-scale stability remains unproven.
- Frozen first-experiment input: **full_frame**. Conditional localization is retained as a later matched ablation and as label-free quality, not silently mixed into the first baseline.

## 1. Structured YOLO montage review

Verdict counts: `{"correct_subject_localization_on_reviewed_frames": 41, "correct_subject_with_context_risk": 10, "mixed_with_background_false_candidate": 3, "mixed_with_localization_collapse": 2}`.

Observed failure types: `{"background_hot_object_candidate": 1, "edge_clipping": 1, "extremely_short_trial": 1, "far_field_small_subject": 3, "hand_or_object_only_candidate": 1, "head_only_candidate": 1, "large_pose_change": 1, "low_confidence_gap": 5, "near_field_edge_clipping": 3, "oversized_box_includes_furniture": 1, "partial_body_crop": 1, "recurrent_low_confidence": 2, "sofa_or_table_candidate": 1, "standing_to_horizontal_pose_change": 1, "subject_exits_frame": 1, "tight_crop_context_loss": 3, "tight_crop_instability": 1, "wrong_small_background_candidate": 1}`.

| Factor | Trials | Frames | Detection rate |
| --- | ---: | ---: | ---: |
| near_field_or_edge_contact | 39 | 185 | 87.0% |
| far_field_small_subject | 17 | 85 | 83.5% |
| seated_or_lying_action_context | 33 | 159 | 85.5% |
| gross_motion_action_context | 10 | 50 | 92.0% |
| single_or_extremely_short | 5 | 13 | 84.6% |

Structured visual review confirms that many high-confidence boxes identify the person, but near-field clipping removes hands/objects, far seated subjects create confidence gaps, and a few low-confidence candidates attach to furniture or hot objects. A numeric person detection is therefore not accepted as a complete action crop. This Codex review does not replace the still-pending independent human authorization gate.
Factor strata are descriptive audit summaries only. Class/action context never enters localization routing or Thermal frame selection.

## 2. Full-frame, YOLO-context, and heat/motion-context audit

- Route rates: `{"full_frame": 0.07835820895522388, "thermal_heat_motion_context": 0.0708955223880597, "thermal_yolo_context": 0.8507462686567164}`.
- Selected bbox area ratio: `{"min": 0.11067519801126764, "p25": 0.2508261383827909, "median": 0.4270571635722672, "p75": 0.6167770445346833, "p95": 1.0, "max": 1.0, "mean": 0.494896191436493}`.
- YOLO-context continuity: `{"min": 0.2734488321848996, "p25": 0.7801614439073058, "median": 0.9025367766724002, "p75": 0.960304709053592, "p95": 0.9861894569285686, "max": 1.0, "mean": 0.8362000518979045}`.
- Final conditional-route continuity: `{"min": 0.11067519801126764, "p25": 0.562828613245904, "median": 0.8565024138442348, "p75": 0.9508571469255305, "p95": 1.0, "max": 1.0, "mean": 0.7463632258911168}`.
- Heat/motion component area: `{"min": 0.11233072916666667, "p25": 0.7950358072916667, "median": 0.9434374999999999, "p75": 0.9834765624999999, "p95": 0.99810546875, "max": 1.0, "mean": 0.8434874906994049}`; expanded area: `{"min": 0.328125, "p25": 1.0, "median": 1.0, "p75": 1.0, "p95": 1.0, "max": 1.0, "mean": 0.9722377232142857}`.
- Heat/motion uses only the uniformly sampled Thermal frames. Motion differences contribute to the union mask, but peaks never choose training frames.
- Heat/motion selectivity rate: **7.1%**; full-frame-equivalent expanded-box rate: **92.9%**. Its high box-coverage proxy mostly comes from retaining almost the whole frame, not from reliable person localization.
- Full-frame retains all action context and is valid for every decodable trial. YOLO context is accepted only at confidence >=0.25 and raw bbox area >=5%; heat/motion rejects components below 5% or above 95%; every failure falls back explicitly.

Route montages:

- `reports/thermal_stage0_5_montages/thermal_route_comparison_montage_01.jpg`
- `reports/thermal_stage0_5_montages/thermal_route_comparison_montage_02.jpg`
- `reports/thermal_stage0_5_montages/thermal_route_comparison_montage_03.jpg`
- `reports/thermal_stage0_5_montages/thermal_route_comparison_montage_04.jpg`
- `reports/thermal_stage0_5_montages/thermal_route_comparison_montage_05.jpg`
- `reports/thermal_stage0_5_montages/thermal_route_comparison_montage_06.jpg`

## 3. Anomaly review

- Short-trial relation assessments: `{"asynchronous_or_partial_capture_needs_metadata": 43, "missing_modality_prevents_count_relation_assessment": 4, "plausible_asynchronous_rate_difference": 35, "short_capture_across_modalities": 33, "thermal_capture_truncated_or_export_incomplete": 7}`.
- IR/Thermal Tukey outliers: **327**; assessments: `{"asynchronous_or_partial_capture_needs_metadata": 194, "ir_depth_capture_truncated_or_export_incomplete": 6, "plausible_asynchronous_rate_difference": 97, "short_capture_across_modalities": 23, "thermal_capture_truncated_or_export_incomplete": 7}`.
- Present and decodable single/short directories are not directory errors. Where Thermal has 1-4 frames and IR/Depth have tens or hundreds, the evidence supports a partial Thermal capture/export candidate, not a proven sensor fault. The reverse asymmetry similarly implicates IR/Depth capture/export. Acquisition metadata is required to distinguish sensor stop from export truncation.
- All canonical rows remain. Availability stays true for decodable short trials; frame scarcity, source uniqueness, duplicate ratio, and fallback route remain quality fields.
- Visual review of all seven short-trial montage pages found genuine short action snippets, but also some room-only frames, edge-only bodies, and partial action fragments. These are visibility/quality defects, not grounds for deleting canonical samples.

Short-trial montages:

- `reports/thermal_stage0_5_montages/thermal_short_trial_montage_01.jpg`
- `reports/thermal_stage0_5_montages/thermal_short_trial_montage_02.jpg`
- `reports/thermal_stage0_5_montages/thermal_short_trial_montage_03.jpg`
- `reports/thermal_stage0_5_montages/thermal_short_trial_montage_04.jpg`
- `reports/thermal_stage0_5_montages/thermal_short_trial_montage_05.jpg`
- `reports/thermal_stage0_5_montages/thermal_short_trial_montage_06.jpg`
- `reports/thermal_stage0_5_montages/thermal_short_trial_montage_07.jpg`

## 4. Rendered pseudocolor and auto-scale indications

- Static-background luminance median span: `{"min": 0.0, "p25": 1.0, "median": 2.0, "p75": 4.0, "p95": 12.049999999999983, "max": 17.0, "mean": 3.3518518518518516}`.
- Static-background hue median span: `{"min": 0.0, "p25": 0.25, "median": 1.0, "p75": 3.0, "p95": 23.399999999999977, "max": 63.0, "mean": 4.462962962962963}`.
- Rendered endpoint-fraction span: `{"min": 0.0, "p25": 7.812499999999989e-05, "median": 0.00016927083333333334, "p75": 0.0002571614583333333, "p95": 0.0006204427083333319, "max": 0.0012369791666666668, "mean": 0.00021870177469135804}`.
- Drift is assessed on low-temporal-variance background pixels and global rendered luminance percentiles. Motion and scene changes can still contaminate these metrics. Without raw temperatures, emissivity, camera range, or palette metadata, neither weak nor strong RGB drift proves an absolute temperature scale.

## 5. Frozen preprocessing decision

- First iFormer-T+TSM development view: `full_frame`.
- Conditional ablation priority: `thermal_yolo_context_then_full_frame`.
- Fallback order: `thermal_yolo_context -> full_frame`.
- Heat/motion status: `offline_quality_diagnostic_not_default_route_due_low_selectivity`.
- Required quality fields: `directory_present, decodable_frame_fraction, distinct_frame_ratio, unique_sampled_source_ratio, duration_bucket, localizer_confidence, raw_bbox_area_ratio, expanded_bbox_area_ratio, bbox_continuity, heat_motion_component_area_ratio, heat_motion_valid, fallback_route, fallback_reason, rendered_auto_scale_evidence_strength`.
- The first development baseline uses the simple full-frame path for every decodable Thermal trial. This avoids route-dependent context loss and makes the matched MobileNetV3 control exact. A later YOLO-context/full-frame ablation may use the same split and sampler, but may not replace the baseline without a controlled result. The audited heat/motion proposal is too close to full-frame to justify online route complexity.
- Singleton and short trials use the frozen 16-target Thermal sampler with explicit repeated-source mask. No canonical sample is deleted, and motion peaks, IR indices, and IR bboxes remain prohibited.

**Stage T0.5 stop:** preprocessing is frozen for the first development experiment, but no training is authorized or performed in this stage.
