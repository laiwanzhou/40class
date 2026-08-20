# Thermal Stage T0 Data and Alignment Audit

## Scope and hard boundaries

- Baseline commit: `c42bb43091c79903e5fde5655c2846c87305895a`.
- Branch: `experiment/thermal-native-expert`; worktree: `D:\work\2026.7.14_kaggle\40class-thermal-native-expert`.
- Population: official train-14 users only. The sealed users `user4,user17,user23,user24`, heldout labels, competition test, and quarantined evidence were not enumerated or opened.
- IR/X3D is frozen. This audit performed no training, tuning, checkpoint selection, or ExpertEvidence write.
- Raw modality data and YOLO weights were opened read-only; outputs are reports and montages in this worktree.
- Deployment ledger at T0: frozen IR X3D + shared YOLO known subtotal `20,644,200` bytes; new Thermal learned artifacts `0` bytes; hard complete-package ceiling `<95,000,000` bytes. A complete-package pass is not claimed before the other retained experts and a trained Thermal candidate are inventoried.

## Executive findings

- Canonical train-14 trials: **2427**. Thermal directory present: **2299**; decodable: **2299**; usable: **2299**. These are separate states.
- Thermal image files: **160125**; decoded: **160125**; corrupt JPEG: **0**; exact duplicate decoded frames: **5**.
- Single-frame trials: **23**; 2-4-frame-or-single trials: **34**; below 13 frames: **122**. None were removed from the canonical population.
- Rendering verdict: `stable_pseudocolor_rendering; grayscale replication rejected; per-frame auto-scale unresolved`. Channel evidence rejects grayscale replication, but rendered RGB JPEGs cannot establish or exclude per-frame automatic temperature scaling because raw radiometric values and scale metadata are absent.
- IR/Thermal common trials: **2214**; median Thermal/IR frame-count ratio: **2.419**. Thermal has independent frame numbering and no shared filename timestamps with IR.
- Motion alignment on 56 stratified trials produced median correlation `0.784` with materially varying offsets/scales. Motion peaks are diagnostic only and are not part of the sampling contract.
- Thermal-native YOLO detection rate: **85.4%** on 268 representative frames; median detected confidence: **0.590**; median adjacent-sample bbox IoU: **0.826**.
- Current evidence supports trial-level fusion. It does not support frame-level IR/Thermal registration or scaled reuse of IR boxes.

## 1. Thermal data audit

`directory_present` means the canonical trial directory exists. `decodable` means at least one image decodes. `usable` means at least one Thermal frame can enter the simple full-frame/native-localization path; defects and short duration remain label-free quality fields.

### Overall

- Frame-count distribution over decodable trials: `{"min": 1.0, "p25": 32.0, "median": 54.0, "p75": 90.5, "p95": 171.0, "max": 595.0, "mean": 69.64984775989561}`.
- Trial-level distinct-frame-ratio distribution: `{"min": 0.9873417721518988, "p25": 1.0, "median": 1.0, "p75": 1.0, "p95": 1.0, "max": 1.0, "mean": 0.9999823531475508}`.
- Rendered palette-curve stability distribution: `{"min": 0.3696536650351585, "p25": 0.9885983502975648, "median": 0.9932038721200551, "p75": 0.9947409688291221, "p95": 0.9977777777777778, "max": 1.0, "mean": 0.9870512283892775}`.
- Resolutions: `{"320x240": 2299}`.
- Rendering categories: `{"stable_pseudocolor": 2299, "unknown_no_decodable_sample": 128}`.
- Pseudocolor stability is assessed from channel spread and hue-versus-luminance palette curves on uniformly located frames. This demonstrates a consistent rendered palette family, not calibrated temperature equivalence across frames.

### By development split

| development_split | Canonical | Present | Decodable | Usable | Missing | Frames min/median/p95/max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train12 | 2039 | 1922 | 1922 | 1922 | 117 | 1/54/172/595 |
| val_user6_user7 | 388 | 377 | 377 | 377 | 11 | 1/59/165/572 |

### By OOF validation owner

| oof_fold | Canonical | Present | Decodable | Usable | Missing | Frames min/median/p95/max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fold_0 | 821 | 808 | 808 | 808 | 13 | 1/50/159/595 |
| fold_1 | 676 | 594 | 594 | 594 | 82 | 1/64/218/497 |
| fold_2 | 930 | 897 | 897 | 897 | 33 | 1/54/164/572 |

### By user

| user_id | Canonical | Present | Decodable | Usable | Missing | Frames min/median/p95/max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| user1 | 153 | 151 | 151 | 151 | 2 | 3/77/254/404 |
| user16 | 186 | 185 | 185 | 185 | 1 | 1/65/171/391 |
| user18 | 179 | 177 | 177 | 177 | 2 | 1/49/133/595 |
| user19 | 188 | 183 | 183 | 183 | 5 | 2/47/133/289 |
| user2 | 168 | 95 | 95 | 95 | 73 | 6/62/228/497 |
| user20 | 159 | 159 | 159 | 159 | 0 | 4/67/148/252 |
| user21 | 133 | 132 | 132 | 132 | 1 | 3/42/170/353 |
| user22 | 194 | 182 | 182 | 182 | 12 | 1/36/122/286 |
| user3 | 163 | 154 | 154 | 154 | 9 | 1/54/181/570 |
| user5 | 160 | 155 | 155 | 155 | 5 | 1/75/192/464 |
| user6 | 203 | 195 | 195 | 195 | 8 | 13/68/173/572 |
| user7 | 185 | 182 | 182 | 182 | 3 | 1/48/163/200 |
| user8 | 169 | 163 | 163 | 163 | 6 | 10/57/156/351 |
| user9 | 187 | 186 | 186 | 186 | 1 | 3/44/118/177 |

Class-level details are preserved in the JSON report to keep this document scannable.

## 2. IR/Depth temporal relationship

- Pairing by directory sort position is forbidden. Each modality is independently naturally ordered, assigned `t=i/(N-1)`, and only then mapped by nearest normalized time as a candidate correspondence.
- IR/Thermal frame-count ratio: `{"min": 0.013888888888888888, "p25": 2.144179894179894, "median": 2.418768328445748, "p75": 2.5625, "p95": 3.5, "max": 57.0, "mean": 2.413698460797521}`.
- Depth/Thermal frame-count ratio: `{"min": 0.013888888888888888, "p25": 2.1481481481481484, "median": 2.4193548387096775, "p75": 2.5625, "p95": 3.5, "max": 57.0, "mean": 2.41414361535343}`.
- IR/Thermal Tukey outliers: **327**; representative extremes are listed in the JSON report.
- IR/Depth filenames carry matching wall-clock timestamps and frame numbers in the audited export. Thermal filenames carry only independent `frame_N` indices; no common timestamp boundary is available.
- Motion diagnostic offset: `{"min": -0.25, "p25": -0.12249999999999989, "median": 0.010000000000000231, "p75": 0.16000000000000036, "p95": 0.23850000000000046, "max": 0.25000000000000044, "mean": 0.009318181818182048}`; scale: `{"min": 0.8, "p25": 0.8800000000000001, "median": 0.9800000000000002, "p75": 1.1000000000000003, "p95": 1.2000000000000004, "max": 1.2000000000000004, "mean": 0.9945454545454548}`; DTW: `{"min": 0.12294963137518683, "p25": 0.2773360060708707, "median": 0.4196148121040711, "p75": 0.5473612639279067, "p95": 0.7352181616977378, "max": 0.825786089740914, "mean": 0.42378554402244356}`.
- Correlation/DTW can diagnose trial-relative lag or rate mismatch. Without synchronized timestamps, calibration targets, or stable cross-trial offset/scale, they do not prove frame identity.

### Evidence boundary

| Claim | Stage T0 evidence | Decision |
| --- | --- | --- |
| Same trial identity | Canonical class/user/trial key and common directories | Sufficient for trial-level late fusion |
| Candidate relative-time correspondence | Independent normalized timelines | Diagnostic only |
| Temporal frame registration | No shared Thermal timestamps; variable count ratio and motion offset/scale | Not established |
| Spatial registration / IR bbox transfer | Different resolution plus no camera calibration or pixel correspondence | Not established; prohibited |

## 3. Thermal-native localization

- YOLO weights: `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` (6255593 bytes), threshold `0.25`, `imgsz=640`.
- Confidence distribution: `{"min": 0.25715744495391846, "p25": 0.43238580226898193, "median": 0.590074896812439, "p75": 0.7437078356742859, "p95": 0.8637099623680115, "max": 0.9062853455543518, "mean": 0.5883573637258538}`.
- Bbox continuity distribution: `{"min": 0.0, "p25": 0.6779647107682492, "median": 0.8260843889666722, "p75": 0.9342081677442755, "p95": 0.9767328717209498, "max": 0.9943450748914524, "mean": 0.7741422969586574}`; IoU >= 0.5 rate: `0.8961038961038961`.
- Failure types: `{"bbox_discontinuity_iou_below_0.25": 3, "edge_clipped_bbox": 126, "low_confidence_below_0.25": 39, "multiple_person_candidates": 29, "tiny_bbox": 1}`.
- Representative coverage: **14 users**, **40 classes**, duration buckets `2_to_8, 33_to_96, 9_to_32, over_96, single_frame`. Per-user, per-class, and per-duration results are in the JSON report.
- Numeric detections are not accepted as ground truth. The montage is the required human check for false people, missed limbs, clipped subjects, furniture/background detections, and context loss.
- Human montage verdict: `conditional_locator_only_full_frame_fallback_required`. Observed modes: near-field subjects frequently touch image boundaries; far seated or crouched subjects produce small or low-confidence boxes; person scale and pose changes can cause low-confidence gaps and box discontinuity; when a subject leaves the frame, a small heated object or background region can be selected as a false person; tight person-only crops risk discarding action-defining table, screen, cup, phone, and room context.
- IR bboxes were neither read nor scaled. If Thermal-native YOLO proves unreliable after human review, retain a heat/motion context crop when label-free confidence is adequate and a full-frame fallback otherwise; encode the route and reliability in availability/quality. Do not train a detector in T0.

Montages:

- `reports/thermal_stage0_montages/thermal_yolo_pose_montage_01.jpg`
- `reports/thermal_stage0_montages/thermal_yolo_pose_montage_02.jpg`
- `reports/thermal_stage0_montages/thermal_yolo_pose_montage_03.jpg`
- `reports/thermal_stage0_montages/thermal_yolo_pose_montage_04.jpg`
- `reports/thermal_stage0_montages/thermal_yolo_pose_montage_05.jpg`
- `reports/thermal_stage0_montages/thermal_yolo_pose_montage_06.jpg`

## 4. Frozen Thermal sampling contract

- Timeline: `Thermal natural frame order with t=i/(N-1); singleton t=0`.
- Targets: `uniform normalized-time targets mapped to nearest Thermal source index`.
- Short trials: `retain canonical row; repeat nearest source index only when a fixed tensor is required; emit source-uniqueness mask and quality`.
- Quality: `directory_present, decodable_frame_fraction, distinct_frame_ratio, unique_sampled_source_ratio, duration_bucket, localizer_confidence, bbox_continuity, fallback_route`.
- The sampler never imports IR indices, never uses IR's 13-frame contract, and never selects motion peaks. Variable duration stays explicit; singleton/short trials remain canonical and use repeated nearest-source indices plus a mask/quality signal when a fixed tensor is required.

## 5. Frozen candidate order after T0

1. **iFormer-T + TSM**: primary budget-friendly Thermal-native candidate.
2. **iFormer-S + TSM**: upgrade only after pretrained-loading provenance, serialized bytes, and complete-package headroom pass.
3. **Pretrained MobileNetV3 + TSM**: matched control using the same Thermal sampler, localization routes, training protocol, and evaluation surface.
4. **Deferred**: VideoMamba, DART, and IR+Thermal early fusion. They are outside the first controlled Thermal generation.

## 6. Approval gate and later evidence contract

After human approval, development starts only on the fixed train12 / user6-user7 validation split. The recipe is then frozen before generating formal train-14 OOF evidence on the shared persisted folds.
Every later Thermal expert must emit 40-class logits, availability, label-free quality/quality-mask, fusion-quality score, hashes, deployed bytes, preprocessing dependencies, and fold lineage through `ExpertEvidence`. Reports must include Accuracy, Macro-F1, worst-user Accuracy, IR unique-correct/oracle-pair metrics, exact deployment bytes, and latency.

**Stage T0 stop:** no formal model was trained. Await human review of this report and montages before Stage T1.
