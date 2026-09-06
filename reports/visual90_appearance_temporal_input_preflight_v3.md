# Visual90 input preflight v3 — authorized training exclusions

Date: 2026-09-06. Supersedes the continuity-blocked status in the original preflight; original artifacts remain unchanged for audit.

## Competition context supplied by the user

Actual competition test may naturally lack some modalities for certain actions. Training and validation are expected to have all modalities. Observed missing development-data paths therefore require investigation and must not be justified by the test-time rule. No competition test files were accessed.

User explicitly authorized trials genuinely missing timestamps to be excluded from training. This authorization does not permit validation exclusion, fabricated timestamps, relabeling, or speculative cross-modality trial matching.

## Executed preflight

| Category | Count | Treatment |
|---|---:|---|
| Canonical training rows | 2039 | Retained in audit; canonical metric separate from fit metric |
| Missing-timestamp training trials | 22 | Excluded from all fitting/sampling/normalization/prior estimation/smoke |
| Other training rows with no IR in index | 82 | Separate visual-data issue, not a timestamp exclusion |
| IR-present training candidates after exclusion | 1935 | All40 classes; geometry/ROI qualification still required |
| Canonical validation rows | 388 | Unchanged, user6/user7 |
| Timestamp-key-paired validation rows | 385 | No continuity blockers found |
| Validation rows without indexed IR/Depth | 3 | Remain in canonical evaluation, no automatic deletion |

Remaining unexcluded continuity blockers:0. Current status:`awaiting_geometry`. This is not a completed geometry, resource or training qualification.

Exclusion is derived from exact legacy numeric-only frame-name recognition and recorded per sample_id in `visual90_appearance_temporal_input_preflight_v3.json`. Other parsing/continuity errors remain blocking; the training permission is not applied to validation. The model/sampler have not yet been implemented, so this preflight defines their required eligible population rather than claiming a completed training run.

## Missing modality investigation

All85 zero-IR canonical rows are Thermal-only in the existing manifest. An exact raw-path check for all85 under IR and Depth_Color found no matching `(class,user,trial)` directory. This is more than an empty manifest field, but it does not establish the original collection cause or rule out a different trial naming/mapping. Do not assume a nearby trial with the same action is its counterpart.

The three validation rows are:

- `train__c15__user6__3-2-3`
- `train__c15__user6__3-3-2`
- `train__c39__user7__4-1-2`

Example neighboring raw folders: class15/user6 has `3-3-1`, `3-3-3`, `6-2-1`, `6-2-2`, `6-2-3` in IR/Depth; class39/user7 has `4-1-1`, `4-1-3`, `7-1-1`, `7-1-2`, `7-1-3`. No matching was inferred from these names. Availability masks remain required for real test missingness, independently of this local data/correspondence anomaly.

## Provenance and next gate

Full machine-readable inventory is a local generated artifact at `reports/visual90_appearance_temporal_input_preflight_v3.json`; the prior inventory is retained. Manifest/split/pose hashes are unchanged from the original preflight. Full pixel readability/integrity and shared ROI geometry are not certified by frame-key inspection. Continue with the prescribed training-only geometry evidence and then encoder/full-batch resource gates; no formal cache or training has been launched.
