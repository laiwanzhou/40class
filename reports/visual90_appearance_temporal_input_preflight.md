# Visual90 input preflight — continuity gate blocked

- Date: 2026-09-06.
- Stage: Task1 input inventory, before geometry qualification or encoder download.
- Status: `blocked_continuity`; no formal feature extraction or A/B training launched.
- Canonical population independently rebuilt from manifest/split: train2039, validation388; both40 classes.
- Source scan found22 distinct training trials (43 modality streams,987 IR frames) with legacy numeric-only filenames, for example `IR_00000074.png` / `Depth_00000074_Color.png`.
- None of these continuity blockers is a validation trial. Validation has385 timestamp-key-paired trials and3 without a usable pair; all388 remain in canonical evaluation.

## Why this stops the current protocol

The v2 spec section3.1 requires trusted timestamps or source-counter semantics verified from export evidence. The legacy names do not contain timestamps. Sequential numeric suffixes alone do not establish the source continuity required by this contract. This report does not claim the videos actually contain gaps or are corrupt.

The available dataset metadata contains manifest/split indexes; the searched project reports and producer-side pose-cache code do not establish the legacy counter export semantics. The older `reports/depth_ir_pose_roi_experiment.md` explicitly excluded one of these legacy trials from its strict paired experiment. Existing treatment is context, not permission to change the new protocol.

## Narrow proposed resolution (not applied)

With user authorization, mark these22 trials as temporally unsupported for this experiment, exclude them from gradient/sampler use, retain them in canonical train evaluation with the prescribed train-only class prior, and preserve all validation388. Do not delete or rename files. The remaining canonical training rows number2017, including1935 with IR frames; those IR rows still cover all40 classes. This is an input-support amendment, not a new user split.

Alternative: provide verifiable export documentation or recover original timestamp metadata, then support their source counters under the existing contract. Do not fabricate timestamps or infer physical continuity from sorted filenames.

## Evidence

Full machine-readable inventory: `reports/visual90_appearance_temporal_input_preflight.json` (local generated artifact; contains every canonical row and the exact22 blocker IDs). It records manifest/split/pose SHA256 values and source-key metadata. It does not certify full pixel integrity or geometry.

- Manifest SHA256: `92d195ed82b71a243355d118280a3a891f00fd5643c76ed3c94a927eea0aecad`.
- Split SHA256: `c65c4826bfdd2d16796021abfccd6b1ece0ef7fb0ec786b19cc62cb60e5e57ca`.
- Pose SHA256: `239f2b88cb2a481e3fefbae3dd6390d6c9c21c5ee73df38b29adc7cd9b0b624d`.

The prescribed8 provisional smoke samples were selected from the timestamp-key inventory, but full readability/geometry selection remains pending. One raw IR/Depth pair was visually inspected; that is insufficient to pass the spec's geometry gate. No ROI geometry pass has been recorded.

## Verification boundary

New tests exercise continuous-segment selection, empty/single-frame bins, verified-counter resets, bounded ROI interpolation, whole-track rejection, geometry fail-closed behavior, training-only smoke selection and legacy filename rejection. Test results are reported at commit/turn completion separately. No accuracy conclusion can be drawn from this preflight.
