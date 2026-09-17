# Data Leakage and Validation Provenance Audit

Date: 2026-09-08  
Scope: read-only audit of the frozen submission and the local source/test manifests. No
test ground truth, leaked mapping, or sample label was opened or reconstructed.

## Executive result

This audit does **not** certify the training ancestry as an outer-pure reconstruction. The frozen
submission is a valid-format artifact and its compact Student alone is below 100 MB, but
the historical teacher/OOF ancestry is not independently source-pure from the evidence
available here. The package is therefore suitable only as a historical candidate with an
explicit provenance limitation, not as proof of leakage-free generalisation.

## Findings

### 1. Historical OOF/calibration provenance is unresolved

The historical reconstruction audit records these limitations:

- the base visual teacher smooths **global OOF across all rows**;
- historical probability artifacts are not sufficient evidence of independent fitting;
- repeat-pair construction requires source-safe rebuilding rather than importing pairs from
  a global supervised bank; and
- no fitted global OOF/calibration model may be relabelled “outer-pure”.

That audit also states that a complete training reconstruction must preserve every fitted
preprocessing/calibration/neighbor ancestor. Consequently, `summary.json` fields such as
`test_labels_read: false`, `rule_uses_labels: false`, and `teacher_agreement: 1.0` are
**self-reported run metadata**, not an independent proof that all historical ancestors were
outer-pure or that no hidden test-derived artifact entered upstream caches.

### 2. Manifest-level subject and label checks

Read-only checks on the current manifests found:

| check | result |
|---|---:|
| training manifest rows | 2,914 |
| distinct training `user_id` values | 18 (`user1`, `user2`, `user3`–`user9`, `user16`–`user24`) |
| test manifest rows | 405 |
| test `user_id` values | all empty |
| test `class_id` | sentinel `-1` for all rows |
| test `class_name` | empty for all rows |
| sample-id overlap | 0 |
| Depth/IR/Skeleton directory-string overlap | 0 for each modality |

Manifest SHA256 values observed during this audit:

- training: `29ecaf62328dd395bed449dd1af2c27cfb63b53d1fce39870ad7bef1931955`
- test: `8b9449a3647b2eb394977d987782450f703d8af9d2cac604e0320f8cd58ca8e6`

These are safe identity/path checks only. They do not prove that feature caches or old
OOF arrays contain no duplicated content.

### 3. Raw Skeleton content-overlap audit

Every raw file below each manifest `skeleton_dir` was hashed with SHA-256. Files below any
path component named `visualizations` were excluded. For each trial, the file hashes were
reduced to a sorted multiset (`file_hash -> multiplicity`) and hashed again with SHA-256.
No sample IDs, paths, class IDs, or train-to-test pairings were emitted.

Results (elapsed wall time: 41.624 s):

| quantity | result |
|---|---:|
| train trials / raw files | 2,914 / 85,248 |
| test trials / raw files | 405 / 9,201 |
| unique train trial digests | 2,893 |
| exact train–test trial-digest intersection | 0 |
| distinct raw-file SHA-256 hash intersection | 0 |
| matching raw-file occurrences (train / test) | 0 / 0 |

The raw-file set intersection was computed independently from the trial-digest comparison
(train unique file hashes: 81,976; test unique file hashes: 8,986). This is an exact-content
check for the covered raw Skeleton files only. It does not audit
RGB/Depth/IR/IMU files, derived caches, historical OOF tensors, or semantic near-duplicates.

### 4. Weight budget: historical risk resolved in the new single bundle

The historical Student (`94,393,969` bytes) plus standalone P28 YOLO pose
(`6,255,593` bytes) would total `100,649,562` bytes and exceed the strict limit. The
new package instead uses one quantized bundle:

`checkpoints/model.pth = 56,471,879 bytes`

`build_manifest.json` records `includes_pose_weights: true` and bundle SHA256
`28d41bf833c191ddf354e8da27b2c9c91ac58d30b7b58aa5f9cd4096d42febde`. The package
weight budget is therefore currently below 100 MB. This is a packaging correction, not
evidence that the historical training ancestry was leakage-free.

### 5. Test metadata and leaderboard feedback

The historical teacher pipeline uses label-free test recording metadata (date/time/session and device
statistics) for repeat/session grouping. This audit found no label column in that test
metadata, but it cannot certify that the metadata cannot act as a side channel in the
presence of the historically confirmed public-data leak. The reported `0.91542` score and
the two-row difference from an earlier submission are external feedback; they are not ground-truth
evidence and must not be used for further row-level edits.

## Not established by this audit

- No claim that the submitted solution used leaked test labels.
- No claim that this packaging task retrained all 30 historical experts.
- No claim that the 0.91542 score transfers to a replacement test set.
- No byte-level equality audit of RGB/Depth/IR/IMU or every historical cache.

## Submission disposition

The single-bundle package now also has a clean-environment raw-input acceptance run
showing405/405 prediction agreement; see ACCEPTANCE_20260911.md. The earlier internal
Skeleton numerical discrepancy remains disclosed. Prediction agreement is neither a new
score nor a leakage certificate. For a replacement test set,
regenerate predictions from the README pipeline using only
the new raw directory and source-fitted artifacts. Do not reuse old per-row decisions,
old test caches, timestamp-to-label associations, or leaderboard-derived edits. Before
submission, produce a fresh manifest hash, a fresh input/cache provenance list, and a
strict sum of every checkpoint actually loaded at inference; reject the package if the
sum is not `<100,000,000` bytes.
