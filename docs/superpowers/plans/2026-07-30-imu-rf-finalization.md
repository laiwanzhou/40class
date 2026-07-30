# IMU compact Random Forest finalization implementation plan

Date: 2026-07-30

This plan executes on local branch `IMU_rf_final`, based directly on
`origin/IMU_rf@6d493d1fa3a183e2bfa6bd602352c9541a0e73bc`. Every behavior change
uses RED -> GREEN. No step pushes, merges, rebases, creates a submission, or
changes a protected external artifact.

## Task 1: Freeze configuration and source discovery

Files:

- Create `configs/imu_rf_final_v1.json`
- Create `src/training/imu_rf_finalization.py`
- Create `tests/imu_rf/test_final_config.py`
- Create `tests/imu_rf/test_reproducibility.py`

RED tests reject every changed RF parameter, another seed, an experimental
feature schema, a non-RF estimator, an incompatible environment, ambiguous
formal records, or a malformed source run. Tests prove the root compact
summary/comparison selects the primary finalist's unique multiseed
confirmation rather than the duplicate structural-screen copy.

GREEN implements strict config loading, root-artifact source selection,
ten-file validation, source provenance collection, and model metadata
validation.

Verify focused tests, `git diff --check`, and commit:

`feat(imu): add compact rf reproducibility gate`

## Task 2: Exact fold-0 reproduction gate

Files:

- Modify `src/training/imu_rf_finalization.py`
- Create `scripts/run_imu_rf_finalization.py`
- Modify `tests/imu_rf/test_reproducibility.py`
- Create `tests/imu_rf/test_finalization_cli.py`

RED tests cover sample/label/class/schema/imputer/config mismatches,
prediction and metric mismatches, the fixed probability tolerance, every
tree array, deterministic tree-state hashing, single-fit enforcement, and
the rule that a failed gate cannot invoke production work.

GREEN builds one fresh run with the existing feature artifact, validates it,
compares source/fresh models and outputs, writes comparison JSON/Markdown,
and atomically publishes the finalization root. The CLI supports preflight
and formal execution from any CWD.

After focused tests pass, run exactly one formal seed-20260725 reproduction.
If `reproducibility_status` is not `exact_match`, stop the entire plan.

## Task 3: Publish the fusion validation reference

Files:

- Modify `src/training/imu_rf_finalization.py`
- Modify `tests/imu_rf/test_reproducibility.py`

RED tests require exactly five files, user IDs, held-out source provenance,
40-class probabilities, manifest hashes, and rejection of production-model
training predictions.

GREEN derives the artifact only from the exact fresh reproduction, validates
it after reopening, and atomically publishes it.

## Task 4: Build and validate the 2,757-sample production matrix

Files:

- Create `src/training/imu_rf_production.py`
- Create `tests/imu_rf/test_production_dataset.py`
- Create `tests/imu_rf/test_production_imputer.py`

RED tests cover 2,184/573 counts, disjointness, union equality with selected
2,757, label/class order agreement, exclusion of test/unlabeled rows,
2,310-feature reconstruction, raw-value/NaN-mask equality, and production
imputer provenance.

GREEN reads the formal training index and Stage 2 artifacts, rebuilds raw
summaries, verifies the v1 extractor contract, fits medians on all 2,757 raw
rows, and produces finite transformed data in production staging.

## Task 5: Fit and package one full-data model

Files:

- Modify `src/training/imu_rf_production.py`
- Create `scripts/train_imu_rf_production.py`
- Create `tests/imu_rf/test_production_package.py`

RED tests cover one-fit enforcement, fixed config/seed, explicit
resubstitution labels, the exact nine-file set, member hashes/bytes,
canonical package hashing, LZMA level 3 reload, and the 8 MiB publication
gate.

GREEN fits one forest, records non-generalization diagnostics, creates the
nine-file package transactionally, reopens every member, and publishes only
when the complete package is at most 8 MiB.

Run the formal production build exactly once. If data, features, or size
gates fail, stop without publishing a formal package.

## Task 6: Implement standard inference

Files:

- Modify `src/inference/__init__.py`
- Create `src/inference/imu_rf_inference.py`
- Create `scripts/run_imu_rf_inference.py`
- Create `tests/imu_rf/test_inference.py`

RED tests cover package/config/schema/imputer/class-order mismatch, Stage 2
feature extraction, finite `[N,40]` probabilities, row sums, argmax,
input-order recovery, identical CSV/NPZ content, current-working-directory
independence, output refusal, transaction cleanup, and absence of submission
behavior.

GREEN implements strict package loading, batch feature extraction/inference,
validated CSV/NPZ publication, and CLI entry.

Use this interface to replay the 573 validation samples against the fresh
reproduction at `atol=1e-15`, then perform two independent reloads on ten
fixed production-training samples.

Commit production and inference implementation in scoped commits:

- `feat(imu): add full-data rf production package`
- `test(imu): validate final rf inference contract`

## Task 7: Full verification and result record

Files:

- Create `docs/superpowers/reports/2026-07-30-imu-rf-finalization-results.md`
- Create `docs/superpowers/reports/2026-07-30-imu-rf-finalization-results.json`

Reopen all three new output roots and validate their exact contracts. Run all
RF tests, all Stage 2 tests, the Stage 1 regression, the full repository,
`py_compile` for every tracked Python file, all three CLI help commands from
outside the repository with `PYTHONPATH` cleared, `git diff --check`, and
`git fsck --full --strict`.

Recompute all protected directory snapshots and require exact equality.
Require zero task-owned staging/backup residues and no large generated blob
in Git. Record commits, exact reproduction evidence, production package
details, inference checks, output hashes, tests, and protections.

Commit:

`docs(imu): record rf finalization results`

Stop with the local branch preserved for independent review. Do not push.
