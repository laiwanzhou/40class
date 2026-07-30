# IMU compact Random Forest finalization design

Date: 2026-07-30

## Scope

This finalizes only the IMU modality. It performs one exact fold-0
reproduction of the frozen compact Random Forest, publishes a validation
reference for later fusion work, fits one production model on the union of
the 2,184 train and 573 validation samples, and exposes a package-validated
inference interface. It does not generate a submission, calibrate
probabilities, tune a model, or incorporate experimental physical, segment,
frequency, or other feature families.

The implementation starts from `origin/IMU_rf` at
`6d493d1fa3a183e2bfa6bd602352c9541a0e73bc` on the local
`IMU_rf_final` branch. Stable and experimental branches remain unchanged and
nothing is pushed.

## Frozen model contract

The feature contract is `imu-rf-summary-v1`, with exactly 2,310 names in the
published order. The estimator is `RandomForestClassifier` with:

- `n_estimators=150`
- `criterion="gini"`
- `max_depth=None`
- `min_samples_split=2`
- `min_samples_leaf=4`
- `max_features="sqrt"`
- `max_leaf_nodes=None`
- `bootstrap=True`
- `max_samples=None`
- `class_weight="balanced_subsample"`
- `n_jobs=-1`
- `random_state=20260725`

Model files use joblib LZMA level 3. Configuration validation compares the
complete estimator parameter map that affects the fitted forest; callers
cannot substitute an experimental schema, another seed, or another
estimator.

## Formal compact source

The compact experiment contains a structural-screen copy and a multiseed
confirmation copy for the same seed. The source locator therefore does not
accept every recursively matching directory. It validates root-level compact
summary and comparison artifacts, requires `trees_150_leaf4` to be the
primary finalist, finds its unique `random_state=20260725` entry in the
formal multiseed comparison, and resolves the referenced multiseed
confirmation run. The run must satisfy the frozen estimator, fold, schema,
feature-count, class-order, manifest, and environment contracts.

The source run's ten files are validated before loading. Its model,
feature-schema, imputer, class-order, config, run, and environment hashes are
recorded. The current scikit-learn major and minor versions must equal the
source metadata versions.

## Reproducibility gate

The reproduction reuses the formal 2,184/573 split, sorted sample IDs,
labels, class order, feature schema, train-only medians, and frozen estimator
parameters. Exactly one model fit is permitted. The fresh run is staged and
atomically published.

Source and fresh runs must agree on sample IDs, labels, class order, feature
schema, imputer medians, complete estimator parameters, all 573 predictions,
the confusion matrix, the zero-F1 class set, and all accuracy, macro
precision, macro recall, macro F1, and weighted F1 values. Probabilities use
`rtol=0` and `atol=1e-15`.

Every tree is compared on `children_left`, `children_right`, `feature`,
`threshold`, `impurity`, `n_node_samples`, `weighted_n_node_samples`, and
`value`. Integer arrays require equality and floating arrays use
`rtol=0, atol=1e-15`. A canonical tree-state SHA-256 is computed from those
arrays. Any mismatch publishes a failed finalization report and prevents all
production fitting.

## Fusion validation reference

Only an exact fresh fold-0 reproduction may provide fusion probabilities.
The five-file validation reference contains sample and user IDs, labels,
predictions, all 40 class probabilities, class order, per-class metrics, and
metadata binding it to both the source and fresh run. The full-data production
model is never used to replace this held-out reference.

## Full-data production set and features

The production set is the disjoint union of the formal fold-0 train and
validation IDs. It must contain exactly 2,757 unique labeled samples, equal
the official `selected_for_run` set, preserve the 40-class order and labels,
and contain no test, unlabeled, or excluded sample.

The formal RF feature artifacts contain already-imputed matrices, so raw
2,310-dimensional summaries are rebuilt read-only from the referenced Stage
2 NPZ files with the stable `extract_summary_features()` implementation. For
all fold-0 train and validation rows, feature names, order, non-missing raw
values, NaN masks, and sample IDs are verified against a reconstruction of
the v1 artifact process. A new production median imputer is fitted once on
all 2,757 raw rows and applied once; the old train-only medians are not
reused.

## Production package

One forest is fitted on all 2,757 transformed rows. There is no production
validation split and no model selection. Resubstitution results are recorded
only under `resubstitution_diagnostics` with `diagnostic_only=true`,
`not_generalization_metric=true`, and `not_for_model_selection=true`.

The atomically published package contains exactly nine files:

1. `model.joblib`
2. `feature_schema.json`
3. `imputer.json`
4. `class_order.json`
5. `resolved_config.json`
6. `inference_metadata.json`
7. `training_data_manifest.json`
8. `training_summary.json`
9. `package_manifest.json`

The manifest binds exact byte counts and SHA-256 values for the other eight
members, total package bytes/MiB, a canonical package hash, package version,
creation commit, and dependency versions. The complete package must be at
most 8 MiB. An oversized staging directory is retained for diagnosis and is
not installed as the formal package.

## Inference contract

The inference loader validates the nine-member package, member hashes,
estimator type and full parameters, feature schema, production imputer, class
order, and metadata before accepting a model. It extracts only
`imu-rf-summary-v1` from Stage 2 artifacts, applies the package imputer, and
returns input-order-recoverable `sample_id`, prediction, 40 probabilities,
and class order.

CSV and NPZ outputs are staged and atomically published into a new directory;
existing output directories are rejected. The NPZ contains `sample_ids`,
`predictions`, `class_probabilities`, and `class_order`. Probabilities must be
finite, shaped `[N, 40]`, sum to one within the approved numeric contract, and
have argmax equal to prediction. The CLI is independent of the current
working directory and never emits a submission.

The formal interface is first checked against the 573-sample fresh
reproduction at `atol=1e-15`. A second smoke test performs two independent
package reloads on ten fixed production-training samples and requires exact
output equality; those training predictions are diagnostics, not a
generalization metric.

## Transactions, protection, and audit

The finalization, production package, and fusion reference each use a unique
sibling staging directory and a final atomic directory rename. Existing
targets and unknown staging/backup residues are rejected. Failure cleanup is
limited to staging created by the current operation.

Before and after execution, every protected RF feature/experiment root is
snapshotted by file count, total bytes, canonical SHA-256, and relative-path
list SHA-256. Any change is Critical. Generated model or data artifacts stay
outside Git. The branch stores only source, tests, configuration, design,
plan, and final result reports.
