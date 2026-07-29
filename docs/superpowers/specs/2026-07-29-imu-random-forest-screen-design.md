# IMU Random Forest Screen Design

## Scope

This experiment lives only on `IMU_rf`, forked from the formal `IMU` commit
`64fb41728db500979c367be959e5e28482ef8c05`. It consumes the accepted fold-0
Grid Stage 2 artifacts and training index without changing them. It does not
alter or replace the formal TCN, and it does not implement fusion.

## Feature contract

`imu-rf-summary-v1` produces one deterministic row per selected sample. The
five sensor positions and sixteen Stage 2 channels remain distinct. Each
sensor/channel pair emits mean, population standard deviation, minimum,
maximum, median, 25th and 75th percentiles, peak-to-peak range, RMS,
first-to-last delta, mean absolute first difference, first-difference standard
deviation, valid count, and valid ratio. Every scalar has an adjacent missing
indicator.

Each sensor additionally emits valid-span duration, valid time-point count,
invalid ratio, contiguous-valid-segment count, longest invalid run, and a
whole-sensor-missing flag. Sample-level features are usable-sensor count,
all-five-valid ratio, sequence length, sequence duration, and overall missing
ratio. These scalars also have adjacent missing indicators.

Only cells selected by `valid_mask` contribute. Padding, invalid cells, and
NaNs never contribute. Unavailable statistics remain NaN until a median
imputer is fitted on train only; validation only applies that frozen imputer.
Identifiers, paths, users, labels, and class names are metadata, never model
features. The final matrix must be finite and have the exact schema order.

The feature artifact directory is transactionally published and contains
exactly train/validation NPZ matrices, schema, imputer, sample manifest, and
provenance. Provenance binds the source Stage 2 canonical tree hash, training
index hashes, schema, train sample IDs, Git commit, every member SHA-256, and a
canonical hash over all members except the self-referential provenance file.

## Frozen RF screen

Both variants use `RandomForestClassifier` with 300 trees, `max_features` set
to `sqrt`, no maximum depth, `min_samples_leaf=2`, bootstrap enabled, and all
available CPU workers. Plain RF uses no class weight. Balanced RF uses
`balanced_subsample`. Seeds are 20260724, 20260725, and 20260726. Validation
never selects or changes these values.

Each run atomically publishes exactly ten files: model bundle, schema,
feature importances, validation predictions and probabilities, confusion
matrix, per-class metrics, resolved config, training summary, and run manifest.
The model bundle binds Python/scikit-learn versions, feature schema and imputer
hashes, class order, seed, and full RF configuration. Strict reload rejects a
binding mismatch.

## Evaluation

The six runs report accuracy, macro precision/recall/F1, weighted F1, zero-F1
classes, prediction concentration, tree depth, duration, and model size.
Three-seed summaries compare variants with the formal CE TCN baseline and the
separately labelled `IMU_test` CE+SupCon candidate. Sample-level comparison
uses the formal TCN validation output to report RF-only correct, TCN-only
correct, both correct, both wrong, and ideal-selector accuracy. No automatic
model replacement, fusion, submission, merge, or push follows from the screen.
