# Raw reconstruction evidence

**Current acceptance:** see `ACCEPTANCE_20260911.md/json` for the newly installed,
isolated environment. The sections below describe historical development-environment
checks and do not certify the current entry scripts or dependency installation by themselves.

The historical frozen-release run is `release_raw_regression.json`: **224.00
seconds, 405 rows, zero prediction differences, exact historical CSV SHA256**.
The earlier `final_raw_regression.json` is an intermediate timestamp-grid experiment,
not the shipped implementation: it also preserved predictions but changed internal
tensors. The release restores the shared IR acquisition grid and uses native
Skeleton timestamps only when IR is absent. Pixel/IMU equality is restored.

The first full fresh-raw run used 405 official input rows and completed in 232.91
seconds on the observed Windows/RTX 5070 Ti Laptop environment. It generated the
same CSV bytes as the frozen scored submission. This proves reproduction of those
predictions on this machine, not accuracy on a new private test set.

The inference runtime received only the raw data directory, unfilled path CSV and
the bundled checkpoint. Historical CSV/cache comparison occurred afterward in the
developer-only regression script. There was no test-ground-truth comparison.

Pixel tensors, frame selections, view masks/qualities and all IMU tensors exactly
match historical caches. One of 405 Skeleton source records differs numerically:
5,956 feature elements (maximum absolute difference 7.54296875) and 470 relation
elements (maximum absolute difference 0.676513671875). Its frame IDs, time axis,
ROI geometry, masks and qualities agree. The newly computed clip scale is
0.81369483; the old cached scale was 0.8122479. The difference is already present
in the source Skeleton feature calculation, before windowing. Its exact historical
generation/version cause remains unresolved; it is not claimed to be roundoff.

This discrepancy did not change any of the 405 predictions. No per-row correction,
exception or cached-value substitution was added. Every row uses the same current
raw preprocessing. The package must not claim bitwise equality of all internal
tensors, nor use these comparisons to alter action labels.

The final-pass report additionally records the hashes of the packaged runtime
sources. `report.json` from an operator's future run records bundle/input/output
hashes, partial status and timing. Preserve these reports when reproducing results.
