# fold_0 14/4 compatibility notes

The repository was searched before changing `fold_0.json`.

## Numeric sample-count search

Command:

```powershell
rg "1916|987|1931|1000|1918|996|1845|1046" scripts src configs
```

Findings:

- `scripts/summarize_task03_baseline_fold0.py:27-32` defines
  `EXPECTED_SAMPLES` for the completed 12-train/6-validation formal Baseline.
  That script is historical-result validation and must not be reused for a new
  14/4 training run until expected counts are computed dynamically from the
  manifest and selected fold.
- `src/train_unimodal.py:248` matched only the unrelated `1000.0` conversion
  from seconds to milliseconds; it is not a split-size dependency.

## Old validation-user search

Command:

```powershell
rg "user1|user5|user6|user16|user18|user19" scripts src configs
```

No old validation-user list is hardcoded in `scripts`, `src`, or `configs`.
`src/data/common.py::load_modality_frames` dynamically reads `train_users` and
`val_users`, so the new JSON fields do not break the loader.

## Additional compatibility findings

- All six YAML files continue to reference `metadata/splits/fold_0.json` and
  were intentionally not modified.
- `scripts/prepare_task01.py::build_fold` still generates the original
  three-way `StratifiedGroupKFold` first fold. Running that script with the
  repository metadata output would overwrite the new 14/4 fold. Do not rerun
  it against this output location unless fold generation is explicitly being
  redesigned. The archived old fold remains the reproducible 12/6 input.
- Existing formal Baseline reports and ignored outputs describe the archived
  12/6 fold and must remain unchanged.
- A later 14/4 Baseline runner should use a new output root and run IDs so it
  cannot collide with `outputs/task03_baseline_fold0`.

No training, test-set access, model change, or YAML change was performed during
this compatibility review.
