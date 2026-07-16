# Task 03 local output cleanup report

Cleanup completed at `2026-07-16 11:29:47 +08:00`.

## Disk usage

- Outputs before: **234,674,082 bytes (223.803 MiB)**.
- Outputs after: **56,439,574 bytes (53.825 MiB)**.
- Space released: **178,234,508 bytes (169.978 MiB)**.

| Deleted directory | Bytes released | MiB released | Classification |
| --- | ---: | ---: | --- |
| `outputs/task03` | 31,316,055 | 29.865 | Initial old-12/6 full-data two-epoch validation. |
| `outputs/task03_smoke` | 32,041,816 | 30.557 | Initial old-12/6 smoke outputs. |
| `outputs/task03_optimized_smoke` | 58,352,864 | 55.650 | Optimized old-12/6 smoke outputs. |
| `outputs/task03_optimized_two_epoch` | 56,523,773 | 53.905 | Optimized old-12/6 two-epoch outputs. |

Deletion used the exact allowlist in
`scripts/cleanup_task03_local_outputs.ps1`. The script resolved every target,
verified that it was a direct child of `outputs`, rejected protected/baseline
names, checked that no target was Git-tracked, and then used literal paths.
A repeated dry-run reported all four targets already absent and made no change.

## Preserved outputs

| Directory | Bytes | MiB | Status |
| --- | ---: | ---: | --- |
| `outputs/task03_baseline_fold0` | 31,341,399 | 29.889 | Protected and fully validated formal 12/6 Baseline. |
| `outputs/task03_worker_probe` | 25,098,175 | 23.935 | `manual_review_required`; retained because no tracked source/report reference identifies ownership. |

The worker-probe contents appear to be three old Depth_Color smoke probes for
workers 0/2/4, but the directory was outside the explicit deletion allowlist.
No other unknown output directory was found.

## Formal Baseline integrity

Before and after cleanup, all six formal runs were reopened and checked for:

- required checkpoint, config, history, metrics, NPZ, and confusion-matrix files;
- temporal-modality normalization statistics;
- `status=passed` and `device=cuda`;
- parameter counts matching the tracked formal summary;
- both checkpoints below 95 MiB;
- readable NPZ archives with logits `[N,40]`, embeddings `[N,128]`, class order
  0-39, and finite labels/logits/embeddings.

The complete formal Baseline tree SHA-256 was unchanged before and after:
`5E3A718B7DEC2D8348C798A9F9C85651A657D2571DC150B0E6A054EE0C2A78BC`.
No formal weight, NPZ, history, metric, configuration, or modality was deleted.

The current 14/4 fold, archived 12/6 fold, manifest, candidate ranking, and
selection report were not modified.

## Smoke script consolidation

- Removed the obsolete `scripts/smoke_test_task03.py` implementation that
  forced `num_workers=0` and overwrote the original smoke summary.
- Consolidated the optimized implementation at
  `scripts/smoke_test_task03.py`; the separate
  `scripts/smoke_test_task03_optimized.py` path is removed.
- The consolidated script reads current YAML worker/batch settings, uses
  `smoke_*` unique run IDs, defaults to `outputs/task03_smoke`, and writes
  `reports/task03_smoke_summary_latest.json`.
- Added `--output-root`, `--report-path`, and `--keep-outputs`.
- Successful session outputs are removed by default; failed runs are retained
  for diagnosis. Cleanup is restricted to unique run directories created by
  that invocation and cannot remove an existing output directory.
- Sample counts are copied from each actual `RESULT_JSON`; no old-fold sample
  count is hardcoded.

Only `python -m compileall src scripts` and
`python scripts/smoke_test_task03.py --help` were run. No smoke or formal
training was started.

## Retained code and reports

The reusable benchmark, formal runner/summarizer, new-fold selector/validator,
all requested small historical reports, and formal reports remain in place.
In particular, `scripts/benchmark_task03_input_pipeline.py` is retained.

The `.gitignore` still protects `outputs/`, `*.pt`, `*.pth`, `*.ckpt`, and
`*.npz`. `git ls-files outputs` returned no paths. Post-clean Git status showed
only the intended smoke-script consolidation, cleanup script, and two cleanup
reports; ignored output deletion creates no Git change.
