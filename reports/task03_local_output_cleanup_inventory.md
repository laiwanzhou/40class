# Task 03 local output cleanup inventory

Inventory captured at `2026-07-16 11:26:50 +08:00`, before deletion.

The scan inspected each run's `config.yaml` and `metrics.json`. None of the
output directories contains a fold snapshot, and their configs reference the
mutable `metadata/splits/fold_0.json` path. Old-fold identification therefore
uses the run date plus sample counts matching the archived 12/6 reports, not
the current contents of that path.

| Directory | Bytes | MiB | Modalities | Run type | Fold evidence | Decision | Reason |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `task03` | 31,316,055 | 29.865 | six modalities | initial full-data 2-epoch validation | old 12/6 sample counts; 2026-07-15 | delete | Superseded by optimized validation and formal Baseline. |
| `task03_baseline_fold0` | 31,341,399 | 29.889 | six modalities | formal 30/40-epoch Baseline | old 12/6 sample counts; formal report | **keep** | Unique complete formal 12/6 artifacts; explicitly protected. |
| `task03_optimized_smoke` | 58,352,864 | 55.650 | six modalities, two runs each | optimized one-epoch smoke | old 12/6 sample counts; 2026-07-15 | delete | Temporary smoke checkpoints and predictions; small Git report is retained. |
| `task03_optimized_two_epoch` | 56,523,773 | 53.905 | six modalities, with repeated visual runs | optimized full-data 2-epoch validation | old 12/6 sample counts; 2026-07-15 | delete | Superseded by formal Baseline; small Git reports are retained. |
| `task03_smoke` | 32,041,816 | 30.557 | six modalities, with repeated early runs | initial one-epoch smoke | old 12/6 sample counts; 2026-07-15 | delete | Obsolete `num_workers=0` smoke artifacts; historical summary remains in Git. |
| `task03_worker_probe` | 25,098,175 | 23.935 | Depth_Color only | three one-epoch probes named `probe_w0/w2/w4` | old 12/6 sample counts; 2026-07-15 | `manual_review_required` | Contents look like worker probes, but no tracked script or report references this output root; it is not auto-deleted. |

Total before cleanup: **234,674,082 bytes (223.803 MiB)**.

The explicit deletion allowlist contains only `task03`, `task03_smoke`,
`task03_optimized_smoke`, and `task03_optimized_two_epoch`.
`task03_baseline_fold0` is permanently protected by the cleanup script.

## Formal Baseline pre-clean validation

The six formal runs passed checks for required files, `status=passed`, CUDA,
parameter counts matching the formal summary, both checkpoints below 95 MiB,
readable NPZ archives, logits `[N,40]`, embeddings `[N,128]`, class order
0-39, and finite labels/logits/embeddings. The three temporal modalities also
contain `normalization_stats.json`.

Pre-clean formal Baseline tree SHA-256:
`5E3A718B7DEC2D8348C798A9F9C85651A657D2571DC150B0E6A054EE0C2A78BC`.
