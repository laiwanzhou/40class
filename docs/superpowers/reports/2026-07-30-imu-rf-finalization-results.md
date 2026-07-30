# IMU compact Random Forest finalization results

## Outcome

`imu-rf-final-v1` passed the reproducibility gate, the full-data production package was published successfully, and the IMU modality is finalized.

```text
reproducibility_status = exact_match
imu_modality_status = finalized
```

No submission was generated and no branch was pushed, merged, rebased, or force-pushed.

## Reproducibility gate

The unique formal source was `trees_150_leaf4`, seed `20260725`, from `random_forest_compact_v1`. The fresh fold-0 run used the same 2,184 training and 573 validation samples, the same train-only median imputer, feature schema, class order, estimator parameters, and environment.

| Evidence | Result |
| --- | --- |
| Source run canonical SHA-256 | `ca3ef446914f27140c5342d36e9358f35a1128d334b10e87e6bb10c5b538f9fd` |
| Fresh run canonical SHA-256 | `b80ab90dddc5cc6ae70877eadc01e84e31ea4418bd43f4ed6c4bc131c5b7533f` |
| Source/fresh model SHA-256 | `1f4fd6efbaaa33915b5aa335acabec4a1c1ae51cfaca4ffd026aaa3adb22c9ef` |
| Prediction mismatches | 0 |
| Maximum probability difference | `1.1102230246251565e-16` |
| Source/fresh tree-state SHA-256 | `bb7e8f9af7f818f20eaf7d4d5bcc4fca91c3352c5bb80e7bee64a2dfc5d16cfc` |

Exact fold-0 metrics were accuracy `0.4240837696`, macro precision `0.3868249057`, macro recall `0.3620987015`, macro F1 `0.3444304400`, and weighted F1 `0.4033194078`. Zero-F1 labels were 8, 13, 18, 24, 25, 26, 27, and 35.

## Full-data production model

The production set is the exact union of fold-0 train and validation: 2,184 + 573 = 2,757 unique labeled samples, overlap 0. Its selected-ID SHA-256 is `a9bfc8bdea778b5dbc289cf1048477b235475086cc0bfbfca51eb6c1c08d66a0`; the label-vector SHA-256 is `8d773cf3b6ffde05dc5aeaffde6442960339519674582e95357ce0a0bc34a58f`.

The model uses the frozen `imu-rf-summary-v1` 2,310-feature schema, a production median imputer fit on all 2,757 raw feature rows, and one 150-tree `RandomForestClassifier` fit with seed `20260725`. The model SHA-256 is `9e47a3606ae5048a665e81599a0b47eb7da1429fd51e81d962ff88fbffa1336f`; the imputer SHA-256 is `cbdbb67094aa066ed29dfc701e258b7def766acbde8de9b54652b956a57c30f4`.

The package contains exactly nine files and is 8,031,457 bytes (7.659394 MiB), below the 8 MiB limit. Its canonical SHA-256 is `b969fbe0e67acb650e1065da3c164adbdcac86bd944dd41eb16204d7d1cc1ced`; the directory snapshot SHA-256 is `48a43f3f5ee764b895129e5be7d1e979529af3f1c9a065e2ed8184a31d6f050a`.

`package_manifest.json` records hashes and exact sizes for the other eight members, plus the total nine-file package size and a canonical hash over those immutable member records. It intentionally does not claim a SHA-256 of itself: embedding a file's own final cryptographic digest in that same file is circular and has no general fixed-point construction.

The recorded resubstitution accuracy (`0.9822270584`) and macro F1 (`0.9865085781`) are explicitly marked diagnostic-only, not generalization metrics, and not usable for model selection.

## Inference and fusion contract

The standard interface loads and validates the production package, reconstructs `imu-rf-summary-v1` features from Stage 2 NPZ files, applies the package imputer, and emits sample IDs, predictions, 40-class probabilities, class order, and restorable input positions in CSV and NPZ form. Inference forces deterministic single-thread tree reduction while restoring the model's frozen `n_jobs` setting afterward.

The fold-0 fresh model was replayed through the same prediction interface for all 573 validation samples. Sample IDs, predictions, and class order were exact; probabilities were within `1e-15` with maximum difference `1.1102230246251565e-16`. Two independent production-package reloads on a fixed 10-sample smoke set produced bit-exact probabilities and passed finite, row-sum, and argmax checks.

The fusion-only validation reference contains exactly five files, 573 samples, and 275,169 bytes. Its canonical SHA-256 is `08e821e0a398e7cf88a28e402b1b90a403bb128902571fe57156a8a48a261820`.

## Verification and protection

| Check | Result |
| --- | --- |
| RF tests | 70 passed |
| Stage 2 tests | 412 passed |
| Stage 1 regression | 68 passed, 1 privilege-dependent skip |
| Full repository | 550 passed, 1 privilege-dependent skip |
| All tracked Python `py_compile` | passed |
| Three CLIs from external CWD | passed |
| `git diff --check` | passed |
| `git fsck --full --strict` | exit 0; pre-existing dangling blobs only |
| Staging/backup residues | 0 |
| Largest tracked blob | 782,978 bytes |

All seven protected external directories retained their starting file counts, byte counts, canonical SHA-256 values, and relative-path-list SHA-256 values. Stable branch refs `IMU`, `IMU_rf`, `IMU_bak`, `IMU_rf_features`, and `IMU_test` were unchanged.
