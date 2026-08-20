# Thermal Train12 / User6-User7 Split Sidecar Audit

## Frozen Decision

Keep the existing `train12 / user6-user7-val` development split. This sidecar records Thermal-specific coverage without modifying `metadata/splits/train12_val2_user6_user7_development.json` (SHA-256 `c65c4826bfdd2d16796021abfccd6b1ece0ef7fb0ec786b19cc62cb60e5e57ca`).

| Population | Canonical | Thermal usable | Class coverage |
| --- | ---: | ---: | ---: |
| user6 | 203 | 195 | 34/40 |
| user7 | 185 | 182 | 33/40 |
| user6+user7 | 388 | 377 | 40/40 |
| remaining train12 | 2,039 | 1,922 | 40/40 |

Across all `14 choose 2 = 91` two-user validation pairs, only `user6+user7` gives full `0..39` Thermal-usable class coverage on both the combined validation side and the remaining train12 side.

## T1-B Metric Contract

- Select checkpoints on combined user6+user7 Macro-F1, with fixed labels `0..39`.
- Use Accuracy and then worst-user Accuracy as secondary metrics; preserve the existing lower-epoch tie break after those metrics.
- Report user6 and user7 Accuracy and Macro-F1 separately, always using labels `0..39` for Macro-F1.
- Do not interpret a class absent for one user as evidence that the model has zero capability on that class.
- The ten combined-validation classes with at most three usable trials are `0, 16, 18, 19, 25, 26, 27, 31, 35, 38`. Do not change architecture from one or two errors in these classes.
- Development metrics do not decide final retention. The shared train-14 OOF result remains authoritative.

No training or sealed/competition data access was performed for this sidecar.
