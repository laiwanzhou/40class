# Six-Modal Program Charter — Subagent Review Brief

## Review target

Repository: `laiwanzhou/40class`

Base before program-charter amendment:

```text
61f4ce93aebce21c817443e43862d4bc290b39db
```

Review the current head of branch:

```text
x3d-s-adaptive-multiclip
```

Primary files:

```text
docs/superpowers/plans/2026-08-10-x3d-s-adaptive-multiclip.md
docs/superpowers/plans/2026-08-11-six-modal-program-charter.md
```

## Purpose of the change

The existing X3D plan already contained a Phase 6-10 multimodal roadmap, but the modality-specific directions and cross-expert compatibility constraints were distributed across later tasks. The amendment inserts Phase 5.5 as a program-level freeze before any non-IR expert is developed.

The program charter must allow each modality model to evolve internally while preventing incompatibility in sample identity, class order, OOF lineage, sparse missingness, evidence schema, fusion features, and aggregate model-size accounting.

## Explicit review scope

This is a **Kaggle competition research plan**, not a production service or deployable enterprise application.

Review only issues that can materially affect:

1. scientific validity;
2. strict cross-subject generalization evidence;
3. leakage between train-14, sealed heldout-4, and competition test;
4. preservation of modality-native information;
5. compatibility of six experts at the evidence/fusion boundary;
6. natural missing-modality semantics;
7. statistical validity of nested fusion evaluation;
8. the `<95,000,000` serialized-byte internal package limit / competition compliance;
9. correctness of the final Phase 10 competition-test CSV lifecycle.

**Do not report production-engineering robustness issues** unless they directly change scientific results or submission correctness. In particular, do not request high availability, operational monitoring, rollback systems, generalized malformed-input handling, broad exception recovery, service-level fault tolerance, production logging frameworks, or maintainability refactors merely because they would be useful in a deployed product.

## Scientific design decisions that are intentional

- Canonical population remains the 3,036-trial sparse union; the all-six intersection is not the training universe.
- `present`, `usable`, and label-free `quality` are separate; only `usable` controls expert availability.
- All retained experts reuse the exact frozen train-14 outer OOF assignment.
- The 40x6 action capability map is a design/diagnosis artifact, not a source of hard-coded fusion weights.
- Capability-map native potential and observed model performance must remain separate; a failed representation does not automatically prove the sensor itself is weak.
- First-generation fusion must be possible using only 40-class logits/probabilities, availability, and label-free quality metadata. Raw heterogeneous embeddings are deliberately excluded from the first registered residual mixer because independently trained fold embeddings are not assumed aligned.
- The Phase 8 safe anchor remains the frozen global-weight masked probability mixture; quality is carried in the interface but need not modulate the first anchor.
- Phase 9 may use quality in the residual gate/model.
- Phase 10 evaluates the sealed heldout-4 once, freezes architecture, refits the frozen recipe using all 18 labeled users, then and only then reads competition test data.
- The first Kaggle/Public-LB result is an external diagnostic of the frozen Phase 10 v1 system. Later leaderboard-driven work must be a separately named experiment generation.

## Modality-family starting points to review

```text
IR        X3D-S adaptive multi-clip; appearance/object-interaction role
Depth     compact geometry-aware spatial + temporal; depth-geometry role
Thermal   Thermal-native lightweight spatial-temporal/video; independent sampler
Skeleton  accepted TCN baseline + lightweight graph-temporal candidate
IMU       compact RF primary; tiny neural candidate only if justified
Radar     raw PointNet-style frame encoder + masked temporal model
```

Do not reject these merely because alternative model families exist. Report an issue only if the direction contradicts available dataset evidence, destroys important native information, creates a fusion incompatibility, or makes the six-modal package scientifically infeasible.

## Questions for the reviewer

Return findings classified as `BLOCKING`, `IMPORTANT`, or `MINOR` and answer:

1. Does Phase 5.5 sufficiently freeze the external compatibility boundary before the five remaining experts are developed?
2. Is there any remaining route by which one expert can silently change sample IDs, class order, outer-fold ownership, or missingness semantics?
3. Does the capability-map A/B/C design correctly separate prior knowledge from measured model behavior?
4. Is any capability prior accidentally allowed to become a hand-coded class routing/fusion rule?
5. Does the first-generation fusion boundary avoid unjustified cross-fold embedding alignment assumptions?
6. Is Phase 7 nested evidence sufficient to prevent level-2 fusion leakage?
7. Does the Phase 8 safe anchor remain a valid standalone fallback for any non-empty expert subset?
8. Does Phase 9 compare the residual against the same nested outer folds without heldout leakage?
9. Is the heldout-4 -> architecture freeze -> all-18 refit -> competition-test CSV lifecycle statistically coherent?
10. Is the model-budget ledger compatible with the known IR+YOLO and IMU sizes without prematurely hard-coding per-expert quotas?
11. Did the amendment unintentionally change the currently running Phase 4 strict-v3 scientific protocol? Ignore formatting-only changes.

End with one of:

```text
APPROVED
APPROVED_WITH_NONBLOCKING_NOTES
CHANGES_REQUIRED
```

For every `CHANGES_REQUIRED` finding, quote the exact conflicting plan clauses and propose the smallest scientific correction. Do not broaden the review into production robustness.