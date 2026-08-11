# Six-Modal Program Charter Independent Review

Reviewed branch head before this report:

```text
ad270374eb3028d2555ce28aa99c0410788029a2
```

Comparison base:

```text
61f4ce93aebce21c817443e43862d4bc290b39db
```

Review scope is intentionally limited to scientific validity, strict cross-subject evidence, multimodal compatibility, competition rules/model budget, and correct Phase 10 submission lifecycle. Production-service robustness, high availability, generalized defensive programming, monitoring, rollback, and unrelated maintainability concerns are out of scope.

## Verdict

```text
APPROVED_WITH_NONBLOCKING_NOTES
```

No blocking scientific or six-modal compatibility defect was found in the Phase 5.5 design.

## Scientific checks

### 1. Current Phase 4 strict-v3 protocol

PASS. The amendment does not change the running strict-v3 model, split, epoch-selection/refit protocol, equal-trial accumulation objective, scheduler horizon, or outer-validation access rule. The diff contains one formatting-only deletion of the isolated `Run:` label before Phase 4 Step 2 prose; it changes no command or scientific instruction.

### 2. Canonical population and cross-modal identity

PASS. The program contract fixes the 3,036-trial sparse union, canonical `sample_id`/`user_id`, one `0..39` class order/hash, separate `present` versus `usable`, and the same Phase 4 outer OOF ownership for all experts. Missing modalities do not change trial membership.

This is sufficient to prevent the most damaging late-fusion incompatibility: independently trained experts silently using different trial universes or class orders.

### 3. Capability-map A/B/C logic

PASS after refinement in commit `9a7d25dfe890991eb893eca45b04918aed4d0bd5`.

The first draft risked conflating a weak tested representation with a weak physical modality. The authoritative charter now separates:

```text
expected_native_potential
observed_strength
observed_model_or_representation
evidence_level
evaluation_scope
```

Thus the old 21-stat Radar TCN may remain a weak A-level observation about that representation while raw-point Radar still has a justified B/C-level native-potential hypothesis. Canonical train-14 A-level OOF evidence has priority for choosing the current model candidate, while historical mixed-split results remain tagged context.

The main X3D plan's Phase 5.5 Step 2 still contains the earlier shorter field list and sentence `A overrides ... C`. This is nonblocking because the same plan explicitly makes `2026-08-11-six-modal-program-charter.md` authoritative for Phase 5.5-10. When Phase 5.5 is implemented, use the charter's refined schema, not the shorthand list.

### 4. Natural complementarity versus artificial diversity

PASS. The plan does not force intentionally different architectures at the expense of sensor modeling. It requires each expert to preserve its native signal first and measures complementarity from canonical OOF unique-correct/error-disagreement/oracle evidence.

The first-family directions are scientifically coherent with the recovered evidence:

```text
IR        appearance / interaction -> X3D-S
Depth     geometry / depth motion -> compact geometry-aware spatial-temporal
Thermal   heat appearance / temporal -> Thermal-native spatial-temporal/video
Skeleton  body topology / motion -> TCN baseline + graph-temporal candidate
IMU       inertial dynamics -> compact RF primary
Radar     point-space / motion -> raw point encoder + masked temporal
```

### 5. ExpertEvidence compatibility boundary

PASS. Every expert must expose the same external trial identity, 40-class logits, availability, quality, provenance, fold lineage, and deployed bytes. Embeddings/engineered summaries remain optional.

This gives heterogeneous model families freedom internally without forcing common hidden-state dimensions or feature semantics.

### 6. Cross-fold embedding compatibility

PASS after amendment. The first registered fusion stack does not require raw heterogeneous embeddings or engineered-summary vectors. Phase 7 carries them only as optional references; Phase 8 anchor uses class evidence; Phase 9 first residual tokens use logits/log-probabilities, modality identity, availability, and label-free quality.

This removes the unjustified assumption that embeddings learned independently by different expert families and OOF refits share one coordinate system.

### 7. Safe-anchor definition versus quality fields

PASS after charter refinement in `9a7d25d...`.

The common interface carries native quality from the beginning, but the first safe anchor remains the already-frozen Phase 8 global-temperature/global-weight masked probability mixture. Quality does not silently change the anchor weighting rule. Phase 9 may use quality only through its separately evaluated residual gate/model.

### 8. Fusion leakage

PASS. Phase 7 retains the nested base-evidence design: for each outer fusion fold, every base expert is trained without the outer-validation users; fusion-training evidence is generated by inner user-OOF; outer-validation evidence is produced by models finalized only on outer-train users.

Phase 8 and Phase 9 consume those same nested folds for reported fusion metrics. Global train-14 OOF is reserved for final refitting/diagnostics and cannot substitute for unbiased level-2 validation.

### 9. Heldout-4 lifecycle

PASS. Heldout evidence may be generated structurally before Phase 10 but remains label-free and quarantined. Expert/fusion architecture decisions are completed from train-14 evidence. Phase 10 then loads the sealed heldout labels exactly once for final evaluation and forbids scientific redesign from that result.

### 10. All-18 production refit and first Kaggle submission

PASS. Task 14 now makes the intended lifecycle explicit:

```text
heldout-4 evaluation
-> freeze architecture
-> all-18 frozen-recipe OOF/refit
-> complete package gate
-> read competition test
-> frozen inference
-> audit CSV
-> user manually submits once
-> record Public-LB score as external diagnostic
```

This matches the intended first external check without sacrificing the sealed local heldout before architecture freeze.

### 11. Model budget

PASS. The `<95,000,000` serialized-byte complete-package limit remains the only hard program budget. Per-modality ranges in the charter are explicitly planning envelopes rather than hard quotas. This prevents an early arbitrary allocation from blocking a high-value expert while still discouraging another 60+ MB visual expert without measured benefit per byte.

## Nonblocking notes for Phase 6 detailed planning

1. The program charter deliberately does not freeze exact ST-GCN variant, Depth geometry channels, Thermal video backbone, or Radar temporal width. Before each modality's first canonical run, create a modality-specific subplan that freezes that experiment's architecture and selection rule while inheriting Phase 5.5 unchanged.
2. The main plan's Phase 5.5 capability-map shorthand should be interpreted through the refined authoritative charter fields noted above. This is a documentation-normalization item, not a scientific blocker.
3. Do not use the capability map to pre-route actions. Its value is to diagnose whether a modality/model learns the information it should plausibly contain and to prioritize the next controlled experiment.
4. If later representation-alignment evidence supports embedding-level fusion, register that as a separate ablation rather than silently extending the first residual mixer.

## Review conclusion

The revised plan now has a clear hierarchy:

```text
Phase 0-5: prove/register IR
Phase 5.5: freeze six-modal compatibility and modality-role charter
Phase 6: develop/freeze the remaining experts under that charter
Phase 7: align leakage-free sparse evidence
Phase 8: safe probability anchor
Phase 9: optional compatible residual correction
Phase 10: sealed heldout -> all-18 frozen refit -> first competition submission
```

The structure is suitable as the program-level roadmap for later Codex work. The detailed model choice for each remaining modality should be made in its own controlled Phase 6 subplan, but those subplans should not reopen the Phase 5.5 compatibility rules.