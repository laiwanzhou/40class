# Six-Modal HAR Program Charter

> **Status:** program-level design authority for Phases 5.5-10. This document does not change the currently running Phase 4 X3D strict-v3 experiment. It becomes binding only after Phase 5 closes and the six-modal program moves to dedicated worktrees/branches.

**Goal:** Preserve scientific validity and cross-modal compatibility while allowing IR, Depth_Color, Thermal, IMU, Skeleton, and Radar experts to evolve independently under the Small Model Track aggregate size limit.

**Research scope:** This is a competition research plan, not a production service. Reviews must focus on scientific validity, cross-subject evaluation, modality information preservation, fusion compatibility, competition compliance, and deployment-byte feasibility. Do not add production-grade fault tolerance, service availability, generalized defensive programming, or maintainability gates unless they are directly required for correct scientific evidence or a valid competition submission.

## 1. Program-level invariants

The following are frozen above all modality-specific model choices:

1. Canonical population is the 3,036-trial sparse union in `metadata/manifest.csv`; never reduce the program to the 2,748 all-six intersection.
2. Trial identity is canonical `sample_id`; user identity is canonical `user_id`; class order is fixed to `0..39` and must share one `class_map_hash`.
3. The outer scientific protocol remains 14 train users plus 4 sealed held-out users. Iterative model-family and hyperparameter development for every later expert uses the frozen shared split `metadata/splits/train12_val2_development.json`: 12 development-train users and validation users exactly `user21,user22`. The former X3D fold0 split is retired for new tuning. Once a candidate recipe is frozen, formal train-14 evidence must still reuse the exact persisted Phase 4 `metadata/splits/train14_oof_3fold.json` assignment and its hash; the shared 12/2 score is development evidence, not a substitute for OOF.
4. `present`, `usable`, and label-free `quality` are distinct. Fusion availability is exactly `usable`; a missing or unusable modality never deletes the canonical trial row.
5. Every retained expert must produce leakage-free train-14 OOF evidence and a structurally label-free held-out evidence archive before fusion development.
6. The complete inference package must remain below the internal ceiling of `95,000,000` serialized bytes, counting each inference-time learned artifact exactly once.
7. Held-out labels are unavailable for model family, architecture, hyperparameter, epoch, calibration, fusion, or expert-membership decisions. Competition test data is unavailable until the Phase 10 production gate has frozen all experiment choices.
8. First-generation fusion compatibility is defined by 40-class logits/probabilities, availability, and label-free quality. Embeddings and engineered summaries may be stored for analysis but are not required or assumed to be cross-fold/cross-expert aligned.

## 2. Modality roles and first model-family directions

These roles describe the native information each expert should preserve. They are not fixed final architectures and they do not require artificial diversity: each expert should first become a strong model of its own sensor signal, then complementarity is measured from OOF errors.

| Modality | Program role | First model-family direction | Native evidence to preserve |
|---|---|---|---|
| IR | high-capacity appearance / interaction expert | X3D-S adaptive multi-clip | person appearance, local object interaction, fine temporal appearance |
| Depth_Color | geometry / depth-motion expert | compact geometry-aware spatial encoder + temporal model | depth ordering, distance, silhouette, relative motion, hand/object geometry |
| Thermal | thermal appearance / temporal expert | Thermal-native lightweight spatial-temporal/video model with independent temporal sampling | body heat silhouette, temperature distribution, temporal heat appearance |
| Skeleton | body-topology / pose-motion expert | lightweight graph-temporal model; compare against accepted root-centered + velocity TCN | joint topology, bones, pose, velocity, whole-body motion |
| IMU | inertial-dynamics expert | compact RF primary; small neural temporal candidate only if justified | bilateral wrist/ankle/waist acceleration, angular motion, rhythm, intensity |
| Radar | point-space / motion expert | raw point-set frame encoder (PointNet-style) + masked temporal model | point geometry, range, radial velocity, spatial dynamics |

Model families may change after controlled shared-12/2 development experiments followed by frozen-recipe train-14 OOF evaluation, but external evidence identity, class order, OOF lineage, availability semantics, and byte accounting may not change.

## 3. 40-action capability-map evidence hierarchy

Maintain a `40 actions x 6 modalities` capability map throughout Phase 6. Each action/modality cell must keep **native-potential priors separate from observed model performance**. Record at least:

```text
expected_native_potential
observed_strength
observed_model_or_representation
evidence_level
evaluation_scope
reason
experimental_status
source_reference
```

Evidence levels are:

- **A — internal CUHK-X/CUHK-S experiment:** direct evidence from this repository. `evaluation_scope` must state whether it is canonical train-14 OOF, another cross-user split, or a diagnostic subset.
- **B — official CUHK-X evidence:** official dataset/benchmark/sensor documentation. It establishes that a modality contains useful signal but must not be treated as numerically comparable when the official split differs from our strict cross-subject protocol.
- **C — sensor/research prior:** physical sensing properties and relevant external HAR research. This is a hypothesis to be tested, not a fusion weight.

Canonical train-14 A-level OOF evidence has the highest priority for selecting the **current model candidate**. Historical A-level evidence remains valuable but is tagged with its original split and cannot be numerically mixed with canonical OOF.

A weak A-level result is evidence about the tested representation/model under its stated scope, not automatically proof that the physical modality has low native potential. For example, an underfitting 21-stat Radar TCN remains evidence that the 21-stat representation failed; it does not erase a C/B-level reason to test raw point-set Radar. Therefore do not collapse `observed_strength` and `expected_native_potential` into one value.

Every retained Phase 6 expert updates its column with new canonical A-level per-class OOF evidence.

The capability map is a **design and diagnosis artifact only**. It must never directly hard-code class-specific fusion weights, routing rules, or labels. Fusion weights are learned only from leakage-free OOF evidence.

## 4. Canonical ExpertEvidence contract

Every retained expert, regardless of internal architecture, must expose or serialize the same external evidence fields:

```text
role
expert_id
sample_ids
user_ids
logits [N,40]
availability [N]
quality
quality_mask
fusion_quality_score [N,1]
class_map_hash
model_sha256
config_sha256
deployed_weight_bytes
preprocessing_dependencies
fold/evidence lineage
```

`labels` are required only for explicitly labeled `oof_train14` evidence and forbidden for `heldout` and `competition_test` evidence. `embeddings`, `engineered_summary`, and other diagnostics are optional.

First-generation safe anchor and first-generation residual fusion must be able to operate without embeddings. This avoids assuming that embeddings from heterogeneous experts or independently trained OOF folds share a coordinate system.

## 5. Shared OOF lineage

All six experts use the same frozen train-14 outer-fold user ownership. A model may have modality-specific preprocessing and a model-specific inner selection procedure, but a trial's formal OOF prediction must be produced by a model for which that trial's user influenced neither preprocessing fitting, weight fitting, epoch/checkpoint selection, nor hyperparameter choice.

For neural experts that require epoch selection, use an inner user split contained entirely inside the outer-train users and refit on all outer-train users for the selected epoch before touching the outer-validation users. For non-iterative experts such as the compact IMU RF, freeze the deterministic fit/feature policy and regenerate predictions on the same outer folds.

## 6. Missing-modality and sparse-union semantics

The canonical row always exists even if one or more modalities are absent or unusable. Each expert independently emits evidence only for usable rows; Phase 7 outer-aligns those sparse archives to the 3,036-row universe.

Fusion rules:

- unavailable experts contribute no evidence and are masked;
- exactly one usable expert must recover that expert's calibrated probability exactly;
- natural missingness patterns are evaluated according to observed support;
- no model may fabricate a prediction merely to fill a missing expert slot.

These are scientific fusion semantics, not general-purpose production robustness requirements.

## 7. Fusion compatibility boundary

The first safe anchor is deliberately low-assumption:

```text
per-expert logits/probabilities
+ availability
        -> scalar calibration
        -> masked non-negative available-expert mixture (safe anchor)
```

Label-free native `quality` and normalized `fusion_quality_score` are carried in the common evidence interface from the start, but the first registered safe anchor does **not** need to use them to modulate its mixture weights. This keeps Phase 8 consistent with the frozen global-weight masked mixture. Quality becomes available to Phase 9's explicitly evaluated residual gate/model without changing the anchor definition.

Phase 9 may add a tiny residual correction, but its first registered version should consume calibrated/log-probability evidence, availability, modality ID, and label-free quality rather than depending on raw heterogeneous embeddings. An embedding/engineered-summary fusion study is a separate later ablation only after an explicit cross-fold representation-alignment analysis.

The safe anchor remains a complete model even if the residual mixer is rejected.

## 8. Model-budget ledger

The only hard budget is the complete-package ceiling `<95,000,000` serialized bytes. Use the following as planning envelopes, not mandatory per-modality quotas:

```text
shared YOLO / visual localization     ~6 MB currently
IR X3D-S                              ~14-16 MB currently
Skeleton                              ~5-12 MB target envelope
IMU RF                                ~6 MB current candidate
Radar                                 ~3-8 MB target envelope
Depth                                 ~8-15 MB target envelope
Thermal                               ~8-15 MB target envelope
calibration + optional residual       ~1-3 MB target envelope
```

Any candidate may exceed its envelope if measured train-14 OOF evidence justifies the benefit per byte and the complete package remains below the hard ceiling. A large expert is not rejected merely for size, but it must earn the budget against alternatives.

## 9. Phase 6 model-family priorities

After Phase 5 freezes IR evidence:

1. **IMU:** canonically rerun the accepted compact RF first. Preserve it as the primary candidate unless a small neural candidate offers a clear standalone or complementary gain per byte.
2. **Skeleton:** reproduce the accepted root-centered + velocity TCN on the canonical folds, then compare a lightweight graph-temporal candidate that explicitly models joint topology and temporal motion.
3. **Radar:** treat the old 21-stat TCN as a failed representation baseline. Compare it with raw variable point sets encoded per frame by a shared PointNet-style MLP/pooling encoder followed by a small masked temporal model.
4. **Depth:** prioritize native geometry and temporal depth change instead of copying another high-capacity IR-like appearance backbone. Historical large-backbone experiments are evidence that capacity alone is not the main bottleneck.
5. **Thermal:** give Thermal an independent temporal sampler and a real spatial-temporal/video candidate. Do not infer weakness from the old shallow MobileNet+mean result alone.

Candidates are judged on canonical cross-user standalone performance, per-class behavior, worst-user behavior, complementary unique-correct/oracle evidence, and deployed-byte cost. Do not deliberately weaken a modality to make it look different from IR; complementarity should emerge from the sensor's native information.

## 10. Program-wide prohibitions

Do not:

1. redefine class order or class IDs inside an expert;
2. create expert-specific outer OOF user folds after the Phase 4 assignment is frozen;
3. delete a canonical trial because one modality is missing;
4. use held-out-4 labels to select any expert or fusion choice;
5. use competition-test/Public-LB results to retroactively choose architecture in the frozen Phase 10 system;
6. assume equal-dimensional embeddings are aligned across expert types or independently trained folds;
7. let an expert bypass `ExpertEvidence` because its native model API is different;
8. exceed the complete byte budget without explicitly reallocating and re-auditing the package;
9. turn A/B/C capability priors into hand-coded class labels or fusion weights.

## 11. Phase 10 lifecycle and first competition submission

The scientific lifecycle is:

```text
Train-14 model development
  -> six expert OOF evidence
  -> nested fusion evidence
  -> anchor / optional residual architecture freeze
  -> evaluate sealed heldout-4 exactly once
  -> freeze evaluation architecture
  -> rebuild OOF/fusion calibration on all 18 users as required by the frozen recipe
  -> train all six retained experts on all usable Train-18 data
  -> complete-package byte audit
  -> competition-test inference
  -> submission CSV
  -> user manually submits once for the first external leaderboard check
```

The first Public-LB score is an external diagnostic of the frozen system, not permission to rewrite the completed Phase 10 evidence. Any later leaderboard-driven iteration must be declared as a new experiment generation with its own audit trail.

## 12. Review scope

Program reviews answer the following questions:

- Does each model direction preserve the information native to its sensor?
- Are all reported generalization results genuinely user-held-out?
- Are modality sample identities, class order, folds, and evidence roles compatible?
- Is the capability-map evidence clearly separated into A/B/C and evaluation scopes rather than mixing priors with results?
- Does a weak model result remain correctly attributed to its tested representation rather than automatically declaring the entire sensor weak?
- Can every retained expert enter the same sparse evidence registry without changing trial semantics?
- Does fusion use leakage-free evidence and handle natural modality availability correctly?
- Does the complete inference model satisfy the competition's aggregate model-size interpretation?
- Is the final test CSV produced only after the scientific architecture is frozen?

Do **not** fail this research plan for missing production-service concerns such as high-availability behavior, generalized malformed-input handling, deployment monitoring, operational rollback, or broad defensive coding when those concerns do not affect experiment validity, competition compliance, or correct test inference.
