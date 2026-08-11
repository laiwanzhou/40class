# Six-Modal Sparse-Evidence Fusion Design

Approved: 2026-08-11 (Asia/Shanghai)

## Objective

Build one deployable CUHK-X Small Model Track classifier from six heterogeneous
modalities: Depth_Color, IMU, IR, Radar, Skeleton, and Thermal. The system must
preserve a high-quality main path for common modality combinations while still
attempting inference for every non-empty raw-modality subset. Rare combinations
may lose accuracy. A trial may fail explicitly only when none of its supplied
modalities can produce usable expert evidence.

X3D-S is the IR expert. It is not the whole multimodal model.

## Frozen Architecture

```text
six heterogeneous specialist experts
        -> sparse OOF evidence registry
        -> calibrated available-expert probability anchor
        -> tiny zero-initialized residual set mixer
        -> final trial probability
```

The system has four layers:

1. Modality-native preprocessing.
2. Heterogeneous specialist experts.
3. A safe calibrated probability anchor.
4. An optional learned residual correction.

Two structural invariants are mandatory:

- Without the residual mixer, the system remains a complete deployable model.
- Without any particular modality, the system remains usable when at least one
  supplied modality produces usable expert evidence.

## Canonical Trial Registry

The authoritative fusion population is the 3,036-row union in
`metadata/manifest.csv`, never the six-modality intersection.

The fixed split is `metadata/splits/fold_0.json`:

- 14 training users, 2,427 union trials;
- 4 held-out users, 609 union trials;
- all held-out users are final evaluation only.

The modality bit order is fixed as:

```text
Depth_Color, IMU, IR, Radar, Skeleton, Thermal
```

Directory-presence patterns are:

| Pattern | Train-14 | Held-out-4 | Meaning |
|---|---:|---:|---|
| `111111` | 2,183 | 565 | all six directories present |
| `111110` | 128 | 14 | Thermal absent |
| `000001` | 85 | 18 | Thermal only |
| `101111` | 18 | 3 | IMU absent |
| `111011` | 8 | 5 | Radar absent |
| `101011` | 4 | 0 | IMU and Radar absent |
| `101110` | 0 | 3 | IMU and Thermal absent |
| `001001` | 1 | 1 | IR and Thermal only |

These bits describe directory presence, not usability. Every expert must report
three separate concepts:

- `present`: the raw modality directory/input exists;
- `usable`: preprocessing and inference produced valid evidence;
- `quality`: label-free inference-time reliability features.

For example, a Radar directory with no valid detections is present but may be
unusable. It must not silently become a normal available expert.

## Preprocessing Layer

### Visual modalities

Pose-guided localization is a reusable visual asset. YOLO11n-pose is the
currently verified locator implementation for the IR path, not a universal
locator for every visual modality.

The first verified route is:

```text
raw IR -> YOLO11n-pose -> ROI coordinates
                         -> IR person-context crop
                         -> aligned Depth crop when registration is proven
```

IR person-context is the primary X3D view. It removes most background while
retaining nearby interaction objects. Interaction/relation ROIs remain optional
future evidence and must be evaluated separately.

Thermal may reuse IR coordinates only after an audit proves common field of
view, resolution mapping, camera geometry, and temporal registration. Otherwise
Thermal uses its own localization strategy. When IR is absent, a visual expert
may attempt a modality-native locator and then a documented full-frame/context
fallback with reduced quality. Sensor-only trials bypass YOLO entirely.

### Sensor modalities

IMU, Skeleton, and Radar keep modality-native preprocessing. No model is forced
to manufacture an image, temporal grid, or neural embedding merely to match
another expert.

## Specialist Layer

The first candidate portfolio is:

| Modality | Candidate | Status |
|---|---|---|
| IR | X3D-S adaptive multi-clip | Current implementation route |
| Depth_Color | compact spatial-temporal/geometric expert | Candidate to select |
| Thermal | compact visual-temporal expert | Candidate to select |
| IMU | compact Random Forest | Strong existing candidate |
| Skeleton | root-centered velocity sequence expert | Strong existing candidate |
| Radar | raw-point frame encoder plus small TCN | Unverified replacement candidate |

Experts are selected on more than standalone Accuracy. Every formal expert
report must include Accuracy, Macro-F1, per-class recall, worst-user Accuracy,
pairwise error agreement, unique-correct count, oracle-pair Accuracy, class-wise
rescues, deployed bytes, and latency.

The existing 21-stat Radar TCN is a failed baseline. PointNet-style frame
encoding is a hypothesis that still requires a controlled experiment.

An aligned IR+Depth co-expert is deferred. It may be proposed only after sparse
OOF evidence shows material error complementarity and the complete weight budget
still passes.

## Evidence Contracts

### Neural model output

Keep the existing `ExpertOutput` contract unchanged:

```text
main_logits
embedding
quality
quality_mask
availability
```

### Serialized fusion evidence

Introduce a separate `ExpertEvidence` archive contract:

```text
required:
  expert_id
  sample_ids
  user_ids
  labels
  logits [N,40]
  availability [N]
  quality [N,Q]
  quality_mask [N,Q]
  class_map_hash
  model_sha256
  config_sha256
  deployed_weight_bytes
  preprocessing_dependencies

optional:
  embeddings [N,D]
  engineered_summary [N,S]
  diagnostics
```

IMU Random Forest evidence can omit embeddings and provide engineered summaries.
Modality-specific adapters convert each expert's available fields to a 64- or
96-dimensional fusion token.

### Two alignment operations

`strict_alignment` remains for matched experiments and paired scientific
comparisons. It requires identical sample sets and class maps.

`outer_evidence_alignment` is used for multimodal fusion. It left-joins each
sparse expert archive to the canonical 3,036 sample IDs, fills neutral tensor
values, and sets `availability=False` for missing rows. It rejects duplicate or
unknown sample IDs, duplicate expert rows, label mismatch, user mismatch, and
class-map mismatch.

## Leakage-Free Evidence Generation

All fusion learning uses predictions generated inside the 14 training users.
Use one shared deterministic three-fold `StratifiedGroupKFold` assignment with
`shuffle=True` and `random_state=20260715`, grouped by `user_id`, for all experts.
Persist the assignment and require all 40 classes in every OOF validation fold.
Each fold trains preprocessing statistics and the expert without the fold's
users, then predicts only the held-out OOF users for rows where that modality is
usable.

The resulting sparse OOF registry contains one canonical row per training-union
trial and an availability mask per expert. A prediction is never generated by a
model trained on the same user's samples.

The four held-out users are predicted only by final experts trained on all 14
training users. Held-out predictions, labels, qualities, and missingness patterns
must not fit temperatures, weights, thresholds, gates, architectures, or residual
hyperparameters.

## Safe Probability Anchor

For each available expert `m`:

```text
p_m = softmax(z_m / T_m)
```

Temperatures and non-negative scalar expert weights are fitted from training-user
OOF evidence only. For a trial:

```text
p_base = sum_m(a_m * w_m * p_m) / sum_m(a_m * w_m)
```

where `a_m` is expert usability, not directory presence. Weights are globally
normalized for identifiability and renormalized over the available subset per
trial. With one usable expert, the anchor exactly returns that expert's calibrated
probability. With zero usable experts, inference fails explicitly with recorded
reasons from every supplied modality.

The anchor is the first six-modal scientific baseline and the permanent fallback
for rare combinations.

## Residual Set Mixer

The optional mixer consumes at most six modality tokens, the availability mask,
and anchor logits. It uses a small DeepSets/gated-MLP or one-to-two-layer masked
self-attention module and emits `delta_logits=[B,40]`.

The final layer is zero-initialized. Define:

```text
z_anchor = log(clamp(p_base, epsilon, 1))
z_final  = z_anchor + g(A,Q) * lambda * delta_logits
p_final  = softmax(z_final)
```

`lambda` controls global residual strength. `g(A,Q)` is a deterministic first-run
support/quality gate in `[0,1]` and answers whether this trial should use learned
correction. Their responsibilities may not be merged.

The first residual-training distribution is deliberately narrow:

- natural train-14 patterns with at least two usable experts and at least 16
  supporting trials;
- occasional single-modality dropout from all-six training rows for stability;
- synthetic-dropout loss receives lower weight than natural-pattern loss;
- single-expert and unsupported rare combinations always use `g=0`;
- no exhaustive generation of arbitrary modality subsets.

The residual candidate is retained only if it improves over the anchor under
user-grouped fusion cross-validation and does not materially regress worst-user
Accuracy, Macro-F1, or common missing-pattern behavior. Otherwise the anchor is
the final system.

## Inference Behavior

For one raw trial:

1. Inventory supplied modalities.
2. Run each supplied modality's preprocessing and expert independently.
3. Mark each expert present/usable and record quality or failure reason.
4. Build calibrated probabilities for usable experts.
5. Compute the anchor over the usable subset.
6. Apply the residual only when `g(A,Q)>0`.
7. Return final probability, anchor probability, expert availability, quality,
   provenance, and diagnostics.

The main path is optimized for all-six and frequent missing patterns. Extreme
subsets are availability features, not equal-accuracy targets. A non-empty raw
subset may still fail when every supplied modality is unusable; the failure must
be explicit and auditable.

## Size and Rule Gate

The internal ceiling remains 95,000,000 serialized bytes for the exact complete
inference bundle. Count each deployed artifact once, including YOLO, all retained
experts, IMU RF and imputer, calibration values, token adapters, residual mixer,
and learned preprocessing. Teacher-only models count as provenance but may be
excluded only when absent from the inference graph.

The historical 65.80 MiB result applies to one specific ResNet18 object-interaction
bundle and demonstrates poor benefit per byte for that candidate; it is not a
general prohibition on ResNet18.

## Acceptance Criteria

- Canonical union membership and 14/4 user split are immutable.
- Every OOF prediction is user-held-out.
- Strict and outer alignment are separate tested operations.
- The anchor supports every non-empty usable expert subset.
- Single-expert inference exactly recovers calibrated expert probability.
- Zero-initialized residual exactly recovers the anchor.
- Rare unsupported patterns bypass learned correction.
- Held-out users never fit fusion choices.
- Anchor and residual are compared at trial, user, class, and missing-pattern level.
- The exact deployable package remains below 95,000,000 bytes.
