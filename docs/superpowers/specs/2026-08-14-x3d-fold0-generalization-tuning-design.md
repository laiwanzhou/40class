# X3D Fold 0 Generalization Tuning Design

Date: 2026-08-14

Status: Approved design; implementation plan pending

Branch: `test/x3d-fold0-generalization`

## Objective

Use the frozen Phase 4 fold 0 split as an explicitly development-only validation
fold and target all three of the following trial-level metrics:

- Accuracy >= 0.6300
- Macro-F1 >= 0.5200
- worst-user Accuracy >= 0.5338

This is a competition-oriented tuning target, not a promise that the target is
reachable under the fixed data and compute budget.

## Scientific Status

Fold 0 is no longer an unbiased OOF evaluation population after this work begins.
Every checkpoint and metric produced by this branch is development evidence only.
It must not replace, overwrite, or be merged into:

- the canonical seed-20260715 Phase 4 OOF archive;
- the registered `ir_x3d_s_k400_pure` Phase 5 ExpertEvidence;
- the frozen Phase 4 retention decision;
- any heldout-4 or competition-test evidence.

The canonical baseline for comparison remains fold 0 Accuracy 0.571250,
Macro-F1 0.486918, and worst-user Accuracy 0.533835.

## Diagnosis Being Tested

The first experiment targets the strongest observed failure mechanism:

1. The complete X3D backbone currently unfreezes after epoch 2.
2. The train-to-validation gap expands rapidly after that transition.
3. Deterministic fold 0 train-eval Accuracy reaches 0.952632 while formal
   outer-validation Accuracy remains 0.571250.
4. The training population contains few independent user domains, making
   subject appearance, clothing, motion style, and retained scene context easy
   shortcuts for a fully trainable video backbone.

Adaptive multi-clip coverage may add capacity for long trials, but existing
duration-bucket results do not show a monotonic penalty for long trials. Clip
count reduction is therefore a second-stage ablation rather than the first
change.

## Invariants

The following remain fixed across all candidates:

- fold 0 outer-train and outer-validation user ownership;
- sample IDs, labels, class order, and usable-IR population;
- historical pose-guided person-context ROI assets;
- 13-frame local clips and deterministic validation sampling;
- trial-level mean-probability aggregation;
- equal-trial gradient accumulation;
- frozen backbone BatchNorm running statistics;
- no heldout-4 or competition-test access.

Every run uses a new run ID and output directory. Existing checkpoints are never
automatically deleted, including after a metric regression.

## Candidate Sequence

### Stage A0: Reproducibility Guard

Before changing training behavior, record hashes for the canonical fold 0
assignment, checkpoint, formal predictions, resolved configuration, and Phase 5
evidence. Verify that the development entry point cannot write into canonical
run directories.

### Stage A1: Generalization-Oriented Fine-Tuning

Make one coherent intervention aimed at subject-domain overfitting:

- extend the head-only warmup;
- after warmup, unfreeze only the final X3D stage instead of the full backbone;
- use a smaller learning rate for the unfrozen stage than the current full
  backbone learning rate;
- keep pretrained BatchNorm running mean and variance frozen;
- add modest label smoothing;
- add IR-appropriate intensity and acquisition perturbations: brightness,
  contrast, gamma, low-amplitude noise, and light blur.

Geometric augmentation remains conservative because ROI geometry and small
interaction objects are action evidence. Augmentation parameters must be frozen
in configuration before a run begins and recorded in the resolved config.

### Stage A2: Controlled Capacity Adjustment

If A1 misses the target, compare a small pre-registered set of unfreeze depths
or warmup lengths. Do not run an open-ended hyperparameter search. Candidate
selection uses the three agreed fold 0 metrics, with Accuracy primary and the
Macro-F1/worst-user constraints acting as hard guards.

### Stage C: Temporal-View Ablation

Only if the best Stage A candidate remains below target, test the teammate-
motivated hypothesis independently by adding either deterministic clip-count
caps for long trials or training-time clip dropout. Keep 13 frames per local
clip and validation aggregation unchanged. Do not combine multiple temporal
changes in the first temporal ablation.

## Evaluation and Stopping

Each completed candidate reports:

- trial-level Accuracy and Macro-F1;
- per-user Accuracy and worst-user Accuracy;
- duration-bucket metrics for `<=13`, `14-32`, `33-64`, and `>64` frames;
- train-eval versus fold0-validation gap when practical;
- selected epoch, trainable parameter count, and artifact hashes.

Stop successfully when a single checkpoint simultaneously reaches all three
target metrics. Stop for human review before deleting artifacts or materially
changing the experiment family. A validation Accuracy decrease greater than two
absolute percentage points from the relevant parent candidate also triggers a
human-review checkpoint; the trained model and logs must be retained.

If the fixed candidate sequence does not reach the target, retain and report the
best honest result and its remaining gap. Do not change evaluation ownership,
sample population, aggregation, or metric definitions to manufacture the target.

## Deliverables

- isolated development configuration and run entry point;
- tests for canonical artifact protection and trainable-stage policy;
- resolved configuration and provenance for every run;
- a fold0 tuning comparison report;
- retained checkpoints and prediction archives;
- no mutation of canonical Phase 4/5 artifacts.
