# VideoMAE wrist-person residual fusion experiment

## Goal

Test whether cached person embeddings can correct the selected VideoMAE wrist
anchor without another backbone run. The wrist path remains the exact anchor;
person enters only through a bounded, zero-initialized residual.

## Data and selection boundary

- Inputs are the valid `p3r1_corrected_*` caches from epoch 14.
- Train only on train12 users with the registered three grouped folds.
- Train every fold for one pre-registered fixed epoch count. Fold labels may be
  used for the one terminal metric, never for epoch or candidate tuning.
- Select the candidate on pooled train12 grouped predictions. Freeze it before
  evaluating user6/user7.
- The VideoMAE checkpoint and cached embeddings remain frozen.

## Required candidates

- `person_fixed10`: fixed person residual gate 0.10.
- `person_no_margin`: person residual without confidence suppression.
- `person_margin`: gate decreases monotonically with wrist Top1-Top3 margin.
- `person_margin_no_aux`: margin candidate without person auxiliary CE.

## Module contract

The fusion module consumes four cached view embeddings, exact wrist-anchor
logits and view weights, availability, and frame count. It returns fused,
anchor, and person logits plus gates and residuals. Its initial fused logits
must exactly equal the anchor. Missing person must force zero person residual.
No hard Top-k may exclude person from the module's training path.

## Loss and reporting

Use fused CE, an anchor guard, and (except the explicit ablation) person-only
auxiliary CE. Report grouped-CV and final Accuracy, fixed-40 Macro-F1,
worst-user Accuracy, Top-3/Top-5, rescue/harm, gate statistics, checkpoint
hashes, exact cache provenance, and per-class recall deltas against the wrist
anchor. The experiment is useful only if it exceeds the historical fixed
wrists `0.745455` reference without regressing worst-user `0.711443`.
