# VideoMAE margin-conditioned Top-3 routing experiment

## Goal

Evaluate whether the selected aggressive VideoMAE checkpoint can exceed its
`0.722078` validation Accuracy without another backbone training run by:

1. caching Depth-on and Depth-off embeddings for all four views;
2. rerouting those embeddings offline so global/person evidence cannot be
   excluded by the trained hard Top-2 wrist collapse; and
3. training a small monotonic margin-conditioned residual that may change only
   the anchor Top-3 classes.

## Data boundary

- Train partition: the configured train12 users only.
- Development evaluation: user6 and user7 only.
- user6/user7 may not enter gradients, normalization, sampler construction, CV
  model selection, or epoch selection.
- The VideoMAE checkpoint remains frozen.

## Required controls

- Selected checkpoint hard Top-2 anchor.
- Depth-off hard Top-2.
- Hard Top-3 and Top-4.
- Four-view soft routing.
- Grouped routing with one wrist distribution and one context distribution.
- Logit-only Top-3 residual.
- Margin-conditioned multi-route Top-3 residual.

Report Accuracy, fixed-40-class Macro-F1, worst-user Accuracy, Top-3/Top-5,
class coverage, rescue/harm counts, view-group weights, and exact cache
provenance. The fixed-wrists `0.745455` result is the minimum useful reference;
the configured `0.75` gate remains the experiment threshold.

## Public seams

- A pure tensor cached-view routing function.
- A Top-3-limited reranker whose correction gate decreases monotonically as
  anchor margin increases.
- Auditable cache and grouped-CV runner interfaces.

