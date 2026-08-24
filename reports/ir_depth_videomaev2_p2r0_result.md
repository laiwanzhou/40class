# IR + Depth VideoMAE P2-R0 result

VideoMAE remained frozen. User6/user7 were evaluated once after CV and final training.

| Candidate | Accuracy | Macro-F1 | Worst-user | Net rescue | HC harm rate |
|---|---:|---:|---:|---:|---:|
| logit_only | 0.706494 | 0.622385 | 0.671642 | -4 | 0.000000 |
| feature_router | 0.693506 | 0.608509 | 0.661692 | -9 | 0.000000 |

- Feature-router gate passed: `False`
- Effective >=0.78 gate: `False`
- Strong >=0.80 gate: `False`
- Scope status: `partial_implementation_of_approved_p2r0`
- Hypothesis status: `narrow_variant_rejected`
- VideoMAE updated: `False`
- P2-B started: `False`

## Interpretation

The implemented narrow cached-feature reranker was rejected. The frozen VideoMAE had
already trained on all train12 users, so router CV began from near-saturated,
non-cross-fitted teacher features and did not predict unseen-user improvement.
The feature router rescued 13 validation trials but harmed 22 (net -9).
This run omitted per-view logits, Top-5 margin, duration, and the no-Depth feature
control, so it does not reject the complete approved P2-R0 hypothesis.
No corrected retry or P2-B launch is authorized without human approval.
