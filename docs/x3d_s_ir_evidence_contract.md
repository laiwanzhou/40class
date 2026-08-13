# X3D-S IR ExpertEvidence Contract

## Identity and scope

The pure X3D-S IR expert is registered as `ir_x3d_s_k400_pure`. It is an IR
specialist for the later six-modal system, not a complete competition model.
A future teacher-assisted model must use a new identity and must not overwrite
these archives.

## Sparse evidence roles

`oof_train14` contains only canonical-seed (`20260715`) strict outer-fold
predictions for usable IR trials. Labels are required because this archive is
the train-14 scientific and fusion-training evidence.

`heldout` contains predictions from one model trained on every train-14 user.
The serialized NPZ must physically omit the `labels` key. Its provenance may
record counts, hashes, availability, and quarantine state, but no held-out
metric is computed before Phase 10.

Every archive contains sample and user IDs, 40-class logits, availability,
native quality and its mask, a scalar fusion quality score, class/config/model
hashes, deployed bytes, preprocessing dependencies, and optional embeddings or
diagnostics. IDs are unique and all numerical evidence is finite.

## Frozen quality mapping

The first-generation IR mapping is
`constant_1_for_first_generation_fusion`. The native quality vector remains in
the archive, but it is not post-hoc converted into a learned reliability score.
Availability and later support rules provide the first fusion gate.

## Frozen finalization policy

The nine Phase 4 selected epochs are `10, 29, 11, 16, 20, 12, 8, 10, 7`.
Their median is 11. The final model therefore trains all train-14 users for
exactly 11 epochs with seed `20260715`, while retaining the 30-epoch cosine
scheduler horizon. This rule is written to
`reports/x3d_s_phase5_finalization_policy.json` before final training starts.

## Deployment accounting

The IR route consists of the final X3D-S checkpoint plus the exact
YOLO11n-pose dependency. The route remains a provisional subtotal inside the
95,000,000-byte complete six-modal package budget.
