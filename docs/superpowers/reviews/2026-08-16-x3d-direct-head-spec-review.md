# Direct-Head Spec Independent Review

**Reviewed:** `docs/superpowers/specs/2026-08-16-x3d-train12-val2-direct-head-design.md`

**Decision:** `APPROVED_WITH_NONBLOCKING_NOTES`

No blocker, heldout4/test access, canonical-evidence mutation, or evident data leakage was found. The independent reviewer confirmed that the parameter groups, head-only warmup, restored shared backbone LR, 2048D archive handling, and frozen decision rule are implementable with the current trainer and contracts.

The reviewer requested six pre-implementation clarifications, all approved by the user and incorporated into the authoritative spec:

1. Treat Direct-Head as a composite replacement of projection, LayerNorm, GELU, and classifier input, not a parameter-count-only causal ablation.
2. Pre-register train metrics, train-validation gaps, parameter counts, per-user and duration metrics as non-decision diagnostics.
3. Add a legacy missing-`head_type` plus projected-checkpoint `strict=True` load/output regression test.
4. Define the embedding as the normalized mean of valid pre-dropout per-clip backbone features; keep `ExpertOutput.embedding` runtime-required but fusion-ignorable.
5. Audit 2048D archive shape/bytes, peak memory, checkpoint bytes, and provisional route bytes.
6. Freeze formal checkpoint selection as Accuracy, then fixed-40 Macro-F1, then earlier epoch; retain Macro-F1 patience 8 and define worst-user over user21/user22.

The amended spec is approved to proceed to implementation planning after final user review.
