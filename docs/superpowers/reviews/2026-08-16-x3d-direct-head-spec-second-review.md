# Direct-Head Spec Second Independent Review

**Reviewed commit:** `1df511d9c403849b212de733e224d3f597998f61`

**Decision:** `CHANGES_REQUIRED`

The second review found no data leakage, heldout4/test access, canonical-evidence mutation, post-hoc threshold selection, or six-modal interface blocker. It identified two contract issues that could compromise the matched experiment and two precision notes:

1. The phrase `pre-dropout backbone features` was ambiguous because the unchanged official X3D feature backbone already contains `Dropout(p=0.5)`. The amended spec now preserves that internal dropout exactly and excludes only the new custom-head `Dropout(0.25)` from the embedding path.
2. The first amendment accidentally omitted the matched partial2 scheduler. The amended spec now freezes the identical two-epoch warmup plus cosine schedule over a 20-epoch horizon for every optimizer group.
3. The legacy exact-output regression now explicitly uses `model.eval()` with identical inputs and masks.
4. Early stopping now explicitly reuses the existing `best_macro_f1` comparator: fixed-40 Macro-F1 first, Accuracy tie-break, patience 8.

The user approved all four corrections on 2026-08-16. A third independent review is required before implementation planning.
