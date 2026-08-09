# Stage 4: Depth-only ordinal exporter and combined manifest

## Status

**Implemented and locally verified.** This stage adds the exporter and manifest contract only. It does not perform the Stage 5 real-data export smoke test or any full export.

## Implemented outputs

- `scripts/export_depth_ordinal_assets.py`
- `tests/test_depth_ordinal_exporter.py`
- `src/data/ordinal_depth.py` now also provides aspect-preserving ordinal crop/letterbox support.

The exporter is designed to write only:

- two ordinal Depth views: context and adaptive relation;
- a separate binary pixel-valid PNG for each Depth view;
- `combined_frame_manifest.csv` referencing the four existing IR files without copying them;
- `depth_ordinal_view_audit.csv`, `export_metadata.json`, and `_SUCCESS`.

## Manifest contract

Each original frame retains:

- split, class, action, sample, user, and original frame index;
- source Depth and source IR paths;
- parsed timestamp, frame id, and inter-frame delta;
- existing four IR file references;
- Depth ordinal and pixel-mask references;
- pose-valid, content-valid, effective-valid, and deterministic reliability fields;
- Depth pixel coverage and temporal-valid state.

No offline temporal sampling, padding, per-frame normalization, or per-trial normalization is performed.

## Mask and content behavior

- JET inversion occurs before crop or resize.
- Depth values use mask-aware interpolation; output masks use nearest-neighbour interpolation.
- Aspect ratio is preserved with invalid zero padding.
- Invalid pixels are cleared again after resize.
- Scalar ordinal Depth crops are rechecked with the approved two-of-three low-information rule.
- Any previously known Depth content-invalid evidence from Stage 2 is retained even if the scalar crop passes the recomputed rule.
- Hard effective masks and deterministic reliability remain separate from pixel-valid masks.

## Verification

- `11` focused codec/exporter tests passed.
- The synthetic integration export confirmed that no IR images are duplicated and that the combined manifest references the existing IR root.
- A read-only full metadata join checked:
  - frames: `84,906`;
  - samples: `2,910`;
  - ROI/effective-view rows: `509,436` each;
  - source Depth rows: `84,906`;
  - paired Depth/IR rows: `84,906`;
  - missing existing IR crop references: `0`.

## Boundary

- Competition test read: no.
- Real ordinal Depth assets exported: no.
- Stage 5 smoke export run: no.
- Full export run: no.
- Training run: no.
- Stage 5 and later stages remain pending.
