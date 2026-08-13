# X3D-S IR Phase 5 Evidence Registration

## Decision

Phase 5 registers the competition-retained pure X3D-S IR specialist as
`ir_x3d_s_k400_pure`. This is sparse expert evidence for the later six-modal
program, not a complete multimodal model.

## Canonical train-14 OOF evidence

- Role: `oof_train14`; canonical seed: `20260715`.
- Population: 2,320 unique usable-IR trials, 14 users, all 40 classes.
- Assignment SHA-256: `2a0dde67ced6f40cf3a163f63d545f1e3cde3545b6ee0eb0291cf1c52c38ea76`.
- Evidence SHA-256: `98517ac3785ab55647b6e8da4e27d31b8096a6115f1cfde9f2607ec861987f21`.
- Model-lineage SHA-256: `dac5dfb22475ecb2a385bf0a419f9a41388b75f6510e3f3b80af19b0964ba59b`.
- Resolved-config-lineage SHA-256: `8b751fade967b13656352e2acf49ecaf6c3a9d4873cd0e1527d033c605746f71`.

Canonical OOF Accuracy is `0.565517`, Macro-F1 is `0.480786`, and worst-user
Accuracy is `0.420690` (`user1`). Class 25 has zero recall. Duration Accuracy is
`0.540541` for `<=13`, `0.587364` for `14-32`, `0.575503` for `33-64`, and
`0.474359` for `>64` frames.

The fixed-budget matched comparison is limited to the common 800-trial outer
fold 0. X3D versus the 10-epoch MobileNet/TCN sanity baseline is `0.571250`
versus `0.283750` Accuracy and `0.486918` versus `0.173536` Macro-F1. X3D has
257 unique-correct trials, the baseline has 27, both are correct on 200, both
are wrong on 316, oracle-pair Accuracy is `0.605000`, and prediction
disagreement is `0.638750`. These are fold-0 complementarity diagnostics, not
a completed three-fold matched experiment or confidence interval.

## Frozen train-14 finalization

The nine Phase 4 selected epochs are `10, 29, 11, 16, 20, 12, 8, 10, 7`.
Their median froze finalization at 11 epochs with seed `20260715` and a
30-epoch cosine horizon before training began. All 2,320 train-14 trials were
used without validation. The final checkpoint has SHA-256
`d913b91dec7273e9619394d7f81b72087f0265e77e3d846bc86d54991f776c58`
and resolved-config SHA-256
`af1920f93c94efd29c2c1c0956cd59364dc419a12fa129dbfb0fd1e5243ebe0f`.

The checkpoint is 14,388,607 bytes; YOLO11n-pose is 6,255,593 bytes. The
provisional IR route subtotal is 20,644,200 bytes, below the 95,000,000-byte
complete-package ceiling. It is not the complete six-modal size result.

## Quarantined held-out evidence

- Role: `heldout`, `evaluation_only=true`, quarantined until Phase 10.
- Population: 590 unique usable-IR trials from only `user4`, `user17`,
  `user23`, and `user24`; availability is 590/590.
- Evidence SHA-256: `420a338adedca97c60601b677dd07935e5e92da3f87b87e0254d832b2d3c34be`.
- The NPZ physically omits the `labels` key. No held-out Accuracy, Macro-F1,
  correctness, class rescue, or other label-derived diagnostic was computed.

The first inference attempt stopped before producing an archive because the
builder passed `[B,K,V=1,C,T,H,W]` directly to the model. A RED regression test
captured the existing trainer contract; the builder now accepts exactly one
deterministic view and removes that dimension before model inference. The
final checkpoint, data, and protocol were unchanged. Superseded derived
archives carrying the less precise base-YAML config hash are preserved under
`superseded_base_yaml_hash/`; the canonical paths contain resolved-config
lineage hashes.

## Quality and claim boundary

Native IR quality and masks remain serialized. The pre-registered first-pass
fusion mapping is the label-free constant `1.0`, with SHA-256
`985bf5ef7a9347dac9dddeea60e185d2d26e05e673123aa49827f6fd627d0cc9`.
It avoids inventing a post-hoc cross-expert reliability scale.

The Phase 4 decision remains: `competition-retained; full matched primary rule
not evaluated`. Phase 5 establishes reproducible evidence and quarantine; it
does not use held-out feedback and does not begin Phase 5.5 or fusion fitting.
