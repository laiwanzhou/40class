# Stage 8 representation smoke gate

## Result

All three fixed Depth input representations completed an end-to-end one-epoch smoke run using the same two train samples, two validation samples, seed, model, optimizer, and frame-budget configuration.

| Representation | Train/validation completed | Checkpoints saved and reloaded | Result |
|---|---|---|---|
| `raw` | yes | yes | pass |
| `relative` | yes | yes | pass |
| `raw+relative` | yes | yes | pass |

The two-sample metrics are deliberately excluded from representation selection.

The first `raw` attempt exposed a BF16 autocast scatter dtype mismatch before completing its first batch. The valid-view output buffer now follows the encoder output dtype, and a CPU BF16 autocast regression test covers this path. The successful retry completed training, validation, checkpoint serialization, strict loading, and both best-checkpoint evaluations.

No competition test data or sensor modality was read. Stage 8 pilot selection uses a separately fixed eight-epoch budget per representation.
