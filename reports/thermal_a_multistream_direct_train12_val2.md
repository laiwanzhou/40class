# Thermal A-direct Train12 / User6-User7 Report

- Status: `completed_stopped_before_a4`
- Selected epoch: `38` of `50`
- Accuracy: `0.27321`
- Macro-F1 (fixed 0..39): `0.18960`
- Worst-user Accuracy: `0.22564`
- NLL: `3.03425`
- Zero-recall classes: `19/40`
- Checkpoint: `15069955` bytes, SHA256 `0756cecc2ae5f76e1c1d1bb7a7e607ffdf52ceebef7ef84e15e894d71665b075`
- A2 package proxy: `21327781` bytes (<95,000,000: `True`)

## Per-user

- user6: Accuracy `0.22564`, Macro-F1 `0.15311`, NLL `3.24267`
- user7: Accuracy `0.32418`, Macro-F1 `0.17734`, NLL `2.81094`

## Policy

Thermal-only, random student initialization, and no teacher logits were used. Route B remains unverified and was explicitly waived by the user, so no paired Route B comparison is claimed. No heldout-4 labels, competition test, IR/Depth inputs, or quarantined evidence were read.

A3 stops here for human review. A4 was not started.
