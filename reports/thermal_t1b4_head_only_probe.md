# Thermal T1-B.4 frozen-backbone head-only probe

- Status: **completed_causal_probe_training_stopped_waiting_human_decision**
- Fixed execution: `8` of `8` epochs; hard stop honored `True`.
- Best epoch: `8`; validation Macro-F1 `0.071199`, Accuracy `0.161804`, worst-user Accuracy `0.158974`.
- Frozen backbone exact before/after: `True`.
- Post-training embedding tail disappeared under preregistered gate: **True**.

## Causal answer

With the official pretrained backbone held exactly fixed, the preregistered block5/final-embedding spike tail disappears. This strengthens the T1-B.3 conclusion that epoch16 backbone fine-tuning created the abnormal tail. The development score measures how far the unchanged pretrained Thermal representation can go with only the original BN-to-Linear head fitted.

Classification performance is negative: final train Accuracy was `0.265869`, while validation Accuracy was `0.161804` and Macro-F1 `0.071199`; `27/40` validation classes had zero recall. The low train score makes insufficient Thermal linear separability the primary limitation, rather than ordinary validation overfitting. Physical-batch-4 classifier BN noise can still reduce the head ceiling, but the normal block5, embedding, and logit ratios rule it out as the source of the T1-B.3 activation tail.

| Precision | Block5 median/max | Embedding median/max | Logit median/max | Gate |
|---|---:|---:|---:|---:|---|
| fp32 | 1.001/1.045 | 1.002/1.018 | 0.922/1.625 | True |
| bfloat16 | 1.001/1.045 | 1.001/1.018 | 0.927/1.649 | True |

## Integrity and scope

- Official pretrained SHA: `7cbd778e3604694eb1a0becbf2e6a22798586f6bb46610a5c22b39880efb967e`.
- The epoch16 model was not loaded. The prior T1-B.3 JSON supplied only the frozen spike/control sample IDs and historical comparison statistics.
- Full-frame, 16-frame normalized-time preprocessing, seed, loss, batch size, and classifier structure were unchanged.
- No crop, BN-free head, activation clamp, residual scaling, heldout labels, competition test, or quarantined evidence was used.
- This is a causal probe only and cannot be promoted from user6/user7 development metrics.
- Training is stopped pending a new human decision.
