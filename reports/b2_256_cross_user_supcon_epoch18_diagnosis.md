# Epoch 18 cross-user SupCon diagnosis

## 1. Main conclusion

Epoch 18 does not show that the inference architecture became worse. The B2 and SupCon checkpoints use the same Depth/IR dual stem, MobileNetV3-Small body, ROI aggregation, GRU, and classifier at inference time. The projection head exists only during training.

The evidence instead supports two conclusions:

1. B2 already contains a strong unseen-user generalization problem.
2. The batch-4 SupCon training protocol did not remove user information and reduced action separability.

Therefore, the primary failure is the contrastive training construction under batch size 4, while the underlying visual representation and limited user diversity remain secondary constraints.

## 2. Deterministic B2 versus Epoch 18

| metric | B2 epoch 25 | SupCon epoch 18 | SupCon minus B2 |
| --- | ---: | ---: | ---: |
| Train Accuracy | 0.976293 | 0.934914 | -0.041379 |
| Validation Accuracy | 0.462712 | 0.423729 | -0.038983 |
| Accuracy gap | 0.513581 | 0.511185 | -0.002396 |
| Train Macro-F1 | 0.975735 | 0.893338 | -0.082397 |
| Validation Macro-F1 | 0.392827 | 0.341988 | -0.050839 |
| Macro-F1 gap | 0.582908 | 0.551350 | -0.031557 |
| Validation loss | 3.409093 | 3.189900 | -0.219193 |
| Top-3 Accuracy | 0.688136 | 0.637288 | -0.050847 |
| Top-5 Accuracy | 0.784746 | 0.730508 | -0.054237 |
| High-confidence errors | 207 | 165 | -42 |
| Predicted classes | 39 | 39 | 0 |
| Zero-F1 classes | 2 | 3 | +1 |

The Accuracy gap changed by only 0.0024. The smaller Macro-F1 gap is mostly caused by lower training fit, not higher validation Macro-F1. SupCon did reduce validation loss and high-confidence errors, so it improved confidence behavior, but this did not translate into better classification.

At the same training point, SupCon epoch 18 and B2 epoch 18 have essentially identical validation Accuracy (both 0.423729) and Macro-F1 (0.341988 versus 0.341681). This rules out a catastrophic architectural failure. The disadvantage appears against the later B2 optimum, which SupCon did not reach before its validation curve degraded.

## 3. Did the embedding become user-independent?

| diagnostic | B2 | SupCon epoch 18 | desired direction | result |
| --- | ---: | ---: | --- | --- |
| Train-user linear probe Accuracy | 0.292672 | 0.301293 | lower | worse |
| Validation-user linear probe Accuracy | 0.620339 | 0.552542 | lower | improved |
| Train same-action cross-user cosine distance | 0.079345 | 0.094295 | lower | worse |
| Validation same-action cross-user cosine distance | 0.366618 | 0.388783 | lower | worse |
| Train-to-validation action probe Accuracy | 0.450847 | 0.405085 | higher | worse |
| Train-to-validation action probe Macro-F1 | 0.381072 | 0.333061 | higher | worse |

The four-user validation probe chance level is 0.25, while the SupCon embedding remains at 0.5525. The 14-user training probe chance level is about 0.0714, while the SupCon embedding reaches 0.3013. User identity therefore remains strongly linearly predictable.

The lower validation-user probe alone is not sufficient evidence of invariance. Same-action cross-user distances increased on both train and validation, and action probe performance fell. The more consistent interpretation is that the representation became less linearly organized overall, rather than specifically removing identity while preserving action.

## 4. Why the objective did not achieve its goal

Each physical batch contains two actions and two users:

```text
action A / user 1
action A / user 2
action B / user 1
action B / user 2
```

Each anchor therefore has only one positive and two negatives. One negative is emphasized because it shares the same user. This satisfies the formal pair rules but supplies a very small contrastive set.

The sampler preserves marginal class exposure, but changes the joint CE batch distribution. Each step contains only two actions, approximately 10% of samples are repeated within an audited epoch, and approximately 10% are not visited in that epoch. The contrastive loss fell from 1.1315 at epoch 1 to 0.0278 at epoch 18. Its weighted contribution at epoch 18 was only about 0.0028, indicating that the four-sample contrastive task had largely saturated without producing the desired clip embedding geometry.

MobileNet BatchNorm receives 4 x 24 x 4 = 384 frame-view images after flattening, not only four images. However, those images are highly correlated because they come from two actions, two users, adjacent frames, and related ROI views. BatchNorm instability is plausible, but this experiment does not isolate it and it should not be treated as the proven primary cause.

## 5. Per-class tradeoff

SupCon improved 17 classes, harmed 20, and left 3 unchanged. It rescued 56 validation samples but harmed 79, for a net rescue of -23.

Largest F1 gains included:

- Wipe_windows_and_tables: +0.285714
- Lie_down: +0.242424
- Stir_drinks: +0.170370
- Read_documents: +0.115789
- Do_stretching_exercises: +0.086538
- Check_the_time: +0.085020

Largest F1 losses included:

- Wash_face: -0.485934
- Brush_teeth: -0.373402
- Make_a_phone_call: -0.352941
- Watch_TV: -0.333333
- Wipe_hands: -0.324393
- Jog_in_place: -0.190476

The F1 change has no meaningful monotonic relationship with train-user count, train support, or validation support (all absolute Spearman correlations below 0.12, all p-values above 0.47). The failure is therefore not explained only by rare classes or classes with few train users.

## 6. Is the model structure responsible?

Not as the direct cause of the Epoch 18 regression: the inference structure is unchanged from B2, and SupCon epoch 18 matches B2 epoch 18.

The structure and data do create an underlying vulnerability. Person crops retain body shape, clothing, appearance, and background cues; the 192-dimensional GRU clip embedding can encode them; and only 14 training users are available. B2's own user probes and 0.5136 Accuracy gap confirm this vulnerability before SupCon is added.

The experiment failed because the new objective did not disentangle action from user under the available contrastive set. It traded some overconfidence for lower action discriminability, while leaving the original user signal largely intact.

## 7. Evidence boundary

This diagnosis supports changing the contrastive training protocol before changing the visual backbone. A normal CE sampler plus a cross-batch feature queue or explicit multi-micro-batch contrastive set would test the intended hypothesis without forcing every CE batch into two actions. BN freezing, warm-up, and other changes must remain separate ablations rather than being combined with that test.

Competition test read: no.
