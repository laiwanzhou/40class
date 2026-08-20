# Thermal MobileNetV3-Small + TSM stopped-run audit

- Status: **stopped after epoch 14 by explicit human decision**
- Best checkpoint: epoch `12`, SHA256 `bd73845f67830349644284a61c3a4800498f5339ba7653f3bac975e17ca2338c`.
- Accuracy / Macro-F1 / worst-user Accuracy: `0.286472` / `0.213697` / `0.251282`.
- user6 Accuracy/Macro-F1: `0.251282` / `0.166066`.
- user7 Accuracy/Macro-F1: `0.324176` / `0.199579`.
- Zero-recall validation classes: `17/40`.

## Stop assessment

Epoch 12 is the observed best. Epochs 13 and 14 regress on the primary metric while train accuracy continues rising, the train-validation gap widens, and validation loss had already reached its minimum at epoch 8. A later best cannot be excluded, but the observed curve supports the approved compute stop because this control is not near an automatic-retention threshold.

The run did not satisfy the preregistered 30-epoch completion contract and is not represented as complete. Epochs 15-30 may in principle contain a later stochastic improvement, but the observed generalization and loss trends do not justify that compute for this development control.

## Matched comparison

| Model | Accuracy | Macro-F1 | Worst-user Accuracy | Stability |
|---|---:|---:|---:|---|
| iFormer-T epoch16 | 0.294430 | 0.195834 | 0.210256 | abnormal fine-tuned activation tail |
| frozen iFormer-T | 0.161804 | 0.071199 | 0.158974 | stable, insufficient representation |
| MobileNet epoch12/14-stop | 0.286472 | 0.213697 | 0.251282 | no matched 10x activation event |

MobileNet minus epoch16 iFormer: Accuracy `-0.007958`, Macro-F1 `+0.017863`, worst-user `+0.041026`.

MobileNet minus frozen iFormer: Accuracy `+0.124668`, Macro-F1 `+0.142498`, worst-user `+0.092308`.

## Activation stability

- FP32 maximum matched feature-block / embedding RMS ratios: `1.606` / `1.192`; 10x event `False`.
- bfloat16 maximum matched feature-block / embedding RMS ratios: `1.457` / `1.208`; 10x event `False`.
- All-validation embedding L2 median/max: `9.938` / `12.681`; robust outliers `0/377`.
- Hooks exact: `True`; state unchanged: `True`.

## Decision

MobileNet does not reproduce the iFormer epoch16 extreme tail, supporting an iFormer-specific fine-tuning response under this recipe. The stopped MobileNet control is substantially stronger than frozen iFormer and more balanced than epoch16 iFormer, but its 0.286 development accuracy and 17 zero-recall classes do not justify automatic promotion or further development compute.

Checkpoint bytes: `3955263`. Provisional deduplicated package: `27599463` bytes; strict limit pass `True`.

This stopped development control cannot auto-promote. Any formal retention requires shared train-14 OOF, IR unique-correct/oracle-pair evidence, deployment bytes, and latency. Training remains stopped.
