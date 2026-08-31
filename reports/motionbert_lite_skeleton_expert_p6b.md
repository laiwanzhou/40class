# MotionBERT-Lite Skeleton Expert P6-B Result

- Status: `stopped_after_b1_rejection`
- B1 passed: `False`
- B2 executed: `False`
- Fusion qualified: `False`

## B1 validation

- Accuracy: `0.118557`
- Macro-F1: `0.017147`
- Worst-user Accuracy: `0.113300`
- Predicted classes: `9/40`
- Zero-recall classes: `36/40`

## Visual complementarity

- Unique rescues: `5`
- Harms: `233`
- Net: `-228`
- Visual + MotionBERT oracle Accuracy: `0.719072`

All fixed B1 gates except unique rescue coverage across both users failed. The frozen pretrained representation is not qualified for partial fine-tuning or multimodal fusion under this experiment.
