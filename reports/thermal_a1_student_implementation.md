# Thermal Route A A1 Student Implementation

**Status:** `completed`

No model was trained and no learned weights were created. Heldout labels, competition test, quarantined evidence, IR/Depth inputs, and frozen IR/X3D evidence were not read or modified.

## Student

- B-X3D-XS: 3,056,634 parameters; 12,570,073-byte state dict.
- A multi-stream: 3,669,874 parameters; 15,049,075-byte state dict.
- The A student shares one randomly initialized X3D-XS encoder and one projection between full-frame and fixed-context views.
- Fusion is exactly 780 values: 256 full, 256 crop, 128 motion, 128 pose, four availability, and eight quality values.
- Missing crop, motion, or pose streams are zeroed after projection and retain finite full-frame logits.

## Trainer contract

- One objective-strategy entry implements direct CE and optional fixed-logit KD without changing the model class.
- AdamW, three-epoch warmup plus cosine, 50-epoch hard stop, masked loss, fixed 40-class metrics, worst-user Accuracy, and checkpoint ordering are frozen.
- Both committed configs remain `training_authorized: false` and the CLI refuses formal execution before creating output.
- The reusable epoch core covers backward, gradient accumulation, gradient clipping, schedule stepping, and selected-checkpoint prediction archives.

## Remaining gate

The full Thermal pose cache is not present. It is not needed for A1 model/trainer contracts, but it must exist before formal A3 training. A2 may probe the fixed model and input shapes, but it must not authorize training or silently replace pose with zeros.

Next action: `a2_runtime_probe`. A2 has not started.
