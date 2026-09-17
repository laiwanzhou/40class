# Organizer Delivery Status

## Required technical material supplied

- `code/`: frozen inference source plus the full local training/teacher/config source
  snapshot in `code/training/project/`. Per-file hashes, syntax and local-reference
  checks are recorded in `code/training/source_manifest.json`.
- `checkpoints/model.pth`: all inference weights, including pose preprocessing,
  **56,471,879 bytes** in one checkpoint. No training-only teacher is loaded at inference.
- `inference.sh` and `inference.ps1`: raw data to CSV, independent of an AI assistant.
- `README.md` and `README_zh-CN.md`: environment, input format, commands and result checks.
- `docs/TRAINING_PIPELINE.md` and `docs/EXTERNAL_RESOURCES.md`: training sequence,
  recorded adaptation settings, pretrained model locations and third-party notices.

## Participant action still required

The actual participant must sign the organizer's honor declaration and place the
signed file at `honor_declaration.pdf`. No signature has been generated on their
behalf. Add it and refresh the ZIP/checksum before sending. Follow the organizer's
shortlist notification for upload channel, actual deadline and any team-specific fields.

Final technical reports and presentations are later-stage deliverables when requested;
they are not represented here by blank files or made prerequisites to this inference package.

## Checks and honest limitations

On2026-09-11, a fresh venv installed the README's dependencies without global/user
site-packages. The full raw405-row PowerShell run finished in349seconds and reproduced
the exact historical CSV bytes.18 input/path tests passed. Requirements, input checks,
Unicode image handling and cache isolation were repaired; the model hash is unchanged.
Current detailed evidence is in ACCEPTANCE_20260911.md/json. Earlier224-second results
refer to the development environment, not this new clean-install acceptance.

Training sources are supplied and statically checked. A new from-scratch training run
was not performed during packaging and is not imposed as an extra organizer requirement.
Historical source scripts may require path configuration and regeneration of their
training artifacts; they are distinct from the portable inference entry point.

The original data-leakage and OOF/calibration limitations remain in LEAKAGE_AUDIT.md.
One historical Skeleton-cache numerical discrepancy did not change any prediction.
Windows/GPU is the tested environment; a clean Linux installation is not claimed.
Generalization to the organizer's unseen data is unmeasured until their verification.

Current organizer replies permit AI coding assistants, automatic model pseudo-labels
and small YOLO preprocessing. Permission to use a method does not waive third-party
license obligations or constitute organizer certification of this particular submission.
