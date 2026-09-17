# Teammate teacher training snapshot

This orphan branch contains only the teammate training-source snapshot used to audit and progressively reconstruct the CUHK-X Small Model Track solution. It intentionally does not inherit files or history from `main` or the owner's in-progress branches.

## Contents

- `teammate_teacher/project/`: complete supplied local training, feature extraction, teacher, evaluation, test, and configuration source snapshot (1,167 files).
- `teammate_teacher/source_manifest.json`: original per-file SHA256 manifest, exclusion record, syntax check, and static local-reference audit.
- `teammate_teacher/package_docs/`: original training, external-resource, third-party-license, leakage, rules, provenance, and reproduction documentation supplied with the package.
- `teammate_teacher/ORIGINAL_PACKAGE_README*.md`: original package usage and scope statements.
- `teammate_teacher/DIFFERENCE_INVENTORY.md`: categorized comparison against the selected `visual_skeleton` baseline.
- `teammate_teacher/MIGRATION_SPEC.md` and `MIGRATION_PLAN.md`: branch scope and verification procedure.

## Deliberate exclusions

This branch does not contain competition data, generated feature caches, predictions, model weights, checkpoints, virtual environments, or the excluded identifier-leakage diagnostic named in the original source manifest. Those artifacts are not required to preserve the source snapshot and may not be redistributed safely.

The snapshot is not claimed to run end to end from an empty directory. Historical scripts retain their original relative layout and some local default paths; external models and regenerated intermediates are described in the supplied documentation. The original package explicitly states that historical OOF/calibration ancestry is not fully certified as outer-pure.

## Integrity check

From the repository root:

```powershell
python teammate_teacher/verify_snapshot.py
```

The verifier checks all 1,167 training-source files against the original manifest, rejects generated model/data extensions, parses every Python file, and verifies that the branch contains no tracked content outside this source-only layout.
