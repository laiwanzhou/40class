# Teacher Source Snapshot Branch Design

## Objective

Create a clean, source-only `teacher` branch in `laiwanzhou/40class` that preserves the teammate's supplied training system for future ablations without incorporating any files or commits from the owner's current work.

## Branch model

`teacher` is an orphan branch with no parent commit. Its root contains only branch-level documentation, exclusion rules, and `teammate_teacher/`. It does not inherit `main`, `skeleton_raw_data`, or any dirty working-tree state.

## Included material

The authoritative source is `CUHK-X_Small_Model_Submission_20260911/code/training/project/`. Every one of its 1,167 files is copied byte-for-byte under `teammate_teacher/project/`. The original source and build manifests, package READMEs, and complete supplied `docs/` tree are retained for provenance, limitations, external-resource disclosure, and third-party notices.

Two new explanatory artifacts are added: this migration specification and `DIFFERENCE_INVENTORY.md`. They do not alter the copied training source.

## Excluded material

No competition dataset, inference checkpoint, teacher weight, feature cache, OOF tensor, Kaggle submission, generated output, virtual environment, or packaging ZIP is included. The file `aligned_multimodal/audit_p117_identifier_leakage_ceiling.py` is absent because the supplied source manifest explicitly excluded it as an unrelated diagnostic; this branch does not reconstruct missing material.

## Integrity and safety

The original SHA256 manifest remains authoritative for `project/`. `verify_snapshot.py` must report exactly 1,167 listed and present files, zero missing/unlisted files, zero hash mismatches, and zero Python syntax failures. A tracked-file scan must reject data/model extensions and files larger than GitHub's normal 100 MB limit. A credential-pattern scan must find no likely secret values before push.

## Interpretation boundary

This branch preserves evidence and implementation, not a claim that every historical candidate entered the final P315 model or that historical OOF estimates are strictly independent. `DIFFERENCE_INVENTORY.md` distinguishes the selected final path, training-time teacher capabilities, and historical experiments where the supplied records permit that distinction.
