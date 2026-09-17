# Teacher Source Snapshot Migration Plan

> **For agentic workers:** preserve copied source bytes; changes belong only in branch documentation and verification utilities.

**Goal:** Publish a clean orphan `teacher` branch containing the complete supplied teammate training-source snapshot and its provenance documents.

**Architecture:** One immutable source tree under `teammate_teacher/project/`, accompanied by the original manifest and package documentation. A small standalone verifier checks byte identity, source completeness, Python syntax, and source-only policy before push.

**Tech Stack:** Git, Python standard library, PowerShell for mechanical copying.

**Spec:** `teammate_teacher/MIGRATION_SPEC.md`

## Global constraints

- Do not modify either submission package or the owner's existing worktrees.
- Do not include datasets, predictions, caches, weights, checkpoints, or virtual environments.
- Preserve all 1,167 original training-source files byte-for-byte.
- Create `teacher` as an orphan branch and push only after local verification passes.

## Tasks

- [x] Create an isolated clone and orphan `teacher` branch.
- [x] Copy the complete source snapshot and provenance documentation.
- [x] Document capability differences and known evidence limits.
- [x] Run manifest, syntax, forbidden-artifact, credential, and tracked-layout checks.
- [x] Commit the verified snapshot.
- [x] Push `teacher` to `origin` and verify the remote head.
