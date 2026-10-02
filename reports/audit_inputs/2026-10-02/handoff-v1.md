# Handoff: CUHK-X Teammate-Style Single-Teacher Pipeline

Date: 2026-09-23 (Asia/Shanghai)

## Objective

Continue the read-only audit and eventual implementation of a fixed-split experiment that rebuilds the teammate's non-voting teacher-to-Student pipeline from raw CUHK-X data. The experiment excludes the historical 30-teacher bank and measures the contribution of one VideoMAE-Large teacher, MC3 visual Student distillation, compact Skeleton/IMU branches, MoBind fusion, repeat/session processing, and unlabeled target adaptation.

No training has started.

## Authoritative artifacts

Do not restate or reconstruct these documents from conversation history. Read them directly.

- Current specification:
  `D:\work\2026.7.14_kaggle\_single_visual_processing_replication\docs\superpowers\specs\2026-09-22-visual-motion-no-vote-ablation-design.md`
- Complete implementation plan (1,113 lines):
  `D:\work\2026.7.14_kaggle\_single_visual_processing_replication\docs\superpowers\plans\2026-09-23-teammate-single-teacher-fixed-split.md`
- Isolated experiment worktree:
  `D:\work\2026.7.14_kaggle\_single_visual_processing_replication`
- Branch:
  `experiment/single-visual-processing-replication`
- Latest local commit:
  `e2bf1a25eb65293fd48a441fd4374661ba0db1d6` (`docs: plan fixed-split single-teacher pipeline`)
- Uploaded teammate source snapshot branch:
  `https://github.com/laiwanzhou/40class/tree/teacher`
- Local read-only teammate source checkout:
  `D:\work\2026.7.14_kaggle\_teacher_branch_upload\teammate_teacher\project`

The experiment worktree is currently clean. Existing dirty `40class` and `40class-x3d-adaptive-multiclip` worktrees belong to the user and must not be committed, reset, or cleaned.

## Frozen split and interpretation

- 12 training users: user1, user2, user3, user5, user8, user9, user16, user18, user19, user20, user21, user22.
- Development users: user6, user7.
- Final refit: all preceding 14 users.
- Final evaluation users: user4, user17, user23, user24 (609 rows).
- The four-user final population covers all 40 classes. IR is present for 591 rows, Depth/Skeleton for 590, IMU for 584; 18 rows have none of the included modalities and remain in the denominator through a frozen training-prior fallback.
- This is fixed-split development evidence, not OOF and not an anonymous official-test estimate.

## Decisions already incorporated

- Three-fold OOF was removed. The class-25 issue disappears because user1 always remains in training/refit.
- user6/user7 are development users, not final-test users.
- user4/user17/user23/user24 are reserved for one frozen final evaluation.
- Historical caches/checkpoints/predictions are not used as training input.
- Target/final caches are rebuilt from raw inputs without label fields.
- A single MCG-NJU VideoMAE-Large teacher is rebuilt from public pretrained initialization; task-specific heads and downstream models are newly trained.
- A compatible MC3 visual Student is distilled, avoiding the invalid 768-to-512 pooled-feature shortcut.
- Skeleton/IMU use teammate P31/P86 processing and one statistical RF IMU teacher.
- Thermal, Radar, MotionBERT, HD-GCN, V-JEPA, LaViLa, InternVideo2, P310 targets, and all expert-bank probabilities are excluded.
- The primary endpoint is A9-12; A9-40 is exploratory.
- At least 20 GiB free output space is required. The last observed D: free space was below that threshold, so execution must not start until storage is expanded or redirected.

## Audit history

An earlier three-agent audit exposed and led to correction of:

- target cache labels entering generation;
- invalid strict-OOF claims with globally fitted ancestors;
- the user1/class-25 OOF failure;
- ambiguous S6 primary output;
- P86/VideoMAEv2 representation incompatibility;
- missing historical caches and underestimated storage/runtime;
- user6/user7 historical reuse.

Those findings are reflected in the current spec and plan; do not apply the old recommendations to superseded versions without checking the current files.

## Final audit completed

Three replacement read-only subagents completed the final audit. All returned `NO-GO` for the current plan.

- Methodology: A9 does not reapply A8, Task 10 controls are inference perturbations rather than matched trained controls, several contrasts are bundles rather than isolated mechanisms, the primary estimand/multiplicity policy is underspecified, and A8/A9 transductive units need to be frozen.
- Leakage/partition integrity: generation currently reads the labeled canonical manifest before stripping final4 fields; A9/freeze accept arbitrary unprovenanced files; exact refit14 ancestry, final4 IDs, session/user boundaries, reveal-once state, and sample-ID joins are not enforced.
- Engineering: multiple planned upstream calls use incompatible schemas or nonexistent arguments; A1/A2 need partition-generic label-optional ports; A9's upstream script hard-requires historical stage IDs and 401/405 rows; shared plan types are inconsistent; pretrained weights need explicit acquisition; D: has insufficient space.

The detailed independent reports were delivered in the conversation by:

- `/root/plan_methodology_final`
- `/root/plan_leakage_final`
- `/root/plan_engineering_final`

Do not begin implementation until spec and plan are revised and reviewed again.

## Next actions

1. Revise both spec and plan in the isolated worktree to address the completed audit.
2. Self-review the revised documents and commit them.
3. Ask the user to approve the revised plan before implementation.
4. After approval, ask for execution mode. Native/inline remains recommended because the 14 tasks are strongly sequential.
5. Before any training, redirect output to a volume with sufficient capacity (C: was reported with about 50.8 GiB free) and run Task 1 preflight.

## Important implementation boundary

The plan is documentation only; none of Tasks 1–14 has been implemented. Do not infer that function signatures shown in the plan already exist.

Do not push, merge, or create a PR unless the user asks. Do not modify either submission package.

## Suggested skills

- `academic-research-suite` in experiment validation mode for synthesizing the final audit.
- `superpowers:receiving-code-review` when applying subagent audit feedback.
- `superpowers:writing-plans` only if plan revisions are required.
- `superpowers:executing-plans` for Native/inline execution after user approval.
- `superpowers:subagent-driven-development` if the user selects per-task subagents.
- `superpowers:systematic-debugging` for any runtime/test failure.
- `superpowers:verification-before-completion` before claims, commits, or final experiment reporting.
