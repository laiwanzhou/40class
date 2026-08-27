# MotionBERT attribution and local modification notice

- Upstream project: `https://github.com/Walter0807/MotionBERT`
- Frozen source commit: `705d3a95354db8bdb696b3492e47a3b5537174ff`
- Upstream license: Apache License 2.0; the unmodified license text is retained
  in `LICENSE`.
- Copied files: `lib/model/DSTformer.py` and `lib/model/drop.py`.
- Local modification: `DSTformer.py` imports `DropPath` through the local
  package-relative path `.drop`; no model computation was changed.
- Purpose: training-only MotionBERT-Lite Skeleton expert qualification. These
  files are not automatically included in the competition inference package.
