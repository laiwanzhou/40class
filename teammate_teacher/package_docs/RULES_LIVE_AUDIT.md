# Current rules verification — 2026-09-08

The organizer's current Kaggle replies were read directly in the browser, including
replies displayed as approximately 18 hours / one day old. This supersedes the old
candidate package's unavailable-discussion warning. No inference code or weights
were changed by this documentation update.

| Topic | Verified organizer clarification |
|---|---|
| [734967](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/734967) | AI coding assistants are allowed; directly calling closed-source APIs/LLMs to solve the task is not. Model pseudo-labeling, including test self-training, and compliant small pretrained models are allowed. |
| [724942](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/724942) | Earlier explicit confirmation that coding assistants are permitted. |
| [729056](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/729056) | All inference weights must be in one checkpoint under100 MB on disk; reduced precision is allowed. Report efficiency. |
| [735601](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/735601) | Preprocessing and recognition models both count toward the total budget; scoring uses one correct label per clip. |
| [729989](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/729989) | Automatic test pseudo-labeling is permitted without truth/manual labels; future unseen data requires generalization. |
| [738333](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/738333) | Small YOLO person-cropping preprocessing is allowed with its weights included. The discussed pretrained video CNN is also accepted. |
| [739668](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/739668) | Stage2 requires code, weights and reproduction. New held-out data addresses advantages from the early leak. |
| [739745](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/739745) | No raw16-bit Depth or privacy-sensitive RGB Skeleton visualizations are supplied for this competition. |

These are summaries of Gvine15's Competition Host replies, not certifications of
this particular submission. Model-assisted pseudo-labels do not authorize test-truth
recovery or manual label edits. Third-party copyrights remain applicable.

The [official website](https://openaiotlab.github.io/CUHK-X-Challenge/) lists the
verification package components: code, checkpoint, inference.sh, README and a signed
honor declaration. Its deadline text has both notification-plus48-hours and
September22 23:59UTC wording. Follow the actual shortlist notification. The Kaggle
countdown tooltip read September15 2026 23:55 China Standard Time during the check.

See RELEASE_STATUS.md for actual unfinished delivery items. AI coding permission
is no longer an unresolved item. Existing run/provenance summaries are historical
records, not a guarantee of accuracy on new data.
