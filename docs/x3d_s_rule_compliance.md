# X3D-S Small Model Track Compliance Record

Access date: 2026-08-10 (Asia/Shanghai)

## Sources

1. Official challenge page: https://openaiotlab.github.io/CUHK-X-Challenge/
2. Competition-host clarification: https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/discussion/711665
3. Kaggle Small Model Track rules page: https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track/rules

The official challenge page states the Small Model Track constraint as: "model size <= 100 MB; no large pretrained backbones." It also encourages CNN, RNN, and Transformer architectures while prohibiting large pretrained foundation models.

The linked Kaggle discussion was previously verified as a competition-host clarification that lightweight pretrained CNN backbones, using ImageNet-pretrained ResNet18 as the example, and knowledge distillation are allowed when the submitted inference model remains lightweight. The discussion body could not be fetched again without an authenticated Kaggle session on 2026-08-10, so this document records that statement as a verified paraphrase rather than presenting it as a fresh verbatim quotation.

The Small Model Track rules URL currently renders competition text that identifies the Large Model Track and describes Large-Track external-model permissions. It therefore does not resolve how the 100 MB limit is measured. Until corrected or clarified by the organizer, this project uses the stricter interpretation below.

## Working Interpretation

- X3D-S is a compact CNN video architecture and is provisionally admissible under the lightweight pretrained CNN clarification.
- Kinetics-400 pretraining is disclosed and remains subject to organizer confirmation for this exact model and source.
- VideoMAE-S is not included in the submitted inference graph without written organizer approval for the exact variant and pretraining source.
- VideoMAE may be used as a training-only distillation teacher because the host clarification permits knowledge distillation. Teacher weights must not be loaded by inference.
- Every learned artifact required to transform an official raw trial into logits counts toward one aggregate inference budget.
- The internal acceptance ceiling is 95,000,000 serialized bytes, leaving headroom below the published 100 MB limit and avoiding MB/MiB ambiguity.
- Each deployable byte is counted once. A custom head embedded in the final X3D checkpoint is not added a second time.
- FP16 conversion, archive compression, and removal of optimizer state are reported but are not the sole basis for compliance.

## Inference Component Inventory

| Component | Required at inference | License | Pretraining data | Parameter count | FP32 parameter bytes | Serialized bytes | SHA-256 | Status |
|---|---:|---|---|---:|---:|---:|---|---|
| `x3d_s` | yes | Apache-2.0 (PyTorchVideo) | Kinetics-400 | 3,794,274 | 15,177,096 | 30,779,313 source checkpoint | `26b95f1605d49650b54049db40ba3a56e023b86b58c3b3e0e10e0992a9c8682f` | probe passed; provisional rule interpretation |
| `x3d_custom_head` | yes | project code | CUHK-X training split | 535,336 | 2,141,344 | 2,144,149 estimate; counted inside final X3D checkpoint | `02c4228cb8ffddeadb77f41e9028e980b95a375fb86579ac674bcead1d1bc1b5` estimate | Phase 0 estimate |
| `yolo11n_pose` | yes | AGPL-3.0 (Ultralytics distribution) | COCO pose | 2,874,462 | 11,497,848 | 6,255,593 | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` | probe passed |
| `videomae_s` | no | checkpoint-specific | checkpoint-specific | not counted | not counted | not counted | not loaded | teacher-only pending written confirmation |

The official X3D source checkpoint contains model, optimizer, and configuration state. Its complete 30,779,313-byte size is used in the conservative Phase 0 aggregate. The extracted inference-only X3D state dict is 15,503,513 bytes with SHA-256 `37eef7a6e800c3a01e141776fc79bbdffdaf49eefe284ab6ca76dc0dcc658030`.

The Phase 0 aggregate is 39,179,055 bytes: complete official X3D source checkpoint plus estimated custom head plus YOLO. This is 41.24% of the 95,000,000-byte internal ceiling. The trained deployment checkpoint must be measured again because the final head will replace the Kinetics classifier and be embedded in one X3D artifact.

## Organizer Clarification Draft

Do not post this draft automatically.

> For the Small Model Track, may we use the official Kinetics-400-pretrained X3D-S checkpoint (about 3.8M parameters) as a lightweight CNN backbone? Separately, would a specific VideoMAE-S checkpoint be allowed if the complete submitted inference package remains below 100 MB, or is VideoMAE considered a prohibited pretrained foundation model? Does the 100 MB cap apply to the sum of every inference-time learned weight file, including YOLO pose preprocessing and fusion modules, and should it be measured using FP32 serialized state-dict bytes?

## Submission Gate

The official pretrained checkpoint loaded under PyTorch 2.7.0 and PyTorchVideo 0.1.5 and produced finite `[1,400]` output for `[1,3,13,182,182]` on CUDA. The conservative aggregate remains below 95,000,000 bytes, so the technical and size gates permit X3D-S experimentation. This is not equivalent to model-specific organizer approval. VideoMAE remains blocked from final inference until written approval is archived.
