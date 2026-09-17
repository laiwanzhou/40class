# CUHK-X Small Model Track — 模型与外部资源说明

## 推理依赖与训练资源分离

最终推理只加载 `checkpoints/model.pth` 提供的Student和YOLO姿态权重。MC3主干已包含
在Student内，不另行下载初始化权重。其他公开模型用于训练源码中的教师／特征分支，
不由新测试集推理入口加载。

100MB约束针对全部推理权重，不是代码ZIP总大小。本包交付训练代码和获取说明，没有
附带原始比赛数据或 `.npy/.npz` 特征缓存。若主办方另行要求训练权重附件，应按通知
独立交付，并遵守第三方许可证。

## 模型来源与获取方式

下列源码路径相对于 `code/training/project/aligned_multimodal/`。模型ID、快照和文件名
取自已有代码／缓存记录。重训时应将历史机器的缓存路径映射到实际下载位置；这些绝对
路径不被当前推理入口使用。

| 组件 | 精确资源 | 源码定位与用途 |
|---|---|---|
| YOLO11n-pose | 包内已嵌入；[Ultralytics](https://github.com/ultralytics/ultralytics)。原pose文件6,255,593 bytes；Ultralytics8.4.33。 | `audit_yolo11_pose_skeleton.py`；自动IR姿态／裁剪，计入最终推理权重。 |
| MC3-18 | torchvision0.24.1的 `MC3_18_Weights.KINETICS400_V1`；[权重直链](https://download.pytorch.org/models/mc3_18-a90a0ba3.pth)。 | `p86_mc3_visual_model.py`；Kinetics400初始化，训练后权重已随Student提供。 |
| VideoMAE | [MCG-NJU/videomae-large-finetuned-kinetics](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics)，另有同组织base/small变体。Large处理器历史快照 `0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4`。 | `download_p46_videomae.py`、`build_p46_videomae_cache.py`及视觉教师源码；不混称VideoMAEv2。 |
| VideoMAEv2 Large | [OpenGVLab/VideoMAEv2-Large](https://huggingface.co/OpenGVLab/VideoMAEv2-Large)，快照 `9981a9c8f77118c421e5228e1b219468a4b0238d`，`model.safetensors`。 | `p90_videomaev2_large_teacher.py`；训练期视觉特征。 |
| VideoMAEv2 distilled | [OpenGVLab/VideoMAE2](https://huggingface.co/OpenGVLab/VideoMAE2)，`distill/vit_b_k710_dl_from_giant.pth`。 | `p90_videomaev2_distilled_teacher.py`；与上行不同的教师。 |
| InternVideo2 distilled ViT-L | [OpenGVLab/InternVideo2_distillation_models](https://huggingface.co/OpenGVLab/InternVideo2_distillation_models)，快照 `449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac`，`stage1/L14/L14_ft_k710_ft_k400_f8/pytorch_model.bin`。 | `p90_internvideo2_l_teacher.py`；训练期视频教师。 |
| V-JEPA2 | [facebook/vjepa2-vitl-fpc16-256-ssv2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2)，缓存记录快照 `4aa02df83918538fc21cfaf576382fa20e489a80`；[官方源码](https://github.com/facebookresearch/vjepa2)。 | `p96_vjepa2_dense24_extractor.py`、`p142_vjepa_token_transformer_oof.py`；冻结视频token。 |
| MotionBERT | [walterzhu/MotionBERT](https://huggingface.co/walterzhu/MotionBERT)；`checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub/best_epoch.bin`、`checkpoint/pretrain/MB_release/latest_epoch.bin`、`checkpoint/pretrain/MB_lite/latest_epoch.bin`。 | `p90_motionbert_teacher.py`；Skeleton教师，各分支按配置选择。 |
| HD-GCN | [官方README及预训练下载](https://github.com/Jho-Yonsei/HD-GCN)；`ntu60_xsub_{joint,bone}_com{1,2,21}.pt`六文件。 | `p91_hdgcn_pretrained_teacher.py`的STREAMS；训练期Skeleton教师。 |
| LaViLa | [官方源码](https://github.com/facebookresearch/LaViLa)；[TimeSformer-B权重](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth)，710,793,107 bytes。 | `download_lavila.py`、`p155_lavila_teacher.py`、`p157_lavila_frame_token_cache.py`；训练期视频token。 |

Hugging Face资源可按repository ID、revision和filename用官方
`huggingface_hub.snapshot_download`／`hf_hub_download`获取。未记录revision的条目不
伪造版本，应保留实际下载文件hash。本次没有执行新下载或训练。
外部源码已核验的Git版本在 `external_resources.json`；实际交付源码以
`code/training/source_manifest.json` 的逐文件SHA256为准，而不是仅凭Git HEAD。

## 许可和数据使用

- Ultralytics本地发行元数据显示AGPL-3.0，见 `THIRD_PARTY_NOTICES.md`。
- torchvision为BSD-3-Clause；预训练数据和权重仍受相应资源条款约束。
- VideoMAEv2／InternVideo的源码许可及模型卡须分别保留，不能把源码许可自动套到权重。
- MotionBERT、HD-GCN、LaViLa、V-JEPA2及预训练数据按官方LICENSE／模型卡约束。
- 比赛原始数据从主办方取得，不随本ZIP重发；训练缓存按源码生成。
- YOLO参赛方法已获主办方确认，不意味着第三方版权义务被免除。

本地存在的EgoVLP和SlowFast候选不因目录存在就被宣称用于本提交；不是运行本包的
必需资源。完整源码可能包含其他研究分支，方法说明和入口决定实际用途。

原YOLO pose文件SHA256：

```text
869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0
```

最终唯一checkpoint为56,471,879 bytes，SHA256见README。本表是资源披露，不声明
每个历史研究分支都参与最终教师，也不声明本次重新训练过所有模型。
