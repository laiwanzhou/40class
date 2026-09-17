# 训练方法与代码使用说明

## 交付范围

训练代码位于 `code/training/project/`，保持原项目的 `aligned_multimodal/` 与
`thermal_baseline/` 层级。本包提供完整本地训练／特征提取／教师／评估源码及配置
快照，而不是只截取推理文件或几个Python import。一个与当前训练／推理无依赖关系的
标识泄露诊断被排除。清单记录所交源文件的哈希、静态引用检查和排除项，见
`code/training/source_manifest.json`。

此处说明训练流程，**不会在评审推理时启动训练**。本次没有从零重训，这是验证范围
说明，不被列为额外提交前置条件。源码保留实际文件名以准确定位依赖；内部实验编号
不是本方案的对外名称。

## 方法流程

1. **数据预处理**：构建官方训练数据清单，提取自动姿态／ROI，对齐IR、Skeleton、
   IMU时间窗。采用官方训练标签，人物划分按代码split配置进行。
2. **教师特征与监督**：公开视觉／骨架模型提取特征，训练分类头和融合教师；会话、
   重复录制、物理模态分支形成蒸馏所需的模型输出。
3. **学生训练**：训练MC3视觉学生、Skeleton/IMU编码器及融合模块，形成基础Student。
4. **伪标签适配**：历史Kaggle无标签输入上的自动教师目标，用于40轮固定配方适配。
   这是模型生成的目标，不是人工真值；新验证数据不进行此步骤。
5. **部署导出**：Student以FP16保存，姿态权重计入同一个checkpoint。新数据仅执行
   包根目录 `inference.sh`／`inference.ps1`。

## 源码入口定位

右列仅为实际文件名，不要求读者理解内部编号。路径相对于
`code/training/project/aligned_multimodal/`。

| 功能 | 源码 |
|---|---|
| 自动姿态及ROI | `build_adaptive_yolo11_pose_skeleton_cache.py`、`build_multiscale_dir_rois.py` |
| 训练输入 | `build_p86_visual_pixel_cache.py`、`build_p86_motion_window_cache.py`及其依赖 |
| 视觉学生训练 | `train_p86_visual_pixel_oof.py` |
| Skeleton/IMU预训练 | `train_p86_mobind_pretrain.py` |
| 融合训练 | `train_p86_mobind_fusion_proxy.py` |
| 基础Student历史参数与调度 | `run_p87s_final_pipeline.py` |
| 教师特征 | 按 `EXTERNAL_RESOURCES.md` 中对应源码定位 |
| 扩展教师与目标融合 | `p307_union_repeat_group_sequence_audit.py`、`p309_union_repeat_group_test.py`、`p310_union_repeat_precedence_teacher.py` |
| 最终学生适配 | `adapt_p87s_test_student.py` |

完整源码也包含其他开发实验，不自动运行，不应当成最终推理依赖。历史调度器含当时
验证记录门槛与多步调用，不能作为新的推理入口。

## 最终适配的实际配置

以下命令来自保存的适配配置和脚本参数定义。仅用于重现训练最后一阶段；基础Student、
自动教师目标和缓存须已按源码生成。**不是新测试集运行命令**。在
`code/training/project/` 下执行，维持项目相对布局：

```bash
python aligned_multimodal/adapt_p87s_test_student.py \
  --base-checkpoint aligned_multimodal/runs/p87s_fusion_all2914_v1/unified_student.pt \
  --structured-targets aligned_multimodal/runs/p310_union_repeat_precedence_teacher_v1/student_test_targets.npz \
  --sequence-cache aligned_multimodal/runs/p87s_test_mc3_sequence_v1 \
  --motion-cache aligned_multimodal/runs/p87s_test_motion_window_t16_v1 \
  --pixel-cache aligned_multimodal/runs/p87s_test_pixel_cache_t16_r160_v1 \
  --output-dir aligned_multimodal/runs/student_adaptation_reproduced \
  --epochs 40 --batch-size 64 --workers 0 \
  --fusion-learning-rate 0.0001 --visual-head-learning-rate 0.00005 \
  --minimum-learning-rate 0.000005 --weight-decay 0.02 \
  --temperature 1.0 --confidence-power 0.0 \
  --adaptation-scope heads_motion_encoder --seed 20260826
```

这是Bash多行写法；PowerShell可合并为一行。使用新输出目录，不覆盖已验证权重。
基础训练参数见历史调度器和 `docs/provenance/visual_student_training_record.json`、
`student_fusion_training_record.json`；适配配置见 `student_adaptation_record.json`。

## 训练输入和复现范围

- 官方原始训练集和历史无标签输入从主办方取得，不随包重发。
- 公开教师源码／权重按资源表获取，训练中间结果按源码生成。没有为了交推理包而
  拷贝开发机器数GB的特征和预测缓存。
- 历史源码有本机缓存绝对路径及资源默认值，重训需按实际位置调整。不能将历史
  研究脚本说成在空目录下一条命令即可全部完成；这不影响已验证的便携推理入口。
- 当前推理代码和权重与发布回归保持一致；训练源码完成语法及本地引用检查。本次
  没有执行新训练，不将“重新训练一次”冒充官方额外要求。
- 历史全局OOF／校准谱系的限制见 `LEAKAGE_AUDIT.md`，不会因文件打包而消失。
