# P24 Task3 姿态与 ROI 源一致性（Pose and ROI Source Parity）

日期：2026-10-28。分支：`experiment/teammate-single-teacher-task1`。

本文件是验收记录（Acceptance Record），唯一指导仍是 [P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接仍是 [P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。

Task3沿用Task2已登记的P28/P29公共硬链接缓存（Hard-linked Public Caches），没有重新生成2,427/609条几何，没有运行YOLO检测器或读取封存标签。正式训练依然由源证据缺项阻断。

## 实现与边界（Implementation and Boundaries）

- `load_pose_roi_sample`通过公共输入、新协议记录及共享校验上下文（Verification Context）读取一条精确ID缓存；验证P29确实以指定P28为父记录、两者ID/用户/帧轴/观测时间相同，核对源几何参数及ROI质量。train12和development2可从refit14公共几何中取精确子集，不复用任何监督模型。
- `_load_ops`改为只读取固定源码清单的元数据，交由原符号加载器逐项校验实际消费的模块及导入依赖；取消此前入口默认全核验1,167文件的行为。时间戳恢复函数（Timestamp Restoration）、两轮检测流程和ROI算法本体没有改写。
- 新CLI `scripts/verify_no_vote_geometry.py`固定选择公共ID顺序中第一条IR可用的train12样本，不根据动作标签、模型分数或ROI质量挑样本。对该样本做原算子重建对照，而不是重跑检测器或全量预处理。
- 检查5个直接源码文件与最终提交包`code/training/project/aligned_multimodal`字节相同：原姿态算子、自适应两轮姿态、ROI生成、VideoMAE裁剪及IR读取。原始包和快照均保持只读。

## 数值与数据验收（Numerical and Data Acceptance）

真实样本有22帧，验收耗时8.614秒：

| 项目（Check） | 结果 |
|---|---|
| P29原始ROI算子重建 | 14个返回字段精确一致，包括boxes、valid、quality、source及关节辅助字段 |
| 两轮检测重试理由（Retry Reasons） | 从缓存primary boxes/keypoints重算，逐元素一致 |
| 帧轴与采集时间（Frame/Acquisition Axes） | 与本条真实IR文件名及Skeleton时间戳文件名对应 |
| 场景、人物、工作区裁剪（Scene/Person/Workspace Crops） | 源crop/resize重建像素、质量、掩码、帧索引与缓存逐元素一致 |
| 记录日内秒（Seconds Since Midnight） | 与时间侧表一致；pixels的epoch秒按采样索引一致 |
| 原始数据／大缓存／权重完整SHA | 验收调用次数0，运行时探针遇到调用即失败 |
| 全量几何重生成／检测器执行 | 均为false |

合成夹具（Synthetic Fixtures）额外覆盖：640/.10与1280/.03两轮设置、fallback替换、短缺口插值不虚构关键点、缺Depth掩码、无绝对时间不造时间、缺工作区回退人物以及双ROI缺失回退整帧。它们不冒充正式数据产物。

15项几何相关测试通过，27.84秒。首轮观察到“全量源码核验”和“缺v3缓存消费接口”两个失败，随后实现转绿。独立审查另发现Task2元数据缓存键被循环变量覆盖，修复后51项针对性集成测试通过（62.91秒），并由审查者小探针复验。真实验收只覆盖这一条样本与所消费的源码，不声称重新核验全部历史缓存内容。

## 运行与产物（Execution and Artifacts）

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/verify_no_vote_geometry.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml --submission-root 'D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)'
```

正式小证明为当前v3 run下`protocol/task3_geometry_parity.json`；日志为`outputs/task5_audit/P24-task3-real-parity.log`，测试日志为`P24-task3-red.log`与`P24-task3-tests.log`。证明记录源码摘要、输入样本ID和协议身份；若内容不同则拒绝覆盖既有证明。

## 复审与后续（Review and Next Step）

独立子智能体已对Task2提交与Task3实现、正式小产物完成复审（Independent Review）。唯一P2问题已修复并经独立探针确认，结论为Task2＋Task3 **GO**，详见[P25复审报告](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P25_Task2_Task3_独立复审.md)。P12最终配置/公开资产、Thermal补充公共清单及P89/A18生产证据仍需补齐。Task4及训练尚未启动。
