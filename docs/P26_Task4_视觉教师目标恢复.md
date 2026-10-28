# P26 Task4 视觉教师目标恢复（Visual Teacher Rectification）

日期：2026-10-28。唯一计划为[P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。本文记录Task4实施和产物状态，不是新计划或交接。

当前状态（In Progress）：视觉头选参、refit及四区A1/六族/十机制目标已经生成并获独立GO；完整全窗公开特征仍在编码。Task4尚未全部完成，Task5没有启动，不读取final4标签。

## 阶段门槛修订（Stage Readiness Ruling）

旧统一学习门槛把P12/P89的证据缺项应用于全部视觉拟合，与规格“P12/KNN不阻塞Task4/5”和本计划Task4顺序矛盾。本轮按规格细化为可审计的`visual_readiness`描述符：绑定原源清单、Task3几何证明和5个消费源文件摘要，只允许A1、六头、先验与固定Ridge机制头。

YAML、协议身份和全局`formal_generation_ready=false`保持不变；P12/专家bank/融合等仍不能越过原门槛。Task5学生如需后续独立放行，应在其实施节点另验证实际依赖，不借用Task4权限。该解释已经独立审查者明确认可，见[P27](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P27_Task4_实施前独立复审.md)。

## 复刻内容（Source-parity Implementation）

- 直接调用源六族特征、StandardScaler→Ridge、sample_weights、aligned_scores_40和fit_temperature；每族12配置，按accuracy/balanced_accuracy/macro-F1/较小power/较小alpha选择。主family固定early_late，不在六族之间再选主头。
- select仅fit train12有效IR，val仅用于预定配置选择；温度仅fit train12预测与标签。refit14继承超参并重新拟合、重新估温。保存原始分数、模型自有head温度和校准分数，目标明确标记in_sample/non-OOF；缺IR始终用本phase训练先验。
- 另训练固定Ridge(alpha3000/power0.75)，以同phase fit的逐clip L2特征均值产生十项原干预；输出T1 raw logits，不消费MC3或学生序列、不套A1温度，也不增加大骨干根节点。
- 全窗使用原uniform16覆盖0–1全时段，scene/person/workspace及1.15/1.40裁剪、原公开VideoMAE-Large/processor/encode；没有把早晚六clip平均当全窗特征。此阶段只生成无标签特征，不学习KNN或混合P12概率。
- 全流程复用一个校验上下文（Verification Context）；辅助头不重复构造TeacherTargets扫描全部特征，主A1使用已验证公共数组的有界检查。旧v2仍走原逻辑，旧模型/69.07%结果不改身份。

## 当前验证（Verification）

源数学、温度、类对齐、机制和完整phase小夹具已通过；原fullwindow.prepare_trial逐帧对照通过，负limit与额外完整特征扫描拒绝探针通过。旧协议身份不变。执行前独立代码GO的具体范围和额度中断限制见P27，不把它当作未来正式产物已通过。

真实train12一条全窗GPU验收通过：features=`[1,3,1024]`、kinetics=`[1,3,400]`、valid=true且全部有限，耗时16.25秒。公开权重48个attention偏置张量验证最大差异0，原生split布局；没有修改权重或训练大视觉骨干。

最终49项针对性测试通过，115.41秒。独立产物审查已核验68组targets/raw scores、102个唯一祖先/记录、phase与ID/先验、raw/T和参数继承，不重读特征大表，头部GO见[P28](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P28_Task4_产物独立验收.md)。

主early_late选择alpha=300、class_weight_power=0.75，select/refit温度分别0.1206300185060063/0.1397821300164829。

| 分区（Partition） | 正确数／分母 | 解释 |
|---|---:|---|
| train12 | 1985/2039（97.35%） | 拟合人口，1957条有效IR |
| development2 | 268/388（69.07%） | 固定user6/user7开发集，385条有效IR |
| refit14 | 2355/2427（97.03%） | 包含开发用户的拟合人口，不是val分数 |
| final4 | 不评分 | 609条，591条有效IR，18条缺失使用refit先验 |

恢复校准后，主头开发准确率仍为69.07%；不能把温度变小、训练目标置信度提高当作跨用户准确率提升。训练／开发有效IR平均置信度约0.9993/0.8112、熵约0.0035/0.6764，后续蒸馏应保留这些实际统计，不根据val改温度。

日志：`outputs/task5_audit/P26-task4-heads.log`、`P26-task4-fullwindow.log`和`P26-task4-fullwindow-acceptance.log`。选定模型及targets写到当前v3 run的`visual_heads/`、`A1/`和`visual_mechanisms/`；全窗写到`fullwindow_features/{refit14,final4}/full/`，逐样本缓存支持同身份接续，完成后才登记完整ArtifactRef。

## 接续命令（Continuation Commands）

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/run_no_vote_visual_teacher.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml --phase source-pipeline
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/run_no_vote_visual_teacher.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml --phase fullwindow --partition all --device cuda
```

执行前先看日志和现存进程，禁止重复启动。完整Task4需六族select/refit模型、两先验、固定机制头、四区A1/六族/十机制targets、refit14/final4真实全窗特征以及必要验收；不能仅凭头训练结束或acceptance样本登记就判完成。
