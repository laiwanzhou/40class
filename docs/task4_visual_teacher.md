# Task4单视觉教师（Single Visual Teacher）

工作树：`D:/work/2026.7.14_kaggle/_single_visual_processing_replication`。分支：`experiment/teammate-single-teacher-task1`。完整计划：[实施计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md)。

## 接口与配方（Interfaces and Recipe）

`src/experiments/visual_teacher.py`新增六clip抽取、固定72候选、一个Ridge分类头、同phase prior与TeacherTargets。四个主要接口均必传`protocol=`；抽取额外接受device、clip_batch和max_trials。真实max_trials产物为partial，正式拟合拒绝；`--smoke`只允许fixture协议。

复用原P46早/晚窗0–.70/.30–1.0，每窗16帧与scene/person/workspace；人物裁剪1.15、工作区1.40，缺工作区回退人物框，缺人物框回退整帧。公开VideoMAE-Large始终冻结；恢复并核验注意力偏置（Attention bias）。

六特征族与P85相等；StandardScaler→RidgeClassifier(lsqr,tol=1e-5,max_iter=5000)。select只拟合valid train12，按development2全388行决胜；refit重新拟合valid refit14，分类头和统计分别注册，不承继select检查点。固定T1输出softmax(Ridge scores)，不复用历史OOF温度。

完整标签无关ROI和公开六clip features分别覆盖refit14/final4；训练/开发按opaque ID取子集。最终标签仍在Task13冻结全部候选前封存。本阶段只能报告development2模型选择结果。

## 执行命令（Commands）

以下命令从上述工作树运行，Python为`D:/Anaconda/envs/PyTorch2.7/python.exe`；使用`-B`避免改写导入源码目录。为便于阅读，以下以`python`代指该解释器。

```powershell
python -B scripts/build_no_vote_pose_roi.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition refit14 --device 0
python -B scripts/build_no_vote_pose_roi.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition final4 --device 0

python -B scripts/run_no_vote_visual_teacher.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --phase extract --partition refit14 --roi-summary outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/pose_roi/refit14/full/summary.json
python -B scripts/run_no_vote_visual_teacher.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --phase extract --partition final4 --roi-summary outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/pose_roi/final4/full/summary.json

python -B scripts/run_no_vote_visual_teacher.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --phase select --features-ref outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/visual_features/refit14/full/artifact.json
python -B scripts/run_no_vote_visual_teacher.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --phase refit --features-ref outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/visual_features/refit14/full/artifact.json
```

预测命令为`--phase predict --partition <分区> --model-phase select|refit --features-ref <该分区特征artifact.json>`。train12/development2使用select模型与refit14 raw features；refit14使用refit模型与refit14 raw features；final4使用refit模型与final4 raw features。预测器只读公共manifest，不读取监督标签；模型、prior和祖先均须验证同phase。

候选明细保存在`A1/select/grid.json`，选定配方`selection.json`、唯一模型`head.joblib`、同phase prior的显式引用`prior_artifact.json`及标准`<partition>_targets.npz`在本run内。NPZ输出sample_ids/class_ids/logits/probabilities/valid，六clip features为独立父产物，不含labels/users/folds旧格式。

## 验收与续跑（Acceptance and Resume）

- 最新72项针对性测试通过；全仓513通过、同名7项既有旧缓存缺失失败，无新增失败。
- 真实train12一条新producer六clip编码成功，features[1,2,3,1024]、kinetics_logits[1,2,3,400]，峰值显存1.449GiB。它仍是partial验收，不是正式教师结果。
- 首次真实编码发现上游torch装饰器来源误判，unwrap校验后保留原inference wrapper；回归修复已通过。
- 缓存绑定协议、权重、ROI、producer SHA、clip batch与原始文件集合。任意identity/内容/新增帧漂移都会拒绝；正式source改动后不能悄悄复用旧特征。诊断旧产物仅在本run输出目录归档。
- 全量ROI已生成2427/609条；完整registry验证、正式六clip抽取及72候选/refit的动态状态见执行日志与交接，不能由上述单样本验收推断已完成全量拟合。

审计记录：[实现独立审计](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-02-task23-task4-independent-audit.md)。用户后续已授权Task4实际完成、工程修复及正式产物独立审计通过后自动推进Task5，本轮止于Task5。

## 2026-10-03校验耗时修订（Verification Cost Revision）

Task4当前实现递归读取P28/P29的相同原始文件清单：204,752个文件在单次模型校验中被读取409,504次，预测入口与登记还会重复这些检查。select/refit与两个预测阶段分别耗时约3小时08分、3小时48分、2小时43分、2小时45分；模型编码本身的日志用时合计约44分钟。此为校验性能问题（Verification Performance Issue），进程存活与中间预测正确不等于校验效率合理。

Task4旧进程已于19:20:46成功完成。已修复调用链，后续采用直接消费文件与记录元数据检查，取消cached head/predict中的大面积原始SHA和祖先文件内容递归复验。旧Task4抽取器实际生成时的原始inventory检查保留，后续常规阶段不调用它；读取ROI时只检查消费单条，模型初始化只检查实际权重并精确绑定既有公开receipt。修复原66项测试、两项新增补丁回归通过，受影响16项再测通过；真实四份targets记录、一次features加载与8条train12复算共4.256秒，原始SHA读取0、logit差0。五份旧源码按原SHA归档于本run的protocol/task4_v1_sources；旧产物引用及模型配方未改变。2026-10-03正式方法学与工程独立审计均GO，Task5已开始TDD实施。
