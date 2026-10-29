# P32 Task7 IMU随机森林教师（IMU Random-Forest Teacher）

日期：2026-10-29，Asia/Shanghai。唯一指导为[P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。本文为实施与验收记录（Implementation and Acceptance Record），不是新plan或handoff。

状态：**Task7实现、正式训练／导出和有限自验已完成（Complete with Bounded Self-verification）**。本轮未请求或执行独立子智能体复审，不冒称独立GO；Task8未启动。

## 源配方与输入（Source Recipe and Inputs）

新增 [imu_rf_teacher.py](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/imu_rf_teacher.py)、[正式CLI](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/scripts/run_no_vote_imu_rf.py)、[真实统计验收CLI](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/scripts/verify_no_vote_imu_rf_source.py) 与 [RF测试](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/tests/test_no_vote_imu_rf.py)。注册器仅新增RF专用readiness描述符及限定阶段放行。

直接调用队友 `build_imu_cache.load_trial/resample_trial(32)` 和 `run_imu_stat_baseline.feature_vector`：五设备共用32点试次时间网格、前六通道acc/gyro、每设备6×8统计，共240维，附10维设备／时间覆盖掩码（Mask）。不使用标准化器，不用Task6的16帧窗口、P31原始点或48维global统计替代。源32点loader对同时间戳求均值、要求设备至少两个时间点、跳过少于21列CSV行等行为全部保留；它与P31 loader并不完全相同。

RF严格为400树、depth18、leaf2、sqrt、balanced_subsample、n_jobs=-1、seed20260723。只训练该阶段实际可用IMU行，使用原样本加删除一个存在设备的副本；`np.random.default_rng(20260723)`不加fold偏移，特征／mask同步清零、标签复制。select拟合train12，refit重新拟合refit14，无OOF、无val选参、不加载select训练权重或历史RF／OOF预测。

两个模型分别导出四人口的两种消费者格式（Consumer Profiles），共8份标准targets：

- `fit_teacher_target`：供A5的RF目标，已存在类为 `log(clip(p,1e-12,1))`，未见类为 `-1e6`。
- `terminal_export`：供终端专家导出，已存在类同上，未见类为 `log(1e-12)`。

类列均按0–39对齐，保留IMU有效mask；缺IMU行输出本phase完整fit标签得到的类别先验（Prior），与源缺类sentinel分开处理，不将缺设备行伪装为有效RF样本。RF的logit temperature固定1，不拟合温度；源bank的T3只在Task11消费时施加。两正式模型实际均覆盖40类，缺类导出行为另由三类小fixture验证，未借val／final标签补类。`TeacherTargets.features=None`，不会伪造视觉features。

四个直接源文件 `build_imu_cache.py/imu_data.py/run_imu_stat_baseline.py/train_final_imu_rf.py` 与最终(1)包逐文件字节相同；未执行旧main、读取p3_imu_oof或进入train_final_imu_rf后半段的S+D融合。

## 阶段门槛与血缘（Readiness and Lineage）

全局P12／Thermal／P89／A18缺项仍在，`formal_generation_ready=false`不改。新增 `imu_rf_readiness` 仅放行既有模型stage `source_expert/imu_rf`，输出目录名为A4；它绑定Task6完成摘要、四个消费源pin及refit14/final4公共32点统计。共享descriptor只有公共source_ops／raw统计祖先，具体RF只带自己phase的标签和fit用户；不放行A5或P12融合。源操作与门槛分离，不改旧协议身份或假造folds。

Task6已完成的公共运动缓存保持不变；RF补建独立标签无关source32统计缓存，train12／development2按精确ID取refit14子集。prior保存在本phase RF监督模型记录内，没有额外跨phase校准或全局统计。标签仅在相应阶段fit，以及全部模型／目标保存后的允许人口结果诊断时读取；预测本身不读取label／旧teacher概率载荷。

## 结果（Results）

| 人口（Population） | 完整正确数／分母 | 完整准确率 | 有效IMU正确数／分母 | 无IMU先验行 |
|---|---:|---:|---:|---:|
| train12 | 1948／2039 | 95.54% | 1910／1914 | 125 |
| development2（user6/7） | 123／388 | 31.70% | 119／369 | 19 |
| refit14 | 2307／2427 | 95.06% | 2265／2283 | 144 |
| final4 | 不评分 | — | 580条有效，不读取标签 | 29 |

refit14包含user6/7，只能称训练拟合结果，不是val。RF的不可用行不等于全模态缺失行：final4中RF不可用29条，而全模态缺失仍为18条，不能把额外11条从后续融合分母删除。

select有效1914行→3828条训练输入，refit有效2283行→4566条训练输入。source32缓存构建分别9.52秒／2.48秒，RF拟合及保存分别2.30秒／2.70秒；正式日志跨度约26.74秒。模型和8份预测共约72.78MiB，两个source32统计缓存约2.64MiB，权重／大缓存不提交Git。该单IMU教师的开发准确率较低，保留源配方，不据此改seed／树数／特征；是否补充视觉错误，留到既定融合与消融判断。

## 验证与接续（Verification and Continuation）

RF配方／dropout测试先2RED再GREEN，消费者sentinel／source32行为再2RED→GREEN，固定phase流水线／RF-only门槛再2RED→GREEN。最终RF／v3 roles／motion共29项针对测试通过48.46秒；包含小型真实sklearn源配方fit对照、缺类／无设备、阶段错配拒绝、select训练禁读开发及refit标签、预测禁读全部标签、幂等复用、RF门槛不能放A5。仅保留Task6全NaN骨骼fixture的源RuntimeWarning，掩码与输出有限性通过。

真实train12单条source32输入 `[5,32,10]`、五设备、240统计／10mask与源函数差0，标签载荷打开0，3.40秒。正式有限验收5.30秒：两模型400树／配置／本phase用户／source摘要／实际dropout向量、四区完整IDs／40类列／mask／概率与先验、8份profile及每区首中尾有效行model→logits数值对照通过，允许开发指标独立复算一致。final标签读取0，未重复fit、未扫raw目录或大motion／pixels／sequence，未把自验称独立复审。

正式命令：

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/run_no_vote_imu_rf.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml
```

当前v3 run下：`imu_statistics32/{refit14,final4}` 为公共统计；`A4/{select,refit}` 为模型；`A4/{phase}/{fit_teacher_target,terminal_export}/{partition}_targets.npz` 为标准目标。权威完成入口 `protocol/task7_complete.json`，绑定ready、P32、自验、真实样本及最终回归摘要；生成时标记 `task7_artifacts_ready.json` 保留原样。源RFcap为 `protocol/task7_rf_capability.json`。正式日志 `outputs/task5_audit/P32-task7-formal.log` 以 `rf_pipeline_complete` 结束，进程已退出。

证据：`P32-task7-recipe-red.log`、`P32-task7-export-red.log`、`P32-task7-pipeline-red.log`、`P32-task7-targeted-tests.log`、`P32-task7-final-regression.log`、`P32-task7-real-sample.log`、`P32-task7-final-acceptance.json/.log`。Task1–6及ROI精度残余保持原状，下一任务为尚未授权执行的Task8 MoBind预训练；其学习配方／阶段依赖仍须单独验收。
