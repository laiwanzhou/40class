# P35 Task8 IR监督恢复与重训（IR Supervision Restoration and Retraining）

日期：2026-10-29，Asia/Shanghai。唯一指导为[P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。本文记录用户批准的[P34-M1](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P34_Task6至8_独立复审.md)修复，不是第二计划或交接。

当前状态：**两阶段24轮重训、四区输出、有限自验及独立复审已完成（Complete / Independently Reviewed GO）**。新版ir_supervision_v2的P34-M1已CLOSED，无新增未解决P1/P2；正式独立结论见[P36](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P36_Task8_IR监督修复独立复审.md)。Task9未启动。

## 修正范围（Correction Scope）

训练人口从旧 `sk|imu` 改为 `sk|imu|同phase A1.valid`。select从1956恢复到1957、refit从2341恢复到2342，各恢复1条仍有视觉监督的IR-only；class_weights与训练Dataset使用完全相同清单和顺序，只有三项全部无效的82／85行不参与fit。部分缺模态及IR-only仍执行源CE、视觉KD和特征损失，不新增分支损失屏蔽；重建、对比及RF项继续使用原mask。

公开motion缓存的值／有效mask和预测阶段各分支prior规则不因恢复训练监督而改变：没有S/I输入的IR-only行也不能在推理中冒称有运动证据。用户划分、A1温度、RF目标、源模型／损失／优化配方、seed、24轮预算以及Task1–7产物均保持原样；不追加超参数或按开发成绩挑epoch。

## 版本与历史（Version and History）

新版本名 `ir_supervision_v2`，输出 `run/A5/ir_supervision_v2/{select,refit}`；capability／ready／complete均使用 `task8_ir_supervision_v2_*` 独立文件。旧 `A5/{select,refit}`、旧resume／预测、P33／P34及原完成／review标记只读保留。验证器兼容旧无revision描述符，新训练强制新revision，预测根据模型revision选择目录，不能读取旧resume或把旧模型追认成已修正。

当前协议identity不变，使用同一协议内新producer／ArtifactRef版本，复用同phase原A1/RF和标签无关motion/transform；全局formal_generation_ready=false及P12／Thermal／P89／A18材料门槛和既有ROI精度残余独立保留。

## 已完成验证（Verified）

- IR-only保留mask与新目录隔离测试先RED再GREEN；补充IR-only源CE/KD测试。该测试最初误用eval模式，未产生训练重建字段，改为源train／mask_ratio.35后通过，没有修改源loss。
- 核心11项测试通过25.81秒，最终MoBind／roles／RF共30项回归通过86.29秒，5个源Transformer norm_first库警告保留。
- 真实128条train12两批各64条，刻意包含恢复的1条IR-only。源Dataset／教师／同步增强、所有loss、参与梯度和AdamW／GradScaler更新后参数差0，非有限梯度0，峰值737.39MiB，10.48秒。proof为 `protocol/task8_ir_supervision_v2_real_batch_acceptance.json`，不覆盖旧proof。
- 独立方法审查重算1957/2342、恢复1/1、排除82/85，确认class_weights同人口且未新屏蔽源loss；独立工程审查确认新旧capability均可验证、旧A5路由保持、无旧resume串用。当前尚未把产物未齐的审查称为全项GO。

正式日志 `outputs/task5_audit/P35-task8-ir-supervision-v2-formal.log`，启动PID39204／session52812；继续时先检查实际进程和history，禁止重复启动。真实数据读取／训练命令仍为原CLI，默认写修正revision：

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/run_no_vote_mobind_pretrain.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml --device cuda
```

训练结束后必须核对新两phase24轮、实际1957/2342清单／类别权重／恢复行证据、四区12份输出与完整分母、旧证据完整性和独立产物复审，再写新complete／当前版本入口并更新唯一P01/P02。先不进入Task9，不改旧P33/P34的历史结论。

## 重训结果（Retraining Results）

select／refit各24轮末态已保存；每轮分别1957／2342条，恢复行1／1、排除真正无证据82／85，训练人口摘要与类别权重人口摘要完全一致。训练过程无开发标签评估／选轮，refit随机重启。原A5保持原人口与原模型，新结果不覆盖旧目标。

| 人口（Population） | Skeleton | IMU | combined额外诊断 |
|---|---:|---:|---:|
| train12 | 1797/2039＝88.13% | 1930/2039＝94.65% | 1947/2039＝95.49% |
| development2（user6/7） | 143/388＝36.86% | 112/388＝28.87% | 151/388＝38.92% |
| refit14 | 2120/2427＝87.35% | 2289/2427＝94.31% | 2318/2427＝95.51% |
| final4 | 不评分；590有效／19prior | 不评分；580有效／29prior | 不评分；590有效／19prior |

相对旧Task8，开发S多4条（+1.03个百分点）、I多1条（+0.26个百分点）、combined诊断少3条（-0.77个百分点）。这次修正恢复了监督人口，不保证所有指标提升，也不能将单seed分数差外推为该样本在其他运行中的固定贡献；未追加调参或选择旧／新中分数更高的版本。后续以通过源方法复审的修正版为当前版本，combined仍不是主链候选。

训练耗时select241.91秒、refit283.05秒，共524.96秒；正式日志跨度562.26秒，约9分22秒。模型与12份预测约29.97MiB，不含resume。PID39204／session52812已结束，没有新自动化或后台训练。

正式有限自验11.49秒，证据 `outputs/task5_audit/P35-task8-final-acceptance.json/.log`：完整IDs／40类列／valid／prior和combined诊断通过；人口／class weights／恢复行的摘要独立重算一致。真实refit模型64条final公共输入推理，教师／标签载荷打开0，S/I logits对正式导出差0；旧Task6／7／8的15项小证据绑定保持。未读取final标签、扫描完整motion／visual大载荷或重跑大骨干。自验中的review=pending保留自验时事实，最终独立结果由P36／新版完成入口单独确认。

独立方法审查已直接核对新两模型及历史、公共mask与A1.valid，裁定P34-M1在新版本**CLOSED／PASS**；不追认旧A5修复。独立工程审查确认新旧capability兼容、目录隔离、类别权重与训练清单共用mask，无新P1/P2。独立产物审查CPU加载两checkpoint、核对12份标准目标／完整分母／分支mask与prior、开发指标及旧模型文件摘要，最终PASS见P36。本记录不扩展为Task9或后续全局链的放行。

完成入口为 `protocol/task8_ir_supervision_v2_complete.json`；当前权威版本入口 `protocol/task8_current.json`绑定新版完成标记和P36独立复审。后续Task9／10等须从该入口读取A5模型及目标，不能将原 `task8_artifacts_ready.json`当作当前修正版。原marker与P33／P34保留历史，未覆盖。修复源代码提交281e770；最终验收文档提交见Git最新日志。
