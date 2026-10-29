# P42 Task11 断点恢复与 P12／结构化目标接续（Resume and Source Target Production）

日期：2026-10-29，Asia/Shanghai。本文件是阶段实施与验收记录（Implementation / Acceptance Record），不是第二份计划。唯一指导仍为 [P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为 [P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。Task11近似复刻仍按[P41](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P41_Task11_近似复刻规格与配置冻结.md)冻结配置执行；旧协议与Task1–10身份不变。

## 恢复状态（Resume Status）

用户明确要求继续暂停任务，覆盖此前“等待继续／不要自动恢复”的暂停状态。原 `pause.json` 保留历史，总 `progress.json` 已更新。训练源码与 `89c1930` 的本地CRLF字节完全一致；与Git LF字节不同是换行格式差异，没有修改源码或恢复身份来绕过校验。

图像教师进程PID5632，已明确打印 `p12_image_resume / completed_epochs=2`，从第3轮恢复。第3轮已完成，耗时986.74秒，当前进入第4轮；该轮每轮平衡采样1920条，不是验证准确率。相对暂停前第1／2轮347.69／257.61秒，本次明显变慢；进程读取量持续增加、GPU实际计算、日志与完整轮恢复点持续产生，没有空转或死亡证据。不根据这一轮外推整条流水线ETA，也没有据此改batch、学习率、AMP或轮数。

当前日志 `outputs/task5_audit/P41-p12-images-resume-20261029-213159.log`，错误日志同名 `-error.log`。PID仍在 `P41-p12-images-formal.pid`，精确当前入口 `P41-p12-images-current-run.json`。原formal日志保留；Skeleton不重复训练。Depth select／refit和Thermal select／refit均固定15轮，最后生成完整四区FP16预测。

## P12／global／KNN正式接口（Production Interfaces）

新增 `src/experiments/p12_source_adapter.py` 与 `scripts/build_no_vote_p12_global.py`，机器契约 `p12_global_fixed_fit_v1`，仍绑定 `task11_reconstruction_v2`。此前P41纯算子现已接入真实父产物登记（Artifact Registration），但真实图像教师尚未齐备，不能称正式P12/global/KNN已生成。

- P12按 Skeleton、Depth、S+D、IMU、S+D+IMU、Thermal、Thermal候选、最终路由八列固定顺序；温度、混合系数和路由只fit本phase标签。IMU校准使用有效S+D与RF共同人口，部署按设备数／5缩放；缺少输入保留同phase先验和false mask。八列不会因缺失而删除或伪造available。
- global按八P12＋六视觉头的14列 `log_softmax`，直接调用原 `fit_simplex`。只fit本phase有视觉detail的训练行，因为本阶段仅在这些行应用该mixture；无视觉detail保持P12 final概率。这是当前完整人口／缺输入适配，明确记录fit行数，不把先验行伪装为实际视觉样本。
- KNN采用原全窗三视野单位化、排除self、三个邻居、0.40混合；邻居池严格是当前被预测分区。train12从共享refit14公共特征按精确ID取子集，不拿development2特征充当train邻居；final4同样独立。不读取标签选邻居。
- 父记录要求精确分区／phase、四新教师、recipe、标签与先验；global另要求P12和六视觉头。参数注入也必须与父payload一致，不能传一组数组却绑定另一份预测记录。新增局部门槛不翻旧全局 `formal_generation_ready=false`。
- 本phase fit P12八分量按源complete表存FP16，held预测存FP32；遇存储溢出直接失败，不静默裁剪RF缺类哨兵。真实两个RF元数据均支持40类，本次未重复读取或统计final标签。

扩展测试发现首次生成global时返回内存FP64、重跑时读取落盘FP32的微小差异（夹具最大约5.32e-8）。现已统一为实际保存的FP32分数；KNN按源独立脚本重新把该分数转FP64后softmax，再平滑，log概率也先计算再存FP32。首次生成与缓存恢复一致，不修改既有Task1–10模型或目标。

## 中间转移模型与目标（Intermediate Transition / Targets）

新增 `source_base_targets.py`、`fit_no_vote_source_base_decoder.py`、`build_no_vote_source_base_targets.py`。直接使用原 `fit_transition_model`、唯一类束搜索后验（Unique-label Beam Posterior）与熵回退（Entropy Backoff）。冻结参数30秒／transition0.30／trigram1／beam50／posterior T1，转移计数平滑沿用原默认alpha0.25；不恢复内部OOF、不搜索开发分数。

| phase | 标签人口（Label Population） | 行数 | 会话数 | 时间未知行 |
|---|---|---:|---:|---:|
| select | train12 | 2039 | 524 | 83 |
| refit | refit14 | 2427 | 621 | 86 |

两份正式转移模型已拟合并有限验收：start/end `[40]`、bigram `[40,40]`、trigram `[40,40,40]`全部有限，指数概率相应轴归一化。只读自身fit标签和已准备公共时间表；原始数据及大缓存未重扫。凭据 `run/protocol/task11_reconstruction_v2/source_base_decoder_fit.json`，验收 `outputs/task5_audit/P42-source-base-decoder-acceptance.json/.log`。

四区公开时间容量预检最长会话均10条，无超过40类容量者；此检查只读公开时间，未读任何类别标签。未知时间不连边。结构化目标仅允许train12/refit14，对development2/final4调用在读取输入之前拒绝。unknown时间保留原emission；无效输入保持原概率并target_mask=false。原后验和entropy backoff在小样本逐值一致。

**转移模型拟合不等于独立source-base适配12轮。** 当前没有正式KNN输入，训练分区结构化目标尚未生成；学生12轮、safe-base、完整20列bank／group／gate／sequence／P310／A8仍未完成，Task12未启动。

## 有界验收与接续（Bounded Checks / Continuation）

P12/global初轮25项针对回归通过114.59秒；加入完整KNN后发现并修复 `Prediction.index` 接口字段错误。最终P12落盘往返＋完整登记链4项通过38.58秒；最终18项生产接口／转移目标／旧v3角色针对测试通过74.50秒。PowerShell接续脚本和三个CLI语法检查通过。没有重跑旧全仓suite或整cache SHA。

正式数据依赖检查按预期拒绝 `fresh depth component is not complete`，发生在拟合标签读取之前；不是把缺失教师用历史预测、普通A7或复制列填满。完整生产端仍需等真实教师输出，而非仅凭测试通过放行。

已启动隐藏后台接续进程PID42704，等待图像教师PID5632结束。脚本 `scripts/continue_no_vote_p12_after_images.ps1`，监督日志 `P42-continuation-supervisor.log`／`-error.log`。它要求三教师FP16 component齐备，再由正式Python接口校验父记录和完整分区，生成P12→global/KNN；随后只生成train12/refit14中间结构化目标。每一步非零退出或缺完成标记都停止，不忽略失败、不推进Task12。

接续的生产日志分别是 `P42-p12-global-continuation.log`／`-error.log` 与 `P42-source-base-targets.log`／`-error.log`。若用户再次暂停，需停止当前运行worker及等待接续者，并把progress标记paused_by_user；不能只停GPU后让等待进程继续。禁止重复启动已有worker／supervisor。

本轮为自验（Self-verification），尚未独立复审，不称Task11 GO。代码提交d689167／f704a13及后续脚本32890fc；最终文档提交以Git为准。main历史文档仍待整个实验完成才更新，两个原有未跟踪历史报告未提交。
