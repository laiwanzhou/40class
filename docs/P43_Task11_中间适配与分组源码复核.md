# P43 Task11 中间适配与分组源码复核（Intermediate Adaptation / Group Source Parity）

日期：2026-10-29，Asia/Shanghai。本文是阶段实现与验收记录（Implementation / Acceptance），不是新计划；唯一指导仍为[P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接仍为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。活动近似复刻规格[P41](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P41_Task11_近似复刻规格与配置冻结.md)和旧协议SHA不改。

## 当前仍缺的步骤（Remaining Work）

Depth／Thermal两教师的select／refit15轮及四区预测仍在运行；最新Depth select已完成6／15轮。其后依次需要正式P12／global／KNN、fit-pool结构化目标、两phase source-base各12轮及四区raw预测，再生成safe probability／joint decision、20列bank、分组分类器及阈值／序列gate／P310／A8，最后完成正式产物验收与独立复审。Task11尚未完成，Task12未启动。

图像worker仍PID5632，最新日志 `outputs/task5_audit/P41-p12-images-resume-20261029-213159.log`。第4／5／6轮分别771.69／730.99／688.69秒，第3轮986.74秒；每轮1920条平衡抽样，不能当作独立train全集或val成绩。没有按分数改训练预算。Skeleton及Task1–10继续复用。

## 独立source-base适配（Own-phase Source-base Adaptation）

新增 `source_base_adaptation.py` 和 `run_no_vote_source_base_adaptation.py`，独立stage为 `source_base_adapt12_reconstructed`。这属于Task11，不是Task12最终目标池适配；不拿原A7的概率或最终A9-12冒充本阶段新模型。

| 项目（Item） | 本阶段冻结规则（Frozen Rule） |
|---|---|
| 初始化（Initialization） | 各自同phase A7 full完整checkpoint；select/refit各自重新开始 |
| 训练人口（Fit Pool） | select只train12；refit只refit14。user6/7的开发预测仍只由select模型给出 |
| 目标（Targets） | 本phase P85 global→KNN→原束后验／entropy backoff，必须是已登记source_base_decoder预测；拒绝普通A1/A7目标作为正式替代 |
| 预算／参数组（Budget / Groups） | 固定12轮、heads，仅visual heads与motion_residual中encoder之外参数可训练 |
| 优化（Optimization） | 原AdamW，fusion5e-5／visual1e-5，cosine→2e-6、wd0.02、batch64、workers0、seed20260812、AMP／clip1 |
| 损失（Loss） | 原T1 KL，confidence_power0，warmup0，不加CE或新阈值筛样 |
| 时间增强（Temporal Augmentation） | 原P86采样索引，同步作用于序列／运动／view quality／time；base Dataset不带label，原LabelFreePseudoDataset继续移除label并加pseudo分布 |
| 输出（Inference） | 四区完整ID／40类raw概率；缺所有学生输入用同phase先验。推理构建不装载A7初始参数或teacher目标，只载自身完整checkpoint |

这里“无development/final目标池适配”有phase边界：select完全不适配user6/7；refit14按既定设计包含user6/7，但该模型只用于final4。两phase均不使用final4做本中间阶段训练。refit14结果不称独立val分数。

训练直接使用原 `adapt_p87s_structured_student.train_label_free` 的损失、模式、参数组、优化和日志。局部AST仅把history初始化／epoch起点接到恢复函数，并在完整epoch结束加保存回调，不修改损失或更新；匹配不到唯一源结构就拒绝。恢复点保存模型、AdamW、scaler、history和Python／NumPy／Torch／CUDA RNG，校验父产物、参数及producer identity。workers0的原RandomSampler使用Torch全局RNG，未另造采样器。

拟合登记只允许完整own-phase A7 full、该fit人口的结构化目标、相应缓存／序列、先验及冻结recipe；旧global false不翻转。阶段kind使用supervised_model表述其祖先的监督用户所有权，配置明确本轮是pseudo／label-free loss，批次不读取真值。

### 验收证据（Acceptance Evidence）

- 初轮适配／旧角色14项通过53.75秒。
- 加强恢复用例使用真实source DataLoader shuffle、batch2+1、dropout与Python/NumPy/Torch随机输入：原连续12轮与第4轮中断→恢复12轮的模型、缓冲区、每轮非耗时指标完全相同；3项通过31.50秒。冻结encoder与heads参数范围、两组末轮LR2e-6及label批次拒绝均核对。
- 四条真实train12缓存输入，含一条IR-only，在CPU上分别执行原函数和恢复包装函数一个批次：KL／更新后全部状态差0，ground_truth_fields_seen=0，总验收56.38秒；保留源Transformer norm_first库warning。
- 这次接口probe用现存同phase A1分布，明确不是正式P85目标；没有登记正式模型，临时probe checkpoint已删除。日志中的训练一致率不是val／test准确率，不证明无损蒸馏；正式入口仍要求P85/global/KNN后验目标。

证据 `outputs/task5_audit/P43-source-adaptation-final-tests.log`、`P43-shuffle-augmentation-resume.log`、`P43-real-source-batch.json/.log`。源实现提交ab93e47，当前只有代码／接口验收，实际12轮模型尚未生成。

## 分组几何与P310退化（Group Geometry / P310 Degeneration）

新增 `source_group_ops.py`，每个(partition,user,date)块直接调用原P137 `group_features`，保持sqrt/full/full、P134 CONFIG、embedding=None和quality_features=false。只将原align_metadata全局函数在局部克隆的函数命名空间内接到本块真实公共时间，不修改源模块或磁盘，也不读取旧训练CSV。未知时间不伪造日期／peer，使用原无peer极限特征。

两用户、同日同时间、两次重复take及未知时间夹具中，已知块与原函数逐值一致，各自peer_count最大1，没有跨用户peer；未知时间peer0。两项分组几何／源码谱系测试通过18.47秒。

更重要的原始依赖区别：P255与P309均从p173同一基础bank出发，P309只比P255多P306列，两者classifier recipe同为C0.03／peer2／balanced=false／frequency_power0。当前单大根roster明确排除了依赖被删根的P306，其他新增physical根同样不在有效20列中。因此剪枝后old/new分组输入和配方相同；在相同完整人口及数据下，P310的new!=old优先覆盖预期全false。

这不是原队友P310普遍无效的结论，也不人为增加旧预测／重复列维持非零route。谱系证据保存于 `run/protocol/task11_reconstruction_v2/group_bank_equivalence.json`，绑定两源文件SHA及活动recipe，明确 `actual_precedence_evaluated=false`。正式bank未产生，尚未计算实际路由；后续必须核对真实相同覆盖／数组，记录自然退化，可按等价公共子计算复用同一分类器，不伪装两个独立专家。

## 后台接续（Background Continuation）

旧等待者42704为阶段历史，当前接续PID13220、worker5632，后续最新PID以 `P42-continuation-current-run.json` 和本记录更新为准。脚本 `continue_no_vote_p12_after_images.ps1` 已扩展为：三教师正式FP16输出齐备→P12/global/KNN→fit-pool结构化目标→Task11 source-base select/refit各12轮与四区raw预测。每步有退出码及完成marker门槛；错误或暂停即停止。

Source-base训练日志将为 `P43-source-base-training.log`／`-error.log`，PID文件 `P43-source-base-training.pid`，完成入口 `source_base_adapt12_component.json`。它之后仍缺safe-base／bank／group／sequence／A8实现，不会启动Task12或冒充Task11完成。

本轮没有新独立复审，不称Task11 GO；原数据、队友快照、两提交包和旧产物身份不改，main历史文档继续等完整实验结束。两份原有未跟踪历史报告未提交。

## 2026-10-30进度检查与恢复（Progress / Recovery）

**最新进度／意外中断恢复（2026-10-30）：** 检查发现旧worker5632与supervisor13220均不存在，无Python异常日志；Depth select仅完成7／15轮，最后完整轮为2026-10-29 22:36:13。Depth refit／Thermal两阶段、P12/global/KNN、中间学生12轮均未启动；Skeleton完成产物仍在。系统未重启，具体退出原因尚未证实，不能把这段时间计作持续训练。

第7轮resume已实际CPU装载核验：history连续1–7、scheduler.last_epoch=7、模型／AdamW／scaler／采样器及全部RNG齐，identity和producer CRLF SHA完全不变。按持续任务授权已从第8轮恢复，日志明确p12_image_resume/completed_epochs=7；未重跑前七轮或修改训练代码。

新worker PID56208，当前日志索引仍P41-p12-images-current-run.json；新supervisor PID29096，索引P42-continuation-current-run.json。两者改为本地WMI创建的hidden detached后台进程，工作目录／脚本／配方不变；没有安装服务或新增定时任务，不宣称能规避系统断电或一切退出。新日志均P43前缀并带20261030时间，不覆盖旧日志。恢复凭据revision/recovery_20261030.json，progress已running_after_unexpected_interruption。

接续范围仍教师→P12/global/KNN→fit-pool targets→Task11 source-base两12轮及四区raw预测，之后停在safe/bank/A8剩余实现之前，不启动Task12。旧ETA因这段停止需顺延。当前Task11未完成，无新独立复审；两份既有未跟踪报告仍不提交。
