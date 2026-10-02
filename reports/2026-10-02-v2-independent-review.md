# v2 三文档独立复审（Independent Review of v2 Documents）

## 材料护照（Material Passport）

- 日期：2026-10-02，Asia/Shanghai。
- 来源：三名独立子智能体（independent subagents）；superpowers:dispatching-parallel-agents。
- 验证状态（Verification Status）：ANALYZED，静态文档与源码核查；未运行新流水线。
- 审计范围：实施计划、实验规格、handoff三个v2文档的一致性、方法学、数据隔离及工程移植契约。未审通用操作安全、权限或目录越界。
- 总结论：**CONDITIONAL GO，可以从Task1开始实施；三处P2局部澄清分别在Task4或Task10前完成。没有维持原v1整体NO-GO的依据。**
- 当前没有新训练、模型下载、commit/push；三个审计文档保持本轮输入版本，另存本报告和证据快照。

## 审计输入与独立性（Inputs and Independence）

| 文档 | 当前路径 | 审计版本SHA256 |
|---|---|---|
| 实施计划P | [计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md) | 399行；057129f5212057d15451975daf07676aa06b80d91a8ef88141aa6cc1d0667139 |
| 实验规格S | [规格](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md) | 161行；867dacb76c8af8f2394d8c644a1c491ebe0eba014aab86acffd692739bb25999 |
| 交接H | [handoff](C:/Users/LaiWanzhou/AppData/Local/Temp/cuhkx_single_teacher_pipeline_handoff_2026-09-23.md) | 76行；835aee93f2a5813ae0c14046e64f976e70c75a7ae85064f116ad0efa7439ce0e |

[本轮原版快照](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/v2)包含三个审计输入，字节及SHA256与上表相同。下文的P/S/H证据行号指向这些快照，未来修订不会覆盖证据。

三名审计员均未参与v2编写。方法学审计使用新上下文；数据与工程审计员在各自独立审计上下文重新完整读取v2，再核对旧报告，没有把作者的“已修复”自检当作通过依据。

| 分工 | 子智能体 | 结论 |
|---|---|---|
| 方法学（Methodology） | /root/v2_review_methodology | GO；一项非阻塞P2措辞一致性问题 |
| 数据隔离（Data Isolation） | /root/audit_20261002_methodology，本轮明确切换为数据审计角色 | GO；未发现新增P1/P2数据隔离缺口 |
| 工程（Engineering） | /root/audit_20261002_engineering | CONDITIONAL GO；两项Task10前训练目标契约待补 |

## 剩余三项局部澄清（Remaining Local Clarifications）

### R1 / P2：冻结公开教师特征的复用范围

证据：[S55](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/v2/specification-v2.md:55)禁止“模型特征”跨select/refit复用；[P209](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/v2/implementation-plan-v2.md:209)允许原始六clip特征共用。

影响：冻结公开VideoMAE、与本实验标签无关的特征，和按本实验监督人口训练的MC3序列特征，应区别对待。当前措辞可能令实现者采用不同缓存政策；不构成必然标签泄露。

最小修订：S55限定禁止共用的是“按本实验监督人口学习的特征、统计量和归一化状态”；明确冻结公开VideoMAE六clip特征可按ID共享。A1 Ridge概率仍分select/refit，A2 sequence按各自checkpoint分开。

落实期限：Task4前。该项不阻止Task1–3的协议与输入实施。

### R2 / P2：选择性锚定缺少冻结A2目标产物

证据：[P136](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/v2/implementation-plan-v2.md:136)的新sequence schema只有sequence/rows/completed；[P285](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/v2/implementation-plan-v2.md:285)的train_fusion没有显式锚点输入；[S113](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/v2/specification-v2.md:113)保留selective anchor weight0.3。

真实源码：
- [train_p86_mobind_fusion_proxy.py:895](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/train_p86_mobind_fusion_proxy.py:895)在默认live_visual_anchor=false时读取batch["anchor_logits"]。
- [p86_cached_motion_data.py:99](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/p86_cached_motion_data.py:99)读取anchor_logits_fp16.npy，254–256行送入batch。
- 原sequence构建器同时生成并保存anchor_logits；新schema没有列出这个依赖。

影响：原样移植loss会缺键；改用不断更新的当前视觉头作为锚点，会改变被冻结的训练目标。该选择不应留到实现者临时猜测。

最小修订：优先保留冻结A2锚点，按同phase的A2 checkpoint生成40类anchor_logits及valid/ID，作为sequence配套或独立ArtifactRef；Task10显式接收并验证祖先，训练中不得更新。补“锚点不随A7参数更新改变”、select/refit祖先、ID重排测试。若改用live anchor，必须明确记录为配方变体。

落实期限：Task10前；建议在Task5定义producer。无需重训大视觉教师或新增教师投票。

### R3 / P2：A7-mask的辅助和可靠性损失须显式屏蔽

证据：[S117](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/v2/specification-v2.md:117)写“缺失损失自然屏蔽”；[P291](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/v2/implementation-plan-v2.md:291)移植训练体，但Task10尚未写具体availability loss mask。

真实源码：
- [fusion_proxy:869](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/train_p86_mobind_fusion_proxy.py:869)无条件计算motion auxiliary CE，1005行按非零权重加入总loss。
- [fusion_proxy:967](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/train_p86_mobind_fusion_proxy.py:967)的默认corrupted reliability标签来自visual/motion正确性的XOR，没有自然要求motion可用。
- Task8的缺失损失测试属于A5，不能替Task10证明A7的上述损失为零。

影响：只清空motion输入和mask仍可能训练空模态分类偏置或可靠性头，与当前匹配对照定义不一致。源码未自动屏蔽不代表已发生数据泄露，属于训练目标契约遗漏。

最小修订：明确移植loss时以控制后的motion availability屏蔽motion_aux和reliability；涉及教师蒸馏时交叉使用该教师valid。全空时返回保留计算图的零loss；增加full/mask/shuffle真实batch验收，检查mask控制相应项为零。将“自然屏蔽”改为“适配层显式屏蔽”。如决定保持原loss，应改描述并明确其优化目标，不能两种行为混用。

落实期限：Task10前，不阻止先实施协议、输入与教师模块。

## 旧方法学问题复核（M1–M4）

| 项目 | v2状态 | 当前证据 |
|---|---|---|
| M1 匹配训练控制 | 文档层已修复 | S117、P288–293：四分支独立train/refit；zero-S/I是单独敏感性，不再替代训练对照 |
| M2 A9差值解释 | 文档层已修复 | S131–137、P317–323、H45：12/40独立A7启动、A8固定目标、raw输出，部署替换差值并另报A9−A7 |
| M3 转导单位 | 文档层已修复 | S121–129、P303–308、P318：partition/user/date、30秒、固定609联合池/591loss/18prior |
| M4 组合效应与统计边界 | 文档层已修复 | S9–11/S145、H34：单seed有限人口描述性，不单独归因ROI/腐败，不声称独立未见用户泛化 |

## 旧数据隔离问题复核（L1–L6）

| 项目 | v2状态 | 当前证据 |
|---|---|---|
| L1 生成读取标签清单 | 文档层已修复 | P174–182：独立可信prepare、白名单/opaque ID/vault，生成不能自动执行prepare |
| L2 缓存隐含类别字段 | 文档层已修复 | P188/P207/P223：label-optional ports，不执行旧main或补回类别 |
| L3 select/refit祖先 | 文档层已修复 | P176/P208/P364–369：训练角色与DAG、重启学生及对应refit父模型 |
| L4 ID/概率/全缺失 | 文档层已修复 | P138–142/P276–279：严格集合和40类join、609输出及18条同一先验 |
| L5 跨样本边界 | 文档层已修复 | P303–308：partition/user/date与断边、dev冻结算子，不按类别/trial分组 |
| L6 来源/冻结/揭示 | 文档层已修复 | P129/P317/P333–337：本run A8来源、16候选、精确ID/祖先/先验、单向revealed |

数据审计另核对了归一化/先验/温度的fit人口、fixture与formal拒绝、A9独立端点和标签不进入目标损失。原始类别路径只用于定位文件不是泄露证据。没有新增P1/P2数据隔离文档问题；实际代码仍须通过所列验收。

## 旧工程问题复核（E1–E10）

| 项目 | v2状态 |
|---|---|
| E1 清单模式/usable/ID/类别 | Task2/3明确重写读取、路径、摘要和缺失行 |
| E2 1384/2914与标签 | Task3–6明确任意分区port，禁旧历史入口与cache reuse |
| E3 公开权重 | Task1明确acquisition/revision/hash与本地目录；执行未验证 |
| E4 pixel/visual Dataset/sequence | Task5明确像素、新single-head targets、无标签推理；另补R2锚点 |
| E5 P31/P86/normalization依赖 | Task6改为P28与pixel帧时间，归一化显式fit/apply |
| E6 MoBind旧人口/训练循环 | Task8/10迁移固定划分循环和Selection；另补R3 loss masks |
| E7 A4→A5 | logits/log probability/40列/valid与真实loader契约明确 |
| E8 A9 stage/401/405 | AdaptationInputs含sequence，新609/591标准targets，移除历史人口限制 |
| E9 时间戳helper未接入 | Task3明确真实读取调用点与训练样本验证 |
| E10 hash/类型/smoke | 共享类型、规范序列化、两参数签名、fixture完整16候选到freeze |

源码实证已确认窗口.70/.30、hybrid直接feature loss为零、MoBind视觉view mean[N,2,1024]、IMU points4/token16/bin52/global48、A8 backed-off trigram和beam后验、A9 heads_motion_encoder均与所引用算子一致。不能把这些静态匹配等同于训练跑通。

## 三文档一致性与实施边界（Consistency and Boundary）

handoff的行数、P/S哈希、基础HEAD及未提交状态准确；它没有声称工作区干净、代码实现或运行通过。当前D盘约46.90GiB，空间不再是阻塞，动态峰值和真实batch速度仍按Task14验收。

复审后的最新阶段状态以本报告为准：旧v1整体NO-GO已解除，v2为有条件可实施。先按Task1–3推进是合理的；在Task4前澄清R1，在Task5/10前落实R2，在Task10前落实R3。没有必要为此重做OOF、增加教师、扩展多seed研究或再进行通用安全审计。

本次没有修改三个审计文档。若用户授权继续修订，可只改这三个局部契约及相关测试步骤，再核对哈希/交接；无需重写已通过的整个方案。真正的模型/标签隔离验收发生在代码实施和真实接口测试阶段。
