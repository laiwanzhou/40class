# 交接文档与实施计划独立审计（Independent Audit）

本报告前半部分保留对v1的独立审计事实；证据行号已指向字节一致的原版输入快照，避免被v2改动覆盖。下方“文档修订记录”描述2026-10-02后续修改，不把旧NO-GO自动改成独立通过。

## 材料护照（Material Passport）

- 来源技能（Origin Skill）：academic-research-suite / experiment-agent；superpowers:dispatching-parallel-agents。
- 模式（Mode）：方案验证与静态可执行性审计（Plan Validation / Static Executability Audit）。
- 日期（Date）：2026-10-02，Asia/Shanghai。
- 验证状态（Verification Status）：ANALYZED。未执行新流水线训练，也没有准确率复现实验。
- 版本（Version）：handoff_plan_audit_v1。
- 主审计对象：用户指定的交接文档（handoff）与完整实施计划（implementation plan）。规格和队友源码只作佐证。
- 结果：三个独立审计均为 **NO-GO：当前文档不能原样指导完整训练**。允许先修订文档；该结论不表示实验目标不可行。

## 审计对象与版本（Scope and Versions）

| 简称 | 文件及定位 | 本次版本 |
|---|---|---|
| H | [交接文档](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/handoff-v1.md) | 2026-09-23 历史快照 |
| P | [完整实施计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md) | 1,113 行；SHA256：`8607b2d7f9b859cb2d20ee3b16bbb403a55f15b01185013835bbc4ba293e1c9d` |
| S | [实验规格](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/specification-v1.md) | 424 行；SHA256：`8570ca4a77db3950696ae04e263997e176aa37c21d0a5d4095c0cba280ba3249` |
| T | [队友训练源码根目录](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal) | 只读源码快照 |

工作树（worktree）为 `D:/work/2026.7.14_kaggle/_single_visual_processing_replication`，分支为 `experiment/single-visual-processing-replication`，HEAD 为 `e2bf1a25eb65293fd48a441fd4374661ba0db1d6`。审计开始时工作树干净。本次只新增本报告，没有修改 H/P/S、训练源码或提交包，没有提交（commit）、推送（push）或训练。

三个审计员使用独立上下文，分别负责实验方法、数据隔离和工程可执行性；先独立核查，再结合用户明确指定的 H 核对旧问题状态。未把旧审计结论直接当作本轮证据。

- 方法学（Methodology）：`/root/audit_20261002_methodology`。
- 数据泄露与分区完整性（Data Leakage / Partition Integrity）：`/root/audit_20261002_leakage`。
- 工程可执行性（Engineering Executability）：`/root/audit_20261002_engineering`。

## 当前状态与交接准确性（Current State / Handoff Accuracy）

| H 中的状态或要求 | 当前核对结果 | 交接修订建议 |
|---|---|---|
| H22/24：分支及 HEAD | 与现况一致 | 保留 |
| H95：Tasks 1–14 为文档，尚未实施 | 抽查协议、融合、适配、总入口及配置文件均不存在；与计划阶段一致 | 写明“本轮新流水线尚未实施”，不泛指整台机器没有训练历史 |
| H71–75：上轮三个审计为 NO-GO | 本轮三个独立审计重新确认核心阻塞项仍存在 | 保留，并链接本报告 |
| H46–47：未用历史产物、目标缓存无标签 | 是规格要求，尚无新流水线可验收 | 改为“必须满足的验收条件”，避免读成已完成 |
| H53/75/91：D 盘不足、必须扩容或改输出盘 | 本轮实测 D 盘约 **46.90 GiB** 空闲，20 GiB 门槛已满足 | 更新当前快照；历史观测本身并非造假，不再作为当前阻塞 |
| H83：先修订，再复审，之后实施 | 与当前缺口一致 | 保留顺序 |
| H88–90：提交、再次批准、选择执行模式 | 属于历史工作流建议，不是审计发现的新研究条件 | 不将其视为本次 commit/push 授权，不重复提出无必要的权限确认 |

已接受的固定划分（fixed split）保持不变：train12 为 user1、2、3、5、8、9、16、18、19、20、21、22；development2 为 user6、7；refit14 为前述14人；final4 为 user4、17、23、24，共609条。18条所有纳入模态缺失的样本仍保留在分母内。

移除严格折外预测（strict OOF）主张、user1/class25 始终留在训练、MC3 原生视觉接口、A9-12 为主要端点、A9-40 为探索端点，均已有设计决定。本轮不要求重新做三折 OOF 或重新改变用户划分。

## 方法学独立审计（Methodology Review）

### M1 / P1：匹配训练对照实际写成推理扰动

证据：[P799](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:799) 对同一个完整 A7 模型循环调用 `apply_control`，而 [S271](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/specification-v1.md:271) 承诺匹配对照（matched controls）。

影响：推理时打乱或遮蔽输入只能测敏感性（sensitivity），不能区分运动内容、增加容量和正则化（regularization）的训练贡献。

最小修订：明确 A7-mask、A7-shuffle-S、A7-shuffle-I 分别训练并 refit，匹配初始化、预算、优化器和选择规则；推理 zero-S/zero-I 单列为敏感性分析。无需重复训练大视觉教师，可复用同一允许分区的上游初始化。

### M2 / P1：A9−A8 的名称与操作图不一致

证据：[P905](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:905)、[P932](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:932) 从 A7 初始化并拟合 A8 伪标签（pseudo targets），未再次应用 A8；[S352](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/specification-v1.md:352) 将 A9−A8 称作适配增量。

最小修订可不增加训练：将当前输出定义为 `A9_raw`，将 A9−A8 称作“以适配模型替换后处理输出的差值”（deployment replacement delta），并报告 A9−A7。若保留“嵌套流水线新增适配”的解释，则必须对 A9 输出再次应用冻结的 A8，并明确该输出与 A9_raw 的区别；不得重新选后处理阈值。

### M3 / P1：转导单位与会话边界未冻结

证据：[P864](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:864)、[P876](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:876)、P905。

最小修订：冻结最终样本 ID 集合、A9 是四用户共同适配还是逐用户适配、会话（session）排序键、用户/日期/录制边界、30秒断开规则、重复组（repeat group）允许连接范围，以及缺失18条的先验策略。转导学习（transductive learning）可以使用已批准的无标签目标数据，但批次定义必须事先明确。

### M4 / P2：收窄组合效应与统计解释

H7、S346 附近的语言应区分独立机制与组合效应（bundle effect）：A2−A1 同时改变骨干、容量和训练过程，应称教师到学生替换/压缩差值；当前矩阵没有分别识别 ROI 或视觉腐败训练的独立增益。A7−A6-VSI 继续称学习融合组合效应即可。

唯一主要比较为 A9-12−A1 在609条上的正确数/准确率差；其他阶段、子组和 A9-40 为描述性或探索性比较（descriptive / exploratory comparison）。固定单随机种子（single seed）不能证明种子稳健性，也不作未经设计的总体显著性声明。此次诊断实验不必为此追加多种子、论文级置信区间或多重比较校正。

复刻忠实度（replication fidelity）另需注明：队友 [adapt_p87s_test_student.py:55](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/adapt_p87s_test_student.py:55) 默认 scope 为 `heads`，P918 使用 `heads_motion_encoder`。后者应记录为明确选择的适配变体，并冻结参数组。

## 数据隔离独立审计（Data Isolation Review）

### L1 / P1：最终标签隔离在清单构造时尚不成立

证据：[P203](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:203) 先读全量带标签规范清单，再删除少数列。真实 [metadata/manifest.csv:1](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/metadata/manifest.csv:1) 还包含 `action_name`，P216 没有删除该动作标签。

最小修订：独立准备步骤产出严格列白名单的最终无标签清单；生成进程不得接收全量带标签清单。训练标签独立连接，增加直接标签字段拒绝、标签文件删除/置换后生成不变的验收。

样本 ID 中的 `cXX` 和原始类别目录是标签代理（label proxy）风险；本轮未发现其被解析成模型特征/目标的证据。路径只用于文件定位并不等于实际作弊。推荐使用不含类别的样本 ID（opaque ID），对必要原始路径规定只定位文件、禁止产生类别目标或类别分组。

### L2 / P1：原提取脚本会读取和写出类别

证据：P314–315、P372、P513 承诺直接调用旧脚本，但队友 ROI、pose、P46、P31 代码强制访问类别、按类别排序，并在 JSON/CSV/NPZ 中写入类别字段。具体源码证据见工程 E1/E2/E5。

最小修订：移植为标签可选（label-optional）、任意分区（partition-generic）提取器；最终分支不能恢复类别来满足旧接口。验收覆盖整个产物目录，不能只检查一个 NPZ。

### L3 / P1：选择与 refit 的监督训练祖先未绑定

证据：P442–453、P651–663 的教师目标与 P792–796 的 `selected.visual_checkpoint/motion_checkpoint` 没有绑定对应训练人口。队友融合脚本会加载视觉和运动检查点，适配脚本还会通过 summary 间接寻找祖先。

最小修订：选参链的全部监督训练祖先只 fit train12，development2 仅预测/选择；最终 refit 链全部对应监督祖先 fit refit14。冻结已选配方，不沿用选择阶段检查点作为完整 refit 的替代物。A9 可有另行声明的 final4 无标签适配祖先。每次加载递归验证祖先有向图（ancestor DAG）的 stage、fit/select/predict/adaptation 用户和内容哈希。

### L4 / P1：样本对齐与缺失回退没有实现契约

证据：[P720](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:720) 直接堆叠概率，P1004–1006 按位置配对标签和预测。609行不等于同一609个样本、同一顺序或同一40类列顺序。

最小修订：概率、mask、元数据、伪标签必须附样本 ID 和类别列映射；连接前检查唯一性与精确集合，再显式重排。增加乱序、重复、缺行和同长度错集合验收。

[P724](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:724) 在所有可用性 mask 为0时返回全零向量。主审用最小数组复现，行和为0，违反概率与先验回退契约。须明确 select 先验 fit train12、final 先验 fit refit14，所有候选补齐609条且行和为1；按现有规格，18条应保留冻结先验。

### L5 / P2：跨样本连接与拟合边界只停留在文字

证据：P857–880 接收任意 sessions/candidate graph。队友 [audit_p87_sequence_decoder.py:123](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/audit_p87_sequence_decoder.py:123) 支持 `known_user` 与 `anonymous_date`；后者可能按日期混合不同用户。

最小修订：明确采用的分组函数和连接边界，禁止跨训练/开发/最终分区的边；transition 的两端必须属于允许拟合人口。A8/A9 的阈值、筛选和停止规则均在 dev 冻结，不能查看 final 预测后手工挑选。

### L6 / P1：冻结文件哈希不等于验证实验来源

证据：[P245](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:245) 只验协议和自身文件；[P994](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:994) 只冻结任意文件映射，未验祖先、必需候选、用户、样本集合或类别列。P911–914 只按名字禁止旧 bank；改名不能证明来源合规。P1075–1084 没有揭示后的单向状态。

最小修订：A9 目标必须可追溯为本 run、本 protocol 的 final4 A8；冻结递归祖先、候选集合和样本契约。持久化 `generating → frozen → revealed` 状态，揭示后禁止新训练、新候选或重新冻结；可以重新展示同一冻结实验的结果。上述是实验标签边界验收，不是通用软件安全审查。

## 工程独立审计（Engineering Review）

### E1 / P1：P28/P29 清单不兼容

P203–223 的规范清单缺 `depth_color_usable/ir_usable/skeleton_usable`；队友 [audit_yolo11_pose_skeleton.py:161](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/audit_yolo11_pose_skeleton.py:161) 用这些列筛样本，缺失会得到空集。该文件399–403的 `safe_name` 还要求三个 `/` 分段，规范 ID 格式不符合。P28读取 `class_name`；P29按 `class_id` 排序并输出类别。

修订：定义规范 ID/source ID 映射、绝对原始路径和可用性模式（schema），移植标签可选的读取/排序/摘要；保留609条缺失人口。

### E2 / P1：A1 调用了 Detail21 专用抽取器

[P370](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:370) 调用 P46；实际 [build_p46_videomae_multiclip_cache.py:53](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/build_p46_videomae_multiclip_cache.py:53) 先筛 `detail_selected` 并强制1384行，170–172之后才应用 `max-trials`。它读取 `class_id` 并输出真实标签。发展388行、refit2427行和final609行都无法原样接入。

修订：复用准备样本/模型编码函数，重写任意行数、标签可选的提取与聚合。不能简单换成 P85 全40类脚本；它强制2914行/18用户并允许历史缓存复用。

### E3 / P1：公开权重取得和本地模型路径分支缺失

P371 的 `snapshot_download(..., local_files_only=True)` 在当前缺少指定 VideoMAE 仓库/修订的缓存上失败。即使下载后，P372把本地路径作为 `--model`，旧 P46 在175–181又将其当 Hub 仓库名下载；工程审计的仓库名静态验证（repo-ID validation）产生 `HFValidationError`。当前默认 torch 缓存也未见 MC3 权重。

修订：单列公开权重获取、修订（revision）和 SHA256 校验；提取器区分本地目录与 Hub ID。明确 YOLO 文件路径/哈希与 MC3 权重路径/哈希。

### E4 / P1：缺独立像素缓存、固定划分训练器及完整教师目标契约

P415–459 消费像素缓存（pixel cache），但未落实构建任务。实际 [build_p86_visual_pixel_cache.py:38](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/build_p86_visual_pixel_cache.py:38) 强制2914行，要求 source ID/class ID；其83–96定义 `images.npy [N,2,T,3,R,R]`、view mask/quality、`source_frame_indices.npy`、completed和rows文件。

旧视觉训练入口先调用历史 `split_universe`，要求2914行和 OOF 字段。已有固定用户模式只跑固定预算后评估，不实现计划的开发集选 epoch。`P86VisualPixelDataset` 又要求四种 logits 与 labels/users/folds；计划只承诺选中 Ridge 的主概率，不能直接接这个 loader。

sequence 构建器也实例化有标签 Dataset 并写类别；P458未传教师路径会回落旧默认。修订必须包含标签可选推理 Dataset、独立像素构建、teacher-target schema、固定划分 select/refit 两种训练循环。复用模型/增强/损失可以，不能只转发旧 CLI。

### E5 / P1：运动缓存参数及时间对齐父依赖错误

[P513](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:513) 的 `--p29-run` 实际不存在；[build_p31_skeleton_imu_cache.py:41](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/build_p31_skeleton_imu_cache.py:41) 接受 `--p28-run`，109–113读取P28的原始帧与骨骼缓存。

[P525](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:525) 的 `--sequence-cache/--normalization` 实际不存在；[build_p86_motion_window_cache.py:33](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/build_p86_motion_window_cache.py:33) 消费 `--pixel-cache/--p31-run`，使用 source frame indices 和真实帧时间对齐，并没有使用计划中的 `NormalizationState`。

修订：改父依赖为P28、像素帧索引；显式实现归一化拟合/应用。可复用运动输出 Skeleton `[N,2,T,17,13]`、IMU `[N,2,T,5,P,16]`及mask。

### E6 / P1：MoBind 入口仍绑定历史人口与固定预算

P648–667、P782–808 的 command wrappers不能绕过源码入口：pretrain/fusion 强制历史1497/973/444计数，refit强制2914行，用户过滤在这些检查之后。现有入口不输出计划所消费的统一 `selection.json`。

修订：移植固定划分选择与refit循环，复用模型和损失；显式实现选择结果及检查点模式。P638承诺的 `enabled_modalities` 也不是原 `P86MoBindLite.forward(motion, mask_ratio=0.0)` 接口，须定义新增适配行为。

### E7 / P1：A4 输出不能接 A5 教师加载器

P594–598 输出 `sample_ids/probabilities/availability`，实际 [p86_mobind_lite_data.py:100](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/p86_mobind_lite_data.py:100) 消费 `imu_logits`或`logits`，可用性字段叫 `valid`。只补logits但不映射mask会将缺失IMU当有效目标。

修订：统一概率/logit转换、40类列顺序和valid mask，加入A4产物通过实际A5 loader的接口验收。

### E8 / P1：A9 绑定历史 stage 和401/405目标，且漏sequence输入

P905–939的输入缺sequence。实际 [adapt_p87s_test_student.py:89](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/adapt_p87s_test_student.py:89) 要求旧stage，99–104消费sequence/motion/pixel，105–114要求特定structured target字段和401或405个有效目标。

修订：移植标签无关适配训练器；声明新 `AdaptationInputs`，包含sequence和609 ID的固定目标模式、可用mask及18 fallback；按祖先契约验证 A7/A8。给旧stage改名不足以修复。

### E9 / P2：时间戳修复未进入实际调用链

P302–309定义helper，但P314–315仍执行原模块；旧 `frame_map` 会将Skeleton时间戳改成counter，P28直接求多模态key交集，可能产生 zero-common skip。原提交 `code/stage_runner.py:15–45` 是通过实际模块替换修复。

修订：明确移植的函数和执行入口，验证真实时间戳样本。不能假设父进程定义helper会影响subprocess。

### E10 / P2：协议哈希与共享签名不一致

[P112](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02/implementation-plan-v1.md:112) 以 `default=str` 序列化 `frozenset(users)`。主审在相同输入、`PYTHONHASHSEED=1/2/3`三个子进程得到三个不同序列化文本。`sort_keys=True`不排序集合，续跑可能误报协议变化。修订为递归规范化、用户排序数组，并纳入所有seed/config/weights身份。

P56–60测试的 `train_users`等属性不在P88–109 dataclass中；P1050测试调用参数与P1069函数签名不一致；smoke止于adaptation40但P1094宣称已冻结预测。先统一类型、签名和验收终点。待实现的函数尚不存在，本身不构成缺陷；承诺直接复用却与实际接口冲突，才是本轮阻塞。

## 本机证据与验证限度（Local Evidence / Limits）

- 主审核验队友快照全部 **1,167 文件**：大小和SHA256均匹配。[source_manifest.json](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/source_manifest.json) 的SHA256为 `a80a20a0f6fe23a5b3288f51a9e4a02daf83157c4b6a598bfb6fa0ec8b73649a`。
- 工程审计实测环境：Python3.12.9、PyTorch2.7.0+cu128、torchvision0.22.0+cu128，CUDA可用；RTX5060 Laptop GPU约7.96GiB。计划所列主要依赖可以发现。无需仅因Windows/PyTorch版本重建环境。
- D盘当前约46.90GiB、C盘约52.54GiB空闲。20GiB预检门槛已满足；完整峰值仍需估计。若四分区各自建立全量像素文件，仅images.npy理论约12.50GiB，还未计其他缓存和多分支检查点。
- 指定VideoMAE修订与MC3默认权重未在工程审计检查的默认缓存中发现；这不证明全机器没有其他副本，实施必须明确取得或指定已验证路径。
- 未执行真实训练批次，不能据环境存在就断言整条流水线跑通；10–20 GPU小时仍是未验证预算。
- 本轮没有证据证明队友作弊或已发生新实验训练泄露。发现的是未来实现会失败或违背声明的具体计划缺口。

## 统计谬误前瞻筛查（Prospective Fallacy Scan）

覆盖11/11类别；本实验尚无新结果，因此下表是方案风险筛查，不是统计结果验证。

| 类型 | 本轮状态与处理 |
|---|---|
| 辛普森悖论（Simpson's paradox） | 无结果可核验；保留总体与逐用户指标 |
| 生态谬误（Ecological fallacy） | 不由四用户有限样本推断所有跨用户总体 |
| 伯克森选择偏差（Berkson's paradox） | 无结果证据；固定人口和可用性子组均须预定义 |
| 碰撞变量偏差（Collider bias） | 无已发生证据；不要依据final正确性筛选样本 |
| 基率忽略（Base rate neglect） | 固定40类macro-F1、逐类与先验回退需明确 |
| 回归均值（Regression to the mean） | 无前后数据可验证；历史“最佳模型”不能证明稳定增益 |
| 幸存者偏差（Survivorship bias） | 18条缺失不得丢弃；全609分母是必须契约 |
| 多处搜索效应（Look-elsewhere effect） | 所有候选冻结后报告；只作描述性比较 |
| 研究者自由度（Garden of forking paths） | 适配池、后处理、候选和端点在揭示前冻结 |
| 相关与因果混淆（Correlation versus causation） | M1/M2/M4需修正对照及阶段差值解释 |
| 反向因果（Reverse causality） | 当前无相关结果，非本轮主要风险 |

## 继续推进的顺序（Next Sequence）

1. 修订 H/P/S：更新空间状态，把尚未建立的隔离改成验收要求；落实M1–M3，并收窄M4解释。
2. 重写输入输出与祖先契约：标签无关final清单、规范样本ID、40类映射、监督fit与无标签adaptation人口、统一fallback、冻结/揭示状态。
3. 将工程任务明确写成“复用模型和函数、移植数据与固定划分训练器”，补像素缓存、公共权重、A4→A5、A9 sequence和target schema；删除不存在的旧CLI参数。
4. 只针对修订项复审；通过后先做预检和真实一条/一批接口验收，再做完整训练。无需重新审查已解决的OOF、class25和旧表示匹配问题。

本报告完成的是独立审计，不是计划修订或训练验收。上轮NO-GO不会仅因清理了磁盘空间而自动解除。

## 2026-10-02文档修订记录（Documentation Revision Record）

用户随后授权“请根据报告进行修改”。已修订[handoff v2](C:/Users/LaiWanzhou/AppData/Local/Temp/cuhkx_single_teacher_pipeline_handoff_2026-09-23.md)、[实施计划v2](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md)与[规格v2](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md)。正文语言改为中文，专业术语保留英文。

M1–M4、L1–L6、E1–E10均有明确修订任务/断言，映射在v2计划末尾。旧CLI调用已改为模型/算子复用和新数据/循环移植；新增公开权重、pixel/sequence、A4 targets、固定select/refit祖先、mask/ID/prior及单向揭示契约。A9保留raw输出，并将A9−A8改为部署替换差值。独立训练对照补齐，ROI/腐败不再声称单独因果贡献。

新增的缓存一致性要求：A7/A9冻结视觉backbone时同时冻结BatchNorm运行统计，保持eval模式；实际运行验收检查参数和buffer哈希。已知时间的全缺失行断开transition；日期/时间未知的行不建边且不猜测位置，其他行仅按可观测时间形成会话。

原计划与规格输入快照SHA256与初审值一致；交接也保存原版。v2是未提交工作区文档，代码Tasks1–14仍未实现。本轮完成的是文档修订与静态自检，不是训练通过或新独立复审。没有模型下载、训练、commit或push。

修订时额外核验了源码配方：P46和P86 pixel的WINDOW_BOUNDS均为(0.0,0.70)/(0.30,1.0)，v2三类窗口已同步。run_p87s_final_pipeline使用hybrid模式；train_p86_visual_pixel_oof仅在feature模式启用直接feature loss及projection。v2因此将feature-weight0.5的配置值与hybrid的实际零feature loss分开记录，保留CE/KD/relation，避免无意新增训练操作。MoBind A5的视觉feature对齐保持生效。

A8移植细节也按源码固定：保留start/end、bigram和backed-off trigram，复用decode_unique_beam_posterior的beam边缘概率而非另写简化一阶解码。beam50/posterior温度1、weight .25/.30/.35与backoff1/2/5明确记录；>40或无合法路径整会话回退，known_user分组及置信门控是固定划分移植策略，不借助final标签处理不适用会话。

修订静态验证：14个任务、17个规格章节及20个审计项映射完整；Python接口代码块语法可解析，43个本地链接/证据行号有效，handoff中的v2文档哈希吻合，git diff --check通过，队友1167文件再次核验一致。验证记录见[文档修订验证JSON](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-02-handoff-plan-revision-verification.json)。这些检查不包含新流水线运行测试或新独立子智能体复审。
