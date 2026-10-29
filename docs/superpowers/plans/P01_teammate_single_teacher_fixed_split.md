# P01 队友单大视觉教师固定划分实施计划（Fixed-Split Source-Parity Implementation Plan）

> 本分支唯一有效计划（Canonical Plan）；唯一交接入口为 [P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。旧实验计划及交接已移入 docs/archive，仅作为历史记录（Historical Records）；reports/audit_inputs 为审计输入快照，不指导当前执行。

> 执行者使用 superpowers:executing-plans；用户明确选择子智能体实施时才使用 subagent-driven-development。2026-10-06独立复审修订为v3.1，Task编号仍为1–14。初版v3的机制producer、KNN依赖和元数据/导入/角色/校验契约已在下文纠正；正式生成仍受逐阶段数值与依赖闭合门槛约束。v2 Task1–5已完成并推送6e31b6d；它们是历史基线，不表示新增v3要求已完成。2026-10-28 Task1已实现并独立复审GO；Task2准备与Task3几何一致性均已完成并获独立复审GO，Task4完整产物生成并通过有限独立验收GO；Task5固定16轮select/refit及四区预测／序列均已完成，有限独立复审GO；2026-10-29 Task6运动缓存／恒等外部转换已实现、正式生成及有限自验通过，未独立复审；Task7源RF教师已训练并完成四区双profile目标、有限自验通过，未独立复审；Task8–14未实施，未训练大视觉骨干。旧规格/计划见reports/audit_inputs/2026-10-04/pre-v3/。

**目标（Goal）：** 一个大视觉教师、固定用户划分，在其余生效操作上按最终提交源码复刻完整路线。
**架构（Architecture）：** 复用源模型/算子，按新人口重新拟合；除教师根节点数量与固定划分替代OOF外，不随意增删归一化、掩码、温度、预算、后处理或适配参数。所有必要数据适配单独登记。
**技术栈（Tech Stack）：** 当前Python3.12、PyTorch2.7/CUDA12.8、torchvision、NumPy、scikit-learn、OpenCV、pytest。
**规格（Specification）：** [v3规格](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md)。
**差异清单（Difference Inventory）：** 以本计划下方“2026-10-29再核对”作为当前结论；[源码一致性修订报告](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-04-plan-v3-source-parity-revision.md)保留历史修订背景，不覆盖当前实现与新增发现。

## 2026-10-29再核对：一致性、例外与蒸馏差距（Source-Parity Recheck）

本次按用户指定的最终包 `D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)` 只读核对，修改文档，不改训练源码、YAML、协议身份、已登记产物或历史独立复审报告，不重新训练，不进入Task6。结论是：**Task1–5工程交付完成，视觉主干的生效模型、损失与优化配方已有源对照；仍不能宣称除两项用户例外外，整个训练链已经完全一致。** 原因包括固定划分的连带变化、下列精度差异，以及尚未实施的Task6–14。当前唯一计划仍是P01，不另建plan或handoff。

### 核对证据与范围（Evidence and Scope）

最终包的[训练说明](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission%281%29/docs/TRAINING_PIPELINE.md)指定基础训练入口为 `run_p87s_final_pipeline.py`，最终适配为 `adapt_p87s_test_student.py` 的实际40轮命令。本次对23个相关小源码文件逐文件比较最终包与本分支使用的队友快照，21个字节相同，两个不同：

- `p89_build_multiexpert_test_logits.py`：最终包删除未启用的 `distill_missing_p46` 两个元专家及其main调用；AST比较确认现存 `build_p86` 等函数不变。当前十机制使用同一VideoMAE固定Ridge数学，不调用历史main或复活被删元专家。
- `p310_union_repeat_precedence_teacher.py`：AST比较main只发现报告中删掉对两个旧候选CSV的比较；生成目标的覆盖规则和 `0.9805/0.0005` 表达式不变。Task11移植最终包的有效规则，不执行快照main中的旧报告读取。

因此，“文件不完全相同”与“生效算子不同”分别登记；不能声称全快照逐字节等于最终包，也不需要改旧source pins来绕过身份校验。本次没有全扫1167文件，没有读取旧教师预测或final标签。过去真实batch的loss／梯度／anchor差0是同输入、同运行环境下的算子证据，不证明历史队友训练轨迹逐位重现。

### 已核实的视觉处理（Verified Visual Operations）

| 操作（Operation） | 最终包的生效行为 | 当前v3状态与证据 |
|---|---|---|
| 时间窗与视野（Windows / Views） | early `[0,.7]`、late `[.3,1]`；scene/person/workspace；16帧、160像素，person/workspace裁剪放大1.15/1.40 | 相同算子和参数；Task2/3已复用与抽样核验几何／pixels；不外推全部历史像素逐值一致 |
| 教师表示（Teacher Representation） | 冻结VideoMAE-Large，逐clip L2、六族特征、StandardScaler→Ridge，原12格/族与决胜规则 | Task4六族共72候选已实现；选参人口与温度估计人口的差异见下表 |
| 基础学生目标（Base Student Targets） | `P86VisualPixelDataset`实际读取 `early_late_logits`，relation消费同六clip教师features；此阶段不是直接读取最终30专家投票目标 | 当前固定early_late校准目标与原始六clip特征；未用最终A8、MC3机制或window_mean替代 |
| 学生结构与初始化（Student Architecture / Initialization） | 公共Kinetics MC3-18 temporal、width512、dropout.18、gated、冻结至layer2及源BatchNorm行为；额外时空候选开关关闭 | 使用同源模型类／公开初始化；Task5已完成。工程上显式装载公共权重，不使用队友训练checkpoint |
| 混合损失（Hybrid Loss） | `CE(label_smoothing=.1) + 1.0·T_KD²·KL + .2·relation`，`T_KD=2`；relation为六clip余弦Gram关系匹配 | 同一输入loss及参与梯度已验收差0；class_weights幂.35、有效mask与归约按源 |
| 未启用损失（Inactive Losses） | 虽命令传 `feature_weight=.5`，但 `mode=hybrid` 时直接feature loss为0；stage distillation weight默认0 | 当前有效权重均0，符合实际分支；不能把增开feature／stage KD当作“补回原方法” |
| 增强和优化（Augmentation / Optimization） | 原subject_robust六clip一致增强，AdamW分组head2e-4/backbone1e-5、wd.08、2轮warmup＋cosine、batch4/accum4、AMP、clip-grad2、singleton策略 | 原函数／同数学；select/refit各16轮末态，公开重新初始化；训练样本与批次顺序差异另记 |
| 序列与锚点（Native Sequence / Anchor） | 本学生的原生layer4序列、同学生anchor用于后续融合 | 四区同phase输出已生成；序列float16 `[N,2,3,16,512]`，anchor与本阶段预测一致 |

直接证据入口：[学生目标与增强](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission%281%29/code/training/project/aligned_multimodal/p86_visual_pixel_data.py:39)、[最终基础训练命令](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission%281%29/code/training/project/aligned_multimodal/run_p87s_final_pipeline.py:164)、[源学生训练与损失](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission%281%29/code/training/project/aligned_multimodal/train_p86_visual_pixel_oof.py:426)、[本轮学生实现](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student_v3.py:55)。

### 明确保留的差异及其连带影响（Explicit Differences and Consequences）

| 类别（Category） | 队友原方法 | 当前实验 | 解释与约束（Interpretation / Constraint） |
|---|---|---|---|
| 用户要求：大教师根数量（Large Teacher Roots） | 最终专家链含多个大视觉根及派生头，不等于30个独立大骨干 | 只保留VideoMAE-Large根；同根六头／机制、合法非大教师按roster保留 | 专家列数、分组统计和后处理输入会自然改变；不能靠重复列／旧预测凑回原结果 |
| 用户要求：划分（Fixed Split vs OOF） | P85头按三折用户OOF选参与生成训练目标；最终学生另走固定预算all-label refit | select训练train12，user6/7作为development2；refit14用于封存final4 | select的val结果与refit的训练结果分开；不引入内部OOF，也不把队友最终官方test分数当此处val基线 |
| 固定划分的连带变化：目标（In-sample Targets） | 学生读取校准后的OOF `early_late_logits`，每行预测来自未fit该行所在折的头 | 学生训练行读取本phase教师的in-sample目标 | 没有使用val/final真值去fit教师，但样本内目标通常更容易拟合，不能宣称等价于源OOF监督；这是实验设计的已知限制，不据此自行加OOF |
| 固定划分的连带变化：头校准（Head Calibration） | `fit_temperature(raw_oof, labels)`后用同一T输出OOF和全训练refit头的test logits | select在train12拟合预测估T；refit在refit14重新估T | 算子及范围相同，统计来源与T继承方式不同；由不生成OOF和不使用held标签校准的边界决定，不称温度数值复现。当前T分别.1206300185/.1397821300 |
| 固定划分的连带变化：选参（Hyperparameter Selection） | 每族12格用全OOF指标选定 | 每族12格在固定development2选择；主family仍固定early_late | user6/7是开发集而非独立test；这些标签已用于头选择，不能称整条路线从未用val选择，也不能把反复看val当新的无偏证据 |
| 数据适配：样本覆盖（Coverage / Missing Inputs） | 源脚本断言2914行；原学生输入基于其历史manifest/cache | 本实验固定完整2039/388/2427/609；有效IR1957/385/2342/591；无IR不进视觉训练，输出按同phase先验补齐 | 2914不能机械硬编码到新划分。此前已保留时间戳恢复和ROI回退；本次未逐行取得队友历史清单做对应差分，不声称覆盖差异只来自用户人数 |
| 数据适配：ID与批次（IDs / Batch Order） | 原sample_id排序及原manifest位置 | 无标签opaque ID排序后按当前人口shuffle；seed相同 | 防止ID编码类别；但同seed不保证原样本顺序、batch组成、随机增强轨迹相同；不是额外网络设计，不宣称训练轨迹等价 |
| 额外诊断（Additional Controls） | 最终包主链不是本实验16候选对照矩阵 | A6平均、mask/shuffle/zero控制及独立12轮适配只作诊断 | 不得回流主监督、按更高val替换主配方，或混称源最终操作 |

当前 `T_head<1` 会先锐化Ridge分数，之后KD再用2软化：学生目标是 `softmax(raw / T_head / 2)`。相对raw分数的有效分母约为select `.24126`、refit `.27956`，不能只看“KD温度2”就把目标称为较软监督。没有队友原OOF温度／目标的本轮数值对照，不能断言谁更软或据此给出调温建议。

### 新发现与尚未闭合的差异（Residual Differences / Open Evidence）

1. **ROI质量值精度（ROI Quality Precision，未要求的数值差异）**：源[像素缓存](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission%281%29/code/training/project/aligned_multimodal/build_p86_visual_pixel_cache.py:71)把 `view_quality` 写为FP16，[Dataset](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission%281%29/code/training/project/aligned_multimodal/p86_visual_pixel_data.py:145)再转FP32；当前[pixel_cache](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/pixel_cache.py:62)／导入缓存直接保留FP32，[Dataset](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/no_vote_datasets.py:39)未补源FP16舍入。只读mmap抽查refit14中前三条有效train12行（位置0/1/3），每条最大舍入差均 `0.000244140625`，clip时间均值差分别约 `7.63e-5/4.58e-5/9.16e-5`。它进入源quality gate，不能保证数值全等；**没有证据把它解释为22个百分点差距的原因**。bool/uint8 mask及int64/int32索引仅存储类型不同、对应值语义相同，应与浮点舍入分开记录。
   - [ ] 严格源精度闭合：若后续要求消除此差异，应先用fixture验证源FP16→FP32量化及quality gate，再用新producer／新产物身份应用于训练与推理；不得覆盖硬链接缓存、只改推理、或给已训练checkpoint追认一致。需要重训A2及受影响后代时另立版本并记录成本。本次只登记，不执行；Task6不消费quality，不因此阻断其独立运动算子工作，但最终Task14严格一致性结论前必须明确解决或显式接受该残余差异。
2. **运行环境（Runtime Environment，非历史训练环境证据）**：当前torch2.7.0+cu128/torchvision.22.0、NumPy1.26.4、sklearn1.6.1、SciPy1.15.3、OpenCV4.10.0、Transformers4.49.0；最终包requirements-lock为torch2.9.1+cu130/torchvision.24.1、NumPy2.4.4、sklearn1.8.0、SciPy1.17.1、OpenCV4.13.0.92、Transformers5.8.1。该锁文件自己注明是干净推理验收环境，**不证明队友当时训练也用了这些版本**。已恢复VideoMAE旧注意力偏置并有局部源对照；不能声称跨版本全程bitwise一致，也不能无证据归因于环境或现在无条件重装／重提取。
3. **后续链尚未实施（Pending Chain）**：Task6运动缓存、Task7 RF、Task8多模态预训练、Task10融合、Task11结构化目标与Task12适配还没有当前v3正式产物。P12 S+D／Thermal最终配置、公开资产与补充清单、P89 safe-base/A18 producer证据仍缺；全局 `formal_generation_ready=false` 保留。已完成的A2-only局部GO不放行这些环节。不能以“计划写了相同”代替实际实现／验收，也不能任意换A5或A7填补缺producer。
4. **局部复审结论的范围（Review Scope）**：P28/P30保留其当时已验收的工程、源配方和有限数值证据；本次补充发现不回写已被完成标记绑定的历史报告。Task5的工程Complete与严格输入精度仍有残余差异可以同时成立，后续报告必须同时说明，不能继续无条件写“除单教师／固定划分以外全部完全相同”。

### 队友用于知识转移的处理及其当前位置（Knowledge-Transfer Mechanisms）

| 处理（Mechanism） | 作用与源事实 | 当前状态及不能声称的结论 |
|---|---|---|
| logits KD＋关系蒸馏（Logit / Relational KD） | T²·KL匹配类别分布，relation匹配六clip相似关系；避免只靠硬标签，允许学生／教师特征维数不同 | Task5已按源执行；不是缺了feature=.5才低分。具体泛化增益仍需消融，不能仅凭公式保证减少acc损失 |
| 校准、增强、冻结与混合监督（Calibration / Regularization） | 原头温度、subject_robust、CE smoothing、低层/BN冻结与小backbone LR共同控制监督和更新 | 已按源算子／预算恢复，校准统计来源不同；增强与正则不保证本划分val提升，不按当前分数追加新配方 |
| 跨模态对齐与监督（Cross-modal Alignment） | 源MoBind含时间token／身体局部／全局对比、重建、视觉KD／特征转移和RF监督；只保留源实际mask与权重 | Task8待实施；不能把未来S/I信息当作Task5已取得的增益 |
| 选择性视觉锚定（Selective Visual Anchor） | 源 `selective_anchor_loss` 只在训练标签判断anchor正确的行约束fusion接近anchor；阶段A只动motion residual，B再开visual heads | Task10待实施；是训练监督下的保留视觉能力约束，不在val/final用真值决定路由，也不保证所有行无harm |
| 结构化教师与末端适配（Structured Targets / Final Adaptation） | 源重复/分组/序列/precedence改善目标；P310选定类.9805、其余.0005，40轮训练heads_motion_encoder，视觉head LR5e-5、motion/fusion1e-4 | Task11/12待实施，目标锐化不是温度软化；conf_power0不进行额外置信度筛行。实际效果取决于新目标质量，不能把队友0.91归因于某一个模块 |

区分**优化损失（Optimization Loss）**和**准确率转移差距（Accuracy Transfer Gap）**：当前select训练日志首轮→16轮KD为 `10.25695→0.25500`，总loss `13.27275→1.66318`；refit KD为 `8.89442→0.26416`。但固定user6/7上A1为268/388=69.07%，A2为182/388=46.91%，相差86条／22.16个百分点；这不是KL损失的单位。样本内有效IR教师1957/1957、学生1945/1957说明训练拟合很高，不能因此断言跨用户知识转移成功，也不能从这一个结果识别唯一原因。旧v2学生48.71%与本轮46.91%还同时改变预算／选择规则／目标校准，不是单因素消融。

**后续执行约束（Execution Decision）：** 保留已授权的单大教师／固定划分，不加内部OOF、不按val重选温度或轮数；先如实保留上述差异，再按各Task移植其有效算子。Task5完成的是基础视觉学生，Task10和12仍会按源训练视觉头，不能称整个最终视觉参数已经冻结。本次没有新的性能消融或独立复审，不重跑模型、全cache或历史测试；任何未解释差异都要在Task14最终一致性声明中显式处理。

## 全局边界（Global Boundaries）

- 工作树D:/work/2026.7.14_kaggle/_single_visual_processing_replication；分支experiment/teammate-single-teacher-task1。原始数据、队友快照与两份提交包只读。
- train12=2039，val user6/user7=388，refit14=2427，final4=609。refit包含val，不能把refit在user6/7上的成绩称val。
- 保留固定划分，不执行三折OOF，也不生成内部OOF。训练目标/温度/门控统计只来自同phase fit数据。不得读取旧OOF/专家预测、按final成绩调参或扩大fit用户。
- 独立子智能体经用户2026-10-04限定授权，仅检查final4类别覆盖，返回“联合覆盖40类=是”；未传回标签或分布，未计算模型成绩。此例外仅为覆盖核实，不给生成进程任何标签授权。仍不假定每名用户40类。
- v3 run_id=fixed-split-single-teacher-v3-source-ops；v2配置及模型保留。复用标签无关缓存需要新run记录、精确ID/轴/配置及来源，不改旧identity；监督模型/targets/学生sequence必须按v3重新生成。
- 大视觉教师根节点只保留VideoMAE-Large。保留同骨干六族头/全窗邻居、VideoMAE固定Ridge机制预测及独立非大视觉专家；teacher_roster对依赖被删根节点的排除逐项说明，不以A5、先验或重复列伪装原专家。
- source truth=最终(1)包实际命令/部署配置→调用路径的源码默认值。快照六个主文件已逐字节一致，其他新消费模块按source_ops_manifest直接核验，不全扫1167文件。
- A6、A7训练controls和12轮适配是额外诊断；不能替代主链或改变主链预算。主端点为最终配方A9-40。
- 保留16候选与最终一次揭示；Task1–2准备已按用户授权执行，后续训练仍需满足源证据与逐阶段门槛。

## 2026-10-29补充：完整知识转移的评估契约（Knowledge-Transfer Evaluation Contract）

用户已批准本节修订。仅补充Task11–13的输出证据、比较指标和解释；Task编号仍为1–14，主端点仍为固定40轮，原主比较 `A9-40−A1` 和16候选名单保持不变。不新增OOF、user6/user7伪标签适配、温度搜索或最佳epoch选择。

### 同人口比较与解释（Matched-population Contrasts）

以下准确率差（Accuracy Difference）均在Task13一次揭示后，按**同一refit阶段、同一final4完整609个ID、同一40类顺序、同一全缺失先验策略**计算，单位为百分点（Percentage Points，`100 × (acc_left−acc_right)`）。不把select模型的user6/7成绩与refit模型的final4成绩相减。完整分母是主报告；按公共 `A8.target_mask` 划分的591条适配行与18条先验回退行另作明确分母的分层报告，不按正确与否选行。

| 比较（Contrast） | 目的与解释（Purpose / Interpretation） |
|---|---|
| `A9-40−A1` | 保留原主比较，描述单大教师条件下完整路线相对基础视觉教师的总体差值 |
| `A8−A1` | 描述保留专家、融合与结构化处理后的最终教师相对基础教师的差值；是组合处理的比较，不单独归因于某个算子 |
| `A9-40−A8` | 直接衡量适配学生相对它实际学习的结构化教师的准确率转移差距（Accuracy Transfer Gap）；负值表示准确率下降，正值表示学生更高 |
| `A9-40−A7` | 描述同一基础融合学生经固定40轮伪标签适配后的准确率差值 |

上述补充是现有候选间的配对诊断（Paired Diagnostics），不新增模型训练或改变主比较。准确率相同不代表逐行预测相同；分别报告正确数、百分点差、相对参考的纠错数（Rescue）和新增错误数（Harm），避免抵消后误称“完全复现教师”。单次、单seed结果只作当前人口的描述，不保证其他新用户或新数据同样无损。

### 无标签一致率与阶段区分（Label-free Agreement and Stage Semantics）

Task12固定端点训练完成后，用 `eval()`、关闭数据增强（Augmentation）、同phase的原始学生推理输出与**真正进入训练的A8伪标签argmax**比较。既不对A9再套A8后处理，也不把含时间增强的训练日志 `student_target_agreement` 当最终一致率。对40轮主端点和已计划的独立12轮诊断分别输出：

- `teacher_student_agreement_all`：相同预测数／完整609行，同时记录分子、分母和disagreement数。
- `teacher_student_agreement_adapted`：相同预测数／A8.target_mask=true的591行；另报18条excluded-prior行的一致数／分母，防止共同prior抬高总体一致率。
- `teacher_student_agreement_before_adaptation`：同一人口上A7与A8的一致数／分母，采用同样分层；复用已生成的小预测表，不重复跑视觉骨干。
- 证据必须绑定A8目标、A7基础模型及对应A9端点／预测的ArtifactRef、精确ID／类顺序和目标mask；重复ID、缺行、不同phase或错误目标引用均拒绝。文件保存于该端点 `output/teacher_student_agreement.json`，无真值label、correct或accuracy字段。

**一致率不是准确率（Agreement Is Not Accuracy）**：无标签阶段只能说明学生是否复现教师的判断，不能给出真值准确率；100%一致不是验收必须达到的分数门槛。低于100%时如实保存差异，不能追加轮数、重选checkpoint、按一致率挑12或40端点，或混入教师预测直接覆盖学生以凑一致。最终准确率及rescue/harm只由Task13揭示后的评估器产生。

user6/7的select阶段成绩继续作为开发划分上的结果，不参与本次伪标签适配。Task12只按原计划对final4公共输入做无标签目标适配（Target Adaptation）；Task13报告明确该输入人口参与过适配，不将其称为适配期间完全未见的新输入验证。不能用本轮46.91%基础学生成绩推断完整流程后的最终蒸馏差距，也不能将目标输入上的匹配外推到另一批未适配样本。

### 已完成Task的复用（Reuse of Completed Tasks）

Task1–5保持完成状态：源操作／协议描述符、公共清单与缓存、姿态／ROI、Task4教师头／温度／四区targets／全窗特征，以及Task5学生checkpoint／四区预测／原生sequence和anchor均继续作为已登记依赖复用。本节不改变YAML、协议身份、producer源码或已训练目标，不重新生成这些产物，不回写P28/P30或已有完成标记；后续报告引用它们的原ArtifactRef。

新增指标在Task11–13实现时只消费已有或届时新生成的小预测表和mask，遵守现有有限校验成本约束。**此前ROI质量FP16舍入的残余差异仍独立待闭合**；这次评估修订本身不要求重训Task5。如果后续另行要求消除该精度差异，按前述严格源精度条款创建新版本并处理受影响后代，不能用“本次可复用”将旧checkpoint追认为严格数值全等。P12／Thermal／P89／A18材料门槛不变，当前仍从未启动的Task6往后推进；本次文档修改不启动训练或最终揭示。

## 校验成本约束（Verification Cost Bound）

沿用已修复的direct-metadata策略。取消raw全目录SHA、全images/features/sequence额外重读、祖先payload递归哈希和每Task全仓旧测试。实际消费的直接小产物/模型每阶段核验一次，Stage Verification Context缓存结果；祖先仅去重元数据遍历。大数组写入时计算摘要，load检查header/mask/ID及少量切片，不扫描整个数组。

每个Task只运行自己的针对性测试和计划要求的一条/一批真实train12样本；输入或实现未变则不重复。原始数据只读；轻量验收不声称验证全部未消费文件。CLI记录初始化/加载/计算/写出/登记耗时与批次进度。用小fixture的读取计数保证raw SHA0、祖先payload0、epoch/batch中资产复查0；不得用一次全量扫描证明“不全扫”。

## 复审重点（Review Focus）

1. 教师头温度与学生KD温度混淆：raw/T_head再除T_KD；T_head只fit训练人口，不能固定1或用val拟合。
2. 原算子未匹配：额外global z-score、强制缺模态CE/KD归零、错误budget/LR/parameter scope须由源函数数值与梯度对照拦截。
3. 专家bank伪装：缺原producer、不匹配ID或被删大根节点的假复制列必须拒绝，不偷读旧预测。
4. 固定划分边界：训练/校准/门控只用fit；val仅允许既定选择或诊断；final标签只给冻结后的评估器。
5. 续跑/校验性能：v2已学习产物不得混入v3；新producer变化不得改旧identity；重复加载与整缓存哈希探针应失败。

## 文件与接口（Files and Interfaces）

保持已实现NoVoteProtocol、ArtifactRef、RowIndex、StageInputs、Prediction、TeacherTargets、Selection等类型；CLI继续显式--config/phase/partition。不向旧loader补假folds、标签或2914计数。新的源码转换、路由与资产描述通过recipe/ArtifactRef纳入协议，不给v2 dataclass增加导致旧身份漂移的隐式默认字段。

| Task | 核心文件及职责 |
|---|---|
| 1 | no_vote_protocol.py/no_vote_weights.py/teammate_source.py；v3配置、source_ops_manifest与teacher_roster |
| 2 | no_vote_manifest.py/artifact_record.py；公共清单、标签隔离、精确缓存复用记录 |
| 3 | pose_roi_adapter.py；复用或按源生成P28/P29 |
| 4 | visual_teacher.py/visual_mechanism.py；六族Ridge、early_late主头、温度、同骨干全窗特征、固定Ridge机制预测 |
| 5 | visual_student.py/no_vote_datasets.py/pixel_cache.py；源16轮MC3、新targets/sequence |
| 6 | motion_cache.py；原P31/P86张量与Source Transform Descriptor |
| 7 | imu_rf_teacher.py；源RF、device dropout及fit预测目标 |
| 8 | mobind_pretrain.py；源24轮MoBind、原缺失损失语义 |
| 9 | simple_fusion.py；明确标注的额外平均/校准诊断 |
| 10 | mobind_fusion.py；主A7源full损失、固定4+20、独立controls |
| 11 | p12_source_adapter.py/source_teacher_bank.py/session_metadata.py/session_repeat.py；P12/全局融合/KNN/安全base、保留专家及原后处理 |
| 12 | target_adaptation.py；最终40轮及独立12轮诊断 |
| 13 | no_vote_evaluation.py；16候选冻结与一次评估 |
| 14 | run_teammate_single_teacher_pipeline.py；编排、成本预检、闭合smoke与报告 |

标准targets保持sample_ids/class_ids/logits/probabilities/valid；A1主目标为校准early_late，辅助六族头分别登记。TeacherTargets.features仍[N,2,3,1024]。sequence仍[N,2,3,16,512]+同phase anchor。motion保留原MOTION_FIELDS；normalization字段引用确定性源转换描述符，不执行跨样本z-score。AdaptationInputs仍显式提供base/targets/pixels/sequence/motion/转换描述符。

## 执行顺序（Execution Order）

所有新增步骤执行测试先失败→最小实现→针对性通过→真实小批次验收→提交当前实验分支。完成一个阶段后再处理依赖；不把伪随机toy输出当正式产物。Task1–3完成v3复用登记后，先重做Task4/5的监督目标和学生，再推进Task6及后续。正式teacher bank需完成保留producer，不能先生成简化A8。

### Task 1：锁定实际源操作与v3配置（Source Recipe and v3 Protocol）

**文件：** 修改no_vote_protocol.py/no_vote_weights.py/teammate_source.py，以及artifact_record.py的最小v3 descriptor注册支持；新建configs/experiments/teammate_single_teacher_fixed_split_v3.yaml、scripts/export_no_vote_source_ops.py、tests/test_no_vote_source_ops.py。descriptor支持必须在首次export返回ArtifactRef之前落地；其余完整角色表在Task2扩展，不把Task1写成必须先完成Task2才可交付。
**接口：** export_source_ops(source_root: Path, output: Path) -> ArtifactRef；resolve_teacher_roster(source_ops: ArtifactRef, retained_large_roots: tuple[str,...]) -> ArtifactRef。不执行旧main，不读历史预测。

- [x] RED：test_final_recipe_overrides_defaults断言MC3主预算16、MoBind24、fusion4+20、40轮visual LR5e-5/scope heads_motion_encoder、12轮scope heads；test_teacher_root_roster断言一个大视觉根、六头不计多个大教师、独立非大专家不被任意删除。
- [x] 从最终(1)命令及源调用导出所有生效参数，逐项附source path/symbol/line、声明是actual override或default。展开p88/p165/p173/P255/P307/P309依赖；被删根节点与依赖专家写明确原因。不替换不兼容输入，缺producer时给确定的失败清单。
- [x] 新v3配置包含原population、source pin、asset绑定、新run ID与本规格参数；公开teacher专用资产从source_roster单独登记。旧v2 YAML/recipe/identity不改。
- [x] pytest tests/test_no_vote_source_ops.py；小fixture只检查实际消费模块，不对全1167文件/公开资产重复下载或全量SHA。导出的recipe才是后续构造模型的唯一参数来源，不能另写一组隐式默认值。
- [x] 提交v3协议实现；本Task的文档修订不等于已完成该代码步骤。

**2026-10-28交付（Delivery）：** Task1实现、最终描述符/配置与CLI幂等导出通过，针对性68项测试通过，独立初审问题修复并实际复审GO。30槽为19保留/9排除/2未闭合。缺P12最终配置与P89/A18生产证据，formal_generation_ready=false继续阻止正式生成；不是所有producer已就绪。详见[Task1说明](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/task1_v3_source_ops.md)。Task2交付见P03。

### Task 2：v3准备与缓存复用（Preparation and Cache Import）

**文件：** 修改no_vote_manifest.py/artifact_record.py及对应CLI/测试。
**接口：** import_public_cache(source: ArtifactRef, destination_protocol: NoVoteProtocol, rows: RowIndex, context: VerificationContext) -> ArtifactRef；build_recording_metadata(inputs: StageInputs, acquisition: ArtifactRef) -> ArtifactRef；学习产物无导入入口。

- [x] RED：test_reuse_raw_only允许P28/P29、公开冻结features、pixels，拒绝v2 heads/targets/A2/sequence；test_import_exact_ids_axes拒绝缺ID/错用户/错窗口/错视野，旧identity不修改。
- [x] 复制前三分区允许标签及四公共清单的明确准备过程；final标签不读、不复制到run。不重新扫描原始数据。新增保留专家模态的公共字段按白名单声明，禁止class/action/trial标签捷径。
- [x] 导入固定采用同D盘硬链接（Hard Link）：新run创建payload路径；新登记的P28/P29/公共initializer/manifest父记录全部位于v3，旧ref只作为provenance字符串/旧digest凭据，不作为新ancestry父节点。按拓扑顺序导入并用memo复用公共祖先；禁止直接放宽registry外部路径或跨protocol规则。所有导入cache immutable、mmap只读，producer拒绝overwrite/r+写入。无法硬链接时明确失败，不偷偷复制多GB或重新提取。
- [x] 定义v3专用Artifact Role Matrix：新增descriptor（source_ops/roster/source_transform/recording_metadata，fit_users为空）、teacher_bank（绑定同phase专家预测、[N,E,40]、availability/fallback及roster）；保留source_expert/<name> supervised_model与predict绑定自己的模型阶段及完整分区；mechanism/六头/global_mixture/safe_base/group_old/group_new/gate有明确stage与父记录。未知stage预测仍拒绝，不能退回“必须adapted_model”旧默认或给所有stage豁免。P12等teacher-only公开资产在recipe资产绑定单独声明，不给v2协议增加隐式字段导致identity变化。
- [x] 新增公共recording_metadata.csv sidecar：sample_id/user_id/partition/recording_date/start_seconds/timestamp_available/time_source；start_seconds按原parse_time解析的日内秒(h×3600+m×60+s+microsecond/1e6)，不能直接传pixels中的epoch秒。从原时间内容或已恢复acquisition IDs提取，不从class/trial ID推断。精确ID连接，不使用旧metadata；未知保持date空/seconds NaN，后处理保留base。教师额外输入另行白名单，无label字段。
- [x] 在artifact_record.py定义公共VerificationContext(protocol)，cache validated_records/direct_files/loaded_arrays/import_memo；keys包含protocol/ref/path。Task4/6/11与其CLI显式共用该context，不能只缓存features。祖先集合去重，注册后消费不再重hash；六头一次加载features，旧大cache消费只走header/ID/mask/少量切片，不能落入_load_features现有full-SHA分支。
- [x] 新测试：test_import_hardlink_bridge_consumable、test_import_rejects_old_learned_parents、test_v3_expert_prediction_roles、test_recording_sidecar_label_free、test_six_heads_load_once、test_shared_ancestors_read_once、test_register_consume_no_rehash。小fixture直接消费完整导入链并验证raw SHA0，不用真实cache扫描。
- [x] 运行标签/角色/ID针对性测试；final覆盖只引用已授权子智能体“是”，不能让准备CLI打开test标签再次检查。

**2026-10-28交付（Delivery）：** 六组真实公共缓存及新祖先登记、四公共清单/三允许标签、四时间侧表已完成，CLI耗时120.41秒。95项针对性测试通过；descriptor父类型补充探针RED→GREEN，角色11项复跑通过。真实train12单条像素与六次特征共享验收通过，raw/权重完整SHA0；没有Task2独立复审。Thermal扩展白名单已定义，实际补充表/P12资产与配置仍待源证据闭合，未伪造缺失。详见[P03](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P03_Task2_准备与缓存复用.md)。

### Task 3：几何与姿态源一致性（Pose and ROI Parity）

**文件：** pose_roi_adapter.py及tests/test_no_vote_pose_roi.py。
**接口：** 沿用build_pose/build_roi(inputs, ..., protocol=...)。

- [x] 复用已验证的全量P28/P29；源ROI两轮、真实timestamp恢复、person/workspace裁剪不变。
- [x] 一条train12 fixture与真实样本核对frame/acquisition/time、ROI quality和缺失fallback；未变部分不重新生成2427/609条几何。
- [x] 仅修复v3导入契约的新失败，保留时间戳适配源码身份，禁止为原main兼容而读取类别路径。

**2026-10-28交付（Delivery）：** 新增精确P28/P29单样本消费接口及有界源算子加载；复用所有既有几何，不重跑检测器。真实train12一条22帧/14个ROI返回字段、源裁剪/质量/时间精确一致，8.614秒；5个直接源文件与最终提交包字节相同。15项初步测试通过；独立审查发现Task2元数据cache key被循环变量覆盖，修复并RED→GREEN，51项集成通过，审查者复验后Task2＋Task3 GO。详见[P24交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P24_Task3_姿态与ROI一致性.md)及[P25复审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P25_Task2_Task3_独立复审.md)。Task4未启动，formal_generation_ready=false不变。

### Task 4：恢复教师目标处理（Visual Teacher Rectification）

**文件：** visual_teacher.py、run_no_vote_visual_teacher.py；新增tests/test_no_vote_visual_teacher_source_parity.py。
**接口：** select_visual_head(train, development, features, protocol=...) -> Selection，主family固定early_late；fit_visual_head(...)->ArtifactRef；predict_visual_teacher使用模型自有T。辅助头/同骨干全窗特征另登记；fit_visual_mechanism_head(fit, features, protocol, context) -> ArtifactRef；predict_visual_mechanisms(model, features, rows, protocol, context) -> Mapping[str,Prediction]。

- [x] RED：test_source_feature_head_selection复用原六族、每族12配置及源tie-break；test_calibration_train_only断言改val/final标签不改变T，原fit_temperature与本实现数值一致；test_ridge_class_alignment复用原floor规则。
- [x] 复用六clip公开特征；72配置生产六族头，每族12配置按源决胜规则在固定val选择，主头固定early_late、不在六族间改主头。候选模型只fit train12，温度只fit train12预测/标签；refit继承已选配置并重fit refit14。训练软目标记录in_sample非OOF，选择指标记录development2，不混淆两者。
- [x] 恢复head温度约0.1–10的源NLL估计，输出raw/T_head；select只train12，refit只refit14。保存温度、目标来源、raw与calibrated分数，不借用旧全局校准。
- [x] Task4只生产同骨干full-window标签无关特征，按build_p85_videomae_fullwindow_cache，不用六clip平均替代。学习KNN依赖P12→全局混合，明确移到Task11；Task4不等待Task6/7/P12，防止执行循环。
- [x] RED：test_visual_mechanism_ridge_source_parity对照audit_p86_teacher_mechanisms.fixed_model_perturbations与p89_build_multiexpert_test_logits.build_p86：VideoMAE六clip逐clip L2、固定StandardScaler→Ridge(alpha3000,power0.75,lsqr/tol1e-5/max_iter5000)、fit归一化特征均值、原十项特征干预、aligned_scores_40。独立小头只fit同phase fit人口，输出raw logits/T=1，不套A1温度，不消费MC3/像素/学生sequence。检查mechanism祖先是固定Ridge而不是A2；此头不增加大骨干数量。
- [x] 源小型score/feature fixture逐元素验收，真实train12小片段记录目标熵/置信度；运行本Task测试。不得将置信度变高当作准确率提升的证明。
- [x] 正式重新登记六族头、A1主头、先验及四区targets。v2的window_mean/T1保持原状，不能改记录宣称已校准。


**阶段就绪说明（Stage Readiness）：** 按本Task及规格明确的不依赖P12/KNN原则，新增绑定source_ops、Task3证明与消费源码的visual_readiness描述符，仅放行A1/六头/class_prior/固定机制；不修改YAML身份或全局formal_generation_ready=false，也不放行后续bank/融合。实施前解释及成本问题修复获独立GO（P27）。正式68组targets/raw与102元数据记录已独立验收GO（P28）；主early_late user6/7=268/388，head温度select0.1206300185/refit0.1397821300。49项针对测试通过；真实全窗单条通过，完整3036条全窗编码已完成，ID/float16 shape/valid及小切片经独立验收GO（P28）；整个Task4完成。详见[P26](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P26_Task4_视觉教师目标恢复.md)。

**完成（Complete）：** 头部68组targets与完整全窗两个ArtifactRef均齐，49项针对测试通过。两区全窗真实编码2034.8秒＋319.0秒（约39.2分钟）；最后独立审查未做大cache SHA或GPU重跑。整体完成标记run/protocol/task4_complete.json。Task4完成时Task5未启动，当前进度见Task5；后续阶段仍需自身源依赖闭合，不能把visual_readiness扩展为全局放行。

### Task 5：源16轮视觉学生与新序列（Visual Student Rectification）

**文件：** visual_student.py/no_vote_datasets.py及CLI；tests/test_no_vote_visual_student.py/test_no_vote_sequence.py。
**接口：** train_visual_student保持select/refit接口；Selection.budget固定epochs16，metric仅描述val；build_sequence仍同phase。

- [x] RED：test_source_fixed_budget_16不按val峰值改轮数；test_source_hybrid_loss_parity使用同一小batch、相同随机状态对比原生效loss与梯度，明确head T与KD T分离；test_v2_targets_rejected拒绝旧T1目标祖先。
- [x] MC3结构、公有初始化、subject_robust、class_weights、源LR调度、batch4/accum4、singleton/GradScaler/clip-grad原样；保留当前已验收的锁与完整epoch恢复。partial不可用行按既定边界记数。
- [x] select训练16轮，末轮一次val报告；refit公开重新初始化16轮。无OOF、无额外early stop或最佳epoch扫描。复用pixels，重建模型、四区标准targets及四份native sequence/anchor。
- [x] Task5不生产十项teacher mechanism。源bank使用Task4固定Ridge在VideoMAE特征上的raw logits；不能用MC3的输入消融或校准early_late主头替代。
- [x] 真实train12一批forward/backward/native sequence与源对照；同phase anchor对预测差在FP16容差内，推理删除教师/label文件仍可运行。只跑此Task回归。
- [x] 正式生成后记录开发成绩及与v2的描述性差，不根据差值改变配方。此时仍不能读取final准确率。

**2026-10-29完成（Complete）：** select与refit均固定16轮末态，公开重新初始化／源loss与梯度／恢复验收通过；53项针对测试通过。四区标准targets和native sequence/anchor完整，有限独立复审整个Task5 GO（P30），无未解决P1/P2。开发182/388=46.91%，比旧v2最佳189/388低7条；未按分数改配方。训练133.85分钟、四区导出6.66分钟，完成标记protocol/task5_complete.json。A2-only student_readiness不改变全局false或旧identity；Task6未启动。详见[P29](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P29_Task5_固定16轮视觉学生.md)。

### Task 6：原运动缓存及恒等外部转换（Motion Cache and Source Transform）

**文件：** motion_cache.py、build_no_vote_motion_cache.py、tests/test_no_vote_motion_cache.py。
**接口：** build_motion_source/windows保持原契约；describe_motion_transform(motion: ArtifactRef, protocol: NoVoteProtocol) -> ArtifactRef；apply_source_transform为恒等外部映射，保留源内物理处理。

- [x] RED：test_no_added_global_standardization断言不读取fit labels、不拟合mean/std、输出字段与源cache/Dataset相同；test_source_window_alignment保留真实pixels索引与跨模态时间窗。
- [x] 移植P31与P86所有MOTION_FIELDS/BOOLEAN/TEMPORAL轴、重采样、身体/旋转补偿及原mask；不把v2额外归一化实现进去。
- [x] normalization父字段仅为确定性Source Transform Descriptor，标注imu_instance_normalization=false；本Task读取／转换接口无额外z-score，后续Task8/10消费者必须继承此约束。源模型内LayerNorm不删除。
- [x] 合成时窗及一条train12真实样本与原纯函数对照。缺设备/Skeleton样本保存原mask，不零填一套伪统计。

**2026-10-29完成（Complete）：** 40项motion／roles／cache-import针对测试通过，补充源Dataset单测通过；真实train12一条22帧／102IMU点，12字段对源函数差0、pixels时间差0、标签打开0。refit14=2427与final4=609的P31源／P86窗口及两个source_transform均生成，train12／development2按ID复用refit14公共缓存，无fit统计；Skeleton有效2341／590、IMU有效2283／580，全缺失85／18保留mask。正式日志跨度约125.24秒，有限产物自验0.88秒，未大数组SHA或全量finite复扫。完成入口protocol/task6_complete.json；本Task未执行独立子智能体复审，Task7未启动，原ROI精度残余与global false不变。详见[P31交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P31_Task6_运动缓存与源转换.md)。

### Task 7：源RF IMU教师（RF Teacher Parity）

**文件：** imu_rf_teacher.py、run_no_vote_imu_rf.py、tests/test_no_vote_imu_rf.py。
**接口：** train_imu_rf/predict_imu_teacher保持标准TeacherTargets。

- [x] RED：test_source_rf_recipe断言400树/depth18/leaf2/sqrt/balanced_subsample/seed20260723；test_source_device_dropout核对原random_present_devices/drop_devices结果、标签副本和mask；test_rf_logits_export比对源log(clip(prob))，不新增温度。
- [x] 32bin原始acc/gyro→原feature_vector240维+10mask，训练原样本+存在设备dropout副本。固定划分的select/refit均按train_final_imu_rf的无fold单次fit调用，dropout RNG=np.random.default_rng(20260723)，不虚构fold或seed偏移。仅fit对应阶段IMU有效数据。
- [x] 同一RF输出两份有明确consumer profile的40列logits：给A5的fit_teacher_target导出沿用run_imu_stat_baseline缺类sentinel=-1e6；目标分区terminal_export沿用train_final_imu_rf缺类log(1e-12)，已存在类均log(clip(p,1e-12,1))。训练目标为in-sample非OOF，但不改consumer数值规则；normalization/temperature不能混用。源bank的RF T3仅在Task11消费时施加，不改变A4原导出。
- [x] test_rf_consumer_sentinel_and_rng覆盖缺类、固定seed删除设备及fit/target两输出；Task7仅fit/导出RF，不等待P12的S+D融合，源train_final_imu_rf后续S+D混合拆到Task11。
- [x] 类列classes_映射、invalid与prior分离，源IMU partial teacher valid原样；不要求为了凑40类而拿val样本补RF。源支持缺类时按同一对齐算法记录。
- [x] sklearn小型fixture/真实一条stat验收；不加载p3_imu_oof历史张量。

**2026-10-29完成（Complete）：** 源32点统计240+10维、400树固定配方、原设备dropout／seed20260723、双consumer sentinel与本phase prior已实现。select1914有效→3828训练行、refit2283→4566；四区共8份标准targets完整，teacher features=None，T1不校准。开发123/388=31.70%（有效119/369）；final609行／580有效／29 RF-prior，不评分，不能将RF不可用混同18全模态缺失。29项针对回归GREEN48.46秒；真实单条统计／mask源差0、标签打开0，正式有限自验5.30秒。RF-only readiness只放source_expert/imu_rf，不改global false，不等待或冒充完成P12 S+D融合。完成标记protocol/task7_complete.json；未执行独立Task7复审，Task8未启动。详见[P32交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P32_Task7_IMU随机森林教师.md)。

### Task 8：源MoBind与缺失损失行为（Motion Pretraining Parity）

**文件：** mobind_pretrain.py及CLI/tests/test_no_vote_mobind_pretrain.py。
**接口：** train_motion_student接受同phase A1/A4与Source Transform Descriptor；预算固定24。

- [ ] RED：test_source_motion_losses_parity对正常、缺IMU、缺Skeleton小batch比较源losses各项与梯度；删去v2强制“缺分支CE/视觉KD全零”的测试断言；保留源重建/对比/RF-valid masks。
- [ ] 原模型最终motion config与全部损失权重/调度、temporal_augment、class_weights按规格8；从原随机seed重新初始化，不偷用已训练S/I分支。
- [ ] 24轮固定，末轮val诊断；refit随机重启同24轮。A1输入改为校准early_late和view-mean features；RF目标遵守阶段/valid。
- [ ] 真实两个train12批次比对缺失项、时间增强及teacher映射；输出A5-S/I/combined和四分区fallback记录。

### Task 9：额外平均融合诊断（Additional Mean-fusion Controls）

**文件：** simple_fusion.py及测试。
**接口：** 保留select_calibration/combine_available，记录role=diagnostic_not_source_main。

- [ ] 测试明确这些候选不进入A7监督目标、A8主bank或主预算选择；ID重排/缺模态prior契约不变。
- [ ] 若执行温度拟合，仅fit预测/标签，保持v2诊断边界0.25–4及identity/fitted两候选；不能称为原最终校准。
- [ ] 输出A6-VS/VI/VSI，完整609行；不因诊断更优改主链。

### Task 10：原融合full与独立训练控制（Fusion Parity and Controls）

**文件：** mobind_fusion.py、run_no_vote_mobind_fusion.py、tests/test_no_vote_mobind_fusion.py。
**接口：** train_fusion(full|mask_motion|shuffle_s|shuffle_i,...)；full严格源操作，controls独立标注。

- [ ] RED：test_full_source_losses_and_modes逐项对照原losses/selective_anchor_loss/set_stage/train_stage；正常/部分缺失时不增加availability gating；test_frozen_motion_dropout_matches_source不因冻结参数而擅自eval全部motion模块。
- [ ] 同phase A2/A5初始化、固定4+20、separate/additive、源腐败/调度/optimizer、frozen backbone/BN与anchor。不得从select权重refit或忽略新的校准目标。
- [ ] full motion_aux/reliability按源生效分支；control额外屏蔽只写control记录，不回流full。四独立训练模型共享初始化、顺序及预算；shuffle仅fit人口特征/mask。
- [ ] 原参数组/冻层/BN/损失与一批train12验收；输出A7及zero-S/I敏感性。源原视觉缓存与v3模型身份不一致时必须拒绝。

### Task 11：原保留专家与后处理（Retained Experts and Original Postprocessing）

**文件：** 新p12_source_adapter.py、source_teacher_bank.py、session_metadata.py、build_no_vote_teacher_bank.py；session_repeat.py及tests/test_no_vote_source_teacher_bank.py/test_no_vote_session_repeat.py。
**接口：** build_p12_source_chain(inputs, producers, protocol, context) -> ArtifactRef；fit_global_mixture(fit, p12, visual_heads, protocol, context) -> ArtifactRef；apply_fullwindow_knn(global_prediction, fullwindow_features, rows, protocol, context) -> Prediction；build_source_bank(roster, inputs, producers, protocol, context) -> ArtifactRef；build_session_blocks(rows: RowIndex, metadata: ArtifactRef) -> list[np.ndarray]；fit_source_postprocessor(bank, metadata, fit, protocol, context) -> ArtifactRef；apply_source_postprocessor(bank, state, metadata, rows, prior, protocol, context) -> Prediction。所有参数metadata属于被预测人口，不得从fit状态推测目标metadata或读旧全局CSV。

- [ ] RED：test_retained_roster_complete拒绝缺producer、历史bank、复制凑列或用A5任意替换源专家；test_source_group_features_parity按实际保留列对照p137，允许teacher数减少产生的自然退化。
- [ ] Task1 manifest逐项指定保留专家原producer/输入/参数；在当前fit人口重新训练非大视觉/非视觉与源派生头。独立小资产按已登记公开权重；不能读旧预测。P231/P238/P253/P306等名称“physical”不等于独立非视觉，必须按实际PATHS核实是否依赖被删VideoMAEv2/InternVideo2；无合法源输入就明确排除，不用当前A5替代。任何未说明差异阻止此Task完成。
- [ ] P12子链在本Task完成并绑定自身producer：train.py/ aligned_model.py给skeleton_logits/depth_logits；Task7 RF给imu_logits/imu_present；thermal_baseline/train_imagenet_oof.py给thermal_logits/thermal_present。复用build_complete_p11_oof的S+D 0.6/0.4与IMU校准混合，analyze_thermal_oof_fusion的温度/21点权重网格及决胜，evaluate_conditional_expert_routing的features/sensitive、StandardScaler→LR(C0.05,max_iter2000)、threshold0.5和单类退化。原折循环改为固定fit→apply，不能填假folds。若必要独立producer实际配置/公开资产/输入尚未锁定，Task1 preflight拒绝正式生成，不以泛泛“移植source”放行。
- [ ] P12八列固定顺序skeleton/depth/sd/imu/sd_imu/thermal/thermal_candidate/final，加Task4六visual头的log_softmax；按build_p85_multiexpert_submission_v1.fit_simplex的softmax权重、L-BFGS-B(maxiter500,ftol1e-12)只用fit标签拟合，给各分区生成其global scores。然后用build_p85_fullwindow_knn_teacher_v3.unit_features/top_neighbors/smooth：本分区无标签全窗特征三邻居（排除self）、0.40混合。不得用A1/A7概率替代global fusion、补oof包装/2241历史正确数，或跨val/final用标签；全缺失边界不参与传播。n<4的fixture按原算法前提拒绝，不靠改邻居数通过。
- [ ] test_p12_global_knn_dependency_order核对P12→global mixture→KNN→bank，没有Task4→Task7→Task6→Task5→Task4循环；逐算子toy数值对照，固定精确专家列顺序。
- [ ] 移植p134原CONFIG/evidence/repeat alignment、p137.group_features/fit_probability/choose_threshold及p165非OOF适配；sqrt/full/full，include_quality_features=false、embedding_lookup=None、C0.03/peer2/balanced=false/frequency_power0；源重复参数原样，不用v2“两peer+cosine+margin”替代。
- [ ] 锁定p165/p173/P255/P309 bank逐行规则：safe=normalise(0.95*source_base_probability+0.05*softmax(RF_terminal_logits/3))；source_base从原P89/safe-base producer产生，不能任取A7替换。缺detail视觉行的full40列及各available=false专家按原规则回退bank第0列，同时保留available=false；这是合法逐行fallback，不是复制假专家凑列。group route=(proposal!=base)&(p_proposal-p_base>=threshold)&visual_available，其中visual_available来自对应detail_ids精确成员资格，不能仅按姓名猜为ir_available。P270序列route不额外AND visual_available；P310仍仅new!=old覆盖，18全缺失最后统一prior。
- [ ] test_source_bank_fallback_and_route覆盖正常、缺IR但有motion、单专家缺失和全缺失；source_base/detail_ids映射必须来自新run原producer，不能用旧官方IDs、旧预测或补造available=true。
- [ ] RED：test_source_sequence_gate_and_precedence对照P270的emission0.65/transition0.45/alpha1/trigram1/beam50和p139五项gate/401点阈值网格/源决胜；old/new group与序列结果按P310 route公式逐元素一致。空disagreement仅恒等返回，不进行空quantile。
- [ ] 所有学习温度/threshold/gate/transition仅fit标签；refit重估于refit14，不能拿val标签代替原source cohorts。group、repeat、sequence均消费Task2新opaque-ID时间sidecar；build_session_blocks按(partition,user,date)外层分块，对每块调用源group_features/date_session_lists/cluster及build_sessions，再将索引映射回全局RowIndex。不得仅给源anonymous_date函数填user字段就认为已隔离。
- [ ] 不调用依赖data[name].split.sessions的旧sessions_for；adapter直接提供显式session index数组。未知时间不连边、输出base；metadata IDs缺失则失败，不能空填取消全部后处理。test_known_user_session_adapter覆盖同日不同用户/跨partition、未知时间、ID乱序、同块与源anonymous_date纯函数结果一致。
- [ ] RED：test_p310_sharpening断言可用行0.9805+39×0.0005、confidence0.9805，全缺失prior且target_mask=false；targets来自最终precedence预测，另保存锐化前group/emission和route，不替代为A7普通softmax。
- [ ] 保存Task12真正消费的A8目标引用、最终precedence预测与target_mask，并保持完整609行／40类及同phase身份；用于后续A8教师对A9学生的精确ID配对，不只保存一次汇总acc或未锐化中间教师结果。此步骤不读final标签。
- [ ] 合成完整会话/专家bank逐算子对照与一条train12接口验收；source_roster+实际参数+所有父记录齐全后才生成正式A8，不用简化替代品闭合流程。

### Task 12：最终40轮适配与12轮诊断（Final Adaptation Recipe）

**文件：** target_adaptation.py、run_no_vote_target_adaptation.py、tests/test_no_vote_target_adaptation.py。
**接口：** adapt_target(inputs: AdaptationInputs, endpoint: Literal[12,40], output: Path) -> ArtifactRef；无final label或accuracy参数。

- [ ] RED：test_final40_recipe断言最终实际LR1e-4/5e-5/min5e-6、scope heads_motion_encoder、seed20260826；test_base12_recipe断言12轮默认scope heads，不能错称两端点同scope。
- [ ] 原LabelFreePseudoDataset、configure_adaptation_parameters、train_label_free、temporal增强、余弦LR、confidence_power0、warmup0按源移植。40是主端点，12为额外诊断，各自从同A7 refit初始化。
- [ ] A8需本run完整保留bank/group/sequence/precedence祖先；Pseudo目标采用锐化概率。联合final609输入，591进入loss、18回退prior，不按最终准确率或confidence选行。
- [ ] 小型无标签batch逐梯度/参数组比对源函数；真实接口只用train12模拟target。训练范围拒绝label键，禁止把12继续训到40。
- [ ] 输出原始模型概率，不再执行A8；两端点都不访问final标签，主比较A9-40−A1。
- [ ] RED：在tests/test_no_vote_target_adaptation.py新增test_teacher_student_agreement_label_free、test_agreement_mask_denominators与test_agreement_is_not_a_selection_gate。小fixture覆盖完整／适配／prior三种分母、重复或缺失ID／错phase拒绝，以及一致率未达100%仍完成固定预算且不追加训练；禁读标签探针必须通过。
- [ ] 按本计划评估契约，对固定40轮与独立12轮端点分别写output/teacher_student_agreement.json：关闭增强、eval模式、A9原始推理对A8训练目标argmax，分开609全体／591适配／18prior计数，附同人口A7对A8适配前一致率及绑定引用。只复用小预测表和mask，不重扫序列、不重复跑骨干；该指标不选择端点、不判断真值准确率。
- [ ] 不新增user6/user7目标适配；训练日志的增强样本一致率只作优化日志，不代替训练完成后的无增强推理一致率。结果说明区分user6/7开发成绩、final4目标输入适配和其他未适配新输入的泛化，不能混称同一val_acc。

### Task 13：完整冻结与一次最终评估（Freeze and Reveal）

**文件：** no_vote_evaluation.py及freeze/evaluate CLI/对应测试。
**接口：** freeze_generation(protocol, registry, candidates) -> Path；evaluate_frozen(generation, private_labels, output) -> Path。

- [ ] 保留16候选：A1、A2、A5-S、A5-I、A6-VS、A6-VI、A6-VSI、A7、A7-mask、A7-shuffle-S、A7-shuffle-I、A7-zero-S、A7-zero-I、A8、A9-12、A9-40。主端点改40，不根据final改名单。
- [ ] RED：test_freeze_requires_v3_source_ops拒绝缺roster/操作证据、v2目标、未闭合bank；test_exact_final_ids_classes_prior核对609/40列/18prior，类别覆盖布尔不代替样本ID校验。
- [ ] prepared→generating→frozen→revealed，不可倒退。freeze只看直接输出/去重元数据及配置，不扫描raw/全cache或打开标签。
- [ ] evaluator先持久化revealed再打开用户显式提供私有标签，报告acc/fixed40 macro-F1、实际test类别支持、各用户/模态/rescue-harm。不得从这一步回到新训练。
- [ ] RED：在tests/test_no_vote_evaluation.py新增test_transfer_contrasts_same_population与test_accuracy_and_agreement_are_distinct。覆盖同人口四项比较、正确数／百分点差／rescue-harm配对关系、acc相同但预测不同，以及不同phase／人口不能混减；一致率报告无标签、准确率报告必须等待revealed。
- [ ] 揭示后保留原主比较A9-40−A1，补充A8−A1、A9-40−A8、A9-40−A7。每项按完整609行及固定A8.target_mask分层报告；A9-40−A8负值表明准确率下降，即使总体acc相同仍报告disagreement／rescue／harm。不以任一差值重新训练、挑端点或变更候选名单。
- [ ] 只用fixture测试状态、标签ID重排和错误集合。现阶段不运行正式揭示。

### Task 14：编排、资源与全链验收（Orchestration and Acceptance）

**文件：** run_teammate_single_teacher_pipeline.py、tests/test_no_vote_orchestrator.py、docs/teammate_single_teacher_fixed_split.md。
**接口：** run_pipeline(preflight|smoke|generate|freeze, config)；evaluate独立。

- [ ] RED：test_dependency_order包含六头/全窗/mechanism/保留非大专家与group/sequence/precedence/sharpening，不止于A7；smoke走16候选直到frozen，不读真实final标签。
- [ ] preflight只读明确资产/磁盘/GPU/配置及producer清单；按保留教师真实依赖估时估空间，不用旧10–20小时假设，不以缩短时间为由跳过算子。
- [ ] 编排每stage复用验证上下文、每完整epoch恢复、OS锁、成功/失败标记及持续日志。Fixture读取计数验证rawSHA0/ancestor-payload0/epoch资产校验0；不真实全扫再记录“很快”。
- [ ] 各Task针对性回归、原源函数同输入数值/梯度/模型config对照、真实train12小批次通过后再正式generate。最终操作清单无未解释差异才freeze。
- [ ] 持续提交当前实验分支。主分支历史回顾文档仍等整个复刻实验完成后更新，不混入此轮计划修订或诊断报告。

## 状态与交接（Status and Handoff）

v2已完成：Task4单头T1开发69.07%，Task5第17轮开发48.71%、四区目标/序列、工程及方法学验收GO。v3 Task1已实现并独立复审GO（68项针对性测试），Task2准备/缓存导入完成（P03）、Task3几何一致性完成（P24），两者实际独立复审GO（P25），Task4头部及完整全窗均已完成并独立GO（P28），Task5已完成并独立GO（P30），Task6运动缓存／恒等转换已完成生成与有限自验（P31，未独立复审）；source roster材料缺口保持完整bank/融合的全局生成阻断。Task7已完成源RF／双consumer输出及有限自验（P32，未独立复审）；下一任务为尚未启动的Task8 MoBind；不得把当前v2权重当v3已通过。

无定时自动化，Task5训练／导出、Task6缓存与Task7 RF进程均已结束；不推断final准确率，不自动进入Task8。2026-10-06原版复审问题已修订，Task1–5各规定节点已实际独立GO；Task5最终53项针对性测试与正式产物有限验收通过，Task6为40项针对回归及有限自验通过。后续按Task8–14各自依赖和复审节点执行，当前局部GO不代表全处理等价，不跳过manifest/roster或P12等材料门槛。v2已退休两resume.pt和四sequence.npy（1.02GiB），正式checkpoint/targets/anchors/公共缓存/来源记录保留；旧seq refs为历史证明，不再宣称可直接完整加载。修改后自检：source_ops覆盖、接口/角色/硬链接导入一致、固定人口/无OOF无矛盾、每主链参数有source、每验证只消费直接输入。
