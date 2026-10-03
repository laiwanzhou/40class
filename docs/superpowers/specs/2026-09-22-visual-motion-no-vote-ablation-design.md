# 固定划分的队友单视觉教师流水线规格（Fixed-Split Single-Teacher Pipeline Specification）

修订日期：2026-10-02，版本 v2。根据[独立审计报告](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-02-handoff-plan-independent-audit.md)修订；旧报告的行号只指向旧版本。当前为设计规格，Task1–3已实现对应接口并验收；模型训练尚未开始，Task3只有真实一条partial验收。

## 1. 目标与解释边界（Objective and Interpretation）

从原始比赛训练数据重建队友的一条教师到学生（teacher-to-student）路线：一个冻结 VideoMAE-Large 视觉骨干、一个选定 Ridge 分类头、MC3 视觉学生（Student）、Skeleton/IMU 分支、MoBind 融合、会话/重复处理和无标签适配。排除历史30教师投票，测量固定人口上的整条路线和阶段替换差值。

“从头训练”指重新提取任务缓存、训练分类头、学生和融合；保留公开视觉/姿态预训练初始化。0.91不是完成标准。ROI与视觉腐败训练是包含的操作，本轮不单独识别它们的因果增益。A2−A1称教师到学生替换/压缩差值（replacement/compression delta）；A7−A6-VSI称学习融合组合效应（bundle effect）。

只报告固定划分的描述性比较（descriptive comparison）。user6/user7已用于历史开发，其他用户的样本统计也曾用于项目设计，因此本实验不证明完全独立的未见用户泛化，也不估计官方匿名测试成绩。单随机种子（single seed）不证明种子稳健性；无需新增多种子或显著性检验。

## 2. 固定人口（Fixed Populations）

| 分区 | 用户 | 规范清单行数 |
|---|---|---:|
| train12 | user1,user2,user3,user5,user8,user9,user16,user18,user19,user20,user21,user22 | 2039 |
| development2 | user6,user7 | 388 |
| refit14 | train12与development2的并集 | 2427 |
| final4 | user4,user17,user23,user24 | 609 |

四分区使用同一规范样本集合；不能把队友2914条人口断言移入本实验。train12/refit14的标签必须覆盖0–39，包括user1的class25。不要因为某模态缺失而删规范行。模型实际拟合支持由可用性掩码（availability mask）定义，另存拟合ID和缺失原因；对每个监督模型检查有效拟合样本的40类支持，缺类时停下并报告，不伪造样本。

final4中IR可用591条、Depth/Skeleton可用590条、IMU CSV文件可用580条（2026-10-02实际检测，早期估计584条），18条全部纳入模态缺失。所有正式候选都预测609行；全部缺失的18条始终使用refit14类别先验，不参与A8传播或A9损失。其他缺失组合按本阶段实际输入掩码处理，例如A1/A2对缺IR条使用先验。

选参阶段：所有监督训练祖先只拟合train12，development2只预测与选择。配方冻结后：A1/A4重新拟合refit14，A2从公开MC3初始化重新训练，A5从原始随机初始化重新训练，A7从对应A2/A5的refit14检查点重新训练。不能以select检查点继续训练替代refit；不得将refit14教师目标送入development2选模。

A9另有已批准的final4无标签适配（unlabeled adaptation）祖先，不能把这种目标域使用误报为监督标签泄露。所有fit/select/predict/adaptation角色分开记录。

## 3. 输入准备与标签隔离（Input Preparation and Label Isolation）

独立可信准备步骤（trusted preparation）可以读现有带标签规范清单，以完成固定用户划分、路径解析和标签分离；它不运行模型、不选参数、不生成正确性报告。模型生成进程不能执行该准备步骤，也不能接收原带标签清单。

准备步骤产生：
- 公共无标签清单：白名单字段为sample_id、user_id、ir_path、depth_path、skeleton_path、imu_path和四个available字段；缺失路径为空。所有分区的缓存rows均采用无标签模式。
- 标签表：train12/development2/refit14分别独立存储sample_id/class_id。最终标签及旧ID到新ID映射只进入独立评估资料目录（evaluation vault），例如 `C:/Users/LaiWanzhou/AppData/Local/Temp/cuhkx_no_vote_labels/<run_id>/`。
- sample_id采用固定命名空间下原规范ID的SHA256十六进制，不暴露cXX；source_id同sample_id。公共侧不导出旧含类别ID或action_name。
- 保留必要原始绝对路径，仅用于文件定位。类别目录名称不得被解析成目标、拟合特征、排序键、重复组或选择条件。所有处理顺序由规范不透明ID（opaque ID）固定。
- timestamp/date/duration/device信息从原始文件内容和时间戳提取，不能从动作类别或trial序号推断类别/重复组。

运行配置（generation config）只包含公共清单及前三分区标签路径，不含final标签路径、旧final报告、Kaggle分数或历史预测。最终评估器（evaluator）在冻结后通过显式参数取得私有标签路径。禁止字段包含class_id、class_name、action_name、label、labels、correct、confusion及其别名；训练标签表是单独允许的输入，不能混入公共缓存。对最终生成路径删除或置换私有标签文件不影响生成数组与文件内容；包含时间戳的日志另测依赖读取边界，不要求运行时间字节相同。

## 4. 排除范围与源码移植（Exclusions and Source Porting）

部署输入为IR视觉、Skeleton、IMU。视觉三视野为scene/person/workspace的IR灰度，MC3内部按公开Kinetics输入转换；Depth用于几何同步和骨骼对齐，不增加独立Depth教师。本轮无Thermal、Radar、MotionBERT、HD-GCN、InternVideo2、V-JEPA、LaViLa、DINOv2或任何历史专家概率。

队友源码只作只读参考。复用模型、预处理算子、增强与损失函数；新建分区无关（partition-generic）、标签可选（label-optional）的提取器、Dataset和固定划分训练循环。不得直接运行旧P46/P85/P86/P87 CLI主入口，恢复类别来兼容它们，或加载旧runs/cache默认路径。1384、2914、1497/973/444和401/405历史计数不属于新人口契约。P310及expert-bank文件即使改名也不得进入训练来源。

## 5. A0 几何与原始缓存（Geometry and Raw Caches）

公开YOLO11n-pose，记录文件哈希；两遍IR姿态检测和跟踪、同步Depth、人物/手臂/工作区ROI（region of interest）。复用队友计算逻辑，排序及输出与类别无关。

将Skeleton帧号映射到原IR时间戳的恢复逻辑直接接入移植后的frame_map读取路径，不以未接入的helper或子进程外补丁代替。缺ROI但存在IR时使用记录过的整帧回退；缺IR不伪造视觉帧。全部缺失行也完成缓存记录，其completed表示处理完成而非模态可用。

P31读取P28的frame_ids、skeleton_h36m_xyz_conf_raw、skeleton_person_count等原始字段。P29仅提供ROI几何。同一只依赖原始输入的缓存及冻结公开VideoMAE的标签无关六clip特征可以由train12/refit14按ID引用复用；按本实验监督人口学习的统计量、归一化、Ridge概率或MC3特征不能跨select/refit身份复用。

## 6. A1 单视觉教师（Single Visual Teacher）

公开初始化：`MCG-NJU/videomae-large-finetuned-kinetics`，revision `0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4`。单独取得权重并记录实际SHA256，本地目录加载不再次当Hub仓库名下载；恢复旧注意力偏置时核验键与形状。

冻结骨干；早窗0–0.70、晚窗0.30–1.0，各16帧，scene/person/workspace三个视野。输出features `[N,2,3,1024]`、kinetics_logits `[N,2,3,400]`和有效掩码，无标签。源图像裁剪与处理器参数跟队友prepare_trial一致；teacher relation/feature目标保留六clip特征，teacher/pixel/motion时间窗必须相同。

仅拟合一个选定40类Ridge头：
- 六特征族early、late、window_mean、early_late、temporal_delta、kinetics。
- 权重指数0.0/0.5/0.75，alpha=300/1000/3000/10000，共72候选。
- 复用队友feature_sets的L2归一化、window mean和差分定义，不重写为不同数学变换。
- 决胜顺序：development2全分母accuracy、fixed40 macro-F1、最差用户accuracy、较小特征维度、较大alpha，最后固定候选ID字典序。

select头仅fit train12；refit同配方仅fit refit14。下游TeacherTargets采用一个选中头的logits/probabilities/valid，不强制构造旧四分类头、labels/users/folds或OOF字段；若需逐窗特征，只读取六clip原特征，不新增专家投票。

2026-10-02实施口径：A1 head temperature固定1，logits为40列Ridge decision scores，probabilities=softmax(logits)。队友旧P85的OOF温度校准不在本固定划分实验中复用；此数值差异作为已冻结复刻变体记录，后续KD温度和融合校准另行处理。原模型数学仍为StandardScaler→RidgeClassifier(solver=lsqr,tol=1e-5,max_iter=5000)。

## 7. A2 MC3 视觉学生（Visual Student）

MC3-18 Kinetics400公开初始化；frames16，resolution160，width512，temporal_modeling=true，freeze-through=layer2，subject_robust同步增强，batch4、accumulation4，seed20260811。训练端只用fit人口的标签和教师目标，推理Dataset不要求任何教师文件或标签。

训练模式固定hybrid：CE + 概率蒸馏温度2.0/权重1.0 + 六clip关系损失0.2，label smoothing0.1；head LR2e-4、backbone LR1e-5、minimum LR1e-5、weight decay0.08、class-weight power0.35。源码命令虽传feature-weight0.5，hybrid分支实际不使用直接特征对齐损失，enable_distillation_projection=false；将配置值与生效损失分开记录，不人为增加该损失或投影。A5的视觉特征对齐仍按第10节生效。

select候选epoch=1–18，每轮在development2全388条评估，按accuracy、macro-F1、最差用户、较早epoch选择；refit从公开权重重启，固定所选epoch，无早停。推理和序列提取仅需学生与像素输入。

原生layer4时间特征为 `[B,2,3,16,512]`；允许源码的时间线性插值，不允许把池化向量复制成伪时间序列。不同select/refit骨干各自建立序列缓存。

## 8. A3 运动缓存与归一化（Motion Caches and Normalization）

从原始Skeleton/IMU重建P31/P86：H36M-17身体相对坐标、部件局部运动、五设备角色WTC/WTLA/WTRA/WTLL/WTRL、原始acc/gyro、补偿和相对quaternion、缺角色掩码。P86时间窗依赖像素缓存的source_frame_indices和原始时间，不依赖视觉sequence embedding。

原始运动字段沿用队友MOTION_FIELDS：Skeleton主张量 `[N,2,16,17,13]`、IMU主张量 `[N,2,16,5,4,16]`及对应mask，窗内采样数points_per_imu_bin固定为源码默认4；bin statistics为52通道、global statistics为48通道，写入缓存schema。

Dataset边界显式拟合/应用有效值均值和标准差，缺失位归零；select统计fit train12，refit统计fit refit14。记录channel/role轴和统计有效ID；禁止dev/final参与统计。原始缓存共用，归一化状态分开。

## 9. A4 单RF IMU教师（Statistical IMU Teacher）

复刻stat_random_forest_device_dropout_aligned；复用run_imu_stat_baseline的统计特征及设备对齐，32时间bin、每设备6个acc/gyro通道各8个统计量，加每设备2个掩码统计。原始acc/gyro单位及重采样算法写入配方，使用P31保留的无损IMU流，不直接把学习后的运动embedding当统计特征。

RF固定n_estimators400、max_depth18、min_samples_leaf2、max_features=sqrt、class_weight=balanced_subsample，seed20260811。拟合数据为原样本加一份每行随机删除一个现有设备的副本，标签同步复制；该固定配方只有一个候选，dev评分用于诊断，不另加RF投票或搜模型族。

输出统一TeacherTargets；概率clip到1e-12并重新归一化，logits=log(probability)，类列0–39；valid是原始IMU可用性，与缺失先验概率分开。兼容适配器显式映射imu_logits/valid，不默认全部有效。

## 10. A5 MoBind 运动预训练（Motion Pretraining）

width96、alignment64、batch64、seed20260811，max epochs24；LR4e-4、minimum2e-5、weight decay0.05、class weight power0.35、label smoothing0.08、mask ratio0.35、dropout0.12。损失：Skeleton/IMU CE、重建、视觉概率蒸馏、视觉feature对齐、部件/时间局部对比、全局语义对齐、IMU教师蒸馏。

移植源码的semantic2.0、token0.25、local0.05、global0.15、reconstruction0.1、distillation0.75/temperature2.0、teacher-feature0.25、IMU-teacher1.0/temperature1.0、contrastive temperature0.08。缺失分支相应损失为0。

select逐轮在development2上用combined motion输出选择epoch，决胜规则同A2；refit重新随机初始化并固定epoch。A5-S/A5-I是同一模型的原生独立分支概率，不宣称分别训练的单模态教师。使用原forward(motion, mask_ratio)及mask构造，不假定原模型支持enabled_modalities参数。

## 11. A6 校准平均对照（Calibrated Mean Controls）

温度在fit人口预测与标签上拟合，边界0.25–4.0；identity T=1与拟合T两种方案在dev比较，以accuracy/macro-F1/最差用户决胜，平局选identity。refit选择拟合方案时仅用refit14重新估温，温度拟合只作校准，不宣称OOF。缺模态行不参与该分支温度拟合。

按ID对齐A2/A5-S/A5-I，产生A6-VS/VI/VSI。按可用分支等权平均；没有可用分支时返回当前fit人口先验，不能返回全零向量。

## 12. A7 融合与匹配训练对照（Fusion and Matched Training Controls）

保留原MC3 seam、分开的Skeleton/IMU编码器、additive global residual；stage A4 + stage B20轮，batch64、workers0、seed20260811，冻结预训练motion encoder。fusion LR4e-4、visual head LR2e-5、weight decay0.05、class weight power0.35、smoothing0.08、distillation温度2.0/权重1.0、relation0.1、motion aux0.35、selective anchor0.3、visual corruption0.75、feature dropout0.3、view dropout0.4。

固定预算4+20，不按final或其他控制分支成绩改变。dev只作诊断。select从A2/A5 select初始化；refit从A2/A5 refit初始化，融合层重置，且监督教师换为A1/A4 refit。选择性锚点来自同phase的冻结A2 checkpoint，Task5生成按ID对齐的anchor_logits/valid并记录祖先；live_visual_anchor=false，A7更新不改锚点。视觉backbone参数及BatchNorm运行统计冻结，每次切训练模式后仍令backbone处于eval，以保证缓存一致；只更新源码visual_head_parameters列出的头部。若后续实验要解冻骨干，属于新配方，必须同步重建sequence。

四个分支各自独立训练并refit：A7、A7-mask、A7-shuffle-S、A7-shuffle-I。共享初始化值、预算、优化器、损失系数及样本顺序；mask控制将两motion输入及mask置零，适配层显式用控制后的motion availability屏蔽motion_aux和reliability，不依赖原源码自动屏蔽；教师损失交叉使用teacher valid与对应输入可用性，空集合返回保留计算图的零loss。shuffle按本分区ID产生置换，连特征与mask一起移动，保留接收行的监督标签和教师目标；训练每epoch使用seed+epoch，评估使用冻结的seed置换，禁止跨分区。单样本分区无合法非恒等置换时测试应显式报告不能生成shuffle控制。推理zero-S/zero-I另列A7-zero-S/I敏感性分析，不替代上述训练控制。控制不触发新视觉教师训练。

## 13. A8 会话与重复后处理（Session and Repeat Processing）

仅处理A7概率。会话键固定(partition,user_id,date)，known_user模式；日期/时间无效时该行不建立边，输出不变。按原始start时间、opaque ID排序；同一用户/日期相邻start差>30秒断开。重复组只在同一会话内形成，不能按类别、trial序号或最终标签分组。

select transition计数fit train12；refit计数fit refit14。复用队友fit_transition_model的start/end、bigram与backed-off trigram统计，alpha0.25；复用decode_unique_beam_posterior的无重复类约束、beam width50、posterior temperature1.0，以保留路径的边缘概率为输出，不用另写一阶Viterbi替代。会话长>40或无合法路径时整会话返回A7，不人为按动作切段；正常会话只对posterior最大概率>=0.80的行应用结构概率，未通过门控返回A7。先transition，后repeat，最多一遍；最少2个同类同意peer，即组含至少3条。repeat门控比较peer同意类的均值概率与接收行该类概率，差至少minimum margin；更新为接收概率与同意peer均值各0.5，所有peer选择取自该步骤输入快照，不原地迭代。全部模态缺失行从连接图剔除并恢复先验；其中已知时间的行打断相邻transition。日期/时间无法定位的行不建立边也不猜测其位置，其他行仅按可观测时间形成会话。

固定候选网格：transition weight=0.25/0.30/0.35、trigram backoff=1.0/2.0/5.0，沿用源码候选；repeat duration relative tolerance=0.10/0.20、probability cosine similarity=0.90/0.95、peer confidence=0.80/0.90、minimum margin=0.05。比较identity、transition-only、repeat-only、combined，按dev accuracy、macro-F1、较少改变行、候选ID决胜。已知用户会话边界、结构概率门控与有限repeat网格是本次固定划分移植的预定策略，不称匿名官方测试配方逐项相同。协议在选参前记录所有候选，不依据final增删。

## 14. A9 无标签适配与端点（Unlabeled Adaptation and Endpoints）

适配池固定为全部final4共609个规范ID，同一模型共同适配；损失mask为至少一纳入模态可用的591条，18条从loss排除且预测恢复先验。每用户报告共享适配模型，不能称四次独立实验。

目标只能是本run、refit14祖先生成的冻结A8概率；固定不更新，不按confidence筛样本，不查看标签，不早停或择checkpoint。输入包括pixel、对应A2 refit的sequence、motion和normalization；部署模型不包含A1大教师。

A9-12从A7 refit独立启动：12轮、seed20260812、scope=heads_motion_encoder、AdamW，fusion与motion LR5e-5、visual head LR1e-5、minimum2e-6、weight decay0.02、temperature1.0、confidence_power0、无warmup、batch64。视觉backbone及BatchNorm统计在eval冻结。scope比源测试适配默认heads更宽，明确记为本次已选择的复刻变体。

A9-40同样独立从A7 refit启动：40轮、seed20260826、fusion与motion LR1e-4、visual head LR1e-5、minimum5e-6；其余同A9-12。不能从12轮结果继续训练到40轮。

A9-12/A9-40均为适配模型的原始输出（raw deployment output），不再次应用A8。主要比较A9-12−A1；A9-12−A8称后处理到适配模型的部署替换差值，另报A9-12−A7。本轮不声称A9−A8是保留A8后的纯新增适配效应。

## 15. 固定候选与评估（Candidates and Evaluation）

必须冻结：A1、A2、A5-S、A5-I、A6-VS、A6-VI、A6-VSI、A7、A7-mask、A7-shuffle-S、A7-shuffle-I、A7-zero-S、A7-zero-I、A8、A9-12、A9-40。候选和角色在选参前写入protocol，不由final表现选择。

每个候选sample_ids完全覆盖final4，class_ids=0–39，probabilities有限非负且行和1；evaluator按私有opaque ID一对一连接，不按数组位置计分。报告correct/609、IR可用591子组、40类macro-F1、四用户、逐类recall、缺模态子组、rescue/harm/net/disagreement、NLL/Brier、A9/A8 agreement。

阶段差值按第1节解释：A2−A1、A6−A2、A7−A6-VSI、A8−A7、A9−A8/A7；只描述本次配方和人口，不能把简单阶段替换全部称作单机制因果贡献。

## 16. 产物来源与单向状态（Provenance and State）

run根目录为 `outputs/teammate_single_teacher_fixed_split/<run_id>/`。所有产物记录协议/配置/源码/公开权重哈希、stage/kind、fit/select/predict/adaptation users和ID集合哈希、类别顺序、父产物哈希、fallback/normalization身份、文件相对路径与哈希。加载检查直接输入与来源记录，不以路径名字判断是否属于禁止bank。

2026-10-03用户修订校验范围（Verification Scope）：Task4运行累计超过10小时，主要瓶颈为重复读取大面积原始文件。后续训练、推理、续跑、审计和freeze取消原始全目录SHA、完整缓存额外扫描及祖先文件内容递归复验。角色、阶段、ID、配置和状态检查可遍历去重后的记录元数据；实际消费时检查schema、有效掩码、概率及实际加载模型。直接输入损坏或被修改时针对性复查，不自动重做全量校验。原始数据依只读约定使用，轻量验证不保证主动发现所有未消费文件的内容漂移；历史SHA记录仍保留，不将轻量检查宣称为全量内容验证。具体执行与Task4完成后的修复验收见实施计划“校验策略修订”。本次只更改校验策略，不改变任何模型配方、实验划分或final标签封存。

protocol的用户集合先排序为JSON数组，所有路径规范化，序列/浮点/布尔明确类型；禁止default=str对frozenset序列化。协议身份包括全部种子、候选、预算、目标池与源码/权重清单身份。

单向状态：prepared → generating → frozen → revealed。freeze核验全部16候选、609 ID、refit14监督祖先、A9的A7/A8目标祖先和18条先验。先持久化revealed状态，再读取最终标签，评估失败可以对同一generation hash重算；不能解锁新训练/新候选/重新冻结。smoke使用独立fixture run，必须走到frozen且不能读取真实final标签。

## 17. 资源与完成条件（Resources and Completion）

2026-10-02已观测D盘约46.90GiB空闲；Python3.12.9、PyTorch2.7.0+cu128、torchvision0.22.0+cu128，RTX5060 Laptop约8GiB显存。20GiB最低门槛已满足，执行时仍重新查剩余量和预计峰值，不能写死本次空间数字。

共享原始像素/几何缓存按ID建立一次，refit引用train+dev；模型sequence与学习统计分开。若重复建立四分区像素，仅images.npy约12.50GiB，因此预检必须计缓存、公开权重、全部select/refit与控制检查点、续跑临时文件的峰值，估计余量不足时停在预检，不先开长训练。

10–20 GPU小时是旧的未验证估计，新增匹配训练对照可能增加预算。用一条/一批真实训练和接口验收实测速度、显存，再记录更新预算。完成前必须通过固定分区/标签/祖先/ID/概率/状态契约以及A0→A9全链验收。文档修订通过不等于代码实现或训练通过。
