# 固定划分的单大视觉教师源码一致性规格（Fixed-Split Single-Large-Visual-Teacher Source Parity）

修订：2026-10-06，v3.1（独立复审纠正producer与接口）。执行计划：[实施计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md)。v2文档已归档到 `reports/audit_inputs/2026-10-04/pre-v3/`；已完成的v2 Task1–5和准确率保留为历史基线。v3是新规格，尚未实现或训练，不将文档修订当作执行完成。

## 1. 目标与允许差异（Goal and Allowed Differences）

按 `CUHK-X_Small_Model_Submission(1)` 的最终训练链复刻操作，研究单个大视觉教师的作用。只允许两项设计差异：大视觉预训练教师根节点（Large Visual Teacher Root）缩减为一个VideoMAE-Large；外层评估继续固定用户划分，不改成三折OOF，也不引入内部OOF。缺失数据、公开ID及当前人口所需适配逐项登记，不能宣称数值等价或单因素因果识别。

保留同一大骨干的多个特征族、分类头、邻居规则和固定Ridge特征敏感性预测；它们不是新增大骨干，也不能被误计为多个大教师。非大视觉/非视觉专家不能因为“单大教师”而被随意删除。只有依赖被移除大视觉根节点且无法在不更换算子的情况下生成的专家，才作为该教师删除的依赖后果明确列入排除清单；不能以新特征、复制概率或假专家补齐。

核心操作来源以最终包的实际命令、部署配置和生效训练分支优先，其次是调用时的源码默认值。不得将未启用的实验选项当作最终组件，或用另一历史候选替代。源码映射与允许差异必须在训练前保存为 `source_ops_manifest.json` 和 `teacher_roster.json`。

这次恢复的是操作一致性（Operator Parity）；训练人口与教师数改变会改变目标和权重，不能保证准确率0.8/0.91。v2的A1=69.07%、A2=48.71%只作历史记录，不用于选择v3额外超参数。

## 2. 固定划分与类别支持（Fixed Split and Class Support）

| 人口（Population） | 用户 | 规范样本 |
|---|---|---:|
| train12 | user1,2,3,5,8,9,16,18,19,20,21,22 | 2039 |
| development2 / val | user6,user7 | 388 |
| refit14 | train12与development2并集 | 2427 |
| final4 / test | user4,user17,user23,user24 | 609 |

保持原已批准边界。配方确定后在refit14重拟合，不能把user6/user7的refit预测叫验证成绩。final4私有标签仅在全部候选冻结后由独立评估器读取。无标签目标域适配可以使用final输入，但不以最终准确率选择预算、阈值或模型。

已核对：train12、development2、refit14联合均覆盖40类；user6单独34类、user7单独33类。用户2026-10-04限定授权独立子智能体仅检查final4类别覆盖，返回“联合覆盖全部40类=是”，未传回样本标签、频率或分用户明细，未计算模型成绩；这次查看覆盖元信息须记录，不宣称测试资料从未被查看。该授权不允许生成进程读标签或据此换用户。仍不假定每名用户完整40类。预测列始终0–39；训练类缺失处理按实际source classes_映射，不能捏造样本。原三折某类只由一名训练用户支持的问题不再作为固定划分的折内门槛。

所有模型权重、校准温度、分组/门控阈值、转换统计只用本阶段fit标签。val仅预测、诊断及从既定12个配置中选择每族分类头，不拟合温度或门控；final不提供标签。训练软目标不再是OOF，而是本阶段fit模型预测，此差异必须写入每份目标记录。不得把它叫OOF或加载旧全人口OOF。

final规范609条，至少一模态可用591条，全缺失18条使用同一refit先验、不参与传播或适配loss。A1/A2缺IR仍回退对应fit先验。完全无输入的拟合行按当前输入适配规则处理并记数，不扩展到将部分缺失样本的源损失任意屏蔽。

## 3. 数据、初始化与复用（Data, Initialization and Reuse）

原始数据、两份提交包、队友快照只读。公共清单保留opaque ID、用户、输入路径和availability，不混入标签或类别目录派生特征。教师专用模态若有源链要求，准备步骤单独声明公开输入字段及用途；不能默默删掉源非大视觉专家或复用其历史预测。

部署输入仍为IR、Skeleton、IMU，Depth用于几何。教师专用输入与公开初始化由teacher_roster的源码依赖决定，单独声明，不自动变成部署输入。VideoMAE-Large、MC3和YOLO使用已核对的公开权重；保留下来的其他源专家需要的公开资产另行显式登记，不借用旧比赛训练权重。

v3运行名固定 `fixed-split-single-teacher-v3-source-ops`。保持v2配置、检查点和目标不可变。可将v2标签无关P28/P29、冻结VideoMAE六clip特征和pixels按精确ID、轴、配置与源记录导入新run。采用同卷硬链接，在新run建立immutable/只读payload及全新同协议manifest/initializer/P28/P29祖先；旧ref只存provenance，不作为ancestry、不登记旧外部payload路径，不改旧identity。导入按拓扑与memo复用digest，不扫描大数组。v2 Ridge/目标/学生/模型sequence不得作为v3监督祖先。2026-10-06旧两resume及四sequence.npy已退休，最终checkpoint/targets/anchors及可复用公开缓存保留。

窗口为0–0.70/0.30–1.0，每窗16帧，scene/person/workspace三视野；像素160，人物/工作区裁剪1.15/1.40，质量与fallback原样。保持已修复的时间戳解析及跨模态时间对齐；不解析类别/trial序号生成规则。

## 4. A1视觉教师目标（Visual Teacher Targets）

复用 `train_p85_videomae_full40_head.feature_sets/sample_weights/aligned_scores_40`、`train_p46_videomae_head.make_model/fit_temperature`。一个冻结VideoMAE-Large，六族early/late/window_mean/early_late/temporal_delta/kinetics；每族权重指数0/0.5/0.75、alpha300/1000/3000/10000。分类头StandardScaler→Ridge(lsqr,tol1e-5,max_iter5000)。

源student固定读取early_late，主A1/A2监督头也固定early_late；不再从六族中用val选window_mean。每族内部按源码accuracy、balanced accuracy、macro-F1、较小class-weight power、较小alpha决胜。固定划分替代OOF后，select候选只fit train12，在user6/user7 val上比较已冻结的每族12个配置；原OOF选参指标改为固定val指标，属于已允许划分差异。refit继承这些配置，只重fit refit14，不重新选择配置。各族头供源后续机制/路由使用，不新增投票来替代源码。

恢复源fit_temperature：对本阶段fit预测与fit标签优化NLL，log-T边界±2.302585，即约0.1–10，输出scores/T。不得固定head T1，不借用队友温度，不用val/final拟合。记录原始分数、温度、目标来源与实际类支持；refit只在refit14重新估计。下游学生的KD温度2与这个head温度分开。

源缺类Ridge对齐为每行min(scores)−max(ptp(scores),1)后填40列，再覆盖实际classes_；不能把缺类列默认为高概率。六clip特征仍为[N,2,3,1024]，与单个大骨干一致。A1标准预测使用校准后的early_late logits。
十项mechanism是audit_p86_teacher_mechanisms固定Ridge头，而非MC3学生：逐clip L2、alpha3000/power0.75、原scaler/Ridge、fit特征均值填补/交换/折叠、aligned_scores_40；只fit本phase，十输出保留raw logits/T1，不套A1校准。对应目标侧p89_build_multiexpert_test_logits.build_p86必须同数学，Task4生成，祖先无A2。full-window标签无关特征也可在Task4提取，但学习global-fusion/KNN明确在Task11完成。

## 5. A2视觉学生（Visual Student）

直接用最终包P86MC3VisualStudent：MC3-18 Kinetics、frames16、resolution160、width512、dropout0.18、gated、temporal=true、freeze-through layer2、BatchNorm统计冻结。exact-time、cross-view、spatial-region、region-temporal、structured-region及直接feature projection均关闭。

复用subject_robust、class_weights、learning_rate_scale和生效hybrid损失：CE(smoothing0.1)+KD(T2,weight1)+six-clip relation0.2；直接feature/stage KD为0。head LR2e-4、backbone LR1e-5、minimum1e-5、AdamW wd0.08、batch4/accum4、clip-grad2、seed20260811、singleton尾批规则均按源代码。不得因val成绩加损失或提高预算。

以实际终端配方固定16轮；select与refit都从公开初始化训练16轮，val最后一次诊断，不以18轮中最佳epoch改变预算。原内层OOF/early-stop评分不移入本固定流程。Selection记录预算来源为源最终命令，而不是最佳val。

像素可复用；新学生/四分区目标/同phase原生sequence和anchor必须重建。[N,2,3,16,512]不能池化复制。移植无标签推理Dataset，不要求teacher或labels文件。保留模型/optimizer/scaler/RNG完整epoch恢复和OS文件锁。

## 6. A3运动处理（Motion Processing）

复用P31和build_p86_motion_window_cache，以及P86MoBindMotionDataset/P86CachedSequenceMotionDataset的实际张量与增强：Skeleton H36M17、13通道；五IMU角色WTC/WTLA/WTRA/WTLL/WTRL、每bin4点、16通道、52bin统计和48global统计。原有身体相对坐标、旋转/补偿、裁剪及物理尺度不变，与pixels源帧一致。

移除v2新增的跨样本fit mean/std。原Dataset直接消费源缓存，只转float32并施加同步时间增强；最终部署imu_instance_normalization=false。保留源算子内已有的物理归一化和模型LayerNorm，不再二次标准化。原normalization接口改为源转换描述符（Source Transform Descriptor），kind为确定性转换，不含fit均值/标准差；消费者执行恒等外部变换，字段值须与源fixture逐元素相同。

## 7. A4 IMU统计教师（IMU Statistical Teacher）

按run_imu_stat_baseline的random_forest_device_dropout：32bin原始acc/gyro统计、240+10维、400树、depth18、leaf2、sqrt、balanced_subsample、n_jobs=-1；源默认seed20260723；固定划分两phase采用train_final_imu_rf无fold拟合路径，dropout RNG直接np.random.default_rng(seed)，不虚构fold/偏移，不随意换视觉seed。

原+每样本删除一个存在设备的副本，复制标签，设备/时间mask同步。训练只用该阶段IMU有效fit人口；实际classes_均log(clip(p,1e-12,1))；fit_teacher_target consumer缺类sentinel沿用baseline -1e6，terminal_export consumer缺类沿用final RF log(1e-12)，分别登记不可混用。源evaluate_imu_oof的目标导出不再校准；bank在其消费阶段使用RF T3，不能因A4无温度而删掉源T3。固定划分下监督训练目标为fit预测，标签来源与非OOF状态明确记录。其他源小教师/非视觉教师按第10节依赖清单保留，RF不能代替一切非视觉专家。

## 8. A5 MoBind预训练（Motion Pretraining）

直接复用P86MoBindLite、原motion_fields/losses/train/evaluate。width96、alignment64、dropout0.12、imu_instance_normalization=false、imu_event_feature_width0、domain_classes0。24轮固定预算；batch64、seed20260811、LR4e-4/min2e-5、wd0.05、class power0.35、smoothing0.08、mask_ratio0.35。

semantic2、token0.25、local0.05、global0.15、reconstruction0.1、visual KD0.75/T2、teacher feature0.25、IMU teacher1/T1、contrastive T0.08。视觉features取view mean保持[N,2,1024]；目标为v3校准early_late。无额外预训练motion checkpoint，无新domain/adversarial/instance norm选项。

恢复源缺失损失行为：重建/局部对比及RF教师KL使用各自源mask；Skeleton/IMU监督CE、视觉KD等原代码按全拟合batch归约的项不因单分支缺失被自行置零。完全无有效输入的规范行仍用公开缺失策略。测试必须直接对照源losses数值/梯度，不能把v2“缺某模态所有损失零”当正确性标准。S/I是同一预训练模型的分支，不冒称两个大教师。

## 9. A6诊断与A7融合（Diagnostic Controls and Fusion）

A6校准等权平均、A7-mask/shuffle及推理zero仍保留为额外消融，不进入主链，不以其结果改主链参数；明确它们不是队友最终训练操作。A6若保留温度拟合，只用fit标签，并标注诊断意义。

A7主模型直接复用separate motion encoder、additive residual、原corruption、losses、selective_anchor_loss和train_stage：stage A4+B20、batch64、seed20260811、fusion LR4e-4、encoder LR1e-4、visual head LR2e-5、minimum1e-5、wd0.05、class power0.35、smoothing0.08、KD1/T2、relation0.1、motion aux0.35、selective anchor0.3。reliability/joint gate/IMU teacher辅助等最终未启用选项为0；视觉腐败0.75、feature dropout0.3、view dropout0.4。

主链按源行为计算motion_aux，不增加v2通用availability门控；受控消融需要的额外mask仅属于control记录。冻结视觉backbone/BN，motion encoder参数按源冻结；保留源码对motion Dropout/train模式的行为，不因requires_grad=false就一律eval。A2 anchor同phase固定，live_visual_anchor=false，选择性anchor只保护源规则判为正确的训练样本。

## 10. A8原后处理与教师依赖（Original Postprocessing and Teacher Dependencies）

取消v2自拟“两peer+cosine+margin混合”和有限transition grid作为主链。源实际链为P255分组→P270序列门控→P307/P309更新分组→P310优先覆盖。移植原函数并保留输入/标签边界，不直接执行旧main或读其runs文件。

teacher_roster逐项展开p88 CANDIDATE_SOURCES、p165 bank、p173新增项及P255/P307/P309 SOURCES。保留一个VideoMAE-Large及其six heads/full-window邻居、同一VideoMAE六clip特征的固定Ridge十项mechanism预测，以及不依赖被删大视觉根节点的小视觉/非视觉专家；删除的根节点及其依赖输出明确登记。源非视觉/物理专家的模型、特征和训练操作须按源码重建，不能拿A5分支随意替换。任一保留槽位缺对应producer/模态时停止该阶段，不用先验概率伪装可用专家，不凑30列。

学习KNN前必须建立P12子链：同phase Skeleton/Depth/IMU/Thermal基础模型、build_complete_p11_oof的S+D0.6/0.4与IMU混合、analyze_thermal_oof_fusion的温度/21点权重及evaluate_conditional_expert_routing的源features/sensitive/LR(C0.05,max_iter2000)/threshold0.5。P12八列按skeleton/depth/sd/imu/sd_imu/thermal/thermal_candidate/final顺序加六visual头，经原log_softmax与fit_simplex(softmax权重,L-BFGS-B/maxiter500/ftol1e-12)仅fit训练标签；再对各预测分区调用全窗unit_features及三邻居/self排除/0.40 smooth，绝不以A1/A7替换父概率或用假OOF包装。顺序为P12→global mixture→KNN→bank，学习KNN不阻塞Task4/5。

分组特征复用p137_group_classifier_selector.group_features：sqrt posterior、full layout、full retained bank、include_quality_features=false、embedding_lookup=None、原CONFIG及重复会话对齐；StandardScaler→LogisticRegression(C0.03,max_iter1000,lbfgs)，peer_weight2、balanced=false、class-frequency power0。原CONFIG保持rank distance2/start gap300/prob similarity0.75/path overlap0.20/length ratio0.80/consensus0.50/alignment penalty0.20/group size3。

序列复用p257.emission(p,base,0.65)及DecoderConfig(gap30,transition0.45,trigram_backoff1,beam50)，fit_transition_model的alpha=1。使用p139.gate_features/select_gate的五项评分、401阈值网格和七分位数、net/rescue/harm/changed决胜，不另造固定posterior>=0.80门槛。没有disagreement/合法会话时沿用源identity/threshold2结果，不计算空分位数。

源base/fallback必须锁定：safe=normalise(0.95*原safe-base producer概率+0.05*softmax(RF terminal logits/3))。各full40列的非detail行、各独立专家available=false行回退bank第0列，同时保留availability；不是复制假专家。P255/P309 group路由必须AND源detail_ids的visual_available，P270 sequence路由没有这个额外AND，P310仍按new!=old。不能猜detail_ids等于ir_available或任用A7当source_base。

原跨cohort阈值拟合在固定划分下改为只用fit预测/标签调用同一函数；不生成OOF，不跨val/final用标签。记录分组/阈值是in-sample固定划分变体，不声称outer-pure。repeat与session候选输入、元数据可在同一无标签分区联合构建，禁止跨partition/known user/date误连、利用类别目录或旧trial ID。该ID/分区适配属于固定人口必要差异。Task2新建公共时间sidecar(sample_id,user_id,partition,recording_date,start_seconds,timestamp_available,time_source)，原秒口径、opaque ID精确连接、未知保留NaN。Task11按(partition,user,date)外层分块调用原anonymous_date/group/repeat纯函数并映射回RowIndex，显式提供session index数组，不调用旧data[name].split.sessions；只填user字段不足以阻止源anonymous_date跨用户连边。

原优先规则：新分组结果与旧分组结果不同时覆盖序列结果，否则保留序列结果。保留源依赖减少后的自然退化，不能改规则凑提升。输出P310式目标：可用行argmax类0.9805，其余39类各0.0005，confidence0.9805；全缺失18行prior且target_mask=false。必须另存锐化前概率、old/new group及route，便于核对，而不是把普通A7概率当最终pseudo targets。

## 11. A9无标签适配（Label-Free Adaptation）

主端点恢复最终包文档实际40轮命令：seed20260826，scope=heads_motion_encoder，fusion/motion LR1e-4、visual head LR5e-5、minimum5e-6、AdamW wd0.02、T1、confidence_power0、warmup0、batch64/workers0。视觉backbone及BN冻结，执行原LabelFreePseudoDataset的时间增强和train_label_free余弦调度；目标为本run A8锐化概率，不读真值。

12轮端点仅保留为原基础调度器配方的额外诊断：seed20260812、scope=heads、fusion LR5e-5、visual head1e-5、minimum2e-6。两端点都从同一A7 refit独立初始化，不从12续到40，不按final成绩挑端点。主比较改为A9-40−A1；A9-12明确不是最终提交配方。输出raw概率，不再套A8。

## 12. 记录、校验与资源（Records, Verification and Resources）

禁止raw全目录SHA、整cache复扫、递归祖先payload校验、每任务重复全仓旧测试。阶段开始仅校验直接消费文件、shape/IDs/class order/availability/config、去重祖先记录角色；Stage Verification Context复用结果，epochs/batches不得触发资产扫描。大数组写入时摘要，load用header/mask/少量真实切片；变动输入才针对性重新核验。不把轻量验收说成全量内容验证。

v3角色表增加无fit的descriptor（source_ops/roster/source_transform/recording_metadata）和同phase teacher_bank；Task1在第一次导出descriptor ArtifactRef前先实现最小注册支持，完整角色表再由Task2扩展。源专家模型使用supervised_model，source_expert/<name>预测显式绑定其roster模型与phase/partition，不套旧未知stage必须adapted_model逻辑。学习global mixture/group/gate/transition保留fit范围；公开teacher-only assets由recipe单独绑定，不漂移v2 dataclass identity。未知角色和不合法父路径继续拒绝。recording_metadata.start_seconds是源日内秒，不是pixels的epoch秒。

公共VerificationContext按protocol/ref/path缓存record/直接文件/已加载数组/import memo；Task4六族拟合/保存/四区预测以及Task6/11共享，不能每次_load_features全文SHA再载或每次verify展开重复祖先列表。导入大数组消费只走header/ID/mask/切片，register与消费复用writer/import digest。小fixture验证六头只载一次和登记→消费无重hash。

每个CLI持续记录加载/计算/写出/登记耗时及批次进度。新门槛用小fixture读取计数证明raw SHA0、祖先payload0、阶段内直接文件只查一次；不为证明性能而执行全量SHA。保留ID错配、错误phase、标签进入推理、missing prior和状态回退的拒绝测试。

固定16候选及单向prepared→generating→frozen→revealed；候选相同ID609、类列40、全缺失先验一致。仅记录已授权独立覆盖检查的“是”；详细test类别支持在冻结后的评估中报告，不允许其他进程重复打开私有标签做覆盖确认。全部source producer/teacher roster闭合后才正式freeze，不用A8简化替代品冒充完成。

资源预检按实际保留的教师依赖更新，不沿用v2只需单视觉+RF的10–20小时估计；公开冻结特征/原始缓存优先精确复用。新增非大视觉专家的计算属于恢复原操作，不擅自跳过以缩短训练。文档修订不启动下载、训练或新自动化。
