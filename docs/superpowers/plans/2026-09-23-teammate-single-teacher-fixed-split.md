# 队友单大视觉教师固定划分实施计划（Fixed-Split Source-Parity Implementation Plan）

> 执行者使用 superpowers:executing-plans；用户明确选择子智能体实施时才使用 subagent-driven-development。2026-10-06独立复审修订为v3.1，Task编号仍为1–14。初版v3的机制producer、KNN依赖和元数据/导入/角色/校验契约已在下文纠正；正式生成仍受逐阶段数值与依赖闭合门槛约束。v2 Task1–5已完成并推送6e31b6d；它们是历史基线，不表示新增v3要求已完成。当前仅修订文档，v3训练未开始。旧规格/计划见reports/audit_inputs/2026-10-04/pre-v3/。

**目标（Goal）：** 一个大视觉教师、固定用户划分，在其余生效操作上按最终提交源码复刻完整路线。
**架构（Architecture）：** 复用源模型/算子，按新人口重新拟合；除教师根节点数量与固定划分替代OOF外，不随意增删归一化、掩码、温度、预算、后处理或适配参数。所有必要数据适配单独登记。
**技术栈（Tech Stack）：** 当前Python3.12、PyTorch2.7/CUDA12.8、torchvision、NumPy、scikit-learn、OpenCV、pytest。
**规格（Specification）：** [v3规格](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md)。
**差异清单（Difference Inventory）：** [源码一致性修订报告](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-04-plan-v3-source-parity-revision.md)。

## 全局边界（Global Boundaries）

- 工作树D:/work/2026.7.14_kaggle/_single_visual_processing_replication；分支experiment/teammate-single-teacher-task1。原始数据、队友快照与两份提交包只读。
- train12=2039，val user6/user7=388，refit14=2427，final4=609。refit包含val，不能把refit在user6/7上的成绩称val。
- 保留固定划分，不执行三折OOF，也不生成内部OOF。训练目标/温度/门控统计只来自同phase fit数据。不得读取旧OOF/专家预测、按final成绩调参或扩大fit用户。
- 独立子智能体经用户2026-10-04限定授权，仅检查final4类别覆盖，返回“联合覆盖40类=是”；未传回标签或分布，未计算模型成绩。此例外仅为覆盖核实，不给生成进程任何标签授权。仍不假定每名用户40类。
- v3 run_id=fixed-split-single-teacher-v3-source-ops；v2配置及模型保留。复用标签无关缓存需要新run记录、精确ID/轴/配置及来源，不改旧identity；监督模型/targets/学生sequence必须按v3重新生成。
- 大视觉教师根节点只保留VideoMAE-Large。保留同骨干六族头/全窗邻居、VideoMAE固定Ridge机制预测及独立非大视觉专家；teacher_roster对依赖被删根节点的排除逐项说明，不以A5、先验或重复列伪装原专家。
- source truth=最终(1)包实际命令/部署配置→调用路径的源码默认值。快照六个主文件已逐字节一致，其他新消费模块按source_ops_manifest直接核验，不全扫1167文件。
- A6、A7训练controls和12轮适配是额外诊断；不能替代主链或改变主链预算。主端点为最终配方A9-40。
- 保留16候选与最终一次揭示；此轮仅修改计划，不执行新准备、下载或训练。

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

- [ ] RED：test_final_recipe_overrides_defaults断言MC3主预算16、MoBind24、fusion4+20、40轮visual LR5e-5/scope heads_motion_encoder、12轮scope heads；test_teacher_root_roster断言一个大视觉根、六头不计多个大教师、独立非大专家不被任意删除。
- [ ] 从最终(1)命令及源调用导出所有生效参数，逐项附source path/symbol/line、声明是actual override或default。展开p88/p165/p173/P255/P307/P309依赖；被删根节点与依赖专家写明确原因。不替换不兼容输入，缺producer时给确定的失败清单。
- [ ] 新v3配置包含原population、source pin、asset绑定、新run ID与本规格参数；公开teacher专用资产从source_roster单独登记。旧v2 YAML/recipe/identity不改。
- [ ] pytest tests/test_no_vote_source_ops.py；小fixture只检查实际消费模块，不对全1167文件/公开资产重复下载或全量SHA。导出的recipe才是后续构造模型的唯一参数来源，不能另写一组隐式默认值。
- [ ] 提交v3协议实现；本Task的文档修订不等于已完成该代码步骤。

### Task 2：v3准备与缓存复用（Preparation and Cache Import）

**文件：** 修改no_vote_manifest.py/artifact_record.py及对应CLI/测试。
**接口：** import_public_cache(source: ArtifactRef, destination_protocol: NoVoteProtocol, rows: RowIndex, context: VerificationContext) -> ArtifactRef；build_recording_metadata(inputs: StageInputs, acquisition: ArtifactRef) -> ArtifactRef；学习产物无导入入口。

- [ ] RED：test_reuse_raw_only允许P28/P29、公开冻结features、pixels，拒绝v2 heads/targets/A2/sequence；test_import_exact_ids_axes拒绝缺ID/错用户/错窗口/错视野，旧identity不修改。
- [ ] 复制前三分区允许标签及四公共清单的明确准备过程；final标签不读、不复制到run。不重新扫描原始数据。新增保留专家模态的公共字段按白名单声明，禁止class/action/trial标签捷径。
- [ ] 导入固定采用同D盘硬链接（Hard Link）：新run创建payload路径；新登记的P28/P29/公共initializer/manifest父记录全部位于v3，旧ref只作为provenance字符串/旧digest凭据，不作为新ancestry父节点。按拓扑顺序导入并用memo复用公共祖先；禁止直接放宽registry外部路径或跨protocol规则。所有导入cache immutable、mmap只读，producer拒绝overwrite/r+写入。无法硬链接时明确失败，不偷偷复制多GB或重新提取。
- [ ] 定义v3专用Artifact Role Matrix：新增descriptor（source_ops/roster/source_transform/recording_metadata，fit_users为空）、teacher_bank（绑定同phase专家预测、[N,E,40]、availability/fallback及roster）；保留source_expert/<name> supervised_model与predict绑定自己的模型阶段及完整分区；mechanism/六头/global_mixture/safe_base/group_old/group_new/gate有明确stage与父记录。未知stage预测仍拒绝，不能退回“必须adapted_model”旧默认或给所有stage豁免。P12等teacher-only公开资产在recipe资产绑定单独声明，不给v2协议增加隐式字段导致identity变化。
- [ ] 新增公共recording_metadata.csv sidecar：sample_id/user_id/partition/recording_date/start_seconds/timestamp_available/time_source；start_seconds按原parse_time解析的日内秒(h×3600+m×60+s+microsecond/1e6)，不能直接传pixels中的epoch秒。从原时间内容或已恢复acquisition IDs提取，不从class/trial ID推断。精确ID连接，不使用旧metadata；未知保持date空/seconds NaN，后处理保留base。教师额外输入另行白名单，无label字段。
- [ ] 在artifact_record.py定义公共VerificationContext(protocol)，cache validated_records/direct_files/loaded_arrays/import_memo；keys包含protocol/ref/path。Task4/6/11与其CLI显式共用该context，不能只缓存features。祖先集合去重，注册后消费不再重hash；六头一次加载features，旧大cache消费只走header/ID/mask/少量切片，不能落入_load_features现有full-SHA分支。
- [ ] 新测试：test_import_hardlink_bridge_consumable、test_import_rejects_old_learned_parents、test_v3_expert_prediction_roles、test_recording_sidecar_label_free、test_six_heads_load_once、test_shared_ancestors_read_once、test_register_consume_no_rehash。小fixture直接消费完整导入链并验证raw SHA0，不用真实cache扫描。
- [ ] 运行标签/角色/ID针对性测试；final覆盖只引用已授权子智能体“是”，不能让准备CLI打开test标签再次检查。

### Task 3：几何与姿态源一致性（Pose and ROI Parity）

**文件：** pose_roi_adapter.py及tests/test_no_vote_pose_roi.py。
**接口：** 沿用build_pose/build_roi(inputs, ..., protocol=...)。

- [ ] 复用已验证的全量P28/P29；源ROI两轮、真实timestamp恢复、person/workspace裁剪不变。
- [ ] 一条train12 fixture与真实样本核对frame/acquisition/time、ROI quality和缺失fallback；未变部分不重新生成2427/609条几何。
- [ ] 仅修复v3导入契约的新失败，保留时间戳适配源码身份，禁止为原main兼容而读取类别路径。

### Task 4：恢复教师目标处理（Visual Teacher Rectification）

**文件：** visual_teacher.py、run_no_vote_visual_teacher.py；新增tests/test_no_vote_visual_teacher_source_parity.py。
**接口：** select_visual_head(train, development, features, protocol=...) -> Selection，主family固定early_late；fit_visual_head(...)->ArtifactRef；predict_visual_teacher使用模型自有T。辅助头/同骨干全窗特征另登记；fit_visual_mechanism_head(fit, features, protocol, context) -> ArtifactRef；predict_visual_mechanisms(model, features, rows, protocol, context) -> Mapping[str,Prediction]。

- [ ] RED：test_source_feature_head_selection复用原六族、每族12配置及源tie-break；test_calibration_train_only断言改val/final标签不改变T，原fit_temperature与本实现数值一致；test_ridge_class_alignment复用原floor规则。
- [ ] 复用六clip公开特征；72配置生产六族头，每族12配置按源决胜规则在固定val选择，主头固定early_late、不在六族间改主头。候选模型只fit train12，温度只fit train12预测/标签；refit继承已选配置并重fit refit14。训练软目标记录in_sample非OOF，选择指标记录development2，不混淆两者。
- [ ] 恢复head温度约0.1–10的源NLL估计，输出raw/T_head；select只train12，refit只refit14。保存温度、目标来源、raw与calibrated分数，不借用旧全局校准。
- [ ] Task4只生产同骨干full-window标签无关特征，按build_p85_videomae_fullwindow_cache，不用六clip平均替代。学习KNN依赖P12→全局混合，明确移到Task11；Task4不等待Task6/7/P12，防止执行循环。
- [ ] RED：test_visual_mechanism_ridge_source_parity对照audit_p86_teacher_mechanisms.fixed_model_perturbations与p89_build_multiexpert_test_logits.build_p86：VideoMAE六clip逐clip L2、固定StandardScaler→Ridge(alpha3000,power0.75,lsqr/tol1e-5/max_iter5000)、fit归一化特征均值、原十项特征干预、aligned_scores_40。独立小头只fit同phase fit人口，输出raw logits/T=1，不套A1温度，不消费MC3/像素/学生sequence。检查mechanism祖先是固定Ridge而不是A2；此头不增加大骨干数量。
- [ ] 源小型score/feature fixture逐元素验收，真实train12小片段记录目标熵/置信度；运行本Task测试。不得将置信度变高当作准确率提升的证明。
- [ ] 正式重新登记六族头、A1主头、先验及四区targets。v2的window_mean/T1保持原状，不能改记录宣称已校准。

### Task 5：源16轮视觉学生与新序列（Visual Student Rectification）

**文件：** visual_student.py/no_vote_datasets.py及CLI；tests/test_no_vote_visual_student.py/test_no_vote_sequence.py。
**接口：** train_visual_student保持select/refit接口；Selection.budget固定epochs16，metric仅描述val；build_sequence仍同phase。

- [ ] RED：test_source_fixed_budget_16不按val峰值改轮数；test_source_hybrid_loss_parity使用同一小batch、相同随机状态对比原生效loss与梯度，明确head T与KD T分离；test_v2_targets_rejected拒绝旧T1目标祖先。
- [ ] MC3结构、公有初始化、subject_robust、class_weights、源LR调度、batch4/accum4、singleton/GradScaler/clip-grad原样；保留当前已验收的锁与完整epoch恢复。partial不可用行按既定边界记数。
- [ ] select训练16轮，末轮一次val报告；refit公开重新初始化16轮。无OOF、无额外early stop或最佳epoch扫描。复用pixels，重建模型、四区标准targets及四份native sequence/anchor。
- [ ] Task5不生产十项teacher mechanism。源bank使用Task4固定Ridge在VideoMAE特征上的raw logits；不能用MC3的输入消融或校准early_late主头替代。
- [ ] 真实train12一批forward/backward/native sequence与源对照；同phase anchor对预测差在FP16容差内，推理删除教师/label文件仍可运行。只跑此Task回归。
- [ ] 正式生成后记录开发成绩及与v2的描述性差，不根据差值改变配方。此时仍不能读取final准确率。

### Task 6：原运动缓存及恒等外部转换（Motion Cache and Source Transform）

**文件：** motion_cache.py、build_no_vote_motion_cache.py、tests/test_no_vote_motion_cache.py。
**接口：** build_motion_source/windows保持原契约；describe_motion_transform(motion: ArtifactRef, protocol: NoVoteProtocol) -> ArtifactRef；apply_source_transform为恒等外部映射，保留源内物理处理。

- [ ] RED：test_no_added_global_standardization断言不读取fit labels、不拟合mean/std、输出字段与源cache/Dataset相同；test_source_window_alignment保留真实pixels索引与跨模态时间窗。
- [ ] 移植P31与P86所有MOTION_FIELDS/BOOLEAN/TEMPORAL轴、重采样、身体/旋转补偿及原mask；不把v2额外归一化实现进去。
- [ ] normalization父字段仅为确定性Source Transform Descriptor，标注imu_instance_normalization=false；全部消费者不再应用额外z-score。源模型内LayerNorm不删除。
- [ ] 合成时窗及一条train12真实样本与原纯函数对照。缺设备/Skeleton样本保存原mask，不零填一套伪统计。

### Task 7：源RF IMU教师（RF Teacher Parity）

**文件：** imu_rf_teacher.py、run_no_vote_imu_rf.py、tests/test_no_vote_imu_rf.py。
**接口：** train_imu_rf/predict_imu_teacher保持标准TeacherTargets。

- [ ] RED：test_source_rf_recipe断言400树/depth18/leaf2/sqrt/balanced_subsample/seed20260723；test_source_device_dropout核对原random_present_devices/drop_devices结果、标签副本和mask；test_rf_logits_export比对源log(clip(prob))，不新增温度。
- [ ] 32bin原始acc/gyro→原feature_vector240维+10mask，训练原样本+存在设备dropout副本。固定划分的select/refit均按train_final_imu_rf的无fold单次fit调用，dropout RNG=np.random.default_rng(20260723)，不虚构fold或seed偏移。仅fit对应阶段IMU有效数据。
- [ ] 同一RF输出两份有明确consumer profile的40列logits：给A5的fit_teacher_target导出沿用run_imu_stat_baseline缺类sentinel=-1e6；目标分区terminal_export沿用train_final_imu_rf缺类log(1e-12)，已存在类均log(clip(p,1e-12,1))。训练目标为in-sample非OOF，但不改consumer数值规则；normalization/temperature不能混用。源bank的RF T3仅在Task11消费时施加，不改变A4原导出。
- [ ] test_rf_consumer_sentinel_and_rng覆盖缺类、固定seed删除设备及fit/target两输出；Task7仅fit/导出RF，不等待P12的S+D融合，源train_final_imu_rf后续S+D混合拆到Task11。
- [ ] 类列classes_映射、invalid与prior分离，源IMU partial teacher valid原样；不要求为了凑40类而拿val样本补RF。源支持缺类时按同一对齐算法记录。
- [ ] sklearn小型fixture/真实一条stat验收；不加载p3_imu_oof历史张量。

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
- [ ] 合成完整会话/专家bank逐算子对照与一条train12接口验收；source_roster+实际参数+所有父记录齐全后才生成正式A8，不用简化替代品闭合流程。

### Task 12：最终40轮适配与12轮诊断（Final Adaptation Recipe）

**文件：** target_adaptation.py、run_no_vote_target_adaptation.py、tests/test_no_vote_target_adaptation.py。
**接口：** adapt_target(inputs: AdaptationInputs, endpoint: Literal[12,40], output: Path) -> ArtifactRef；无final label或accuracy参数。

- [ ] RED：test_final40_recipe断言最终实际LR1e-4/5e-5/min5e-6、scope heads_motion_encoder、seed20260826；test_base12_recipe断言12轮默认scope heads，不能错称两端点同scope。
- [ ] 原LabelFreePseudoDataset、configure_adaptation_parameters、train_label_free、temporal增强、余弦LR、confidence_power0、warmup0按源移植。40是主端点，12为额外诊断，各自从同A7 refit初始化。
- [ ] A8需本run完整保留bank/group/sequence/precedence祖先；Pseudo目标采用锐化概率。联合final609输入，591进入loss、18回退prior，不按最终准确率或confidence选行。
- [ ] 小型无标签batch逐梯度/参数组比对源函数；真实接口只用train12模拟target。训练范围拒绝label键，禁止把12继续训到40。
- [ ] 输出原始模型概率，不再执行A8；两端点都不访问final标签，主比较A9-40−A1。

### Task 13：完整冻结与一次最终评估（Freeze and Reveal）

**文件：** no_vote_evaluation.py及freeze/evaluate CLI/对应测试。
**接口：** freeze_generation(protocol, registry, candidates) -> Path；evaluate_frozen(generation, private_labels, output) -> Path。

- [ ] 保留16候选：A1、A2、A5-S、A5-I、A6-VS、A6-VI、A6-VSI、A7、A7-mask、A7-shuffle-S、A7-shuffle-I、A7-zero-S、A7-zero-I、A8、A9-12、A9-40。主端点改40，不根据final改名单。
- [ ] RED：test_freeze_requires_v3_source_ops拒绝缺roster/操作证据、v2目标、未闭合bank；test_exact_final_ids_classes_prior核对609/40列/18prior，类别覆盖布尔不代替样本ID校验。
- [ ] prepared→generating→frozen→revealed，不可倒退。freeze只看直接输出/去重元数据及配置，不扫描raw/全cache或打开标签。
- [ ] evaluator先持久化revealed再打开用户显式提供私有标签，报告acc/fixed40 macro-F1、实际test类别支持、各用户/模态/rescue-harm。不得从这一步回到新训练。
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

v2已完成：Task4单头T1开发69.07%，Task5第17轮开发48.71%、四区目标/序列、工程及方法学验收GO。v3新增要求未实现。进入实施时先Task1–3协议/复用→Task4目标恢复→Task5源16轮重训→Task6–14；不得把当前v2权重当v3已通过。

无定时自动化，Task5旧后台进程已正常结束。本轮不新建自动化，不训练，不推断final准确率。2026-10-06独立复审原版NO-GO；本次修订后还需原审查者确认文档问题闭合。放行仅指进入Task1–3实现与针对性验证，不代表已经证明全处理等价或可以跳过manifest/roster直接长训练。v2已退休两resume.pt和四sequence.npy（1.02GiB），正式checkpoint/targets/anchors/公共缓存/来源记录保留；旧seq refs为历史证明，不再宣称可直接完整加载。修改后自检：source_ops覆盖、接口/角色/硬链接导入一致、固定人口/无OOF无矛盾、每主链参数有source、每验证只消费直接输入。
