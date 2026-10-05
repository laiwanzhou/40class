# v3 可实现性独立复审（Independent Feasibility Review）

结论：**NO-GO，当前文本不宜直接进入 v3 正式生成（Formal Generation）**。固定用户分区（Fixed Split）和单个大视觉根节点（Single Large Visual Root）的边界清楚；六族头、原后处理、固定训练预算和低成本校验方向也已恢复。仍有三项 P1：mechanism 的实际生产者被误读、全窗 KNN 的监督上游未形成可执行顺序、公开时间元数据与源分组函数的接口没有闭合。另有三项 P2 工程契约需要写清。

复审基线为工作树 HEAD `8d1f3943b05b76e81d71e5163b5ec82537219301`；审查的是尚未实施的 v3 计划，不把当前 v2 实现的差异直接当作 v3 已失败。只允许用户明许的一个大视觉根节点和固定分区替代折外预测（Out-of-Fold, OOF）；下列建议不恢复 OOF，不新增全缓存校验，不用简化算法替代源操作。

## P1-1：十项机制预测的生产者不是 MC3 学生（Mechanism Producer）

**计划位置：** [全局边界，行18](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-06/pre-review/plan-v3.md:18)、[Task5，行114](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-06/pre-review/plan-v3.md:114)；规格第10节也写“同一 MC3 的十项 mechanism”。Task5 明确要求“同学生”且“不新增训练模型”，这会把错误来源变成实施约束。

**源码证据：** 最终包 `audit_p86_teacher_mechanisms.py:19–28` 读取 VideoMAE 六 clip 特征并固定 `alpha=3000`、`class_weight_power=0.75`；`:121–147` 的 `fixed_model_perturbations` 对 `[N,2,3,D]` 特征做均值填补、窗口/视野交换与折叠；`:208–225` 对归一化 VideoMAE 特征拟合 Ridge，再对十种干预调用 `aligned_scores_40`。此处没有 MC3、像素输入或学生检查点。`p88_oof_candidate_ensemble.py:49–66` 的 bank 槽位恰好读取这一产物的十列。

**影响：** 在 MC3 上做相似干预，或者用 Task4 经 val 选参且校准后的主头直接代替固定机制头，都会改变专家语义，超出已允许的两项差异；还会产生错误的 A2 祖先。

**最小修订：** 将此生产者移到 Task4 或 Task11，写明固定 Ridge 与 fit 人口特征均值（Fit Feature Mean）；只移除源 OOF 外层循环，select 用 train12、refit 用 refit14 独立拟合固定头和均值，对各自预测分区生成十列。保持源 feature normalization、填补、类列对齐与原始 logits，不擅自套 A1 的温度。这个新增轻量头仍依赖同一个 VideoMAE-Large，不构成第二个大教师。增加真实源特征上的逐元素对照（Elementwise Parity），并断言机制预测祖先是该固定头，而非 A2。

## P1-2：全窗 KNN 还需要 P12 全局混合，上游缺口被 Task4 顺序掩盖（Full-window KNN Dependency）

**计划位置：** Task4 行102要求调用 `build_p85_fullwindow_knn_teacher_v3`，而执行顺序行62及215要求完成 Task4/5 后才推进 Task6–14。Task11 行173才笼统安排保留专家重新训练。

**源码证据：** `build_p85_fullwindow_knn_teacher_v3.py:23` 的 `DEFAULT_FUSION` 是 `p85_multiexpert_submission_v1/fusion_logits.npz`；`:96–105` 和 `:118–120` 对其全局融合概率施加 `smooth`，全窗特征只提供邻居。`smooth:67–74` 是三邻居、0.40 混合，并不是全窗独立分类器。该全局融合由 `build_p85_multiexpert_submission_v1.py:29–48` 的八项 P12 专家加六族视觉头生成，`:88–108` 用 `fit_simplex` 拟合非负全局权重。P12 又通过 `build_complete_p11_oof.py:196–243` 消费 Skeleton/Depth、IMU、Thermal 及原残差/路由链；不只有 Task7 的裸 RF。

**影响：** 仅做全窗特征提取不能生成 `p85_teacher`。若等 Task11 才生产 P12，Task4“正式闭合 fullwindow/KNN”无法完成；若用六 clip 平均、A1 概率或 A7 输出填入 fusion，则是更换算子。源 main 还要求旧 `oof_*` 字段并断言旧人口正确数2241（`:96–109`），不能直接运行或补假 folds。

**最小修订：** Task1 的依赖图（Dependency Graph）显式增加 P12 子链→`fit_simplex` 全局混合→全窗 `smooth`，逐项列出 producer、输入、参数及保留/排除原因。Task4只完成全窗标签无关提取和六头；将正式 KNN 概率生产排到保留 P12 producer 完成之后、源 bank 组装之前。用同 phase 的 fit 预测拟合原混合，对每个无标签预测分区单独应用原 KNN；移除旧 OOF 包装与固定历史正确数，保留纯函数。若某 P12 子项实际依赖被删的大视觉根，记录其依赖后果并按保留列拟合原混合，不能任意删掉独立小专家。

**P12应锁定的入口与输出（Producer Interface）：** Task1/11需明确同phase S+D模型的独立 `skeleton_logits/depth_logits`、Task7同phase RF `imu_logits/imu_present`、独立 Thermal模型 `thermal_logits/thermal_present`，及以下源派生过程：`build_complete_p11_oof.py:196` 的0.6/0.4 S+D混合、`:208–216` 的IMU校准/混合、`analyze_thermal_oof_fusion.py:162–175,189–230` 的thermal温度/21点权重网格和决胜、`evaluate_conditional_expert_routing.py:132–167` 的features/sensitive样本/StandardScaler→LR(C0.05,max_iter2000)/0.5路由及无二类退化规则。只拆出fit→apply接口，移除外层三fold遍历，不向这些旧loader补假folds。八输出顺序严格为 skeleton/depth/sd/imu/sd_imu/thermal/thermal_candidate/final，再加六visual列，给 `build_p85_multiexpert_submission_v1.py:147,163–164,199–205` 的原log_softmax→simplex→加权scores；KNN只消费对应分区的该scores与全窗features。S+D源训练路径为 `train.py`/`aligned_model.py`，Thermal路径为 `../thermal_baseline/train_imagenet_oof.py:23–25` 的ThermalOOFDataset/ThermalResNetTSM；具体最终配置、公开初始化及teacher-only输入由Task1从实际命令锁定，不沿用旧训练权重。该修订把学习KNN放Task11、全窗提取留Task4，能消除Task4等待Task7/P12、Task6又等待Task5的顺序闭环。

## P1-3：不跨已知用户的会话约束尚无传入路径（Metadata and Session Adapter）

**计划位置：** Task2 行81准备公共清单；Task11 行170只接受未定义的 `metadata`，行174移植 `p137.group_features`，行176要求同分区/known user/date。规格第10节同样要求避免跨已知用户连接。

**现有接口与源码：** `no_vote_manifest.py:17–18,29–35` 的公共白名单只有 opaque ID、用户、四模态路径及 availability，没有 `recording_date/start_seconds`。源 `audit_p87_sequence_decoder.py:101–119` 以 sample ID 精确对齐日期/开始时间，因此不能直接读取旧 ID 的元数据文件。`p137_group_classifier_selector.py:96–103` 调用 `date_session_lists`；`p89_global_repeat_decoder.py:49–59` 内部固定 `anonymous_date`，会把同日期的不同已知用户一起分 session/cluster。`p139_soft_sequence_gate.py:42–55` 的 `sessions_for` 又要求旧 `data[name].split.sessions`，当前 `StageInputs/RowIndex` 不提供此结构。

**影响：** 直接移植会出现 ID 缺失；以空日期/NaN 兜底会静默取消原后处理；只给源 metadata 填 user 字段也不能阻止原 anonymous-date 函数跨用户合并。该问题不是调阈值可以修复。

**最小修订：** Task2 增加独立公共时间描述文件（Public Timestamp Sidecar）及精确 schema：opaque sample_id、user_id、partition、recording_date、start_seconds、timestamp_available、时间来源。从已有几何/像素时间元数据及已核实解析器提取，不能解析 class/trial 或打开私有标签；未知时间保持 unknown。Task11 增加明确 adapter：按 `(partition,user,date)` 外层分块，把每块原日期/秒数传给源分组纯函数，再映射回原 RowIndex；sequence 输入改为这些显式 session 数组，避免构造假旧 cohort。对跨用户同日、跨 partition、未知时间、ID重排做小 fixture 对照。应记录这一适配属于已批准固定人口差异。

## P2-1：跨 run 缓存导入需要明确桥接记录，否则旧路径与父记录会被拒绝（Cache Import Bridge）

**计划位置：** Task2 行78–82规定原 ref、digest 复用和新 ref，但未定义旧记录是 ancestry parent 还是 provenance，以及旧 payload 如何被新 loader 消费。

**硬约束：** `artifact_record.py:79–82` 只读当前 run 内记录；`:127–128` 要求所有 ancestry 的 protocol identity 等于当前 run；`:185–197` 要求 payload 在当前 run 内。`visual_teacher.py:97–101` 还要求新 features 有 P29 和 public initializer 直接父记录。因此把旧 ref 直接作为父节点，或在新 ref 直接登记旧路径，均会失败。v2 公开 features/pixels 的正式 refs 存在；此次仅检查其元数据，refit features 记录4855个文件、2427行，pixels记录7个文件、2427行，没有读 payload 或依赖 `resume.pt`。

**最小修订：** 固定一种策略：新 run 用受控硬链接（Hard Link）建立允许缓存的本地 payload 路径，按旧凭据复用 digest；新 run 重新登记允许的几何/public initializer/manifest 祖先，旧 ref 仅在来源字段（Provenance）保存，不混入新监督 ancestry。若选择外部只读映射，必须明确 registry/loader 的专用进口类型和路径解析规则，不能放宽所有路径检查。测试应覆盖跨协议旧父拒绝、允许缓存导入后完整消费、旧学习产物/sequence拒绝、import SHA读取次数0。不得重新生成已完成缓存，也不得修改旧 identity。

## P2-2：新 recipe/roster/transform/专家预测缺产物角色表（Artifact Role Matrix）

**计划位置：** Task1 行67返回 source_ops/roster ArtifactRef；Task6 行121返回 Source Transform Descriptor；Task11新增专家预测与 bank；计划行34宣称保持原接口。

**现有注册器：** `artifact_record.py:19–21,95–101` 的 kind 集合没有显式确定性描述符；`statistics` 被视作需要 fit_users 的学习产物。`:150–160` 对所有未知 stage 的 `predictions` 要求同 stage `adapted_model` 祖先，因此原专家/mechanism/A4独立预测按自己的 stage 登记会被拒绝。`no_vote_protocol.py:102–103` 还固定只能有 videomae/mc3/yolo 三种初始化。新产物未实现属正常，但计划应给实施者可通过的角色设计。

**最小修订：** Task1/2写出 v3专用角色表：source recipe/roster/恒等 transform 为无 fit_users 的确定性元数据；专家模型为同 phase 学习产物；专家预测须绑定 roster producer、40类列、完整分区和自己的模型祖先；bank 为同phase专家集合。公开资产按 roster 另设 teacher-only bindings，不迫使 v2 protocol 字段/identity漂移。举例，独立 P12 thermal 源链有 `thermal_baseline` 输入与公开初始化（`build_complete_p11_oof.py:34–35,218–223`），当前四模态公共清单和三资产约束不能自动覆盖；被删根的 P231/P238/P253/P306 则不能因名称“physical”就误保留，其源 PATHS 实际消费 VideoMAEv2/InternVideo2。保留/排除须由真实根依赖判断。

## P2-3：校验上下文当前只缓存 features，Task4热点与 ancestry展开还未被约束（Verification Cost）

**计划位置：** 行26–30、Task14 行209要求每阶段核验一次并去重元数据，方向正确；但 Task4/6/11文件清单未明确接入共用上下文。

**现有行为：** `visual_student.py:103–108` 的 StageVerificationContext仅实现 `feature_table` 缓存，`records` 尚未用于 registry；`visual_teacher.py:94–114` 每次 `_load_features` 都 `verify_file` 全读SHA再加载所有 NPZ字段，`:155` 保存头时再次调用。扩展六头时若复用这一调用路径，会反复读取同一features。`artifact_record.py:122–143` 的 checked只活在单次verify中，且 `descendants` 在共享父 DAG 中反复展开列表；`:219` 登记后verify还会再次校验直接文件。它已避免祖先payload扫描，但不等于所有阶段“只查一次”。

**最小修订：** 在 Task2 定义公共 Stage Verification Context，将已核验record、直接小文件及已加载数组按 protocol/ref/path缓存；Task4六族选择、六头保存、四区预测显式接受同一个上下文。旧大数组导入和load路径使用 header/ID/mask/小切片而不调用现有 full SHA 分支；写出阶段复用 writer digest。祖先元数据遍历用唯一记录集合，不重复构建展开列表。用小fixture计数覆盖“六头一次加载”“登记→消费不重hash”“共祖先只读一次”，无需真实全cache扫描或全suite。这里不估计新增路线耗时；应先以最终闭合 roster 做 preflight。

## 最小放行条件（Minimal GO Gates）

1. 修正十项机制的 source producer，并为全窗 KNN补齐 P12/global-mixture 的 phase依赖与执行位置。
2. 给出 opaque ID/time sidecar、按已知用户/分区外层调用源group/session的适配接口与小fixture。
3. 在 Task1–2明确 import bridge、v3角色表/teacher-only assets、所有后续阶段共享的 bounded verification context。

上述修订完成且静态依赖图没有未解释槽位后，可 **GO进入 Task1–3 实施与针对性验证**；不因此宣称 v3训练已完成或允许正式 freeze。正式训练前仍须按计划验证真实 train12小批次与逐算子对照。

本次仅阅读计划、规格、源代码及少量正式产物元数据；没有读取 final私有标签或旧 final评分，没有训练、下载、全suite、raw/cache SHA或修改生产文件/计划/产物，也没有把 v2续跑文件当证据。文件引用行号按审查时内容。
