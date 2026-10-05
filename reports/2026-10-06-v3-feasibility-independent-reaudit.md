# v3.1 可实现性独立再审（Independent Feasibility Re-audit）

结论：**GO，仅放行 Task1–3 实施与针对性验证（Implementation and Targeted Verification）**。初次独立复审的三项 P1 与三项 P2 均已在计划层面闭合；本次没有发现需要再次阻止这三项任务启动的文档缺口。该结论不表示 v3 已实现、不表示算子数值已通过验收，也不放行正式长训练、正式生成（Formal Generation）或冻结（Freeze）。

基线仍为 HEAD `8d1f3943b05b76e81d71e5163b5ec82537219301`，审查对象是2026-10-06未提交的 v3.1 计划/规格 diff。初审记录见 [独立复审报告](2026-10-06-v3-feasibility-independent-review.md)。本次重点核实该报告的六条发现，没有扩大为长期项目或数据安全审计。

## 六条发现的处理结果（Finding Resolution）

| 原发现 | v3.1计划证据 | 独立核对与结论 |
|---|---|---|
| P1-1：误把十项机制预测当 MC3 学生干预 | 行18、46、102–107、119；规格行9、95 | 已改为同 VideoMAE 六 clip 特征的独立固定 Ridge：逐 clip L2、alpha3000/power0.75、fit均值、十项源干预、raw logits/T1，不消费 A2。与 `audit_p86_teacher_mechanisms.py:121–147,208–225` 及终端 `p89_build_multiexpert_test_logits.py:162–173` 一致。**文档闭合。** |
| P1-2：全窗 KNN 缺 P12/global mixture 上游和执行位置 | 行106、177、181–183；规格行97 | 标签无关全窗特征留 Task4；学习 KNN 移 Task11，显式 P12八列+六头→原 log_softmax/simplex→本分区三邻居/0.40 smooth→bank。P12锁定 S+D、RF、Thermal、温度/21点权重、LR路由及退化。与 `build_p85_multiexpert_submission_v1.py:29–48,88–108,147–164` 和 `build_p85_fullwindow_knn_teacher_v3.py:50–74,96–120` 依赖一致。**文档闭合。** |
| P1-3：opaque ID/time元数据及known user/session接口未闭合 | 行78、84、177、188–189；规格行105 | 增加精确七字段公共时间 sidecar；预测接口显式接收目标人口metadata；按partition/user/date外层分块调用原anonymous-date纯函数，再映射全局索引，直接传session数组，不调用旧data.split.sessions。源 `align_metadata:101–119` 的ID连接和 `date_session_lists:49–59` 的跨用户问题均有具体适配路径。**文档闭合。** |
| P2-1：跨run缓存无法通过旧registry路径/协议约束 | 行80、82、86；规格行38 | 同D卷硬链接建立新payload；新manifest/initializer/P28/P29父链，全新protocol记录；旧ref仅provenance；只读mmap、builder禁止覆盖/r+，按拓扑和memo复用digest，失败不重新提取。既保留 `artifact_record.py:79–82,127–128,185–197` 的边界，也给新consumer合法输入。**文档闭合。** |
| P2-2：descriptor、bank、独立专家预测与公开资产角色缺失 | 行71、83、86、180–181；规格行119 | 新descriptor无fit_users；teacher_bank同phase；source_expert预测绑定自己的模型、roster、phase/partition；global mixture/group/gate明确角色。未知stage继续拒绝；teacher-only资产纳入recipe，不漂移v2 dataclass identity。实际producer/资产/输入缺失会阻止生成。**文档闭合。** |
| P2-3：校验上下文只缓存features、重复SHA/祖先展开 | 行25–27、78、85–86、177、222；规格行117、121–123 | 公共VerificationContext缓存records/direct files/arrays/import memo，Task4拟合/保存/四区预测及Task6/11显式共享；禁止旧full-SHA加载支路、祖先去重、register→consume复用摘要，并有六头一次加载/共祖先一次读/不重hash的fixture计数。**文档闭合。** |

## 依赖顺序复核（Dependency Order）

最小修订成立：Task4先产生六族头、固定机制头及同骨干全窗标签无关特征；Task5可以只消费A1/pixels完成16轮学生；Task6/7随后提供motion/RF；Task11在自身S+D/Thermal producer与Task7 RF齐备后，建立P12、global mixture及KNN再组bank。Task4不再等待Task7/P12，因此消除了先前 Task4→Task7→Task6→Task5→Task4 的执行闭环。

P12入口现已写清：S+D模型独立skeleton/depth logits，RF logits/mask，Thermal logits/mask；保留原派生混合、残差及路由，不以A1/A7概率替代global scores。真实S+D/Thermal模型的最终配置、teacher-only公共资产与输入仍由Task1依据实际命令锁定；这是计划中明确安排的实施产物，不是本轮已有实证。任何必要独立producer未锁定时，行181的preflight拒绝条款须生效。

metadata的原秒数口径也有可核实来源：`build_p85_recording_metadata.py:77–86` 将真实最早时间分成日期与日内秒数。实施时应按该口径把现有acquisition时间转换到新sidecar，避免把UTC epoch秒直接写成日内秒。文档已经要求源口径与同块数值fixture，不需要重新读私有标签或类别/trial路径。

## 放行范围与尚须通过的门槛（GO Scope and Remaining Gates）

1. **Task1–3可开始实现。** source_ops/roster必须实际逐槽闭合：独立小视觉/非视觉专家保留；P231/P238/P253/P306等根据真实大根PATHS决定依赖排除；不得用名称、先验或A5替换专家。新增公开资产与teacher-only输入需要实际绑定。
2. **导入与新角色需真实小fixture验收。** 包括硬链接新payload、新协议祖先、旧学习产物拒绝、registry/loader完整消费、只读保护、时间sidecar精确ID及跨用户/分区测试。Task1出口descriptor的基础角色应随Task1–2实现顺序安排到首次登记前，不能在registry尚不支持时提前登记。
3. **成本约束需小fixture计数验收。** rawSHA0、祖先payload0、六头一次载入、共祖先一次读取、register→consume无重hash、epoch/batch资产复查0。不得用全cache扫描或整仓suite证明性能。
4. **正式生成仍须后续阶段门槛。** 源函数同输入数值/梯度、每Task针对性回归与规定真实train12小批次通过；manifest/roster/所有保留producer实际闭合后才允许正式生成或freeze。轻量header/mask/切片验收不宣称全量内容验证。

计划行230和规格行38已说明v2两resume及四旧sequence数组退休、正式refs作为历史记录保留。此次再审不读取这些退休数组、不依赖resume；也不将历史sequence refs当作仍可完整消费的cache。其余公开缓存的精确导入由Task2验证。

本次仅只读核对未提交文档diff及相关源函数；没有运行训练/下载/完整suite，没有raw/cache SHA，没有读取final私有标签、旧final评分或任何已退休payload；没有修改计划、生产文件或实验产物。仅新增本报告。
