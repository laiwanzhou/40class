# P02 当前实验交接（Current Experiment Handoff）

更新：2026-10-29，Asia/Shanghai。工作树`D:/work/2026.7.14_kaggle/_single_visual_processing_replication`；分支`experiment/teammate-single-teacher-task1`，origin为`laiwanzhou/40class`。

本文件是本分支唯一交接入口（Canonical Handoff），后续原位更新。唯一指导为[P01计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)。历史文档在docs/archive/P04–P23，审计输入快照仅供历史证据，不作为新计划。

## 当前完成状态（Current Status）

**用户要求暂停（Paused by User，2026-10-29 20:47）：** 已停止核验身份后的Depth／Thermal训练进程PID47268，当前无此后台训练；不要自动恢复。Depth select已完成2／15轮，`run/task11_reconstruction_v2/p12_depth/select/resume.pt`已实际用CPU装载核验：identity相符，history连续1、2，模型／AdamW／scheduler／scaler／sampler及全部RNG齐，scheduler.last_epoch=2。恢复从第3轮开始；当前未完整轮的进度舍弃，不增加预算。Skeleton两阶段及FP16四区产物已完成并保留；Depth refit与Thermal尚未启动，Task11未完成、Task12未启动。暂停凭据 `run/protocol/task11_reconstruction_v2/pause.json`，总progress已标paused_by_user。恢复时先看进程，再使用原producer89c1930对应的同脚本与配置；日志改用新的resume日志，保留原formal日志。当前分支后续提交2f94ebe只增加纯融合算子与文档，未改运行中的image producer，因此不影响本恢复点。等待用户明确继续。


**最新续接（2026-10-29 20:45）：Task11近似复刻进行中，Depth／Thermal正式训练在跑，不要重复启动。** 主进程PID47268，代码producer `89c1930`（已推送）；日志 `outputs/task5_audit/P41-p12-images-formal.log`，错误 `P41-p12-images-formal-error.log`，PID `P41-p12-images-formal.pid`。Depth select第1轮347.69秒已完成，后续按15轮select→15轮refit→四区FP16导出，再Thermal。每完整轮resume保存模型/optimizer/scheduler/scaler/全部RNG和WeightedRandomSampler RNG，恢复只重做未完整轮。先看进程/日志/history，不能仅凭进程存在或编码100%认完成。运行脚本 `scripts/run_no_vote_p12_images.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml`；不要改正在运行的p12_visual_reconstruction.py，否则新进程identity会拒绝续跑。

Skeleton已完成原两15轮与FP16导出，权威指针 `run/protocol/task11_reconstruction_v2/p12_skeleton_current.json`。两FP32 master来源Git69eed6d SHA已验证，不按当前扩展后的源码重写旧record。四区完整2039/388/2427/609，valid1939/385/2324/590，prior100/3/103/19；原共同三模态帧/首人姿态人口，与Task8不同。FP16逐张量为master.half、整数计数不变；保留旧FP32模型和预测。验收 `outputs/task5_audit/P41-p12-skeleton-acceptance.json/.log`，由 `scripts/check_no_vote_p12_skeleton_component.py` 完成；不要重复全3千小缓存或大raw哈希。

本轮24项针对性测试GREEN30.66秒，含训练第4轮中断→15轮恢复与连续训练的模型/AdamW/scheduler/scaler/RNG/采样顺序等价；两真实32条train12批原Dataset差0，优化器更新前梯度有限、参数改变、AMP跳步0、峰值约2.8GiB。源Depth函数step后zero_grad，旧probe误判空梯度为溢出；已改optimizer pre-step hook，保留失败日志，无配方变更。真实成功证据 `P41-p12-images-real-batch.json` 与 `-accepted.log`。Thermal ImageNet V1已从C盘缓存跨盘复制到run并登记完整SHA/官方URL，初始化对原类逐参数一致。workers0、torch threads4的工程覆盖单独披露。

纯 `p12_fusion.py` 新增S+D/IMU设备权重/Thermal校准路由/global simplex/三邻居source算子，4测试GREEN24.34秒。尚未正式fit或生成P12/global/KNN；路由敏感目标只有一类时用原cross-fit函数常数fallback而非最终main的直接拒绝，实际触发时必须记录这个固定划分适配，不构造伪样本。不能把纯算子通过当完整bank闭合。

当前活动配置仍task11_reconstruction_v2/SHA39f5c5a5a67a6d128ae089dfc619ccfdb537aea92b595aaf9d2d4bd7c67ba89c，recipe.artifact.json稳定复用；不要因trainer扩展重登记recipe。当前总入口 `run/protocol/task11_reconstruction_v2/progress.json` 明确Task11 false/A8 false/Task12 false。P41已更新实际进展；后续待两图像教师/四区预测、正式P12八分量/global/KNN、中间source-base adapt12、safe base/bank/group/gate/sequence/A8。原Task1–10和严格旧预检保持历史，新独立复审未执行，不声称Task11 GO；final私有标签不读。main历史文档仍等全实验结束再更新。两个既有untracked历史报告不提交。


**最新授权/状态：用户同意改为近似复刻并要求继续工作。** 已冻结活动`configs/experiments/task11_reconstruction_v2.json`，基础protocol仍2b563c...、Task1–10复用，旧P39/P40与严格预检不改。当前P01顶部授权补充和[P41](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P41_Task11_近似复刻规格与配置冻结.md)指导后续；不能继续把缺历史config_used/decoder摘要作为无条件停工理由，也不能翻旧globalfalse冒充实际producer完成。

v2准备已成功10.27秒：P11 source Skeleton/Depth/Thermal JSON固定15末轮，Depth明确随机init，Thermal ImageNet V1；中间decoder30/.3/1/50/T1+source backoff、独立source-base adapt12仅本phase fit、不适配dev/final（与原target池不同，已显式登记）；safe .95adaptedraw+.05 RF/T3及原固定repeat配置。A18/P128含被删大根排除，20保留列。Thermal路径准备最初因公共行全部路径为空失败；v2保留unknown观察值，不反查私有mapping或猜不存在。四区found/knownabsent/unknown=1840/117/82、374/11/3、2214/128/85、574/17/18，完整分母不变。标准输入表unknown无可消费mask0，另有observation表真实保留unknown。receipt run/protocol/task11_reconstruction_v2/preparation.json。两个输入测试GREEN，下一步实现真实P12数据与教师训练；Task11仍未完成、A8未生成、Task12不推进，无新训练已启动。此前v1配置是失败准备草稿，不作为活动配置。

**当前最新：Task9／10三独立复审PASS（P39）；用户纠正后续为Task11不是Task12。Task11已推进会话/纯算子/谱系预检，尚未完成，正式A8阻断。** 当前review入口protocol/task9_10_independent_review_P39.json，Task9/10无未解决P1/P2，支持前十步继续复用；旧Task9 registry/roles证据SHA绑定4e148cb历史blob，当前共享校验器改变不改其旧complete。

Task11新增session_metadata.py、source_teacher_bank.py及preflight CLI，30 targeted GREEN14.34秒。四区公共时间sidecar实验收4.01秒，不读labels/预测或大cache；按(partition,user,date)调用原anonymous_date/session-list并映射全局索引，unknown时间不连边。P310优先覆盖/锐化纯接口已实现但无正式A8输入。task11_source_closure_v1.json是evidence-only版本化视图，parent原roster，明确A18实际含被删VideoMAEv2/InternVideo2而排除，旧协议/祖先不改。真实CLI预期exit2=BLOCKED_FORMAL_SOURCE_BANK，不是A8完成；无task11_complete。

**本轮阻塞材料：** P12 p11_fp16所选S+D配置/命令与公开init receipt；Thermal实际_fp16所选checkpoint/转换链绑定（runner只证明非_fp16输出），以及四区thermal_path/available公共sidecar；safe-base新发现依赖P85 KNN→原中间结构目标/decoder→独立adapt12→P87S raw→P89，不能直接A7或最终A9-12替代。三fold decoder摘要/固定划分参数来源和P89 joint grouping证据尚需闭合。A18配置本身存在，不再误报完全缺失；应排除其多大根输入而非A5凑列。详见[P40](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P40_Task11_来源闭合与会话适配.md)。Task11其余完整P12/global/KNN/bank/group/sequence未实现，Task12不推进，无新训练/自动化/final标签读取。以下“Task11未授权/未启动”等是此前历史。

**当前最新：Task1–10已实施，Task10八组4＋20、四区24targets及最终有限自验完成；Task11未启动。** 入口run/protocol/task10_complete.json；交付[P38](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P38_Task10_原融合与独立训练控制.md)。42 targeted GREEN43.74秒，另旧固定24恢复专项1GREEN9.36秒；真实64train12（含IR-only1）源Dataset/A/B loss/grad/update0；最终57.48秒验收八模型／配对初始化与每轮顺序／人口/CW／24小targets／prior。真实refit/full在64final公共输入的full/zero-S/I logits差0，教师目标/标签打开0、initializer参数装载0。final六候选均609/591valid/18prior，不评分。

八组保留训练时间1400.40秒，续跑训练＋导出流程1589.0秒；首次序列化失败后的full/select从A4末态重跑B20，旧新全部history除seconds外相同，未追加模型预算。主实现1e9648f、保存修复8dbd621已推送，最终收尾见Git日志。八模型均各自同phase A2/ir_supervision_v2 A5初始化；A7与训练控制不混祖先，zero-S/I共用full checkpoint。当前开发full210/388=54.12%，control181/195/198，zero187/205，只记录不据分数改配方。按用户要求全部任务完成后统一评估。

本轮无独立子智能体复审，不称独立GO；完整bank的P12/Thermal/P89/A18门槛、已有Task5精度残余继续保留。最新训练PID45888/session18327及验收session59416均结束，无后台训练或自动化。下一步P01 Task11尚未授权／实施；此前“Task10进行中／未完成”等段落保留阶段历史，以此最新段为准。source bool兼容子类应在后续A7使用中继续保留，不能直接换回Torch2.7下报错的源forward；旧原始数据、队友快照、Task1–9模型／完成记录不改。

**当前最新：用户授权推进Task10，训练前41项针对回归及真实64批源对照通过，八组正式4＋20训练进行中，Task10未完成。** 实现源separate/additive/clip，直接调用train_stage，四独立full/mask_motion/shuffle_s/shuffle_i，select/refit均只用自身A2和task8_current指向的新A5。运行日志outputs/task5_audit/P38-task10-formal.log；后续先查日志、进程及各A7*目录artifact，不重复启动。Stage A保存阶段级恢复，B未完成时从A末态重跑B。Task10完成需八模型、四分区六候选24targets、最终有限验收与complete，不能只看full结束。交付进行中[P38](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P38_Task10_原融合与独立训练控制.md)。

续跑更新：首次full/select算完但保存最终checkpoint因nested mappingproxy不可pickle退出，原session41747已结束。共享_save_torch元数据序列化修复已42tests GREEN43.74s；主融合producer/训练identity不改，不动既有模型。新session18327从已有A4/RNG恢复后重跑B20，再继续其余七组；最新日志outputs/task5_audit/P38-task10-formal-resume.log，旧formal错误日志与小history副本保留。先查新进程/日志，不重复启动；当前仍未完成Task10。

源三维safe_mask索引在Torch2.7失败，只在适配子类将首padding列与empty做OR，独立flatten参考真实梯度/更新差0；不改快照。Task10消费quality和anchor按源FP16舍入再FP32，原Task5产物不改。真实64条含IR-only1，A/B loss/grad/update0，B原GradScaler首步overflow按源自动恢复后有限；峰值596.44MiB，17.40秒。41tests GREEN43.21秒。原数据/快照只读，global false/P12等门槛继续保留，A6/训练控制均禁止回流主链；Task11未启动。以下“Task10未启动”是此前Task9完成时历史。

**当前最新：Task1–9已实施，Task9额外平均融合诊断已完成四区12targets及有限自验；Task10未启动。** 本次用户要求先完成Task9，所有Task完成后再统一评估。A6消费Task5 A2和task8_current.json指向的ir_supervision_v2 A5-S/I，identity/fitted两策略只在dev A6-VSI选择；选中fitted，refit只在refit14重估温。33 targeted GREEN22.20秒；正式生成18.58秒，自验22.42秒。final4每候选609行、591有效／18prior、不评分；无网络重训或大缓存扫描。

当前Task9入口为run/protocol/task9_complete.json，交付[P37](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P37_Task9_额外平均融合诊断.md)。诊断角色diagnostic_not_source_main，注册器阻止其进入A7监督祖先及A8主bank；global false未改变。Task9本轮只有有限自验，没有独立子智能体复审，不能称独立GO。开发记录VS184/388、VI183/388、VSI196/388，仅作阶段记录。下一步是P01 Task10的源融合与控制，不以A6结果改主预算。旧前置完成标记/P35/P36保持原样；以下段落中的“Task9未启动”等均为此前历史，当前以本段为准。

最新：**Task8修正版ir_supervision_v2已完成两阶段24轮、四区输出及独立GO，P34-M1对新模型CLOSED；Task9未启动。** select1957／refit2342、各恢复1IR-only，训练与class_weights人口一致；新dev S143/388=36.86%、I112/388=28.87%、combined诊断151/388=38.92%。30 targeted GREEN86.29秒，真实两64批含恢复行的源loss/grad/update差0；64final公共行实际模型禁teacher/labels打开0、logits差0。独立工程／方法／产物三者完成，最终P36确认新模型及12targets和旧证据完整。交付[P35](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P35_Task8_IR监督恢复与重训.md)、复审[P36](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P36_Task8_IR监督修复独立复审.md)。当前入口protocol/task8_current.json→task8_ir_supervision_v2_complete.json；后续必须用其refs，不硬编码旧A5。

新模型根run/A5/ir_supervision_v2/{select,refit}，独立capability/ready/complete。旧A5／原Task8 complete／P33／P34保留原身份和条件性历史，未追认修复。Task1–7可复用，全局false/P12等缺项与ROI精度残余仍在；不自动推进Task9。训练select241.91s/refit283.05s，日志跨度562.26s；PID39204/session52812已结束，无后台训练或自动化。源码修复281e770已推送，最终记录提交见Git日志。以下为阶段历史状态，当前以上述新入口为准。

最新授权：用户已明确要求恢复IR-only并重训Task8。该授权后已完成修复，人口改为 `sk|imu|同phase A1.valid`，select1957／refit2342，各恢复1条并同步class_weights；新模型／输出写入 `A5/ir_supervision_v2`，capability／ready／complete均独立版本，旧A5/P33/P34不覆盖。Task1–7复用，仍各24轮随机重启，无val调预算；现已在新版完成重训与独立复审并关闭P34-M1；旧记录保持历史。Task9未启动。

最新独立复审（2026-10-29）：**Task6／7 PASS，Task8 CONDITIONAL，P34-M1未解决。** 三个独立子智能体完成工程／方法／产物审查，详见[P34](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P34_Task6至8_独立复审.md)。A5按motion union剔除83／86行中各有1条IR-only仍具A1监督，改变源CE/KD／类别权重与batch人口；不能把事后披露当明确批准。建议恢复该行训练并版本化重生A5后复审，或由用户明确接受额外人口适配。本轮仅审计，不修改生产代码／模型，不代替用户接受，不推进Task9。Task1–7可复用。原task8_complete仍是自验完成历史，新的task6_8_independent_review_P34.json记录条件性审查，优先读此状态；下述“已完成／未独立审”属于当时实施记录。

工程独立CPU恢复等价加强验证PASS（连续24 vs第7轮中断，dropout/shuffle/三RNG数据，model/optimizer/scaler/RNG/history一致）；P3完整缓存复用不重查raw IMU变化，仅在原输入变化后触发，无当前漂移证据。产物独立核验实际模型、mask/IDs/12份A5及8份RF预测与允许开发指标；未读final标签／重训／GPU复验／全cache扫描。

Task8实施收尾记录：**当时v3.1 Task1–8实施完成，Task9未启动；当前独立方法结论见上方CONDITIONAL。** Task8源MoBind select/refit各24轮末态与四区S/I/combined诊断共12份输出已齐，27项针对回归GREEN68.29秒、两真实64样本批次源Dataset/损失/梯度/更新差0、实际无教师/标签载荷推理64行差0，有限自验通过。开发S35.82%（139/388）、I28.61%（111/388）、combined诊断39.69%（154/388），final不评分。本轮无独立子智能体复审；权威入口task8_complete.json，交付[P33](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P33_Task8_MoBind运动预训练.md)。Task1–7、全局门槛与ROI精度残余保留，旧段落中的未启动状态仅为历史。

Task7收尾记录：**当时v3.1 Task1–7已完成，Task8未启动。** Task7源RF／四区双profile targets与有限自验已完成，29项针对回归GREEN48.46秒、真实source32统计／mask源差0、正式自验5.30秒；本轮未执行独立子智能体复审。select训练1914有效样本＋设备dropout副本，refit2283＋副本；开发123/388=31.70%，final609／580有效／29 RF-prior，不读取final标签。source_expert/imu_rf单独readiness不放A5/P12或改global false。权威入口task7_complete.json，交付[P32](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P32_Task7_IMU随机森林教师.md)。Task1–6产物保持有效；下面的“Task7未启动”为此前Task6收尾时的历史状态。

Task6收尾记录：**当时v3.1 Task1–6已完成，Task7未启动。** Task1–5保留此前实际独立GO；2026-10-29 Task6按用户授权完成运动缓存／恒等外部转换，40项针对回归及补充源Dataset对照通过，正式refit14=2427／final4=609的源和窗口缓存、两个source_transform已齐，有限自验通过；本Task未请求或执行独立子智能体复审。权威入口`run/protocol/task6_complete.json`，交付[P31](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P31_Task6_运动缓存与源转换.md)。PID24744/session8676已退出，无后台训练或新自动化；用户只授权到Task6。

2026-10-29用户批准补充完整知识转移的评估契约（Knowledge-Transfer Evaluation Contract），已原位写入P01及Task11–13：保留主比较A9-40−A1，新增A8−A1／A9-40−A8／A9-40−A7；Task12端点后无增强eval输出教师—学生一致率，分别报告609全体／591适配／18prior并绑定A8目标与A7/A9引用。100%一致不设门槛、不追加训练或选端点；准确率和rescue/harm只在Task13冻结揭示后计算。保持user6/7未参与伪标签适配，区分开发成绩和目标输入已参与适配的结果。仅文档／后续评估输出修订，不改YAML、协议或producer；Task1–5正式产物及原完成标记／复审继续复用，ROI精度差异与P12/P89等后续门槛仍独立保留。Task6未启动，没有新训练或标签揭示。

2026-10-29用户要求再次对照最终(1)包并注明差异，结论已原位补入唯一[P01的再核对章节](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)。23个直接小源码中21个与快照字节相同，P89/P310两文件仅无效元专家／旧报告比较不同，当前相关生效算子未改。新增发现：源pixels质量值FP16存储后转FP32，当前直接FP32，三条train12抽样舍入差最大2.44e-4；已列待闭合精度差异，本次不改缓存或模型、不重训。固定划分另带来in-sample目标、校准来源／refit重新估T、val选参及ID批次顺序等差异；不再无条件宣称只两项差异且全链完全一致。P28/P30与完成标记保持历史原样。Task5基础视觉完成，Task10/12仍会训练视觉头；Task6未启动，P12/P89/A18全局门槛不变。

Task1、Task2＋3、Task4和Task5均经实际独立复审GO。Task4完成全部六族select/refit头、两个先验、两phase固定Ridge机制头、四区68组targets/raw scores与两个完整全窗特征。Task5两阶段固定16轮与四区预测／原生序列已完成，Task6现已完成源运动缓存及恒等转换；此前两条2026-10-29修订记录中的“Task6未启动”仅指其修订当时，当前状态以上方最新项为准。

Task4代码已推送`202d57c`；最终完成文档的提交号见本任务ledger和Git最新日志。ledger仍位于`.superpowers/sdd/2026-09-23-teammate-single-teacher-fixed-split/progress.md`，保留旧目录名以连续记录；首行指导路径已更新P01。

当前run：`outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops`。协议SHA：`2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。当前审查入口为`run/protocol/task6_8_independent_review_P34.json`，Task8为CONDITIONAL；原实施完成入口`run/protocol/task8_complete.json`保留自验历史；`task7_complete.json`继续绑定RF模型／双consumer目标；`task5_complete.json`继续绑定学生模型／四区产物与独立验收，`task6_complete.json`绑定公共运动缓存及自验；`task4_complete.json`仍保留此前视觉教师头部和全窗特征的完成依据。旧`task4_heads_complete.json`的fullwindow_completed=false仅保留头部完成当时事实，不重写旧身份。

## Task4结果与验收（Results and Acceptance）

主头固定early_late，选定alpha300／class_weight_power0.75；select/refit温度分别为0.1206300185060063／0.1397821300164829，均仅由本phase fit人口估计。train12为1985/2039，development2为268/388（69.07%），refit14为2355/2427。refit包含val，不能称val成绩；final4不评分、correct=null。

六族每族12格，共72格按源tie-break选择；主头不在六族间再选择。十机制源固定Ridge(alpha3000/power0.75)及同phase L2均值，保持raw T1，未使用MC3或旧序列。训练目标明确in_sample/non-OOF。

全窗为真正0–1 uniform16、scene/person/workspace原裁剪与公开冻结VideoMAE编码，不是早晚特征均值。refit14特征float16[2427,3,1024]、logits[2427,3,400]、valid2342；final4为[609,3,1024]/[609,3,400]、valid591。两个根在`run/fullwindow_features/{refit14,final4}/full/artifact.json`。真实编码2034.8＋319.0秒，合计39.2分钟；初始化48偏置张量maxdiff0，模型在两区共享。

最终49项受影响测试通过；头部独立核验68组targets/raw与102记录，ID/phase/先验/raw÷T/配置继承/无A2祖先均通过。全窗独立核对完整ID/数组头/valid、每区首行和8个trial小切片、验收样本对正式样本逐值一致、共享公开初始化父及源参数。大feature全表重读0、大cache SHA0、GPU重跑0、final标签读取0；不外推未抽查payload全部完整或有限。

交付与证据：

- [P26 Task4交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P26_Task4_视觉教师目标恢复.md)
- [P27实施前复审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P27_Task4_实施前独立复审.md)
- [P28最终产物独立GO](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P28_Task4_产物独立验收.md)
- `outputs/task5_audit/P26-task4-final-tests.log`、`P26-task4-heads.log`、`P26-task4-fullwindow.log`
- `outputs/task5_audit/P26-task4-fullwindow-acceptance.json`：自验header/ID/mask/首行/小trial，验收对正式特征差0。

审查额度曾在实施前汇总时中断，随后恢复并真实完成两轮产物验收；从未把额度受限当作通过。

## 门槛与下一任务（Readiness and Next Task）

旧全局门槛把P12/P89缺项阻断全部学习，与规格Task4/5不等待P12冲突。本轮独立认可的修订是`visual_readiness`描述符绑定source_ops、Task3证明与5个消费源pin，只放行A1/六头/class_prior/固定机制；没有改变YAML、旧协议身份或全局`formal_generation_ready=false`，不放行P12/专家bank/融合。

Task5已按P01完成源16轮MC3、新targets消费和同phase native sequence/anchor，实际正式产物已独立验收GO。Task6现已完成raw运动缓存及公共确定性source_transform；Task7源RF现已完成，Task8现已完成源MoBind，下一任务是Task9诊断，未启动；RF-only readiness不把A5或完整P12融合一并放行。student_readiness仍只绑定Task4 complete和A2本phase目标，不把旧v2学生权重当作v3已完成。P12 S+D最终配置、Thermal最终配置/公开资产/额外公共清单，以及P89 safe-base/A18完整producer证据仍未闭合，完整teacher bank正式生成及最终冻结仍受阻断；不得填默认值、删必要专家或直接改true。

## 数据与复用边界（Data and Reuse Boundaries）

固定train12=2039、val user6/7=388、refit14=2427、封存test user4/17/23/24=609。训练/校准/门控仅fit本phase；不做OOF，不读旧专家预测或用final分数调参。此前用户只授权独立子智能体返回final联合40类覆盖“是”，不允许再次读取标签。

Task2只复制四公共清单和前三分区允许标签，未读/复制final标签；P28/P29、公开冻结features、pixels和公开initializer按同卷硬链接新登记，新祖先全部v3，旧ref仅provenance。独立共享VerificationContext修复了for key覆写缓存键导致重复JSON解析的问题。Task3一条22帧14ROI字段/原crop/quality/time逐值一致；原timestamp恢复函数与两轮操作不改。详情[P03](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P03_Task2_准备与缓存复用.md)、[P24](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P24_Task3_姿态与ROI一致性.md)、[P25](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P25_Task2_Task3_独立复审.md)。

原始数据、队友快照和两份提交包只读。main历史文档等待完整实验结束再更新。旧v2 69.07%教师/48.71%学生保持历史；两个resume和四旧sequence数组已清理，不能把退休sequence作为新祖先。仍保留公共几何/特征/pixels/权重、正式checkpoint/targets/来源记录。两份原有未跟踪历史报告不顺手提交。

## 校验成本（Verification Cost Bound）

不重新SHA原始目录/全images/features/sequence，不递归读祖先payload，不每Task重跑旧全仓suite。新直接输入只查一次、祖先元数据去重、数组只检查header/ID/mask和消费切片。v3辅助头返回Prediction，主A1用显式bounded TeacherTargets，旧默认full检查保持；InitVar不纳入旧dataclass fields/协议身份。不要为“证明不全扫”实际先做一轮全扫描。


## Task5完成（Task5 Complete，2026-10-29）

用户要求Task5完成后独立复审，已全部完成：select与refit各16轮末态，refit公开重新初始化，不按val挑epoch。四区标准targets和原生sequence/anchor已齐，整体完成入口`run/protocol/task5_complete.json`绑定两模型、四区产物、ready摘要及最终独立报告。旧ready的review=pending保留生成时事实，不修改旧身份；以新的complete入口为准。Task6未开始。

学生train12=1975/2039（96.86%，有效IR1945/1957）；固定user6/7开发=182/388（46.91%，macro-F1 .3521767785）；refit14=2343/2427（96.54%，有效IR2313/2342），含val用户不能称val；final4=609行、591有效／18prior，不读取标签、不评分。旧v2学生最佳189/388=48.71%，当前低7条（-1.80个百分点），未观察到提升；同时改变了预算／选择规则／目标校准等，不能据此断言单一原因或追加调参。

实际训练select3661.98秒＋refit4368.81秒=133.85分钟；四区预测／sequence联合导出399.41秒=6.66分钟，总计约2小时20分钟，不含前置实现测试。PID37608/session18655已结束，不可重复启动旧任务。正式日志`outputs/task5_audit/P29-task5-training.log`以source_student_pipeline_complete结束。

源代码已提交推送0630ce7；完成文档提交见最新Git日志／ledger。53项针对性测试通过81.10秒；真实train12批次loss／参与梯度／anchor与源参考差0。完整恢复夹具证明16轮末态／只末轮一次val／refit重新初始化，中断恢复不重复完整epoch。单独student_readiness只放A2且绑定Task4 complete、源pins、pixels、MC3公开init和本phase校准early_late目标；共享描述符无学习祖先，实际模型直接ownphase教师父。

最终独立审查者实际核验两模型／16轮history／源配方、ownphase祖先、四区targets完整IDs／40类／prob／valid／prior、四份native `[N,2,3,16,512]` float16 header与只读mmap、全部small anchor和prediction逐值一致，以及16条sequence小切片；开发指标独立复算一致。真实正式refit模型禁读teacher／label／public initializer载荷推理通过（模拟不可用、未删除真实文件），打开尝试0，native和直接anchor差0。未读final标签、未扫描大sequence／pixel／feature或raw。整个Task5 GO，无未解决P1/P2；未外推未抽payload完整性、独立重训或后续Task6+。

详情：[P29交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P29_Task5_固定16轮视觉学生.md)、[P30最终独立GO](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P30_Task5_独立复审.md)。证据`outputs/task5_audit/P29-task5-final-tests.log`、`P29-task5-real-batch.log`、`P29-task5-final-acceptance.json/.log`。inference_only protocol保留同receipt/pins/identity，只不要求未消费public teacher/init文件存在；实际checkpoint/pixels/prior照常校验。

协议身份仍2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a，全局formal_generation_ready=false不变，P12/Thermal/P89/A18材料门槛仍约束后续teacher bank／融合。Task6已按后续授权完成，Task7现已按后续授权完成，Task8也已按后续授权完成，下一步只有收到授权才进入P01 Task9，不按本轮低分改配方。保持唯一P01计划／P02交接；原数据、快照及两提交包只读，main历史文档仍等完整实验分支结束。

## Task6完成（Motion Cache Complete，2026-10-29）

新增motion_cache.py与构建／真实样本验收CLI，原P31骨骼／IMU、P46身体旋转／设备补偿、P86精确pixels窗口和12字段／BOOLEAN／TEMPORAL语义保留；原窗口FP16/uint8、global stat FP32，附带绝对时间元数据FP64。source_transform为公共descriptor、external identity、cross_sample_fit=false、imu_instance_normalization=false，没有fit标签或mean/std。五个直接源模块与最终包字节一致。

refit14=2427（Skeleton2341／IMU2283／全缺失85），final4=609（590／580／18）；各两阶段均完整。train12/development2按精确ID复用refit14公共缓存，不做跨人口统计，后续拟合仍仅限本phase用户。当前无IR行也无运动输入，保留全false mask；其他输入若无IR但有motion且缺源timeline则拒绝，不伪造时间。两区各1条counter-only/no-IMU使用源10Hz回退，其他非空行用已恢复acquisition时间。

7缺接口RED→GREEN，timestamp验证RED→GREEN，FP64时间存储RED→GREEN。40项motion/roles/cache-import初轮GREEN49.74秒，源Dataset扩展后最终同40项回归GREEN46.07秒，扩展单测另GREEN3.66秒；源全NaN Skeleton fixture保留源warning且mask/finite通过。真实train12一条22帧/102IMU点，12窗口字段源差0、timestamp0、区间点数守恒，标签载荷打开0，6.52秒。正式源/窗口计算合计102.15秒，日志跨度约125.24秒；登记载荷273.67MiB。有限产物自验0.88秒，完整IDs/headers/mask/源索引/小时间表、首中尾与缺失行小切片、两个transform绑定及训练/开发子集均通过；无大数组SHA／完整finite复扫、视觉重跑或final标签读取。

完成入口task6_complete.json绑定ready、P31交付及自验／真实样本摘要。正式缓存run/motion/{refit14,final4}/full/{source,windows}；normalization见ready中的source_transform引用。大数组后续用load_motion只读mmap/header/stamp／消费切片，不能调用完整SHA路径重扫。日志outputs/task5_audit/P31-task6-formal.log，最终小证据P31-task6-final-acceptance.json/.log；无独立Task6复审，不冒称独立GO。Task7需自己的RF源配方／fit范围与阶段门槛验收。

## Task7完成（Source IMU RF Complete，2026-10-29）

按源build_imu_cache的独立32点IMU重采样生成240统计＋10mask，不用Task6 P31/16帧/48global统计替代；重复时间戳／至少2点／CSV>=21列等原规则保留。四RF直接源模块与最终包字节相同。400树/depth18/leaf2/sqrt/balanced_subsample/n_jobs-1/seed20260723，存在设备dropout只用原seed、标签复制，无fold偏移／温度拟合或val调参。RF输出registry stage为source_expert/imu_rf、磁盘目录A4，TeacherTargets.features=None；双profile分别缺类-1e6和log1e-12，T1，bank T3留Task11。

新增imu_rf_readiness只绑定Task6 complete、公有source_ops与refit14/final4 source32统计，无学习共享祖先；模型直接用本phase标签，先验保存在RF模型配置内。原协议身份/global false不改，P12/Thermal/P89/A18门槛独立保留。Task1–6与ROI精度残余不变，main历史文档仍等完整实验结束。

train12=1948/2039（有效1910/1914），dev=123/388=31.70%（有效119/369），refit=2307/2427（有效2265/2283）；refit含dev不能称val。final609行580有效/29RF-prior、correct=null；全模态缺失仍18，不能删除其余RF缺失行。正式统计12秒、两个RF拟合保存2.30+2.70秒，日志跨度26.74秒，模型与8预测约72.78MiB、统计2.64MiB。

29项RF/roles/motion最终回归通过48.46秒，既有全NaN Skeleton fixture保留1个源warning；真实train12 source32[5,32,10]、五设备、240/10源差0且标签打开0，3.4秒。正式有限自验5.3秒核对models/phase/400树/真实dropout、完整ID/类列/valid/prior/8profiles及每区首中尾有效model推理和允许标签指标；未读final标签／旧OOF、未重扫大cache／重复fit，未执行独立Task7复审。下一Task8需独立的源配方／fit范围验收，本Task不自动推进。

## Task8完成（Source MoBind Complete，2026-10-29）

源P86MoBindLite width96/alignment64/dropout.12、无instance norm/event/domain；源loss／class weights／时间增强、AdamW4e-4→2e-5/wd.05/batch64/AMP/clip2，固定24无val选轮，refit随机重启；完整epoch模型/optimizer/scaler/全RNG恢复与锁保留。新Dataset只optional训练标签，先FP32再view mean给视觉[N,2,1024]，A1校准early_late＋A4 fit_teacher_target同phase精确引用，RF无效清零保留mask。不自行屏蔽部分缺分支CE/KD。motion_readiness共享descriptor只有公共motion/identity transform/source_ops祖先，学习目标仅body分phase绑定，具体A5带ownphase目标；仅放A5 fixed24，不放A7或完整P12。

train fit1956（排除83无运动，保留42partial），refit fit2341（排除86无运动）。完整人口S/I独立valid/prior；源无combined学习头，额外available-mean输出只诊断，不进主bank或16候选。train S1793/I1931/combined1955，分母2039；dev S139/I111/combined154，分母388；refit S2129/I2297/combined2317，分母2427。final609行S590/I580/combined590有效，分别19/29/19prior，不读标签／不评分；全模态缺失仍18，不能删额外motion缺失行。

核心8TDD testsGREEN18.20s（正常/缺I/缺S逐loss+gradient、fresh config、Dataset同步和教师、epoch2中断→24、人口拒绝、combined）；两真实GPU批次128train12全部Dataset/teacher/augmentation/loss/gradient/更新差0、非有限梯度0、峰值737.39MiB、7.70s。最终27 targeted GREEN68.29s，旧RF禁oof测试原误拦Python源文件，缩为禁OOF载荷扩展名后通过；未放松旧预测／final标签限制。4条Transformer norm_first库warning保留。

select训练83.48s/refit101.70s，共185.18s；日志跨度213.37s。PID39904/session51895已退出，12targets齐，模型＋输出29.97MiB（不含resume）。正式有限自验10.37s核对两历史/fit/源摘要及四区IDs/class40/prob/valid/prior、combined与开发指标；真实refit模型64条final公共输入禁teacher/labels载荷打开0，S/I logits对导出差0，未扫大motion/visual或读final标签。完成入口task8_complete.json绑定ready/P33/小验收及最终回归；无新独立Task8复审，不称GO。下一Task9未启动，P12/Thermal/P89/A18门槛和既有视觉精度差异不变。
