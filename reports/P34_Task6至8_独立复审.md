# P34 Task6至8独立复审（Independent Review）

日期：2026-10-29，Asia/Shanghai。用户明确要求在Task8后独立复审骨骼／IMU处理、RF目标、缺模态损失和跨模态对齐。唯一指导为[P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)；本文只记录复审结论，不新建计划或交接。

审查范围固定为 `git diff 3c3dda6...bfcb3a9`：Task6提交f1f72c4、Task7提交2a64270、Task8提交bfcb3a9；目标完整SHA为 `bfcb3a9f0261ebd7fe70dde23bcbde73b9dbcea1`。本轮没有修改训练代码、重训、修正预测或推进Task9。P31/P32/P33和原完成标记保留自验／实施时事实，不回写为独立通过。

## 总体结论（Overall Verdict）

**Task6／Task7通过（PASS）；Task8有条件通过（CONDITIONAL），尚不构成无保留的源方法一致性通过。** 三名独立审查者分别完成工程／规范（Engineering / Standards）、规格／方法（Specification / Methodology）及正式产物（Formal Artifacts）审查，再由主智能体汇总。工程与规格维度分别呈现，不以工程测试通过覆盖方法差异。

独立审查者为 `/root/p34_engineering_review`、`/root/p34_methodology_review`、`/root/p34_artifact_review`。最终发现按唯一问题去重：一项条件性P2方法差异、一项P3缓存复用限制及低优先级代码异味；无P1。P2影响当前A5训练定义，不能仅补文字就追认为严格源等价。没有证据表明其解释了当前准确率差距，不从单次复审推断修正后的分数。

## 工程与规范（Engineering / Standards）

独立结论：**PASS（限定工程范围），未发现影响当前已产正式结果的P1/P2工程缺陷。** 固定24轮末态、refit重新初始化、阶段教师绑定、ID／用户映射、mask／dtype、prior和combined诊断语义符合当前实现约束。未发现额外项目硬编码规范违规；用户要求的中文文档、Pxx编号及唯一plan／handoff继续保持。

**P3／P34-E1：完整缓存复用不重查原始IMU改变。** [motion_cache.py:238](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/motion_cache.py:238)在完整artifact存在时直接返回，跳过后续逐trial输入stamp检查；[imu_rf_teacher.py:98](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/imu_rf_teacher.py:98)也只复用已登记完整统计。若原路径CSV被修改而manifest未变，再次调用可能使用陈旧缓存。**未发现当前原始输入发生变化的证据，不认定现有模型受到影响。** 当前复用以原始输入保持不变为条件；未来若需支持可变输入，再补轻量inventory／版本化失效规则，不因此增加全目录SHA或重扫大缓存。

低优先级代码异味（Code Smells，判断项而非规范硬错误）：三个新模块重复 `_ref`，两个模块重复NPZ写入／摘要逻辑，后续可提取共享工具。本次不为风格重构已冻结producer，避免无必要地产生新模型身份。

独立加强的CPU恢复验证：连续24轮与第7轮中断后重建模型／DataLoader、扰乱RNG再恢复，使用dropout、shuffle及Dataset内部Python／NumPy／Torch随机量；最终模型、优化器、scaler、全部CPU RNG和排除耗时的逐轮history逐值／bitwise一致。证据为[恢复验证日志](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/.superpowers/sdd/2026-09-23-teammate-single-teacher-fixed-split/P34-engineering-resume/result.log)。CUDA恢复仅静态核对，没有重复GPU训练或源批次验收。

## 规格与方法（Specification / Methodology）

独立结论：**Task6／Task7 PASS，Task8 CONDITIONAL。** 未发现运动字段、旋转／重采样、RF统计／种子／双sentinel、A5源loss各项或跨模态对齐实现缺失；12个直接消费源码与最终包字节一致。combined只作显式诊断，分支valid／prior属于已披露的预测适配。

**P2／P34-M1：IR-only样本被排除训练，尚不能认定是已获明确授权的源操作。** [规格第30行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md:30)与[第81行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md:81)的“完全无输入／完全无有效输入”可能指全模态缺失，也可能指该阶段无motion；不能把这项歧义直接解释为允许舍弃仍有视觉监督的样本。

[当前过滤代码](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/mobind_pretrain.py:277)用 `sk | imu`决定训练行，[类别权重人口](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/mobind_pretrain.py:292)也随 `y[valid]`改变。实际select／refit分别排除83／86行，其中各有1条A1有效、但S/I均无输入的IR-only行。A1有效人口1957／2342，A5训练人口1956／2341。这一点由方法和产物审查者分别核实，不需要final标签。

源[CE计算](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission%281%29/code/training/project/aligned_multimodal/train_p86_mobind_pretrain.py:388)、[视觉KD／特征损失](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission%281%29/code/training/project/aligned_multimodal/train_p86_mobind_pretrain.py:449)不会因为无motion自动删除这些行；源all-label路径按完整训练索引取行。排除会改变训练样本、CE／KD贡献、class weights、shuffle及batch组成，**影响当前两个A5 checkpoint及使用它们的后代，不是仅报告上的差异**。P33／P01完成段已披露执行事实，但事后披露不等于事前明确批准额外方法差异。

待决处理（Resolution Required）：

- **建议方案：恢复仍有有效视觉监督的IR-only行参与源CE／KD及对应类别权重计算**，明确真正全模态缺失行的排除边界；按修正人口重新随机训练两个24轮A5，使用新producer／产物版本，不覆盖旧A5、旧身份或原完成证据。之后做针对性回归与独立复审。Task1–7可继续复用，不需要因本问题重训视觉或RF。
- 另一种选择是用户明确接受“无motion即排除”作为额外人口适配（Population Adaptation），将其写入规格的允许差异后保留现有A5；此时不得继续宣称除单大教师／固定划分外其余操作完全一致。本轮未代替用户作此接受。

在裁定或修复闭合前，不将当前A5作为已获无保留源方法验收的输入自动推进后续训练。该条件不否定已经实际完成的24轮、现有输出的数值正确性或此前Task1–7的有效性。

## 正式产物独立核验（Independent Artifact Verification）

独立结论：**按现有实施设定产物一致，严格方法裁定仍为CONDITIONAL。**

- Task6：refit14／final4的2427／609条source、12字段数组头、少量mask切片、identity transform、完整ID／窗口索引及pixels小时间表一致，没有跨样本拟合统计。
- Task7：实际加载两RF，均400树／250维；有效拟合1914→3828、2283→4566。8份目标有效数1914／369／2283／580及分区映射正确，开发独立复算123／388。
- Task8：CPU实际加载两checkpoint，双40类头、width96；各24轮、每轮1956／2341样本，首末LR4e-4／2e-5、同phase祖先正确。12份小预测完整ID／40列、逐行prior和combined可用分支平均通过，开发S／I／combined复算139／111／154，分母均388。
- 15项完成证据摘要匹配；P34-M1的两阶段各1条IR-only排除在实际model.json／history和公共mask／A1.valid中得到确认。未发现其他P1/P2/P3产物问题；全局false及局部scope保持。

未读取final私有标签、类别分布或final准确率；未重训、GPU复验、全扫大数组或递归祖先载荷。mmap headers／抽样不能证明全部未抽payload完整；不将代码正确性或当前目标输入表现外推为新用户必然同分。

## 后续状态（Next State）

本轮只完成审计及结论登记，不修改正式生产代码／模型。原 `task6_complete.json`、`task7_complete.json`、`task8_complete.json`保留当时自验完成的历史事实；新增 `protocol/task6_8_independent_review_P34.json`绑定本报告和原完成标记，记录Task6／7 PASS、Task8 CONDITIONAL及未解决P34-M1。它不是新训练产物或新全局放行。

P01／P02原位更新审查状态，Task9未启动；后续先处理P34-M1，不能把“Task8实施完成”误读成“全部源方法问题已闭合”。工程维度共1项P3（最严重为条件性陈旧缓存复用），方法维度共1项条件性P2（最严重为IR-only训练人口），产物维度复核同一P2而非新增第二问题。
