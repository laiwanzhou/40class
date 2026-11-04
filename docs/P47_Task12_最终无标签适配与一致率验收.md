# P47 Task12 最终无标签适配与一致率验收（Final Label-free Adaptation and Agreement Acceptance）

日期：2026-11-04，Asia/Shanghai。唯一指导仍为[P01计划](superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](handoffs/P02_current_handoff.md)。本报告记录实施与验收，不构成新计划。

## 范围与配方（Scope and Recipes）

用户明确推进Task12。仅使用final4公共输入和本run Task11的A8伪标签（Pseudo Targets），不读取final4真值标签、不计算最终准确率（Accuracy），不进入Task13冻结／揭示。

| 端点（Endpoint） | 固定轮数 | 更新范围（Adaptation Scope） | Fusion / Visual LR | 最低LR | 随机种子 |
|---|---:|---|---|---|---:|
| A9-40 主端点（Primary） | 40 | heads_motion_encoder | 1e-4 / 5e-5 | 5e-6 | 20260826 |
| A9-12 诊断（Diagnostic） | 12 | heads | 5e-5 / 1e-5 | 2e-6 | 20260812 |

共同为AdamW weight_decay0.02、batch64、workers0、KL T1、confidence_power0、emission_warmup0；沿用原余弦调度（Cosine Schedule）与时间增强（Temporal Augmentation），冻结视觉骨干／BN。两个端点各自从**同一A7 refit full**初始化，独立目录、优化器和恢复身份；不能把12轮继续训练到40轮。

609条最终输入保持完整，其中591条A8.target_mask=true进入loss，18条全缺失行不进入loss并回退同一refit先验。没有按置信度、开发或最终成绩筛行、选择轮次或追加训练。

## 实现与输入绑定（Implementation and Bindings）

训练producer提交`7a05347`。新增`src/experiments/target_adaptation.py`，直接复用队友`LabelFreePseudoDataset`、`make_loader`、`configure_adaptation_parameters`、`train_label_free`及Task11已验证的完整轮恢复包装。`RejectLabelsDataset`在源wrapper删除label前先拒绝任何真值字段；正式CLI另拒绝打开三个已配置监督标签payload。

Task12必须绑定已完成且独立GO的Task11凭据、正式final4 A8、原A7 refit及精确pixels／sequence／motion／normalization。注册角色（Artifact Roles）只放行具体A9-12／A9-40完整契约，不翻旧全局ready标记。模型记录保留目标／基础模型SHA、两种端点配方、完整ID／用户／类轴与591条loss ID哈希。

预测使用eval、关闭增强，输出学生原始概率（Raw Probabilities），不再执行A8后处理。无标签一致率（Label-free Agreement）复用小预测表，绑定A8、适配前A7、适配后A9和模型引用，分别记录完整609／适配591／先验18的分子、分母和disagreement；它不是真值准确率，也不参与端点选择。

## TDD与源对照（TDD and Source Parity）

首批8项测试先因producer缺失RED，再GREEN。补充精确模型契约后9项通过（17.14秒）；最终含旧source-base、后处理、角色与迁移的30项针对性回归通过（38.20秒）。测试包括完整12／40轮toy源更新逐状态一致、原日志除耗时一致、低于100%一致率仍跑满预算、错误端点resume拒绝、错ID／phase／mask拒绝、缺父输入和错target拒绝。

独立生成前审计`p47_task12_review`发现P2：缓存命中只验stage和population，尚未绑定请求模型版本。新增精确model直接父引用／stage／endpoint／target SHA／refit／raw检查，并在预测入口要求输入引用完全一致；先RED再GREEN，独立复审CLOSED。修复发生于正式A9生成之前，没有既有正式结果受影响。

真实接口仅取train12四条含IR-only样本，不接触最终标签、不登记正式模型。对照原`P87STestCachedSequenceMotionDataset`：除既有ROI质量FP16精度例外，所有消费tensor精确一致；质量先转FP16后的数值也精确一致。原FP32质量与当前消费最大差异为0.000244140625，继续明确披露，不声称全输入逐位相同。40轮scope的一轮真实CPU探针与原函数loss及全部模型更新state差0，真值字段0，用时29.60秒。探针用已有train12 A1作接口目标，不冒称正式A8替代。

## 正式生成与验收（Formal Generation and Acceptance）

正式入口为`scripts/run_no_vote_target_adaptation.py`，先40轮，再独立初始化12轮；日志`outputs/task5_audit/P47-target-formal.log`及`-error.log`，进程入口`P47-target-current-run.json`，状态`run/protocol/task12_status.json`。正式生成于20:49:33开始、20:51:55完成，约142秒；40轮训练记录合计37.57秒，12轮合计8.59秒，其余为依赖验证、初始化、保存和推理。复用已完成的视觉骨干序列，未重跑或微调大视觉教师，因此小学生头部适配较快。正式监督标签payload打开次数0。

有限验收脚本`scripts/check_no_vote_task12.py`只检查直接模型／小输出及去重元数据，核对预算、共同基础模型、完整目标／输入引用、raw预测、先验、三分母一致率和缓存复用不改写。禁止再次全扫raw／images／features／sequence或按epoch重复资产哈希。最终有界自验PASS，59.21秒；两个端点的预算／同基础模型／精确目标／输入／raw概率／先验／报告绑定及缓存复用不改写均通过，监督标签打开0。独立正式产物复审PASS／GO，无剩余P1／P2／P3，详见[P48](../reports/P48_Task12_独立复审结论.md)。最终凭据为`run/protocol/task12_complete.json`与`task12_independent_review_P48.json`；原`task12_artifacts_ready.json`生成瞬间pending／task12_complete=false保持原字节，不追改历史。

## 无增强一致率结果（Agreement Results without Augmentation）

| 阶段（Stage） | 完整609匹配 | 适配591匹配 | 先验18匹配 |
|---|---:|---:|---:|
| 适配前A7 | 438/609（71.92%） | 420/591（71.07%） | 18/18 |
| A9-40 | 607/609（99.67%） | 589/591（99.66%） | 18/18 |
| A9-12 | 581/609（95.40%） | 563/591（95.26%） | 18/18 |

40轮主端点与教师仍有2条预测不同，12轮诊断有28条；固定预算正常结束，没有为了达到100%一致率追加训练。18条共同prior单列，避免混入有效适配分母。这些数字仅表示模仿A8的程度，**不能当作最终分类准确率**。未读取真值，也未据此选择端点：40轮始终是预先指定主端点。

## 已有差异与解释边界（Existing Differences and Limits）

Task11已授权的近似配方、单根／固定划分及合法bank剪枝继续有效；Task5/10质量FP16残余不因本Task消失。Task12对final4输入做过无标签适配（Transductive Target Adaptation），不能把其结果外推为对另一批从未适配样本的同等泛化能力。开发user6／7未另做本阶段伪标签适配，不能把refit成绩称为开发验证成绩。

Task13负责冻结所有16候选并按已授权方式一次揭示；当前不执行。main历史回顾文档继续等待全部实验完成。
