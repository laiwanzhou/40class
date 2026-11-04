# P50 Task13 独立复审与冻结代次验收（Independent Review and Frozen-generation Acceptance）

日期：2026-11-04，Asia/Shanghai。审计者`p49_task13_review`使用独立新上下文只读审查。唯一指导为[P01](../docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，交付[P49](../docs/P49_Task13_候选冻结与固定划分最终评估.md)。

## 代码复审与修复（Code Review and Fixes）

审计者发现P2：初版`metadata_only`跳过小型descriptor payload SHA，冻结快照也未完整绑定已消费小文件。纯模拟数据（Fixture）复现修改`source_transform.operators`后仍通过，普通验证正确拒绝。

已修复两时点：小descriptor／capability／recording／model JSON/CSV按record.files expected SHA验证，缓存命中仍重查；冻结代次保存全部实际消费小文件的path/SHA，揭示前再次检查。原始缓存／权重／监督标签payload仍排除，实际recipe作为明确例外验证。新增回归先RED再GREEN，独立复审CLOSED。

正式揭示另要求独立审计批准同一generation SHA及protocol。错误SHA、缺审计、数据漂移均在读取标签前拒绝；revealed必须先持久化，失败也不可重置评估。

性能修订：只缓存小文件路径列表，不缓存验证结论；递归不可变协议缓存原canonical SHA，新配置实例重新计算。独立相关5测试通过（4.08秒），哈希语义不变。主线程72项回归通过（46.39秒），原protocol／目录迁移／角色测试继续通过。

## 精确冻结代次审计（Exact-generation Review）

审计对象SHA256：`0a120e444d6142ed72213f1d02d5af4587dbb57a361605ddf890bf732cecc4ee`。protocol为`2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。

独立审计在builtins.open／io.open双探针下，仅允许冻结绑定白名单，禁止写入、模型／序列大载荷、监督标签和原带标签`metadata/manifest.csv`。使用`load_frozen_predictions`通过全部SHA：176份Artifact Record、308份已消费小文件绑定，实际读取310个唯一小文件。

复核16候选固定顺序、609唯一ID、四名封存用户、0–39类列、公开final4清单完整行序；四模态可用性OR与A8.valid均为591／18；每个候选的18条全缺失概率逐元素等于同一40类训练先验，先验有限、非负且和为1。run_state=frozen，generation/state SHA一致，truth_reads=0，accuracy_computed=false。

精确冻结代次结论GO，无剩余P1／P2／P3。审计者首个附加选择器同时匹配公开recording与标准清单而断言退出，明确选择标准`protocol/final4.csv`后完成核对；没有读取任何真实标签，不视为模型或冻结失败。

## 审计凭据登记（Review Registration）

冻结文件中的逻辑旧目录与本机新物理目录是同一目录联接（Directory Junction）。原内置approve辅助函数在两种路径间作字面边界比较，会误拒绝。为保持已冻结generation与评估器源码不变，正式登记使用独立`review_no_vote_frozen.py`，将两个目录都解析成物理路径后，仍检查真实run/protocol边界、frozen状态、state引用、generation SHA和review GO。

真实历史别名→新目录的fixture先RED再GREEN（3.57秒），完整模拟评估成功且冻结字节不变。此登记只更新状态中的审计引用，不修改预测、协议或冻结代次；不把错误路径视为批准。

最终一次真实标签评估完成后，产物一致性检查可仅使用聚合混淆矩阵（Confusion Matrix）、公开冻结预测和计数关系，不重复打开封存真值，不按结果训练。


## 登记与一次评估后验收（Registration and Post-evaluation Acceptance）

独立审计确认行政登记CLI GO，真实目录联接回归独立1项通过（3.54秒）。登记只更新review引用，原冻结SHA和评估器字节不变；正式评估使用已批准的同一代次。

一次评估成功后，审计者仅白名单读取metrics与冻结绑定小文件，唯一311个小文件；正式标签／原manifest／模型／raw读取0，evaluate重跑0，训练0。16候选混淆矩阵的总数609、对角正确数、Accuracy、类别支持／实际联合40类和fixed40 macro-F1全部一致；预测列边际与冻结argmax分布一致。用户／模态／A8-mask分母与正确数加总一致。

四项contrast在完整／适配／prior分层上的人数、正确数、delta=rescue−harm、百分点及分层加总正确，disagreement与公开冻结预测逐行比对一致。A9-40与A8均467/609，有2处预测不同但rescue=harm=0；两处均为双方错误而错类不同。A9-40−A1为+22条／+3.612479个百分点，32救回／10新增错误；相对A7为+69条／+11.330049个百分点，92／23。

末次聚合验收PASS，无未解决问题。state和metrics.state均为revealed；两者引用的generation SHA仍为0a120e444d6142ed72213f1d02d5af4587dbb57a361605ddf890bf732cecc4ee。未重复打开真值，不追加训练／选择。当前批准范围至Task13，Task14未启动。
