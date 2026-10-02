# CUHK-X 单视觉教师流水线交接（Single-Teacher Pipeline Handoff）

更新时间：2026-10-02，Asia/Shanghai；版本v2。文件保留历史日期命名，内容已更新。下一会话先复核本次文档修订，再推进实施。

## 当前任务与授权（Task and Authorization）

用户要求根据[独立审计报告](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-02-handoff-plan-independent-audit.md)修改交接与实施计划。本轮修改了实施计划、相关规格及本文件；原始数据、队友训练源码和两个提交包均未修改。未启动训练、下载模型、commit、push、merge或PR。

用户更早批准了固定划分的队友单视觉教师复刻方向。本轮不是重新讨论方向：排除历史30教师投票，保留单VideoMAE-Large教师、MC3学生、Skeleton/IMU、MoBind、会话/重复处理及无标签适配。完整配方只在权威规格和计划中，勿从聊天片段重建。

## 权威文档与工作位置（Authoritative Artifacts）

- [完整实施计划v2](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md)，当前399行。
- [实验规格v2](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md)，当前161行。
- [独立审计与修订记录](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-02-handoff-plan-independent-audit.md)。
- [审计原版输入快照](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-02)：implementation-plan-v1.md、specification-v1.md、handoff-v1.md。原审计行号指向这里，不能套用到v2行号。
- 工作树（worktree）：`D:/work/2026.7.14_kaggle/_single_visual_processing_replication`。
- 分支（branch）：`experiment/single-visual-processing-replication`。
- 基础已提交HEAD：`e2bf1a25eb65293fd48a441fd4374661ba0db1d6`。v2是工作区中未提交修改，不能仅git show HEAD读取最新版。
- 队友源码快照（source snapshot）：`D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project`；源码清单在其上级source_manifest.json。
- [已上传teacher分支](https://github.com/laiwanzhou/40class/tree/teacher)是只读来源，不包含本次新实验实现。

本轮计划SHA256：`057129f5212057d15451975daf07676aa06b80d91a8ef88141aa6cc1d0667139`。
本轮规格SHA256：`867dacb76c8af8f2394d8c644a1c491ebe0eba014aab86acffd692739bb25999`。
如果之后再修订，更新本交接记录，不要将哈希更新误视为实验产物已存在。

## 必须保留的设计决定（Frozen Decisions）

- 不做三折折外预测（OOF）；训练train12为2039规范行，user6/user7为development2共388行，refit14为前14人2427行。
- 最终评估（final4）为user4/user17/user23/user24共609条；18条全部模态缺失保留先验回退（prior fallback），不删分母。
- user1始终在train/refit，原class25 OOF问题不再成立。固定用户名单及有效拟合类别验收在规格第2节。
- 用户不希望继续旧视觉接口匹配：重建一个冻结公开VideoMAE-Large+单Ridge分类头，重新训练兼容MC3学生；不使用腕部视觉旧路线或历史任务缓存。
- 其他模态优先复用队友原算子、模型及损失，不新建教师投票。
- 这只是固定人口的描述性消融（descriptive ablation），不能凭历史0.91或单seed声称完全独立跨用户泛化。

## 本轮审计及修订状态（Audit and Revision Status）

2026-10-02三名独立审计员对v1都给NO-GO。核心问题已在v2文档层面逐项落实，映射见计划末尾Audit Disposition；尚未实现代码，因此不能称数据隔离或接口测试已经通过。

本轮关键修订：
1. 可信输入准备与模型生成分离，白名单公共manifest、opaque ID、私有final标签；旧脚本类别字段不能补回final。
2. 明确select教师只fit train12、refit祖先只fit refit14；目标、ID、mask、类别列及所有祖先递归校验。
3. 将旧CLI包装承诺改为数据/循环移植（port），补真实pixel/sequence/P28→P31→P86依赖、公开权重和A4→A5目标契约。
4. A7-mask/shuffle-S/shuffle-I独立训练并refit；zero-S/I仅为推理敏感性。
5. A9-12/40都是原始部署输出，不再应用A8；A9−A8是部署替换差值，同时报A9−A7。主比较仍A9-12−A1。
6. 会话固定partition/user/date，A9固定final4联合无标签池；12/40各自从A7 refit启动，18缺失行排除适配损失。
7. 冻结完整候选与祖先、按ID评估、单向generating/frozen/revealed状态、稳定规范哈希、闭合fixture smoke。
8. 冻结视觉backbone时连BatchNorm运行统计保持eval，避免sequence缓存与模型失配。

这些是本次已批准方向中的最小澄清与接口修订。暂不要求新增OOF、多seed、ROI/腐败独立消融或论文级统计推断。旧审计的缺口不构成队友已经作弊的证据。

## 资源与实施状态（Resources and Implementation）

2026-10-02观测D盘约46.90GiB、C盘约52.54GiB空闲；旧交接中的“D不足、必须改盘”不再成立。仍需执行时重新计算全部缓存与控制模型峰值。共享原始缓存可以减少重复像素存储。

现有Python3.12.9、PyTorch2.7.0+cu128、torchvision0.22.0+cu128、RTX5060 Laptop约8GiB；没有运行真实新流水线batch，不能视为运行兼容性通过。公开VideoMAE指定revision和MC3权重需要显式取得或指定hash验证的本地副本；旧10–20 GPU小时是未验证预算，匹配对照会增加工作量。

新流水线Tasks1–14均尚未实现，模型配置/函数签名/测试是待建接口。没有新准确率、最终候选或已冻结实验。修改文档不等于通过独立复审或真实训练验收。

## 下一步（Next Steps）

1. 读取本文件、规格v2、计划v2及报告修订映射，检查当前工作区差异，勿将v1历史行号当v2。
2. 如用户继续要求独立复审，仅审修改项；无需重新审计通用操作安全、权限或越界，重点仍是方法、数据泄露和实际源码契约。
3. 复审解决文档阻塞后，按Task1–14推进实现：先协议/权重/输入与祖先，再几何/教师/学生/运动/对照，最后A8/A9和冻结。
4. 正式长训练前通过预检、针对性测试、真实train12一条/一批接口验收与独立fixture全链smoke；此后生成全部16候选，再冻结，最后独立评估器取得final标签。
5. 按用户当时的明确授权决定是否commit/push；不要擅自提交主仓库或旧worktree的未提交实验。原数据/提交包/队友源码保持只读。

## 建议技能（Suggested Skills）

- superpowers:receiving-code-review：继续核对审计反馈，区分文档缺口与已发生问题。
- superpowers:writing-plans：必要的文档修订；当前方案已确定，不重新引入泛化设计讨论。
- superpowers:executing-plans：本机顺序实施，Task依赖密集，适合直接执行。
- superpowers:dispatching-parallel-agents：仅在用户明确要求独立复审时使用不同上下文审计员。
- academic-research-suite：实验验证与有限人口结果解释，不扩大到长期学术流程。
- superpowers:systematic-debugging：实际接口、训练或测试失败时先定位根因。
- superpowers:verification-before-completion：验证再声明完成；明确静态检查与真实训练的区别。
