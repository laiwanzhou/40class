# Task2/3与Task4独立审计（Independent Audit）

审计对象：`9005c2e`及本轮工作区修复；分支`experiment/teammate-single-teacher-task1`。三名审计者均未编写实现，复用了已保留历史的独立子智能体：audit_20261002_engineering、audit_20261002_methodology、v2_review_methodology。未读取final4私有标签。本文记录实施审计，不等同最终模型成绩。

## Task2/3门槛与修复（Gates and Fixes）

| 发现（Finding） | 处理与证据（Disposition and Evidence） |
|---|---|
| 只有一条partial ROI，不能供完整拟合 | 改用refit14/final4各一次全量标签无关预处理；缓存必须complete且精确匹配2427/609 IDs。 |
| Numeric IR未对齐timestamp Skeleton，21条可用骨骼被丢弃 | 按唯一counter匹配，真实22条逐一复核：21条全部完整对齐并保存acquisition_ids；另1条保持缺失。 |
| 4条旧记录存在numeric/timestamp重复别名 | 只允许唯一时间戳与数字别名且文件SHA完全相同的合并；多时间戳或冲突内容拒绝。真实17/37/50/67组别名均逐字节相等。 |
| 无IR时新的时间字段缺失 | P28/P29统一输出等长空acquisition_ids，不伪造绝对时间。 |

工程、数据边界、方法三方均同意补全原始ROI并进入Task4；正式教师拟合另受完整feature和有效40类支持门槛约束。帧号、别名和空时间字段均有RED→GREEN回归；相关Task2/3测试26通过。

## Task4复刻与来源边界（Parity and Provenance）

- 冻结公开VideoMAE-Large；refit14原始六clip features按opaque ID供train12/development2引用，不把refit监督模型送入开发选择。
- select的StandardScaler、Ridge和class weights只拟合valid train12；dev按全部388行评分。refit重建模型，仅拟合refit14，不继续训练selected模型。
- 缺IR使用相应fit人口全部规范标签得到的40类prior，valid=false；final预测仍保留全部609行，不读取其标签。
- 方法审计独立对照P46/P85：六特征族、三种power样本权重、四个alpha的模型参数逐元素/参数相等；六clip的96张合成裁剪帧含缺ROI回退也完全相等。主审另外保存source-math-parity.log复核数学。
- A1固定head temperature=1，scores→softmax；旧OOF温度未复用。这是规格明确披露的固定划分变体，区别于后续KD温度和融合校准。

## Task4工程修复（Engineering Fixes）

1. 上游encode的torch.inference_mode装饰器导致来源误判：校验inspect.unwrap后的原始定义，仍返回完整装饰函数。标准库decorator回归RED→GREEN；保持inference_mode。
2. Registry原只核验已知文件内容，可能忽略新增原始帧：提取/续跑任何快速返回前核验P29原始文件集合；head/predict直接读features也核验，两个实际漏报回归RED→GREEN。
3. 完整ROI允许max_trials诊断子集，但验证整个父缓存的输入集合；partial features仍不可进入正式拟合。
4. prior采用显式同phase prior_artifact.json，避免扫描历史记录时误选/歧义；receipt回归RED→GREEN。

数据边界、方法与工程复审均GO。工程审计者接续历史后独立重跑teacher/source共19测试通过，另以新增IR帧合成复现确认旧续跑漏洞已拒绝；没有剩余Important代码阻塞。最新工作区相关72测试通过，真实GPU六clip复测成功，峰值1.449GiB。正式CLI仍须自行完整验证父产物。

## 尚待执行与非阻塞项（Pending Work and Non-blocking Finding）

- 两分区全量ROI已生成，summary.complete=true，逐项registry与schema核验进行中。最新真实GPU编码复测已通过；refit14正式特征抽取已启动（入口先核验完整父产物）。正式72候选选择与refit尚未开始，不能报告模型准确率。
- P2：Depth–Skeleton semantic字段当前直接复制IR侧，缺Depth未独立mask。P29/Task4及原P31/P86不消费此字段，不阻塞当前阶段；Task6前必须核对其消费者并在需要时做独立遮罩处理，避免随意修改当前producer导致整批缓存身份失效。
- 已有7个全仓失败均依赖清理掉的旧缓存；不恢复废弃路线来掩盖基线失败。

详细日志位于本工作树`outputs/task4_audit/`（Git忽略）。最终执行状态另由Task4验收报告与交接文档记录，不从本静态报告推断全量训练完成。
