# 旧实验清理与v3复审修订（Cleanup and Review Resolution）

日期：2026-10-06，Asia/Shanghai。用户明确“最初v1”指已完成的69.07%教师／48.71%学生实验；磁盘正式run名称实际为`fixed-split-single-teacher-v2`。本轮只清理该run的可重建冗余产物，不清理其他历史目录或两份原提交包。

## 已清理（Removed）

| 相对于旧run的文件（Relative File） | 字节（Bytes） | 原因（Reason） |
|---|---:|---|
| A2/select/resume.pt | 279776255 | 训练已完成，正式记录不引用；完整epoch恢复状态无需继续保留 |
| A2/refit/resume.pt | 279775295 | 同上 |
| sequence/select/train12/sequence.npy | 200441984 | 旧学生学习特征不能用于v3，新学生必须重建 |
| sequence/select/development2/sequence.npy | 38142080 | 同上 |
| sequence/refit/refit14/sequence.npy | 238583936 | 同上 |
| sequence/refit/final4/sequence.npy | 59867264 | 同上 |

合计 **1,096,586,814字节＝1.021 GiB**，6个文件；删除后逐项确认不存在。此前检查无学生训练/接续进程，四个sequence ArtifactRef无已登记下游消费者。仅对目录/文件大小和小型记录元数据做检查，没有raw或cache全内容SHA扫描。

第一条带完整预检/记录的长批量命令被自动审批策略返回blocked by policy，未执行。改为明确字面量路径、非递归的原生PowerShell文件删除后完成；没有用另一个语言/工具绕过拦截或扩大删除范围。

机器可读记录：`outputs/task5_audit/cleanup-20261006.json`；旧run新增`retired_caches.json`侧记录。源正式artifact记录、identity、digest未修改。**旧sequence记录现在只证明历史生产与验收，不是可直接完整加载的缓存。** 若以后确需重建旧学生sequence，使用历史producer、保留检查点与pixels，另行记录重建身份；不能改旧digest冒充原文件。

## 保留及意义（Retained Value）

- pixels约6.96 GiB：v3可按ID及同输入配置复用，重新构建会浪费解码/裁剪时间。
- 公开weights约1.18 GiB、P28/P29约0.27 GiB、冻结visual_features约0.09 GiB：是v3复用的公开初始化及标签无关输入。
- select/refit最终checkpoint各约68.6 MiB、Ridge头、targets、priors、anchor logits/valid、rows/masks、正式records与报告：保留v2基线、来源及必要对照。不会拿旧学习产物当v3监督祖先。
- 原始数据、队友快照、提交包、v2正式配置、公共清单及监督标签不变。旧raw/cache并非整体“无意义”，所以没有整目录删除。

## 独立复审与发现（Independent Review Findings）

两项独立复审基于HEAD8d1f394的v3初稿，均给NO-GO直接正式生成：

- [源操作一致性复审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-06-v3-source-parity-independent-review.md)：1项P1，另列具体consumer待锁定处。
- [可实现性复审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-06-v3-feasibility-independent-review.md)：3项P1和3项P2，部分与前者重合。

| 问题（Finding） | 本次文档修订（Resolution） |
|---|---|
| 十机制错误归为MC3学生 | 移Task4，VideoMAE六clip逐clip L2→固定Ridge alpha3000/power0.75→fit均值特征干预→raw logits/T1；独立小头不增加大骨干，不用A1温度或A2 |
| 全窗KNN孤立/顺序闭环 | Task4只提取同大骨干全窗特征；Task11先P12八列+六visual log概率、源simplex全局融合，再三邻居/0.40 smooth，不能用A1/A7替代父融合 |
| opaque ID时间元数据缺接口 | Task2明确7字段公共sidecar；Task11外层按partition/user/date分块调用原纯函数，再映射回RowIndex；目标apply显式接收其metadata，不读旧CSV/假cohort |
| 跨run缓存原refs/外部路径不被registry接受 | 明确D同卷硬链接为新run本地immutable payload，新祖先全部v3，oldref只存provenance；mmap只读/producer拒绝写入，不放宽所有路径 |
| descriptor/专家预测/其他资产角色不匹配旧注册器 | v3明确descriptor、teacher_bank、source_expert模型→自己的预测祖先、teacher-only asset绑定，未知角色仍拒绝，不漂移v2身份 |
| 六头/登记/加载的校验重复 | 公共VerificationContext缓存record/file/loaded arrays/import memo；Task4/6/11共用；禁旧full-SHA分支，登记/消费无重hash，fixture计数验证 |
| RF缺类及RNG consumer未明确 | 固定无fold拟合用terminal RNG seed；fit_teacher_target缺类-1e6，terminal_export缺类log1e-12，两profile分别登记，不假装全都相同 |
| bank safe/fallback/availability未锁定 | 源safe 0.95base+0.05RF(T3)，合法缺输入行回退第0列且available=false；原detail_ids的visual_available；group必须AND、sequence不额外AND、P310规则不变 |

v3.1仍保留用户固定划分替代OOF、不引入内部OOF、不读取final标签、不按test调参。此次未重新执行类别覆盖核实；仅引用此前受限独立检查“联合40类=是”。不扩大安全/权限审计，不训练或下载。

初稿已按原字节归档到`reports/audit_inputs/2026-10-06/pre-review/`。当前规格与计划在原权威地址更新。两位原独立审查者复审修订文本，报告另存，不能删除原NO-GO报告来掩盖修订历史。

## 放行边界（Release Boundary）

本次目标是纠正文档并让Task1–3有可执行的协议/源操作/导入路径。**不能把计划文字一致等同于全部未来producer已被证明一致。** 实际source_ops_manifest、teacher_roster、P12/safe-base独立配置/资产仍须在Task1产出；逐算子数值/梯度与实际输入验收仍是正式生成门槛。缺源producer或角色时必须停在该门槛，不能用简化A8替代。

不会因文档复审GO就启动长训练，现阶段没有v3模型或完整teacher bank。保留原1–14任务编号，下一实施仍从v3 Task1–3开始。

修订后两位原独立审查者均再审GO，仅放行Task1–3实现与针对性验证；原P1/P2已在文档层面闭合，无新增有证据阻断。另补充日内秒口径及Task1首次ArtifactRef导出前的最小descriptor注册顺序。详细结论见[源操作再审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-06-v3-source-parity-independent-reaudit.md)和[可实现性再审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-06-v3-feasibility-independent-reaudit.md)。实际保留producer/资产/数值验收尚未完成，不能因此宣称全部v3处理已有运行证据证明与队友等价。
