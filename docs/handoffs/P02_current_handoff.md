# P02 当前实验交接（Current Experiment Handoff）

更新：2026-11-04，Asia/Shanghai。本文件是本分支唯一交接入口（Canonical Handoff），后续原位更新；唯一指导为[P01计划](../superpowers/plans/P01_teammate_single_teacher_fixed_split.md)。归档和审计输入快照仅作历史，不作为另一份plan或handoff。

## 当前状态（Current Status）

**Task13已完成完整冻结、一次授权最终评估与独立聚合验收；Task14尚未启动。真实run已revealed，禁止继续生成／训练、改候选或重复评估。**

工作树`D:/work/26.7.14_kaggle/_single_visual_processing_replication`，分支`experiment/teammate-single-teacher-task1`，origin为`https://github.com/laiwanzhou/40class.git`。用户只改父目录名，真实项目目录为26.7.14_kaggle；历史2026.7.14_kaggle保留同文件目录联接（Directory Junction）。显式七锚点迁移配置为configs/experiments/project_root_relocation.json，旧协议、记录和产物SHA保持。原ROI提取生产器raw_input_hashes的全套旧断点恢复未认证，不宣称全旧路线可重启。

## 权威入口（Authoritative Records）

Run为outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops。

- run/run_state.json：state=revealed；内含冻结代次和审计登记绑定。真实生成已关闭。
- run/protocol/task13_complete.json：本Task完整结果、证据及审计入口。
- run/protocol/task13_frozen_generation.json：精确16候选和消费文件绑定，SHA256为0a120e444d6142ed72213f1d02d5af4587dbb57a361605ddf890bf732cecc4ee。
- run/protocol/task13_freeze_review.json：精确代次独立GO；run/protocol/task13_independent_review_P50.json记录完整复审。
- run/evaluation/task13_final4_once/metrics.json：唯一一次真实评估结果，不重复计算或打开原真值。
- 原Task11／12 complete及其独立审计记录继续保留；生成瞬间pending／false和当时task13_started=false属于历史，不覆盖当前完成事实。

基础协议身份仍2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a；Task11修订task11_reconstruction_v2，配方SHA39f5c5a5a67a6d128ae089dfc619ccfdb537aea92b595aaf9d2d4bd7c67ba89c。不可修改旧身份或描述符绕过校验。

## 最终结果（Final Results）

固定final4为user4/17/23/24共609条，591条有效输入、18条prior。实际真值联合覆盖40类，不假定每名用户分别覆盖全部40类。

| 阶段（Stage） | 正确/609 | 准确率（Accuracy） | 40类宏平均F1（Macro-F1） |
|---|---:|---:|---:|
| A1 单大视觉教师 | 445/609 | 73.07% | 68.16% |
| A2 基础视觉学生 | 394/609 | 64.70% | 58.86% |
| A7 基础融合 | 398/609 | 65.35% | 59.21% |
| A8 完整教师表与后处理 | 467/609 | 76.68% | 71.36% |
| A9-12 诊断适配 | 460/609 | 75.53% | 70.17% |
| A9-40 主端点 | 467/609 | 76.68% | 71.32% |

主比较A9-40−A1：+22条／+3.61个百分点，rescue32／harm10。相对A7：+69条／+11.33个百分点，rescue92／harm23。A9-40对A8准确率无净损失，但2条预测不同且两者在这两条都错，不能声称逐行或全部指标完全相同。18prior全部错误，固定591分母的主端点为467/591=79.02%；主指标仍完整609。

全部16候选、四项配对比较、用户／模态及A8.mask分层和40×40混淆矩阵均保留在metrics.json；交付[P49](../P49_Task13_候选冻结与固定划分最终评估.md)，独立复审[P50](../../reports/P50_Task13_独立复审与冻结代次验收.md)。

## 标签、冻结与一次性边界（Truth, Freeze and One-time Boundary）

用户本轮明确使用固定划分真实标签，覆盖旧“暂不正式揭示”。旧temp final_labels.csv不存在；原prepare记录docs/task23_input_pose_roi.md指定本分支metadata/manifest.csv。冻结前只stat定位；精确代次独立GO后先写revealed，再以canonical_manifest模式按原NAMESPACE SHA映射opaque IDs，仅提取冻结609行，核对用户归属。没有使用官方Kaggle隐藏测试标签，也没有用开发user6/7替代final4。

真值仅在唯一一次真实评估步骤使用。之后独立验收只读混淆矩阵／聚合计数／冻结预测，真实标签读取0、evaluate重跑0、训练0。不能把已揭示final结果用来改预算、阈值、候选或追加训练。

Task12使用过final4公共输入和A8伪标签，属于转导式目标适配（Transductive Target Adaptation），不是对另一批完全未适配输入的纯归纳成绩。单根／固定划分、缺历史记录的近似配方、20列合法bank剪枝、既有quality FP16残余继续披露，不能宣称单因素严格因果消融。

## 实现与验证（Implementation and Verification）

Task13代码736b14e；迁移安全审计登记CLI为2ec0591。固定16候选与四项contrast；revealed先持久化、失败不重置。ArtifactRegistry和phase_lock拒绝冻结／揭示后的生成训练。metadata_only模式不读raw／权重／模型／序列大payload，已消费小descriptor等仍验SHA且揭示前复查。

72项针对性回归46.39秒通过，另junction审计登记1项3.57秒通过并独立复审。真实预检12.70秒、冻结12.19秒，176去重metadata／308消费小文件，禁止标签打开0。初版开销profile和失败日志保留；优化不重写旧SHA。精确代次审计GO，最终聚合验收PASS，无剩余阻断。

原内置approve辅助函数会因旧逻辑路径与新物理路径比较而误拒绝；正式登记使用scripts/review_no_vote_frozen.py，将两个根都resolve后执行同等或更严格边界，保持被冻结评估器源码和generation字节不变。不要为修复历史辅助接口改写已冻结代次。

## 下一步（Next Step）

只剩Task14编排与全链交付收尾，当前未启动；真实run已revealed，后续编排测试应使用隔离fixture，不能在该run重训或再次揭示。main历史回顾文档仍等整个复刻实验完成再更新。

原始数据、队友快照和两份提交包只读。两份旧untracked报告reports/2026-10-02-history-cache-reuse-inventory.md与reports/2026-10-02-old-visual-worktree-retirement.md不顺手提交／删除。正式Python为D:/Anaconda/envs/PyTorch2.7/python.exe -B；没有自动化、新服务或存活训练worker。
