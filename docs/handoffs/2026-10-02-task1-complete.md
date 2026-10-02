# CUHK-X Task1完成交接（Task1 Handoff）

更新时间：2026-10-02，Asia/Shanghai；本轮只完成Task1。用户授权新建实验分支并推送，未授权本轮继续Task2或训练。

## 位置与版本（Locations and Versions）

- 工作树：D:/work/2026.7.14_kaggle/_single_visual_processing_replication。
- 实验分支：experiment/teammate-single-teacher-task1。
- 基础提交：e2bf1a25eb65293fd48a441fd4374661ba0db1d6；新提交及远端状态以最终push验证为准。
- 最新计划：docs/superpowers/plans/2026-09-23-teammate-single-teacher-fixed-split.md，Task1已勾选，其余尚未实施。
- 最新规格：docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md。
- 实施/测试证据：reports/2026-10-02-task1-implementation.md及reports/2026-10-02-task1-verification.json。
- 使用说明：docs/task1_fixed_split_protocol.md。
- 队友源码：D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project；1167文件校验，未修改源码或提交包。

## 已完成（Completed）

Task1已实现四个聚焦模块、公开权重管理CLI、完整YAML配方和共同类型。35个新测试全部通过；全仓476通过、7个旧数据/实验产物缺失失败，集合与基线相同，无新增失败。

公开VideoMAE-Large、MC3-18、YOLO11n-pose已取得并在CPU初始化验证；不读取比赛样本、不推理视频、不训练。权重存放于Git忽略的outputs目录，完整SHA256与模型架构记录在本地protocol/weights_manifest.json。

本机Transformers4.49直接验证48个原生q_bias/v_bias张量，误差0；新版query/key/value恢复分支有回归测试，不宣称本机运行Transformers5。

独立实现审计发现顶层来源正确但同名传递依赖可来自旧工作树，已回归复现并修复：现在验证cached dependencies及导出函数/类来源。原三个文档复审的P2也已明确配置/契约：公开冻结特征可共享、同phase冻结A2 anchor、显式motion_aux/reliability masks。

## 继续时注意（Continuation）

1. 不重做Task1。先运行acquire_no_vote_weights.py --verify-only核验当前资产和协议；新机器需取得teacher分支快照或传--source-root。
2. 用户下一步若授权，继续Task2的可信prepare、公共manifest与标签分离。当前这些路径仅配置，文件尚未生成。
3. 不用旧缓存恢复7个基线失败；它们不属于新流水线依赖。完整失败名称在实施报告。
4. 仍沿用train12/dev2(user6/7)/refit14/final4(user4/17/23/24)固定划分；不能把Task1权重验证等同于最终标签隔离或准确率通过。
5. 对代码/文档继续变更后重验身份；最终标签读取仍只在后续Task13冻结后。

## 建议技能（Suggested Skills）

- superpowers:executing-plans：按授权任务顺序实施。
- superpowers:test-driven-development：先失败测试，再实现。
- superpowers:systematic-debugging：定位实际接口/测试失败。
- superpowers:verification-before-completion：验证后报告，区分已有与新增失败。
- academic-research-suite：解释固定人口描述性实验结果。

计划SHA256：78d5f935752480157c7dddab089fa780234d7a8d568340054455b2d9484e5ef3
规格SHA256：6b53b17f5cf1bb6824f4eaf5c5ff28cf42aab0ebf897d0553e8daee1b22813a8
