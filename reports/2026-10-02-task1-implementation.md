# Task1 实施与验证（Implementation and Verification）

实验分支：`experiment/teammate-single-teacher-task1`。完成范围为协议、公共类型、1167源码校验、公开权重取得及模型CPU初始化；没有比赛样本读取、数据准备或训练。

## 验证结果（Verification Results）

- 测试先行（TDD）：初始30测试因模块缺失失败，再实现至通过；完整SHA校验新增2个失败测试后通过；独立审计发现依赖来源缺口，新增回归测试复现后修复。
- Task1针对性测试：35 passed / 0 failed。
- 全仓库（repository suite）：476 passed / 7 failed；原始基线441 passed /同样7 failed，没有新增失败。
- 公开权重取得及CPU初始化完成，`--verify-only`重新验证通过。
- 队友源码1167个文件大小/哈希符合原清单；加载器检查顶层、传递依赖及导出函数来源，不执行旧main。
- VideoMAE-Large：304,270,736参数、1024隐藏维、400类、16帧；本机Transformers4.49原生q_bias/v_bias共48个张量验证，误差0。
- MC3-18：11,695,440参数、400类公开初始化。
- YOLO11n-pose：2,874,462参数、17×3姿态点。

详细文件SHA256、协议身份和模型结果见[验证JSON](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-02-task1-verification.json)。权重本体位于Git忽略的outputs目录，不上传。

## 已有失败（Existing Failures）

下列失败在实现前就已存在，都是旧数据/实验产物缺失；没有重建已废弃缓存：

1. `test_motion_attribute_dataset.py::test_real_population_is_canonical_before_cache_generation`
2. `test_motionbert_skeleton_dataset.py::test_real_population_and_projection_ownership_are_frozen`
3. `test_report_motionbert_lite_skeleton_expert.py::test_report_recomputes_metrics_and_stops_after_failed_b1`
4. `test_run_motion_attribute_expert.py::test_motion_attribute_smoke_updates_all_groups_without_validation_gradient`
5. `test_run_motionbert_lite_skeleton_expert.py::test_smoke_loads_pretraining_and_updates_head_only`
6. `test_x3d_s_online_roi_parity.py::test_real_online_pipeline_matches_exported_training_rois`
7. `test_x3d_s_real_manifest_contract.py::test_real_manifest_is_split_safe_ordered_and_adaptive`

## 实施决定（Rulings）

- 用户明确只完成Task1，因此不遵循技能连续实施全14任务的通用建议。若范围判断错误会提前启动未经本轮要求的长训练，因此本轮停止于初始化验证。
- 权重网络/初始化逻辑独立为no_vote_weights.py，与纯协议及类型分开；成本是增加一个聚焦模块，避免协议加载无意触发网络或模型初始化。
- 基线失败只记录，不重建旧缓存；后果是全仓仍有7个已知失败，但Task1针对性测试全部通过且失败集合不扩大。
- 本次任务冻结完整配方所需的三处复审澄清：公开冻结特征可共享、固定同phase A2 anchor、显式motion辅助/reliability masks。这里只更新契约与配置，不提前实现Task4/5/10。
- 独立实现审计发现传递依赖来源遗漏；按一次修复流程补充回归测试、修复并重新跑针对性/全仓/真实初始化校验，没有循环重复独立审计。

## 后续（Next）

Task2尚未开始。使用说明和可重跑命令见[Task1说明](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/task1_fixed_split_protocol.md)。Task1完成不等于数据隔离、整个流水线或最终准确率验收完成。
