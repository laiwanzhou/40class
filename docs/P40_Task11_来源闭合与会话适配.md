# P40 Task11 来源闭合与会话适配（Source Closure and Session Adapter）

日期：2026-10-29，Asia/Shanghai。分支 `experiment/teammate-single-teacher-task1`。

**状态：Task11 已推进来源预检和可独立实现的算子，但尚未完成。正式教师库（Teacher Bank）和 A8 生成被预检阻断。** Task9／10 已由三个独立审查者复审 PASS，见 [P39](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P39_Task9_Task10_独立复审.md)。用户纠正后续为 Task11，本轮不推进 Task12。

本文件是交付与阻塞证据，不是新计划或交接。唯一 [P01 计划（Plan）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md) 与 [P02 交接（Handoff）](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md) 原位维护。

## 已完成部分（Implemented Scope）

1. **显式会话适配（Session Adapter）。** `session_metadata.py` 只消费新的公共时间侧表，按精确 ID／user 对齐，以 `(partition,user,date)` 分块后调用原 `build_sessions(...,'anonymous_date')` 和 `date_session_lists`，映射回全局行索引。未知时间行不连边并单独保留，后续算子应对其返回 base；拒绝少 ID／重复 ID／错误用户／伪装的未知时间。没有从动作名或旧会话对象恢复会话。
2. **P310 优先覆盖和锐化（Precedence and Sharpening）。** `source_teacher_bank.precedence_targets` 实现 `route=new_group!=old_group`，route 行覆盖 sequence，其余保留 sequence；有效行被选类为 0.9805，另外39类各 0.0005。全缺失行保持同 phase 先验、target_mask=false，保留原中间 route／precedence_prediction 和标准目标字段。这只是纯算子，尚无正式 A8 输入，不将 toy 输出登记为正式教师。
3. **生产链拒绝（Producer Rejection）。** 预检拒绝缺保留 producer、多余排除列、历史输出冒充新 producer、被删大视觉根。当前版本化闭合文件是 evidence-only，不能靠翻转 ready 布尔值变成正式 bank 就绪证明；未知 schema／无生产链证明也拒绝。
4. **A18 谱系修正（Lineage Correction）。** 对 pinned `a18_full_teacher_data.py` 实际消费语句及 `p100a_global_teacher_data.py` 路径作 AST 检查，确认 A18 依赖 VideoMAEv2 和 InternVideo2，当前 Task11 视图显式排除。新增 `protocol/task11_source_closure_v1.json`，parent 绑定原 roster 路径／摘要，逐文件绑定只读来源证据。它是未完成的版本化闭合证据，**不是注册完成的 TeacherRoster／TeacherBank Artifact**；完整消费端待后续闭合后再绑定。没有覆盖旧 roster、更新旧协议摘要、改 Task1–10 祖先或用 A5 替换 A18。

## 真实公共时间侧表验收（Public Metadata Acceptance）

| 分区（Partition） | 完整行数 | 已知时间 | 未知时间 | 源会话数 | 重复会话列表数 |
|---|---:|---:|---:|---:|---:|
| train12 | 2039 | 1956 | 83 | 524 | 33 |
| development2 | 388 | 385 | 3 | 97 | 6 |
| refit14 | 2427 | 2341 | 86 | 621 | 39 |
| final4 | 609 | 590 | 19 | 171 | 9 |

每行仍在完整分母内；这些是**无标签时间可用性**计数，不是正确率或类别支持。final 的19条未知时间与18条全模态缺失不是同一概念，额外未知时间行仍保留其有效模态输入，不删除或伪造时间。跨用户／跨分区连边为0，公共时间验收标签载荷打开尝试0。未读取 final 真值、分布或类别覆盖。

TDD：会话接口先3 RED、纯 bank／谱系／锐化接口先4 RED；补充未知就绪证明拒绝先 RED 后修复。8项新核心测试通过7.28秒；最终加现有角色／ArtifactRegistry受影响测试，**30 passed，14.34秒**。正式公共预检4.01秒，CLI退出码2表示如期阻断正式生成，不是成功生成A8，也不是进程崩溃。没有原始目录扫描、全缓存哈希、网络训练或GPU运算。

运行命令：`python -B scripts/preflight_no_vote_teacher_bank.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml`。证据 `outputs/task5_audit/P40-task11-preflight.json/.log`、`P40-task11-regression.log`，源码 `session_metadata.py`、`source_teacher_bank.py`。

## 仍需闭合的材料（Outstanding Evidence）

### P12 所选训练配置（Selected Training Configuration）

`build_complete_p11_oof.py` 消费 `aligned_multimodal/runs/p11_fp16_oof`。源包有多份 Skeleton／Depth 候选 JSON，尚无对应输出实际选中 `config_used.json`、完整训练命令或 checkpoint→配置／FP16转换绑定。不能挑一个同名相近配置就称复刻。Depth 公共路径已存在，不把它报成缺失模态。

Thermal 的 `run_imagenet_oof_folds.ps1` 证明 `resnet18_tsm_imagenet_12f.json` 用于 `thermal_baseline/runs/p11_thermal_imagenet`：12帧、144×192、15轮、seed20260720、ImageNet预训练等可知；尚未证明它到实际消费的 `p11_thermal_imagenet_fp16` 所选 checkpoint／转换链。源公共 initializer enum 明确为 `ResNet18_Weights.IMAGENET1K_V1`，缺的不是名称知识，而是实际所选配置与本 run 资产 receipt 绑定。

当前 Thermal 还缺公共 sidecar：`sample_id,user_id,thermal_path,thermal_available`，须与四分区 ID／user 精确对应。现有四模态 manifest 不能证明 Thermal 覆盖，缺表不能默认全不可用。对两份提交包的小配置／说明文件搜索未发现所需 `config_used.json` 或 P11 运行摘要；只读源包中的候选配置不等于选中证据。

### safe-base 中间适配生产链（Intermediate Adapted Source-base）

源实际依赖是：

`P85 KNN → 原中间结构目标＋decoder → 独立12轮适配 → P87S学生raw → P89 base_probability／safe decision → 原教师库`。

`p165_deployable_group_teacher.py` 的 `TEST_BASE` 读取 P89 文件内 `base_probability`；`p89_build_final_test_submissions.py:302` 明确该概率来自 `softmax(p87s_final_test_predictions_v1/student_logits.npy)`。`run_p87s_final_pipeline.py:285` 在此之前先运行12轮 `adapt_p87s_test_student.py`。这套旧中间目标来自 `build_p87s_test_structured_targets.py`：P85 KNN emission、三 outer-fold decoder摘要中位配置、fit transition、beam posterior和 entropy backoff。

因此它既不是当前A7普通softmax，也不是最终P310 A8目标。P01 Task11需先补齐该中间生产链及固定划分下的decoder参数来源／safe joint grouping选参契约，不能用Task12的最终A9-12倒填。旧外折摘要未打包时也不能凭默认参数代替。需要的证据包括对应 `p87_sequence_decoder_v1/summary.json`／实际命令与配置绑定、生产者输入与阶段边界；不需要导入其历史已训练权重或OOF预测。

### 当前进度的边界（Status Boundary）

已排除的A18从“未知小专家”修正为被删大根依赖，不表示剩余谱系都闭合；p89／p128等仍需追清。当前 `task11_source_closure_v1.json` 保留六组阻塞项。未实现完整 P12训练／global mixture／KNN／group模型／sequence gate／safe-base中间适配，不生成占位专家列、不复制先验凑bank，不创建 `task11_complete.json` 或正式A8／Task12产物。

Task9／10独立复审已放行，Task11的可独立源码部分已推进。下一步需先取得上述**精确配置与生产链证据**，再继续正式训练和A8生成；此前前十步模型保持可复用，旧global `formal_generation_ready=false` 与协议身份不变。
