# P02 当前实验交接（Current Experiment Handoff）

2026-10-28，Asia/Shanghai。工作树D:/work/2026.7.14_kaggle/_single_visual_processing_replication，分支experiment/teammate-single-teacher-task1。Task1实现与独立复审GO，Task2实现及真实准备、Task3几何一致性均已完成并获独立复审GO，Task4未开始，无新训练或自动化。本文件是本分支唯一交接入口（Canonical Handoff）；保留当前路径，后续直接更新，不新增日期版handoff。

权威计划：[v3.1计划](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)。结果：[Task1交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/task1_v3_source_ops.md)。复审：[独立GO报告](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-28-v3-task1-independent-reaudit.md)。完整历史ledger在本工作树`.superpowers/sdd/2026-09-23-teammate-single-teacher-fixed-split/progress.md`。

新代码包括no_vote_source_ops.py、export_no_vote_source_ops.py、v3配置，以及artifact_record/no_vote_protocol/no_vote_weights的限定扩展。源代码AST导出47小文件，774664字节；30槽19保留/9排除/2未闭合；配置与源配方逐字段绑定，descriptor payload/唯一source父合法性检查，旧v2身份不变。最终68项针对性测试通过；初审1P1+3P2实际复审关闭。没有final标签读取、旧预测复用、raw/大cache全SHA。

v3 run为outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops；配置只绑定protocol/source_ops_manifest_task1_final.json及teacher_roster_task1_final.json。旧中途描述符保留历史、不改identity。公开权重为继承凭据的同卷硬链接，不重新下载，不声称重新初始化验收。

**正式生成仍阻断（Formal Generation Blocked）**：缺P12 S+D最终配置、Thermal最终配置及P89 safe-base/A18完整生产证据；formal_generation_ready=false。不能伪造默认值、删掉必要小专家或直接设true。Task1/2接口交付不等于整条teacher bank就绪。下一步按P01为Task4目标恢复补齐上述源证据；可继续讨论实现，但当前不能启动正式监督生成或训练。

固定划分不变：train12/val user6与7/refit14/test user4、17、23、24，不做OOF。final联合40类只有此前独立布尔核实授权，不得再次读取标签；一次最终评估仍在全部候选冻结之后。原始数据、队友快照和提交包只读；main历史文档等待完整实验结束。

v2历史69.07%教师/48.71%学生保留，旧resume与sequence数组已清理，几何/冻结特征/pixels/公开权重/正式checkpoint/targets及refs保留。v3不能把旧监督产物或退休sequence当新祖先。

## Task2交付与接续（Delivery and Continuation）

详见 [P03 Task2交付](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P03_Task2_准备与缓存复用.md)。新增no_vote_cache_import.py/no_vote_roles.py及prepare_no_vote_v3_inputs.py，扩展公共VerificationContext；禁止v3调用旧canonical标签准备入口。四公共清单和三份允许标签已复制；没有读取或复制final标签，没有重新扫描raw目录。

真实准备120.41秒完成：refit14/final4各自P29、冻结features、pixels六个根记录，P28/initializer/manifest按新协议拓扑登记、祖先共享；payload同卷硬链接（Hard Links）、按不可变缓存（Immutable Cache）消费，旧ref仅作为来源凭据（Provenance）。来源摘要继承历史凭据，本轮不声称重新校验全部旧payload。四份recording_metadata位于run/protocol/recording_metadata/，日内秒不使用pixels epoch秒，未知时间保留NaN。

95项受影响针对性测试通过（52.14秒）；随后补充source_transform父记录类型探针并RED→GREEN，角色模块11项复跑通过。真实train12一条无标签像素样本通过，shape=[2,16,3,160,160]；六次特征加载共享对象，像素mmap_mode=r，验收raw/权重全SHA调用0。Task2初交付时未独立审；本轮Task3后已由独立审查者连同Task2实际复审GO（P25），唯一缓存问题已修复复验。

正式产物入口：run/protocol/task2_prepared.json；证据outputs/task5_audit/P04-task2-formal-prepare.log、P03-task2-real-acceptance.json及P03-task2-final-tests.log。run仍为outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops，protocol SHA=2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a，旧v2身份未修改。

Thermal额外公共字段白名单（Whitelist）已定义为sample_id/user_id/thermal_path/thermal_available，实际补充清单尚未提供；缺失表示尚无证据，不表示所有样本无Thermal。Task11生产前必须补齐清单、原配置和公开资产绑定。未触碰main工作树或两份提交包。历史资料已移入docs/archive/P04–P23，不作为当前计划或交接；审计原输入快照保留。


## Task3完成与复审（Task3 Completion and Review）

交付见[P24](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/P24_Task3_姿态与ROI一致性.md)，独立审查见[P25](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P25_Task2_Task3_独立复审.md)。本轮只增加load_pose_roi_sample、单样本源对照CLI、针对性测试与有界_load_ops加载；不改compatible_frame_map/acquisition_ids/_process_trial本体，不重新生成2427/609条几何。

真实样本22帧、14个ROI字段精确一致，retry flags、scene/person/workspace crop/quality/masks/frames/times和日内秒一致；5个直接源码与最终提交包code/training/project/aligned_multimodal字节一致。CLI耗时8.614秒，无detector/raw全SHA；正式小证明run/protocol/task3_geometry_parity.json，日志outputs/task5_audit/P24-task3-real-parity.log。

审查者独立核对六导入根/12唯一祖先，全部新protocol/新run/无学习祖先。发现Task2 ArtifactRegistry.read中for key覆写元组缓存键，只缓存SHA却重复解析JSON；已改for field并加Path.read_text计数RED→GREEN。审查者同一正式manifest连续读3次，修复前3次JSON/不同对象，修复后1次JSON/同一对象/tuple key。修复后51项针对集成测试通过62.91秒，P25最终GO仅Task2＋3。无需重跑Task2两分钟导入或更改旧身份。

Task4–14未开始，正式生成仍受上述P12/Thermal/P89/A18证据阻断。后续坚持唯一P01计划/P02交接、有限直接输入校验，禁止重启旧全raw或全cache SHA。
