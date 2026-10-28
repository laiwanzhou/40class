# P03 Task2 准备与缓存复用（Preparation and Cache Import）

日期：2026-10-28。分支：`experiment/teammate-single-teacher-task1`。唯一计划为 [P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为 [P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。本文是交付记录（Delivery Record），不是另一个计划或交接。

Task2完成现有四模态准备与缓存复用；Task3及后续训练没有启动。Task1已发现的源证据缺项仍保持`formal_generation_ready=false`。

## 准备结果（Preparation Results）

| 分区（Partition） | 公共样本数 | 可提取时间的样本数 | 标签用途 |
|---|---:|---:|---|
| train12 | 2039 | 1956 | 复制允许的训练标签 |
| development2 | 388 | 385 | 复制允许的user6/7验证标签 |
| refit14 | 2427 | 2341 | 复制允许的重拟合标签 |
| final4 | 609 | 590 | 没有读取、复制或检查标签 |

公共清单沿用精确不透明样本ID（Opaque IDs）与用户顺序。没有重新扫描原始数据目录。时间来自P28 acquisition_ids，不解析class/trial ID；`start_seconds`为日内秒（Seconds Since Midnight），未知日期留空、时间NaN、可用性false。上述时间数量不等于IR可用数量，不删除未知行。

真实准备耗时120.41秒，六个导入根为refit14和final4的P29、公开冻结VideoMAE特征及pixels；它们共享的新P28、公开initializer和manifest祖先全部位于v3。train12/development2从refit14公共缓存按精确ID取子集，不继承监督模型或预测。

缓存使用同卷硬链接（Hard Links）；不额外复制多GB像素。新记录保存来源protocol/ref/digest/路径、大小与mtime，旧记录不改identity，旧ref不进入新祖先链（Ancestry）。导入继承历史摘要凭据，只验证元数据、ID/轴、掩码和消费切片，不声称本轮重做了全缓存完整性扫描。已登记导入目录禁止producer覆盖或r+续写；加载为只读内存映射（Read-only Memory Mapping）。

## 接口与角色（Interfaces and Roles）

- `import_public_cache(source, destination_protocol, rows, context, source_protocol=...)`：首次显式提供已验证旧协议；拒绝heads、targets、A2、sequence及任何包含学习祖先的缓存。失败不回退大文件复制或重新提取。
- `VerificationContext(protocol)`：共享元数据/直接文件校验、已加载数组和导入memo。键包含protocol/ref/path，stat变化失效；祖先只遍历去重元数据。六个头可用同一个context调用`_load_features`，不再各自加载/全SHA。
- v3角色表（Artifact Role Matrix）：`source_ops/teacher_roster/source_transform/recording_metadata`为无fit描述符；`source_expert/<name>`、`visual_head/<family>`、`visual_mechanism`预测绑定自身同phase模型。`global_mixture/safe_base/group_old/group_new/gate/sequence/fullwindow_knn`显式限定学习类型；未知预测stage拒绝。teacher_bank要求同分区、完整ID、按列顺序的专家预测及roster父记录和`[N,E,40]`/availability/fallback轴。
- teacher-only公开资产（Public Assets）必须在recipe的`teacher_only_assets`中显式绑定origin和文件摘要；本轮不虚构P12资产，不给v2 dataclass添加默认字段。
- Thermal补充公共表白名单为`sample_id/user_id/thermal_path/thermal_available`；额外class/action/trial列拒绝。实际表尚未提供，不把未知误写为全部不可用。P12/Task11生成前仍需准备该输入和源配置/资产证据。

准备命令（工作树内执行）：

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/prepare_no_vote_v3_inputs.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml --source-config configs/experiments/teammate_single_teacher_fixed_split.yaml --cache-refs outputs/task5_audit/P04-task2-source-cache-refs.json
```

不要用旧`prepare_no_vote_inputs.py`重新打开canonical带标签清单；v3已明确拒绝该入口。准备CLI的显式引用清单只列六个公共缓存根，不遍历历史监督产物来找候选。

## 验证证据（Verification Evidence）

95项受影响测试通过，52.14秒；另补source_transform必须有descriptor类型source_ops父记录的探针，先失败再修复，角色模块11项再次通过。没有运行全仓旧测试或原始数据全SHA。

真实train12一条像素样本验收通过：`[2,16,3,160,160]`，无label/teacher字段，像素mmap为r；六次特征加载共享对象；验收中raw/权重完整SHA调用为0。只证明实际消费与声明的有界检查（Bounded Checks），不外推到未消费文件。

产物和日志在当前v3 run或工作树的以下位置：

- `outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v3-source-ops/protocol/task2_prepared.json`
- `outputs/task5_audit/P04-task2-formal-prepare.log`
- `outputs/task5_audit/P03-task2-final-tests.log`
- `outputs/task5_audit/P03-task2-descriptor-red.log`与`P03-task2-descriptor-green.log`
- `outputs/task5_audit/P03-task2-real-acceptance.json`

Task2没有独立子智能体复审；计划下一复审节点为Task3后，进入长训练前仍需满足P01各门槛。没有读取final标签或更新main历史文档。

## 文档入口整理（Canonical Documents）

本分支只保留P01指导与P02交接，持续原位更新。16份旧实验计划、3份旧交接和1份旧路线文档移到`docs/archive/P04–P23`，标记为历史资料，不再指导本分支；原审计输入快照保持原样。P03仅记录本轮交付事实。
