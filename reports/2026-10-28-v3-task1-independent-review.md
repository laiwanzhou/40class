# Task1 独立审查报告（Independent Review）

审查日期：2026-10-28。基线（Base）：`9972da2`。范围为 Task1 源操作导出、30 槽教师谱系（Teacher Lineage）、v3 配置绑定、最小描述符（Descriptor）注册及公开初始化凭据硬链接复用。未读取 final 标签、历史评分、旧预测或大缓存；未训练，未运行全测试套件或全源码 SHA。

## 初审结论（Initial Decision）

**NO-GO：初审发现 1 个 P1 与 3 个 P2，正在交由主智能体修复。** `formal_generation_ready=false` 与缺失 P12 最终配置/P89 生产链凭据是正确阻断，不作为缺陷。

### P1：YAML 生效参数与源配方可以漂移（Configuration Drift）

位置：`src/experiments/no_vote_protocol.py` 的 v3 `source_ops_bindings` 分支。

初始实现只验证两个描述符的哈希及顶层标记，没有验证 YAML recipe 与 manifest.recipe 的对应关系。保持原两份描述符及其 SHA 不动，把 `visual_student.fixed_epochs` 改为 999、`head_learning_rate` 改为 99，并向 recipe.retained_large_roots 添加 vjepa2，`load_protocol` 均成功；返回配方确实包含这些值。已用临时 YAML 复现，公开权重只检查凭据与大小。

影响：后续构造可使用一套与已锁源操作不一致的配置，但仍拥有合法 v3 协议身份。违反计划 Task1“导出的 recipe 才是后续构造模型的唯一参数来源”。建议从源 recipe 建立明确运行时投影（Runtime Projection），统一比较预算、别名、端点、教师根与所有生效字段。

### P2：描述符登记与 CLI 验证强度不一致（Descriptor Validation）

位置：`src/experiments/artifact_record.py` 的 descriptor 角色检查。

初始实现允许任意 `{}` 文件以 `kind=descriptor, phase=public, config={source_contract_version:3}` 登记为 `source_ops` 或 `teacher_roster`，后者无任何父记录（Parent）仍通过 register/verify。小复现仅在临时目录建立协议副本及小 JSON，没有写正式产物。

影响：仅从合法 ArtifactRef 无法知道它是否为协议绑定的源描述符；调用 API 与正常 CLI 得到的可信度不同。建议验证唯一 JSON、协议内路径、绑定 digest、必要内容，并强制 roster 唯一父记录为对应 source_ops。

### P2：P128 的可闭合导入链没有展开（Incomplete Producer Lineage）

位置：`src/experiments/no_vote_source_ops.py` 的 `_original_slots` / `producer_names` 根扫描。

初始输出把 `p128_hierarchical_multimodal` 写为 `roots=[] / unresolved`。但源码证据已足够确认其依赖被删根：`p128_hierarchical_multimodal_full_oof.py` 导入 `p91_hierarchical_multimodal_teacher.build_data`；后者导入 `p91_unrestricted_fusion_teacher.build_cohorts`，该函数 146–147 行直接消费 VideoMAEv2 和 InternVideo2 特征，且 hierarchical.build_data 的 streams 使用这两类输入及 VideoMAEv2 depth/thermal/hand。

最小复现：`compile_source_ops(source_root)` 后查找该槽，初始值为 `(p128_hierarchical_multimodal, [], unresolved)`；仅阅读上述三个小 Python 文件即可核实反例。

影响：这不是外部证据缺失，而是 Task1 的谱系追踪未完成。应记录至少 `videomaev2/internvideo2` 及调用路径并排除（Excluded），保持真正缺证据的 P89/A18 为 unresolved。无需执行任何 producer。

### P2：部分生效构造参数未进入导出配方（Missing Effective Parameters）

位置：`src/experiments/no_vote_source_ops.py` 的 student `_kwargs` 白名单及 fusion 数值筛选。

初始 `visual_student` 未导出源 `fusion_mode=gated`、`temporal_modeling=true` 与 exact-time/cross-view/spatial/region/structured 全关闭等构造参数；`fusion` 因只保留数值/bool 与两个特例字符串，漏掉真正控制构造分支的 `--fusion-position=clip`。机制/视觉头还没有明确导出源码 StandardScaler→Ridge 的 `solver=lsqr, tol=1e-5, max_iter=5000`。

最小复现：检查 `compile_source_ops(...)[recipe]` 的键；与 `train_p86_visual_pixel_oof.build_model`、`P86MC3VisualStudent.__init__`、`train_p86_mobind_fusion_proxy.parse_args`（104–115 行 fusion-position）及 `train_p46_videomae_head.make_model` 对照。所有检查仅涉及源文本。

影响：锁定了文件 SHA 不等于已提供后续模型唯一可用的完整构造参数。建议完整导出所选分支的参数，并对这些参数附实际调用或默认值来源；不可要求后续 Task 再猜一套隐式默认值。

## 已核实的正面结果（Verified Positives）

- 初审编译仅实际读取 43 个源/说明文件；不导入旧 producer，不读取旧数组。
- MC3 16 轮、MoBind 24 轮、fusion 4+20 与最终 40 轮适配覆盖均来自实际命令；12 轮额外端点保持独立。
- 当前 30 槽原始顺序、六族头与十个机制槽得到保留；P231/P238/P253/P306 的 VideoMAEv2/InternVideo2 删除后果没有被伪造特征填补。
- 硬链接复用公开初始化沿用已验证凭据、来源固定及大小检查，没有重新哈希公开大权重；保留验证范围说明。
- 没有把正式生成受阻本身判作 Task1 错误。

## 修复后复审（Re-review）

待主智能体完成针对性修复后更新；本段未完成前，不应把初审发现视为已关闭。
