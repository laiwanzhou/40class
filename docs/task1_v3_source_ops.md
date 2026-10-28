# v3.1 Task1 源操作与教师清单（Source Operations and Teacher Roster）

2026-10-28完成Task1接口、描述符导出及独立复审。仅Task1交付通过；未进入Task2或启动训练，未读取final标签。

## 已交付（Delivered）

- `no_vote_source_ops.py`通过AST读取最终调度器、实际适配命令及所需源码默认值，不导入或执行旧producer、不读取旧预测。每项参数及算子带path/symbol/line/SHA来源。
- 新v3 YAML与源配方建立明确运行时投影（Runtime Projection），固定MC3 16轮、MoBind 24轮、fusion 4+20、最终适配40轮及其实际学习率。更改轮数、LR、融合位置或增加大视觉根节点都会拒绝。
- `source_ops`、`teacher_roster`作为v3限定descriptor登记。必须为有效绑定JSON；roster有且只有对应source_ops父记录。不能登记`{}`或无父roster，不能借新kind放宽v2规则。
- 公共权重通过同卷硬链接复用已核对凭据，明确标注继承验证；不重新读取大权重SHA、不重新下载或声称重新做CPU初始化。v2协议身份不变。

实际消费47个源码/说明文件，共774664字节；没有全量读取1167个源文件内容，没有raw或大cache扫描。

30个原专家槽位：**19保留、9因被删大视觉根依赖排除、2来源仍待补齐**。同一VideoMAE六族头及十项固定Ridge机制不是多个大骨干。P128经hierarchical→unrestricted实际依赖链确认VideoMAEv2/InternVideo2根，因此明确排除。

## 正式生成门槛（Formal Generation Gate）

`formal_generation_ready=false`保持生效，缺项不能以猜默认值消除：

1. P12 S+D最终`config_used.json`或等价实际命令、架构及公开初始化绑定。
2. P12 Thermal最终配置/实际命令及ImageNet初始化绑定。
3. P89 safe-base/A18完整producer及其配置、资产、固定划分输入。
4. 对应两项未闭合槽位：`p89_safe_probability`、`a18_best_session`。

Task1的导出与阻断接口已经完成；上述原实验材料缺口仍阻止整条路线正式训练。本报告不把描述符成功当作所有teacher producer已实现。

## 复现与验证（Reproduction and Verification）

从实验工作树执行：

```powershell
D:/Anaconda/envs/PyTorch2.7/python.exe -B scripts/export_no_vote_source_ops.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml
```

最终绑定文件位于新run的`protocol/source_ops_manifest_task1_final.json`与`teacher_roster_task1_final.json`。中途旧描述符保留历史、不改旧身份；当前YAML只绑定最终版本。重复CLI输出同一ArtifactRef。

源操作记录SHA：`ffe653c21d9b10003593e50ed5692748ccbe2b7aa9afc380fd9aa4937d1f64e6`。

教师清单记录SHA：`d518d84c3a6f0e5da5cee230fbed1d0d9a852cebeaadbd14bcf6b10a7676585c`。

v3协议SHA：`2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`。

v2协议SHA仍为：`2640dfa234af5f18b6d438f6834223d6ceeb8b445afb35c1d91de5f3c7bddd5c`。

最终针对性回归（Targeted Regression）：68 passed in 22.71s，日志`outputs/task5_audit/v3-task1-final-verification-20261028.log`。未运行无关旧全仓测试。独立初审1P1+3P2均以回归或独立探针关闭；额度中断未当作通过，恢复后已实际复审与CLI重跑。

报告：[初审](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-28-v3-task1-independent-review.md)、[独立复审GO](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-28-v3-task1-independent-reaudit.md)。
