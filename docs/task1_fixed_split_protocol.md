# Task1 协议与公开初始化（Protocol and Public Initialization）

本实验分支只实现Task1：固定分区、公共类型、源码校验、公开权重取得/初始化验证。没有实现Task2数据准备或后续训练，未读取比赛样本。模型权重位于Git忽略的outputs目录，不推送至仓库。

## 使用步骤（Usage）

在仓库根运行；现有队友源码快照位于相邻的`_teacher_branch_upload/teammate_teacher/project`。新机器需先取得仓库`teacher`分支源码，或通过`--source-root`指定已取得的project目录。源码清单位于project上级，必须匹配1167文件及固定SHA256。

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/acquire_no_vote_weights.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml
```

该命令验证源码，下载/复用公开VideoMAE-Large、MC3-18、YOLO11n-pose权重，在CPU上初始化并校验模型结构，不执行视频推理或训练。VideoMAE复用旧目录时只读原始公开权重，并用完整官方文件SHA256核验；配置与处理器文件来自指定公开修订（revision）。MC3与YOLO使用固定公开URL及完整SHA256。

已有权重时使用离线验证（offline verification）：

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/acquire_no_vote_weights.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --verify-only
```

三个模型的初始化验证通过后才生成正式协议身份（sealed protocol identity）。`load_protocol(..., verify_assets=False)`只供bootstrap/配置检查使用，其`identity()`会拒绝作为已验证协议。公开清单和训练标签在Task2之前不存在，Task1只定义其路径，不读取或生成它们。

## 本地产物（Local Artifacts）

`outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/protocol/`中：

- `verified_source.json`：源码清单身份与可用上游符号。
- `weights_manifest.json`：公开来源、全部文件SHA256、模型初始化与偏置验证。
- `resolved_protocol.json`：规范化、不可变配置及实际资产绑定。
- `task1_complete.json`：Task1完成证据，不表示Task2或训练完成。

协议包括train12=2039、development2=388、refit14=2427、final4=609及固定用户名单。生成配置没有final标签路径；独立评估资料目录属于后续Task2/13。fixture协议及权重只能用于测试，正式协议拒绝它们。

## 测试与范围（Tests and Scope）

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B -m pytest tests/test_no_vote_protocol.py tests/test_teammate_source.py tests/test_no_vote_weights.py -q
```

源码加载器同时检查顶层模块、已缓存的快照依赖、导出函数/类的来源，避免旧工作树的同名辅助模块进入新实验。导出第三方函数不被标作队友快照函数；第三方库应直接导入并记录版本。

Transformers4.49使用原生分离q_bias/v_bias；此环境验证48个偏置，误差为0，不需要重写偏置。新版query/key/value偏置布局也有回归测试及恢复路径，但没有声称本机运行了Transformers5。

本实验沿用已确认的单教师配方；配置中已固定同phase A2 anchor、显式motion辅助/可靠性loss masks以及公开冻结特征共用规则。这些是Task4/5/10的后续实现契约，不是Task1训练结果。
