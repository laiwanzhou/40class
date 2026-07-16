# 任务03阶段进度：正式 Baseline 完成与单模态优化路线

## 当前状态

- 旧 `fold_0` 的 12 名训练用户 / 6 名验证用户正式六模态 Baseline 已完成。
- 旧划分已原样归档为
  `metadata/splits/fold_0_12train_6val_20260715.json`，对应历史报告和训练输出继续保留。
- 已仅根据 `metadata/manifest.csv` 的数据结构穷举 3060 个四用户验证组合，生成新的 14 名训练用户 / 4 名验证用户 `fold_0.json`。
- 候选排名、评分定义、六模态样本比例和类别覆盖记录在
  `metadata/splits/fold_0_14train_4val_candidates.csv` 与
  `metadata/splits/fold_0_14train_4val_selection_report.md`。
- 用户选择没有使用模型预测、Accuracy、loss、混淆矩阵或测试集信息。

## 阶段边界

新 14/4 划分改变了验证用户，因此其结果不能与旧 12/6 Baseline 做严格的一对一精度比较。旧 Baseline 是已完成的历史基准，新划分用于检查增加训练用户后六种原始单模态方案的表现。

本阶段只生成和验证划分，不修改六个训练 YAML，不修改模型、batch size、input size、帧数、序列长度、学习率或训练轮数，也不启动训练。

## 下一步

1. 人工审核新 `fold_0.json`、前 50 名候选和选择报告。
2. 审核通过后，使用原始六模态正式配置在新 14/4 fold 上重新运行 Baseline，并使用独立输出根目录。
3. 新 14/4 Baseline 完成并分析收敛、类别覆盖与错误模式后，再开始单变量参数消融。
4. 在新 Baseline 完成前，不进行 input size、帧数、序列长度、学习率、batch、增强或候选模型消融。

本文件是任务03后续工作的阶段补充，不替代原始 DOC-03 或旧正式 Baseline 报告。
