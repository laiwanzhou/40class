# 多候选 Skeleton 的视觉身份选择诊断

## 方法

- 范围限定为 fold-0 的 14 个 train 用户和既有完整配对样本；未读取 held-out validation 用户或 competition test。
- 用单候选帧拟合一个全局、固定的 Skeleton 3D→YOLO 2D 线性投影，不进行逐帧自由拟合。
- 对多候选帧中的每个 Skeleton 分别计算与 IR 主 YOLO pose 的 12 个公共关节 RMSE，选择 RMSE 最低者。
- `relative_margin=(second-best-best)/second-best`。margin 越大，视觉证据越明确；这里只给出可辨识度，不把无标注选择冒充 ground truth accuracy。

## 覆盖与结果

| 指标 | 数值 |
|---|---:|
| 原始审计中的 train-fold 多候选 JSON | 3049 |
| 可参与视觉选择的多候选帧 | 2506 |
| 对 train-fold 多候选 JSON 的覆盖率 | 82.1909% |
| 涉及 trial | 211 |
| 2/3/4 候选帧 | 1825 / 669 / 12 |
| 选择 JSON 第一个候选的比例 | 96.0096% |
| 视觉排序改选非第一候选 | 100 |
| margin ≥10% | 98.8428% |
| margin ≥20% | 97.4062% |
| margin <20%，标记 ambiguous | 65 |
| median relative margin | 0.768315 |

## 伪干扰验证

单候选帧作为已知正候选，并从同一动作、不同 trial 抽取一个 Skeleton 作为干扰。每个目标用户都使用排除该用户后拟合的投影，避免目标用户泄漏进投影校准：

| 策略 | 覆盖率 | Top-1 |
|---|---:|---:|
| 全部 pair | 100.0000% | 93.4750% |
| winner margin ≥10% | 93.6000% | 95.9001% |
| winner margin ≥20% | 86.9875% | 97.6146% |

这个 benchmark 检验姿态形状是否能排除同动作干扰，但真实多候选经常是镜像、反射或近似同步人物，通常比随机同动作干扰更难，因此不能把该 Top-1 当作真实多人选择准确率。

## 判断

- 可以为每帧候选产生可复现的视觉匹配排序，完整逐帧结果见 `multi_person_candidate_decisions.csv`。
- margin 较高的帧可以自动选人；margin 较低时，原始 Skeleton 已丢失图像位置和平移，单凭规范化姿态无法可靠区分真人、镜像或做相似动作的人。
- 第一版建议只自动接受 margin≥20% 的候选；其余标记为 ambiguous，不补造身份。若要覆盖更多帧，应增加序列级动态规划和人工抽样核验，而不是默认信任 JSON 第一项。
