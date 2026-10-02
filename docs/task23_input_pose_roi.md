# Task2/3 输入与姿态ROI（Input Preparation and Pose ROI）

Task2已生成完整公共清单和分离标签；Task3代码移植完成，真实train12一条22帧验收及续跑校验通过。尚未全量构建2039/388/609条ROI，不进入Task4教师训练。

## 输入准备（Trusted Preparation）

在实验工作树根运行：

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/prepare_no_vote_inputs.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --source-manifest metadata/manifest.csv --public-output outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/protocol --private-output C:/Users/LaiWanzhou/AppData/Local/Temp/cuhkx_no_vote_labels/fixed-split-single-teacher-v2 --data-root D:/work/2026.7.14_kaggle/datasets/Small-Model-Track/train
```

这是允许读原带标签清单的独立准备步骤，不运行模型。公共清单包含不透明样本ID（opaque ID）、用户、四个原始模态定位路径和可用性；最终标签及旧ID映射只存入私有目录。生成入口不调用准备脚本，也不接收私有目录路径。

| 分区 | 全行数 | IR文件可用 | Depth/Skeleton文件可用 | IMU CSV文件可用 | 所有纳入模态缺失 |
|---|---:|---:|---:|---:|---:|
| train12 | 2039 | 1957 | 1956 | 1914 | 82 |
| development2 | 388 | 385 | 385 | 369 | 3 |
| refit14 | 2427 | 2342 | 2341 | 2283 | 85 |
| final4 | 609 | 591 | 590 | 580 | 18 |

IMU统计按队友读取器的CSV规则，仓库文本不算传感器数据；实际文件可用580与早期估计584不同，已同步规格。CSV存在不代表其每个设备都有有效数值，Task6会进一步验证有效设备和时间点。样本分母和用户划分没有变化。

## 姿态与ROI（Pose and ROI）

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/build_no_vote_pose_roi.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition train12 --max-trials 1 --device cpu
```

该验收输出标记为partial，不可当作全量缓存使用。实际验证的是user5一条22帧IR，22帧Skeleton完全按时间戳对应；使用公开YOLO推理，没有教师/学生训练。未来全量构建去掉max-trials，各分区分别输出；正式学习产物拒绝未完成的partial祖先。

原P28/P29 main、类别排序及类别摘要已移除；调用已验证快照中的检测、跟踪、重试、插值和几何函数。以IR观察帧为主轴，Depth/Skeleton只在准确对应时传递或对齐。缺ROI保持整帧回退，缺IR不会伪造视觉帧。

续跑（resume）验证配置、权重、生产代码、输入文件集合/内容、每条输出cache SHA及最终产物记录；新增帧、中断缓存损坏或生产者变化均不能静默复用。旧诊断只归档在本轮outputs中，原数据与队友源码保持只读。

## 来源注册表（Artifact Registry）

产物记录保存文件/源码/原始输入hash、config、protocol、用户角色、样本ID、类别列、父记录及complete状态。递归验证select/refit监督人口；final预测必须有匹配的refit监督模型祖先，A9必须接完整final4 A8并绑定同一个A7/refit。预测声明分区后按公共清单精确连接，允许refit14预测包含其开发用户，但不允许把refit教师用来选择development2。

公开JSON/CSV/NPZ直接标签字段被拒绝。准备状态与输入摘要不含私有final标签路径，生成读取器不读原带标签清单。完整验证见reports/2026-10-02-task23-verification.json。

本轮相关测试57项通过；全仓库498项通过、7项旧缓存缺失失败，失败集合与Task1基线相同，没有新增失败。数据、私有标签及ROI NPZ留在本地Git忽略目录，不推送。
