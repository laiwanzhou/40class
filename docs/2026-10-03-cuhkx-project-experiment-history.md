# CUHK-X项目工作与实验结果时间线（Project and Experiment History）

## 阅读范围与材料说明（Material Passport）

- 整理日期：2026-10-03，Asia/Shanghai。复核状态：**ANALYZED（已核对记录，未重新训练或补做消融）**。
- 首要来源：`laiwanzhou/40class` GitHub实际远端分支及其已提交报告。本次已fetch并核对31条非teacher远端分支，另单独核对teacher快照；不是只看main，也不是把本地同名文件当远端内容。
- 时间顺序优先取Git提交/报告首次归档时间。run名中的`seed20260715`只是随机种子，不能当训练日期；报告晚于实际运行时，以首次可追溯记录为准。
- ACC统一用百分比，差值用百分点（Percentage points，pp）。保留所有能查证的主要分支ACC，即使没有消融；缺少该路线自身结果时写“未发布可核实ACC”，不借用父分支结果填空。
- 小范围本地补充只保留三个影响主线判断的结果：层级多模态、Visual90正式结果和9月基础教师重建。它们的实现可在远端祖先/实验分支追溯，但最终结果文件尚非原GitHub报告；本页明确标记**本地补充（Local supplement）**并记录文件SHA256。这次将其聚合数值首次归档到main，不上传权重、逐样本标签或预测数组。
- 本文是历史记录，不是新模型选参表。历史14/4报告曾使用user4/user17/user23/user24作验证；这四名用户不能声称在整个项目历史中从未被观察。新固定流程仍隔离自己的final4标签，历史分数不用于修改已冻结的A1–A9配方。

术语：准确率（Accuracy，ACC）、宏平均F1（Macro-F1）、折外预测（Out-of-fold，OOF）、感兴趣区域（Region of interest，ROI）、时序卷积网络（Temporal Convolutional Network，TCN）、随机森林（Random Forest，RF）、批归一化（Batch Normalization，BN）、骨骼（Skeleton）、惯性测量单元（Inertial Measurement Unit，IMU）、红外（Infrared，IR）、热成像（Thermal imaging，Thermal）。模型名称保留官方英文。

## 评估口径必须先分清（Evaluation Protocols）

| 口径 | 用户与分母 | 如何使用历史ACC |
|---|---|---|
| P0：早期六模态fold0 | 正式基线的原划分，模态有效验证约987–1046条 | 只在同一模态、同一旧人口内比较；后来的14/4不能直接与它相减。 |
| P1：14训练用户/4验证用户 | user4、17、23、24验证；规范609条，视觉可用约590，IMU处理后573 | 7月底至8月初多数视觉/IMU实验。40类与难动作148条、Target16的222条必须再分开。 |
| P2：严格train14三折OOF | 内部用户选择epoch→重新拟合→外折评价；Skeleton2341条、IR2320条 | 比较同一OOF配置的消融；一个seed的canonical结果与三seed均值不是同一个指标。 |
| P3：X3D五用户开发集 | user18、20、21、3、9；800条 | A2/A3/A4-T的开发结果，允许同人口比较；不是user6/7成绩。 |
| P4：user21/user22开发集 | 可用IR324条 | Partial2/Partial1/分层LR在这一人口上成组比较。 |
| P5：user6/user7开发集 | 可用IR385、Thermal377，或全规范388条 | 最新主要开发口径；385与388的分母不能混用，开发ACC也不是独立最终test。 |

**历史“最高ACC”不是一条从20%一路累加到74%的同口径曲线。** 下面的每个阶段都记录当时的分母；不同口径的分数保留，但不硬算机制贡献。配对oracle需要真值挑选，是诊断上限（Oracle upper bound），不能称为可部署ACC。

## 1. 7月14–16日：数据清单、六模态基线和输入加速

工作从文档、规范manifest和按用户划分开始。第一轮统一搭建40类单模态模型：视觉使用从零初始化的MobileNetV3-Small＋时间均值池化，IMU/Skeleton/Radar使用轻量TCN。先跑两轮验证训练链，再完成较长基线。原main提交从`a7d8dea`到`73fcd4c`可追溯。[S01][S03]

| 模态 | 首次2轮ACC | 输入优化后2轮ACC | 正式最佳ACC | 正式Macro-F1 | 正式验证N |
|---|---:|---:|---:|---:|---:|
| IMU | 18.24% | 19.35% | 25.53% | 18.60% | 987 |
| Skeleton | 27.70% | 30.10% | 47.50% | 38.12% | 1000 |
| Radar | 8.94% | 8.73% | 13.45% | 2.57% | 996 |
| IR | 13.20% | 15.30% | 20.10% | 11.49% | 1000 |
| Thermal | 16.54% | 18.64% | 21.03% | 8.36% | 1046 |
| Depth_Color | 13.70% | 14.20% | 22.20% | 14.69% | 1000 |

**有益之处：** 建立了可比较的输入、类别列、验证输出与资源基线；更充分训练让所有模态在各自旧人口上高于两轮结果。输入优化把worker从0改为4，平均epoch加速约1.89–3.69倍，视觉吞吐提升明显。[S02]

**代价与边界：** 输入加速是效率收益，不能把两轮ACC波动全归于worker；Radar优化后点估计还稍降。正式视觉ACC仍在20%左右，说明**这套小型全帧/均值池化配方**较弱，不证明IR/Depth/Thermal模态本身没有潜力。7月16日随后更换为14/4划分，后续分数不能与这一表直接做差。

## 2. 7月16–31日：IMU结构化处理、统计特征与紧凑随机森林

先处理CSV结构、时间戳、质量控制、缺失mask与10Hz网格，再形成仅训练人口拟合的插补/归一化。`IMU`分支主要发布这些接口；正式神经TCN参考ACC **28.80%**、CE＋SupCon参考均值 **30.66%**见后续RF报告，它们是处理后口径，不能与987条的25.53%硬算增益。[S04]

| 方案 | ACC | Macro-F1 | 验证口径/资源 |
|---|---:|---:|---|
| Balanced RF，三seed均值 | 41.07% | 32.23% | 同RF开发口径；约103.41 MiB |
| 150树备用方案，三seed均值 | 41.13% | 32.53% | 约7.63 MiB |
| 150树＋min_samples_leaf=4，三seed均值 | **41.88%** | **33.52%** | 约5.92 MiB |
| 正式复现seed20260725 | **42.41%（243/573）** | **34.44%** | 2184训练/573验证；全数据部署包约7.66 MiB |

**对ACC有益：** 紧凑森林相对接受的Balanced RF三seed均值增加**0.81pp**，Macro-F1增加1.28pp，大小约减少94.3%。统计特征/RF方案也观察到高于该阶段TCN的ACC，但算法、特征与容量共同改变，不能把差距全赋给单个统计量。[S04][S06]

**副作用/边界：** 更激进压缩候选只有40.66%，不是越小越好。无损序列化保留预测，**它的作用是减小文件，不是提高ACC**。2757条全数据生产模型的重代入ACC98.22%属于训练诊断，不能当泛化结果；神经/树模型在手部小动作上仍存在显著类别差异。[S05]

## 3. 7月31–8月5日：Depth视野、姿态ROI、IR配对及交互专家

### 3.1 六块Depth视野没有达到预期

`depth-six-patch`在14/4可用视觉590条上取得**22.54%**、Macro-F1 **13.51%**。其注意力出现固定空间偏好，难动作表现弱。没有同14/4的普通Depth基线，因此不能拿旧1000条的22.20%声称六块视野提升0.34pp。[S07]

### 3.2 姿态ROI与配对IR确有局部增益，也有类别损伤

| 实验 | ACC | Macro-F1 | 同组比较 |
|---|---:|---:|---|
| E0全局Depth，难动作148条 | 30.41% | 26.25% | 同596训练/148验证 |
| E1姿态ROI Depth | **37.16%** | **31.14%** | 相对E0 **+6.76pp ACC** |
| 严格公共子集Depth-only | 37.84% | 25.97% | 同595训练/148验证 |
| Depth＋严格配对IR | **40.54%** | **36.33%** | 相对公共子集 **+2.70pp ACC** |
| 上述双模态模型，IR归零 | 27.70% | 17.81% | 推理mask控制 |
| 上述双模态模型，IR样本打乱 | 25.00% | 24.31% | 推理shuffle控制 |

ROI集中到人体/上肢/手部后，该难动作子集ACC提高；IR归零/错配使同一双模态模型明显退步，支持它确实利用了IR及配对信息。不过mask/shuffle是推理敏感性，不等于重新训练后的单模态因果差值。[S08][S09]

**副作用：** ROI增益并不覆盖所有动作：擦碗、服药等曾退步，电视/游戏仍弱；E1峰值显存约2.59GiB，E0约0.68GiB。加入IR也让一些动作F1下降，验证loss变大。增加视图/模态有空间与时间开销，不应只看总体ACC。

### 3.3 扩到40类和交互专家后的结果

| 方案，14/4视觉590条 | ACC | Macro-F1 | 解释 |
|---|---:|---:|---|
| 40类Depth/IR ROI best-accuracy，epoch9 | **45.08%** | 32.98% | 可追溯停止结果 |
| 同运行best-macro，epoch8 | 44.41% | 33.93% | 少0.68pp ACC，但保护更多小类 |
| 交互MobileNet＋TCN专家 | **46.27%** | 34.98% | 相对epoch8基线+1.86pp；纠错22、误伤11 |
| 交互ResNet18专家 | 46.10% | **35.62%** | 纠错18、误伤8；ACC略低于MobileNet专家 |

**收益：** 交互专家对餐具、翻页、部分桌面操作有价值；ResNet18改善部分原先损伤动作与Macro-F1、校准误差。

**副作用：** 交互MobileNet专家让零F1类别从5增至6，并伤到服药/体温/擦碗等；ResNet18模型参数字节约59.56MB，对照MobileNet约18.61MB，未带来更高整体ACC。best-accuracy与best-macro的选择也会改变小类覆盖。[S10][S11][S12]

## 4. 8月6–10日：跨用户对比学习、Target16与全序列路线

`B2-256`后续基准是**46.27% ACC / 39.28% Macro-F1**，来自epoch25的同590条验证测量。这里与上面的交互专家不是同一检查点，虽然ACC数值相同。

| 路线 | 当时ACC | 主要优劣点 |
|---|---:|---|
| 用户SupCon，epoch18 | **42.37%**，相对B2 **−3.90pp** | 高置信错误207→165、loss下降，但泛化ACC与动作线性探针退步；对比任务很快饱和。 |
| Target16分层专家，目标16类222条 | **36.04%**；同子集B2为32.43% | 小动作专家闭集点估计+3.60pp，仍严重跨用户过拟合。 |
| Target16分层专家接回40类 | **48.14%**；B2为46.27% | +1.86pp；25纠错/14误伤。alpha=1为本验证扫描出的诊断，不是独立部署证明。 |
| Target16线性残差，目标16类 | **32.88%** | 极小头只训练3088参数，但收益仅+0.45pp。 |
| 线性残差接回40类 | **46.27%**，净增0 | 最好alpha=0等于不用它；缩小容量丢掉原分层专家收益。 |
| 原型SupCon分支 | **未发布可核实ACC** | 远端有实现，只有继承的原SupCon报告；不能写成该原型方案也跑了42.37%。 |
| Full-sequence multiscale TCN分支 | **未发布可核实ACC** | 发布了全序列方案，未见本路线正式结果报告。 |
| Scratch dual-spatial full-sequence分支 | **未发布可核实ACC** | 有实现，不能凭无大模型或父分支分数判定本实验已经失败。 |

用户对比学习的训练准确率下降，并不代表泛化变好了：B2与SupCon的训练/验证差距几乎相同，差距变小主要来自训练拟合下降；每批只含两类动作等采样变化也是混杂因素。[S13][S14][S15]

**序数Depth/IR-primary路线：** 先导表示raw/relative/raw+relative ACC分别为25.25%/27.46%/26.27%；更长正式训练最好ACC **27.80%**，best-macro检查点ACC **25.59%**，停止时仅21.86%。训练拟合继续上升而验证没有恢复，反对“再加epoch就能回到B2约46%”的解释；但整个输入/空间/时间包同时变化，不能单独怪罪TCN或relative表示。[S16][S17]

## 5. 8月12–13日：视频预训练与严格跨用户OOF；骨骼预处理选择

### 5.1 X3D-S Kinetics预训练IR路线

canonical严格train14 OOF取得 **56.55% ACC / 48.08% Macro-F1**，2320条。三seed为56.55%/56.94%/55.00%，均值56.16%；seed17有已记录恢复偏差，仍固定seed15为canonical。[S18]

在同800条fold0上，X3D为**57.13%**，固定10轮MobileNet/TCN sanity为**28.38%**，观察差距28.75pp。**这支持整条视频预训练路线更有竞争力，但对照只有一折短预算，不是完整三折匹配消融，不能把28.75pp全叫“预训练权重的净收益”。**

确定性训练评价96.57%与正式OOF56.55%相差40.02pp；最长视频桶仍弱。使用视频预训练没有自动消除跨用户泛化瓶颈。训练log里的随机增强准确率也不能直接代替eval-mode训练ACC来算gap。[S19]

### 5.2 Skeleton骨长尺度、图结构和时序消融

以下都在同严格嵌套train14 OOF、2341条上比较；与早期47.50%不同口径。[S20]–[S24]

| 方案 | ACC | Macro-F1 | 相对C1与决定 |
|---|---:|---:|---|
| C0：整条trial统一骨长尺度 | 38.70% | 25.02% | 预处理对照 |
| C1：逐帧骨长尺度 | **40.92%** | **30.08%** | 相对C0 **+2.22pp**；保留 |
| D1：轻量ST-GCN | **23.58%** | 12.54% | **−17.34pp**；所有14用户均未改善，低训练拟合支持该配方欠拟合 |
| D2：Joint/Bone双表示TCN | 41.18% | 29.27% | +0.26pp但Macro-F1下降，未通过替代标准 |
| T1：Segment-aware TCN | 41.95% | 32.44% | +1.03pp，但区间含0；多分段子集没改善，不能归因阻断跨gap边 |
| T2：T64→T96 | 42.25% | 32.12% | +1.32pp但收益集中fold0，只有5/14用户改善，未替代C1 |

C1是较清楚的**预处理正向证据**：C1−C0的既有配对bootstrap区间约+0.81至+3.63pp，报告也记录fold0反向。图结构或更细时序不是天然更好；当时D1失败仅评价这个轻量实现/配方，不代表所有ST-GCN都无效。T1/T2总体均值较高，也不能忽略稳定性和机制对照。

## 6. 8月14–16日：X3D解冻范围、学习率与分类头；骨骼三seed集成

### 6.1 user21/user22，324条

| X3D方案 | ACC | Macro-F1 | 同组差值 |
|---|---:|---:|---|
| Partial2，解冻最后两块 | **55.25%** | 41.87% | 单独开发基准 |
| Partial1，只解冻最后一块 | **50.31%** | 39.99% | −4.94pp ACC，长视频更受损 |
| Partial2但骨干分层降LR | **52.16%** | 40.91% | −3.09pp，骨干漂移下降却未转化为更好泛化 |

少解冻、低学习率都不能自动称为“更不容易过拟合”。这组确有匹配消融；但没有同324条的全骨干训练对照，不能据此证明Partial2优于全解冻。[S25][S26][S27]

### 6.2 五用户800条开发与user6/user7的385条开发

| 方案 | ACC | Macro-F1 | 人口 |
|---|---:|---:|---|
| A2完整时间覆盖 | 60.75% | 52.68% | 五用户800条 |
| A3开发候选 | 61.00% | 52.56% | 同800条 |
| A4-T，训练clip keep=0.5 | **61.50%** | 52.59% | 相对A2 +0.75pp；长时长桶退步 |
| Partial2匹配参考 | **53.25%** | 42.16% | user6/user7，385条 |
| Direct-Head | **51.69%** | 42.38% | 同385条，ACC −1.56pp、最差用户 −5.97pp |

Direct-Head缩小自定义头、减小训练/验证gap与错误置信度，却损失了可迁移决策能力。**gap较小不等于ACC较好。** [S28][S29][S30][S31][S32]

A3相对A2的ACC只有+0.25pp，同时最差用户ACC从53.93%降到52.81%；保留其61.00%的实际成绩，但不依据这一个差值推断某个超参数的独立贡献。

### 6.3 骨骼三seed等权集成

C1三个成员ACC **40.92%/40.71%/39.94%**；固定等权平均达到 **42.38% / 31.56% Macro-F1**，比canonical C1增加**1.45pp**。既有报告纠错108/误伤74，9/14用户改善，并通过当时替代条件。代价是推理参数/计算约三倍。这是有收益的集成，但不保证“教师数越多就一直增长”。[S33]

## 7. 8月20–21日：固定视野、Depth残差与Thermal路线

### 7.1 先解释61.50%与51.69%的误比较

历史复核指出两个实验验证用户不交叉且进入对方训练集；原始9.81pp差距不能都算分类头退步。用同canonical OOF比较人口，五用户57.13%与user6/user7的49.35%相差**7.77pp**，约占原始差距79%。这只是人口难度的同协议诊断，不是证明全部差距已经因果分解。真正匹配的Direct-Head效应是−1.56pp，A4-T dropout效应为+0.75pp。[S34]

### 7.2 user6/user7视野与Depth，385条

| 方案 | ACC | Macro-F1 | 收益或副作用 |
|---|---:|---:|---|
| Adaptive Partial2 | 53.25% | 42.16% | 参考 |
| 单个全局分层Single13 clip | **52.21%** | 41.14% | −1.04pp，算力减少但没有提高ACC |
| Single13＋trial固定context框 | **53.77%** | 42.75% | 对moving-context **+1.56pp** |
| 直接扩展raw四通道stem | **45.71%** | 35.54% | 后续报告记录的失败参考，对固定context IR **−8.05pp** |
| IR＋绝对Depth残差anchor | **54.81%** | 43.31% | 对固定context IR **+1.04pp** |
| IR＋ordinal-motion Depth残差 | **54.81%** | **46.20%** | 对绝对Depth ACC持平，Macro-F1 +2.90pp；仍未到0.63门槛 |

固定context避免人体框随帧移动的视野变化，已有匹配结果支持这套改法；直接扩展四通道stem退步，保持重复IR路径、经零初始化9参数adapter加入Depth的方案恢复到54.81%。这支持保留预训练输入路径的重要性，但不是纯参数量消融。45.71%的来源是后续报告的failed_expanded_stem_reference，不能假称资源资格head自己发布了正式训练报告。把Depth改成显式位移/速度/幅值主要改善类别平衡，而不是新增整体ACC；它与绝对Depth各独对16条，oracle58.96%不能当实际融合成绩。[S35][S36][S37][S38][S57]

### 7.3 Skeleton再做均衡采样：少数类改善不代表主ACC改善

保持三seed C1集成不变，只改train-scope平方根逆频采样：ACC **42.38%→40.79%（−1.58pp）**，Macro-F1 **31.56%→32.36%（+0.80pp）**，Macro-F1差值区间含0；纠错96/误伤133。未通过预先约定的替代标准，保留原E1。这是一个实际负向ACC消融，不能把“类别更均衡”写成无条件提升。[S39]

### 7.4 Thermal没有重复IR预训练收益，数值修复也存在性能代价

以下为user6/user7的377条Thermal可用样本；中断/停止运行记录**当时已观察最佳值**，不冒充完成所有计划epoch。[S40]–[S45]

| 方案或诊断 | 当时ACC | Macro-F1 | 状态与含义 |
|---|---:|---:|---|
| iFormer-T＋TSM，17轮后停止，选epoch16 | **29.44%** | 19.52%（首次归档） | logits极端尾部，loss剧烈波动；后续诊断重算Macro-F1约19.58% |
| official iFormer冻结骨干，只训头8轮 | **16.18%** | 7.12% | 激活异常消失，但表征线性可分性不足 |
| MobileNet＋TSM，14轮后停止，epoch12 | **28.65%** | **21.37%** | 比iFormer低0.80pp ACC，但最差用户与Macro-F1更好，无同样激活尾部 |
| 从零多流Thermal学生，50轮 | **27.32%** | 18.96% | 局部/全局多流未达到设定目标 |
| R(2+1)D-18 Kinetics教师C1 | **没有可用已发布验证ACC** | — | 旧不均衡运行作废；GitHub有均衡采样修复，未发布合法重跑最终分数 |

BN诊断的checkpoint ACC29.44%；替换为trial人口统计24.93%、frame人口统计21.22%、identity BN约6.10%。它们只是推理控制，不是可部署候选；降低logit尺度/NLL并没有自动提高ACC。冻结骨干对照支持“微调引发激活异常”，但也把分类能力降到16.18%，不能把稳定性修复等同性能收益。[S41][S42]

C1旧运行出现多数类collapse；已验证低显存分段反传与联合图数值一致，因此不能把失败怪到分段反传。均衡采样修正还需后续真实结果，本回顾不补训练。[S45]

## 8. 8月23–26日：VideoMAEv2视觉教师、路由修正与层级多模态

### 8.1 多视图VideoMAEv2与缓存路由，user6/user7可用IR385条

| 方案 | ACC | Macro-F1 | 状态/副作用 |
|---|---:|---:|---|
| 初始八流IR/Depth视觉教师 | **71.43%** | 63.09% | 8流oracle84.16%只表示互补上限，非部署分数 |
| P2-R0 logit-only缓存重排 | **70.65%** | 62.24% | 相对教师净损4条，窄变体失败 |
| P2-R0 feature-router | **69.35%** | 60.85% | 净损9条，不否定未实施完整P2-R0假设 |
| 更新视觉教师/full_hard2，主要腕部路由 | **72.21%** | 63.77% | 相对其uniform路由71.43%净纠错3条；不是纯人体/全局上下文教师 |
| 合法fixed-epoch CV选择margin_routes | **71.17%** | — | train-CV93.95%未转化为更高开发ACC；拒绝新路由 |
| wrist-person，仅最终预选候选person_fixed10 | **72.21%** | 63.67% | 3纠错/3误伤，净增0；拒绝缓存融合 |

初始教师的**模型推理消融**：去person_context ACC69.09%、去global67.27%；仅保留left_hand_object72.99%。这说明语义来源存在互补与干扰，不证明“只训腕部”会优于其它视野，也不能事后从385条选择最优mask当新正式模型。[S46]

路由CV使用已经在全部train12上学过的基础视觉特征，出现近饱和93%–94%并不等于基础教师对CV用户独立；即使user6/7没有进入训练，也不能把这个分数当全链跨用户泛化。[S47][S48][S49]

**无效结果也保留事实，但不列为成功：** 旧P3-R1用同用户fold选择epoch，其pooled score偏乐观；person辅助头曾未共享到fusion通路；另一次先把全部候选都在最终开发人口评价，违反当时selected-only契约。旧表出现的72.99%不能冒充已修正融合的成绩，修正后仅保留72.21%的selected-only结果。这些问题不等于证明读取过官方test。[S50][S51][S52]

### 8.2 层级多模态，完整388条（本地补充）

实现可追溯`33a6fcf`及远端当前单教师分支的祖先；下面的完成结果原在本地报告L01/L02，不伪装成原远端报告。

| 候选 | ACC | Macro-F1 | 最差用户ACC | 同组增益 |
|---|---:|---:|---:|---|
| visual_only | **70.62%（274/388）** | 61.18% | 63.55% | 基准 |
| visual_skeleton | **74.23%（288/388）** | 66.38% | 70.44% | +3.61pp；32纠错/18误伤，净14 |
| visual_skeleton_imu | **73.71%（286/388）** | **67.41%** | **70.94%** | 相对VS ACC−0.52pp，但Macro-F1＋1.03pp、最差用户＋0.49pp |

这是当前已确认固定user6/7的历史最优ACC。加入Skeleton在同一候选组中提供正向证据；继续加入IMU存在评价目标取舍，不能把它整体判为无信息。`visual_imu`当时没有单独运行，不能补造那条消融。模型组合/训练也有改变，因此仍是分支组合效应，不能全赋给某一种骨骼预处理。

与上一节385条的72.21%不同，本表包含3条缺IR样本、使用全388分母；不能直接把74.23−72.21当纯Skeleton净增益。

## 9. 8月27–9月8日：MotionBERT、动作属性和双视觉编码器

**MotionBERT Lite冻结骨骼专家**在388条上ACC **11.86%**、Macro-F1 **1.71%**，只预测9/40类；B1筛选失败后停止，未继续B2/B3。视觉＋MotionBERT真值oracle71.91%也不支持足够互补，不把“引入大预训练骨骼模型”写成自动收益。[S53]

**Motion-attribute分支**发布了运动属性缓存、训练实现和资源资格验证；本次GitHub没有找到该专家正式最终ACC，因此只记录工作，不能拿已存在的输入缓存当模型完成结果。

**Visual90（DINOv2外观＋VideoMAEv2时序）**的输入与资源验证已发布在当前实验分支祖先，原独立分支未发布；正式聚合结果来自本地L03。[S54][S55]

| 本地正式候选，388条开发集 | ACC | Macro-F1 |
|---|---:|---:|
| temporal_visual | **66.75%（259/388）** | 60.35% |
| appearance_temporal | **68.04%（264/388）** | 61.59% |

新增外观候选相对时序候选观察到+1.29pp，23纠错/18误伤，净5条；训练fit ACC两者都约99.90%，开发显著低于训练。原记录将appearance_increment标记为`mixed_or_unproven`，且未达到目标。没有新补充消融，不把这5条全部因果归给DINOv2，也不因模型更大就推断应该优于74.23%多模态路线。

## 10. 9月11–17日：推理检查与相同用户基础教师重建（本地补充）

先检查队友提交内容、依赖和推理兼容性，再转向训练方法对比。提交/推理检查没有形成新的本队训练ACC，且主要属于本地执行记录，本页不扩写成主要实验分支。

冻结VideoMAE-Large Kinetics公共权重、早晚两窗×scene/person/workspace六clip，再用train12内部三折72配置选择Ridge；**没有多教师投票、会话规则、test伪标签适配或学生蒸馏**。1957条实际训练输入，全部388条开发样本；385验证ROI、3缺IR仍以训练prior预测。报告与聚合结果见L04/L05。

| 指标 | 原visual_skeleton | 冻结基础视觉教师 | 差值 |
|---|---:|---:|---:|
| ACC | **74.23%（288/388）** | **67.53%（262/388）** | **−6.70pp** |
| Macro-F1 | 66.38% | 61.04% | −5.35pp |
| user6 ACC | 70.44% | 64.04% | 少13条 |
| user7 ACC | 78.38% | 71.35% | 少13条 |

train12内部OOF选参ACC **70.82%**不是388条最终开发成绩。新教师纠正旧基线25条、误伤51条；真值oracle80.67%不可部署。

**最重要的负向证据：只换队友这一冻结大视觉教师，没有补齐差距。** 原模型使用IR/Depth/Skeleton与任务微调，新模型是IR冻结骨干＋Ridge，不能据此证明VideoMAE-Large骨干本身弱于VideoMAEv2，也不能把队友整套系统的收益全归因网络大小。

## 11. 9月17–10月2日：源码归档、固定划分无投票复刻和清理

`teacher`分支发布队友源码快照；不把团队提供的官方提交约91%记作自己的可复现分支ACC。它与user6/7开发集没有同一测试人口，不能直接宣布约17pp就是某个后处理的贡献。

9月22–23日将方案改为一套从原始数据重新构建的单视觉教师＋Skeleton/IMU＋学生/融合/后处理流水线，明确排除历史30教师投票，固定train12→development2，再refit14→final4。10月2日完成协议/权重、清单/标签隔离、P28/P29和Task4接口审计；numeric IR/Skeleton映射、装饰器来源、缓存新增帧漏检等缺陷有回归修复。[S56]

**这一轮尚无正式ACC。** 已完成完整ROI和单样本GPU编码验收；10月3日整理时全量refit14特征已进入实际GPU编码，但不等于完成72候选分类头及学生训练。前置原始文件/祖先校验曾造成明显等待，属于工程副作用。划分隔离、缺类检查、时间戳恢复与缺失行保留是可信度/完整性修复；没有前后成对ACC，不能把它们各写成已测量涨分。没有新增机制消融结果，因此不能把工程完成或72个单元测试通过算成准确率收益。

用户已移除部分旧小视觉工作树，GitHub远端源码/报告仍可追溯。清理是资源管理，不改变已有科学结论；`outputs/`和checkpoint被Git忽略，只有源码提交不能保证恢复权重与本地最终结果。

## 12. 哪些改法有益，哪些代价已经被观察到？（Evidence Summary）

### 已有正向ACC证据（Positive ACC Evidence）

| 改法 | 已观察到的ACC变化 | 证据边界 |
|---|---:|---|
| 姿态ROI替代全局Depth | +6.76pp | 难动作148条同组，不覆盖全部40类 |
| 严格配对加入IR | +2.70pp | 公共148条，mask/shuffle支持输入依赖 |
| Target16层级条件专家 | 46.27%→48.14% | alpha在同验证集诊断选出，待独立确认 |
| 骨骼逐帧尺度C1 | 38.70%→40.92% | 严格OOF，有fold异质性 |
| 骨骼三seed固定平均 | 40.92%→42.38% | 推理成本约三倍 |
| 时间clip dropout | 60.75%→61.50% | 五用户800条，长视频存在损伤 |
| trial固定context | 52.21%→53.77% | user6/7，同视图族匹配 |
| 小Depth残差adapter | 53.77%→54.81% | 同user6/7，仍远低于目标 |
| 层级视觉＋Skeleton | 70.62%→74.23% | 本地388条候选组，组合增益 |
| RF150树＋leaf4 | 41.07%→41.88%三seed均值 | 减小模型且更高ACC；区别于无损序列化 |

这些数值不能相加，不能推出未来A9一定到91%。视频预训练整条路线的较大差距和Visual90的+1.29pp也已记录，但没有完整单变量证据。

### 已观察的负向ACC或性能取舍（Negative Effects and Trade-offs）

| 改法 | 已观察结果 | 不能误写的原因 |
|---|---|---|
| 用户SupCon | −3.90pp，置信错误减少 | 校准/置信改善不等于分类改善 |
| 极小Target16残差头 | 全40类净增0 | 参数更少不保证保留迁移容量 |
| ST-GCN轻量图模型 | −17.34pp | 当前配方欠拟合，不等于所有图模型无效 |
| Partial1 / 分层低LR | −4.94 / −3.09pp | 过度约束骨干可能损失有效表征 |
| Direct-Head | −1.56pp，但gap缩小 | 训练拟合减少不自动提高跨用户性能 |
| 全局Single13 | −1.04pp | 算力/效率收益有准确率成本 |
| 直接扩展四通道stem | 45.71%，相对固定context IR−8.05pp | 引入Depth的路径改变可能损伤既有预训练表示 |
| Skeleton均衡采样 | −1.58pp，Macro-F1略增 | 主ACC与少数类目标发生取舍 |
| 层级模型增加IMU | −0.52pp ACC，Macro-F1＋1.03pp | 不能只据主ACC说IMU全无价值 |
| person/context与更复杂路由 | 多次持平或下降 | 分支互补存在，简单叠加可能误伤 |
| 冻结Thermal骨干抑制异常 | 29.44%→16.18% | 数值稳定修复有表示/优化代价 |
| 冻结基础VideoMAE-Large | 67.53%，低于74.23% | 更大模型不是本项目提升的充分条件 |

## 13. GitHub分支覆盖与未发布结果（Remote Branch Coverage）

下面覆盖本次远端可见的所有主要分支。表中ACC是该分支**自身可追溯结果**或明确标明的继承/本地补充，不把相同报告在多个分支的继承算作重复实验。

| 首次阶段 | 远端分支 | 当时可核实ACC/状态 |
|---|---|---|
| 7月14–16日 | main | 六模态正式基线见第1节；这是项目初始记录，不是当前所有路线已合并进main |
| 7月16–27日 | IMU | 处理/TCN实现；28.80%是后续RF报告引用的参考，不假称该分支另有正式报告 |
| 7月29–31日 | IMU_rf | 三seed紧凑RF41.88%，正式seed42.41% |
| 7月31日 | depth-six-patch | 22.54%，590条 |
| 8月3日 | depth-pose-roi-expert | 30.41%→37.16%，148条 |
| 8月3日 | depth-ir-pose-roi-expert | 37.84%→40.54%，公共148条 |
| 8月4日 | depth-ir-pose-roi-40class | 45.08% best-accuracy / 44.41% best-macro，590条 |
| 8月4日 | depth-ir-object-interaction-tcn-expert | 46.27%，590条 |
| 8月5日 | depth-ir-object-interaction-resnet18-expert | 46.10%，590条 |
| 8月6日 | b2-256-cross-user-supcon | 42.37%，B2参考46.27% |
| 8月7日 | b2-256-cross-user-prototype-supcon | 实现已发布；没有该原型分支自身正式ACC |
| 8月7日 | b2-256-target16-hierarchical-e2 | 目标16类36.04%；接回全40类48.14%诊断 |
| 8月7日 | b2-256-target16-linear-residual-e2 | 目标16类32.88%；接回全40类46.27% |
| 8月8日 | b2-256-full-sequence-multiscale-tcn | 实现已发布；没有该全序列路线自身正式ACC |
| 8月8日 | depth-ir-scratch-dual-spatial-fullseq | 实现已发布；没有该scratch路线自身正式ACC |
| 8月10日 | ir-ordinal-fullseq-pipeline | 最好27.80%，590条；停点21.86% |
| 8月12–13日 | x3d-s-adaptive-multiclip | canonical OOF56.55%，三seed均值56.16% |
| 8月12–20日 | skeleton_raw_data | C0/C1至三seed与均衡采样的全部消融见第5–7节 |
| 8月16日 | test/x3d-fold0-generalization | 五用户A2/A3/A4-T60.75/61.00/61.50%；另一user6/7 Partial2/Direct53.25/51.69%，不能混成一条曲线 |
| 8月20日 | experiment/x3d-direct-head-generation-d | Direct51.69%；人口差距诊断，不是9.81pp纯头损失 |
| 8月20日 | experiment/x3d-single13-global-user6-user7 | 52.21%，385条 |
| 8月20日 | experiment/x3d-single13-fixed-context-user6-user7 | 53.77%，385条 |
| 8月20–21日 | experiment/x3d-single13-fixed-context-ir-depth4-user6-user7 | 本分支head只发布资源/四worker资格；后续ordinal-motion报告引用其失败raw四通道stem ACC45.71%，明确区别于绝对Depth adapter54.81% |
| 8月20–21日 | experiment/x3d-ir-anchored-adapter-user6-user7 | 绝对Depth残差anchor54.81%由后续阶段状态追溯；不把raw四通道stem与9参数adapter混同 |
| 8月21日 | experiment/x3d-ordinal-motion-adapter-user6-user7 | 54.81%，ACC对anchor持平、Macro-F1提高 |
| 8月20日 | experiment/thermal-native-expert | 官方权重/输入资格；没有该资格审计自己的训练ACC |
| 8月20日 | experiment/thermal-iformer-t-t1b | 中断iFormer29.44%；frozen16.18%；MobileNet28.65% |
| 8月21日 | experiment/thermal-route-a-workflow | 多流Thermal学生27.32% |
| 8月21日 | experiment/thermal-route-c-teacher | 不均衡旧运行无效；没有合法重跑最终ACC |
| 8月23–25日 | experiment/videomae-wrist-person-residual | 教师71.43/72.21%、重排70.65/69.35%、合法路由71.17%、最终融合72.21% |
| 9月17日 | teacher | 队友源码快照；不构成新的本队独立ACC报告 |
| 9月22–10月2日 | experiment/teammate-single-teacher-task1 | 固定划分复刻方案/Task1–4实现与审计；尚无正式新ACC |

`design/hierarchical-multimodal-midfusion`、MotionBERT、motion-attribute和Visual90原独立本地分支名不在当前远端heads清单中，但其实现/部分报告可通过已发布的单教师分支祖先访问；它们不应因此从重要工作历史中完全消失。最终未发布结果已按本地补充单列；其余仅本地小试验不扩展。

## 14. 解释ACC时保留的限制（Interpretation Limits）

本次按统计解释清单检查11项：用户汇总与单用户反转、群体均值外推个体、筛样本偏差、条件选择混杂、类别基率、极值回归、只保留完成者、最佳结果挑选、分析路径自由度、相关当因果、反向解释。主要实际风险是**混划分、缺模态删分母、反复开发选最优、恢复偏差、少数用户贡献集中、没有配对实验却硬归因**。

既有paired bootstrap/McNemar只引用原报告，不新增统计检验。样本级置信区间不自动等于按用户簇重新抽样的跨新用户不确定性；部分结果比较次数较多，不能把未经全局多重比较校正的历史p值包装成一项统一确认性研究。

ACC低能说明某个已执行配方在该人口表现弱；训练ACC高而验证低支持泛化风险；主ACC不变但Macro-F1/误伤结构改变支持目标取舍。它们**不能自动定位唯一原因**。因此这份历史既保留失败分支的实际分数，也保留尚未被消融隔离的疑问，不为了补叙事而补实验。

## 来源索引（Sources）

以下GitHub链接固定到已核对的提交SHA，减少分支移动造成的引用漂移；远端分支被删除并清理孤立对象后仍可能失效，因此本页同时归档关键聚合数值，建议保留远端实验分支。项目原始仓库是`D:/work/2026.7.14_kaggle/40class`；本文件在独立main工作树`_project_history_main/docs`编写，只提交本页，不提交旧分支未提交工作。

- [S01: 7月15日六模态两轮结果](https://github.com/laiwanzhou/40class/blob/64fb41728db500979c367be959e5e28482ef8c05/reports/task03_two_epoch_run_report.md)
- [S02: 输入流水线优化结果](https://github.com/laiwanzhou/40class/blob/64fb41728db500979c367be959e5e28482ef8c05/reports/task03_optimized_two_epoch_report.md)
- [S03: 7月16日六模态正式基线](https://github.com/laiwanzhou/40class/blob/64fb41728db500979c367be959e5e28482ef8c05/reports/task03_baseline_fold0_report.md)
- [S04: IMU紧凑随机森林与消融](https://github.com/laiwanzhou/40class/blob/91ba87a8d9025cad5bb9f42accb31596e06994e1/docs/superpowers/reports/2026-07-30-imu-compact-random-forest-results.md)
- [S05: IMU部署包与复现结果](https://github.com/laiwanzhou/40class/blob/91ba87a8d9025cad5bb9f42accb31596e06994e1/docs/superpowers/reports/2026-07-30-imu-rf-finalization-results.md)
- [S06: IMU正式验证表现](https://github.com/laiwanzhou/40class/blob/91ba87a8d9025cad5bb9f42accb31596e06994e1/docs/superpowers/reports/2026-07-31-imu-rf-action-performance.md)
- [S07: Depth六块视野结果](https://github.com/laiwanzhou/40class/blob/6f22cfe077ab4ef68b34df5f1d0c6760cc41c9bd/reports/depth_six_patch_fold0_14train_4val.md)
- [S08: Depth全局与姿态ROI比较](https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/depth_pose_roi_experiment.md)
- [S09: Depth/IR配对和mask/shuffle消融](https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/depth_ir_pose_roi_experiment.md)
- [S10: 40类视觉停止与检查点取舍](https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/depth_ir_pose_roi_40class_early_stop_analysis.md)
- [S11: 物体交互TCN专家](https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/object_interaction_tcn_expert_experiment.md)
- [S12: ResNet18专家比较](https://github.com/laiwanzhou/40class/blob/d5ffac835912344827e21117ac48fe73bcf77c2c/reports/object_interaction_resnet18_expert_experiment.md)
- [S13: 用户对比学习诊断](https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/b2_256_cross_user_supcon_epoch18_diagnosis.md)
- [S14: Target16层级专家](https://github.com/laiwanzhou/40class/blob/362f9c6f14d5cf1a070e2c639c08ab5b5eaca57c/reports/target16_hierarchical_e2_experiment.md)
- [S15: Target16线性残差](https://github.com/laiwanzhou/40class/blob/53c8e9aa452f3b0df910c45c164bf837fb397ce1/reports/target16_linear_residual_e2_experiment.md)
- [S16: Depth表示先导比较](https://github.com/laiwanzhou/40class/blob/c8d656f5d2c27f5c823d77c828fbfba77ef9899e/reports/depth_ordinal_stage8_pilot_comparison.md)
- [S17: 序数Depth正式训练停止分析](https://github.com/laiwanzhou/40class/blob/c8d656f5d2c27f5c823d77c828fbfba77ef9899e/reports/depth_ordinal_stage9_early_stop_diagnosis.md)
- [S18: X3D严格OOF与匹配sanity](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_ir_context_oof_report.md)
- [S19: X3D训练与OOF泛化差距](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_ir_train_vs_oof_generalization.md)
- [S20: 骨骼C0/C1预处理消融](https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_c0_c1_strict_oof/skeleton_c0_c1_strict_oof_report.md)
- [S21: 轻量ST-GCN](https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_lightweight_stgcn_strict_oof/skeleton_lightweight_stgcn_strict_oof_report.md)
- [S22: Joint/Bone双表示](https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_joint_bone_tcn_strict_oof/skeleton_joint_bone_tcn_strict_oof_report.md)
- [S23: 骨骼分段感知TCN](https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_segment_aware_tcn_strict_oof/skeleton_segment_aware_tcn_strict_oof_report.md)
- [S24: 骨骼T96时间分辨率](https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_t96_tcn_strict_oof/skeleton_t96_tcn_strict_oof_report.md)
- [S25: user21/user22 Partial2](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_partial2_report.md)
- [S26: Partial1容量消融](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_partial1_report.md)
- [S27: 分层学习率消融](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_layerwise_lr1_report.md)
- [S28: X3D开发A2](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_fold0_a2_report.md)
- [S29: X3D开发A3](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_fold0_a3_report.md)
- [S30: 时间clip dropout A4-T](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_fold0_a4_t_report.md)
- [S31: user6/user7 Partial2](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_partial2_report.md)
- [S32: 直接分类头消融](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_direct_head1_report.md)
- [S33: 骨骼三seed集成](https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_c1_seed_ensemble_strict_oof/skeleton_c1_seed_ensemble_strict_oof_report.md)
- [S34: 跨用户群准确率差距复核](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_direct_head_vs_a4_t_diagnosis.md)
- [S35: Single13全局时间窗](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_single13_global_report.md)
- [S36: 固定人物context](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_report.md)
- [S37: X3D各阶段与Depth残差结果](https://github.com/laiwanzhou/40class/blob/ffee876f260dcb51e61e330cdb528f8aa5ac0043/reports/x3d_s_phase_status.md)
- [S38: 序数运动Depth消融](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_ordinal_motion_adapter_report.md)
- [S39: 骨骼均衡采样消融](https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_c1_balanced_seed_ensemble_strict_oof/skeleton_c1_balanced_seed_ensemble_strict_oof_report.md)
- [S40: Thermal iFormer中断结果](https://github.com/laiwanzhou/40class/blob/c3be5496d1d13a8d98289ffefbb4dc74c1572b18/reports/thermal_iformer_t_tsm_train12_val2_interrupted_audit.md)
- [S41: Thermal BN诊断](https://github.com/laiwanzhou/40class/blob/c3be5496d1d13a8d98289ffefbb4dc74c1572b18/reports/thermal_t1b1_bn_diagnostic.md)
- [S42: Thermal冻结骨干对照](https://github.com/laiwanzhou/40class/blob/c3be5496d1d13a8d98289ffefbb4dc74c1572b18/reports/thermal_t1b4_head_only_probe.md)
- [S43: Thermal MobileNet停止结果](https://github.com/laiwanzhou/40class/blob/c3be5496d1d13a8d98289ffefbb4dc74c1572b18/reports/thermal_mobilenetv3_tsm_epoch14_stopped_audit.md)
- [S44: Thermal多流学生结果](https://github.com/laiwanzhou/40class/blob/50e7d711b268fd8fb4cf4fd156215926f5ea77b8/reports/thermal_a_multistream_direct_train12_val2.md)
- [S45: Thermal C1无效运行与采样修正](https://github.com/laiwanzhou/40class/blob/8aa30884b001eda3221bcf054cae2ee97f9eba11/reports/thermal_c1_invalid_imbalanced_run.md)
- [S46: VideoMAEv2视图消融](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_p2a_diagnostics.md)
- [S47: P2-R0窄变体重排器](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_p2r0_result.md)
- [S48: 修正后的P3-R1路由](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_p3r1_fixed_epoch_result.md)
- [S49: 修正后的wrist-person融合](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_wrist_person_residual_selected_only_result.md)
- [S50: 无效P3-R1结果隔离](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_p3r1_label_tuned_invalid.md)
- [S51: 全候选验证无效归档](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_wrist_person_residual_all_candidates_validation_invalid.md)
- [S52: 辅助头未连通的无效归档](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_wrist_person_residual_disconnected_aux_invalid.md)
- [S53: MotionBERT停止报告](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/motionbert_lite_skeleton_expert_p6b.md)
- [S54: Visual90输入规范](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/visual90_appearance_temporal_input_preflight_v3.md)
- [S55: Visual90资源及smoke报告](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/visual90_appearance_temporal_resource_smoke.md)
- [S56: 固定划分单教师新流程审计](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/2026-10-02-task23-task4-independent-audit.md)
- [S57: Depth失败四通道stem与残差anchor的聚合数值](https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_ordinal_motion_adapter_report.json)

### 本地补充来源指纹（Local Supplement Fingerprints）

- **L01**：`40class-x3d-adaptive-multiclip/reports/hierarchical_multimodal_teacher_fixed_user6_user7.md`；SHA256 `06301c037284be8abd39b1aeec272b23014fde9be18f3d266f499f75d4ff1e08`。
- **L02**：`40class-x3d-adaptive-multiclip/reports/hierarchical_multimodal_teacher_fixed_user6_user7.json`；SHA256 `3c04c50ecbec1bb7ae3a673636914e7db8fbbf0691e59b7fbb78251cd947f7c7`。
- **L03**：`40class-x3d-adaptive-multiclip/outputs/visual90_appearance_temporal/formal_v1/result.json`；SHA256 `73531819169deeb747a32368f384fae7e2696c28686bece7c97af367f197a640`。
- **L04**：`teacher_comparison_20260913/REPORT.md`；SHA256 `bdf5fc37465414ee8f8ae54b8efb9799187dfba6553cab5cc4e108a235e9d081`。
- **L05**：`teacher_comparison_20260913/result.json`；SHA256 `776d56d75dd12de75161e7c15845a14e1e1d1a4f10ec8bb2d1e930920226c296`。

下面定义正文的引用链接（Reference links）：

[S01]: https://github.com/laiwanzhou/40class/blob/64fb41728db500979c367be959e5e28482ef8c05/reports/task03_two_epoch_run_report.md "7月15日六模态两轮结果"
[S02]: https://github.com/laiwanzhou/40class/blob/64fb41728db500979c367be959e5e28482ef8c05/reports/task03_optimized_two_epoch_report.md "输入流水线优化结果"
[S03]: https://github.com/laiwanzhou/40class/blob/64fb41728db500979c367be959e5e28482ef8c05/reports/task03_baseline_fold0_report.md "7月16日六模态正式基线"
[S04]: https://github.com/laiwanzhou/40class/blob/91ba87a8d9025cad5bb9f42accb31596e06994e1/docs/superpowers/reports/2026-07-30-imu-compact-random-forest-results.md "IMU紧凑随机森林与消融"
[S05]: https://github.com/laiwanzhou/40class/blob/91ba87a8d9025cad5bb9f42accb31596e06994e1/docs/superpowers/reports/2026-07-30-imu-rf-finalization-results.md "IMU部署包与复现结果"
[S06]: https://github.com/laiwanzhou/40class/blob/91ba87a8d9025cad5bb9f42accb31596e06994e1/docs/superpowers/reports/2026-07-31-imu-rf-action-performance.md "IMU正式验证表现"
[S07]: https://github.com/laiwanzhou/40class/blob/6f22cfe077ab4ef68b34df5f1d0c6760cc41c9bd/reports/depth_six_patch_fold0_14train_4val.md "Depth六块视野结果"
[S08]: https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/depth_pose_roi_experiment.md "Depth全局与姿态ROI比较"
[S09]: https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/depth_ir_pose_roi_experiment.md "Depth/IR配对和mask/shuffle消融"
[S10]: https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/depth_ir_pose_roi_40class_early_stop_analysis.md "40类视觉停止与检查点取舍"
[S11]: https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/object_interaction_tcn_expert_experiment.md "物体交互TCN专家"
[S12]: https://github.com/laiwanzhou/40class/blob/d5ffac835912344827e21117ac48fe73bcf77c2c/reports/object_interaction_resnet18_expert_experiment.md "ResNet18专家比较"
[S13]: https://github.com/laiwanzhou/40class/blob/04d18ee775d7ffafeeac6da6bde83c5a8d7f16c2/reports/b2_256_cross_user_supcon_epoch18_diagnosis.md "用户对比学习诊断"
[S14]: https://github.com/laiwanzhou/40class/blob/362f9c6f14d5cf1a070e2c639c08ab5b5eaca57c/reports/target16_hierarchical_e2_experiment.md "Target16层级专家"
[S15]: https://github.com/laiwanzhou/40class/blob/53c8e9aa452f3b0df910c45c164bf837fb397ce1/reports/target16_linear_residual_e2_experiment.md "Target16线性残差"
[S16]: https://github.com/laiwanzhou/40class/blob/c8d656f5d2c27f5c823d77c828fbfba77ef9899e/reports/depth_ordinal_stage8_pilot_comparison.md "Depth表示先导比较"
[S17]: https://github.com/laiwanzhou/40class/blob/c8d656f5d2c27f5c823d77c828fbfba77ef9899e/reports/depth_ordinal_stage9_early_stop_diagnosis.md "序数Depth正式训练停止分析"
[S18]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_ir_context_oof_report.md "X3D严格OOF与匹配sanity"
[S19]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_ir_train_vs_oof_generalization.md "X3D训练与OOF泛化差距"
[S20]: https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_c0_c1_strict_oof/skeleton_c0_c1_strict_oof_report.md "骨骼C0/C1预处理消融"
[S21]: https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_lightweight_stgcn_strict_oof/skeleton_lightweight_stgcn_strict_oof_report.md "轻量ST-GCN"
[S22]: https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_joint_bone_tcn_strict_oof/skeleton_joint_bone_tcn_strict_oof_report.md "Joint/Bone双表示"
[S23]: https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_segment_aware_tcn_strict_oof/skeleton_segment_aware_tcn_strict_oof_report.md "骨骼分段感知TCN"
[S24]: https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_t96_tcn_strict_oof/skeleton_t96_tcn_strict_oof_report.md "骨骼T96时间分辨率"
[S25]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_partial2_report.md "user21/user22 Partial2"
[S26]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_partial1_report.md "Partial1容量消融"
[S27]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_layerwise_lr1_report.md "分层学习率消融"
[S28]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_fold0_a2_report.md "X3D开发A2"
[S29]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_fold0_a3_report.md "X3D开发A3"
[S30]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_fold0_a4_t_report.md "时间clip dropout A4-T"
[S31]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_partial2_report.md "user6/user7 Partial2"
[S32]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_direct_head1_report.md "直接分类头消融"
[S33]: https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_c1_seed_ensemble_strict_oof/skeleton_c1_seed_ensemble_strict_oof_report.md "骨骼三seed集成"
[S34]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_direct_head_vs_a4_t_diagnosis.md "跨用户群准确率差距复核"
[S35]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_single13_global_report.md "Single13全局时间窗"
[S36]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_report.md "固定人物context"
[S37]: https://github.com/laiwanzhou/40class/blob/ffee876f260dcb51e61e330cdb528f8aa5ac0043/reports/x3d_s_phase_status.md "X3D各阶段与Depth残差结果"
[S38]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_ordinal_motion_adapter_report.md "序数运动Depth消融"
[S39]: https://github.com/laiwanzhou/40class/blob/bdd5d7313320336b5f2e6eae34df4e7adf2918d3/reports/skeleton_c1_balanced_seed_ensemble_strict_oof/skeleton_c1_balanced_seed_ensemble_strict_oof_report.md "骨骼均衡采样消融"
[S40]: https://github.com/laiwanzhou/40class/blob/c3be5496d1d13a8d98289ffefbb4dc74c1572b18/reports/thermal_iformer_t_tsm_train12_val2_interrupted_audit.md "Thermal iFormer中断结果"
[S41]: https://github.com/laiwanzhou/40class/blob/c3be5496d1d13a8d98289ffefbb4dc74c1572b18/reports/thermal_t1b1_bn_diagnostic.md "Thermal BN诊断"
[S42]: https://github.com/laiwanzhou/40class/blob/c3be5496d1d13a8d98289ffefbb4dc74c1572b18/reports/thermal_t1b4_head_only_probe.md "Thermal冻结骨干对照"
[S43]: https://github.com/laiwanzhou/40class/blob/c3be5496d1d13a8d98289ffefbb4dc74c1572b18/reports/thermal_mobilenetv3_tsm_epoch14_stopped_audit.md "Thermal MobileNet停止结果"
[S44]: https://github.com/laiwanzhou/40class/blob/50e7d711b268fd8fb4cf4fd156215926f5ea77b8/reports/thermal_a_multistream_direct_train12_val2.md "Thermal多流学生结果"
[S45]: https://github.com/laiwanzhou/40class/blob/8aa30884b001eda3221bcf054cae2ee97f9eba11/reports/thermal_c1_invalid_imbalanced_run.md "Thermal C1无效运行与采样修正"
[S46]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_p2a_diagnostics.md "VideoMAEv2视图消融"
[S47]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_p2r0_result.md "P2-R0窄变体重排器"
[S48]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_p3r1_fixed_epoch_result.md "修正后的P3-R1路由"
[S49]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_wrist_person_residual_selected_only_result.md "修正后的wrist-person融合"
[S50]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_p3r1_label_tuned_invalid.md "无效P3-R1结果隔离"
[S51]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_wrist_person_residual_all_candidates_validation_invalid.md "全候选验证无效归档"
[S52]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/ir_depth_videomaev2_wrist_person_residual_disconnected_aux_invalid.md "辅助头未连通的无效归档"
[S53]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/motionbert_lite_skeleton_expert_p6b.md "MotionBERT停止报告"
[S54]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/visual90_appearance_temporal_input_preflight_v3.md "Visual90输入规范"
[S55]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/visual90_appearance_temporal_resource_smoke.md "Visual90资源及smoke报告"
[S56]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/2026-10-02-task23-task4-independent-audit.md "固定划分单教师新流程审计"
[S57]: https://github.com/laiwanzhou/40class/blob/b8587bed1493c60178ce47a81ca49882ab35ddff/reports/x3d_s_train12_val2_user6_user7_single13_fixed_context_ordinal_motion_adapter_report.json "Depth失败四通道stem与残差anchor的聚合数值"
