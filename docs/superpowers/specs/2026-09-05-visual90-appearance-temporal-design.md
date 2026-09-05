# Visual90：物品外观与局部动作过程教师实验

## 状态与研究问题

- 日期：2026-09-05。
- 分支：`experiment/visual90-appearance-temporal`，起点 `e9fd351`。
- 用户已批准对话中的实验方向及新分支执行；本书面协议等待审阅。
- 第一阶段只比较 A/B 两个冻结骨干、可学习中期融合候选，不启动 Skeleton、IMU、Thermal、Radar 或学生蒸馏。
- 研究问题：在相同四视图、局部片段和训练协议下，独立预训练的物品外观特征，是否改善跨用户视觉动作分类？
- 假设：保留局部空间信息、视图身份、时间顺序，再联合外观与运动证据，比继续修改已有 logits 的门控更值得投入。假设尚未验证，不承诺达到 0.90。

## 1. 已有证据及解释边界

当前固定验证人口为 388。旧视觉模型正确 274，视觉加骨骼正确 288；至少正确 350 才达到 0.90。旧视觉 Top-3 正确 346，Top-5 正确 354。因此固定 Top-3 重排不足以达到目标，固定 Top-5 重排需要极高的纠错成功率。

证据工件：

- `reports/hierarchical_multimodal_teacher_fixed_user6_user7.md`
- `outputs/hierarchical_multimodal_midfusion_stage1/fixed_user6_user7/visual_only/validation_predictions.npz`
- `outputs/hierarchical_multimodal_midfusion_stage1/fixed_user6_user7/visual_skeleton/validation_predictions.npz`
- `reports/ir_depth_videomaev2_p3r1_fixed_epoch_result.md`
- `reports/motion_attribute_expert_result.md`

历史结果是参照，不是与新实验完全匹配的因果对照。A/B 会同时改变旧输入及融合结构，A 相对旧模型的差异不能单独归因于帧数、裁剪、对比损失或骨干冻结。B 相对 A 检验的是新增外观分支的整体效用，不是控制参数量后的纯预训练效应。

Motion Attribute 的原失败门禁和旧报告保持不变。冻结特征筛选失败不证明后续骨干适配必然失败；本阶段也不自动启动适配。

## 2. 人口、划分与数据隔离

- 唯一划分：`metadata/splits/train12_val2_user6_user7_development.json`。
- 训练用户：user1、user2、user3、user5、user8、user9、user16、user18、user19、user20、user21、user22。
- 验证用户：user6、user7。规范人口固定 train=2039、validation=388，类别顺序 0..39。
- 入库前核对 sample_id 唯一性、训练验证 ID/用户互斥、两侧 40 类支持、与规范 manifest 的逐行成员一致性。
- 不使用三折、OOF、多种子、heldout4 或比赛 test。
- 验证可用于本轮已声明的 A/B 开发比较，但不得进入梯度、归一化统计、采样器、对比队列、伪标签、自监督适配或错误驱动的逐样本规则。
- user_id 只用于训练配对和评估分组，不能成为模型输入。路径、类别目录名、标签编码也不能进入特征。
- 验证属于反复使用的开发集，不能称作独立最终泛化测试。
- 无可用视觉的样本不从规范指标中删除：使用训练集类别先验输出，同时单列 supported 指标和支持率。训练损失不使用无视觉样本；先验仅由规范训练标签拟合。
- Watch_TV 等覆盖不足按既有决定记录并搁置，不改划分或从验证指标排除。

## 3. 固定输入协议

### 3.1 时间证据

每条 trial 按原始帧序划分四个等长区间，区间内均匀选择 16 个位置，得到固定 `[4,16]` 索引。短区间允许重复帧；记录重复率和有效 mask。该方案是局部片段采样，不声称 16 帧必然在原始帧序上相邻。禁止运动峰值选择和依据预测错误挑帧。

在每个片段的第 0、5、10、15 个位置提取 DINO 外观特征，总计 16 个时间位置。它们严格是 A 已观察帧的子集，不给 B 额外原始帧。两候选共享裁剪、顺序和真实可用性。

记录原始帧键及区间边界；可信时间戳存在时保存真实时间，否则明确使用归一化帧序，不能伪装成秒。遇到不连续源片段不跨 gap 合成运动，使用 mask 标识并审计。

### 3.2 四视图与局部裁剪

保留 global、person、left_hand_object、right_hand_object 四个身份，不能在跨视图交互之前合并成 context/wrist 两组。

使用现有 YOLO pose 缓存和 `ObjectInteractionROIBuilder` 的原始方向框作为几何来源；实施前核对缓存坐标系、原始尺寸、帧对应和来源哈希。禁止把 person-crop 坐标直接当作全图坐标。使用 `configs/experiments/ir_depth_videomaev2_vit_b_p0.yaml` 中已记录的 ROI 参数，不利用验证标签调 ROI。

person 和腕部框采用片段内平滑中心、局部稳定尺度，平滑必须保留有效检测框覆盖范围。禁止恢复为整条 trial 的单一腕部框。保存每帧框中心、宽高及源图归一化坐标；保留位置变化，避免跟随裁剪消除运动线索。

短缺失可在同一有效段内插值，最大连续缺失为 3 个源帧；长缺失置不可用。只有没有有效人物检测时允许按既有授权退化为 global-only；存在人物但腕部不可用时保留人物和 global，不虚构腕部特征。任何其他读取异常不得被吞掉并伪装成无 pose。

IR 重复灰度三通道；Depth 保留已有 depth_color 表达，不能当作已校准公制深度。两者使用已验证对应关系和相同裁剪。若只能证明时间同步，不能声称像素几何配准；像素级残差必须另有配准证据，否则采用同时间同视图 token 交互。

两分支输入分辨率首轮固定 224，不把插值放大当作新增真实细节。首轮不另加第五/六视图；手-头、双手关系通过已有视图和位置 token 表达。

## 4. 骨干与缓存

### 4.1 VideoMAE

复用官方兼容 `VideoMAEV2ViTBase` 及严格加载器，从同一 K710 蒸馏预训练开始，而不是加载旧验证选中的动作分类头：

- 权重：`C:/Users/LaiWanzhou/.cache/torch/hub/checkpoints/vit_b_k710_dl_from_giant.pth`。
- SHA256：`8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec`。
- 字节数：173574417。
- 冻结全部骨干，eval/inference 模式，去掉原分类头。
- 每个片段/视图保留 8 个 tubelet 时间位置，空间网格池化到 2x2，而非全空间平均；缓存维度 `[2 modalities,4 clips,4 views,8 times,4 cells,768]`。

### 4.2 DINOv2

首轮只使用官方 `dinov2_vitl14`（无 registers），冻结、eval 模式。官方模型卡给出 ViT-L 为 1024 维、patch14，224 输入产生 16x16 patch 网格；采用固定 4x4 空间池化保留 16 个区域，并保留独立 CLS token。

缓存 IR 外观维度 `[4 clips,4 views,4 times,17 tokens,1024]`。Depth 不另跑 DINO。

下载前读取官方实现与许可证、锁定不可变源码 revision；下载后计算 checkpoint SHA256/字节数并写 provenance lock，再允许正式提取。不得伪造尚未下载权重的哈希，不从可变 main 分支静默执行代码。严格核对 key/shape 覆盖率，只有显式不用的分类头可例外。

这是 RGB 预训练向 IR 迁移的待验证假设。不得引用 ImageNet 成绩推导本任务准确率。

### 4.3 可靠性与成本

以 FP16、分片或 mmap 形式保存，不把整个大缓存压缩后一次性装入内存。每条记录包含 sample_id、split、标签、帧键、视图可用性、ROI/时间坐标、源清单和来源哈希。浮点运算归一化使用 float32。

缓存身份绑定 manifest、split、ROI 配置与缓存、源码版本、两骨干权重及变换协议。完成标记最后原子写入；断点续提取只接受全套哈希一致且已校验完成的分片。最终验证人口必须完整，不能带缺失分片评估。

先逐个骨干加载并释放，避免 8GB GPU 同时驻留。先用仅训练样本的 8 条固定冒烟集合测量冷/热缓存读取、CUDA 同步计时、峰值显存和单样本字节数；按训练源帧长度四分位每组两个 sample_id 排序最小样本确定，不看结果选样。

资源硬门：峰值 allocated <7300 MiB，估计磁盘用量不超过所在盘可用空间的 70%，预计单次全量特征提取不超过 12 小时。越门只报告并请求调整，不自动更换小骨干或降输入。12 小时是预算上限，不是运行时间估计。

## 5. A/B 深融合接口

两个候选共享一套 256 维、4 attention heads、2 层的轻量分层融合：先片段内部处理区域和视图关系，再跨四片段处理顺序。必须显式加入视图身份、时间顺序、空间单元及 ROI 坐标，并应用逐 token mask。

- A `temporal_visual`：IR VideoMAE 为主，同时间同视图 Depth token 经零初始化的有界残差适配器进入 IR 特征。无 Depth 时严格恢复 IR 路径。保留四视图，不使用静态等权 logits 平均。
- B `appearance_temporal`：与 A 完全相同的基础路径；视觉 query 在分类前对 DINO 区域 token 做 cross-attention，残差进入时空视觉特征。保留 A 的直接视觉路径，不采用最终分类概率平均或 Top-k 硬重排。
- 输出统一 40 类 logits、归一化 trial embedding、可审计分支有效性与残差幅度。attention 权重不是模态贡献的因果证明。
- 初始新增外观残差为零，验证初始 A/B 基础输出一致；也验证经过优化后外观适配路径确实有梯度并可改变输出。零初始化不能保证训练后无 harm。
- 测试所有视图缺失、单侧手缺失、Depth 缺失、异常 NaN、时间/视图置换以及 batch=1。

## 6. 首轮训练协议

- 只训练新投影、适配、融合与分类模块；两大骨干不进入 optimizer。
- 固定单 seed=20260715、30 epochs、batch=32，最后不足 batch 保留；不依据验证选择 epoch。
- AdamW，lr=0.0003、weight_decay=0.05，前 2 epochs 线性 warmup 后 cosine，gradient_clip=5.0。
- 损失：40 类 CE（label_smoothing=0.05）加 0.10 倍跨用户监督对比项，temperature=0.10；对比对象为融合后的 256 维归一化 trial embedding。
- 每轮预生成共同样本序列：均匀选类别，再选两个不同的可用训练用户，各取一个 trial，组成 pair；16 pair 为一 batch。只有一个用户的类别取两个不同 trial，若无第二 trial 则重复并标记无跨用户正例。每轮 pair 数为 ceil(视觉支持训练行数/2)，使用有放回抽样。
- 对比正例必须同类、不同用户、不同 trial；无合格正例的 anchor 跳过对比损失但保留 CE；批次中相同 sample_id 不互作对比项。报告实际正例覆盖和每类/用户曝光。
- A/B 共用完全相同的样本序列、优化步数、共同模块初始化和输入；新增模块的 RNG 独立，不能改变共同模块 dropout 随机序列。
- 缓存阶段不声称做了在线像素增强。训练 token dropout=0.10，仅作用于有效 token 且每条有视觉 trial 至少保留一个真实 token；验证关闭。更复杂增强和骨干适配不在首轮自动执行范围。
- 每 epoch 存 latest checkpoint，包含模型、optimizer、scheduler、随机状态、sampler/epoch、配置和缓存来源哈希。中途续训必须还原后续样本序列；正式完成目录不得被静默覆盖。
- loss/logits/gradients 任一非有限即停止并记录；不静默跳过坏 batch。两候选全部完成后统一读取验证结果，不根据 A 的验证结果改 B。

## 7. 评估与投资决策

固定 epoch30，eval 模式无增强，对规范 train2039 和 val388 逐条计算 train_acc、val_acc、Macro-F1（全部40类）、worst-user、每用户准确率、NLL、Top3/Top5、零召回类别和支持率。另报视觉支持子集，不能替代规范指标。

逐 sample_id 对齐 A/B 及旧基线，报告 rescue、harm、net、每类变化和用户分解。打印计数及分母。吃食物、喝水、吃药、使用餐具、手机等类别单列，仅用于解释，不据验证样本拟合类别例外规则。

建议性投资规则，非统计显著性检验：B 相比 A 净增至少 8/388、worst-user 不下降，且 B 达到至少 319/388（约0.82），才标记为明确值得讨论后续适配。其余结果标记混合/未达到阶段目标，不宣布 IR 或 DINO 的能力上限。该规则及0.85里程碑均不是预期收益承诺。

0.90 目标对应至少350/388；两用户、单seed结果不能支持稳定跨用户泛化声明。任何后续 adapter 微调、增加 Skeleton、额外消融或蒸馏，都需要先汇报本轮结果并确定新阶段协议。

## 8. 实施与验收顺序

1. 审阅本 spec 后，写独立实施 plan，沿用 Inline Execution，不默认派子代理。
2. TDD 实现人口/采样/ROI 深接口，校验几何来源与无泄露边界；不改旧实验接口。
3. 锁定两骨干来源，实现特征提取和可恢复缓存；用训练样本冒烟、实测预算。
4. TDD 实现 A/B 融合、跨用户配对损失、恢复训练与配对报告。
5. 一组任务完成后统一审计，再启动全量缓存和 A/B 固定训练；后台进程隐藏窗口、独立日志和状态文件，不使用替代计划的额外实验。
6. 完成后检查来源哈希、预测人口、复算指标、全量回归和 git diff 检查，再提交本实验文件并更新临时 handoff。新分支不默认推送远程。

## 9. 代码边界与交付位置

新增模块集中在 `src/experiments/visual90_config.py`、`src/data/visual90_dataset.py`、`src/models/visual90_encoders.py`、`src/models/visual90_fusion.py`、`src/training/visual90_training.py`。

新增入口：`scripts/cache_visual90_features.py`、`scripts/run_visual90_experiment.py`；配置：`configs/experiments/visual90_appearance_temporal.yaml`；测试：对应 `tests/test_visual90_*.py`。

新输出仅写 `outputs/visual90_appearance_temporal/`，新报告仅写 `reports/visual90_appearance_temporal*`。不覆盖历史 checkpoints、缓存、指标或 dirty Motion Attribute 文件。学生从头训练和模型包大小限制留到后续蒸馏阶段，本轮教师骨干不作为学生包交付。

## 参考与材料护照

- 来源技能：academic-research-suite / experiment-agent，plan 模式；Superpowers 设计流程。
- 验证状态：历史计数已从本地档案复算；新干预 UNVERIFIED；书面设计 v1 待审阅。
- DINOv2 官方模型卡：https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- DINOv2 论文：https://arxiv.org/abs/2304.07193
- Supervised Contrastive Learning：https://arxiv.org/abs/2004.11362
- AdaptFormer（后续阶段参考，不是首轮已实现内容）：https://arxiv.org/abs/2205.13535
