# Visual90：物品外观与局部动作过程教师实验

## 状态与研究问题

- 创建日期：2026-09-05；修订日期：2026-09-06，v3（保留v2五项审计修订，补充赛事缺模态规则和缺时间戳训练排除授权）。
- 分支：`experiment/visual90-appearance-temporal`，起点 `e9fd351`。
- 用户已批准对话中的实验方向、新分支执行及五项审计修订；本版本为修订后的实施依据，不表示代码、几何核验或资源冒烟已经完成。
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
- 赛事背景（用户2026-09-06补充）：实际test中某些动作天然缺模态；train/val预期模态齐全。因此mask/fallback是推理能力要求，不是把开发数据缺路径自动判作正常的理由。本阶段不访问test，不根据动作标签推定或伪造可用模态。实际train/val缺目录、缺文件或仅有Thermal的孤立trial，单独列为数据/对应关系异常；不据此擅自合并trial、删除验证行或声称赛事允许开发集缺模态。
- 入库前核对 sample_id 唯一性、训练验证 ID/用户互斥、两侧 40 类支持、与规范 manifest 的逐行成员一致性。
- 不使用三折、OOF、多种子、heldout4 或比赛 test。
- 验证可用于本轮已声明的 A/B 开发比较，但不得进入梯度、归一化统计、采样器、对比队列、伪标签、自监督适配或错误驱动的逐样本规则。
- user_id 只用于训练配对和评估分组，不能成为模型输入。路径、类别目录名、标签编码也不能进入特征。
- 验证属于反复使用的开发集，不能称作独立最终泛化测试。
- 无可用视觉的样本不从规范指标中删除：使用训练集类别先验输出，同时单列 supported 指标和支持率。训练损失不使用无视觉样本；先验仅由排除规则后实际合格的训练标签拟合，不使用验证或训练排除行标签。
- 用户已授权确实缺少时间戳的训练trial不参与训练。按原始文件名检测并固化排除清单（当前22条）；这些trial不进入CE、对比损失、sampler、归一化、训练先验拟合或训练侧冒烟。保留规范2039条的审计/评估行，单列 `train_fit_acc`（实际合格训练样本）和 `train_canonical_acc`（全2039条，排除行使用其余合格训练标签拟合的先验），不能把后者全称为模型训练集记忆准确率。验证388条不变；验证若出现该异常，不自动套用训练排除政策。
- Watch_TV 等覆盖不足按既有决定记录并搁置，不改划分或从验证指标排除。

## 3. 固定输入协议

### 3.1 时间证据

先解析帧键并检查源连续性，再将 trial 按原始有序帧列表划分四个区间：边界为 `floor(k*N/4)`，k=0..4，采用左闭右开区间。每个区间仅选择其中一个连续源段，按段内真实帧数最多、起始源帧最早的顺序唯一确定；选段不使用标签、预测或腕部置信度。随后在该段首尾之间以 `floor(linspace(0,L-1,16)+0.5)` 取16个索引，得到固定 `[4,16]` 布局。

源连续性必须在采样前判定，不能把有意稀疏采样后的大间隔当作源 gap。已核实含义的逐帧计数器出现重置或步长不等于来源规定值时切段；有可信时间戳时，非递增时间或相邻间隔超过该 trial 正相邻间隔中位数的3倍也切段，两种证据取并集。该中位数只用于该输入的无标签时间完整性检查，不作模型归一化统计。计数器步长需有导出格式证据，不能把任意文件序号当作连续帧计数器。对已明确识别为 `IR_数字.png` / `Depth_数字_Color.png` 等旧无时间戳格式的训练trial，先按第2节授权排除，不阻塞其余样本的缓存；不为其虚构时间，不调用骨干。其他两种证据均不足的情形仍标记 `continuity_unverified` 并阻止正式缓存，先核实来源，不能把重复键、损坏文件或验证异常一起静默排除。

短段通过重复本段真实帧填足16帧，包括 L=1 的全重复；重复是可编码真实输入，另记 unique-frame fraction，不把它当作零填充的无效帧。空区间使用索引 -1、mask=false，不调用骨干，也不借用相邻区间帧。此方案保留固定片段数，但有 gap 时不再保证覆盖区间内所有连续段；报告选段覆盖率和弃用帧数。禁止跨源 gap 插值、运动峰值选择和依据预测错误挑帧。

在每个可编码片段的第 0、5、10、15 个位置提取 DINO 外观特征，总布局为16个时间位置。它们严格是 A 已观察帧的子集，不给 B 额外原始帧。两候选共享选段、裁剪、顺序和可用性；若 IR 的某 clip-view 不可编码，该 clip-view 的 DINO 也不提取，不能单独救回原协议丢弃的帧。

记录原始帧键、区间边界、segment_id、gap 原因和最终16帧来源；可信时间戳存在时保存真实时间，否则明确使用归一化帧序，不能伪装成秒。IR/Depth 对应项须共用区间和选段，不能把各自不同片段标为同步。mask 用于已经独立编码的片段之间的下游融合，不用于声称可以消除编码阶段已发生的跨段混合。

### 3.2 四视图与局部裁剪

保留 global、person、left_hand_object、right_hand_object 四个身份，不能在跨视图交互之前合并成 context/wrist 两组。

使用现有 YOLO pose 缓存和 `ObjectInteractionROIBuilder` 的原始方向框作为几何来源；实施前核对缓存坐标系、原始尺寸、帧对应和来源哈希。禁止把 person-crop 坐标直接当作全图坐标。使用 `configs/experiments/ir_depth_videomaev2_vit_b_p0.yaml` 中已记录的 ROI 参数，不利用验证标签调 ROI。

person 和腕部框采用片段内平滑中心、局部稳定尺度，平滑必须保留有效检测框覆盖范围。禁止恢复为整条 trial 的单一腕部框。保存每帧框中心、宽高及源图归一化坐标；保留位置变化，避免跟随裁剪消除运动线索。

ROI 只允许对同一已选连续源段内、两端均有有效检测的内部短缺失插值，最大连续缺失为3个源帧；不外推首尾缺失，也不跨 gap 或从其他区间借框。平滑和尺度估计仅使用已选段，不能让未选段的框影响本段输入。短缺失补齐后，如某视图在已选段首尾覆盖范围内仍有任一无效 ROI，该整个 clip-view 标记不可用，不调用 VideoMAE；不按视图重新选段，避免 IR、Depth、DINO 的时间错配。

不可编码视图仅在输出缓存位置写零并 mask=false，禁止把部分零帧/NaN帧送入无 mask 的冻结 VideoMAE，再用输出 mask 补救。该保守策略可能降低腕部支持率：必须按用户、类别、视图报告补齐率、整段弃用率和 trial 视觉支持率，不得自动放宽3帧上限。只有没有有效人物检测时允许按既有授权退化为 global-only；存在人物但腕部不可用时保留可用人物和 global，不虚构腕部特征。因部分人物 ROI 失效导致 context 缺失，应记录该原因，不能伪称 YOLO 从未识别人。其他读取异常不得被吞掉并伪装成无 pose。

必要回归：固定帧键、选段和尺寸，改变 gap 另一侧或未选段的像素/ROI，不得改变本段缓存；不可用 clip-view 不调用骨干且无法影响其他 clip-view；空段、单帧段、等长段选择和缺失1/3/4帧都须有测试。这些是后续实现验收要求，不是本次文档修订已经通过的测试。

IR 重复灰度三通道；Depth 保留已有 depth_color 表达，不能当作已校准公制深度。时间配对和局部 ROI 几何对应为两个独立合同：时间戳/帧ID相同、分辨率相同、旧代码曾共用框，均不能单独证明框可迁移。

### 3.3 局部 ROI 几何核验前置门

正式提取前生成独立几何核验报告，绑定数据来源/采集配置、原始尺寸、坐标原点/轴向、pose 所在模态、crop-to-original 变换、训练样本 ID/帧键和原始输入哈希。报告必须包含共享框覆盖同一 person/左腕/右腕区域的证据，而不仅是文件配对通过。

优先检查赛事/传感器导出元数据中明确的坐标映射。再用第4.3节固定8条训练冒烟样本，每条四个选段的中间采样帧（索引7，空段跳过），生成带同框叠加的 IR/Depth 配对图进行核验；图中不显示动作标签或预测。逐项记录人物和可见左右腕区域是否被同名框覆盖、是否错侧、错尺度或无法判断。某区域在两模态不可辨时记 `unverifiable`，不能记通过。若该集合不覆盖某种采集尺寸/配置，则仅从训练侧按 sample_id 顺序补该配置的样本。将核验结论及核验者写入报告，不以神经网络高置信度代替几何证据。

这是局部覆盖的工程证据，不是完美像素级配准或全数据保证；报告声明其适用采集配置和限制，运行时遇到新尺寸/配置必须停止。存在反例、坐标映射不明或关键区域始终无法核验时状态为 `geometry_unverified`，阻止正式缓存；先报告并确定模态各自 ROI 或禁用局部 Depth 的修订协议，不能自动切换。仅改成 token 交互不能解除这个前置门，因为交互无法恢复已经裁掉的内容。

几何门通过后，本阶段只在同连续片段、同视图的空间 token 集合之间做交互，加入时间编码并允许片段内跨 tubelet 时间位置读取；不跨源 gap，不把相同空间单元索引强制视为像素对应。该范围与第5节的32 query×32 key一致，不是逐tubelet的4×4独立交互。实现测试需验证未知几何状态不能调用共享局部裁剪、已知坐标变换的合成框映射正确；实际数据通过与否由单独报告证明，本 spec 不预先标记通过。

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

先逐个骨干加载并释放，避免8GB GPU同时驻留。先用仅训练样本的8条固定冒烟集合测量冷/热缓存读取、CUDA同步计时、峰值显存和单样本字节数。选择方法固定为：从具有可读成对源图的训练 trial 中按 `(源帧数,sample_id)` 排序，以 `floor(k*N/4)` 分成四组，各组取 sample_id 最小的两个；不看标签或预测。无效选段/ROI样本也保留在支持率报告，不静默换成容易通过的样本。不足8条时报告资源资格不足，不借用验证样本。

全量缓存前另设正式训练资源门：A/B分别使用 batch=32、完整 token 布局和全有效 mask，完成3次 forward/backward/optimizer step，包含CE、跨用户对比损失、正式精度、dropout、优化器状态和实际注意力实现。可用确定性的合成特征/标签完成最坏布局压力测试（不用于精度或模型资格结论），再用上述训练侧小缓存跑真实梯度路径。不能用 batch=1 或只有 forward 的结果替代；不得把实际无效ROI假标成真实可用以声称数据正确。

两个阶段均记录同步后的时间、peak allocated/reserved、设备空闲显存；训练门还记录 loss/梯度有限、optimizer状态改变、初始零残差及后续新增分支梯度。采用先warmup再测量，避免漏掉惰性分配的AdamW状态。所有冒烟参数/optimizer丢弃，正式A/B从共同冻结初始化与seed重新开始，不能把压力测试当成预训练。

资源硬门：提取和正式batch训练两阶段的峰值 allocated 均 <7300 MiB，且真实运行不得OOM；估计磁盘用量（含权重、缓存、索引和并存临时分片）不超过所在盘可用空间的70%；预计单次全量特征提取不超过12小时。全布局FP16特征本体为每条3.625MiB，2427条约8.592GiB，不等于峰值显存。两阶段冒烟都通过才允许全量提取；越门只报告并请求调整，不自动降低batch、替换骨干或降输入。12小时是预算上限，不是运行时间估计。

## 5. A/B 深融合接口

两个候选共享256维、4 attention heads的分层主干：一层片段内Transformer和一层跨片段Transformer。投影后的IR每片段为 `4 views*8 times*4 cells=128` 个token；Depth交互限定在同片段、同视图的32个query与32个key之间。B的外观cross-attention限定为同片段的128个视觉query读取 `4 views*4 times*17 tokens=272` 个DINO key，不把四个片段拼成一次全局密集cross-attention。

外观/Depth残差进入视觉token后，运行片段内Transformer；再以每个视图独立的可学习pool query压缩到4个视图token/片段。四片段共16个token，附加1个trial query，经跨片段Transformer输出trial表征。必须显式加入视图身份、时间顺序、空间单元及ROI坐标；所有交互和池化使用mask，全空集合返回零特征/不可用标记，禁止全负无穷softmax产生NaN。A/B独立训练但共享上述结构；B仅多出已声明的外观路径。正式资源冒烟必须使用这一拓扑，不能临时换成平均池化或较少token。

- A `temporal_visual`：IR VideoMAE 为主，同连续片段、同视图的Depth token经带时间编码的32×32交互及零初始化有界残差适配器进入IR特征，允许片段内跨tubelet读取。无Depth时严格恢复IR路径。保留四视图，不使用静态等权logits平均。
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
- 对比正例/负例/忽略项、重复ID及归约严格遵循下方公式；CE仍对采样batch中全部出现项求均值，保留既定有放回采样的曝光含义。报告实际正例覆盖、唯一ID数量和每类/用户曝光。
- A/B 共用完全相同的样本序列、优化步数、共同模块初始化和输入；新增模块的 RNG 独立，不能改变共同模块 dropout 随机序列。
- 缓存阶段不声称做了在线像素增强。训练 token dropout=0.10，仅作用于有效 token 且每条有视觉 trial 至少保留一个真实 token；验证关闭。更复杂增强和骨干适配不在首轮自动执行范围。
- 每 epoch 存 latest checkpoint，包含模型、optimizer、scheduler、随机状态、sampler/epoch、配置和缓存来源哈希。中途续训必须还原后续样本序列；正式完成目录不得被静默覆盖。
- loss/logits/gradients 任一非有限即停止并记录；不静默跳过坏 batch。两候选全部完成后统一读取验证结果，不根据 A 的验证结果改 B。

### 6.1 跨用户 SupCon 的完整集合与归约

仅对比项按 sample_id 去重：按共同采样顺序保留每个ID第一次出现的embedding，丢弃其余出现项的对比资格，不改变CE；同ID若标签或用户不一致立即报错。使用首次出现而非跨dropout副本平均，确保重复ID不会改变第三个anchor的候选权重。唯一集合记为 U。

对 anchor i 定义：正例 `P_i={j∈U: y_j=y_i 且 user_j≠user_i}`；负例 `N_i={j∈U: y_j≠y_i}`；自身及同类同用户的其余样本全部忽略，不进入分母。分母集合 `D_i=P_i∪N_i`。有正例的anchor使用：

`L_i = -(1/|P_i|) * sum_{p∈P_i}[ dot(z_i,z_p)/0.10 - logsumexp_{j∈D_i}(dot(z_i,z_j)/0.10) ]`

z为L2归一化embedding，分母与logsumexp用float32稳定计算。最终对比损失对有正例的唯一anchor求均值；无正例anchor不进入该均值，整批无正例时返回 `z.sum()*0` 的可微零。没有负例但存在正例时仍按同一公式计算，不另造负例。

必要手算测试：a=(类0,user1)、b=(类0,user2)、c=(类0,user1)、d=(类1,user3)，a的正例只有b，负例只有d，c被忽略；当相关相似度相等时a的损失为ln(2)。重复插入同ID的b且首次embedding不变时，全部对比损失不变；同类同用户整批及单样本批次得到有限可微零。这些用例在实现阶段验证，不能以仅检查loss有限代替集合正确性。

## 7. 评估与投资决策

固定 epoch30，eval模式无增强，主训练指标 `train_acc=train_fit_acc` 只对实际合格训练样本计算；另对规范train2039逐条报告 `train_canonical_acc`，禁止两者混称。val_acc仍对全388条计算。报告Macro-F1（全部40类）、worst-user、每用户准确率、NLL、Top3/Top5、零召回类别和支持率，明确每项分母和排除原因。另报视觉支持子集，不能替代规范验证指标。

逐 sample_id 对齐 A/B 及旧基线，报告 rescue、harm、net、每类变化和用户分解。打印计数及分母。吃食物、喝水、吃药、使用餐具、手机等类别单列，仅用于解释，不据验证样本拟合类别例外规则。

决策拆成两个独立维度，不用外观增量门覆盖绝对目标：

- `absolute_progress`：A/B各自按正确计数报告 `<319`、`319..329`（达到0.82）、`330..349`（达到0.85）、`>=350`（达到0.90）。同时保留全部Macro-F1及用户结果；准确率达标不等于稳定性或其他指标达标。
- `appearance_increment`：B相对A净增至少8/388且worst-user不下降，记 `positive`；未满足记 `mixed_or_unproven`，负净增另记 `regression`。这是预声明的投资标准，不是显著性检验。
- `next_discussion`：任何候选达到350先报告本划分的0.90目标已达；未达350但A达到319、B未有明确增量时，保留讨论A适配的路径；B达到319且增量positive时，讨论外观路径后续投入；其他结果报告仍需诊断。所有分支都只产生报告，不自动训练、蒸馏或扩大候选。

决策回归案例：A=350、B=354，即使B净增不足8，仍必须报告两者绝对目标已达、外观增量未获该投资标准支持；A=330、B=328时报告A达到0.85、B回退，不能把整个视觉路线判为失败。319/330/350及8个净增都是投资/目标阈值，不是收益预测；未达标不宣布IR或DINO的能力上限。

0.90 目标对应至少350/388；两用户、单seed结果不能支持稳定跨用户泛化声明。任何后续 adapter 微调、增加 Skeleton、额外消融或蒸馏，都需要先汇报本轮结果并确定新阶段协议。

## 8. 实施与验收顺序

1. 审阅本 spec 后，写独立实施 plan，沿用 Inline Execution，不默认派子代理。
2. TDD 实现人口/采样/ROI 深接口，校验几何来源与无泄露边界；不改旧实验接口。
3. 锁定两骨干来源，实现特征提取和可恢复缓存；先完成编码前连续段/缺失合同与训练侧几何门，再运行提取器冒烟。
4. TDD 实现 A/B 融合、跨用户配对损失、恢复训练与配对报告。
5. 一组任务完成后统一审计，完成正式batch=32的A/B反向资源门；两阶段冒烟及数据门均通过后，再启动全量缓存和A/B固定训练。后台进程隐藏窗口、独立日志和状态文件，不使用替代计划的额外实验。
6. 完成后检查来源哈希、预测人口、复算指标、全量回归和 git diff 检查，再提交本实验文件并更新临时 handoff。新分支不默认推送远程。

## 9. 代码边界与交付位置

新增模块集中在 `src/experiments/visual90_config.py`、`src/data/visual90_dataset.py`、`src/models/visual90_encoders.py`、`src/models/visual90_fusion.py`、`src/training/visual90_training.py`。

新增入口：`scripts/cache_visual90_features.py`、`scripts/run_visual90_experiment.py`；配置：`configs/experiments/visual90_appearance_temporal.yaml`；测试：对应 `tests/test_visual90_*.py`。

新输出仅写 `outputs/visual90_appearance_temporal/`，新报告仅写 `reports/visual90_appearance_temporal*`。不覆盖历史 checkpoints、缓存、指标或 dirty Motion Attribute 文件。学生从头训练和模型包大小限制留到后续蒸馏阶段，本轮教师骨干不作为学生包交付。

## 参考与材料护照

- 来源技能：academic-research-suite / experiment-agent，plan 模式；Superpowers 设计流程。
- 验证状态：v2五项审计修订保留；v3记录用户新增背景和训练排除授权。历史计数已复算，基础数据接口测试通过，但完整编码隔离、实际几何门、资源资格和实验效果仍未通过实测。
- DINOv2 官方模型卡：https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- DINOv2 论文：https://arxiv.org/abs/2304.07193
- Supervised Contrastive Learning：https://arxiv.org/abs/2004.11362
- AdaptFormer（后续阶段参考，不是首轮已实现内容）：https://arxiv.org/abs/2205.13535
