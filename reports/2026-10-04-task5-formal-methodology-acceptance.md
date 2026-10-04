# Task5 正式产物方法学独立验收（Formal Methodology Acceptance）

日期：2026-10-04，Asia/Shanghai。审计者：`/root/audit_20261002_methodology`。审计时HEAD为`faf8980eb54e31ac1344feeafec42a7c7af668a3`。

## 结论与范围（Verdict and Scope）

**GO：Task5 正式开发选择、refit及四分区targets/sequence通过本次方法学轻量独立验收（lightweight independent acceptance）。没有发现未解决的P1/P2方法学问题。** 本轮没有为开发成绩重新调参、重选方法或重训模型。

本次重新核对实际正式产物，不以此前代码GO替代验收。读取select/refit history、selection、model/completion/identity小型资料，四区公共manifest、四份小型targets、anchor logits/valid/completed、sequence数组头及每区各一条有效/缺失序列切片；只用development2允许标签复算开发成绩。对34个去重祖先记录（ancestor records）检查元数据及引用摘要，不打开祖先payload。

没有调用生产ArtifactRegistry.verify，没有读取final4私有标签，没有打开raw、images或全量feature缓存，没有做sequence全数组扫描或祖先payload哈希，没有训练或修改生产文件。本轮只写本报告。既有代码审计及工程正式验收不重复执行。

## 18轮开发与第17轮选择（18-epoch Development and Epoch-17 Selection）

[select history](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/A2/select/history.json)连续覆盖epoch1–18。独立按既定accuracy、fixed40 macro-F1、最差用户accuracy、较早epoch决胜，最优为epoch17，与[selection.json](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/A2/select/selection.json)、登记模型budget_epochs及完成标记一致。

第14、17、18轮accuracy同为189/388；宏F1依次0.3546943955、0.3648354050、0.3526577796。第17轮按第二决胜项胜出；即使第18轮最差用户accuracy更高，也不能越过macro-F1规则改选第18轮。

正式development2 targets与允许的开发标签按opaque ID一对一连接后，独立复算：

| 开发人口（Development Population） | 正确数/全分母 | accuracy |
|---|---:|---:|
| user6 | 95/203 | 46.7980% |
| user7 | 94/185 | 50.8108% |
| 合计（total） | 189/388 | 48.7113402% |

fixed40 macro-F1=`0.36483540499875017`；最差用户accuracy=`0.46798029556650245`。三项值及开发ID摘要都与epoch17、selection.metric完全一致。分母为全部388条，包括3条缺IR先验回退，不只计算385条IR有效行。

该值是使用同一开发集选择18个epoch后的描述性开发成绩（descriptive development result），不是final4成绩。本轮不分析或修正分数，不改变冻结配方。

## 重新拟合与配方执行（Fresh Refit and Executed Recipe）

| 检查项（Check） | select | refit |
|---|---:|---:|
| 训练轮数 | 18 | 17 |
| 登记/选择budget | 17 | 17 |
| 规范拟合人口 | train12 / 2039 | refit14 / 2427 |
| valid_fit_rows | 1957 | 2342 |
| 每epoch实际样本消费 | 1956 | 2342 |
| 每epochoptimizer_steps | 123 | 147 |

[refit history](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/teammate_single_teacher_fixed_split/fixed-split-single-teacher-v2/A2/refit/history.json)连续覆盖epoch1–17，每条仅包含epoch/training，没有开发或final准确率用于择checkpoint。两阶段valid拟合ID摘要与公共IR有效子集一致；singleton丢弃规则的实际样本数、accumulation4末尾更新次数均符合已审查配方。

select/refit日志均记录public_init=true；模型直接父节点包含visual_student_initializer/public_weights，而refit不存在select A2监督模型父节点。refit使用对应refit14 A1目标和prior，从公开MC3初始化重新拟合；不是继续训练select检查点。两阶段completion.artifact、identity及对应artifact引用一致，refit completion.selection=null。

两个model_recipe与selection.config一致：MC3 temporal、16帧、160分辨率、layer2冻结、subject_robust、batch4/accumulation4、seed20260811、class-weight power0.35。A1祖先head温度为1；A2 KD温度为2、权重1、relation权重0.2、feature直接项及stage KD为0。所有history损失项有限，loss与CE+KD+0.2×relation在数值容差内一致；未因开发成绩改成其他损失。

本次没有重新加载大型checkpoint或独立重训；fresh初始化依据是此前已GO生产路径、实际初始化日志及本轮产物父角色，而非声称复验全部权重字节。

## 四区目标、原生序列及锚点（Targets, Native Sequences and Anchors）

| 分区（Partition） | 模型/先验phase | 规范行数 | valid IR | prior回退 |
|---|---|---:|---:|---:|
| train12 | select | 2039 | 1957 | 82 |
| development2 | select | 388 | 385 | 3 |
| refit14 | refit | 2427 | 2342 | 85 |
| final4 | refit | 609 | 591 | 18 |

四份targets均为A2/predictions/predict、complete=true、formal；键精确为sample_ids/class_ids/logits/probabilities/valid，没有labels、旧OOF或多头字段。ID集合、顺序、user对应及class_ids=0–39与公共manifest/登记记录一致。logits/probabilities形状[N,40]，概率有限、非负、行和1；valid与公共IR可用性逐条一致。无效行概率与同phase float32 prior逐元素完全相同；有效行符合softmax(logits)。

四区sequence记录绑定对应phase的同一个A2模型及与targets相同的pixels父引用。sequence.npy头部均为[N,2,3,16,512] float16；anchor_valid与targets.valid逐元素相同、completed全部true。anchor_logits与对应targets.logits最大绝对差为0。

每区读取一条有效序列，均有限且首末时间槽不同，支持原生时间特征（native temporal features）；每区一条缺失序列切片为0。此为有限切片检查，没有宣称全sequence内容扫描通过。全部小型anchor及预测数组已核对；四区rows.csv逐行等于公共manifest。

直接targets摘要（target SHA256）：

| 分区 | SHA256 |
|---|---|
| train12 | 03f1a67296bda3165773bc4bfc07a52d85d9116831012d70dacf6c1426b3a5c4 |
| development2 | 217b0961f1af82f115ef193269a4c9c4a3d164be60caf98630e17d75bab2a7fb |
| refit14 | d9bc9b899aa25d42f69de40938806eebac595661050999398a5a93f32bd3d8d7 |
| final4 | 396ca09b7fc7f848d5de96956b1ef91f721296f74fca6bd4ea51d4f6b1db68a1 |

可恢复引用分别为A2/select或refit目录内的`<partition>_targets.artifact.json`，以及sequence/select或refit的`<partition>/artifact.json`。未修改任何旧产物身份。

## 阶段祖先与最终标签封存（Phase Ancestry and Final-label Seal）

去重检查34个元数据记录：监督模型、统计只含A1、A2、class_prior；监督标签仅labels_train12/labels_refit14；预测角色只含A1/A2。其余为公共manifest、P28/P29、pixels、冻结visual_features、公开pose/visual/MC3初始化及visual_sequence。没有历史OOF输出、expert bank、P310或adapted_model祖先。

对每个A2模型分别检查整条学习祖先（learned ancestry）：select的全部监督模型/统计均为select、fit train12；refit均为refit、fit refit14。四区targets/sequence对应的模型、prior和原始pixel父节点无跨phase混用；共享标签无关原始缓存不构成监督祖先泄露。

[student-continuation.log](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/task5_audit/student-continuation.log)记录select于01:57:05结束，refit于03:01:42结束，final4预测直到03:12:29之后才开始；末尾03:14:02记录sequence-final4完成和TASK5_CLI_STAGES_COMPLETE。epoch17选择可由全部开发history独立确定；refit history没有评价字段，final4没有标签或accuracy进入selection/completion。结合实际记录与此前已审查代码，本轮选择没有使用最终分数（final score）。

本次仅读取development2标签，不读取、连接或评分final4标签。final609条及18条先验仍封存；run_state为generating，尚未到整个A1–A9的16候选总冻结或揭盲（reveal）。本报告不提前报告最终成绩，也不宣称完整实验已完成。

## 验收边界（Acceptance Boundary）

没有需要修改正式配方或重生成Task5模型/预测的未解决方法学问题。本GO限于正式Task5产物及其开发选择；工程正式验收另有独立报告。后续任务应使用已登记、同phase的targets/sequence/anchor，不用本报告作为重新调参、最终标签揭示或全量原始内容复验的理由。
