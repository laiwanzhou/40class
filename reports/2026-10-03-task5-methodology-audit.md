# Task5 方法学与数据边界独立审计（Methodology and Data-boundary Audit）

日期：2026-10-03，Asia/Shanghai。审计者：`/root/audit_20261002_methodology`。基础提交（base commit）：`25b8ec0570d92573db99375f7087c48aa4bdee10`；本次审计该基础之上的Task5工作区实现，审计者未编写实现。

## 结论（Verdict）

**NO-GO：当前版本尚不应开始正式学生训练（formal Student training）。** 发现一项冻结选择预算未绑定的P1，以及一项训练尾批策略与源码不同、尚未明确披露的P2。修订均局限于选择验证、DataLoader及针对性回归，不要求重新抽取像素或重选研究方法；正式训练尚未开始，没有需要撤销的正式A2结果。

其余已审查的像素布局、质量字段、同步增强、公开MC3初始化、冻结参数/BatchNorm、hybrid损失、优化器和训练/推理分区边界，与当前固定划分规格相容。修订后需复核具体修改项再给GO，不能沿用此前Task4审计结论。

## 范围与证据限度（Scope and Verification Limits）

已阅读计划Task5、规格S7及轻量校验修订（lightweight validation revision），检查pixel_cache.py、no_vote_datasets.py、visual_student.py、三个CLI、相应新测试和必要的队友源码函数。独立读取正式像素缓存的记录、rows.csv、数组头及view_valid/quality/source_frame_indices/source_time_seconds/completed等小型字段；images.npy仅以只读mmap取得shape/dtype，未扫描图像内容。

没有读取final4私有标签，没有遍历原始数据，没有执行raw/全缓存SHA，没有重新运行GPU训练。本轮只写本报告。29项新测试通过和真实GPU批次结果来自已有验收日志，不把它们写成本次重新执行。

## Important发现与最小修订（Findings and Minimal Fixes）

### F1 / P1：refit预算没有绑定已登记的select模型（Unbound Selected Epoch Budget）

位置：[visual_student.py:199](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:199)–220。

当前refit使用registry.read读取选择模型，检查stage/phase及selection.config等于当前recipe，再直接取selection.budget['epochs']，只验证它落在1–18范围。未要求该值等于已登记select模型config['budget_epochs']，也未校验selected.config['model_recipe']与当前recipe一致。

触发场景：selection.json中的epoch被误改为范围内其他值，或接口传入与登记模型预算不一致的Selection。即使选择模型引用不变，refit仍可按另一预算训练。这违背“公开初始化、按已冻结选中epoch重新训练”的契约。

影响：实际refit不再对应已记录开发选择，后续A2差值解释及来源追溯（provenance）不完整。这是实现接受错误预算的缺口，不是已发生泄露或已发生错误训练的证据。

最小修订：使用轻量registry.verify(selection.fit_artifact, 'A2', 'select')验证该直接选择产物，要求kind=supervised_model，并同时绑定selected.config['model_recipe']==recipe与selected.config['budget_epochs']==selection.budget['epochs']。加入“合法范围内但与选择模型不同的epoch”拒绝回归；继续使用fresh公开初始化，不把select模型加为refit初始化父节点。

### F2 / P2：singleton尾批策略与参考源码不同（Singleton-tail Policy Mismatch）

位置：[visual_student.py:218](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:218)；参考[train_p86_visual_pixel_oof.py:244](D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project/aligned_multimodal/train_p86_visual_pixel_oof.py:244)–255。

新DataLoader未设drop_last，默认保留所有尾批。参考源码明确使用drop_last=shuffle and len(dataset)%batch_size==1。正式select有效IR人口为1957条，batch4，因此当前实现每轮会产生1条singleton尾批；参考条件会丢弃该轮随机顺序中的单样本尾批。refit的2342条则保留2条尾批，两种实现该处一致。

影响：每轮样本消费数、最后一个梯度累积组（gradient accumulation group）及更新轨迹与参考源码不同。现有real4验收和整除的合成训练不能覆盖此分支。MC3分类头使用LayerNorm且BatchNorm3d运行统计冻结，不能据此断言当前singleton必然崩溃；问题是未明确的复刻差异（replication deviation）。

最小修订优先复用参考drop-singleton条件，保持较大尾批可用，并在history记录实际消费样本数及optimizer_steps；valid拟合支持1957与该轮实际消费1956必须区分。若本次明确冻结保留所有有效样本的变体，则必须在规格/配方中披露该选择并覆盖singleton/末尾累积回归，不把它称为DataLoader逐项相同复刻。无需重建像素缓存。

## 正式像素与时间字段（Formal Pixels and Time Fields）

| 分区（Partition） | 行数 | images shape/dtype | 可用IR | 缺失回退 | 绝对时间全部有限的行 |
|---|---:|---|---:|---:|---:|
| refit14 | 2427 | [2427,2,16,3,160,160] uint8 | 2342 | 85 | 2341 |
| final4 | 609 | [609,2,16,3,160,160] uint8 | 591 | 18 | 590 |

两份rows.csv逐行等于相应公共manifest；记录stage=pixels、phase=raw、complete=true、fixture=false。view_valid的逐样本可用性与公共IR标志一致；quality有限且范围[0,1]；completed全部true。source_frame_indices和source_time_seconds均为[N,2,16]。

每个分区有一条可用IR记录未取得完整绝对时间，按已批准策略保留NaN而非伪造counter时间。这是明确的数据支持限制（data-support limitation），不是本轮阻塞；后续运动时间对齐须消费其真实mask/可观测索引，不通过标签或trial排序补造时间。本次不要求再扫raw定位或修补。

像素构建的early/late窗口0–.70/.30–1、16帧、scene/person/workspace、人物裁剪1.15和工作区裁剪1.40、cv2.INTER_AREA，以及quality/valid回退计算，与build_p86_visual_pixel_cache一致。三视野轴不是RGB轴；MC3内部将灰度复制为Kinetics输入三通道。新增float32 quality存储符合已定schema。

## 已通过的配方与数据边界检查（Passed Recipe and Boundary Checks）

- **公开初始化（public initialization）：** initialize_student用验证过的本地MC3权重逐块strict加载stem/layer1–4，头部按seed重新初始化；select和refit均调用public_init=True，refit不加载selected checkpoint权重。初始化40类、width512、frames16、temporal_modeling=true、gated融合和dropout0.18；关闭直接蒸馏投影。
- **冻结和BatchNorm（freeze and BN）：** 使用源模型freeze_low_level(layer2)，冻结stem/layer1/layer2参数。源train覆盖保持全部BatchNorm3d在eval，layer3/4卷积仍可训练；不是只冻结参数而继续更新BN运行统计。
- **同步增强（synchronized augmentation）：** 直接复用源_subject_robust_augment；同一空间平移、水平翻转、时间重采样、gain/bias作用于两窗三视野。Dataset用返回temporal_indices同步移动valid、quality、source frame index和time字段；teacher clip目标保留源配方的六clip语义。
- **hybrid损失（hybrid losses）：** CE使用40类权重指数0.35及smoothing0.1；KD温度2/权重1按teacher_valid屏蔽；relation权重0.2，Gram关系及clip pair mask与源码代数一致，并交叉teacher_valid。直接feature和stage KD保持计算图零值，不因配置feature_weight=0.5误接直接特征损失。CE训练人口过滤到pixel可用行，并要求teacher可用性匹配。
- **类别权重（class weights）：** counts仅来自有效fit标签，要求40类支持。对权重再除以全类均值是统一比例；加权CE的mean归约分子与归约分母同乘此比例，数学上不改变该损失，相比源码仅有浮点舍入差异，不是另一个权重策略。
- **优化（optimization）：** AdamW，backbone/head LR1e-5/2e-4、wd0.08、foreach=false；两轮warmup及随后cosine比例与源码公式一致。minimum LR1e-5按源码用作head ratio，backbone随同一比例变化；不能将其解释为所有参数组都截断到1e-5。accumulation4、AMP/GradScaler、clip norm2及不满累积组的末尾step与原训练体相同。
- **开发选择（development selection）：** select只使用有效train12标签和对应A1 select目标；development2推理Dataset无教师、无增强，按全388条及同phase prior计分。epoch1–18，accuracy/macro-F1/最差用户/更早epoch字典序决胜符合S7；这是已批准固定划分规则，不恢复旧入口的加权selection_score或early stopping。
- **teacher/prior边界（phase boundaries）：** teacher目标必须精确匹配拟合分区的ID/user/class order及已登记概率文件；现有registry限制train12预测不能有refit监督祖先、refit不能有select监督祖先。prior明确从A1同phase已登记统计取得。共享refit14原始pixels/公开六clip特征不把开发标签并入select训练。
- **推理隔离（inference isolation）：** predict/sequence只创建无labels/teacher的Dataset，加载A2检查点、公共pixels、同phase prior与manifest；没有加载teacher文件的CLI路径。新合成测试删除全部监督label/teacher文件后仍能完成predict/sequence，其结果来源记录保留metadata引用，而不再读取删除的监督文件。
- **原生序列及锚点（native sequence and anchor）：** 使用源encode_backbone_sequence和forward_from_backbone_sequence生成[B,2,3,16,512]及anchor logits；按同phase A2身份及pixels引用登记。select只能对应train/dev，refit对应refit/final；无IR位置序列零值、anchor为先验且valid=false。文件锚点不随后续A7更新重新产生。

## 已有验收证据与尚未验证内容（Existing Evidence and Unverified Work）

[outputs/task5_audit/targeted.log](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/task5_audit/targeted.log)记录29项针对性测试通过。[real-batch.json](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/task5_audit/real-batch.json)记录真实train12四条GPU验收：2.481秒、峰值2.072508GiB；loss3.746219、CE3.501929、KD0.241507、relation0.013918，feature/stage直接项为0。冻结前缀无梯度、head有非零梯度，formal_training=false。

real_batch_acceptance.py还明确断言原生序列首末时间槽不同、数组有限，并记录shape[4,2,3,16,512]。这些是真实接口验收，未覆盖完整optimizer epoch、1957条尾批、18轮开发选择或选中预算的正式refit；不能把2.481秒当成全epoch速度或正式A2准确率。

修复F1并解决F2的源码/变体口径后，只需针对性回归和独立复审修改项；不要求重复已通过的像素生成、公开教师训练或全量SHA。Task5正式训练仍须在上述门槛关闭后开始。
