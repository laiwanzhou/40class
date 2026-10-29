# P31 Task6 运动缓存与源转换（Motion Cache and Source Transform）

日期：2026-10-29，Asia/Shanghai。唯一指导为[P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。本文仅为Task6实施与验收记录（Implementation and Acceptance Record）。

状态：**Task6已完成工程实现、正式生成和有限自验（Complete with Bounded Self-verification）**。本轮没有请求或执行Task6独立子智能体复审，不将自验称为独立GO；Task7未启动。

## 源处理与接口（Source Processing and Interfaces）

新增 [motion_cache.py](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/motion_cache.py)、[正式构建CLI](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/scripts/build_no_vote_motion_cache.py)、[真实样本验收CLI](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/scripts/verify_no_vote_motion_source.py) 和 [Task6测试](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/tests/test_no_vote_motion_cache.py)。

- `build_motion_source`：从已登记P28提取Skeleton原始坐标／置信度、人数及采集时间；直接复用源P31的骨骼特征、IMU全点读取与时间区间分配，保存标签无关的逐样本P31缓存，不执行历史main或读取类别字段。
- `build_motion_windows`：使用pixels的实际源帧索引及时间戳，调用源P46身体坐标旋转与P86 IMU聚合，保留每bin4点、16通道、52维bin统计、48维global统计及全部掩码。
- `describe_motion_transform`：生成 `source_transform` 公共确定性描述符（Deterministic Descriptor）；外部变换为identity，`cross_sample_fit=false`、`imu_instance_normalization=false`，无fit用户、跨样本mean/std或标签依赖。
- `apply_source_transform`／`motion_item`：只保留源物理转换后的值，按原MOTION_FIELDS／BOOLEAN／TEMPORAL字段及轴读取；布尔字段转bool，其他字段转FP32，同步时间增强索引由源方法产生，不增加z-score。后续Task8/10应消费此描述符及同ID数据，不恢复旧额外归一化。

五个直接源模块与最终(1)包逐文件字节一致：P31 preprocessing、P46 event preprocessing、P86 motion window builder、cached motion data、MoBind motion data。仅导入被消费源码与函数，不全扫1167文件、不执行历史训练器。

## 人口、数值与复用（Population, Numerics and Reuse）

| 缓存人口（Population） | 完整行数 | Skeleton有效行 | IMU有效行 | 全无运动／IR行 |
|---|---:|---:|---:|---:|
| refit14 | 2427 | 2341 | 2283 | 85 |
| final4 | 609 | 590 | 580 | 18 |

“有效”来自源输出mask，不以文件路径存在代替实际关节／设备有效性；两种模态的有效集合可以不同。train12的2039条与development2的388条按精确ID取refit14公共缓存子集，避免复制缓存；没有拟合跨人口统计，不读取它们的标签，后续模型仍须仅fit对应阶段训练用户。

refit14源缓存包含68203帧／317963条IMU点，final4包含17715帧／83754条IMU点。各有1条counter-only且无IMU的时间轴，沿源10Hz回退计算骨骼导数；其他非空行使用P28恢复的真实采集时间，未用数值帧号冒充绝对时间。

12个MOTION_FIELDS保持源存储精度：一般数值FP16、布尔掩码uint8，`imu_global_statistics`为FP32以保留平方差等较大统计。附加公共源帧索引为int32、绝对采集时间为FP64；绝对epoch秒不压成FP32。P31内部IMU／相机相对时间仍按源FP32计算，不能混淆两种时间口径。

缺Skeleton／设备时保留源mask与零占位；全缺失行不生成伪统计或伪有效设备。当前公共清单无“缺IR但有motion”行；若其他输入出现该情况且没有已闭合源时间轴，本适配器明确拒绝，不发明时间。已登记源＋窗口载荷约273.67 MiB，不含辅助逐trial JSON与日志。

Task1–5、原始数据、队友快照及两提交包均未修改。ROI质量精度残余与P12／Thermal／P89／A18门槛仍独立保留；协议身份仍为 `2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`，全局正式生成门槛仍false。Task6是raw cache／公共descriptor，不需要借用A2局部放行或添加学习祖先。

## 验证证据（Verification Evidence）

- 7项初始缺接口测试RED→GREEN；真实像素绝对时间校验先RED再实现；绝对时间FP64保存断言先RED再修复，避免存储元数据失精度。首次集成夹具ID不合规，修正为64位opaque ID而未放宽正式校验。
- 40项motion／v3 roles／cache-import针对性测试初轮通过49.74秒，源Dataset时间增强／字段对照扩展后，最终同40项回归通过46.07秒；扩展单测也单独通过3.66秒。源全NaN Skeleton fixture保留一个 `All-NaN slice encountered` 警告，掩码和输出有限性通过；未改源算子隐藏警告。
- 真实train12一条22帧／102条IMU点，12个窗口字段与源旋转／聚合函数结果在源存储dtype下逐值差0，pixels绝对时间差0，P31区间点数守恒，label payload打开尝试0；6.52秒。未将验收partial产物当正式缓存或后续父节点。
- 正式产物自验0.88秒：完整ID／用户／人口、12类数组headers、完成mask、原始pixels索引及小时间表全部对齐；每区首／中／尾及两条缺失行抽样有限性、mask、Dataset类型和外部identity验证通过；两份transform绑定对应motion。未扫描整个数值载荷，也未声称抽样证明所有未抽payload完整性。
- 源逐trial NPZ写入时产生摘要；大窗口数组写入时产生摘要。后续读取只核验登记元数据／文件stamp、headers、IDs／mask及消费切片，并以只读mmap打开；测试拒绝后续重新SHA整个NPY。没有raw全目录SHA、视觉骨干重跑或final标签读取。

正式命令：

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/build_no_vote_motion_cache.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml --partition all
```

日志 `outputs/task5_audit/P31-task6-formal.log` 以 `motion_pipeline_complete` 结束；PID24744／session8676已退出。源／窗口阶段计算合计102.15秒，日志创建至最终写入约125.24秒；不包含前置实现及测试。无后台训练或新自动化。

产物位于当前v3 run的 `motion/{refit14,final4}/full/{source,windows}`，各有artifact.json；normalization引用是 `protocol/motion_transform_<refsha>.json` 的source_transform描述符。权威完成入口为 `protocol/task6_complete.json`，绑定原ready、自验与真实样本证据及本记录；`task6_artifacts_ready.json`仅表示生成时已齐备。

可追溯小证据：`P31-task6-red.log`、`P31-task6-time-red.log`、`P31-task6-precision-red.log`、`P31-task6-targeted-tests.log`、`P31-task6-final-regression.log`、`P31-task6-source-dataset-test.log`、`P31-task6-real-sample.log`、`P31-task6-final-acceptance.json/.log`。Task7源RF尚未实施，需要它自己的fit范围／源配方验收，不能把本Task raw-cache完成视作全局监督放行。
