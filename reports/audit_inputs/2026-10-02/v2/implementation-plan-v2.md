# 队友单视觉教师固定划分实施计划（Fixed-Split Single-Teacher Implementation Plan）

> 执行者须按任务使用 superpowers:executing-plans；若用户另行明确选择子智能体实施，使用 superpowers:subagent-driven-development。下列任务均尚未实施，复选框表示未来验收，不能视为已运行结果。

**目标（Goal）：** 在train12/development2/refit14/final4固定划分上移植一条无历史多教师投票的Visual/Skeleton/IMU流水线，并生成可解释的阶段比较。
**架构（Architecture）：** 复用队友模型、预处理算子、增强和损失；新建标签无关缓存、固定划分训练循环和来源验证。旧CLI是源码参考，不是新接口。公共权重和原始数据缓存可共用，监督训练祖先按select/refit分离。
**技术栈（Tech Stack）：** Python3.12、PyTorch2.7/CUDA12.8、torchvision、transformers、scikit-learn、NumPy、pandas、OpenCV、Ultralytics、pytest。
**规格（Specification）：** [v2规格](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/specs/2026-09-22-visual-motion-no-vote-ablation-design.md)。
**审计基线（Audit Baseline）：** [2026-10-02独立审计](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-02-handoff-plan-independent-audit.md)。本计划于2026-10-02修订为v2；旧1113行版本见报告中的输入快照（input snapshot）。

## 全局约束（Global Constraints）

- 工作树：`D:/work/2026.7.14_kaggle/_single_visual_processing_replication`；分支：`experiment/single-visual-processing-replication`。只在此实施新代码。
- train12=user1,user2,user3,user5,user8,user9,user16,user18,user19,user20,user21,user22，共2039行；development2=user6,user7，共388行；refit14为前两者并集2427行；final4=user4,user17,user23,user24，共609行。
- final4的18条全部模态缺失始终保留refit14先验；IR可用591、Skeleton/Depth可用590、IMU可用584。规范行数与模型实际可用拟合行数分开记录。
- 不做三折OOF，不重新匹配旧视觉教师接口，不使用历史任务权重、缓存、标签衍生final报告、Kaggle分数、P310或expert bank。
- 视觉六clip来自IR的早晚两窗/scene-person-workspace三视野；Depth提供几何同步，不增加第二视觉教师。所有模型都是40类，列顺序0–39。
- select的全部监督祖先fit train12，dev仅预测/选择；refit的全部对应监督祖先fit refit14。A9单独记录final4无标签适配角色。
- run根：`outputs/teammate_single_teacher_fixed_split/<run_id>/`；源码根：`D:/work/2026.7.14_kaggle/_teacher_branch_upload/teammate_teacher/project`。不能回落队友runs/cache默认目录。
- 当前只修订文档，不训练、commit或push。将来任务完成时先形成可检查的本地差异；是否提交由当时用户授权决定。
- 正式生成不得访问final标签路径；独立可信准备可以读取原规范清单，仅分离输入/标签，不拟合模型或选择参数。
- 资源门槛至少20GiB，并须满足实测峰值加余量；2026-10-02观测D盘46.90GiB，可用性仍须执行前重新查询。
- 所有超参数以规格v2为准，写入配置后先冻结身份，再选参。所有阶段为单seed描述性实验，不承诺0.91或显著性。

## 复核重点（Review Focus）

1. 直接标签字段与类别代理：final公共manifest/缓存/摘要不得含action_name等标签；原始路径只定位文件。
2. 祖先混用：dev必须拒绝refit14教师；refit必须拒绝select学生初始化；A9必须验证A7/A8的本run来源。
3. 同长度错集合/乱序：按opaque ID严格一对一连接，禁止纯位置堆叠或评估。
4. 缺失与时间戳：全mask行必须先验回退，completed不等于available；真实Skeleton时间恢复须接入读取链。
5. 重启与揭示：规范哈希跨进程稳定；revealed之后禁止新生成；smoke必须冻结自己的fixture候选。

## 文件职责（File Ownership）

新文件位于 `src/experiments/`、`scripts/`、`tests/`、`configs/experiments/`。不修改队友快照或提交包。每个任务的文件清单也包含相应测试；测试依赖由Task1的fixtures提供。

| 任务 | 主要模块与职责 |
|---|---|
| 1 | no_vote_protocol.py、no_vote_types.py、teammate_source.py：协议、公共类型、源码验证、公开权重 |
| 2 | no_vote_manifest.py、artifact_record.py：可信准备、ID/标签分离、祖先与产物注册表 |
| 3 | pose_roi_adapter.py：标签可选P28/P29、真实时间戳恢复 |
| 4 | visual_teacher.py：分区无关VideoMAE特征、单Ridge头及targets |
| 5 | pixel_cache.py、visual_student.py、no_vote_datasets.py：像素、固定划分MC3、无标签推理及sequence |
| 6 | motion_cache.py：P31/P86原始动作窗、显式归一化 |
| 7 | imu_rf_teacher.py：原统计特征与单RF、标准IMU targets |
| 8 | mobind_pretrain.py：固定划分运动预训练、分支预测 |
| 9 | simple_fusion.py：温度与可用分支均值、先验 |
| 10 | mobind_fusion.py：四个独立训练对照及推理zero敏感性 |
| 11 | session_repeat.py：固定会话、有限开发网格、A8 |
| 12 | target_adaptation.py：标签无关A9、独立12/40轮端点 |
| 13 | no_vote_evaluation.py：候选冻结、单向状态、ID连接评估 |
| 14 | run_teammate_single_teacher_pipeline.py：依赖编排、预检、闭合smoke、执行记录 |

## 公共接口与数据模式（Shared Interfaces and Schemas）

Task1在no_vote_types.py定义以下类型，后续任务不能重新定义同名但不同字段的类型。Path是已解析绝对路径，ArtifactRef指向record.json及其sha256；所有样本索引使用同一RowIndex。

```python
@dataclass(frozen=True)
class Partition:
    name: Literal["train12", "development2", "refit14", "final4"]
    users: tuple[str, ...]      # 排序后唯一
    expected_rows: int

@dataclass(frozen=True)
class NoVoteProtocol:
    run_id: str
    run_root: Path
    source_root: Path
    partitions: Mapping[str, Partition]
    public_manifests: Mapping[str, Path]
    supervised_labels: Mapping[str, Path]  # 不得有final4键
    weights: Mapping[str, Path]
    recipe: Mapping[str, object]
    def identity(self) -> str: ...

@dataclass(frozen=True)
class ArtifactRef:
    record_path: Path
    sha256: str

@dataclass(frozen=True)
class RowIndex:
    sample_ids: tuple[str, ...]
    user_ids: tuple[str, ...]
    class_ids: tuple[int, ...]  # 概率列的0..39，不是每行标签

@dataclass(frozen=True)
class Prediction:
    index: RowIndex
    logits: np.ndarray         # [N,40]
    probabilities: np.ndarray  # [N,40]
    valid: np.ndarray          # [N]，本分支可用
    artifact: ArtifactRef

@dataclass(frozen=True)
class TeacherTargets:
    prediction: Prediction
    features: np.ndarray | None  # 视觉[N,2,3,1024]，IMU为None

@dataclass(frozen=True)
class Selection:
    stage: str
    config: Mapping[str, object]
    budget: Mapping[str, int]
    metric: Mapping[str, float]
    fit_artifact: ArtifactRef
    development_ids_sha256: str
    def write(self, path: Path) -> None: ...

@dataclass(frozen=True)
class StageInputs:
    partition: Partition
    public_manifest: Path
    labels: Path | None
    parents: Mapping[str, ArtifactRef]

@dataclass(frozen=True)
class AdaptationInputs:
    rows: RowIndex
    base: ArtifactRef
    targets: ArtifactRef
    pixels: ArtifactRef
    sequence: ArtifactRef
    motion: ArtifactRef
    normalization: ArtifactRef
```

ArtifactRecord及ArtifactRegistry由Task2定义：stage/kind/phase、protocol/config/source/weight身份、fit/select/predict/adaptation的users与ID集合、parent refs、normalization/prior refs、row-set hash、class order、相对文件路径及hash。ArtifactRegistry只能解析本run记录或已验证公开权重/源码；递归验证记录与文件，拒绝祖先环、角色错配或未经声明的输入。共享类型中raw数组经验证后才能构造Prediction/TeacherTargets。Task1另导出load_protocol(config: Path) -> NoVoteProtocol，后续CLI统一使用它，不调用未定义的from_yaml或train_users等属性。

数据模式（schema）：
- 公共rows.csv：sample_id/user_id和四原始路径及available字段；源ID同opaque sample_id，不能夹带旧类别ID。
- features.npz：sample_ids、features[N,2,3,1024]、kinetics_logits[N,2,3,400]、valid[N]、class_ids[40]；不含labels。
- targets.npz：sample_ids、class_ids、logits[N,40]、probabilities[N,40]、valid[N]；视觉特征作为独立父产物。旧loader的四头logits与folds被移植后的loader取代。
- pixel目录：images.npy[N,2,16,3,160,160] uint8，最后一个3是IR视野而不是独立RGB通道；view_valid/quality[N,2,16,3]、source_frame_indices[N,2,16]、source_time_seconds、completed[N]、rows.csv。源码MC3内部按灰度复制为Kinetics三色通道。
- sequence目录：sequence.npy[N,2,3,16,512]、rows.csv、completed；identity包含MC3 checkpoint和pixels哈希。移植推理Dataset完全不读teacher/labels。
- motion目录：队友MOTION_FIELDS各自.npy及mask/rows/completed；Skeleton主字段[N,2,16,17,13]、IMU主字段[N,2,16,5,4,16]，points_per_imu_bin=4；IMU bin statistics52通道/global statistics48通道。字段顺序从p86_cached_motion_data.py固定到本run schema。
- normalization.json：channel/role轴、mean/std、有效fit ID集合hash、fit_users；缺失值不计统计，std下限1e-4、应用后缺失位置归零。
- prior.json：按全部规范fit标签计数的40类先验、fit用户/ID及hash。select=train12；final=refit14。
- predictions_unlabeled.npz：sample_ids/class_ids/probabilities/valid；所有正式final候选609条，行和1，全缺失18条为相同refit14先验。

所有数组join先验证ID唯一、精确集合与类别列，再显式重排；公共缓存中未知标签别名直接拒绝。训练Dataset从独立标签表连接监督标签，推理/适配Dataset没有label键。

## 执行方式（Execution Convention）

每个任务按“测试先失败 → 最小实现 → 针对性测试通过 → 保存本地差异”推进。下面是将来需创建的新脚本接口，不是声称当前已存在。运行目录为本工作树，Python为 `D:/Anaconda/envs/PyTorch2.7/python.exe`。脚本建立本仓库import路径，队友模块只由验证过的源码加载器定位；每次调用显式传入本run父产物，不消费历史默认路径。

每个训练CLI统一接受 `--config --phase select|refit|predict --smoke`；任务专有参数列在对应接口。阶段输出采用run根中同名目录的select/refit子目录。select返回Selection及模型ArtifactRef；refit只读取Selection的config/budget，显式接收refit父模型，不读selected checkpoint替代初始化。

smoke是独立fixture protocol/run_id，使用合成小人口和模拟公开权重进行全链闭合；按fixture的expected_rows验收，不能标记formal completed或生成正式准确率。真实一条/一批验收只取train12，不读取final标签；固定40类拟合支持的正式检查不因smoke而被关闭到正式配置。

### Task 1：冻结协议、源码与公开初始化（Protocol, Source and Weights）

**文件：** 新建上述三个模块、configs/experiments/teammate_single_teacher_fixed_split.yaml、scripts/acquire_no_vote_weights.py、tests/test_no_vote_protocol.py、tests/test_teammate_source.py、tests/test_no_vote_weights.py；tests/conftest.py补合成labels/IDs、fake artifact DAG与fixture protocol。

**输入/输出：** 规格v2与源码manifest → NoVoteProtocol、verified_source.json、weights_manifest.json。配置包含全部S中的配方、种子、候选、固定目标池和分区；无需重新访问final标签决定划分。

- [ ] 先写test_protocol_hash_across_processes：PYTHONHASHSEED=1/2/3得到同hash；改变预算、seed、用户、weight/source身份任何一项得到不同hash。
- [ ] 先写test_partitions_exact：train/dev/final两两不交、refit=train∪dev、18用户；test_no_final_label_config拒绝final标签路径；test_source_hash_drift拒绝任意源码字节变更。
- [ ] 先写test_local_weight_directory：本地模型目录绝不作为Hub repo_id；缺修订或hash不符失败。
- [ ] 实现load_protocol及identity的递归规范化：集合变排序数组、Path转规范字符串、dict键排序，序列保留语义顺序，JSON允许有限数值，禁止default=str。recipe包含source_manifest/weights_manifest的内容hash；每次加载先验证实物，不能仅按路径计算协议身份。
- [ ] verify_source逐一验证1167文件，加载root/aligned_multimodal函数前完成；保存使用的符号/源码文件哈希，不执行main。
- [ ] acquisition显式下载公开VideoMAE指定revision、MC3 KINETICS400_V1和YOLO11n-pose，或接受已存在的本地副本；记录完整文件SHA256。运行初始化目录加载并核验旧attention bias恢复。下载和训练是不同步骤。
- [ ] 验证：`python -m pytest tests/test_no_vote_protocol.py tests/test_teammate_source.py tests/test_no_vote_weights.py -v`。实际下载命令为`python scripts/acquire_no_vote_weights.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml`，仅在实施时执行。

### Task 2：独立准备、标签分离与产物验证（Preparation and Provenance）

**文件：** no_vote_manifest.py、artifact_record.py、scripts/prepare_no_vote_inputs.py、tests/test_no_vote_manifest.py、tests/test_artifact_record.py。
**接口：**
`prepare_inputs(source: Path, protocol: NoVoteProtocol, public_output: Path, private_output: Path) -> Mapping[str, Path]`；
`load_stage_inputs(protocol: NoVoteProtocol, partition: str, phase: str) -> StageInputs`；
`ArtifactRegistry.verify(ref: ArtifactRef, expected_stage: str, expected_phase: str, expected_ids: RowIndex) -> ArtifactRecord`。

- [ ] test_final_whitelist：含action_name/class_id的原清单经独立准备后公共字段严格相等白名单，public中没有旧cXX ID，final标签只在private_output。
- [ ] test_generation_never_reads_private_labels：移除/置换私有final标签后重新生成相同fixture预测；文件读取探针拒绝生成进程接触canonical labelled manifest或private目录。
- [ ] test_ancestor_roles：dev拒绝fit refit14的A1/A4目标；refit拒绝select A2/A5；A9只允许本run refit A7/A8，伪装改名bank拒绝。
- [ ] 实现prepare仅在独立CLI执行：读取3036行规范清单，按固定用户生成四分区，opaque ID=sha256固定命名空间+原sample_id，检查碰撞/唯一。先准备public与前三分区label表，再把final标签和旧ID映射写private；public运行配置不保存private_output。
- [ ] 移植原路径解析与available检测，缺失保持规范行；unknown label fields拒绝。metadata后续从原始记录提取，不从action/trial排序推断。
- [ ] 实现ArtifactRegistry完整DAG验证及规范ID索引、class order与stage-role规则；label表只允许前三分区supervised角色；所有公共缓存一律无标签。
- [ ] 验证：`python -m pytest tests/test_no_vote_manifest.py tests/test_artifact_record.py -v`。独立准备命令：
  `python scripts/prepare_no_vote_inputs.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --source-manifest metadata/manifest.csv --public-output outputs/teammate_single_teacher_fixed_split/<run_id>/protocol --private-output C:/Users/LaiWanzhou/AppData/Local/Temp/cuhkx_no_vote_labels/<run_id>`。
  生成入口不自动调用prepare，不自动接收原manifest或final标签。

### Task 3：移植P28/P29并接入时间戳恢复（Pose and ROI Port）

**文件：** pose_roi_adapter.py、scripts/build_no_vote_pose_roi.py、tests/test_no_vote_pose_roi.py。
**接口：** `build_pose_roi(inputs: StageInputs, pose_weights: ArtifactRef, output: Path) -> Mapping[str, ArtifactRef]`，返回p28/p29。只消费公共manifest；inputs.labels即使存在也不传入提取器。
**复用：** audit_yolo11_pose_skeleton、build_adaptive_yolo11_pose_skeleton_cache、build_multiscale_dir_rois的frame/pose/track/ROI算子；迁移其read_rows/safe_name/summary与完成检查，不调用旧main。

- [ ] test_label_free_pose_rows：无class_name、depth_color_usable旧列、斜杠source_id的行仍可处理；缺失行完成但available=false。
- [ ] test_timestamp_alignment真实训练样本：Skeleton counter恢复为IR时间戳后共同帧非空，结果等于提交stage_runner验证过的映射；无匹配时明确标记，禁止按最近标签样本补齐。
- [ ] 在实际移植frame_map调用点接入映射，不在父进程定义未使用helper；ID同opaque值，排序不读class字段，JSON/CSV/NPZ都不输出类别。
- [ ] 写p28必要原字段，p29写ROI；缺ROI使用整帧回退，缺IR不伪造。同步记录帧时间/质量/原因及raw input hashes。
- [ ] 验证：`python -m pytest tests/test_no_vote_pose_roi.py -v`；真实一条仅train12：`python scripts/build_no_vote_pose_roi.py --config configs/experiments/teammate_single_teacher_fixed_split.yaml --partition train12 --max-trials 1`，产物标partial不能伪装正式完成。

### Task 4：移植A1抽取器与单Ridge头（Visual Teacher Port）

**文件：** visual_teacher.py、scripts/run_no_vote_visual_teacher.py、tests/test_no_vote_visual_teacher.py。
**接口：**
`extract_visual_features(inputs: StageInputs, roi: ArtifactRef, weights: ArtifactRef, output: Path) -> ArtifactRef`；
`select_visual_head(train: StageInputs, development: StageInputs, features: ArtifactRef) -> Selection`；
`fit_visual_head(refit: StageInputs, features: ArtifactRef, selection: Selection) -> ArtifactRef`；
`predict_visual_teacher(model: ArtifactRef, features: ArtifactRef, rows: RowIndex, prior: ArtifactRef) -> TeacherTargets`。

- [ ] test_arbitrary_label_free_population：N=1/388/609的模拟rows不要求detail_selected/class_id，不先断言1384再max-trials；shape分别[N,2,3,1024]/[N,2,3,400]。
- [ ] test_single_selected_head：72grid按S规则决胜，只选一个head；生成targets只含标准键与40类顺序，不要求旧四头、labels/users/folds。
- [ ] 移植P46的prepare_trial/模型编码/聚合，与P85 feature_sets、Ridge训练数学；禁用label读取、旧complete-count和P85历史缓存reuse。窗口0–.70/.30–1，各16帧，与pixels/motion同步；冻结公开骨干。
- [ ] select模型fit train12预测train/dev，下游dev只用此头；selection记录配方/预算/开发ID。refit头重新fit refit14，预测refit/final。
- [ ] 原始六clip features可共用；分类头、概率及监督祖先分开。缺IR预测先验且valid=false，KL/feature蒸馏loss按valid屏蔽。
- [ ] 验证：`python -m pytest tests/test_no_vote_visual_teacher.py -v`；一条编码确认模型键和FP16数值，不执行72grid全训练直到阶段验收。

### Task 5：补像素缓存、固定划分A2和无标签sequence（Pixels, Student and Sequence）

**文件：** pixel_cache.py、visual_student.py、no_vote_datasets.py、scripts/build_no_vote_pixels.py、scripts/run_no_vote_visual_student.py、scripts/build_no_vote_sequence.py、tests/test_no_vote_pixels.py、tests/test_no_vote_visual_student.py、tests/test_no_vote_sequence.py。
**接口：**
`build_pixels(inputs: StageInputs, roi: ArtifactRef, output: Path) -> ArtifactRef`；
`train_visual_student(phase: Literal["select","refit"], fit: StageInputs, development: StageInputs | None, pixels: ArtifactRef, teacher: TeacherTargets, selection: Selection | None) -> tuple[ArtifactRef, Selection | None]`；
`build_sequence(model: ArtifactRef, pixels: ArtifactRef, rows: RowIndex, output: Path) -> ArtifactRef`。

- [ ] test_pixels_contract精确检查Shared Schema，source frame index/time与early/late源帧一致；view axes不是RGB渠道；raw completed可以含不可用行。
- [ ] test_inference_without_teacher：删全部teacher/labels文件后学生推理和sequence仍运行；test_select_refit_teacher_roles按Task2DAG拒绝错误教师。
- [ ] test_mc3_native_sequence：非恒定真实clip得到[B,2,3,16,512]，与原encode_backbone_sequence一致；不得重复池化向量伪造时间；允许原源码线性插值。
- [ ] 移植build_p86_visual_pixel_cache的cache结构、P86VisualPixelDataset增强、MC3模型/训练体。移除split_universe、历史2914断言、all-label旧入口；Dataset接受标准TeacherTargets，阶段logit辅助项固定关闭，不构造虚假旧多头/OOF目标。
- [ ] A2配置完全采用S7；select每轮评价全388并选择1–18 epoch，保存统一Selection；refit重新公开初始化按选中epoch训练。保留原hybrid CE/KD/relation，enable_distillation_projection=false；feature-weight0.5只记配置值、实际直接feature loss为0，不能将它误接入hybrid。test_hybrid_active_losses核对该行为与源码一致。
- [ ] 推理Dataset只接受pixels/rows/masks，sequence为select/refit各自构建；两者checkpoint身份不同不能复用。原始pixels建立一次，分区indices引用。
- [ ] 验证：`python -m pytest tests/test_no_vote_pixels.py tests/test_no_vote_visual_student.py tests/test_no_vote_sequence.py -v`；真实train12一批forward/backward与一条sequence验收，记录显存与用时。

### Task 6：移植P31/P86运动窗及拟合归一化（Motion Port and Normalization）

**文件：** motion_cache.py、scripts/build_no_vote_motion_cache.py、tests/test_no_vote_motion_cache.py。
**接口：**
`build_motion_source(inputs: StageInputs, p28: ArtifactRef, output: Path) -> ArtifactRef`；
`build_motion_windows(inputs: StageInputs, p31: ArtifactRef, pixels: ArtifactRef, output: Path) -> ArtifactRef`；
`fit_normalization(motion: ArtifactRef, fit: StageInputs) -> ArtifactRef`；
`apply_normalization(motion: ArtifactRef, state: ArtifactRef, rows: RowIndex) -> Mapping[str, np.ndarray]`。

- [ ] test_p31_reads_p28_fields：必须读P28 frame_ids/raw Skeleton，不接受只有P29 ROI的假父产物。
- [ ] test_window_alignment：读pixels/source_frame_indices与原时间；更换embedding数值不能改变原始运动窗；两种缺设备/缺骨骼mask正确。
- [ ] test_normalizer_scope：dev/final数据极端扰动不改变select均值；refit明确train∪dev；无有效值时mean0/std1，缺失输出0。
- [ ] 移植P31原始字段与P86运动window函数。不会把不存在的--p29-run、--sequence-cache或--normalization传给旧CLI；normalization单独实现并纳入Dataset/记录。
- [ ] 固定五角色、13 Skeleton与16 IMU通道、points_per_imu_bin=4；记录MOTION_FIELDS顺序、52/48统计通道与源码版本；有label字段的旧rows/summary移除。
- [ ] 验证：`python -m pytest tests/test_no_vote_motion_cache.py -v`；一个train12跨模态样本对比像素帧与运动时间窗。

### Task 7：单RF IMU教师与A4→A5契约（RF Teacher Contract）

**文件：** imu_rf_teacher.py、scripts/run_no_vote_imu_rf.py、tests/test_no_vote_imu_rf.py。
**接口：** `build_imu_statistics(p31: ArtifactRef, rows: RowIndex) -> ArtifactRef`；
`train_imu_rf(phase: str, inputs: StageInputs, statistics: ArtifactRef) -> ArtifactRef`；
`predict_imu_teacher(model: ArtifactRef, statistics: ArtifactRef, rows: RowIndex, prior: ArtifactRef) -> TeacherTargets`。

- [ ] test_seeded_device_dropout：复制拟合标签与样本后每行删除一个存在设备，角色/掩码同步、同seed一致；无设备行不训练。
- [ ] test_rf40_and_valid：probabilities[N,40]按classes_对齐，logits=log(clip后归一化概率)，valid=false不会因先验概率有效而变true。
- [ ] test_a4_to_a5_loader：标准targets经新motion Dataset显式映射imu_logits/valid，缺IMU的teacher KL为0；没有valid键直接失败。
- [ ] 使用P31保留原始无损IMU重采样32bin，复用run_imu_stat_baseline.feature_vector/drop_devices，240统计维+10mask维，不能直接套P86学习embedding。
- [ ] S9单一RF配方，选参候选只有一个，dev成绩仅记录；select/refit分别拟合允许人口，seed20260811，原样本+dropout副本；不创建ensemble或读取旧OOF文件。
- [ ] 验证：`python -m pytest tests/test_no_vote_imu_rf.py -v`；RF接口smoke使用至少支持40类的合成统计fixture。

### Task 8：固定划分MoBind预训练（Motion Pretraining Port）

**文件：** mobind_pretrain.py、scripts/run_no_vote_mobind_pretrain.py、tests/test_no_vote_mobind_pretrain.py。
**接口：** `train_motion_student(phase: str, fit: StageInputs, development: StageInputs | None, motion: ArtifactRef, normalization: ArtifactRef, visual_teacher: TeacherTargets, imu_teacher: TeacherTargets, selection: Selection | None) -> tuple[ArtifactRef, Selection | None]`。

- [ ] test_missing_losses_zero：无IMU的CE/teacher/reconstruction/local项有效掩码为0，无Skeleton同理。
- [ ] test_motion_branch_output：原forward(motion,mask_ratio)产生两个原生branch logits，禁用不相关mask不能改变独立分支；不调用不存在的enabled_modalities参数。
- [ ] 从train_p86_mobind_pretrain迁移batch损失与模型，绕过历史计数和OOF入口；visual logits映射标准single-head目标，features按view-mean保留[N,2,1024]语义，IMU目标用显式valid。
- [ ] 参数采用S10，select1–24 epoch用combined motion在全388上选预算；refit从相同随机seed原始初始化训练选中预算，不沿用select权重。normalization和两teacher全部切到refit身份。
- [ ] 输出A5-S/I/combined、分支valid、统一fallback及Selection。S/I是共享预训练模型独立分支，不声称两次单模态监督训练。
- [ ] 验证：`python -m pytest tests/test_no_vote_mobind_pretrain.py -v`；两个train12批次核查全部有效损失与mask，不读取final标签。

### Task 9：校准和简单融合（Calibration and Simple Fusion）

**文件：** simple_fusion.py、scripts/run_no_vote_simple_fusion.py、tests/test_no_vote_simple_fusion.py。
**接口：** `select_calibration(fit_predictions: Mapping[str,Prediction], fit_labels: Path, dev_predictions: Mapping[str,Prediction], dev_labels: Path) -> Selection`；
`combine_available(branches: Sequence[Prediction], prior: ArtifactRef) -> Prediction`。

- [ ] test_all_missing_prior：三个mask均0返回prior，非零行和1；混合两可用分支得到等权概率。
- [ ] test_join_permutation：teacher ID乱序后显式重排结果相同；重复/少ID/同长度错集合/错40列顺序失败。
- [ ] 拟合每branch温度边界.25–4；仅比较整体identity温度策略与整体fit温度策略两个候选，按dev的A6-VSI评分选策略，同策略用于VS/VI。selectfit只用train12；refit策略为fit时只用refit14重估T。
- [ ] 补齐A6-VS/VI/VSI全609预测；先验与分支valid独立。in-sample温度记录为校准诊断，不作OOF声明。
- [ ] 验证：`python -m pytest tests/test_no_vote_simple_fusion.py -v`。

### Task 10：A7与独立匹配训练对照（Fusion and Training Controls）

**文件：** mobind_fusion.py、scripts/run_no_vote_mobind_fusion.py、tests/test_no_vote_mobind_fusion.py。
**接口：** `train_fusion(phase: str, variant: Literal["full","mask_motion","shuffle_s","shuffle_i"], inputs: StageInputs, visual: ArtifactRef, motion: ArtifactRef, sequence: ArtifactRef, teacher: TeacherTargets, normalization: ArtifactRef) -> ArtifactRef`；
`predict_fusion(model: ArtifactRef, inputs: StageInputs, variant: str) -> Prediction`。

- [ ] test_matched_controls_have_distinct_training：四variant有独立model/checkpoint/training-record，但初始化值、配置、预算4+20和batch顺序相同。
- [ ] test_shuffle_features_and_masks：每epoch donor permutation只含fit人口，特征与mask一起移动，接收行标签与teacher不移动；eval置换由sorted IDs/seed固定，与batch划分和输入顺序无关。
- [ ] test_refit_parents：select加载A2/A5 select，refit加载A2/A5 refit；模型加载严格验证config/state keys，而非修改stage字符串欺骗旧入口。
- [ ] 迁移fusion_proxy的建模/损失/两阶段训练体，不跑split_universe或固定1497/973/444及2914主入口。S12配方固定，不用控制的dev分数改变full预算。
- [ ] 冻结视觉backbone参数及BatchNorm运行统计，每次model.train后将backbone保持eval；冻结预训练motion encoder，visual head可更新。sequence依赖不变backbone，记录其身份；test_cached_backbone_unchanged同时核对参数与buffer哈希。四variant各做select和refit；mask时两运动分支均置零，shuffle按S12定义。
- [ ] 附加A7-zero-S/I只在训练完成full模型上推理，是敏感性分析而非匹配对照；与独立训练control目录分开。
- [ ] 验证：`python -m pytest tests/test_no_vote_mobind_fusion.py -v`；一批full/mask/shuffle数据检查真实损失和参数冻结。

### Task 11：冻结会话与重复处理（Session and Repeat Processing）

**文件：** session_repeat.py、scripts/run_no_vote_session_repeat.py、tests/test_no_vote_session_repeat.py。
**接口：** `build_sessions(inputs: StageInputs, metadata: ArtifactRef) -> list[tuple[str,...]]`；
`select_session_operator(train: StageInputs, development: StageInputs, train_probability: Prediction, dev_probability: Prediction, metadata: ArtifactRef) -> Selection`；
`apply_session_operator(inputs: StageInputs, probability: Prediction, transition: ArtifactRef, selection: Selection, prior: ArtifactRef) -> Prediction`。

- [ ] test_no_cross_partition_user_date_edges：相同timestamp但不同用户/日期不连边；已知时间的全缺失行断开相邻transition；无法定位日期/时间的行不建立边，不推测其位置，输出不变；gap>30断开，opaque ID解决同时间排序。
- [ ] test_repeat_needs_two_peers：只有一个同意peer不改；所有missing18不得作为peer或被覆盖；概率/metadata同步乱序后结果按ID一致。
- [ ] 从原录制元数据函数移植时间提取；会话known_user加partition键，邻接gap用start差，repeat限制同会话；不按动作/旧trial编号/类别路径构建图。
- [ ] 复用audit_p87_sequence_decoder.fit_transition_model与decode_unique_beam_posterior：start/end、bigram、backed-off trigram、alpha=.25、beam50、posterior温度1及无重复类约束。test_structured_posterior_matches_source在合成合法会话比较完整40列边缘概率；len>40/无合法路径时整会话回退，禁止按动作截断。结构posterior.max>=.80才应用，否则A7不变。先transition后repeat，repeat用冻结输入一次评估，满足至少两个高置信同意peer、duration tolerance、cosine门控与margin才更新，否则返回当前emission；更新为接收概率与同意peer均值各0.5，margin比较peer同意类概率与接收该类概率。
- [ ] S13有限网格先记录protocol，dev比较identity/transition-only/repeat-only/combined，决胜accuracy、macroF1、argmax改变行更少、候选ID；refit transition统计换为refit14，阈值/算子不重选。
- [ ] 输出A8 targets与metadata/transition/selection完整父记录，禁止final手工选择。
- [ ] 验证：`python -m pytest tests/test_no_vote_session_repeat.py -v`。

### Task 12：移植标签无关A9（Label-Free Adaptation Port）

**文件：** target_adaptation.py、scripts/run_no_vote_target_adaptation.py、tests/test_no_vote_target_adaptation.py。
**接口：** `adapt_target(inputs: AdaptationInputs, endpoint: Literal[12,40], output: Path) -> ArtifactRef`；
CLI新增 `--endpoint 12|40`，无final-label或eval-accuracy参数。

- [ ] test_targets_require_a8_ancestry：改名历史targets不能通过；只接受本run、refit14 A7祖先的冻结final4 A8，609精确ID，class order0–39。
- [ ] test_591_loss_609_output：联合609池适配，18全缺失target_mask=false；591有至少一模态进入loss，不按confidence筛选，输出仍609且18为prior。
- [ ] test_adaptation_parameter_scope：visual_head_parameters及motion_residual参数可训练，视觉backbone参数与BatchNorm运行统计冻结并保持eval；12/40都从同A7独立初始化，40不能从12续训。
- [ ] 复用LabelFreePseudoDataset、train_label_free和configure_adaptation_parameters的算法，移植build_model/load paths/Dataset；删除历史stage==all2914与401/405条件，不引入旧P310 structured字段默认加载。
- [ ] 输入强制pixel/sequence/motion/normalization全部明确；使用无标签batch，遇label键立刻失败；targets标准概率映射为pseudo_probability，confidence_power0、warmup0，无epoch准确率选择。
- [ ] 参数采用S14：12轮fusion/motion5e-5、visual head1e-5、min2e-6；40轮fusion/motion1e-4、visual head1e-5、min5e-6；AdamW wd.02、T1、batch64、对应seeds。scope=heads_motion_encoder是明确复刻变体。
- [ ] A9-12/40直接输出raw概率，不重新A8；主报告将A9−A8称deployment replacement，并另报A9−A7。
- [ ] 验证：`python -m pytest tests/test_no_vote_target_adaptation.py -v`；两个合成无标签批次确认梯度scope，真实接口验收只在train12模拟target，不能查看final准确率。

### Task 13：完整冻结与一次揭示（Freeze and Reveal）

**文件：** no_vote_evaluation.py、scripts/freeze_no_vote_generation.py、scripts/evaluate_no_vote_ablation.py、tests/test_no_vote_evaluation.py。
**接口：** `freeze_generation(protocol: NoVoteProtocol, registry: ArtifactRegistry, candidates: Mapping[str,ArtifactRef]) -> Path`；
`evaluate_frozen(generation: Path, private_labels: Path, output: Path) -> Path`。
状态记录run_state.json，transition为prepared→generating→frozen→revealed，不能倒退。

- [ ] test_freeze_exact_candidates_ids：必须有S15全部16候选、609 ID、40列、有限行和1、18同prior；缺候选/换row set/parent drift/fallback drift都失败。
- [ ] test_evaluator_id_join：private_labels乱序结果相同，duplicate/missing/extra拒绝；不以expected_rows=609替代ID验证。
- [ ] test_revealed_blocks_generation：revealed后generate/refit/adapt/freeze全拒绝；同generation hash的评估可幂等重算，失败后不解锁训练。
- [ ] generation_complete保存candidate ID、可恢复相对路径、record/hash和DAG闭包，不只保存name→hash；不含final标签内容或其文件hash。
- [ ] evaluator先核验generation、持久化revealed状态，再打开用户显式提供的private标签；标签只进入evaluation目录。揭示后的剩余过程仅输出S15指标，不增加候选。
- [ ] 验证：`python -m pytest tests/test_no_vote_evaluation.py -v`；独立评估命令为
  `python scripts/evaluate_no_vote_ablation.py --generation outputs/teammate_single_teacher_fixed_split/<run_id>/generation_complete.json --labels C:/Users/LaiWanzhou/AppData/Local/Temp/cuhkx_no_vote_labels/<run_id>/final_labels.csv --output outputs/teammate_single_teacher_fixed_split/<run_id>/evaluation`。

### Task 14：依赖编排、预检与闭合smoke（Orchestration and Acceptance）

**文件：** scripts/run_teammate_single_teacher_pipeline.py、tests/test_no_vote_orchestrator.py、docs/teammate_single_teacher_fixed_split.md。
**接口：** `run_pipeline(mode: Literal["preflight","smoke","generate","freeze"], config: Path) -> None`；evaluate保留独立CLI。测试使用同一两参数签名，不另造第三参数接口。

- [ ] test_orchestrator_dependency_order：全链遵循下表select/refit顺序，A9目标从A8来；任何父身份变化拒绝续跑。
- [ ] test_smoke_reaches_frozen：smoke使用fixture人口/目标池走全部16候选直到freeze，而不是止于adaptation40；formal入口拒绝fixture缓存/权重/Selection。
- [ ] test_preflight_resources：查询当前磁盘、GPU和env；预计峰值+至少20%余量不足则只停止预检；不把2026-10-02空间观测当永久可用。
- [ ] preflight报告来源/权重清单、公共输入/标签边界、拟合支持、cache字节和多branch检查点峰值；真实batch再记录用时/显存，修订旧10–20小时估计。
- [ ] 明确prepared状态只能由可信准备提交，generate不能自行读source labelled manifest。模式顺序：preflight检查 → 各task针对性验收 → fixture smoke frozen → formal generate → formal freeze；只有独立evaluator能揭示。
- [ ] 验证新测试全集：`python -m pytest tests -k "no_vote or artifact_record or teammate_source" -v`。此命令按pytest节点关键词选择新模块，避免依赖Windows shell展开文件通配符。
- [ ] 实施完成后可运行：
  `python scripts/run_teammate_single_teacher_pipeline.py preflight --config configs/experiments/teammate_single_teacher_fixed_split.yaml`；
  `python scripts/run_teammate_single_teacher_pipeline.py smoke --config configs/experiments/teammate_single_teacher_fixed_split.yaml`；
  generate/freeze同接口，正式长训练仅在预检、阶段验收和smoke都通过后开始。
- [ ] 保存协议身份、配方、实际支持/缺失、stage速度、resume历史和验收结果。文件职责/接口与本计划保持一致，不能用通过mock测试替代真实schema接口验收。

## 全链依赖图（Dependency and Ancestry Schedule）

| 顺序 | select链 | refit/final链 |
|---|---|---|
| 输入 | 独立准备train12/dev公共输入与标签 | 独立准备refit14/final4，final标签进入vault |
| 无学习缓存 | A0 raw/geometry/pixels/P31/P86按opaque ID | 同原始缓存引用，不能复用已学习统计 |
| A1/A4 | fit train12，预测train/dev | 同冻结配方重fit refit14，预测refit/final |
| A2/A5 | 公开/随机初始化 + train12教师 + select normalization | 公开/随机重启 + refit14教师 + refit normalization |
| A6 | 仅train预测拟温，dev选identity/fitted策略 | refit预测拟温，同策略，不看final |
| A7与3训练control | A2/A5 select初始化，监督fit train12，dev诊断 | A2/A5 refit初始化，fit refit14，final预测 |
| A8 | transition fit train12，dev选有限算子 | transition refit14，同阈值应用final4 |
| A9 | 只做来源与无标签接口验收，不用final标签选预算 | A7 refit独立启动两次，A8 fixed targets，final4联合无标签池 |
| 冻结/揭示 | fixture全链单独闭合 | 全16候选冻结609 ID后一次揭示 |

## 修订项验收映射（Audit Disposition）

| 审计项 | 修订任务/规格 |
|---|---|
| M1 匹配对照 | Task10；S12独立train/refit与zero敏感性分开 |
| M2 A9解释 | Task12/13；S14/15明确raw deployment replacement |
| M3 转导单位 | Task11/12；S13/14固定会话与联合目标池 |
| M4 组合/统计语言 | Task13；S1/15描述性单seed、ROI/腐败不单独归因 |
| L1/E1 清单与标签 | Task2/3；白名单、opaque ID、可信准备与生成分离 |
| L2/E2 旧提取器标签/计数 | Task3/4/5/6；明确port，不调用main |
| L3 监督祖先 | Task2/4/5/7/8/10；上表完整select/refit DAG |
| L4 概率与ID/缺失 | Task2/9/13；按ID连接、全mask先验 |
| L5 会话边界 | Task11；known_user+partition，禁止类别分组 |
| L6 冻结与揭示 | Task2/12/13/14；目标来源、16候选、单向状态 |
| E3 公开权重 | Task1；取得/revision/hash、本地目录分支 |
| E4 像素/视觉Dataset | Task5；cache/targets/推理/sequence均新接口 |
| E5 运动父输入/归一化 | Task6；P28+pixel source frame indices |
| E6 MoBind历史人口 | Task8/10；固定划分训练体与统一Selection |
| E7 A4→A5格式 | Task7/8；logits/valid与实际loader契约 |
| E8 A9历史stage/401/405 | Task12；AdaptationInputs包含sequence，新609/591契约 |
| E9 时间戳接入 | Task3；真实调用点与train样本验收 |
| E10 哈希/类型/smoke | Task1/14；排序序列化、共享类型、闭合freeze |

## 自检与下一步（Self-Review and Next Step）

规格各项可定位到Task1–14；新接口类型均在Task1/2定义，训练来源和缓存身份见依赖表。旧CLI参数错误已从执行路径移除，剩余源码函数只是需移植的参考。测试步骤均对应具体缺陷和断言，不声称现已实现或已通过。

此v2已完成文档修订，需要针对修改项做静态复审；独立审计通过之前不启动正式训练。当前用户授权为修订报告暴露的问题，未据此自动commit/push或开始GPU实验。后续实施按本计划先做协议/来源和无标签接口，再做模型任务。
