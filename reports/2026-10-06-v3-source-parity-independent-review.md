# v3 源操作一致性独立复审（Independent Source-Parity Review）

结论：**NO-GO**。当前规格（Specification）和实施计划（Implementation Plan）有 **1 项 P1 明确偏差，0 项已证实 P2 数值偏差**：十项 mechanism 专家的生产模型被写成 MC3 学生，实际源 producer 是冻结 VideoMAE-Large 特征上的固定 Ridge 分类头。此错误会改变 A8 教师库（Teacher Bank）及 A9 伪标签（Pseudo Labels），须先修订文档再实施对应步骤。

复审基准为工作树 HEAD `8d1f3943b05b76e81d71e5163b5ec82537219301`，最终 `(1)` 包的实际命令／调用路径及公开模型元数据（Public Model Metadata）。未执行训练、预测或测试套件；未读取私有 final4 标签、旧 final 结果、历史预测数组；未计算 final 分数，也未执行原始数据／缓存 SHA 扫描。下面的“吻合”仅表示文档要求与读取的源操作一致，不代表尚未实现的 v3 已通过数值／梯度验收。

固定划分（Fixed Split）替代所有 OOF、仅保留单个大视觉根节点（Large Visual Root），属于用户明确允许的差异。本次不要求恢复三折、内部 OOF、原 cohort 人口或按其标签拟合阈值。train12、dev=user6/7、refit14 和封存 final4 的边界作为既定条件；类别覆盖仅引用已有独立布尔核实，没有重复打开 final 标签。

## P1：mechanism 的模型与特征来源错误（Wrong Mechanism Producer）

**计划位置：** [计划第114行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-06/pre-review/plan-v3.md:114) 要求“恢复同学生十项mechanism变换输出”，并写“不新增训练模型”；[计划第18行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-06/pre-review/plan-v3.md:18) 同样称“同学生mechanism预测”。[规格第94行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-06/pre-review/spec-v3.md:94) 更明确写“同一MC3的十项mechanism预测”。

**源证据（Source Evidence）：**

- [audit_p86_teacher_mechanisms.py:25](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)/code/training/project/aligned_multimodal/audit_p86_teacher_mechanisms.py:25) 固定 `FROZEN_ALPHA=3000`、`FROZEN_CLASS_WEIGHT_POWER=0.75`。[第174行](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)/code/training/project/aligned_multimodal/audit_p86_teacher_mechanisms.py:174) 对 VideoMAE 六 clip 特征逐 clip L2 归一化（L2 Normalization），第208–225行展平为 early/late 全视野特征，重新拟合 `make_model(FROZEN_ALPHA)`，用 fit 人口归一化特征均值做干预，再调用 `aligned_scores_40`。没有 MC3 forward。
- [fixed_model_perturbations:101](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)/code/training/project/aligned_multimodal/audit_p86_teacher_mechanisms.py:101) 的 baseline、drop scene/person/workspace/early/late、swap early/late、collapse early/late、swap person/workspace、collapse view identity 均直接操作 `[N,2,3,1024]` 特征。
- 测试侧 producer [p89_build_multiexpert_test_logits.py:162](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)/code/training/project/aligned_multimodal/p89_build_multiexpert_test_logits.py:162) 的 `build_p86` 第163–173行同样读取 VideoMAE train/test features，逐 clip L2，固定 Ridge 重新拟合，fit 特征均值干预，输出 raw logits；**没有 head temperature 校准**。
- [p88_oof_candidate_ensemble.py:50](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)/code/training/project/aligned_multimodal/p88_oof_candidate_ensemble.py:50) 第50–65行直接列入这些 `fixed_model_predictions.npz` 十项 logits；[p89_full40_scale_invariant_transfer.py:120](D:/work/2026.7.14_kaggle/CUHK-X_Small_Model_Submission(1)/code/training/project/aligned_multimodal/p89_full40_scale_invariant_transfer.py:120) 第120–128行在目标侧按原 logits softmax 加入 bank。并非一个可任意替换成 MC3 的诊断名称。

**影响（Impact）：** 即使保留十项同名输出，MC3 图像／序列干预也无法复现源 bank 列。将 Task4 通过 dev 选中的 early_late 头直接当 mechanism 头，也不保证一致：源 mechanism 的 alpha/power 固定，输出不除 Task4 的 `T_head`。此差异既不是删掉多大根节点的必要后果，也不是固定划分替代 OOF 的必要后果。

**最小修订（Minimal Revision）：** 把上述三处改为“同一 VideoMAE-Large 六 clip 特征的固定 Ridge mechanism 头”，移到 Task4 或 Task11 producer；允许独立拟合这个小分类头，仍只有一个大视觉骨干。显式锁定 `StandardScaler→Ridge(lsqr,tol=1e-5,max_iter=5000), alpha=3000, class_weight_power=0.75`、逐 clip L2、fit 人口特征均值、十项原变换、`aligned_scores_40` 和 raw logits/T=1。固定划分下只在本 phase fit 人口拟合头和均值；不生成 OOF、不加载历史预测。增加 fixture 对照 train/test 两条 producer 的 logits，以及 producer 不依赖 MC3 的断言；同步更新 roster、依赖顺序与成本预检。

## 已核对吻合的主链操作（Checked Main-Chain Operations）

| Task | 文档要求与源证据 | 判断 |
|---|---|---|
| 4 | 六族特征含逐 clip L2、window mean 二次 L2、kinetics row standardization；每族 power 0/0.5/0.75 × alpha 300/1000/3000/10000；accuracy→balanced accuracy→macro-F1→较小power→较小alpha。源 `train_p85_videomae_full40_head.py:85–104,149–150,194–215`；缺类 floor 第61–70行；`train_p46_videomae_head.py:94–108,128–138` 的 scaler/Ridge 和 log-T ±2.302585。主 Dataset 实际读取 early_late：`p86_visual_pixel_data.py:43–46`。 | 规格第42–50行、计划第98–103行吻合；fit 预测替代 OOF 与 dev 选参为获准变体。 |
| 5 | 最终调度器 `run_p87s_final_pipeline.py:149–192` 覆盖 hybrid、mc3_18_temporal、layer2、16帧160、LR2e-4/1e-5、wd0.08、KD1、relation0.2、smoothing0.1、subject_robust、batch4/accum4、固定16轮。`train_p86_visual_pixel_oof.py:474–523` hybrid 不计算有效 feature/stage KD；第409–415行源 warmup/cosine；第985–996行 all-label 路径重新 seed、固定预算、不 early stop。公开最终 metadata 中宽度512、dropout0.18、gated及所有额外时空开关与规格一致。 | MC3 训练配方吻合；仅 mechanism 要求存在上述 P1。`“minimum1e-5”` 是 LR 调度参数，源按相同 ratio 缩放两组，不能误实现成每组绝对下限1e-5。 |
| 7 | `run_imu_stat_baseline.py:70–101,150–157,193–215` 的存在设备删除、标签副本、400/depth18/leaf2/sqrt/balanced_subsample、RF seed20260723、预测 log clip；`evaluate_imu_oof.py:372–374` 原样导出 imu logits。 | A4 不额外校准的要求吻合；缺类 sentinel 与 phase dropout RNG 仍需下节锁定。 |
| 8 | `train_p86_mobind_pretrain.py:67–112` 默认结构／权重；最终调度器第236–242行启用 RF KD weight1、24轮。`p86_mobind_lite_data.py:86–93` 原 features 直接 view mean；源 losses 第388–399、451–469行对分支 CE／视觉KD／feature loss不作新增可用性门控，RF KD 第472–479行用 teacher valid。 | 规格第76–80行、计划第143–145行吻合。 |
| 10 | 最终调度器第262–281行 separate、4+20、冻结 pretrained encoders、KD1、anchor0.3、reliability0、corruption0.75/dropout0.3/view0.4；源 `set_stage:774–789`、`train_stage:1039–1045,1117–1155` 两阶段参数组及 Dropout 模式；`losses:869–876` motion_aux全batch，`selective_anchor_loss:798–806` 仅anchor正确行。 | 规格第86–88行、计划第162–165行吻合；必须使用原参数组名单与逐阶段新 optimizer，不能仅比总 LR。 |
| 11 | `p134_frozen_repeat_consensus.py:31–39` CONFIG；`p137_group_classifier_selector.py:31–158,161–197` sqrt/full/full、peer2、C0.03、frequency0及40列类对齐。`p270_fixed_emission065_transition045.py:13–17` 使用 P255概率和预测，emission0.65、transition0.45、alpha1、gap30/trigram1/beam50；`p139_soft_sequence_gate.py:63–129` 五评分、401网格、七分位数与net/rescue/harm/changed决胜。`p310_union_repeat_precedence_teacher.py:56–68` 是 `new!=old` 覆盖 sequence，再0.9805+39×0.0005锐化。 | 规格第96–102行、计划第174–177行明确值与规则吻合；producer/逐行fallback的完整操作尚未落盘证明。 |
| 12 | 最终包 `docs/TRAINING_PIPELINE.md:61–65` 的40轮、LR1e-4/5e-5/min5e-6、wd0.02、T1/confidence0、scope heads_motion_encoder、seed20260826；`adapt_p87s_test_student.py:45–59,83–85` 给12轮默认和warmup0。`configure_adaptation_parameters:222–239` 40范围包含整个 motion_residual.encoder；`train_label_free:275–329` 余弦、KL与clip-grad1。最终 `code/infer_student.py:42` 直接logits.argmax，没有A8回流。 | 规格第106–108行、计划第185–189行吻合。 |

## 必须在 Task1 产物中锁定的空缺（Unresolved Operation Locks）

下列是当前未完整展开的操作证据，不把“Task1尚未实现”本身列作代码 bug，也不把允许的固定人口差异另列 P1/P2。它们不能仅靠“复制 source”四字宣称已验证。

1. **RF 的缺类值与阶段 RNG（Missing-Class Fill and Phase RNG）：** [计划第133–135行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-06/pre-review/plan-v3.md:133) 只锁定实际 classes 的 log clip，未给缺类 sentinel；源训练目标 `run_imu_stat_baseline.py:178` 是 `-1e6`，目标侧 `train_final_imu_rf.py:117` 是 `log(1e-12)`。前者 dropout RNG 第194行是 seed+fold，最终 refit 第83行是 seed。无 folds 的新 phase 必须明确这两类 consumer 采用哪条源路径及映射，禁止补虚构fold；fixture加入缺类列和选中设备逐项比较。A4目标不加温度并不代表 A8消费RF时也必须 T1。
2. **保留bank的 base、availability和fallback（Base / Availability / Fallback）：** `p165_deployable_group_teacher.build_test_bank:229–242` 对RF使用 **T3**，以 **0.95 base+0.05 RF** 构造 safe probability；缺detail visual行的各full40列使用该safe probability。`p309_union_repeat_group_test.py:22` 的独立专家 unavailable行也回退bank第0列，第23行group route另AND `visual_available`；P255第18行同样如此。[计划第172–177行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-06/pre-review/plan-v3.md:172) 未显式列出这些值／门控，Task1应逐consumer记录并测试“正常行、缺IR但有motion、某专家缺输入、全缺失”。源允许的逐行fallback与“复制假专家凑列”应明确区分，不能用后者拒绝条件删掉前者。
3. **全窗邻居不是仅 feature 算子（Full-Window KNN Dependency）：** [计划第102行](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/audit_inputs/2026-10-06/pre-review/plan-v3.md:102) 指名源KNN但未展开父融合。`build_p85_fullwindow_knn_teacher_v3.py:23,26–27,67–74,102–120` 是源全局融合概率的3邻居、weight0.40平滑；`build_p85_multiexpert_submission_v1.py:29–47,83–101` 父节点是8个P12输出与6个VideoMAE头的log概率simplex融合（L-BFGS-B,maxiter500,ftol1e-12），不是对A1 early_late直接做KNN。Task1需要继续展开P12与safe-base实际producer及公开初始化、每个保留／排除理由；若删除大根导致某派生专家不可保留，按已授权规则排除并记原因，不能悄悄换父概率。

未发现以固定 val 替代 OOF、重新拟合 fit-only温度／阈值、known user/date分区会话隔离、全缺失 prior／target_mask适配本身违反当前授权。本报告没有要求这些步骤回到原人口或内部OOF。空disagreement时identity返回为文档显式适配，应同时覆盖group的 `choose_threshold` 和sequence的 `select_gate`，原函数在quantile前没有空数组guard。

## 放行条件（Release Condition）

先修订 P1 mechanism producer 的上述文档要求；Task1实际导出的 source_ops_manifest／teacher_roster 必须覆盖固定Ridge头、safe-base、P12→全局融合→全窗KNN以及各consumer缺类／缺输入语义，再执行计划已有的小fixture数值／梯度对照。当前没有足够证据把尚未生成的完整保留bank称为已证明的操作一致性，故不建议按现文档直接进入对应正式训练阶段。
