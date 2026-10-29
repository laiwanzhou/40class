# P33 Task8 MoBind运动预训练（MoBind Motion Pretraining）

日期：2026-10-29，Asia/Shanghai。唯一指导为[P01](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/superpowers/plans/P01_teammate_single_teacher_fixed_split.md)，唯一交接为[P02](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/docs/handoffs/P02_current_handoff.md)。本文仅为实施与验收记录（Implementation and Acceptance Record），不另建计划或交接。

状态：**Task8已完成两阶段固定24轮训练、四区输出及有限自验（Complete with Bounded Self-verification）**。本轮没有请求或执行独立子智能体复审，不称独立GO；Task9未启动。

## 生效模型与源配方（Effective Model and Source Recipe）

新增 [mobind_pretrain.py](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/mobind_pretrain.py)、[训练CLI](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/scripts/run_no_vote_mobind_pretrain.py)、[真实批次源验收CLI](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/scripts/verify_no_vote_mobind_source.py) 与 [Task8测试](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/tests/test_no_vote_mobind_pretrain.py)。注册器仅添加A5专用readiness描述符，不把全局门槛改true。

直接使用源 `P86MoBindLite`：width96、alignment64、dropout.12、40类、imu_instance_normalization=false、imu_event_feature_width=0、domain_classes=0。每阶段从seed20260811重新随机初始化全部参数，不读取旧S/I预训练权重或用select权重初始化refit；视觉大骨干不在本阶段训练。

24轮末态、batch64、保留尾batch、原同步temporal_augment、class_weights幂.35、AdamW LR4e-4／min2e-5／wd.05、原cosine调度、AMP／GradScaler和clip-grad2。固定预算期间没有val loader、early stop或最佳epoch选择。保留模型／优化器／scaler／Torch、CUDA、Python与NumPy RNG完整epoch恢复、阶段锁和原子checkpoint写入。

损失直接调用 `train_p86_mobind_pretrain.losses`：semantic2、token.25、local.05、global.15、reconstruction.1、视觉KD.75／T2、教师feature.25、RF KD1／T1、contrastive T.08、mask_ratio.35；未启用的asymmetric Skeleton→IMU／domain项权重仍0。部分缺模态的CE／视觉KD按源batch归约，不自行全屏蔽；局部／重建／RF等项使用源自己的mask。模型内部LayerNorm保留，外部只执行Task6的identity转换。

A1为当前v3、同phase的校准early_late目标，features先转FP32再取view mean，保持 `[N,2,1024]`；A4必须使用同phase `fit_teacher_target`，RF不可用行按源Dataset清零并保留valid=false。实际学习祖先仅来自相应phase，未读取历史OOF预测或调用旧main。五个直接源模块与最终(1)包字节一致。

## 训练覆盖与输出含义（Coverage and Output Semantics）

本阶段只训练至少一种运动输入有效的行：train12=1956／2039，refit14=2341／2427。分别83／86行完全没有有效运动输入，保留在完整报告与预测分母、用同phase完整fit标签得到的先验回退；这与全模态缺失不是同一集合，包含IR-only行。train12的42条部分缺模态行保留训练和源损失，不以缺分支为由删除。四区输出按各自S/I有效mask分别回退。

源模型只有Skeleton和IMU两个分类头，没有独立combined学习头。按Task8要求另保存可用分支概率等权平均的 **combined诊断（Available-mean Diagnostic）**；该项明确标注 `diagnostic_available_mean_not_source_learned_head`，不冒充新增源专家或流入主teacher bank，不增加Task13候选。S/I输出仍为原分支logits，combined不改变模型训练。

| 人口（Population） | Skeleton | IMU | combined额外诊断 |
|---|---:|---:|---:|
| train12 | 1793/2039＝87.94% | 1931/2039＝94.70% | 1955/2039＝95.88% |
| development2（user6/7） | 139/388＝35.82% | 111/388＝28.61% | 154/388＝39.69% |
| refit14 | 2129/2427＝87.72% | 2297/2427＝94.64% | 2317/2427＝95.47% |
| final4 | 不评分；590有效／19prior | 不评分；580有效／29prior | 不评分；590有效／19prior |

refit包含user6/7，其成绩不能称val。开发有效分母分别S385、I369、combined385；上述主表始终用完整388。最终总体全模态缺失仍18，不能因为motion-only的19／29先验行而删掉额外样本；后续视觉融合可能仍有输入。本阶段的独立S/I准确率不是Task10融合结果，更不是Task12伪标签适配后的最终知识转移成绩；没有据此改配方。

## 门槛、产物与成本（Readiness, Artifacts and Cost）

`motion_readiness`只放A5 fixed24，绑定Task4／6／7完成摘要、源pins、公共motion／source_transform及各phase的A1/RF目标引用。共享descriptor不把select/refit学习祖先混在一起，实际A5模型直接带自己的phase目标；不放A7／P12或完整bank。协议身份仍 `2b563c9d2b0c4ee62297a1ec05062811f7059ee1778c7090cefede2cc507618a`，全局formal_generation_ready=false、P12／Thermal／P89／A18缺项及ROI质量精度残余均不变。Task1–7的模型／缓存和原始数据／两提交包未修改。

正式训练select83.48秒、refit101.70秒，合计185.18秒；日志跨度213.37秒（约3分33秒，含加载／保存与导出，不含前置实现／测试）。模型与12份标准输出约29.97MiB，不含两个resume。PID39904／session51895已结束，无后台训练或自动化。

当前v3 run的 `A5/{select,refit}` 保存checkpoint.pt、model.json、history.json、resume.pt、identity.json和artifact.json；四区各有skeleton／imu／combined三份标准targets，共12份。checkpoint携带源model_state／model_config供后续同phase融合初始化。`protocol/task8_complete.json`是整体完成入口，绑定原 `task8_artifacts_ready.json`、两模型／四区输出及本记录和验收证据；P01/P02仍为唯一指导与交接。

## 验收证据（Acceptance Evidence）

- 3种正常／缺IMU／缺Skeleton的小批次：源loss每项与参与参数梯度差0，缺分支CE／视觉KD仍生效；配置／同seed随机初始化测试通过。Dataset测试验证精确ID、同步时间索引、FP32 view mean、RF无效清零，以及推理没有label／teacher字段。
- 完整epoch恢复夹具先在2轮后中断，恢复后严格24轮且所有样本都参与每轮训练，LR首末4e-4／2e-5，不产生validation字段。最初断言错误地把`retrieval`中的`val`认成验证指标，修正测试前缀检查，未改训练逻辑。
- 真实train12两批各64条：与源Dataset的增强／教师／标签／motion逐值相同；所有loss、参与梯度、源AdamW／GradScaler更新后的参数差0，非有限梯度0；第一批含3条partial，第二批无partial。峰值737.39MiB，7.70秒。
- 最终MoBind／v3 roles／RF共27项针对回归通过68.29秒。初轮旧RF测试误把名称含oof的Python源码当旧缓存，改为只禁止OOF载荷扩展名，仍禁止旧预测及final标签；4个Transformer norm_first/nested_tensor库警告保留，不影响通过。
- 正式有限自验10.37秒：两个24轮历史／fit用户／本phase祖先／源摘要、完整2039/388/2427/609 IDs、40列、12份小targets／prob／mask／prior、combined诊断及允许标签指标复算均通过。真实refit模型对64条final4公共输入禁读教师／标签载荷推理，打开尝试0，S/I logits与已导出结果差0。未扫描整个motion／visual大payload、重跑大教师或读取final标签，未将自验称独立复审。

正式命令：

```powershell
& 'D:/Anaconda/envs/PyTorch2.7/python.exe' -B scripts/run_no_vote_mobind_pretrain.py --config configs/experiments/teammate_single_teacher_fixed_split_v3.yaml --device cuda
```

证据位于 `outputs/task5_audit/`：`P33-task8-loss-red.log`、`P33-task8-engine-red.log`、`P33-task8-phase-red.log`、`P33-task8-core-tests.log`、`P33-task8-targeted-regression.log`、`P33-task8-final-regression.log`、`P33-task8-real-batches.log`、`P33-task8-formal.log`、`P33-task8-final-acceptance.json/.log`。真实批次小证明为 `run/protocol/task8_real_batch_acceptance.json`。下一任务为尚未启动的Task9诊断对照，不自动执行Task9或后续融合。
