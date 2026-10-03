# Task5 方法学修复最终独立复审（Final Methodology Re-review）

日期：2026-10-04，Asia/Shanghai。审计者：`/root/audit_20261002_methodology`；未参与实现修订。基础提交（base commit）为25b8ec0，本次审查其后的Task5工作区修复。

## 结论与范围（Verdict and Scope）

**GO：此前方法学审计发现的两项阻塞已关闭，未发现这两项修复的新增方法学问题。** 该结论支持正式学生训练（formal Student training）的方法学门槛；teacher features来源绑定、阶段锁、续跑及完成发布等工程修订，由另一个独立审查负责，其结论仍须合并。

本次只复核[前轮报告](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/2026-10-03-task5-methodology-audit.md)的F1/F2、对应源码及回归，并运行无文件写入的纯内存小型fixture。不重新读取像素/原始数据、不做全缓存SHA、不读取final4私有标签、不运行训练或GPU验收；只写本报告。此前已通过且未涉及本轮修订的像素和数学检查不重复执行。

审计时源码身份（source identity）：visual_student.py SHA256=`b1ebbc5ec081bcd5c1c9385210e3de84e1b0337e19953c95b0ecfcf8fbff6594`；test_no_vote_visual_student.py SHA256=`55ab454ef7a74755145ad28af7b9ba6ddb3e39b72e3a1d1a15fb20f2cfaa9859`。这些仅为小型源码身份，不是原始数据内容复验。

## F1 / P1：冻结选择预算绑定——已关闭（Selected-budget Binding: Closed）

- [外层入口](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:220)先轻量验证selection.fit_artifact为A2/select，要求kind=supervised_model，并要求登记模型config['budget_epochs']与selection.budget['epochs']相等；范围内但不同的epoch在初始化/阶段写入前被拒绝。
- [内层训练体](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:258)复用该已验证selected_record；同时核对selection.config、selected.config['model_recipe']与当前recipe一致，预算再次绑定，development ID hash精确匹配。refit仍从公开MC3重新初始化，没有读取selected权重替代初始化。
- [回归测试](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/tests/test_no_vote_visual_student.py:179)将选择预算改为合法范围内另一epoch，明确要求拒绝。
- 本审计另执行纯内存fixture：登记预算5而请求4时外层与内层均拒绝；错误Selection recipe与错误development ID hash均在读取实际标签、模型初始化或训练之前拒绝。fixture以模拟登记元数据隔离该分支；不声称它替代真实产物校验。

原“只检查1–18范围、不绑定已选预算”的接受路径已消除，无需修改预期epoch范围或重新选择方法。

## F2 / P2：单样本尾批策略——已关闭（Singleton-tail Policy: Closed）

[训练DataLoader](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:286)固定shuffle=True，并恢复drop_last=len(dataset)%batch_size==1，与队友训练loader的shuffle/drop-singleton条件一致；[推理DataLoader](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:179)仍shuffle=False且不丢尾批。

纯CPU、纯内存范围序列检查结果：

| 人口（Population） | 有效fit表/预测表 | batch | drop_last | 每epoch实际消费/预测 | 最后批大小 |
|---|---:|---:|---|---:|---:|
| select训练 | 1957 | 4 | true | 1956 | 4 |
| refit训练 | 2342 | 4 | false | 2342 | 2 |
| development推理 | 388 | 4 | false | 388 | 4 |

class weights仍在[完整有效fit标签](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:299)上用指数0.35计算，未改成仅该轮实际采到的标签；即select依据1957行、refit依据2342行。history的[training.samples与optimizer_steps](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/src/experiments/visual_student.py:170)记录真实消费和更新次数；config.valid_fit_rows表示有效拟合支持，不能与select每轮1956次样本消费混写。

accumulation4、末尾不满累积组仍step、AdamW、两轮warmup/cosine、CE/KD/relation配方及epoch决胜规则没有因本修复改变。未要求重建已经complete的2427/609像素缓存。

## 验证证据与边界（Evidence and Limits）

[final-targeted.log](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/task5_audit/final-targeted.log:8)记录30 passed、1 warning、24.63秒；[review-regression.log](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/outputs/task5_audit/review-regression.log:8)另记录30 passed、1 warning、23.25秒。本审计读取这些已有日志，没有声称重新运行30项全套测试。

本次独立纯内存检查全部通过：外层错误预算、内层错误预算、recipe错配、开发ID hash错配、1957训练尾批、2342训练尾批及388全分母推理。未访问真实标签/缓存，未调用训练体的有效训练路径。

正式训练在复审请求时尚未开始。本报告不是A2正式结果或最终准确率验收；完成select/refit后仍需核对所选epoch、有效支持/实际消费、同phase教师与先验、预测与sequence/anchor来源。合并独立工程GO后，可以按现有授权推进Task5，不需要额外OOF、多seed或全量原始数据复验。
