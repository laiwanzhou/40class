# P27 Task4 实施前独立复审（Pre-execution Independent Review）

日期：2026-10-28。审查者为独立子智能体`p25_task23_geometry_review`，复用其Task2/3只读审查席位，未由实现者自审代替。

审查范围为Task4未提交代码、源数学对照、阶段门槛（Stage Readiness）及小夹具，未把尚未生成的正式模型或全窗缓存判为完成。

## 已给出的放行（Explicit Authorization Scope）

审查者明确认可：规格第51/97行及P01 Task4要求视觉头／全窗特征不等待P12/KNN。以`visual_readiness`描述符单独绑定源清单、Task3证明及消费源码，只允许A1、六族视觉头、class_prior和固定Ridge机制头；不改变协议身份或全局`formal_generation_ready=false`，不扩展到A7/P12/专家bank。此阶段化解释符合规格。

源六族72配置、原tie-break、fit-only温度、aligned floor、固定Ridge十干预和全窗uniform16＋三视野裁剪与源函数一致。修复下列P2后，审查者明确给出 **GO启动正式视觉头拟合＋一条真实train12全窗GPU验收**；完整全窗生成须先通过真实编码与权重兼容检查。

## P2问题与独立复验（Finding and Verification）

辅助头原先都构造TeacherTargets，导致其构造器对完整features做`np.isfinite(...).all()`：审查者小探针对同一features连续构造7次，实测全表扫描7次；四分区预计重复28次，违反有界验证要求。

已修复：辅助头直接返回Prediction，主A1对已经按ID/掩码验证的不可变特征使用显式有界构造器（Bounded Constructor），只检查固定三行切片；旧构造器默认完整检查不变。回归先失败再通过。独立探针确认新路径仅对`(3,2,3,1024)`切片检查，保留原features对象；`dataclasses.fields`仍仅prediction/features，InitVar不改变旧字段或协议身份；旧默认构造仍拒绝未抽中行NaN。

## 放行的实际限度（Limits）

源夹具的两次剩余失败均为测试适配：原`safe_relative`要求三段source_id，以及裁剪后视频帧尺寸不同不能把整个列表强制转成同形数组。已分别改成合规范例路径和逐帧比较，没有修改源裁剪算法；最终两项针对复测通过。

审查者已通过消息明确给出上述代码／方法放行，但当时最终汇总工具返回额度限制。本文件只记录已实际完成的只读核对、P2复验和明确的执行前GO，**不把额度中断当作正式产物独立验收通过**。随后审查恢复，实际头部产物已另获GO，见[P28](D:/work/2026.7.14_kaggle/_single_visual_processing_replication/reports/P28_Task4_产物独立验收.md)；完整fullwindow仍待生成后验收。不得据本文件直接进入Task5。
