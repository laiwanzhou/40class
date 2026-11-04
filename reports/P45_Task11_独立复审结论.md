# P45 Task11 独立复审结论（Independent Review Conclusions）

整理日期：2026-11-04；以下三项为中断前2026-10-30已完成的独立审计（Independent Audit）结果，来自本聊天保存的审计结论，不冒称本日重新运行三次审计。实施交付见[P44](../docs/P44_Task11_近似复刻交付与产物验收.md)，目录迁移新增复审另见[P46](../docs/P46_项目目录迁移与历史身份兼容.md)。

## 方法学复审（Methodology Review）

审计者`p44_method_review`：PASS，无未解决P1／P2。核对源safe公式及解码顺序、P134／P137分组、P270／P139序列门控、20列合法剪枝及old／new等价、P310锐化／先验和固定人口fit边界。独立14项针对性测试通过（166.08秒）；误用基础Anaconda导致的首次导入失败已换正式环境纠正，未当作通过。

发现P3：计划写alpha=1，与源`fit_transition_model`默认0.25不符；实现实际为0.25。P01已在`ba11c7e`纠正，本次同步纠正规格中同一文字错误，不改训练配方或产物。状态：CLOSED。

## 工程复审与修复（Engineering Review and Fix）

审计者`p44_engineering_review`首次为CONDITIONAL：P2指出A8元数据校验尚可能接受同refit phase但不同预测分区的bank／safe。实际正式产物绑定正确，缺口位于拒绝错误输入的门槛。

修复`ba11c7e`：A8、bank、safe的partition与完整RowIndex必须一致；A8所用safe必须恰为bank的唯一safe父引用。缓存复用（Cached Reuse）校验精确bank／model引用，保持已发布NPZ不变。新增回归先RED再GREEN；工程审计者独立3项测试通过（8.47秒），复审结论P2 CLOSED、工程PASS，无新增问题。

## 正式产物复审（Formal Artifact Review）

审计者`p44_artifact_review`：PASS，限定已授权近似复刻。

- 四区精确样本ID／用户／40列与有效掩码、同phase祖先成立。
- 四个bank共80个列实例逐一对应新生成父预测；fallback和availability逐值正确，没有历史教师输出或复制凑列。
- safe公式独立复算、P270原序列算子及P139时间／视觉路由对照通过；A8锐化、先验、old／new等价与P310 route正确。
- P12／global／KNN／source-base输出的ID／概率／掩码正确；KNN本分区排除自身并与保存邻居平滑结果一致。
- 仅检查237份去重元数据及100份直接小输出；未读最终或其他标签，未计算准确率，未加载大checkpoint、特征缓存或重跑GPU。

## 放行范围（Release Scope）

三项复审无未解决P1／P2，允许关闭Task11近似复刻交付。局部GO不等于已执行Task12，不等于证明处理提高准确率，也不消除P41／P44已经披露的近似和既有精度残余。目录改名属于后续新变化，必须另行通过P46的迁移验收及独立复审后，才写最终收尾标记。
