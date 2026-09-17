# Third-party notices — CUHK-X Small Model Submission

核对日期：2026-09-08。本文只记录本地安装/项目来源证据，不构成法律意见。

## Ultralytics / YOLO

- 本地 Python 环境检测到 `ultralytics==8.4.33`。
- 本地 package metadata 的 License 字段为 `AGPL-3.0`，主页为 <https://ultralytics.com>。
- 官方源码仓库及许可证入口：<https://github.com/ultralytics/ultralytics>、<https://github.com/ultralytics/ultralytics/blob/main/LICENSE>。
- YOLO11 发布记录：<https://github.com/ultralytics/assets/releases/tag/v8.3.0>。
- 实际提交使用 `assets/models/yolo11n-pose.pt`（不是普通检测版 YOLO11n）；本次构建实际计算 SHA256 为 `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`，6,255,593 bytes，嵌入唯一 `checkpoints/model.pth`。历史记录中的普通检测模型 hash 不适用于此权重。

## 提交风险提示

已将本地可取得的相关LICENSE原文保存在 `licenses/`，来源文件hash见其manifest。
这些文件按原文保留，不改写为本项目许可证，也不代替各模型卡／数据条款。

2026-09-08已读到主办方topic738333回复，轻量YOLO自动预处理的参赛方法获允许；
这解决了赛道方法许可问题，不代表第三方版权／许可证条款被主办方替代。
本包确定在推理期使用YOLO，并已将权重放入唯一checkpoint，不将其伪称为仅开发期工具。

官网对 CUHK-X 规则写有获奖方案/代码需按 Apache 2.0 开源的要求，而本地 Ultralytics 包显示 AGPL-3.0。若最终提交或获奖开源范围包含 Ultralytics 代码、链接组件或其权重，AGPL 与官网 Apache 2.0 要求可能存在许可证兼容性冲突；本文件不作法律结论。提交前应向主办方和合格的软件许可顾问确认：

1. 完整披露Ultralytics作为推理期依赖及所用权重，不遗漏预处理组件；
2. 若最终推理依赖 YOLO 权重或运行时，官网 Apache 2.0 义务与 AGPL 条件如何满足；
3. YOLO 权重及 COCO/相关预训练资产是否需要单独披露和保留原始许可文本。
