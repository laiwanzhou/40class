# 独立环境、逐步README验收

日期：2026-09-11。本次不是仅把代码复制到另一个文件夹后调用原来的Python环境。
解压原提交ZIP到新目录；复制主办方的原始测试文件与空预测CSV；建立未启用全局／用户
site-packages的venv，再安装README中的依赖并执行入口。原训练／特征缓存没有复制。

## 实际步骤与结果

| 步骤 | 执行结果 |
|---|---|
| 校验原ZIP并解压 | 原文件SHA256核验通过 |
| `python -m venv .venv` | Python3.12.7，`include-system-site-packages=false` |
| 安装torch／torchvision | 官方CUDA13.0源安装2.9.1／0.24.1成功 |
| 安装依赖及锁定清单 | 成功；完整版本见根目录requirements-lock.txt |
| `python -m pip check` | 无依赖冲突 |
| 检查实际Python与CUDA | 使用新venv，用户site关闭，CUDA可用 |
| 输入／安全测试 | 18项通过 |
| 按PowerShell入口运行405条原始数据 | 成功，约349秒，输出顺序正确、非部分运行 |
| 按 `inference.sh` 运行同一405条 | Git Bash + 新Windows环境成功，约297秒 |
| 输出比对 | 两入口均与历史最强提交CSV逐字节相同，0条预测差异 |

CSV SHA256：

```text
a9796cde28c8a999623a1ee388047c97d2d1553fd9bf10a9ed30f8ca7768561c
```

模型权重保持不变：56,471,879字节，包含Student和姿态预处理权重。
具体环境、代码哈希和验收结果见 `ACCEPTANCE_20260911.json`。

## 本次查出的真实问题与修正

1. 旧依赖文件没有列出共享工具间接导入的transformers、huggingface-hub和safetensors。
   已补齐，并在新venv安装验证；未回退到原全局库。
2. 当前机器继承的代理导致官方HTTPS索引连接失败。确认直连正常后，仅在验收子进程
   清除代理继承并设置NO_PROXY；没有关闭TLS校验或修改系统代理。README记录了该分支。
3. Windows OpenCV原生文件名读取不支持本次中文目录，曾把有效IR判为不可读。
   清单检查改为读取文件字节后解码；后续图像读取使用固定Ultralytics的Unicode安全实现。
4. 增加所有CSV行、重复表头等验证；隔离YOLO/HuggingFace/Torch库设置与缓存。
5. 两入口优先使用包内venv，避免未激活时意外调用全局Python。

这些修正没有改变训练权重，也没有逐样本修改预测。失败日志与成功日志均保留在验收
工作目录；正式ZIP不包含venv、原始数据或中间缓存。

## 验收边界

- 本次验证的是提供权重后的原始数据到CSV流程；未重新训练教师或学生。
- Bash入口在Windows的Git Bash下验证，**不宣称已完成原生Linux安装验证**。
- 原测试集输出一致不能保证新人物／新测试集取得相同准确率，也不是无泄漏证书。
- 诚信声明尚需真实参赛者签署，不能用技术验收代替签字。正式发送前按README运行
  `verify_package.py --for-submission` 并人工核对签名。
