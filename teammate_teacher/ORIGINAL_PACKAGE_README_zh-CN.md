# CUHK-X 小模型赛道：提交与复现说明

英文对应版本：[README.md](README.md)。本说明面向人工操作人员，运行过程不需要与
Codex、ChatGPT或原开发者对话。

## 1. 这份包能做什么

使用固定的小模型，从主办方提供的新原始数据生成 `path,prediction` 两列CSV。
程序自行建立清单、姿态／ROI与传感器缓存，再完成推理；不读取旧测试集的预测文件或缓存。

本方案历史Kaggle分数 **0.91542** 来自参赛者提交后的回报，只绑定原405条测试数据和
指定CSV，不保证新测试集取得同样分数。冻结发布版曾在本机从原始文件完整运行405条，
此前约224秒生成与原已计分CSV一致的结果。2026-09-11又完成真正干净环境验收：
新目录解压、全新venv、不读取用户全局库、使用独立原始数据副本，经PowerShell入口
处理405条约349秒，CSV仍逐字节一致。详见 `docs/ACCEPTANCE_20260911.md`。

**代码、模型、双语说明和训练资源说明已整理；诚信声明仍须本人签署。**
状态见 `docs/RELEASE_STATUS.md`。本次补全依赖、加强CSV与中文路径读取、隔离库缓存，
没有改变模型权重或重新训练。验收通过不等于主办方已批准本方案，发包渠道和期限以通知为准。

## 2. 文件结构与提交阶段

```text
README.md                    英文操作说明
README_zh-CN.md               本中文操作说明
requirements.txt             依赖版本
requirements-lock.txt        本次干净Windows环境完整依赖锁定
verify_package.py            文件完整性和正式提交材料检查
inference.sh                 Linux/Bash入口
inference.ps1                Windows PowerShell入口
checkpoints/model.pth        唯一模型文件，含Student和YOLO
code/runtime.py              新数据到CSV的统一调度入口
code/legacy/                 冻结预处理与模型源码依赖
code/training/project/        完整本地训练／教师源码和配置快照
tests/                      输入和安全性测试
docs/                       当前规则、来源、验证记录及限制
kaggle/submission.csv         原已计分CSV，仅用于旧Kaggle数据
```

`honor_declaration.pdf` 应由真实参赛者使用主办方表格签署后放到本目录根下。目前没有
该文件；未制作虚假官方表格，也没有代签。补齐材料后必须重新打包和更新校验记录。

区分两种提交：

- **Kaggle阶段**：提交或选择 `kaggle/submission.csv`，不是上传整个代码ZIP。
- **主办方复现阶段**：按晋级通知交代码权重包，随后用新数据执行下面的推理命令。
  不能拿旧CSV当新数据答案；交ZIP后仍须通过主办方复现／新数据验证。

具体期限与上传渠道以主办方通知为准，官网对后续期限存在不同表述，详见规则记录。

## 3. 环境要求与安装

解压最终ZIP后，先执行 `python verify_package.py` 检查文件哈希和技术材料。
正式发送前，加入本人签署的声明后再执行 `python verify_package.py --for-submission`。
此程序只能检查PDF存在及文件结构，不能代替本人核对签名；缺声明时正式提交检查会失败。

已验证环境为Windows、Python3.12.7、NVIDIA RTX5070 Ti Laptop（12GB显存）、驱动581.29、
PyTorch2.9.1+cu130、torchvision0.24.1+cu130。为405条输入推荐至少16GB内存、20GB空闲
磁盘空间（含环境、原始数据副本及临时文件）；数据更多时增加空间。建议使用GPU。
CPU运行不保证满足现场时限。

先切换到解压后本README所在的包目录，确认现有Python为3.12，再建立不读取全局库的
独立环境。Windows PowerShell依次执行：

```powershell
python --version
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

若激活脚本受到执行策略限制，下面命令直接使用 `.\.venv\Scripts\python.exe` 代替
`python`，不需要修改全局安全策略。不要开启 `--system-site-packages`。
Linux对应命令为 `python3.12 -m venv .venv` 和 `source .venv/bin/activate`。
在新环境中继续：

```powershell
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -r requirements-lock.txt
python -m pip check
python -c "import sys,site,torch,torchvision; print(sys.executable); print(site.ENABLE_USER_SITE); print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

核对Python路径位于 `.venv`，用户全局库标志为 `False`；使用GPU时最后一行应为 `True`。
上述命令不代表已在所有Linux、
驱动或GPU上安装测试成功。不要直接在原有复杂环境里升级／覆盖依赖。

原开发环境存在旧torchvision元数据不一致；本次新venv已经消除混合安装影响，实际
torch/torchvision版本与依赖检查均通过，没有修改原全局环境。

环境安装需要网络；**推理不需要账户、API密钥、LLM或额外模型下载**。程序设置
`YOLO_OFFLINE=true`，关闭Ultralytics在线检查／遥测。离线评审机器应提前准备对应依赖
安装包。不要未经回归验证自行换Ultralytics或PyTorch版本。

若安装报 `SSL: UNEXPECTED_EOF_WHILE_READING`，先检查代理连接，不要关闭TLS证书校验。
本机验收遇到继承的本地代理连接失败，但直接访问官方索引正常。仅在确认网络允许
直连时，可在当前PowerShell会话清除代理变量后重试原安装命令：

```powershell
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY -ErrorAction SilentlyContinue
$env:NO_PROXY = '*'
```

`NO_PROXY`避免Python又自动读取系统代理。这不修改系统代理设置；重新打开终端可
重新继承原环境。不适用于必须使用代理的网络。

## 4. 新测试数据怎么放

以下是示意目录，不是实际已提供的新测试集：

```text
D:\new_data\
  test.csv
  small_model_track_test\
    sample_0001\
      Depth_Color\
      IR\
      Skeleton\predictions\
      IMU\
      Thermal\
      Radar\
    sample_0002\
      ...
```

对应的空预测CSV示例：

```csv
path,prediction
small_model_track_test/sample_0001/,
small_model_track_test/sample_0002/,
```

输入可以只有 `path` 列，也可以另有空的 `prediction` 列；不接受其他列、已有预测、
标签、重复样本名、绝对路径或越界路径。保留官方 `path` 的内容与顺序。
输入条数不固定为405，也不要求样本目录名以 `SM_test` 开头。

`DataDir` 可以是包含CSV路径前缀文件夹的上一级，也可以是该前缀文件夹本身。
**不要改名原始模态文件**：帧ID与时间戳用于同一片段的传感器对齐，不用于外部标签映射。

当前冻结预处理的边界：

- 需要非空Skeleton记录及可恢复的时间轴。
- 缺失／不可读IR或缺失IMU时使用缺失掩码，不替换为旧预测。
- 历史ROI构建需要可用且同步的Depth/IR/Skeleton组合；否则视觉分支被屏蔽。
- 只有帧计数、又无法恢复绝对时间的Skeleton会明确报错，不伪造时间进行IMU对齐。
- 如果主办方改变了数据格式或未给路径清单，应先取得格式说明，不能自行猜测样本标签。

## 5. Windows人工运行

在本README目录打开PowerShell，激活上面的环境后执行（路径按实际位置替换）：

```powershell
& .\inference.ps1 -DataDir 'D:\new_data' -TestCsv 'D:\new_data\test.csv' -OutputDir 'D:\new_result'
```

两个入口脚本优先使用本包 `.venv` 中的Python，即使没有激活也不会误用全局库。
PowerShell还可用 `-PythonExecutable` 指定其他环境的解释器。仅当包内没有 `.venv`
时才使用PATH中的Python；先核对环境。输出报告记录实际解释器路径。

必须使用新的输出目录。程序不会覆盖已有 `submission.csv` 或 `report.json`。
默认自动使用可用GPU；如需明确选择设备，使用Python入口：

```powershell
python code/runtime.py --bundle checkpoints/model.pth --data-root 'D:\new_data' --test-csv 'D:\new_data\test.csv' --output 'D:\new_result_gpu' --device cuda
```

CPU模式把 `--device cuda` 改为 `--device cpu`。不要因CPU能启动就假定满足现场时限。
如果PowerShell脚本受执行策略限制，可直接使用Python入口，无需修改全局安全策略。

## 6. Linux/Bash人工运行

```bash
bash /path/to/package/inference.sh /path/to/new_data /path/to/new_data/test.csv /path/to/new_result
```

或在包目录使用：

```bash
python code/runtime.py --bundle checkpoints/model.pth --data-root /path/to/new_data --test-csv /path/to/new_data/test.csv --output /path/to/new_result
```

Bash入口已提供，但本交付未完成干净Linux系统的完整安装／端到端测试。

## 7. 什么结果才能交

成功退出后，输出目录包含：

```text
new_result/
  submission.csv       本次新数据预测；要提交的结果
  report.json          条数、是否部分运行、输入／模型／输出哈希与时间
```

人工检查：

1. 命令退出成功，不能在异常后提交部分文件或旧结果。
2. `report.json` 中 `partial` 必须为 `false`，`rows` 等于官方输入行数。
3. CSV仅含 `path,prediction` 两列，顺序与输入一致，每条预测为0～39整数。
4. 保存运行报告和控制台日志；提交的是本次输出，而非包内旧 `kaggle/submission.csv`。

**正式运行绝对不要使用 `--max-rows`**：这是开发期小样本检查选项，即使成功也不是完整
提交。`--keep-work` 可保留新生成缓存用于排错；缓存包含比赛数据，不得公开上传。

## 8. 方法与模型大小

推理流程为：原始文件 → YOLO自动姿态 → 确定性ROI → IR像素与Skeleton/IMU时间窗
→ MC3/MoBind紧凑Student → argmax类别。对新数据不执行Teacher、会话解码器、自训练
或人工逐条修正。

Student浮点权重以FP16保存，加载进FP32模块，GPU前向使用AMP。YOLO权重也包含在同一个
`checkpoints/model.pth`，该文件为 **56,471,879字节**，不是只计算识别模型而漏算预处理。
checkpoint SHA256：

```text
28d41bf833c191ddf354e8da27b2c9c91ac58d30b7b58aa5f9cd4096d42febde
```

这是一版明确的低精度存储方案，不声称其浮点权重与历史FP32权重逐位一致。

## 9. 已有验证及不能据此证明的事情

- 405条原始数据发布版重跑：约224秒，输出与历史已计分CSV的SHA256完全一致。
- 独立解压后改名两条样本运行：成功，任意样本数／输入顺序按要求处理。
- 18项输入、安全及中文路径测试通过；本次记录新的入口代码哈希，模型哈希不变。
- 像素和IMU与历史缓存一致；1条Skeleton源特征有历史生成差异，但没有改变任何预测。
  数值范围和限制在 `docs/RAW_REPRODUCTION_NOTES.md` 中保留。

训练与测试Skeleton原始文件内容哈希交集为0，但这不是所有模态、缓存或历史OOF均无
泄露的证明。历史全局OOF／校准来源仍有限制，不能把旧验证值称为重新独立确认的完整
嵌套验证。没有因比较旧CSV而对个别样本加入覆盖标签。

## 10. 训练源码、规则与待补材料

Student在官方训练人物上拟合，之后使用旧Kaggle无标签输入上的模型伪标签进行适配。
训练期Teacher及会话／重复录制逻辑不进入新测试集推理流程。

`code/training/project/` 保存完整本地训练与教师源码快照；逐文件哈希和静态检查见
`code/training/source_manifest.json`。训练顺序、实际适配命令见 `docs/TRAINING_PIPELINE.md`，
资源下载和用途见 `docs/EXTERNAL_RESOURCES.md`。`docs/provenance/` 的绝对路径是历史
训练记录，不属于推理配置。本次未从零重训，不将额外重训当作主办方要求；历史验证
谱系的限制照实保留，不能因打包而宣称新验证成绩。

已直接读到主办方允许AI代码助手、自动伪标签、轻量YOLO预处理的回复，不再需要重复
索取AI编程许可。仍禁止人工测试标签、测试真值或闭源LLM/API直接解题。
官方许可口径与第三方版权义务是两回事；保留相关组件及权重的许可说明。

正式发包前还需要真实签署的诚信声明，并遵循官方交付通知；若晋级后续
报告／答辩阶段，再按通知准备报告和演示材料。详见本包 `docs/RELEASE_STATUS.md`。
不要宣称本包已经获得主办方合规认证。

## 11. 常见问题

| 现象 | 处理 |
|---|---|
| 找不到样本目录 | 对照CSV路径前缀检查DataDir，勿改标签或套用旧manifest |
| 输入CSV被拒绝 | 移除额外列／已有预测，检查重复、空值和路径格式 |
| 输出目录已有结果 | 换新目录，不要误交上一次生成的CSV |
| 缺Skeleton或时间轴错误 | 核对官方解压结构／文件完整性；不要凭空合成时间 |
| CUDA不可用 | 核对GPU、驱动与实际torch安装；CPU只作另行评估的运行选项 |
| 显存不足 | 通过Python入口尝试 `--batch-size 1` 并重新验证；YOLO另有固定批量，不能保证该参数解决所有显存问题 |
| 需要保留排错数据 | 使用 `--keep-work` 并保留日志，勿公开原始数据与缓存 |

可以先运行不加载模型的安全测试：

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

这些测试不能替代新数据推理、完整训练复现或主办方验证。
