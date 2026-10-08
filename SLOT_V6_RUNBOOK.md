# 四槽抽取 v6：扩大监督数据的复现说明

## 研究问题与公开时序

v5 混合系统在独立测试的严格实体 F1 为 57.21%，低于预设的 60% 门槛。v6 检验：把字符 BIO 的训练集扩大至 1,600 条后，沿用**冻结 v4 DPO 与原阈值 `−0.08`**，能否在新句子上超过旧 v5 混合系统。主候选是“新 BIO 默认；DPO 输出合法、非空、原文可复制且高置信度时整句替换”；主要比较是**新混合减旧混合**，两者在同一批新句子上运行。新旧 BIO 和 DPO 单模型另行报告。任何收益都属于整体系统，不能称为 DPO 单模型提升。

[v6 设计](reports/massive-slots-v6-design.md)与切分在本地 BIO 开发集评测前固定；**BIO 开发集评测之后**才首次发布到 GitHub，但公开发生在 v6 主混合系统开发集及封存测试集推理之前。不要把 BIO 开发集结果表述成 GitHub 公开预注册后的结果。

## 数据、许可与本机条件

数据来自 [MASSIVE 1.0 官方归档](https://github.com/alexa/massive#accessing-and-processing-the-data)的 `zh-CN` train，遵守 CC BY 4.0。排除官方 dev/test 的同组文本、旧意图训练和四槽 v1–v5 的全部使用或封存组，按强归一化原句分组。先按固定种子 `20261003` 取自然频率 **test 600／dev 300**，再按类型配额取 **train 1,600**（四类各 128、困难负例 544、纯负例 544）。原句、标注、模型和逐题预测只在本地 Git 忽略目录；[来源和许可清单](DATA_AND_MODEL_LICENSES.md)说明模型及代码授权。

| 固定文件 | SHA-256 |
| --- | --- |
| `data/massive-zh/slots-v6/manifest.json` | `7982b3760333a364ebf17c91556f9978b32811fdacc0321db3129b6abee41a70` |
| `data/massive-zh/slots-v6/train.jsonl` | `ea1cd537dcfd924b24eec7997b692a42ae9df757c52f89c904358665ec06a3d3` |
| `data/massive-zh/slots-v6/dev.jsonl` | `6ee1812c5ac9a12e44047cd57e5973be606b9dcdf546d1d30ccf96f4ff1a157c` |
| `data/massive-zh/slots-v6/test.jsonl` | `482efe79ef69a7a43fd7feb9128df8985db663a08b38d17a057237d4bf96293b` |

本轮在约 **16 GB 内存、无独立 GPU** 的 Windows CPU 环境运行。复现需要固定的 MASSIVE 归档、v1–v5 本地清单、旧 v2 BIO 权重、Qwen2.5-0.5B-Instruct 基座及 v4 DPO adapter。开发评测记录的主要包版本为 CPU PyTorch `2.14.0`、Transformers `5.17.0`、PEFT `0.21.0`、bitsandbytes `0.50.2`、scikit-learn `1.7.2`；选择锁还核对其他依赖和模型文件指纹。Qwen 推理在 CPU 上耗时较长；本轮混合开发集 300 条实测约 **19 分钟**。缺少上游数据或权重时，公开仓库本身不能重建完整实验。

## 固定执行顺序与已公布结果

在**全新隔离检出**的仓库根目录、已配置训练依赖的 `.venv` 中按顺序执行。准备脚本会核对旧资产和上表四个指纹；已有不同数据时拒绝覆盖。BIO 与主评测入口也拒绝覆盖已有结果。BIO 沿用原实现的训练随机种子 `20260928`，数据切分及 bootstrap 使用 `20261003`。

```powershell
.\.venv\Scripts\python.exe scripts\prepare_massive_slots_v6.py
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v6_bio.py
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v6_hybrid.py --role dev
```

已公布的 [BIO 开发集报告](reports/massive-slots-v6-bio-dev.json)给出新 BIO F1 **65.74%**。[主混合开发集报告](reports/massive-slots-v6-dev.json)在同一批 300 条句子上给出：旧 v5 混合 **61.28%**、新混合 **66.10%**，差值 **+4.83 个百分点**，组配对 95% 区间约 **+0.27 至 +9.78 个百分点**；新混合阳性 F1 **73.24%**、无目标失败率 **10.58%**，JSON／结构／复制合法率各 **100%**。这些仅是**开发集结果**。固定门槛（总体及阳性 F1 各 ≥60%、无目标失败 ≤20%、三项合法率各 ≥95%）均通过，因而生成一次性[选择锁](reports/massive-slots-v6-selection-lock.json)，文件 SHA-256 为 `2387005995d9a511dea6d1d81cb200614ed95845bf4e2eafec179bb0834a83b9`。

选择锁发布后，测试入口要求显式解封和上述锁 SHA，先核对数据、代码、模型、依赖、开发门槛及不存在既有输出，再读取测试样本。固定命令如下；该单次测试**已于 2026-10-08 本地完成**，报告为 `reports/massive-slots-v6-test.json`，入口拒绝覆盖既有结果：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v6_hybrid.py --role test --unlock-test --expected-selection-lock-sha256 2387005995d9a511dea6d1d81cb200614ed95845bf4e2eafec179bb0834a83b9
```

独立成功条件还要求新混合相对旧混合的严格实体 F1 **配对 95% 区间下界大于 0**。该主要比较差值为 **+1.82 个百分点**，区间 **−3.58 至 +7.19**，下界不大于 0，因此**七项检查中六项通过、总体未通过**：绝对门槛首次全部达标，但相对旧 v5 混合的增量未被确认。测试完成后只根据锁定入口的聚合报告判断，不修改阈值或重选模型。

## 已公布的独立测试结果

[完整结论](reports/massive-slots-v6-study-summary.md)与[机器可读报告](reports/massive-slots-v6-test.json)给出 600 条新句子上的五个系统：旧 BIO **55.70%**、新 BIO **61.39%**、v4 DPO **42.62%**、旧 v5 混合 **60.77%**、新混合 **62.59%** 严格实体 micro-F1。新混合的阳性子集 F1 **71.12%**、无槽失败 **45/433（10.39%）**、JSON／结构／复制合法率各 **100%**，逐句全对 **81.8%**。新旧混合各替换 BIO 回答 **80/600** 次，替换次数相同。

## 次级实验：同 1,600 组的 Qwen QLoRA SFT

可选的同 1,600 组 Qwen 4-bit QLoRA SFT 是[设计](reports/massive-slots-v6-design.md)中的次级实验，**不参与主系统选型，也不参与上面的主验收**：主实验的测试已在 2026-10-08 完成并判定，本实验不能追溯修改那个判定。主实验报告当时不含任何 SFT 数字，这一点不变。

补做时按设计固定 400 optimizer steps、学习率 `3e-4`、批 1／梯度累计 4、`max_length=256`、种子 `20261003`，只评最终权重。训练入口 `scripts/train_massive_slots_v6_sft.py` 会在训练前用真实 tokenizer 逐条执行模板与遮罩预检，任何超长样本即停止而不静默截断。评测**只在开发集**进行，入口 `scripts/evaluate_massive_slots_v6_sft.py` 在结构上不构造测试集路径；[协议](reports/massive-slots-v6-sft-protocol.md)说明为什么不为它解封测试集（设计要求的"测试前另行固定方案"窗口已过）。

结果写入 `reports/massive-slots-v6-sft-dev.json` 后，本段应补上实际数字，在此之前不要把 SFT 说成已完成或有成绩。
