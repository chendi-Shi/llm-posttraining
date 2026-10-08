# 本地大模型后训练：中文结构化抽取与意图分类

本项目在约 16 GB 内存、无独立 GPU 的本机上运行 Qwen2.5-0.5B-Instruct 的 4-bit QLoRA SFT/DPO，并以字符 BIO 为同任务对照。数据来自许可清楚的 MASSIVE 中文子集；每轮固定训练、开发、测试的原句组和模型选择规则，公开聚合评测与失败案例分析。仓库不含原始数据、权重或逐题预测。

**当前最重要的结果（v6，600 条新测试组）：**扩大 BIO 监督数据后的混合系统严格实体 F1 为 **62.59%**，相对旧混合系统的点估计高 **1.82 个百分点**，但配对 95% 区间为 **−3.58 至 +7.19**。因此预设的绝对 F1 门槛通过，**主要相对收益门槛未通过**；不能宣称可靠超越旧系统。[完整结论](reports/massive-slots-v6-study-summary.md)、[机器可读报告](reports/massive-slots-v6-test.json)和[复现说明](SLOT_V6_RUNBOOK.md)列出选择锁、文件指纹及限制。设计中的 1,600 组 QLoRA SFT 是独立的次级研究，只在开发集评估，不进入这组测试结论。

项目从 v1 四槽抽取起步：384 条训练、100 条开发、200 条确认，另有同量字符 BIO 基线。[v1 复现步骤](SLOT_RUNBOOK.md)、[训练环境清单](reports/massive-slots-training-provenance.json)和[研究报告](reports/massive-slots-study-summary.md)列出当时的结果。针对 v1 无槽误报，v2 SFT 与 v3 DPO 继续做纠错实验；v3 DPO 将无槽失败从 126/167 降至 28/167，但总体严格实体 F1 仅 47.12%，未达到预设门槛，两个旧确认集仍封存。[同样本继续 SFT 对照](reports/massive-slots-v3-matched-control-summary.md)展示新增数据与偏好目标的探索性比较。[项目展示页](PORTFOLIO_CASE_STUDY.md)和[岗位证据](JOB_APPLICATION.md)说明可写进简历的结果与边界。

| 四槽抽取，确认集 200 条 | 严格实体 micro-F1 | 逐句全对 | 无槽失败（共 40 条） |
| --- | ---: | ---: | ---: |
| Qwen 基座 | 2.55% | 0.0% | 40/40 输出无效 |
| Qwen 4-bit QLoRA SFT（96 步） | 60.43% | 40.0% | 40/40：34 条误报、6 条无效 |
| 字符 BIO + LinearSVC（同 384 条训练数据） | **68.05%** | **57.5%** | 7/40 误报 |

SFT 相对未微调基座的确认集 F1 提升 **57.88 个百分点**（配对 95% 区间 +50.39 至 +64.44）；BIO 的 F1 点估计再高 **7.61 个百分点**，但配对区间下界略低于 0。SFT 的无槽句误报仍高，是**后训练研究成果，不是自动抽取服务**。确认集按开发集预定规则只评一次，数据是实体富集挑战集；不能把分数直接当作真实流量表现。意图分类小模型与四槽混合系统都有本机 HTTP 端点，每个预测都要求人工复核；云服务器尚未提供。

## v2：自然频率留出与无槽失败修复

新数据集仅用 MASSIVE 官方中文 train，排除旧项目全部训练／评测原句组及官方 dev/test 的强归一化重复组。先锁定不按标签重抽的开发集 250 条、确认集 400 条，再构造 640 条含目标槽、困难负例和纯负例的训练集。预定比较同一 Qwen 基座的第 80／160 步 SFT；只有开发集 JSON、结构与复制合法率均达到 95%，且无槽失败率不高于 30%，才允许打开一次新确认集。第 160 步开发集实体 F1 **42.17%**、无槽失败 **126/167**，复制合法率 **91.2%**；同量 BIO 为 **61.76%** F1、**9/167** 无槽失败。选择器拒绝两个 SFT 候选，**没有生成选择锁，也没有评测 400 条确认集**。完整数据、偏离记录和开发集结论见[预先设计](reports/massive-slots-v2-design.md)与[结果报告](reports/massive-slots-v2-study-summary.md)。

## v3：DPO 纠错的收益与召回代价

从未用过的 MASSIVE 官方 train 原句组构造 256 对偏好样本，以 v2 SFT 为初始策略和冻结参考，在 CPU 上完成 64 步 DPO。相同 250 条开发集上，无槽失败从 v2 SFT 的 **126/167（75.45%）** 降至 **28/167（16.77%）**，配对下降 95% 区间为 **+50.90 至 +65.87 个百分点**；但阳性句 F1 从 **67.01%** 降至 **54.75%**。总体 F1 从 **42.17%** 到 **47.12%**，配对增量区间 **−1.67 至 +11.20 个百分点**，跨 0。总体 F1、阳性句 F1 和可靠总体改进三项预设门槛未通过，因此**不生成选择锁，也不打开确认集**。同量 BIO 开发集 F1 为 **61.76%**、无槽失败 **9/167**。训练样本、权重和最终保存精度的限制见 [v3 设计](reports/massive-slots-v3-design.md)、[复现步骤](SLOT_V3_RUNBOOK.md)与 [完整结果](reports/massive-slots-v3-study-summary.md)。

## 同样本继续 SFT：探索性方法对照

在已看过 v3 DPO 开发集后，我从同一 256 对偏好数据提取正确回答，按[事先记录的对照方案](reports/massive-slots-v3-matched-control-design.md)从同一 v2 adapter 继续 SFT 64 步。固定开发集上，继续 SFT 总体实体 F1 **49.24%**、阳性句 F1 **67.36%**、无槽失败 **78/167**。DPO 相对该对照将无槽失败进一步降至 **28/167**，但阳性句 F1 降至 **54.75%**，总体 F1 差值的配对 95% 区间为 **−8.05 至 +3.46 个百分点**。这只能解释当前开发集的权衡，不能把差异唯一归因为 DPO 损失，也不改变 v3 未过门槛及 400 条确认集封存的决定。[完整报告](reports/massive-slots-v3-matched-control-summary.md)

## v4：平衡阳性偏好的失败检验

在[预先固定的新切分](reports/massive-slots-v4-design.md)上，v4 将全新 DPO 偏好对的阳性占比从 25% 提到 50%，完成 64 步训练。另抽的 250 条自然频率开发集上，v4 阳性子集 F1 为 **66.29%**，高于同集冻结 v3 DPO 的 **60.49%**；但无槽失败从 **26/171** 升到 **98/171**，总体严格实体 F1 从 **52.69%** 降至 **44.79%**，配对差值 95% 区间为 **−14.18 至 −2.42 个百分点**。v4 未通过预设门槛，两个 400 条确认集仍封存。[复现步骤](SLOT_V4_RUNBOOK.md)与[完整失败报告](reports/massive-slots-v4-study-summary.md)保留训练和评测证据。

## v5：独立测试中的混合推理相对收益

在已看过的 v4 开发集上固定一个概率门控：BIO 为默认回答，仅在 v4 DPO 给出合法、可复制且高置信度的非空槽位时替换。规则、阈值 `−0.08`、新测试集和成功门槛均在推理前通过[方案](reports/massive-slots-v5-design.md)与[选择锁](reports/massive-slots-v5-selection-lock.json)公开。**600 条全新自然频率测试组**上，严格实体 F1 为 BIO **49.75%**、v4 DPO 单模型 **41.65%**、固定混合系统 **57.21%**；混合相对 BIO 提高 **7.46 个百分点**，配对 95% 区间 **+2.10 至 +13.23**。混合的阳性子集 F1 为 **65.78%**，无槽失败 **50/429（11.66%）**，JSON／结构／原文复制合法率均为 100%。**预设绝对 F1 ≥60% 未达，因此总体门槛失败**；它是混合系统的独立相对收益，不能称为 DPO 单模型成功或可自动上线。[完整报告](reports/massive-slots-v5-study-summary.md)和[复现步骤](SLOT_V5_RUNBOOK.md)保留完整边界。

## v6：扩大 BIO 训练集，绝对门槛达标但相对门槛未过

在[预先固定](reports/massive-slots-v6-design.md)的 1,600 条新训练组上重训字符 BIO，并沿用**冻结 v4 DPO 与同一阈值 `−0.08`**。600 条全新自然频率测试组上，严格实体 micro-F1 为旧 BIO **55.70%**、新 BIO **61.39%**、v4 DPO **42.62%**、旧 v5 混合 **60.77%**、新混合 **62.59%**。新混合的阳性子集 F1 **71.12%**、无槽失败 **45/433（10.39%）**、JSON／结构／原文复制合法率各 **100%**，逐句全对 **81.8%**。

**这是本项目四槽抽取首次达到预设绝对 F1 ≥60% 门槛；但 [v6 预设独立成功条件](SLOT_V6_RUNBOOK.md)还要求新混合相对旧 v5 混合的配对 95% 区间下界大于 0，实测差值为 +1.82 个百分点、区间 −3.58 至 +7.19，因此七项检查中六项通过、总体未通过。** 不能说 v6 复现并超过 v5，也不能称为 DPO 单模型提升或可自动上线。[完整结论](reports/massive-slots-v6-study-summary.md)与[复现步骤](SLOT_V6_RUNBOOK.md)保留边界。

设计中的同 1,600 组 Qwen QLoRA SFT 次级实验另行进行，**固定配方与开发集评测[协议](reports/massive-slots-v6-sft-protocol.md)已单独固定**：它不进入主系统选型、不参与主验收，也不为它解封测试集，因此它的数字不能与上面的 62.59% 混在一起报告。

## 旧意图分类任务与人工复核服务

| 模型 | 训练集 | 官方 test 准确率 | 官方 test macro-F1 | 当前用途 |
| --- | ---: | ---: | ---: | --- |
| Qwen 基座 | — | 26.53% | 16.15% | 对照 |
| Qwen 4-bit QLoRA SFT | 594 条 | 29.32% | 21.18% | 后训练研究，不用于自动路由 |
| 字符 TF-IDF + LinearSVC | 11,514 条 | 83.76% | 79.30% | 人工复核服务的默认候选 |

线性模型在排除与完整 train/dev 相同文本的 2,727 条 test 上，准确率为 **82.80%**、macro-F1 为 **77.99%**。同一官方 test 此前已用于 SFT 研究，因此这个结果是锁定模型的确认，不能称为全新盲测。模型仍缺真实请求、未知意图、逐类充分样本及拒判校准，不能自动执行用户指令。详细证据见[线性基线测试摘要](reports/massive-linear-test-summary.md)和[发布与上线验收](RELEASE_READINESS.md)。

从官方 MASSIVE 1.0 归档重建线性模型后，可按[服务运行说明](SERVICE_RUNBOOK.md)启动本地接口；原创代码和文档采用 [Apache 2.0](LICENSE)，[数据与模型许可](DATA_AND_MODEL_LICENSES.md)说明第三方来源和署名。[云服务器部署说明](CLOUD_DEPLOY.md)记录容器构建和访问边界。当前尚无云服务器和对外服务地址。

四槽抽取系统另有 `--backend slots` 的本机端点 `POST /v1/slots`（v6 BIO＋冻结 DPO，阈值 `−0.08`，与已公布评测口径一致），同样只返回人工复核候选。[服务侧证据](SERVING_EVIDENCE.md)记录冷启动、单请求延迟分位数、分阶段耗时与并发扫描的测量方法，并明确标出哪些数字因与训练争抢 CPU 而无效；真实流量容量、延迟 SLO 与拒判质量仍然未验收。

## 旧实验：中文语音助理意图分类 SFT/DPO

项目已从宽泛的回答偏好试验切换到可逐题核对的中文语音助理意图分类。使用 CC BY 4.0 的 MASSIVE 中文数据：594 条近似均衡 SFT 训练样本、全量官方 dev/test；主指标为 59 个测试类别上的 macro-F1，并报告准确率和逐类支持数。

## 本轮结果

使用 594 条训练样本对 Qwen2.5-0.5B-Instruct 做 4-bit QLoRA SFT；在 300 条分层 dev 子集上通过预设门槛后，构造了 418 组仅来自训练集的 DPO 偏好对。DPO 在 dev 上没有显示相对 SFT 的可靠增益，因此没有进入最终测试。

在完整 2,974 条 test 上，SFT 准确率为 29.3%（基座 26.5%，+2.8 个百分点），macro-F1 为 21.2%（基座 16.1%，+5.0 个百分点）。按意图分层的配对 bootstrap（2,000 次）给出的 macro-F1 差值 95% 区间为 +3.45 到 +6.70 个百分点，准确率差值区间为 +1.51 到 +4.17 个百分点；达到预先定义的收益门槛。这个结果支持小规模任务适配有效，但不代表通用对话能力提升。DPO 的 dev 差值为 +1.35 个百分点，95% 区间 −2.99 到 +5.61，未达到门槛。

汇总指标和配对比较见 [最终测试比较](reports/massive-test-comparison.json)、[SFT 开发集比较](reports/massive-dev-comparison.json) 与 [DPO 开发集比较](reports/massive-dpo-dev-comparison.json)。逐项预测留在本地，不随公开仓库提交。许可、数据清单、评测规则和复现命令见 [实验设计](EXPERIMENT_DESIGN.md)、[数据许可清单](data/DATA_LICENSES.md)及[数据与模型发布说明](DATA_AND_MODEL_LICENSES.md)。

## 上一版实验：OASST2 宽泛回答偏好（已封存）

独立的 CPU QLoRA 实验：4-bit 量化冻结基座模型，先做监督微调（SFT），再用偏好数据做直接偏好优化（DPO）。项目用于跑通完整方法并比较基座、SFT、SFT+DPO 的表现。

## 本机配置与范围

- Intel Core i5-1130G7，8 个逻辑线程；Intel Iris Xe 集成显卡；16 GB 内存。
- 当前 Python 3.14 环境没有训练依赖。另一套 Conda 环境 rl-cpu 有 Python 3.10 和 CPU 版 PyTorch 2.14，可供本项目的隔离虚拟环境复用。
- 本地先使用 Qwen/Qwen2.5-0.5B-Instruct，batch size 1，序列长度不超过 256；先跑少量样本观察速度和内存，再逐步扩大。
- 4-bit 主要减少基座权重占用。CPU 没有 CUDA 加速，训练耗时会较长。

bitsandbytes 当前官方支持矩阵包含 Windows x86 CPU 上的 QLoRA 4-bit，要求 AVX2；QLoRA 冻结量化基座，只训练额外的 LoRA 参数。参考：[bitsandbytes 安装与支持](https://huggingface.co/docs/bitsandbytes/installation)、[PEFT 量化训练说明](https://huggingface.co/docs/peft/developer_guides/quantization)。

## 建立本项目环境（PowerShell）

从项目根目录运行。使用现有 CPU PyTorch 环境创建项目专属 venv，不改动 Conda 环境：

~~~powershell
$py = '<本机已安装 CPU PyTorch 的 Python 3.10 可执行文件路径>'
& $py -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\check_environment.py
~~~

默认从 ModelScope 官方仓库下载基座模型，保存在 models/Qwen2.5-0.5B-Instruct，并保留上游许可文件：

~~~powershell
python scripts\download_model.py
~~~

## 数据格式

SFT 每行一条 JSON，包含对话列表形式的 prompt 与 completion。DPO 每行包含 prompt、chosen 和 rejected 对话列表。列表项使用 role 和 content 字段，格式与 TRL 的对话数据一致。

`data/sft.sample.jsonl` 和 `data/dpo.sample.jsonl` 是用于检查训练链路的合成样例，不用于能力结论。正式试训使用同一 OASST2 固定版本筛出的 256 条中文 SFT 对话和 64 组中文 DPO 偏好对。许可、筛选规则和来源记录见 [data/DATA_LICENSES.md](data/DATA_LICENSES.md) 及两个 manifest。

## 许可清楚的 SFT 和 DPO 数据

数据源是 [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)，数据卡声明许可证为 Apache-2.0。整理脚本固定到数据仓库提交 `179dd21fc55192153d94adb0e0ce8f69e222bf75`。SFT 只选中文、审核通过、未删除、非合成的消息链和 rank 0 助手回复；DPO 从同一个已审核中文 prompt 下配对 rank 0 与 rank 1 助手回复。两类数据都按本地 Qwen tokenizer 过滤，确保完整序列不超过 256 token，再以固定随机种子抽样。源消息 ID、筛选版本和文件校验值记录在 manifest 中。

数据审核字段不等于事实正确性评测。项目另从同一固定版本的 validation split 建立独立偏好评测集，并在评分脚本中检查评测源消息与 SFT、DPO 训练清单没有重叠。

## 流程烟雾检查

~~~powershell
python scripts\train_sft.py --model models\Qwen2.5-0.5B-Instruct --train-file data\sft.sample.jsonl --output-dir outputs\sft-smoke --max-steps 1 --max-length 128
python scripts\train_dpo.py --model models\Qwen2.5-0.5B-Instruct --train-file data\dpo.sample.jsonl --sft-adapter outputs\sft-smoke --output-dir outputs\dpo-smoke --max-steps 1 --max-length 128
~~~

流程确认后再增加样本和训练步数。适配器输出在 outputs/，可通过 scripts/evaluate.py 做定性对比。

## 用许可清楚的数据做一次本地 SFT

若尚未生成数据，先运行整理脚本（首次需从 Hugging Face 下载约 67 MB 的数据文件）：

~~~powershell
python scripts\prepare_oasst2_sft.py --count 256
~~~

然后运行 10 步 CPU 试训：

~~~powershell
python scripts\train_sft.py --train-file data\sft.licensed.jsonl --output-dir outputs\sft-oasst2-pilot-clean --max-steps 10 --max-length 256
~~~

生成 DPO 偏好数据并从 SFT adapter 开始试训：

~~~powershell
python scripts\prepare_oasst2_dpo.py --count 64 --max-length 256
python scripts\train_dpo.py --model models\Qwen2.5-0.5B-Instruct --train-file data\dpo.licensed.jsonl --sft-adapter outputs\sft-oasst2-pilot-clean --output-dir outputs\dpo-oasst2-pilot --max-steps 5 --max-length 256
~~~

上面的 10 步 SFT、5 步 DPO 命令是快速试跑。完整一轮对照实验使用下面的命令，分别训练 256 条 SFT 样本一轮和 64 组 DPO 偏好对一轮：

~~~powershell
python scripts\train_sft.py --train-file data\sft.licensed.jsonl --output-dir outputs\sft-oasst2-1epoch --max-steps 64 --max-length 256
python scripts\train_dpo.py --model models\Qwen2.5-0.5B-Instruct --train-file data\dpo.licensed.jsonl --sft-adapter outputs\sft-oasst2-1epoch --output-dir outputs\dpo-oasst2-1epoch --max-steps 16 --max-length 256
~~~

## 独立偏好评测

评测集从 OASST2 validation split 按与 DPO 相同的审核和 token 长度规则构造，当前有 33 组 rank 0 / rank 1 偏好对。评测脚本计算 chosen 与 rejected 完成部分在模型下的对数似然，并比较哪一项更高。复跑命令：

~~~powershell
python scripts\prepare_oasst2_dpo.py --split validation --count 1000 --max-length 256 --output data\dpo.eval.jsonl
python scripts\evaluate_preferences.py --model models\Qwen2.5-0.5B-Instruct --sft-adapter outputs\sft-oasst2-pilot-clean --dpo-adapter outputs\dpo-oasst2-pilot --eval-file data\dpo.eval.jsonl --output reports\oasst2_preference_eval.json
~~~

两次评测都使用同一 33 组验证集，短试跑报告为 [reports/oasst2_preference_eval_short_pilot.json](reports/oasst2_preference_eval_short_pilot.json)，完整一轮报告为 [reports/oasst2_preference_eval_1epoch.json](reports/oasst2_preference_eval_1epoch.json)：

| 训练轮次 | 模型 | token 平均对数似然 chosen 胜率 | 平均 token 对数似然差 |
| --- | --- | ---: | ---: |
| 基线 | 基座 | 60.6%（20/33） | 0.8443 |
| 快速试跑 | SFT，10 步 | 60.6%（20/33） | 0.8136 |
| 快速试跑 | SFT + DPO，5 步 | 60.6%（20/33） | 0.8179 |
| 完整一轮 | SFT，64 步 | 60.6%（20/33） | 0.6241 |
| 完整一轮 | SFT + DPO，16 步 | 60.6%（20/33） | 0.6558 |

两轮实验的整段对数似然 chosen 胜率也都相同，为 30.3%（10/33）。完整一轮中，DPO 的 token 平均分差比 SFT 高约 0.0317，但仍低于基座；胜率没有改变。因此，这套小数据和当前训练设置没有显示出偏好提升，完整一轮结果还显示相对基座的分差变差。33 组样本太少，不能据此断言所有 SFT/DPO 方法无效，但当前配置不应作为成功效果展示。整段对数似然会受回答长度影响。该评测只检验 OASST2 排名偏好的拟合情况，不是通用能力或生成质量基准。

## 匿名生成对比

为了检查对数似然指标是否对应到实际回答质量，项目还为同一 33 个验证提示分别生成基座、SFT、DPO 回答，并按每题随机打乱 A/B/C 标签。一次单一 AI 盲评按切题、遵循指令、准确性和清晰度比较回答：基座单独胜出 15 题，SFT 胜出 5 题，DPO 胜出 2 题，11 题打平。把平局得分均分后，胜出份额分别为 47.5%、30.8% 和 21.7%，没有显示后训练优于基座。公开仓库保留[评分摘要](reports/oasst2_blind_generation_score_summary.md)；包含原始对话文本的逐题材料留在本地。

此评分由单一 AI 完成，不是人工或多评审盲测；每条生成最多 96 个 token，部分回答被截断。它是小样本初步诊断，不能代替正式能力基准或人类偏好评测。

复现命令：

~~~powershell
python scripts\generate_blind_eval.py --sft-adapter outputs\sft-oasst2-1epoch --dpo-adapter outputs\dpo-oasst2-1epoch --eval-file data\dpo.eval.jsonl --max-new-tokens 96
~~~

## 本机烟雾训练结果

合成烟雾检查曾用 3 条 SFT 样例和 3 组 DPO 偏好样例各训练 1 步；SFT 约 26 秒/步，DPO 约 32 秒/步。它只验证训练链路可运行，不代表模型能力已经提升。

快速试跑 SFT：OASST2 中文子集 256 条，10 步，约 6 分 40 秒，平均训练 loss 2.488，覆盖约 0.156 个 epoch。适配器保存在 `outputs/sft-oasst2-pilot-clean`。

快速试跑 DPO：64 组 rank 0/rank 1 偏好对，5 步，参考分数预计算约 2 分 30 秒、优化训练约 5 分 27 秒，平均训练 loss 0.678，覆盖约 0.313 个 epoch。适配器保存在 `outputs/dpo-oasst2-pilot`。

完整一轮对照训练：SFT 64 步、平均 loss 2.474，运行约 70 分钟；DPO 16 步、平均 loss 0.678，参考分数预计算约 2 分 30 秒、优化训练约 31 分钟。adapter 分别保存在 `outputs/sft-oasst2-1epoch` 和 `outputs/dpo-oasst2-1epoch`。两种训练都成功运行，但独立验证没有显示胜率提升；训练 loss 下降不等于验证效果改善。
