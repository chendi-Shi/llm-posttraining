# 本地大模型后训练：中文结构化抽取与意图分类

本项目在约 16 GB 内存、无独立 GPU 的本机上运行 Qwen2.5-0.5B-Instruct 的 4-bit QLoRA SFT/DPO，构造许可清楚的数据、冻结模型选择、严格评测，并保留负结果。主研究已切换到 MASSIVE 1.0 中文四槽 JSON 抽取：384 条训练、100 条开发、200 条确认，另有同量数据的字符 BIO 基线。[主实验复现步骤](SLOT_RUNBOOK.md)、[公开训练环境清单](reports/massive-slots-training-provenance.json)和[槽位研究报告](reports/massive-slots-study-summary.md)列出实际命令、模型哈希、局限与结果；求职视角见 [岗位证据](JOB_APPLICATION.md)。仓库不含原始数据、权重或逐题预测。

| 四槽抽取，确认集 200 条 | 严格实体 micro-F1 | 逐句全对 | 无槽句误报 |
| --- | ---: | ---: | ---: |
| Qwen 基座 | 2.55% | 0.0% | 0/40 有效误报，40/40 输出无效 |
| Qwen 4-bit QLoRA SFT（96 步） | 60.43% | 40.0% | 34/40 |
| 字符 BIO + LinearSVC（同 384 条训练数据） | **68.05%** | **57.5%** | 7/40 |

SFT 相对未微调基座的确认集 F1 提升 **57.88 个百分点**（配对 95% 区间 +50.39 至 +64.44）；BIO 的 F1 点估计再高 **7.61 个百分点**，但配对区间下界略低于 0。SFT 的无槽句误报仍高，是**后训练研究成果，不是自动抽取服务**。确认集按开发集预定规则只评一次，数据是实体富集挑战集；不能把分数直接当作真实流量表现。小模型 HTTP 服务仍针对**旧意图分类任务**，每个预测都要求人工复核，云服务器尚未提供。

## 旧意图分类任务与人工复核服务

| 模型 | 训练集 | 官方 test 准确率 | 官方 test macro-F1 | 当前用途 |
| --- | ---: | ---: | ---: | --- |
| Qwen 基座 | — | 26.53% | 16.15% | 对照 |
| Qwen 4-bit QLoRA SFT | 594 条 | 29.32% | 21.18% | 后训练研究，不用于自动路由 |
| 字符 TF-IDF + LinearSVC | 11,514 条 | 83.76% | 79.30% | 人工复核服务的默认候选 |

线性模型在排除与完整 train/dev 相同文本的 2,727 条 test 上，准确率为 **82.80%**、macro-F1 为 **77.99%**。同一官方 test 此前已用于 SFT 研究，因此这个结果是锁定模型的确认，不能称为全新盲测。模型仍缺真实请求、未知意图、逐类充分样本及拒判校准，不能自动执行用户指令。详细证据见[线性基线测试摘要](reports/massive-linear-test-summary.md)和[发布与上线验收](RELEASE_READINESS.md)。

从官方 MASSIVE 1.0 归档重建线性模型后，可按[服务运行说明](SERVICE_RUNBOOK.md)启动本地接口；原创代码和文档采用 [Apache 2.0](LICENSE)，[数据与模型许可](DATA_AND_MODEL_LICENSES.md)说明第三方来源和署名。[云服务器部署说明](CLOUD_DEPLOY.md)记录容器构建和访问边界。当前尚无云服务器和对外服务地址。

## 当前实验：中文语音助理意图分类

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
