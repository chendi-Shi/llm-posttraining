# 新实验设计：中文语音助理意图分类

## 研究问题

在本机 CPU 的 Qwen2.5-0.5B-Instruct 4-bit QLoRA 设置下，用少量许可清楚的中文标注数据做 SFT，能否提升模型识别中文语音助理请求意图的能力？只有 SFT 显示可重复收益后，才评估 DPO。

任务只涉及通用语音助理请求，不含投研或金融内容。

## 任务定义

输入一条中文单轮请求，输出 MASSIVE 定义的一个意图标签，例如 `alarm_set`。提示要求只输出标签，不加解释；避免每条样本重复一长串标签表。

训练集包含 60 个标签，评测时使用候选标签 trie 做约束解码：基座、SFT 和 DPO 都只能输出同一组标签。这把问题定义为闭集分类，避免自由生成格式错误掩盖分类能力，同时不使用 dev/test 答案。

## 数据与许可

使用 Amazon Science 发布的 [MASSIVE 1.0](https://github.com/alexa/massive)，仅取 `zh-CN`。数据许可证为 CC BY 4.0；数据仓库代码的 Apache-2.0 许可证单独适用代码。整理脚本保留官方归档中的数据许可证，并在 manifest 记录版本、源文件 SHA-256、划分计数、标签分布和训练 ID。再分发时注明来源、许可证并引用 MASSIVE 与其英文种子数据 SLURP 的论文。

官方中文划分实际核验为：

| 用途 | 官方划分 | 样本数 | 处理 |
| --- | --- | ---: | --- |
| SFT | train | 594 | 固定种子抽样，每个意图最多 10 条；`cooking_query` 只有 4 条，全部保留，不复制稀有样本 |
| DPO 决策与流程检查 | dev | 300 | 从完整 dev 按意图固定种子分层抽样；共 59 个意图，缺少 `audio_volume_other` |
| 完整开发集 | dev | 2,033 | 原始全量文件保留，必要时作错误分析 |
| 最终测评 | test | 2,974 | 全量保留；共 59 个意图，缺少 `cooking_query` |

train、dev、test 的 ID 无交叉。训练抽样种子为 `20260924`。测试 macro-F1 对 test 实际出现的 59 个意图取平均，并同时报告逐类支持数，避免把没有测试样本的标签算入平均。

## 训练对照

1. **基座**：原始 Qwen2.5-0.5B-Instruct，4-bit 量化推理。
2. **SFT**：用 594 条近似平衡样本训练一轮，只预测意图标签。现有 trainer 每步累计 4 条，设 149 optimizer steps，序列上限 128。固定抽样样本的完整序列为 60–83 token。
3. **DPO（条件阶段）**：只有 SFT 在 dev 上达到下面的提升门槛后才运行。对 SFT 在训练子集上预测错误的样本，正确标签作 chosen，受约束解码出的错误标签作 rejected；不使用 dev/test 标签构造偏好对。错误样本不足 50 条时跳过 DPO。DPO 完成后再用 dev 与 SFT 比较，确定最终候选模型。

本机为 16 GB 内存 Intel CPU。新数据 1 步烟雾训练用时 18.5 秒；完整 149 步预计约 45–60 分钟，实际以训练日志为准。

## 测评设计

对所有模型使用相同提示、60 个标签、512 token 输入上限和贪心 trie 约束解码；输出最多 16 token。先在 dev 比较基座与 SFT，必要时再比较 DPO 与 SFT。选定最终候选后，基座、SFT 和入选的 DPO adapter 在 test 上各跑一次。

**主要指标：macro-F1**，在 test 有支持的 59 个类别上等权平均。同时报告总体准确率、每类精确率/召回率/F1/支持数、场景准确率、超出标签词表比例和输入截断数。

在固定的 300 条分层 dev 子集上决定是否继续 DPO；在 dev 与最终 test 的模型对照中，都按真实意图类别分层、逐题配对 bootstrap 2,000 次，报告 macro-F1 和准确率差值的 95% 区间。预先定义“有实际意义的提升”为 macro-F1 至少增加 2 个百分点且区间下界大于 0；test 不用于训练、提示选择或是否运行 DPO 的决定。

## 运行步骤

下载官方数据包并整理数据：

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
Invoke-WebRequest https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz -OutFile data\raw\amazon-massive-dataset-1.0.tar.gz
python scripts\prepare_massive.py --source data\raw\amazon-massive-dataset-1.0.tar.gz
```

训练一轮 SFT：

```powershell
python scripts\train_sft.py --train-file data\massive-zh\train.jsonl --output-dir outputs\massive-sft --max-steps 149 --max-length 128
```

先用 dev 的 10 条分层子集检查输出，再用固定的 300 条分层子集决定是否运行 DPO：

```powershell
python scripts\evaluate_massive.py --eval-file data\massive-zh\dev.jsonl --max-examples 10 --output reports\massive-dev-smoke.jsonl
python scripts\evaluate_massive.py --eval-file data\massive-zh\dev.jsonl --max-examples 300 --output reports\massive-base-dev.jsonl
python scripts\evaluate_massive.py --eval-file data\massive-zh\dev.jsonl --adapter outputs\massive-sft --max-examples 300 --output reports\massive-sft-dev.jsonl
python scripts\compare_massive.py --gold data\massive-zh\dev.jsonl --base reports\massive-base-dev.jsonl --candidate reports\massive-sft-dev.jsonl --allow-subset
```

通过 SFT 门槛后，才从 SFT 在训练样本上的错误预测准备 DPO：

```powershell
python scripts\evaluate_massive.py --eval-file data\massive-zh\train.jsonl --adapter outputs\massive-sft --output reports\massive-sft-train.jsonl
python scripts\prepare_massive_dpo.py --train-file data\massive-zh\train.jsonl --predictions reports\massive-sft-train.jsonl --output data\massive-zh\dpo.jsonl
$pairs = (Get-Content data\massive-zh\dpo.jsonl).Count
$steps = [int][math]::Ceiling($pairs / 4)
python scripts\train_dpo.py --model models\Qwen2.5-0.5B-Instruct --train-file data\massive-zh\dpo.jsonl --sft-adapter outputs\massive-sft --output-dir outputs\massive-dpo --max-steps $steps --max-length 128
```

少于 50 组偏好对时，脚本停止且不写 DPO 数据。DPO 在 dev 上与 SFT 比较后冻结最终候选，再各跑一次最终 test：

```powershell
python scripts\evaluate_massive.py --eval-file data\massive-zh\dev.jsonl --adapter outputs\massive-dpo --max-examples 300 --output reports\massive-dpo-dev.jsonl
python scripts\compare_massive.py --gold data\massive-zh\dev.jsonl --base reports\massive-sft-dev.jsonl --candidate reports\massive-dpo-dev.jsonl --base-name SFT --candidate-name DPO --allow-subset
python scripts\evaluate_massive.py --eval-file data\massive-zh\test.jsonl --batch-size 16 --output reports\massive-base-test.jsonl
python scripts\evaluate_massive.py --eval-file data\massive-zh\test.jsonl --adapter outputs\massive-sft --batch-size 16 --output reports\massive-sft-test.jsonl
python scripts\evaluate_massive.py --eval-file data\massive-zh\test.jsonl --adapter outputs\massive-dpo --batch-size 16 --output reports\massive-dpo-test.jsonl
python scripts\compare_massive.py --gold data\massive-zh\test.jsonl --base reports\massive-base-test.jsonl --candidate reports\massive-sft-test.jsonl --candidate-name SFT
python scripts\compare_massive.py --gold data\massive-zh\test.jsonl --base reports\massive-base-test.jsonl --candidate reports\massive-dpo-test.jsonl --candidate-name DPO
```

## 当前状态

官方数据包已下载，594 条训练数据及全量 dev/test 已生成；划分数量、意图覆盖、ID 隔离和归档内许可证已核验。之前自由格式槽位抽取的 10 条 dev 检查耗时约 197 秒，且格式错误较多，因此本轮改为闭集意图分类。

正式 SFT、DPO 和最终评测均已完成。SFT 在 300 条分层 dev 子集上相对基座 macro-F1 提升 9.21 个百分点，95% 区间 +5.43 到 +12.42；准确率提升 8.00 个百分点，95% 区间 +4.33 到 +11.67，达到预设门槛。DPO 相对 SFT 的 dev macro-F1 提升 1.35 个百分点，但 95% 区间为 −2.99 到 +5.61，未达到预设门槛，因此未在 test 上评估 DPO。

完整 test（2,974 条）上，基座准确率/macro-F1 为 26.53%/16.15%，SFT 为 29.32%/21.18%；SFT 的准确率差值为 +2.79 个百分点（95% 区间 +1.51 到 +4.17），macro-F1 差值为 +5.03 个百分点（95% 区间 +3.45 到 +6.70）。结果达到预设收益标准，但只支持该中文闭集意图分类任务上的适配效果，不应外推为通用对话能力提升。

结果文件：[最终 test 比较](reports/massive-test-comparison.json)、[SFT dev 比较](reports/massive-dev-comparison.json)、[DPO dev 比较](reports/massive-dpo-dev-comparison.json)。

## 来源与署名

- [MASSIVE 数据集卡与许可](https://huggingface.co/datasets/AmazonScience/massive)
- [MASSIVE 官方仓库：数据字段、下载与标注格式](https://github.com/alexa/massive/blob/main/README.md)
- [MASSIVE 官方数据 NOTICE](https://github.com/alexa/massive/blob/main/NOTICE.md)
- Fitzgerald et al., [MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages](https://aclanthology.org/2023.acl-long.235/)
- Bastianelli et al., [SLURP: A Spoken Language Understanding Resource Package](https://aclanthology.org/2020.emnlp-main.588/)
