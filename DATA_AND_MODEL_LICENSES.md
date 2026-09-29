# 数据、模型与发布材料

本项目原创代码和文档采用 [Apache License 2.0](LICENSE)，版权声明见根目录 [NOTICE](NOTICE)。本文件记录中文意图分类与四槽抽取实验所用的第三方材料；第三方数据和模型仍分别遵守下列许可，根目录代码许可不改变其条款。

## MASSIVE 1.0 中文数据

- 来源：Amazon Science 的 [MASSIVE 官方仓库](https://github.com/alexa/massive)、[官方 1.0 下载包](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz)及[数据集卡](https://huggingface.co/datasets/AmazonScience/massive)。本项目只取 `zh-CN`。
- 数据许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)。请区分数据许可与 MASSIVE 仓库代码的 Apache-2.0 许可，见[官方 NOTICE](https://github.com/alexa/massive/blob/main/NOTICE.md)。
- 版权及来源署名：MASSIVE © Amazon.com, Inc. or its affiliates。MASSIVE 源于 [SLURP](https://github.com/pswietojanski/slurp) 英文文本的翻译与本地化；研究引用见 [MASSIVE 论文](https://aclanthology.org/2023.acl-long.235/)和 [SLURP 论文](https://aclanthology.org/2020.emnlp-main.588/)。
- 本项目的改动：Qwen SFT 实验从官方 `train` 按意图固定种子抽取 594 条；线性基线用官方 `train` 全部 11,514 条训练。两者均使用官方 `dev`、`test` 评测；测试与训练/开发文本存在重复，报告单列去重结果。确切 SFT 抽样规则见 `scripts/prepare_massive.py` 产生的 `data/massive-zh/manifest.json`；线性模型的训练与选择规则见 `scripts/evaluate_massive_linear.py` 和 `reports/massive-linear-dev.json`。
- 四槽抽取研究也只从官方 `zh-CN` 数据派生。v1 从官方 `train` 建 384／100／200 的训练、开发、确认切分；v2 先排除旧项目用过的文本组及官方 `dev`／`test` 的强归一化重复组，再从官方 `train` 锁定 640／250／400 的新切分。两轮仅抽取 `date`、`time`、`place_name`、`person`，并将目标输出改为严格 JSON；具体过滤、分组、抽样和文件哈希见 [v1 报告](reports/massive-slots-study-summary.md)与 [v2 预先设计](reports/massive-slots-v2-design.md)。公开仓库仅保留聚合统计与哈希，不再分发派生原句。

本仓库默认不提交原始归档或整理出的训练、开发、测试样本。如发布包含 MASSIVE 文本的材料，应保留官方数据归档内的 `LICENSE`、给出上述来源和版权署名、链接 CC BY 4.0，并说明抽样、清洗或其他改动。生成的 `data/massive-zh/MASSIVE-LICENSE.txt` 是官方归档内许可证的本地副本。服务文档也应列明训练数据来源。

从项目根目录重新下载并整理：

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
Invoke-WebRequest https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz -OutFile data\raw\amazon-massive-dataset-1.0.tar.gz
python scripts\prepare_massive.py --source data\raw\amazon-massive-dataset-1.0.tar.gz
```

## Qwen2.5-0.5B-Instruct 模型

- 来源：[Qwen 官方 Hugging Face 仓库](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)和[官方 ModelScope 仓库](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)。本项目的下载脚本从 ModelScope 下载权重。
- 模型许可：[Apache License 2.0，Qwen 官方许可文件](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE)，含 Alibaba Cloud 版权声明。本仓库的 `licenses/Qwen2.5-0.5B-Instruct-LICENSE` 是该官方固定提交中的许可副本，并非本项目脚本的许可证。此许可适用于当前 **0.5B-Instruct** 模型；更换型号时需重新核对对应模型仓库。
- 本项目的改动：使用 4-bit 基座运行，并训练 LoRA SFT 适配器。实验没有将 4-bit 基座导出为新的独立量化权重文件。

本仓库默认不提交基座权重、tokenizer 或训练适配器。运行 `python scripts\download_model.py` 会把基座下载到被 Git 忽略的 `models/Qwen2.5-0.5B-Instruct/`，并确保该目录包含上游 `LICENSE`；若下载快照缺少该文件，脚本会复制上述官方许可副本。如果以后分发基座、合并模型或包含模型的安装包，请将 Apache-2.0 许可全文和上游版权声明随包提供，标明微调或量化修改；若上游快照带有 `NOTICE`，也保留相关署名。发布适配器时同样应记录它依赖的基座名称、来源和许可。

## Git 发布边界

项目的 `.gitignore` 排除了 `models/`、`outputs/`、`data/raw/`、`data/massive-zh/`、历史 OASST2 整理数据，以及逐题预测和含原文的旧报告。可公开的汇总指标保留在 `reports/`。发布前检查待提交文件清单；`git add -f` 会绕过这些排除规则。

`llm-posttraining` 虽位于另一工作区内，但已在本目录建立独立 Git 仓库。发布时只从本目录检查和提交文件，避免将外层工作区的文件带入该公开项目。
