# 数据来源与许可

## 当前实验：AmazonScience/MASSIVE 1.0（zh-CN）

- 来源：[MASSIVE 官方仓库](https://github.com/alexa/massive)和[数据集卡](https://huggingface.co/datasets/AmazonScience/massive)；官方下载包为 MASSIVE 1.0。
- 数据许可证：CC BY 4.0。仓库代码的 Apache-2.0 许可证单独适用代码，不代替数据许可证；另见[官方 NOTICE](https://github.com/alexa/massive/blob/main/NOTICE.md)。
- 整理方式：Qwen SFT 从官方 train 按 60 个意图分层抽取，每类最多 10 条；`cooking_query` 仅有 4 条，全部纳入；该训练样本按本机 Qwen tokenizer 的完整对话长度过滤。独立线性基线使用官方 train 的全部 11,514 条；两项实验均保留官方 dev/test 作评测。
- 复现种子：`20260924`。数据源版本、文件哈希、观察到的 split 数量、训练 ID 和标签分布由 `scripts/prepare_massive.py` 写入 `data/massive-zh/manifest.json`。
- 署名与引用：再分发或发布衍生数据时注明 MASSIVE 来源及 CC BY 4.0，引用 MASSIVE 论文与其英文种子数据 SLURP 论文。完整实验规则见 `EXPERIMENT_DESIGN.md`。

## 已封存实验：OpenAssistant/oasst2

- 来源：[Hugging Face 数据集卡](https://huggingface.co/datasets/OpenAssistant/oasst2)；项目仓库：[LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
- 数据集卡声明：Apache-2.0。
- 使用版本：`179dd21fc55192153d94adb0e0ce8f69e222bf75`。
- 数据卡说明该版本消息收集截至 2023-11-05，并包含中文消息、消息角色、父子关系、审核结果、删除标记、合成标记和回复排名字段。
- 整理规则：保留 `lang=zh`、`review_result=true`、`deleted=false`、`synthetic=false` 的消息链；仅用 rank 为 0 的助手回复；限制链长不超过 12 条、prompt 与回复各不超过 1800 个 Unicode 字符、按 Qwen tokenizer 计的完整对话不超过 256 token；固定种子 `20260924` 抽取 256 条。
- 原始助手回复 ID 与 prompt 父消息 ID 写入 `sft.licensed.jsonl`；完整版本、筛选规则、样本 ID 清单见 `sft.licensed.manifest.json`。

## 从同一数据源构造的 DPO 偏好对

- 使用相同的 OASST2 提交版本和 Apache-2.0 数据卡声明。
- 每组来自同一个已审核中文 prompt：rank 0 回复作为 chosen，rank 1 回复作为 rejected；两边均未删除、非合成、审核通过，且文本不同。
- prompt 上下文不超过 12 条消息；chosen 与 rejected 各自和 prompt 组成的完整 Qwen chat-template 序列均不超过 256 token。
- 固定种子 `20260924` 从 630 组符合筛选条件的候选中抽取 64 组。prompt、chosen、rejected 的原始消息 ID、仓库版本、筛选规则和 JSONL SHA-256 记录在 `dpo.licensed.manifest.json`。

数据卡的审核结果用于筛除被拒绝/删除的消息，不代表事实正确性认证。样本尚未逐条人工校对，因此训练结果需要独立评测，不应把数据卡许可证等同于对每条内容权利的单独法律意见。

## 独立偏好评测集

- 从上述同一 OASST2 固定版本的 `validation` split 构造，数据卡声明许可证为 Apache-2.0；不参与训练。
- 应用相同语言、审核、删除、合成、对话长度及 token 长度过滤，保留 33 组 rank 0/rank 1 回复。
- `dpo.eval.manifest.json` 记录评测样本的源消息 ID、随机种子、版本及 `dpo.eval.jsonl` 的 SHA-256。`scripts/evaluate_preferences.py` 会校验该哈希和 validation split，并拒绝与 SFT 或 DPO 训练 manifest 有源消息 ID 重叠的数据。
- 评测按 chosen 与 rejected 完成部分的序列对数似然及每 token 平均对数似然评分。33 组样本仅构成小型偏好代理指标，不能证明通用能力、事实准确性或生成质量改善。
