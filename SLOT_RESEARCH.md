# 中文结构化抽取：预注册实验方案

## 研究问题

在只有 CPU 和约 16 GB 内存的条件下，对 Qwen2.5-0.5B-Instruct 进行 4-bit QLoRA SFT，能否提高中文语音助理请求中四类槽位的严格抽取效果？本实验同时测量格式遵循与逐字复制能力。它是后训练研究；当前人工复核服务仍由独立的小模型提供。

## 数据与任务

使用 MASSIVE 1.0 的 `zh-CN` 官方 **train** 分区，只研究 `date`、`time`、`place_name`、`person` 四类槽位。输入是原始 `utt`，输出是且仅是 JSON，例如 `{"slots":[{"type":"time","value":"明天早上"}]}`；`value` 必须是原句中的连续子串，空目标为 `{"slots":[]}`。重复槽位保留，不做同义归一或预测后的修复。

`scripts/prepare_massive_slots.py` 固定种子 `20260928`，按 NFKC、casefold、删除空白后的原句分组，排除同组标注冲突，只从每组取一条。固定 **384 train / 100 dev / 200 confirmation**；每个划分覆盖四类目标和无目标句。三个划分的组互不重叠。dev 和 confirmation 还排除旧意图分类 SFT 的 594 条训练文本组。所有新划分均源于官方 train；旧项目已使用官方 test，因此这里不把官方 test 称为新盲测。进一步核对发现新 train 有 5/384、dev 有 2/100、confirmation 有 3/200 的归一化原句也存在于旧项目曾使用的官方 dev/test 中；因此 confirmation 只对**新槽位任务和本轮模型选择**保持留出，不能称作完全未见过的文本。模型预训练也可能接触过同源数据，无法据此断言完全无污染。

这些划分是**富集目标槽位的挑战集**：dev 和 confirmation 各有 80% 的句子含目标槽位，而官方 `zh-CN` train 中相应比例约为 32%。这里的实体 F1 和无槽句误报率不能直接当作真实流量上的表现；面向部署还要在自然频率样本和未知请求上评测。

官方归档 SHA-256、许可、源 ID、组哈希和新文件哈希写入本地 `data/massive-zh/slots/manifest.json`。原始语句和模型权重不进入公开代码仓库；数据遵守 MASSIVE 的 CC BY 4.0，代码遵守 Apache 2.0。

## 冻结的比较对象

1. **基座 Qwen**：原始 Qwen2.5-0.5B-Instruct，4-bit 推理；提示和生成设置与 SFT 完全相同。
2. **槽位 SFT**：从同一原始 Qwen 权重开始，用 384 条 train 的 `prompt`/`completion` 做 QLoRA，不续训旧意图分类 adapter。LoRA rank 8，目标模块 `q/k/v/o_proj`，batch 1，梯度累计 4，最大长度 160；长度核验表明训练样本最长 157 token。固定学习率 `3e-4`、96 optimizer steps（约一轮），线性退火，并保存第 48 与第 96 步。两个检查点都在 dev 上评测；按严格实体 micro-F1 选择，若相同则选较短训练的第 48 步。所有尝试须记录。
3. **监督式 BIO 基线**：仅用同一 384 条训练语句与标签拟合字符级模型，用同一 100 条 dev 评测。该基线回答任务是否本来适合更简单的模型。

本机基座权重 `model.safetensors` 的 SHA-256 为 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`，`tokenizer.json` 的 SHA-256 为 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`；复跑时应先核对这两个文件哈希，再比较数值。

另外在旧意图分类任务上做 64 条**训练集**记忆诊断，检查旧 SFT 的学习能力；这不是泛化成绩，也不用于挑选新任务的 confirmation 样本。

## 指标和决策

主要指标为 `(type, 原文 value)` 多重集合的实体 micro-F1：每个预测实体最多匹配一个标准实体，重复预测会产生额外假阳性。非法 JSON、错误结构、非原文子串全部按预测失败处理，另外分别报告 JSON 合法率、结构合法率、复制合法率。辅助指标为逐句完全正确率、各类型 F1 和无槽位句误报率。

Qwen 基座与 SFT 使用相同的贪心自由生成：输入上限 256 token、不截断，最多新生成 64 token，批量 8；不加 JSON 语法约束，也不修复输出。全部 684 条固定样本的目标完成部分最长 52 token（含结束 token），因此 64 token 足以容纳标准答案。先在 dev 比较三种方法并分析错误。只用 dev 选择模型及生成设置，记录每次尝试。SFT 若相对基座的实体 micro-F1 至少提高 5 个百分点，且按原句组配对 bootstrap 的 95% 区间下界大于 0，才称为开发集上的明确收益。和 BIO 基线的差距必须完整报告，即使 SFT 较弱。

冻结模型、提示和解码设置后，才对 200 条 confirmation 各评一次。confirmation 只用于确认既定结果，不用于再次调参；若不支持开发集结论，应如实报告失败并重新设计独立验证。小样本、同域数据和单次随机种子限制结论外推。DPO 只有在 SFT 已有稳定收益且能获得有意义、许可明确的偏好数据和独立评测时才立项。

## 可复现步骤

[主实验复现说明](SLOT_RUNBOOK.md)按顺序列出官方下载、旧意图训练集、槽位划分、SFT 检查点、BIO、开发集配对比较及显式解锁的确认集命令；[公开运行清单](reports/massive-slots-training-provenance.json)记录本次训练的软件版本、配置和文件哈希。

```powershell
python scripts\prepare_massive_slots.py --source data\raw\amazon-massive-dataset-1.0.tar.gz
python scripts\train_sft.py --train-file data\massive-zh\slots\train.jsonl --output-dir outputs\massive-slots-sft-lr3e4-96 --max-steps 96 --max-length 160 --learning-rate 0.0003 --seed 20260928 --save-steps 48
```

开发集评测脚本为 `scripts/evaluate_massive_slots.py`，同量数据 BIO 基线为 `scripts/evaluate_massive_slots_bio.py`；确认集分别使用评测脚本的 `--role confirmation --unlock-confirmation` 和 `scripts/evaluate_massive_slots_bio_confirmation.py`。确认入口要求基座、tokenizer、adapter 或 BIO 模型文件的已冻结 SHA-256。实际模型选择与哈希见 [冻结记录](reports/massive-slots-selection-lock.json)，数值、配对区间、错误分析及执行边界见 [研究报告](reports/massive-slots-study-summary.md)。48 项离线测试已通过，覆盖目标结构、去重、计分和确认门禁。

## 来源

- [MASSIVE 官方仓库](https://github.com/alexa/massive)
- [MASSIVE 数据集卡及许可](https://huggingface.co/datasets/AmazonScience/massive)
- [Qwen2.5-0.5B-Instruct 模型卡及许可](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
