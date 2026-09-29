# 四槽抽取 v2：自然频率留出与拒判实验

本轮只用 MASSIVE 1.0 `zh-CN` 官方 train 分区建样本。开发集和确认集在剔除所有旧训练/评测原句及官方 dev/test 的强归一化原句后，先按固定哈希顺序抽取，不按标签比例重抽。原始数据、逐题预测、权重和 v2 划分留在 Git 忽略目录。确认集只在开发集选出合格模型、写入冻结锁后使用一次。

## 1. 重新生成锁定数据

```powershell
.\.venv\Scripts\python.exe scripts\prepare_massive.py --source data\raw\amazon-massive-dataset-1.0.tar.gz
.\.venv\Scripts\python.exe scripts\prepare_massive_slots.py --source data\raw\amazon-massive-dataset-1.0.tar.gz
.\.venv\Scripts\python.exe scripts\prepare_massive_slots_v2.py
```

v2 脚本核对官方归档 SHA-256、旧意图训练文件、v1 三份划分及官方 dev/test 原句，按 NFKC、大小写折叠、去空白和 Unicode 标点后的键排除重复或近重复；冲突标注组直接剔除。应得到 `640 train / 250 dev / 400 confirmation`。本机锁定数据的 SHA-256：train `08916652746cc880aaa7fe17cd999f4bfbeb3834e55e5c43fbbe8148233372cd`；dev `354c8cb707813f2d148d06e905e9a169ba8606a3d8cba51f857e928dc17ce912`；confirmation `0975c57f2738041629d4842e8545fafef9806e99b06d2ea58306aaba31380e8d`。本机 dev 有 83 条含目标槽位、167 条无目标槽位；confirmation 有 103/297 条。

## 2. 训练与开发集选择

`train` 为四类目标槽位各 64 条阳性，加 192 条含非目标槽位的难负例及 192 条无标注负例。只改数据方案，不改 Qwen2.5-0.5B-Instruct 基座、四槽提示、LoRA 模块和 rank、学习率或贪心生成设置。全部 640 条样本已经通过真实 TRL 模板与 completion 掩码检查，最长 154 token，`max-length 160` 不截断。

```powershell
.\.venv\Scripts\python.exe scripts\check_sft_template_mask.py --train-file data\massive-zh\slots-v2\train.jsonl --max-length 160 --all
.\.venv\Scripts\python.exe scripts\train_sft.py --train-file data\massive-zh\slots-v2\train.jsonl --output-dir outputs\massive-slots-v2-sft-lr3e4-160 --max-steps 160 --max-length 160 --learning-rate 0.0003 --seed 20260928 --save-steps 80
```

在新 dev 上以相同 `batch-size 8`、`max-input-tokens 256`、`max-new-tokens 64` 评估第 80/160 步。v2 评测入口在生成前后核对基座、tokenizer 与 adapter 的 SHA-256，并把本地逐题预测文件 SHA-256 写入聚合报告；逐题预测只写到忽略的 `_tmp/`。旧 v1 SFT 只作为冻结参考，不参与 v2 检查点选择。

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v2.py --system v2_step80 --role dev
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v2.py --system v2_step160 --role dev
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v2.py --system v1 --role dev
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v2_bio.py --role dev
```

选择器要求 JSON、结构和原文复制合法率均至少 95%，无目标槽位句的失败率（输出非法或错误地产生目标槽位）最多 30%。合格者按严格实体 micro-F1、低失败率、早检查点依次选择；都不合格时拒绝写锁，也不打开确认集。

```powershell
.\.venv\Scripts\python.exe scripts\select_massive_slots_v2.py
```

冻结锁只包含哈希、设置和聚合开发集结果，也固定 v1 SFT 和同训练数据 BIO 对照的权重哈希。公开报告不得含原句、source ID 或逐题预测。开发集参考比较可用 `scripts\compare_massive_slots_v2.py --role dev`，并传入至少两个 `--system 名称=本地逐题预测文件` 与 `--output reports\massive-slots-v2-dev-comparison.json`。

选择器还核对 SFT 运行清单中的训练文件、种子、完成步数、学习率、序列长度及保存步数。选择锁固定基座与 adapter 目录的全部直接文件哈希；确认脚本只接受固定锁路径和固定输出路径。公开报告不得含原句、source ID 或逐题预测。

## 3. 一次性确认和结论

**此节在锁文件存在且人工核对其 `selected_step`、全部哈希及开发集门禁后才执行。**先记录 `reports\massive-slots-v2-selection-lock.json` 的 SHA-256。Qwen 确认脚本要求显式解锁，并在读取确认样本前核对基座、tokenizer 与 adapter。旧 v1 SFT 及同 640 条训练数据的 BIO 模型可以作为预先冻结的参考；任何确认结果都不得用来反选模型或改阈值。

```powershell
$lock = Get-Content reports\massive-slots-v2-selection-lock.json -Raw | ConvertFrom-Json
$lockSha = (Get-FileHash reports\massive-slots-v2-selection-lock.json -Algorithm SHA256).Hash.ToLowerInvariant()
$v2System = "v2_step$($lock.selected_step)"
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v2.py --system $v2System --role confirmation --unlock-confirmation --expected-selection-lock-sha256 $lockSha
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v2.py --system v1 --role confirmation --unlock-confirmation --expected-selection-lock-sha256 $lockSha
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v2_bio.py --role confirmation --unlock-confirmation --expected-selection-lock-sha256 $lockSha --expected-manifest-sha256 $lock.manifest_sha256 --expected-model-sha256 $lock.reference_systems.bio.model_sha256 --expected-dev-report-sha256 $lock.reference_systems.bio.dev_report_sha256
.\.venv\Scripts\python.exe scripts\compare_massive_slots_v2.py --role confirmation --unlock-confirmation --expected-lock-sha256 $lockSha --eval-file data\massive-zh\slots-v2\confirmation.jsonl --system bio=_tmp\massive-slots-v2-bio-confirmation-predictions.jsonl --system v1=_tmp\massive-slots-v2-v1-confirmation-predictions.jsonl --system "v2=_tmp\massive-slots-v2-${v2System}-confirmation-predictions.jsonl" --metrics bio=reports\massive-slots-v2-bio-confirmation.json --metrics v1=reports\massive-slots-v2-v1sft-confirmation.json --metrics v2=reports\massive-slots-v2-sft-confirmation.json --output reports\massive-slots-v2-confirmation-comparison.json
```

比较脚本要求 `bio → v1 → v2` 的预定系统顺序及每个模型的聚合评测报告，逐一核对模型、数据、生成设置和逐题预测哈希；只以 `v2_minus_v1` 作为主要成败比较，BIO 配对为参考。预定成功条件是 v2 对 v1 的实体 F1 配对区间下界大于 0、无槽失败率下降区间下界大于 0、且 v2 在确认集的无槽失败率不高于 30%。阳性子集 F1 和格式合法率是防止误读的辅助指标，不用于事后选模型。确认输出使用独占文件名，已有文件时拒绝覆盖。新确认集仍是 MASSIVE 同源留出，不能称为真实生产流量；在未配备服务器前也不代表云端推理服务已上线。
