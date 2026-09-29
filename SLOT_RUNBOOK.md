# 中文四槽后训练实验复现

从仓库根目录在 PowerShell 中运行。主实验的方案、结果与限制分别见 [预定方案](SLOT_RESEARCH.md)、[研究报告](reports/massive-slots-study-summary.md)。这里列出实际使用的路径、参数和执行顺序。原始 MASSIVE 数据、逐题预测和模型权重留在本地；不要将 `data/raw/`、`data/massive-zh/`、`outputs/` 或 `_tmp/` 上传到公开仓库。

## 1. 环境和官方数据

使用 Python 3.10 与 CPU 版 PyTorch；本次训练的实际软件版本见[公开运行清单](reports/massive-slots-training-provenance.json)。`requirements.txt` 给出兼容范围，不锁定完整依赖图，因此不同版本可能无法重现完全相同的权重哈希。先按 [README 的环境步骤](README.md#建立本项目环境powershell) 建好虚拟环境并安装依赖，再运行：

```powershell
python scripts\check_environment.py
python scripts\download_model.py
New-Item -ItemType Directory -Force data\raw | Out-Null
Invoke-WebRequest -Uri https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz -OutFile data\raw\amazon-massive-dataset-1.0.tar.gz
python scripts\prepare_massive.py --source data\raw\amazon-massive-dataset-1.0.tar.gz
python scripts\prepare_massive_slots.py --source data\raw\amazon-massive-dataset-1.0.tar.gz
```

`prepare_massive.py` 先重建旧意图 SFT 的 594 条训练样本；槽位划分脚本用这份文件排除开发和确认集中的旧训练原句。槽位划分应得到 **384/100/200** 条 train/dev/confirmation。官方归档、旧训练文件和三个新划分的 SHA-256 见[公开运行清单](reports/massive-slots-training-provenance.json)；划分脚本也会核对官方归档版本，并在本地生成 `data/massive-zh/slots/manifest.json`。

## 2. 训练与开发集模型选择

本次训练在约 16 GB 内存的 CPU 电脑上用时约 101 分钟。训练前检查全部 384 条样本的对话模板和 completion 掩码：

```powershell
python scripts\check_sft_template_mask.py --train-file data\massive-zh\slots\train.jsonl --max-length 160 --all
python scripts\train_sft.py --train-file data\massive-zh\slots\train.jsonl --output-dir outputs\massive-slots-sft-lr3e4-96 --max-steps 96 --max-length 160 --learning-rate 0.0003 --seed 20260928 --save-steps 48
```

下面三次 Qwen 开发集评测都使用同一贪心生成设置：`batch-size=8`、`max-input-tokens=256`、`max-new-tokens=64`。脚本默认值不同，复跑时必须显式传入这些参数。`_tmp/` 中的逐题预测仅供本地配对比较，不公开。

```powershell
python scripts\evaluate_massive_slots.py --role dev --eval-file data\massive-zh\slots\dev.jsonl --metrics reports\massive-slots-base-dev.json --predictions _tmp\massive-slots-base-dev-predictions.jsonl --batch-size 8 --max-input-tokens 256 --max-new-tokens 64
python scripts\evaluate_massive_slots.py --role dev --adapter outputs\massive-slots-sft-lr3e4-96\checkpoint-48 --eval-file data\massive-zh\slots\dev.jsonl --metrics reports\massive-slots-sft-step48-dev.json --predictions _tmp\massive-slots-sft-step48-dev-predictions.jsonl --batch-size 8 --max-input-tokens 256 --max-new-tokens 64
python scripts\evaluate_massive_slots.py --role dev --adapter outputs\massive-slots-sft-lr3e4-96 --eval-file data\massive-zh\slots\dev.jsonl --metrics reports\massive-slots-sft-step96-dev.json --predictions _tmp\massive-slots-sft-step96-dev-predictions.jsonl --batch-size 8 --max-input-tokens 256 --max-new-tokens 64
python scripts\evaluate_massive_slots_bio.py
python scripts\compare_massive_slots.py --role dev --eval-file data\massive-zh\slots\dev.jsonl --system base=_tmp\massive-slots-base-dev-predictions.jsonl --system sft48=_tmp\massive-slots-sft-step48-dev-predictions.jsonl --system sft96=_tmp\massive-slots-sft-step96-dev-predictions.jsonl --system bio=_tmp\massive-slots-bio-dev-predictions.jsonl --output reports\massive-slots-dev-comparison.json
```

按 [预定方案](SLOT_RESEARCH.md#指标和决策)，只用开发集的严格实体 micro-F1 在第 48/96 步之间选 SFT；平局选第 48 步。本次第 96 步为 58.15%，第 48 步为 57.51%，因此选择第 96 步。选择及运行文件哈希在打开确认集预测前写入[冻结记录](reports/massive-slots-selection-lock.json)。复跑若得到不同选择，应先查数据、模型和软件版本，不得按本次确认集成绩反选检查点。

## 3. 确认集：先核哈希，再显式解锁

以下命令用于复核**本次已完成的一次确认实验**，不要为当前项目反复运行确认集或用它调参。第三方若独立训练得到不同权重哈希，应记录自己的检查点和冻结规则，再用新的留出样本确认，不能把结果称作本次模型的精确复现。先检查本地文件与公开冻结记录一致：

```powershell
$lock = Get-Content reports\massive-slots-selection-lock.json -Raw | ConvertFrom-Json
$prov = Get-Content reports\massive-slots-training-provenance.json -Raw | ConvertFrom-Json
function Assert-Sha256([string]$Path, [string]$Expected) { $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant(); if ($actual -ne $Expected.ToLowerInvariant()) { throw "SHA-256 mismatch: $Path" } }
Assert-Sha256 data\raw\amazon-massive-dataset-1.0.tar.gz $prov.source_archive_sha256
Assert-Sha256 data\massive-zh\train.jsonl $prov.old_intent_sft_train_sha256
Assert-Sha256 data\massive-zh\slots\train.jsonl $prov.slot_train_sha256
Assert-Sha256 data\massive-zh\slots\dev.jsonl $lock.dev_file_sha256
Assert-Sha256 data\massive-zh\slots\confirmation.jsonl $lock.confirmation_file_sha256
Assert-Sha256 data\massive-zh\slots\manifest.json $lock.locked_data_manifest_sha256
Assert-Sha256 models\Qwen2.5-0.5B-Instruct\model.safetensors $lock.base_model_sha256
Assert-Sha256 models\Qwen2.5-0.5B-Instruct\tokenizer.json $lock.tokenizer_sha256
Assert-Sha256 outputs\massive-slots-sft-lr3e4-96\adapter_model.safetensors $lock.selected_adapter_sha256
Assert-Sha256 outputs\massive-slots-bio.joblib $lock.bio_model_sha256
```

**只有上述检查全部通过、且开发集选择已经冻结，才运行以下一次性确认命令。**脚本的 `--unlock-confirmation` 是有意设置的门禁；Qwen 命令还会再次校验基座、tokenizer 和 adapter 的哈希，BIO 命令会校验模型与数据哈希。

```powershell
python scripts\evaluate_massive_slots.py --role confirmation --unlock-confirmation --eval-file data\massive-zh\slots\confirmation.jsonl --expected-model-sha256 $lock.base_model_sha256 --expected-tokenizer-sha256 $lock.tokenizer_sha256 --metrics reports\massive-slots-base-confirmation.json --predictions _tmp\massive-slots-base-confirmation-predictions.jsonl --batch-size 8 --max-input-tokens 256 --max-new-tokens 64
python scripts\evaluate_massive_slots.py --role confirmation --unlock-confirmation --adapter outputs\massive-slots-sft-lr3e4-96 --eval-file data\massive-zh\slots\confirmation.jsonl --expected-model-sha256 $lock.base_model_sha256 --expected-tokenizer-sha256 $lock.tokenizer_sha256 --expected-adapter-sha256 $lock.selected_adapter_sha256 --metrics reports\massive-slots-sft-confirmation.json --predictions _tmp\massive-slots-sft-confirmation-predictions.jsonl --batch-size 8 --max-input-tokens 256 --max-new-tokens 64
python scripts\evaluate_massive_slots_bio_confirmation.py --unlock-confirmation --expected-model-sha256 $lock.bio_model_sha256
python scripts\compare_massive_slots.py --role confirmation --unlock-confirmation --eval-file data\massive-zh\slots\confirmation.jsonl --system base=_tmp\massive-slots-base-confirmation-predictions.jsonl --system sft96=_tmp\massive-slots-sft-confirmation-predictions.jsonl --system bio=_tmp\massive-slots-bio-confirmation-predictions.jsonl --output reports\massive-slots-confirmation-comparison.json
```

发布前可运行 `python -m unittest discover -s tests -q` 和 `python scripts\check_release_tree.py`。离线 CI 检查代码与文件边界，不会重新下载数据或训练模型；公开聚合报告中的数值需要按本页流程在具备模型和原始数据的机器上复核。
